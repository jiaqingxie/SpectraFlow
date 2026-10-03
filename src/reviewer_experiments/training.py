"""Unified Flow/Direct training with fixed splits and recoverable budget accounting."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from model_flow import ConditionalFlowMatching
from train_flow import (spectral_importance_weight, weighted_elastic_loss,
                        derivative_shape_loss, local_ot_loss)
from audit_paper_reproducibility import legacy_order
from .common import ROOT, LazyPairs, digest, save_json, validate_split


def device_for(args):
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Allocate a GPU; CPU requires explicit --device cpu.')
    return torch.device(args.device)


def model_for(job, device):
    torch.manual_seed(job['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(job['seed'])
    model = ConditionalFlowMatching(image_size=(job['side'],)*2, backbone='vibradit',
        dit_hidden_dim=job['hidden'], dit_depth=job['depth'], dit_num_heads=job['heads'],
        dit_patch_size=job['patch'], sigma_min=job['sigma_min']).to(device)
    # Match legacy Flow/Direct Xavier initialization while preserving adaLN zeros.
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            if '.adaLN' in name or name.startswith('velocity_field.final_layer.'):
                continue
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
    return model


def endpoint(model, source, mode, method, steps=8, rk4=False):
    if method == 'direct':
        t = torch.zeros(len(source), device=source.device, dtype=source.dtype)
        return (source + model.velocity_field(source, t, condition=source, target_mode=mode)).clamp(0, 1)
    return model.sample(source, target_mode=mode, num_steps=steps, use_rk4=rk4)


def loss_terms(model, source, target, mode, job, apply_endpoint):
    variant = job['variant']
    tf = target.reshape(len(target), -1)
    if variant in ('no_peak', 'basic'):
        weight = torch.ones_like(tf)
    else:
        weight = spectral_importance_weight(tf, 1., .5, .5)
    if job['method'] == 'flow':
        pred, reference, _ = model(source, target, target_mode=mode)
    else:
        t = torch.zeros(len(source), device=source.device)
        pred = model.velocity_field(source, t, condition=source, target_mode=mode)
        reference = target - source
    primary = weighted_elastic_loss(pred.reshape(len(source), -1), reference.reshape(len(source), -1), weight)
    reconstruction = torch.zeros_like(primary)
    if apply_endpoint:
        prediction = endpoint(model, source, mode, job['method'], job['steps']).reshape(len(source), -1)
        reconstruction = weighted_elastic_loss(prediction, tf, weight)
        if variant not in ('no_shape','basic'):
            reconstruction = reconstruction + .05 * derivative_shape_loss(prediction, tf)
        if variant not in ('no_ot','basic'):
            reconstruction = reconstruction + .02 * local_ot_loss(prediction, tf, window_size=64)
    total = primary + job['endpoint_weight']/job['endpoint_probability'] * reconstruction
    return total, primary, reconstruction


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def validation_loss(model, loader, mode, job, device):
    model.eval()
    total, count = 0., 0
    for source, target, _ in loader:
        source, target = source.to(device), target.to(device)
        pred = endpoint(model, source, mode, job['method'], job['steps'])
        value = (.6 * (pred-target).abs() + .4 * (pred-target).square()).reshape(len(source),-1).mean(dim=1)
        total += value.sum().item()
        count += len(source)
    return total/count


def train(job, args):
    device = device_for(args)
    job = dict(job)
    if job['direction'] not in ('ir2raman', 'raman2ir'):
        raise ValueError('Unknown translation direction')
    if args.budget_seconds:
        if args.budget_seconds <= 0:
            raise ValueError('Time budget must be positive')
        job.update(budget='equal_training_seconds', max_train_seconds=args.budget_seconds,
                   name=job['name']+f"_seconds{args.budget_seconds:g}")
    folder = args.output/'training'/job['name']
    folder.mkdir(parents=True, exist_ok=True)
    config_file = folder/'config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != job:
        raise ValueError('Existing run has a different configuration')
    if (folder/'last.pt').exists() and not args.resume:
        raise ValueError('Run already exists; use --resume or a different run name')
    if digest(job['split']) != job['split_sha256']:
        raise ValueError('Prepared split manifest changed')
    frame = pd.read_csv(job['split'])
    validate_split(frame)
    permutation = legacy_order(job['side']) if job['variant']=='legacy_order' else None
    source_path, target_path = job['source'], job['target']
    if job['direction'] == 'raman2ir':
        source_path, target_path = target_path, source_path
    datasets = {part: LazyPairs(source_path,target_path,frame.loc[frame.split==part,'row_id'],permutation)
                for part in ('train','valid')}
    model = model_for(job, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=job['learning_rate'], betas=(.9,.999))
    mode = 2 if job['direction']=='ir2raman' else 0
    endpoint_rng = np.random.default_rng(job['seed']+100000)
    state = dict(epoch=0, next_batch=0, updates=0, examples_seen=0, train_seconds=0.,
                 validation_seconds=0., best_val_loss=float('inf'), endpoint_updates=0)
    if args.resume and (folder/'last.pt').exists():
        checkpoint = torch.load(folder/'last.pt', map_location=device, weights_only=True)
        if checkpoint['config'] != job:
            raise ValueError('Resume configuration differs')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        state = checkpoint['progress']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
        endpoint_rng.bit_generator.state = checkpoint['endpoint_rng']
    save_json(config_file, job)
    def checkpoint(path):
        payload = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                       config=job, progress=state.copy(), torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                       endpoint_rng=endpoint_rng.bit_generator.state,
                       split_sha256=job['split_sha256'], preprocessing='legacy_patch_permutation' if permutation is not None else 'physical_order',
                       model_type=job['method'], direct_time=0.,
                       source_code_sha256={p:digest(ROOT/'src'/p) for p in ('model_flow.py','reviewer_experiments/training.py','reviewer_experiments/common.py')})
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
    batches_per_epoch = math.ceil(len(datasets['train'])/job['batch_size'])
    planned_updates = batches_per_epoch * job['epochs']
    val_loader = DataLoader(datasets['valid'],batch_size=job['batch_size'],shuffle=False,num_workers=0,
                            generator=torch.Generator().manual_seed(job['split_seed']+998000))
    exhausted = False
    # A time-budget run may need more epochs than the equal-update comparison.
    epoch_cap = job['epochs'] if not job['max_train_seconds'] else 1000000
    for epoch in range(state['epoch'], epoch_cap):
        order = torch.randperm(len(datasets['train']),generator=torch.Generator().manual_seed(job['split_seed']+epoch)).tolist()
        start_batch = state['next_batch'] if epoch==state['epoch'] else 0
        loader = DataLoader(Subset(datasets['train'],order[start_batch*job['batch_size']:]),
                            batch_size=job['batch_size'],shuffle=False,num_workers=0,
                            generator=torch.Generator().manual_seed(job['split_seed']+epoch+999000))
        for offset, (source, target, _) in enumerate(loader, start=start_batch):
            model.train()
            synchronize(device)
            begin = time.monotonic()
            source, target = source.to(device), target.to(device)
            progress = state['train_seconds']/job['max_train_seconds'] if job['max_train_seconds'] else state['updates']/planned_updates
            lr = job['learning_rate'] * (.01 + .99*.5*(1+math.cos(math.pi*min(progress,1.))))
            for group in optimizer.param_groups:
                group['lr'] = lr
            apply_endpoint = endpoint_rng.random() < job['endpoint_probability']
            optimizer.zero_grad(set_to_none=True)
            total, primary, reconstruction = loss_terms(model,source,target,mode,job,apply_endpoint)
            if not torch.isfinite(total):
                raise RuntimeError(f'Nonfinite loss at update {state["updates"]}')
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            synchronize(device)
            state['train_seconds'] += time.monotonic()-begin
            state['updates'] += 1
            state['examples_seen'] += len(source)
            state['endpoint_updates'] += int(apply_endpoint)
            state.update(epoch=epoch,next_batch=offset+1)
            if state['updates']%100==0:
                print(job['name'],state['updates'],float(total.detach()),flush=True)
            if state['updates']%1000==0:
                checkpoint(folder/'last.pt')
            if job['max_train_seconds'] and state['train_seconds']>=job['max_train_seconds']:
                exhausted=True
                break
        synchronize(device)
        val_start=time.monotonic()
        val=validation_loss(model,val_loader,mode,job,device)
        synchronize(device)
        state['validation_seconds']+=time.monotonic()-val_start
        if val < state['best_val_loss']:
            state['best_val_loss']=val
            checkpoint(folder/'best.pt')
        if not exhausted:
            state.update(epoch=epoch+1,next_batch=0)
        checkpoint(folder/'last.pt')
        with (folder/'history.jsonl').open('a') as f:
            f.write(json.dumps(dict(**state,validation_loss=val))+'\n')
        print(job['name'],f'epoch {epoch+1}, validation {val:.6g}',flush=True)
        if exhausted:
            break
    for dataset in datasets.values():
        dataset.close()
    save_json(folder/'summary.json',dict(**state, status='complete',config=job,device=str(device),
              gpu_name=torch.cuda.get_device_name() if device.type=='cuda' else None,
              trainable_parameters=sum(p.numel() for p in model.parameters()),
              best_checkpoint_sha256=digest(folder/'best.pt'),
              cost_definition='synchronized transfer+forward+loss+backward+optimizer; validation and HDF5 loading reported separately',
              caveat='equal updates is not equal compute; use the separate seconds-budget comparison'))
    return folder/'best.pt'
