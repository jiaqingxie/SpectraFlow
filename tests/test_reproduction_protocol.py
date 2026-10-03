"""Protocol checks using tiny synthetic spectra; no scientific benchmark claims."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from audit_paper_reproducibility import legacy_order
from build_nist_qm9s_overlap import read_qm9s_mapping
from model_flow import ConditionalFlowMatching
from reviewer_experiments.common import LazyPairs, digest, peak_metrics, validate_split
from reviewer_experiments.prepare import chemical_keys, group_partition
from reviewer_experiments.training import train


class ProtocolTests(unittest.TestCase):
    def test_identity_and_scaffold_groups(self):
        a,b = chemical_keys('CCO'),chemical_keys('CCN')
        self.assertNotEqual(a[1],b[1])
        self.assertEqual(a[2],b[2])
        self.assertTrue(chemical_keys('O=S(=O)(O)O')[2])
        keys=['C','CC','CCC','CCO','CCN']
        labels=group_partition(keys)
        self.assertEqual(dict(zip(keys,labels)),dict(zip(keys[::-1],group_partition(keys[::-1]))))

    def test_group_leakage_is_rejected(self):
        frame=pd.DataFrame(dict(row_id=[0,1,2,3],identity=['a','b','c','d'],split_group=['a','b','c','d'],
                                split=['train','valid','calibration','test']))
        validate_split(frame)
        frame.loc[3,'identity']='a'
        with self.assertRaisesRegex(ValueError,'leaks'):
            validate_split(frame)

    def test_qm9s_mapping_rejects_wrong_counts_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mapping.txt'
            path.write_text('1\tC\n2\tN\n')
            self.assertEqual(read_qm9s_mapping(path,2),['C','N'])
            with self.assertRaises(ValueError): read_qm9s_mapping(path,3)
            path.write_text('2\tN\n1\tC\n')
            with self.assertRaises(ValueError): read_qm9s_mapping(path,2)

    def test_solvers_count_network_evaluations(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.calls=0
            def forward(self,x,t,condition=None,target_mode=None):
                self.calls+=1
                return torch.full_like(x,.1)
        model=ConditionalFlowMatching(image_size=(4,4),backbone='vibradit',
                    dit_hidden_dim=24,dit_depth=1,dit_num_heads=3,dit_patch_size=4)
        counter=Constant(); model.velocity_field=counter
        source=torch.zeros(2,1,4,4)
        a=model.sample(source,num_steps=8)
        self.assertEqual(counter.calls,8)
        counter.calls=0
        b=model.sample(source,num_steps=2,use_rk4=True)
        self.assertEqual(counter.calls,8)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(a,torch.full_like(a,.1))

    def test_physical_peak_coordinates(self):
        wave=np.arange(100.)*2+400
        reference=np.exp(-((wave-460)/3)**2)+.6*np.exp(-((wave-540)/3)**2)
        prediction=np.exp(-((wave-466)/3)**2)+.6*np.exp(-((wave-546)/3)**2)
        m=peak_metrics(reference,prediction,wave,tolerance=10)
        self.assertEqual(m['peak_recall'],1.)
        self.assertAlmostEqual(m['peak_position_mae'],6.)
        self.assertEqual(peak_metrics(reference,prediction,wave,tolerance=4)['peak_recall'],0.)
        order=legacy_order(10)
        np.testing.assert_array_equal(np.sort(order),np.arange(100))
        np.testing.assert_array_equal(reference[order][np.argsort(order)],reference)

    def test_lazy_normalization_and_recoverable_training(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            axis=np.linspace(400,4000,64)
            source=np.stack([np.exp(-((np.arange(64)-(8+i))/3)**2) for i in range(16)]).astype('float32')
            target=np.roll(source,8,axis=1)
            for name,data in [('ir',source),('raman',target)]:
                with h5py.File(directory/f'{name}.h5','w') as f:
                    f['spectra']=data; f['x_axis']=axis
            frame=pd.DataFrame(dict(row_id=np.arange(16),identity=[f'm{i}' for i in range(16)],
                split_group=[f'm{i}' for i in range(16)],split=['train']*8+['valid']*4+['calibration']*2+['test']*2))
            path=directory/'split.csv'; frame.to_csv(path,index=False)
            ds=LazyPairs(directory/'ir.h5',directory/'raman.h5',[0,7])
            a,b,row=ds[1]
            self.assertEqual(row,7); self.assertEqual(a.shape,(1,8,8))
            self.assertEqual(int(b.argmax())-int(a.argmax()),8)
            ds.close()
            job=dict(source=str(directory/'ir.h5'),target=str(directory/'raman.h5'),split=str(path),
                split_sha256=digest(path),name='synthetic_reverse',direction='raman2ir',variant='full',
                seed=0,split_seed=2,side=8,hidden=24,depth=1,heads=3,patch=4,sigma_min=.01,
                learning_rate=.0002,batch_size=4,epochs=1,endpoint_weight=.5,endpoint_probability=1.,steps=2,
                budget='equal_updates',max_train_seconds=None,method='direct')
            args=SimpleNamespace(device='cpu',threads=2,budget_seconds=None,output=directory/'runs',resume=False)
            checkpoint=train(job,args)
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            self.assertEqual(payload['config']['direction'],'raman2ir')
            self.assertTrue(np.isfinite(payload['progress']['best_val_loss']))
            args.resume=True
            train(job,args)
            resumed=torch.load(checkpoint,map_location='cpu',weights_only=True)
            for name in payload['model_state_dict']:
                torch.testing.assert_close(payload['model_state_dict'][name],resumed['model_state_dict'][name])
            job.update(name='synthetic_flow',direction='ir2raman',method='flow')
            args.resume=False
            flow=train(job,args)
            self.assertTrue(flow.exists())


if __name__=='__main__':
    unittest.main()
