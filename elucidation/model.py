from __future__ import annotations

import torch
import torch.nn as nn

from modules import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    LearnableClassEmbedding,
    clones,
    subsequent_mask,
)


class SpectralEncoding(nn.Module):
    def __init__(self, d_model: int = 256, patch_size: int = 8, dropout: float = 0.1, spectral_channel: int = 1):
        super().__init__()
        self.encoding = nn.Conv1d(spectral_channel, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = LayerNorm(d_model)
        self.class_encoding = LearnableClassEmbedding(d_model, dropout)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, input_spectra: torch.Tensor) -> torch.Tensor:
        # input: (B, C, L)
        x = self.encoding(input_spectra).transpose(1, 2)  # (B, T, D)
        x = self.norm(x)
        x = self.class_encoding(x)                         # prepend cls token
        x = self.positional_encoding(x)
        return x


class MolecularEncoding(nn.Module):
    def __init__(self, d_model: int = 256, vocab_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.mask_token = nn.Parameter(torch.randn(d_model))

    def forward(self, input_ids: torch.Tensor, mask_token_id: int) -> torch.Tensor:
        x = self.embedding(input_ids)
        mask_positions = (input_ids == mask_token_id).unsqueeze(-1)
        x = torch.where(mask_positions, self.mask_token, x)
        x = self.positional_encoding(x)
        return x


class SpectralEncoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, d_ff: int = 1024, nlayer: int = 6, dropout: float = 0.1):
        super().__init__()
        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, self_attn, feed_forward, dropout)
        self.layers = clones(layer, nlayer)
        self.norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


class MolecularEncoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, d_ff: int = 1024, nlayer: int = 2, dropout: float = 0.1):
        super().__init__()
        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, self_attn, feed_forward, dropout)
        self.layers = clones(layer, nlayer)
        self.norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


class MolecularDecoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, d_ff: int = 1024, nlayer: int = 6, dropout: float = 0.1, vocab_size: int = 181):
        super().__init__()
        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        src_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = DecoderLayer(d_model, self_attn, src_attn, feed_forward, dropout)
        self.layers = clones(layer, nlayer)
        self.norm = LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, vocab_size),
            nn.Tanh(),
            nn.Linear(vocab_size, vocab_size),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        return self.proj(x)


class SpectrumToSmilesModel(nn.Module):
    """
    vib2mol-like generation architecture:
      spectral_encoding -> spectral_encoder -> molecular_encoder -> molecular_decoder
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        spectral_channel: int = 1,
        encoder_layers: int = 6,
        molecular_encoder_layers: int = 2,
        d_ff: int = 1024,
    ):
        super().__init__()
        self.spectral_encoding = SpectralEncoding(d_model=d_model, spectral_channel=spectral_channel)
        self.spectral_encoder = SpectralEncoder(d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=encoder_layers)
        self.molecular_encoding = MolecularEncoding(d_model=d_model, vocab_size=vocab_size)
        self.molecular_encoder = MolecularEncoder(
            d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=molecular_encoder_layers
        )
        self.molecular_decoder = MolecularDecoder(
            d_model=d_model, nhead=nhead, d_ff=d_ff, nlayer=num_decoder_layers, vocab_size=vocab_size
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def _prepare_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        # allow (B, L) or (B, C, L)
        if spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(1)
        return spectrum

    def forward(
        self,
        spectrum: torch.Tensor,
        tgt_ids: torch.Tensor,
        causal: bool = True,
        tgt_key_padding_mask: torch.Tensor | None = None,
        mask_token_id: int = 4,
    ) -> torch.Tensor:
        spectrum = self._prepare_spectrum(spectrum)

        src_embeds = self.spectral_encoding(spectrum)
        src_hidden = self.spectral_encoder(src_embeds, mask=None)

        tgt_embeds = self.molecular_encoding(tgt_ids, mask_token_id=mask_token_id)
        keep_mask = None
        if tgt_key_padding_mask is not None:
            keep_mask = (~tgt_key_padding_mask).to(tgt_ids.device)
        tgt_hidden = self.molecular_encoder(tgt_embeds, mask=keep_mask)

        tgt_mask = None
        if causal:
            tgt_mask = subsequent_mask(tgt_ids.size(1)).type_as(tgt_ids.data)

        logits = self.molecular_decoder(tgt_hidden, src_hidden, src_mask=None, tgt_mask=tgt_mask if tgt_mask is not None else keep_mask)
        return logits

    @torch.no_grad()
    def greedy_decode(self, spectrum: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 256, mask_token_id: int = 4):
        spectrum = self._prepare_spectrum(spectrum)
        src_embeds = self.spectral_encoding(spectrum)
        src_hidden = self.spectral_encoder(src_embeds, mask=None)

        bsz = spectrum.size(0)
        ys = torch.full((bsz, 1), bos_id, dtype=torch.long, device=spectrum.device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=spectrum.device)

        for _ in range(max_len - 1):
            tgt_embeds = self.molecular_encoding(ys, mask_token_id=mask_token_id)
            tgt_hidden = self.molecular_encoder(tgt_embeds, mask=None)
            tgt_mask = subsequent_mask(ys.size(1)).type_as(ys.data)
            logits = self.molecular_decoder(tgt_hidden, src_hidden, src_mask=None, tgt_mask=tgt_mask)
            next_logits = logits[:, -1, :]
            nxt = torch.argmax(next_logits, dim=-1)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            finished |= (nxt == eos_id)
            if finished.all():
                break
        return ys
