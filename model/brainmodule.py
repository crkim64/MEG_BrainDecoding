# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import random
import typing as tp
from functools import partial

import mne
import torch
import torchaudio as ta
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


# ==========================================
# 1. Helper Classes (from Common.py)
# ==========================================

def pad_multiple(x: torch.Tensor, base: int):
    length = x.shape[-1]
    target = math.ceil(length / base) * base
    return torch.nn.functional.pad(x, (0, target - length))


class ScaledEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data /= scale
        self.scale = scale

    def forward(self, x):
        return self.embedding(x) * self.scale


class SubjectLayers(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class ConvSequence(nn.Module):
    def __init__(self, channels: tp.Sequence[int], kernel: int = 4, dilation_growth: int = 1,
                 dilation_period: tp.Optional[int] = None, stride: int = 2,
                 dropout: float = 0.0, leakiness: float = 0.0, groups: int = 1,
                 decode: bool = False, batch_norm: bool = False, dropout_input: float = 0,
                 skip: bool = False, scale: tp.Optional[float] = None, rewrite: bool = False,
                 activation_on_last: bool = True, post_skip: bool = False, glu: int = 0,
                 glu_context: int = 0, glu_glu: bool = True, activation: tp.Any = None) -> None:
        super().__init__()
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        
        dilation = 1
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            if k == 0 and dropout_input:
                layers.append(nn.Dropout(dropout_input))

            if dilation_growth > 1:
                assert kernel % 2 != 0
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(Conv(chin, chout, kernel, stride, pad,
                               dilation=dilation, groups=groups if k > 0 else 1))
            dilation *= dilation_growth
            
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]

            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context), act))
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x


class PositionGetter:
    INVALID = -0.1

    def __init__(self) -> None:
        self._cache: tp.Dict[int, torch.Tensor] = {}
        self._invalid_names: tp.Set[str] = set()

    def get_recording_layout(self, recording) -> torch.Tensor:
        index = recording.recording_index
        if index in self._cache:
            return self._cache[index]
        else:
            info = recording.mne_info
            layout = mne.find_layout(info)
            indexes: tp.List[int] = []
            valid_indexes: tp.List[int] = []
            for meg_index, name in enumerate(info.ch_names):
                name = name.rsplit("-", 1)[0]
                try:
                    indexes.append(layout.names.index(name))
                except ValueError:
                    if name not in self._invalid_names:
                        self._invalid_names.add(name)
                else:
                    valid_indexes.append(meg_index)

            positions = torch.full((len(info.ch_names), 2), self.INVALID)
            x, y = layout.pos[indexes, :2].T
            x = (x - x.min()) / (x.max() - x.min())
            y = (y - y.min()) / (y.max() - y.min())
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            positions[valid_indexes, 0] = x
            positions[valid_indexes, 1] = y
            self._cache[index] = positions
            return positions

    def get_positions(self, batch):
        meg = batch.meg
        B, C, T = meg.shape
        positions = torch.full((B, C, 2), self.INVALID, device=meg.device)
        for idx in range(len(batch)):
            recording = batch._recordings[idx]
            rec_pos = self.get_recording_layout(recording)
            positions[idx, :len(rec_pos)] = rec_pos.to(meg.device)
        return positions

    def is_invalid(self, positions):
        return (positions == self.INVALID).all(dim=-1)


class FourierEmb(nn.Module):
    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        n_freqs = (self.dimension // 2)**0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb


class ChannelDropout(nn.Module):
    def __init__(self, dropout: float = 0.1, rescale: bool = True):
        super().__init__()
        self.dropout = dropout
        self.rescale = rescale
        self.position_getter = PositionGetter()

    def forward(self, meg, batch):
        if not self.dropout:
            return meg
        B, C, T = meg.shape
        meg = meg.clone()
        positions = self.position_getter.get_positions(batch)
        valid = (~self.position_getter.is_invalid(positions)).float()
        meg = meg * valid[:, :, None]

        if self.training:
            center_to_ban = torch.rand(2, device=meg.device)
            kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
            meg = meg * kept.float()[:, :, None]
            if self.rescale:
                proba_kept = torch.zeros(B, C, device=meg.device)
                n_tests = 100
                for _ in range(n_tests):
                    center_to_ban = torch.rand(2, device=meg.device)
                    kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
                    proba_kept += kept.float() / n_tests
                meg = meg / (1e-8 + proba_kept[:, :, None])
        return meg


class ChannelMerger(nn.Module):
    def __init__(self, chout: int, pos_dim: int = 256,
                 dropout: float = 0, usage_penalty: float = 0.,
                 n_subjects: int = 200, per_subject: bool = False):
        super().__init__()
        self.position_getter = PositionGetter()
        self.per_subject = per_subject
        
        self.embedding = FourierEmb(pos_dim)
        
        n_freqs = len(torch.arange((pos_dim // 2)**0.5))
        actual_pos_dim = (n_freqs ** 2) * 2

        if self.per_subject:
            self.heads = nn.Parameter(torch.randn(n_subjects, chout, actual_pos_dim, requires_grad=True))
        else:
            self.heads = nn.Parameter(torch.randn(chout, actual_pos_dim, requires_grad=True))
        
        self.heads.data /= actual_pos_dim ** 0.5
        self.dropout = dropout
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def forward(self, meg, batch):
        B, C, T = meg.shape
        meg = meg.clone()
        
        positions = self.position_getter.get_positions(batch)
        embedding = self.embedding(positions)
        
        score_offset = torch.zeros(B, C, device=meg.device)
        score_offset[self.position_getter.is_invalid(positions)] = float('-inf')

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=meg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float('-inf')

        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            subject = batch.subject_index
            heads = self.heads.gather(0, subject.view(-1, 1, 1).expand(-1, cout, pos_dim))
        else:
            heads = self.heads[None].expand(B, -1, -1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", meg, weights)
        if self.training and self.usage_penalty > 0.:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out

# ==========================================
# 2. Main Model: ImageDecodingMEG
# ==========================================

class TemporalAggregation(nn.Module):
    def __init__(self, time_steps: int):
        super().__init__()
        self.weights = nn.Linear(time_steps, 1)

    def forward(self, x):
        # x: (B, C, T) -> (B, C)
        return self.weights(x).squeeze(-1)


class BrainModule(nn.Module):
    def __init__(self,
                 in_channels: tp.Dict[str, int],
                 out_dim_clip: int, # [수정] CLIP Head Output Dimension
                 out_dim_mse: int,  # [수정] MSE Head Output Dimension
                 time_len: int, 
                 hidden: tp.Dict[str, int],
                 depth: int = 4,
                 kernel_size: int = 5,
                 growth: float = 1.,
                 dilation_growth: int = 2,
                 dilation_period: tp.Optional[int] = None,
                 skip: bool = True,
                 post_skip: bool = True,
                 scale: tp.Optional[float] = 1.0,
                 rewrite: bool = True,
                 groups: int = 1,
                 glu: int = 1,
                 glu_context: int = 1,
                 glu_glu: bool = True,
                 gelu: bool = False,
                 conv_dropout: float = 0.0,
                 dropout_input: float = 0.0,
                 batch_norm: bool = True,
                 relu_leakiness: float = 0.0,
                 n_subjects: int = 4,
                 subject_dim: int = 0,
                 subject_layers: bool = True,
                 subject_layers_dim: str = "input", 
                 subject_layers_id: bool = False,
                 embedding_scale: float = 1.0,
                 merger: bool = True,
                 merger_pos_dim: int = 512,
                 merger_channels: int = 270,
                 merger_dropout: float = 0.2,
                 merger_penalty: float = 0.,
                 merger_per_subject: bool = True,
                 dropout: float = 0.,
                 dropout_rescale: bool = True,
                 initial_linear: int = 0,
                 initial_depth: int = 1,
                 initial_nonlin: bool = False,
                 subsample_meg_channels: int = 0,
                 ):
        super().__init__()
        
        # --- Standard ConvNet Initialization ---
        if set(in_channels.keys()) != set(hidden.keys()):
            raise ValueError("Keys mismatch")

        if gelu:
            activation = nn.GELU
        elif relu_leakiness:
            activation = partial(nn.LeakyReLU, relu_leakiness)
        else:
            activation = nn.ReLU

        assert kernel_size % 2 == 1

        self.merger = None
        self.dropout = None
        self.subsampled_meg_channels = None
        if subsample_meg_channels:
            assert 'meg' in in_channels
            indexes = list(range(in_channels['meg']))
            random.Random(1234).shuffle(indexes)
            self.subsampled_meg_channels = indexes[:subsample_meg_channels]

        self.initial_linear = None
        if dropout > 0.:
            self.dropout = ChannelDropout(dropout, dropout_rescale)
        if merger:
            self.merger = ChannelMerger(
                merger_channels, pos_dim=merger_pos_dim, dropout=merger_dropout,
                usage_penalty=merger_penalty, n_subjects=n_subjects, per_subject=merger_per_subject)
            in_channels["meg"] = merger_channels

        self.post_merger_linear = nn.Sequential(
            nn.Conv1d(270, 270, 1), # 1x1 Conv acts as Linear
            activation()
        )

        if initial_linear:
            init = [nn.Conv1d(in_channels["meg"], initial_linear, 1)]
            for _ in range(initial_depth - 1):
                init += [activation(), nn.Conv1d(initial_linear, initial_linear, 1)]
            if initial_nonlin:
                init += [activation()]
            self.initial_linear = nn.Sequential(*init)
            in_channels["meg"] = initial_linear

        self.subject_layers = None
        if subject_layers:
            assert "meg" in in_channels
            meg_dim = in_channels["meg"]
            dim = {"hidden": hidden["meg"], "input": meg_dim}[subject_layers_dim]
            self.subject_layers = SubjectLayers(meg_dim, dim, n_subjects, subject_layers_id)
            in_channels["meg"] = dim

        self.subject_embedding = None
        if subject_dim:
            self.subject_embedding = ScaledEmbedding(n_subjects, subject_dim, embedding_scale)
            in_channels["meg"] += subject_dim

        # Compute channel sizes
        sizes = {}
        for name in in_channels:
            sizes[name] = [in_channels[name]]
            sizes[name] += [int(round(hidden[name] * growth ** k)) for k in range(depth)]

        params: tp.Dict[str, tp.Any]
        params = dict(kernel=kernel_size, stride=1,
                      leakiness=relu_leakiness, dropout=conv_dropout, dropout_input=dropout_input,
                      batch_norm=batch_norm, dilation_growth=dilation_growth, groups=groups,
                      dilation_period=dilation_period, skip=skip, post_skip=post_skip, scale=scale,
                      rewrite=rewrite, glu=glu, glu_context=glu_context, glu_glu=glu_glu,
                      activation=activation)

        self.encoders = nn.ModuleDict({name: ConvSequence(channels, **params)
                                       for name, channels in sizes.items()})

        # --- KEY CHANGES: Linear Projection & Heads ---
        
        final_conv_channels = sum([x[-1] for x in sizes.values()]) # Usually 320
        projected_dim = 2048 # From ImageDecodingMEG Paper

        # 1. Linear Projection (320 -> 2048)
        self.linear_projection = nn.Sequential(
            nn.Conv1d(final_conv_channels, projected_dim, 1),
            activation()
        )

        # 2. Temporal Aggregation (on 2048 channels)
        self.temporal_aggregator = TemporalAggregation(time_len)

        # 3. Dual Heads [수정]
        # 각각 CLIP Head와 MSE Head로 분리
        self.head_clip = nn.Linear(projected_dim, out_dim_clip)
        self.head_mse = nn.Linear(projected_dim, out_dim_mse)

    def forward(self, inputs, batch):
        subjects = batch.subject_index

        # Pre-processing
        if self.subsampled_meg_channels is not None:
            mask = torch.zeros_like(inputs["meg"][:1, :, :1])
            mask[:, self.subsampled_meg_channels] = 1.
            inputs["meg"] = inputs["meg"] * mask

        if self.dropout is not None:
            inputs["meg"] = self.dropout(inputs["meg"], batch)

        if self.merger is not None:
            inputs["meg"] = self.merger(inputs["meg"], batch)

            if hasattr(self, 'post_merger_linear'):
                inputs["meg"] = self.post_merger_linear(inputs["meg"])

        if self.initial_linear is not None:
            inputs["meg"] = self.initial_linear(inputs["meg"])

        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)
        
        length = inputs["meg"].shape[-1]
        if self.subject_embedding is not None:
            emb = self.subject_embedding(subjects)[:, :, None]
            inputs["meg"] = torch.cat([inputs["meg"], emb.expand(-1, -1, length)], dim=1)

        # ConvNet Backbone
        encoded = {}
        for name, x in inputs.items():
            encoded[name] = self.encoders[name](x)

        inputs = [x[1] for x in sorted(encoded.items())]
        x = torch.cat(inputs, dim=1) # (B, 320, T)

        # --- Image Decoding Logic ---
        
        # 1. Linear Projection: (B, 320, T) -> (B, 2048, T)
        x = self.linear_projection(x)

        # 2. Temporal Aggregation: (B, 2048, T) -> (B, 2048)
        x_aggregated = self.temporal_aggregator(x)

        # 3. Dual Heads Output [수정]
        out_clip = self.head_clip(x_aggregated) # (B, out_dim_clip)
        out_mse = self.head_mse(x_aggregated)   # (B, out_dim_mse)

        return out_clip, out_mse