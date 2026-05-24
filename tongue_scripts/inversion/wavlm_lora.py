# part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
# and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
import argparse
import copy
import os
from collections import OrderedDict
from functools import lru_cache
from typing import Callable, Optional

import loralib as lora
import numpy as np
import torch
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v2
import transformers.models.wavlm.modeling_wavlm as wavlm
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import normalize
from torch.nn.utils import weight_norm
from torchaudio.compliance import kaldi
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WavLMModel,
)


class WavLMEncoderLayer(nn.Module):
    def __init__(self, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.config = config

        if self.config.finetune_method == "lora":
            self.feed_forward.intermediate_dense = lora.Linear(
                config.hidden_size,
                config.intermediate_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.feed_forward.output_dense = lora.Linear(
                config.intermediate_size,
                config.hidden_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.attention.k_proj = lora.Linear(
                config.hidden_size,
                config.hidden_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.attention.q_proj = lora.Linear(
                config.hidden_size,
                config.hidden_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.attention.v_proj = lora.Linear(
                config.hidden_size,
                config.hidden_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )
            self.attention.out_proj = lora.Linear(
                config.hidden_size,
                config.hidden_size,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                merge_weights=False,
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        index=0,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states)
        )

        # Adapter
        outputs = (hidden_states, position_bias)

        return outputs


class WavLMWrapper(nn.Module):
    def __init__(self, lora_rank=4, lora_alpha=4, hidden_dim=256, output_class_num=4):
        super(WavLMWrapper, self).__init__()
        # 1. We Load the model first with weights
        self.backbone_model = WavLMModel.from_pretrained(
            "microsoft/wavlm-large",
            output_hidden_states=True,
            use_safetensors=True,
        )
        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method = "lora"
        self.model_config.lora_rank = lora_rank
        self.model_config.lora_alpha = lora_alpha

        # 3. Config encoder layers with adapter or embedding prompt
        self.backbone_model.encoder.layers = nn.ModuleList(
            [
                WavLMEncoderLayer(
                    self.model_config, has_relative_position_bias=(i == 0)
                )
                for i in range(self.model_config.num_hidden_layers)
            ]
        )
        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        # 5. Freeze the weights
        for name, p in self.backbone_model.named_parameters():
            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.regression_head = nn.Linear(self.model_config.hidden_size, 16)

    def forward(self, x, inp_mask=None, length=None):
        # 1. feature extraction and projections
        if inp_mask is not None:
            x = self.backbone_model(x, attention_mask=inp_mask).last_hidden_state
        else:
            x = self.backbone_model(x).last_hidden_state

        predicted = self.regression_head(x)

        return predicted
