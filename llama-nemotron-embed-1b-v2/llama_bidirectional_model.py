from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, HybridCache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.utils import logging



logger = logging.get_logger(__name__)


class LlamaBidirectionalConfig(LlamaConfig):
    model_type = "llama_bidirec"

    def __init__(
        self, pooling="avg", temperature=1.0, **kwargs,
    ):
        self.pooling = pooling
        self.temperature = temperature
        super().__init__(**kwargs,)


class LlamaBidirectionalModel(LlamaModel):
    config_class = LlamaBidirectionalConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False
        self.config._attn_implementation = "eager"

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # Generates bi-directional attention.
        causal_mask = _prepare_4d_attention_mask(attention_mask, input_tensor.dtype)
        return causal_mask


