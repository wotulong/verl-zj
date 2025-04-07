import numbers
import torch
from megatron.core import ModelParallelConfig
from torch import nn
from verl.models.deepseekv2.configuration_deepseek import DeepseekV2Config

from apex.normalization.fused_layer_norm import fused_rms_norm_affine
from verl.utils.megatron import sequence_parallel as sp_utils

class ParallelDeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, megatron_config: ModelParallelConfig, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        deepseek attention中有K,V的低秩映射，需要传hidden_size
        """
        super().__init__()
        if isinstance(hidden_size, numbers.Integral):
            normalized_shape = (hidden_size,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.variance_epsilon = eps

        if megatron_config.sequence_parallel:
            sp_utils.mark_parameter_as_sequence_parallel(self.weight)

    def forward(self, hidden_states):
        return fused_rms_norm_affine(input=hidden_states,
                                     weight=self.weight,
                                     normalized_shape=self.normalized_shape,
                                     eps=self.variance_epsilon,)
                                     #memory_efficient=True)
