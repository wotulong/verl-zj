from typing import Optional, Tuple

import torch
from torch import nn
from verl.models.deepseekv2.configuration_deepseek import DeepseekV2Config
from megatron.core import ModelParallelConfig

from .parallel_moe import ParallelDeepseekV2MoE

from.parallel_attention import ParallelDeepseekV2Attention, ParallelDeepseekV2AttentionRmPad
from .parallel_mlp import ParallelDeepseekV2MLP
from .parallel_rmsnorm import ParallelDeepseekV2RMSNorm

from verl.utils.megatron_utils import TransformerConfig, convert_deepseekv2_config



class ParallelDeepseekV2DecoderLayer(nn.Module):

    def __init__(self, config: DeepseekV2Config, megatron_config: ModelParallelConfig, layer_idx: int):
        super().__init__()
        self.config: TransformerConfig = convert_deepseekv2_config(config, megatron_config)
        self.hidden_size = config.hidden_size

        self.self_attn = ParallelDeepseekV2Attention(config, megatron_config)

        self.mlp = (
            ParallelDeepseekV2MoE(config, megatron_config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else ParallelDeepseekV2MLP(config, megatron_config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = ParallelDeepseekV2RMSNorm(
            config.hidden_size, megatron_config, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = ParallelDeepseekV2RMSNorm(
            config.hidden_size, megatron_config, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states
        return outputs

class ParallelDeepseekV2DecoderLayerRmPad(nn.Module):

    def __init__(self, config: DeepseekV2Config, megatron_config: ModelParallelConfig, layer_idx: int):
        super().__init__()
        self.config: TransformerConfig = convert_deepseekv2_config(config, megatron_config)
        self.hidden_size = config.hidden_size

        self.self_attn = ParallelDeepseekV2AttentionRmPad(config, megatron_config)

        self.mlp = (
            ParallelDeepseekV2MoE(config, megatron_config)
            if (
                    config.n_routed_experts is not None
                    and layer_idx >= config.first_k_dense_replace
                    and layer_idx % config.moe_layer_freq == 0
            )
            else ParallelDeepseekV2MLP(config, megatron_config)
        )
        self.input_layernorm = ParallelDeepseekV2RMSNorm(
            config.hidden_size, megatron_config, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = ParallelDeepseekV2RMSNorm(
            config.hidden_size, megatron_config, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states  # (total_nnz // sp, 1, hidden_size)

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # (total_nnz // sp, 1, hidden_size) -> all-gather (total_nnz, 1, hidden_size)
        # -> col + row -> reduce-scatter -> (total_nnz // sp, 1, hidden_size)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       position_ids=position_ids,
                                       sequence_length=sequence_length,
                                       indices=indices,
                                       cu_seqlens=cu_seqlens,
                                       max_seqlen_in_batch=max_seqlen_in_batch)

        hidden_states = residual + hidden_states

        # Fully Connected
        # shape changes same as attn
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs