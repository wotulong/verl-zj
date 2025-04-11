import math
from typing import Optional, Tuple

import torch
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core import ModelParallelConfig
from torch import nn
from verl.models.deepseekv2.configuration_deepseek import DeepseekV2Config

from verl.models.deepseekv2.megatron.layers.parallel_linear import QParallelLinear, MergedColumnParallelLinear
from verl.models.deepseekv2.megatron.layers.parallel_rmsnorm import ParallelDeepseekV2RMSNorm
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

import torch.nn.functional as F
from einops import rearrange

from verl.utils.megatron import tensor_parallel as tp_utils
from verl.utils.megatron_utils import convert_deepseekv2_config
from verl.workers.megatron_workers import logger


class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2LinearScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2DynamicNTKScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )

# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case
def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )

# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV2
class ParallelDeepseekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, megatron_config: ModelParallelConfig):
        super().__init__()
        transformer_config = convert_deepseekv2_config(config, megatron_config)

        self.config = config
        self.megatron_config = megatron_config
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        # assign values after tp
        tp_size = mpu.get_tensor_model_parallel_world_size()

        self.num_heads_per_tp = self.num_heads #// tp_size
        self.hidden_size_per_tp = self.hidden_size #// tp_size

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        if megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            assert row_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        self.is_causal = True

        if self.q_lora_rank is None:
            # self.q_proj = nn.Linear(
            #     self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            # )
            # print(f"\n\nq_lora_rank==============>self.hidden_size:{self.hidden_size}, self.num_heads_per_tp:{self.num_heads_per_tp}, self.q_head_dim: {self.q_head_dim}, ")
            self.q_proj = QParallelLinear(input_size=self.hidden_size,
                                          num_heads=self.num_heads_per_tp, head_dim=self.q_head_dim, bias=False,**column_kwargs)
        else:
            # self.q_a_proj = nn.Linear(
            #     self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            # )
            # 此处直接使用ColumnParallelLinear
            self.q_a_proj = tensor_parallel.ColumnParallelLinear(input_size=self.hidden_size, output_size=self.q_lora_rank, bias=config.attention_bias, **column_kwargs)

            # self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
            self.q_a_layernorm = ParallelDeepseekV2RMSNorm(config.q_lora_rank, megatron_config)

            # self.q_b_proj = nn.Linear(
            #     config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            # )

            self.q_b_proj = QParallelLinear(input_size=self.q_lora_rank, num_heads=self.num_heads_per_tp, head_dim=self.q_head_dim, bias=False, **column_kwargs)

        # no parallel by zsk
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )

        # self.kv_a_proj_with_mqa = MergedColumnParallelLinear(input_size=self.hidden_size, gate_ouput_size=config.kv_lora_rank, up_output_size=config.qk_rope_head_dim, bias=config.attention_bias, **column_kwargs)

        # kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_a_layernorm = ParallelDeepseekV2RMSNorm(config.kv_lora_rank, megatron_config)

        # self.kv_b_proj = nn.Linear(
        #     config.kv_lora_rank,
        #     self.num_heads
        #     * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
        #     bias=False,
        # )

        head_dim = self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim
        self.kv_b_proj = QParallelLinear(input_size=config.kv_lora_rank, num_heads=self.num_heads_per_tp, head_dim=head_dim, bias=False, **column_kwargs)

        # self.o_proj = nn.Linear(
        #     self.num_heads * self.v_head_dim,
        #     self.hidden_size,
        #     bias=config.attention_bias,
        # )

        # input ids 貌似不是并行的，修改下，by zsk 20250405
        self.o_proj = tensor_parallel.RowParallelLinear(input_size=self.num_heads * self.v_head_dim, output_size=self.hidden_size, bias=config.attention_bias, input_is_parallel=False,
                                                        skip_bias_add=False, **row_kwargs)

        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None: # dp2-lite true
            q = self.q_proj(hidden_states)[0]
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)[0]))[0]

        q = q.view(bsz, q_len, self.num_heads_per_tp, self.q_head_dim).transpose(1, 2)

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))[0]
            .view(bsz, q_len, self.num_heads_per_tp, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # if torch.distributed.get_rank() == 0:
        #     # bsz:272, q_len:8, q_pe.shape:torch.Size([272, 16, 8, 64]), k_pe.shape:torch.Size([272, 1, 8, 64]), sin.shape:torch.Size([8, 64]), cos.shape:torch.Size([8, 64]), position_ids.shape:torch.Size([8, 272])
        #     print(f"bsz:{bsz}, q_len:{q_len}, q_pe.shape:{q_pe.shape}, k_pe.shape:{k_pe.shape}, sin.shape:{sin.shape}, cos.shape:{cos.shape}, position_ids.shape:{position_ids.shape}")
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads_per_tp, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads_per_tp, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, self.num_heads_per_tp, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads_per_tp, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads_per_tp, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads_per_tp, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads_per_tp * self.v_head_dim)

        attn_output = self.o_proj(attn_output)[0]
        return attn_output

"""
Remove padding Attention
- Using Flash-attn 2
- Compatible with sequence parallel
"""

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

def apply_rotary_pos_emb_rmpad(q, k, cos, sin, position_ids, indices, sequence_length):
    batch_size = position_ids.shape[0]

    q = pad_input(q, indices, batch_size, sequence_length)  # (batch_size, seqlen, num_head, head_dim)
    k = pad_input(k, indices, batch_size, sequence_length)
    cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = index_first_axis(rearrange(q_embed, "b s ... -> (b s) ..."), indices)
    k_embed = index_first_axis(rearrange(k_embed, "b s ... -> (b s) ..."), indices)

    return q_embed, k_embed


from flash_attn.layers.rotary import apply_rotary_emb

# use flash-attn rotary embeddings with rmpad
# cos/sin shoudl be: (seq_length, rotary_dim / 2)
def apply_rotary_pos_emb_rmpad_flash(q, k, cos, sin, cu_seqlens, max_seqlen):
    q_embed = apply_rotary_emb(q,
                               cos,
                               sin,
                               interleaved=False,
                               inplace=False,
                               cu_seqlens=cu_seqlens,
                               max_seqlen=max_seqlen)
    k_embed = apply_rotary_emb(k,
                               cos,
                               sin,
                               interleaved=False,
                               inplace=False,
                               cu_seqlens=cu_seqlens,
                               max_seqlen=max_seqlen)
    return q_embed, k_embed

class ParallelDeepseekV2AttentionRmPad(ParallelDeepseekV2Attention):
    # MLA 参考：https://github.com/NVIDIA/Megatron-LM/blob/6ba97dd37150a6bfba03d31808674211cf2a4d0d/megatron/core/transformer/multi_latent_attention.py
    def forward(self,
                hidden_states: torch.Tensor,
                position_ids: Optional[torch.LongTensor] = None,
                sequence_length: int = None,
                indices: torch.Tensor = None,
                cu_seqlens: torch.Tensor = None,
                max_seqlen_in_batch: int = None):
        '''
        total_nnz:所有批次中所有序列的有效总和
        cu_seqlens:标记每个序列的长度累计和
        max_seq_in_batch:
        seuquence_length:全局序列长度
        '''

        total_nnz, q_len, hidden_size = hidden_states.size() #torch.Size([1150, 1, 2048])
        # bsz = cu_seqlens.shape[0] - 1

        # if torch.distributed.get_rank() == 0:
            # # hidden_states.shape:torch.Size([845, 1, 2048]), self.q_head_dim:192
            # print(f'hidden_states:{hidden_states.shape}, self.num_heads:{self.num_heads}, self.q_head_dim:{self.q_head_dim}')
            # print(f'sequence_length:{sequence_length}, cu_seqlens:{cu_seqlens}, max_seqlen_in_batch:{max_seqlen_in_batch}') #2560


        if self.megatron_config.sequence_parallel:
            # print("==========> using sequence_parallel, self.megatron_config.sequence_parallel:{self.megatron_config.sequence_parallel}")
            total_nnz = total_nnz * mpu.get_tensor_model_parallel_world_size()
            # print(f"self.kv_lora_rank:{self.kv_lora_rank}, self.v_head_dim:{self.v_head_dim}")

        # 映射q维度
        if self.q_lora_rank is None:
            # 走这里
            q = self.q_proj(hidden_states)[0]
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)[0]))[0]

        # error shape '[845, 2560, 16, 192]' is invalid for input of size 2595840
        # if torch.distributed.get_rank() == 0:
            # shape q:torch.Size([845, 1, 3072])
            # print(f"\n===============>shape q:{q.shape}, ")
        # batch size 改为自动计算
        q = q.view(total_nnz, self.num_heads, self.q_head_dim)#.transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        # mqa:[576,2048], kv_lora_rank:512? qk_rope_head_dim:64?
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # if torch.distributed.get_rank() == 0:
        #     #shape k_pe:torch.Size([845, 1, 64]), self.qk_rope_head_dim：64， total_nnz：845
        #     print(f"\n===============>shape k_pe:{k_pe.shape}, self.qk_rope_head_dim：{self.qk_rope_head_dim}， total_nnz：{total_nnz}")
        
        # [845, 1, 64]
        # k_pe = k_pe.view(total_nnz, 1, self.qk_rope_head_dim)#.transpose(1, 2)

        kv = (
            # [845, 1, 512] ->  [845, 1, 4096], self.num_heads:16 ->[845, 1, 16, 256] ?
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))[0]
            .view(total_nnz, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        )
        # if torch.distributed.get_rank() == 0:
            # kv shape:torch.Size([845, 16, 1, 256]), self.qk_nope_head_dim:128, self.v_head_dim:128
            # print(f"==============>kv shape:{kv.shape}, self.qk_nope_head_dim:{self.qk_nope_head_dim}, self.v_head_dim:{self.v_head_dim}")
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        kv_seq_len = value_states.shape[-2] # [bsz, q_len, num_heads, v_head_dim]

        cos, sin = self.rotary_emb(value_states, seq_len=sequence_length)
        cos, sin = cos[:, :cos.shape[1] // 2], sin[:, :sin.shape[1] // 2]  # flash attn only needs half

        # if torch.distributed.get_rank() == 0:
        #     # parallel_attention_before:max_seqlen_in_batch:464,cu_seqlens:tensor([  0, 464, 845], device='cuda:0', dtype=torch.int32),q_pe:torch.Size([845, 16, 64]), k_pe.shape:torch.Size([845, 1, 64]),cos:torch.Size([464, 32]), sin:torch.Size([464, 32])
        #     print(f'parallel_attention: hidden_size:{hidden_states.shape}, kv_seq_len:{kv_seq_len}, value_states.shape:{value_states.shape}')
        #     # parallel_attention: kv_seq_len:1, value_states.shape:torch.Size([845, 16, 1, 128])
        #     print(f'parallel_attention_before:max_seqlen_in_batch:{max_seqlen_in_batch},cu_seqlens:{cu_seqlens},q_pe:{q_pe.shape}, k_pe.shape:{k_pe.shape},cos:{cos.shape}, sin:{sin.shape}')

        # (batch, seqlen, nheads, headdim)
        # kv_seq_len:1, value_states.shape:torch.Size([1283, 16, 1, 128])
        # q_pe:torch.Size([1283, 16, 1, 64]), [bsz, num_heads, q_len, q_head_dim]
        # k_pe.shape:torch.Size([1283, 1, 1, 64]),[bsz, 1, q_len, head_dim]
        # cos:torch.Size([1, 32]), sin:torch.Size([1, 32])
        #hidden_size:torch.Size([1150, 1, 2048])
        # max_seqlen_in_batch:590,cu_seqlens:tensor([   0,  560, 1150]
            # assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
            # total_seqlen, nheads, headdim = x.shape
            # batch_p_1 = cu_seqlens.shape[0]-->3
            # batch = batch_p_1 - 1-->2
            # seqlen = max_seqlen-->590
        # seqlen_ro, rotary_dim = cos.shape
        # assert sin.shape == cos.shape
        # rotary_dim *= 2
        # assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
        # assert headdim <= 256, "Only support headdim <= 256"
        # assert seqlen_ro, 1 >= seqlen,590, "seqlen_ro must be >= seqlen"
        
        q_pe, k_pe = apply_rotary_pos_emb_rmpad_flash(q_pe,
                                                    k_pe,
                                                    cos,
                                                    sin,
                                                    cu_seqlens=cu_seqlens,
                                                    max_seqlen=max_seqlen_in_batch)
        
        # if torch.distributed.get_rank() == 0:
        #     # parallel_attention: kv_seq_len:1, value_states.shape:torch.Size([845, 16, 1, 128])
        #     print(f'parallel_attention: kv_seq_len:{kv_seq_len}, value_states.shape:{value_states.shape}')
        #     # parallel_attention_after:q_pe:torch.Size([845, 16, 64]), k_pe.shape:torch.Size([845, 1, 64]),cos:torch.Size([464, 32]), sin:torch.Size([464, 32])
        #     print(f'parallel_attention_after:q_pe:{q_pe.shape}, k_pe.shape:{k_pe.shape},cos:{cos.shape}, sin:{sin.shape}')
        
        # 转换维度
        # q_nope = q_nope.view(total_nnz, self.num_heads, q_len, self.qk_nope_head_dim)
        # q_pe = q_pe.view(total_nnz, self.num_heads, q_len, self.qk_rope_head_dim)

        query_states = k_pe.new_empty(total_nnz, self.num_heads, self.q_head_dim)
        query_states[:, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, self.qk_nope_head_dim:] = q_pe

        # k转换维度
        k_nope = k_nope.view(total_nnz, self.num_heads, self.qk_nope_head_dim)
        # k_pe_expanded = k_pe.unsqueeze(1)  # 插入 num_heads 维度 -> [1150, 1, 1, 64]
        # k_pe_expanded = k_pe_expanded.expand(-1, self.num_heads, -1, -1) 

        key_states = k_pe.new_empty(total_nnz, self.num_heads, self.q_head_dim)
        key_states[:, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, self.qk_nope_head_dim:] = k_pe

        # self.q_head_dim: 192, self.v_head_dim:128
        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        # query_states = query_states.transpose(1, 2)
        # key_states = key_states.transpose(1, 2)
        # value_states = value_states.transpose(1, 2)


        if self.megatron_config.sequence_parallel:
            sequence_parallel_pad = total_nnz - cu_seqlens[-1]
            total_nnz = cu_seqlens[-1]  # total_nnz before sp padding
            query_states = query_states[:total_nnz]
            key_states = key_states[:total_nnz]
            value_states = value_states[:total_nnz]


        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        input_dtype = query_states.dtype

        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = (
                    self.q_proj.weight.dtype
                    if self.q_lora_rank is None
                    else self.q_a_proj.weight.dtype
                )

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # error [-1, 16, 128]' is invalid for input of size 2595840
        attn_output_unpad = flash_attn_varlen_func(
                query_states.contiguous().view(-1, self.num_heads, self.q_head_dim), #[total_nnz,1,num_heads, head_dim]
                key_states.contiguous().view(-1, self.num_heads, self.q_head_dim),
                value_states.view(-1, self.num_heads, self.q_head_dim), # 改为q_dim， 因为前面补齐了
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen_in_batch, # 此处建议是1？
                max_seqlen_k=max_seqlen_in_batch,
                dropout_p=dropout_rate,
                softmax_scale=None,
                causal=True,
            )
        attn_output_unpad = attn_output_unpad.to(input_dtype)
        # if torch.distributed.get_rank() == 0:
        #     # attn_output_unpad shape:torch.Size([845, 16, 192])
        #     print(f"attn_output_unpad shape:{attn_output_unpad.shape}")
        if self.q_head_dim != self.v_head_dim:
            # [845, 16, 192] -> [845, 16, 128]
            attn_output_unpad = attn_output_unpad[:, :, : self.v_head_dim]

        attn_output_unpad = attn_output_unpad.reshape(
            total_nnz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()

        # sequence parallel reduce_scatter is performed inside RowColumnParallel if enabled
        # Here we need to repad
        if self.megatron_config.sequence_parallel:
            attn_output_unpad = F.pad(attn_output_unpad, pad=(0, 0, 0, 0, 0, sequence_parallel_pad))

        attn_output_unpad = self.o_proj(attn_output_unpad)[0]
        return attn_output_unpad


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from:https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite/file/view/master?fileName=modeling_deepseek.py&status=1
# zsk add 20250410
class ParallelDeepseekV2AttentionRmpadV2(ParallelDeepseekV2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, megatron_config: ModelParallelConfig):
        super().__init__(config, megatron_config)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None: # dp2-lite true
            q = self.q_proj(hidden_states)[0]
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)[0]))[0]

        q = q.view(bsz, q_len, self.num_heads_per_tp, self.q_head_dim).transpose(1, 2)

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))[0]
            .view(bsz, q_len, self.num_heads_per_tp, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # if torch.distributed.get_rank() == 0:
        #     # err output bsz:272, q_len:8, q_pe.shape:torch.Size([272, 16, 8, 64]), k_pe.shape:torch.Size([272, 1, 8, 64]), sin.shape:torch.Size([8, 64]), cos.shape:torch.Size([8, 64]), position_ids.shape:torch.Size([8, 272])
        #     print(f"bsz:{bsz}, q_len:{q_len}, q_pe.shape:{q_pe.shape}, k_pe.shape:{k_pe.shape}, sin.shape:{sin.shape}, cos.shape:{cos.shape}, position_ids.shape:{position_ids.shape}")
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads_per_tp, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads_per_tp, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        # if past_key_value is not None:
        #     cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        #     key_states, value_states = past_key_value.update(
        #         key_states, value_states, self.layer_idx, cache_kwargs
        #     )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = 0. # self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (DeepseekV2RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = (
                    self.q_proj.weight.dtype
                    if self.q_lora_rank is None
                    else self.q_a_proj.weight.dtype
                )

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)[0]

        return attn_output

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in DeepseekV2FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=None,
                causal=True,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
