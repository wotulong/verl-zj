# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.megatron_utils import print_rank_0, unwrap_model
from megatron.core import mpu
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallel as LocalDDP
from torch.nn.parallel import DistributedDataParallel as torchDDP
import torch
import time
from verl.models.deepseekv2.megatron.checkpoint_utils.model_pp_map import deepseekv2_lite_schedule

import torch
import torch.distributed as dist


# deepseekv2-lite 模型参数
# embed：
#     param: model.embed_tokens.weight     shape: torch.Size([102400, 2048])
# transformer：总共28层 （0-27）
#  0层，无专家
#  attention
#     param: model.layers.0.self_attn.q_proj.weight   shape: torch.Size([3072, 2048])
#     param: model.layers.0.self_attn.kv_a_proj_with_mqa.weight   shape: torch.Size([576, 2048])
#     param: model.layers.0.self_attn.kv_a_layernorm.weight   shape: torch.Size([512])
#     param: model.layers.0.self_attn.kv_b_proj.weight    shape: torch.Size([4096, 512])
#     param: model.layers.0.self_attn.o_proj.weight   shape: torch.Size([2048, 2048])
#  mlp
#     param: model.layers.0.mlp.gate_proj.weight      shape: torch.Size([10944, 2048])
#     param: model.layers.0.mlp.up_proj.weight    shape: torch.Size([10944, 2048])
#     param: model.layers.0.mlp.down_proj.weight      shape: torch.Size([2048, 10944])
#  其他
#     param: model.layers.0.input_layernorm.weight    shape: torch.Size([2048])
#     param: model.layers.0.post_attention_layernorm.weight   shape: torch.Size([2048])
## 1-17层有专家及共享专家
#   expert层：
#   attention层：
#     param: model.layers.1.self_attn.q_proj.weight   shape: torch.Size([3072, 2048])
#     param: model.layers.1.self_attn.kv_a_proj_with_mqa.weight   shape: torch.Size([576, 2048])
#     param: model.layers.1.self_attn.kv_a_layernorm.weight   shape: torch.Size([512])
#     param: model.layers.1.self_attn.kv_b_proj.weight    shape: torch.Size([4096, 512])
#     param: model.layers.1.self_attn.o_proj.weight   shape: torch.Size([2048, 2048])
#   expert 层：
#     param: model.layers.1.mlp.experts.0.gate_proj.weight    shape: torch.Size([1408, 2048])
#     param: model.layers.1.mlp.experts.0.up_proj.weight      shape: torch.Size([1408, 2048])
#     param: model.layers.1.mlp.experts.0.down_proj.weight shape: torch.Size([2048, 1408])
#   share expert层：
#     param: model.layers.1.mlp.shared_experts.gate_proj.weight   shape: torch.Size([2816, 2048])
#     param: model.layers.1.mlp.shared_experts.up_proj.weight     shape: torch.Size([2816, 2048])
#     param: model.layers.1.mlp.shared_experts.down_proj.weight   shape: torch.Size([2048, 2816])
#   其他
#     param: model.layers.1.input_layernorm.weight    shape: torch.Size([2048])
#     param: model.layers.1.post_attention_layernorm.weight   shape: torch.Size([2048])
# other:
#     param: model.norm.weight    shape: torch.Size([2048])
#     param: lm_head.weight   shape: torch.Size([102400, 2048])


# 获取全局rank id
def _megatron_calc_global_rank(tp_rank: int = 0, dp_rank: int = 0, pp_rank: int = 0):
    """given TP,DP,PP rank to get the global rank."""

    tp_size = mpu.get_tensor_model_parallel_world_size()
    dp_size = mpu.get_data_parallel_world_size()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    assert (tp_size * dp_size * pp_size == torch.distributed.get_world_size()
           ), f"{tp_size} x {dp_size} x {pp_size} != {torch.distributed.get_world_size()}"
    # We only support TP-DP-PP grouping, for correctness when resharding
    return (pp_rank * dp_size + dp_rank) * tp_size + tp_rank


def _megatron_calc_layer_map(config):
    """Calculate the mapping of global layer_idx to local layer_idx
    Returns:
        layer_map (Dict: int -> tuple(int, int, int)):
            mapping from the global layer index to
            a tuple of (pp_rank, virtual_pp_rank, layer_idx inside model)
    """
    from megatron.core import mpu

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1

    pp_schedule = deepseekv2_lite_schedule()
    # pp_size = len(pp_schedule)
    # virtual_pp_size = len(pp_schedule[0])
    assert len(pp_schedule) == pp_size
    assert len(pp_schedule[0]) == virtual_pp_size

    layers_pp = [sum(vp) for vp in pp_schedule]
    n_layers_schedule = sum(layers_pp)
    # print("\n\npp total layers:-----------------", n_layers_schedule)
    assert n_layers_schedule == config.num_hidden_layers, f"n_layers_schedule: {n_layers_schedule}, config.num_hidden_layers:{config.num_hidden_layers}"

    layer_map = dict()
    # num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size
    # assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers

    for pp_rank_idx in range(pp_size):
        for virtual_pp_rank_idx in range(virtual_pp_size):
            layer_offset = sum(layers_pp[:pp_rank_idx]) + sum(pp_schedule[pp_rank_idx][:virtual_pp_rank_idx])
            for layer_idx in range(pp_schedule[pp_rank_idx][virtual_pp_rank_idx]):
                layer_map[layer_offset + layer_idx] = (
                    pp_rank_idx,
                    virtual_pp_rank_idx,
                    layer_idx,
                )
    return layer_map


def merge_megatron_ckpt_deepseekv2(wrapped_models, config, dtype, is_value_model=False, tie_word_embeddings=False):
    """Merge sharded parameters of a Megatron module into a merged checkpoint.

    Args:
        wrapped_modelss (list of megatron.core.distributed.DistributedDataParallel):
            The local DDP wrapped megatron modules.
        config (str or None):
            HF config for model
        dtype: model params type
        is_value_model: if model is value model
        tie_word_embeddings: tie_word_embeddings
    Returns:
        state_dict (dict):
            The merged state_dict in rank 0, and an empty dictionary in other ranks.
    """
    start_time = time.time()

    def _get_gpt_model(model):
        return model

    dp_rank = mpu.get_data_parallel_rank()  # 在数据并行组中的编号
    pp_size = mpu.get_pipeline_model_parallel_world_size()  # 流水并行数量
    pp_rank = mpu.get_pipeline_model_parallel_rank()  # 在流水并行组中的编号
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1  # vp数量
    mp_group = mpu.get_model_parallel_group()  # 获取并行组rank信息，即一个dp组的rank id列表

    # 确认rank0是几个并行组的rann0
    if dist.get_rank() == 0:
        assert mp_group.rank() == 0, f"mp_rank:[{mp_group.rank}] != 0 on rank #0"
        assert pp_rank == 0, f"pp_rank:[{pp_rank}] != 0 on rank #0"
        assert dp_rank == 0, f"dp_rank:[{dp_rank}] != 0 on rank #0"

    if not isinstance(wrapped_models, (list, tuple)):
        wrapped_models = list(wrapped_models)

    assert len(wrapped_models) == virtual_pp_size
    # 获取指定模型pp及vp切分下的模型配置
    pp_schedule = deepseekv2_lite_schedule()
    layers_pp = [sum(vp) for vp in pp_schedule]
    n_layers_schedule = sum(layers_pp)
    assert n_layers_schedule == config.num_hidden_layers, f"n_layers_schedule:{n_layers_schedule}, config.num_hidden_layers:{config.num_hidden_layers}"

    # num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size
    # assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers

    models = [None] * len(wrapped_models)

    # 解析封装的网络层？
    for i, wrapped_model in enumerate(wrapped_models):
        models[i] = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))
        assert len(models[i].model.layers
                  ) == layers_pp[pp_rank], 'len model layers {} not equal to num_layers_per_model {}'.format(
                      len(models[i].model.layers), layers_pp[pp_rank])

    state_dict = dict()

    def _get_cpu_tensor(tensor: torch.Tensor):
        if tensor is None:
            return None
        if tensor.device == torch.device("cpu"):
            return tensor.detach().clone()
        return tensor.detach().cpu()

    def _broadcast_tensor(tensor, name, src_pp_rank) -> torch.Tensor:
        """broadcast tensor across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        # 非当前所属rank，不执行操作
        if torch.distributed.get_rank() == src_rank:
            if tensor is None:
                weight = None
                tensor_shape = None
            else:
                weight = tensor
                tensor_shape = weight.shape
        else:
            weight = None
            tensor_shape = None

        obj_list = [tensor_shape]
        # 将当前rank权重广播到整个数据组，mp为dp组rank列表
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        tensor_shape = obj_list[0]

        if tensor_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tensor:[{name}] not exist, skip collect")
            return

        if weight is None:
            weight = torch.empty(
                tensor_shape,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        # 当weight为None时，广播空张量？？
        dist.broadcast(weight, src=src_rank, group=mp_group)

        # 将权重保存到rank0的cpu中
        if torch.distributed.get_rank() == 0:
            state_dict[name] = _get_cpu_tensor(weight)

    def _broadcast_tp_shard_tensor(tensor, name, src_pp_rank, concat_dim=0, mutate_func=None) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        if torch.distributed.get_rank() == src_rank:
            if tensor is None:
                chunk_shape = None
            else:
                chunk_shape = tensor.shape
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{name}] not exist, skip collecting")
            return

        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        chunk_tensors = [None] * tp_size

        # 收集tensor分片
        for i in range(tp_size):
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank)
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            # 是当前
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            if torch.distributed.get_rank() == 0:
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        # 保存完整tensor
        if torch.distributed.get_rank() == 0:
            full_tensor = torch.concat(chunk_tensors, dim=concat_dim)
            if mutate_func is not None:
                full_tensor = mutate_func(full_tensor)
            state_dict[name] = full_tensor

    def _broadcast_tp_shard_tensor_gate_up(tensor, gate_name, up_name, src_pp_rank) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        if torch.distributed.get_rank() == src_rank:
            if tensor is None:
                chunk_shape = None
            else:
                chunk_shape = tensor.shape
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not exist, skip collecting")
            return

        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        chunk_tensors = [None] * tp_size

        for i in range(tp_size):
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank)
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            if torch.distributed.get_rank() == 0:
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        # 因为gate_weight和up_weight合并了，所以需要切分还原
        if torch.distributed.get_rank() == 0:
            full_tensor = torch.concat(chunk_tensors, dim=0)
            intermediate_size_tp = config.intermediate_size // tp_size
            gate_weight_list = []
            up_weight_list = []
            for i in range(tp_size):
                gate_up_weight_tp = full_tensor[intermediate_size_tp * 2 * i:intermediate_size_tp * 2 * (i + 1)]
                gate_weight_tp = gate_up_weight_tp[:intermediate_size_tp]
                up_weight_tp = gate_up_weight_tp[intermediate_size_tp:]
                gate_weight_list.append(gate_weight_tp)
                up_weight_list.append(up_weight_tp)

            state_dict[gate_name] = torch.cat(gate_weight_list, dim=0)
            state_dict[up_name] = torch.cat(up_weight_list, dim=0)

    def _broadcast_tp_shard_tensor_gate_up_moe(tensor, gate_name, up_name, src_pp_rank, scale=1) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        if torch.distributed.get_rank() == src_rank:
            if tensor is None:
                chunk_shape = None
            else:
                chunk_shape = tensor.shape
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not exist, skip collecting")
            return

        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        chunk_tensors = [None] * tp_size

        for i in range(tp_size):
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank)
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            if torch.distributed.get_rank() == 0:
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        # 因为gate_weight和up_weight合并了，所以需要切分还原
        if torch.distributed.get_rank() == 0:
            full_tensor = torch.concat(chunk_tensors, dim=0)
            intermediate_size_tp = config.moe_intermediate_size * scale // tp_size
            gate_weight_list = []
            up_weight_list = []
            for i in range(tp_size):
                gate_up_weight_tp = full_tensor[intermediate_size_tp * 2 * i:intermediate_size_tp * 2 * (i + 1)]
                gate_weight_tp = gate_up_weight_tp[:intermediate_size_tp]
                up_weight_tp = gate_up_weight_tp[intermediate_size_tp:]
                gate_weight_list.append(gate_weight_tp)
                up_weight_list.append(up_weight_tp)

            state_dict[gate_name] = torch.cat(gate_weight_list, dim=0)
            state_dict[up_name] = torch.cat(up_weight_list, dim=0)

    def _broadcast_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name, src_pp_rank):
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        if torch.distributed.get_rank() == src_rank:
            if tensor is None:
                chunk_shape = None
            else:
                chunk_shape = tensor.shape
        else:
            chunk_shape = None

        obj_list = [chunk_shape]
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        chunk_shape = obj_list[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{q_name}] not exist, skip collecting")
            return

        buffer_tensor = torch.empty(
            chunk_shape,
            dtype=dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        chunk_tensors = [None] * tp_size

        for i in range(tp_size):
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank)
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            if torch.distributed.get_rank() == 0:
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        if torch.distributed.get_rank() == 0:
            full_tensor = torch.concat(chunk_tensors, dim=0)
            q_weight_list = []
            k_weight_list = []
            v_weight_list = []
            hidden_size_per_head = config.hidden_size // config.num_attention_heads

            if config.num_key_value_heads >= tp_size:
                q_size_tp = config.hidden_size // tp_size
                kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
                total_size = q_size_tp + 2 * kv_size_tp
                for i in range(tp_size):
                    qkv_part = full_tensor[i * total_size:(i + 1) * total_size]
                    q_part = qkv_part[:q_size_tp]
                    k_part = qkv_part[q_size_tp:q_size_tp + kv_size_tp]
                    v_part = qkv_part[q_size_tp + kv_size_tp:total_size]
                    q_weight_list.append(q_part)
                    k_weight_list.append(k_part)
                    v_weight_list.append(v_part)
            else:
                q_size_tp = config.hidden_size // tp_size
                kv_size_tp = hidden_size_per_head
                total_size = q_size_tp + 2 * kv_size_tp
                for i in range(tp_size):
                    qkv_part = full_tensor[i * total_size:(i + 1) * total_size]
                    q_part = qkv_part[:q_size_tp]
                    k_part = qkv_part[q_size_tp:q_size_tp + kv_size_tp]
                    v_part = qkv_part[q_size_tp + kv_size_tp:total_size]
                    q_weight_list.append(q_part)
                    if i * config.num_key_value_heads % tp_size == 0:
                        k_weight_list.append(k_part)
                        v_weight_list.append(v_part)

            state_dict[q_name] = torch.cat(q_weight_list, dim=0)
            state_dict[k_name] = torch.cat(k_weight_list, dim=0)
            state_dict[v_name] = torch.cat(v_weight_list, dim=0)

    # empty cache before collecting weights
    # 释放当前未使用的张量，以释放GPU显存。
    torch.cuda.empty_cache()
    # Embeddings
    # -------------------
    if dp_rank == 0:
        # Embeddings
        # -------------------
        # print_rank_0("collecting embeddings...")
        gpt_model_module = _get_gpt_model(models[0])
        _broadcast_tp_shard_tensor(
            gpt_model_module.model.embed_tokens.weight if pp_rank == 0 else None,
            "model.embed_tokens.weight",
            src_pp_rank=0,
        )

        # dp2 不同层不一样架构，需要通过pp rank进行额外判断
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # Transformer layers
        # -------------------
        layer_map = _megatron_calc_layer_map(config)
        for layer in range(config.num_hidden_layers):
            # print_rank_0(f"collecting layer #{layer}...")
            layer_name = f"model.layers.{layer}"
            src_pp_rank, src_virtual_pp_rank, src_layer_idx = layer_map[layer]

            gpt_model_module = _get_gpt_model(models[src_virtual_pp_rank])
            # 加个判断
            sync_layer = gpt_model_module.model.layers[src_layer_idx] if pp_rank == src_pp_rank else None

            # 第一层和其他层都有的，包括MLA和input，output
            _broadcast_tensor(
                sync_layer.input_layernorm.weight if pp_rank == src_pp_rank else None,
                f"{layer_name}.input_layernorm.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attn.q_proj.weight if pp_rank == src_pp_rank else None,
                f"{layer_name}.self_attn.q_proj.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tensor(
                sync_layer.self_attn.kv_a_proj_with_mqa.weight if pp_rank == src_pp_rank else None,
                f"{layer_name}.self_attn.kv_a_proj_with_mqa.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attn.kv_a_layernorm.weight if pp_rank == src_pp_rank else None,
                f"{layer_name}.self_attn.kv_a_layernorm.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attn.kv_b_proj.weight if pp_rank == src_pp_rank else None,
                f"{layer_name}.self_attn.kv_b_proj.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attn.o_proj.weight if pp_rank == src_pp_rank else None,
                f"{layer_name}.self_attn.o_proj.weight",
                src_pp_rank=src_pp_rank,
            )

            _broadcast_tensor(
                sync_layer.post_attention_layernorm.weight if pp_rank == src_pp_rank else None,
                f"{layer_name}.post_attention_layernorm.weight",
                src_pp_rank=src_pp_rank,
            )
            # 第一层：MHA网络
            if 0 == layer:
                # print(f"=================>>src_pp_rank:{src_pp_rank}, src_layer_idx:{src_layer_idx}")
                _broadcast_tp_shard_tensor_gate_up(sync_layer.mlp.gate_up_proj.weight if pp_rank == src_pp_rank else None,
                                                   f"{layer_name}.mlp.gate_proj.weight",
                                                   f"{layer_name}.mlp.up_proj.weight",
                                                   src_pp_rank=src_pp_rank)
                _broadcast_tp_shard_tensor(
                    sync_layer.mlp.down_proj.weight if pp_rank == src_pp_rank else None,
                    f"{layer_name}.mlp.down_proj.weight",
                    concat_dim=1,
                    src_pp_rank=src_pp_rank,
                )
            # 其他层，混合专家网络
            else:
                # 普通专家
                for expert_id in range(config.n_routed_experts):
                    expert_name = f"experts.{expert_id}"
                    _broadcast_tp_shard_tensor_gate_up_moe(sync_layer.mlp.experts[expert_id].gate_up_proj.weight if pp_rank == src_pp_rank else None,
                                                       f"{layer_name}.mlp.{expert_name}.gate_proj.weight",
                                                       f"{layer_name}.mlp.{expert_name}.up_proj.weight",
                                                       src_pp_rank=src_pp_rank)
                    _broadcast_tp_shard_tensor(
                        sync_layer.mlp.experts[expert_id].down_proj.weight if pp_rank == src_pp_rank else None,
                        f"{layer_name}.mlp.{expert_name}.down_proj.weight",
                        concat_dim=1,
                        src_pp_rank=src_pp_rank,
                    )
                # gate
                _broadcast_tensor(
                    sync_layer.mlp.gate.weight if pp_rank == src_pp_rank else None,
                    f"{layer_name}.mlp.gate.weight",
                    src_pp_rank=src_pp_rank,
                )

                # 共享专家
                _broadcast_tp_shard_tensor_gate_up_moe(sync_layer.mlp.shared_experts.gate_up_proj.weight if pp_rank == src_pp_rank else None,
                                                   f"{layer_name}.mlp.shared_experts.gate_proj.weight",
                                                   f"{layer_name}.mlp.shared_experts.up_proj.weight",
                                                   src_pp_rank=src_pp_rank, scale=2)
                _broadcast_tp_shard_tensor(
                    sync_layer.mlp.shared_experts.down_proj.weight if pp_rank == src_pp_rank else None,
                    f"{layer_name}.mlp.shared_experts.down_proj.weight",
                    concat_dim=1,
                    src_pp_rank=src_pp_rank,
                )

        # Final Layernorm
        # -------------------
        # print_rank_0("collecting final layernorm...")
        gpt_model_module = _get_gpt_model(models[-1])
        _broadcast_tensor(
            getattr(gpt_model_module.model.norm, "weight", None),
            "model.norm.weight",
            src_pp_rank=pp_size - 1,
        )

        if tie_word_embeddings:
            print_rank_0(f"tie word embedding skip load lm_head...")
        else:
            print_rank_0("collecting lm_head...")

            if is_value_model:
                _broadcast_tensor(getattr(gpt_model_module.lm_head, "weight", None) if pp_rank == pp_size - 1 else None,
                                  "reward_head.weight",
                                  src_pp_rank=pp_size - 1)

            else:
                _broadcast_tp_shard_tensor(
                    getattr(gpt_model_module.lm_head, "weight", None) if pp_rank == pp_size - 1 else None,
                    "lm_head.weight",
                    src_pp_rank=pp_size - 1,
                )

    dist.barrier()

    torch.cuda.empty_cache()
    if torch.distributed.get_rank() == 0:

        for k, v in state_dict.items():
            if dtype != v.dtype:
                state_dict[k] = v.to(dtype)

    print_rank_0(f"merge megatron ckpt done, time elapsed {time.time() - start_time}s")
    return state_dict
