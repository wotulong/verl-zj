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

import torch
import time
from typing import Dict, Any, Callable, Optional
import torch.distributed as dist

from verl.models.deepseekv2.megatron.checkpoint_utils.model_pp_map import deepseekv2_lite_schedule
from tqdm import tqdm


# deepseekv2-lite 模型参数
# embed：
#     param: model.embed_tokens.weight     shape: torch.Size([102400, 2048])
# transformer：总共18层 （0-17）
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
## 1-26层有专家及共享专家
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
#     param: model.layers.1.mlp.gate.weight   shape: torch.Size([64, 2048])
#     param: model.layers.1.input_layernorm.weight    shape: torch.Size([2048])
#     param: model.layers.1.post_attention_layernorm.weight   shape: torch.Size([2048])
# other:
#     param: model.norm.weight    shape: torch.Size([2048])
#     param: lm_head.weight   shape: torch.Size([102400, 2048])

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


def load_state_dict_to_megatron_deepseekv2(state_dict,
                                           wrapped_models,
                                           config,
                                           params_dtype,
                                           is_value_model=False,
                                           tie_word_embeddings=False):
    """Load merged state_dict to sharded Megatron module in training.
    """
    from verl.utils.megatron_utils import print_rank_0, unwrap_model
    from megatron.core import mpu
    from megatron.core.transformer.module import Float16Module
    from megatron.core import DistributedDataParallel as LocalDDP
    from torch.nn.parallel import DistributedDataParallel as torchDDP

    start_time = time.time()

    def _get_gpt_model(model):
        return model

    def broadcast_params(module):
        for param in module.parameters():
            torch.distributed.broadcast(param.data,
                                        src=mpu.get_data_parallel_src_rank(),
                                        group=mpu.get_data_parallel_group())

    dp_rank = mpu.get_data_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    mp_group = mpu.get_model_parallel_group()

    if torch.distributed.get_rank() == 0:
        assert mp_group.rank() == 0, f"mp_rank:[{mp_group.rank}] != 0 on rank #0"
        assert pp_rank == 0, f"pp_rank:[{pp_rank}] != 0 on rank #0"
        assert dp_rank == 0, f"dp_rank:[{dp_rank}] != 0 on rank #0"

    # wrapped_models: 当前rank持有的模型
    if not isinstance(wrapped_models, (list, tuple)):
        wrapped_models = list(wrapped_models)

    assert len(wrapped_models) == virtual_pp_size

    # 获取指定模型pp及vp切分下的模型配置
    pp_schedule = deepseekv2_lite_schedule()
    layers_pp = [sum(vp) for vp in pp_schedule]
    # num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size
    # vw-lite均匀切分报错：AssertionError: num_layers_per_model: 13 * pp_size: 2 * virtual_pp_size: 1 != config.num_hidden_layers: 27
    # assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers, f'num_layers_per_model: {num_layers_per_model} * pp_size: {pp_size} * virtual_pp_size: {virtual_pp_size} != config.num_hidden_layers: {config.num_hidden_layers}'

    models = [None] * len(wrapped_models)

    for i, wrapped_model in enumerate(wrapped_models):
        models[i] = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))
        gpt_model_module = _get_gpt_model(models[i])
        # print_rank_0(f'=============================>len(gpt_model_module.model.layers): {len(gpt_model_module.model.layers)}, num_layers_per_model:{layers_pp[pp_rank]}, pp_rank: {pp_rank}')
        assert len(gpt_model_module.model.layers) == layers_pp[
            pp_rank], f'len(gpt_model_module.model.layers): {len(gpt_model_module.model.layers)}, num_layers_per_model:{layers_pp[pp_rank]}, pp_rank: {pp_rank}'

    def _broadcast_tensor(tensor, name, do_broadcast=True) -> torch.Tensor:
        """broadcast tensor from rank0 across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        if torch.distributed.get_rank() == 0:
            if name in state_dict:
                weight = state_dict[name]
                tensor_shape = weight.shape
            else:
                tensor_shape = None
        else:
            weight = None
            tensor_shape = None

        obj_list_bt = [tensor_shape]
        if do_broadcast:
            # print(f"===============>>> broadcast_object_list:{obj_list}, group:{mp_group}")
            dist.broadcast_object_list(obj_list_bt, src=0, group=mp_group)
        tensor_shape = obj_list_bt[0]

        if tensor_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tensor:[{name}] not in state_dict, skip load")
            return

        if tensor is None:
            tensor = torch.empty(
                tensor_shape,
                dtype=params_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        if torch.distributed.get_rank() == 0:
            tensor.data.copy_(weight)
        dist.broadcast(tensor, src=0, group=mp_group)

    def _broadcast_tp_shard_tensor_vocab(tensor, name, chunk_dim=0, mutate_func=None, do_broadcast=True) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == 0:
            if name in state_dict:
                full_weight = state_dict[name]

                if mutate_func is not None:
                    full_weight = mutate_func(full_weight)
                tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
                chunk_shape = tensor_chunk[0].shape
            else:
                chunk_shape = None
        else:
            chunk_shape = None

        obj_list_vocab = [chunk_shape]
        if do_broadcast:
            dist.broadcast_object_list(obj_list_vocab, src=0, group=mp_group)
        chunk_shape = obj_list_vocab[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        else:
            assert (tensor.shape == chunk_shape
                    ), f"rank #{torch.distributed.get_rank()} tensor {name} shape {tensor.shape} != {chunk_shape}"
            sync_tensor = torch.empty_like(tensor, device=torch.cuda.current_device(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == 0:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    def _broadcast_tp_shard_tensor(tensor, name, chunk_dim=0, mutate_func=None, do_broadcast=True) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == 0:
            if name in state_dict:
                full_weight = state_dict[name]
                if mutate_func is not None:
                    full_weight = mutate_func(full_weight)
                tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
                chunk_shape = tensor_chunk[0].shape

                # tp_size: 2, param name: model.layers.0.self_attn.q_proj.weight, full_weight size: torch.Size([3072, 2048]
                # print(f"\n\n============>>tp_size: {tp_size}, param name: {name}, full_weight size: {full_weight.shape}, chunk_shape: {chunk_shape}")
            else:
                chunk_shape = None
        else:
            chunk_shape = None

        obj_list_tst = [chunk_shape]
        
        if do_broadcast:
            dist.broadcast_object_list(obj_list_tst, src=0, group=mp_group)
        chunk_shape = obj_list_tst[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        else:
            assert (tensor.shape == chunk_shape
                    ), f"rank #{torch.distributed.get_rank()} tensor {name} shape {tensor.shape} != {chunk_shape}"
            sync_tensor = torch.empty_like(tensor, device=torch.cuda.current_device(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == 0:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    def _broadcast_tp_shard_tensor_gate_up(tensor, gate_name, up_name, do_broadcast=True) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if torch.distributed.get_rank() == 0:
            if gate_name in state_dict and up_name in state_dict:
                gate_weight = state_dict[gate_name]
                up_weight = state_dict[up_name]
                new_gate_up_weight = torch.empty(config.intermediate_size * 2,
                                                config.hidden_size,
                                                dtype=params_dtype,
                                                device=torch.cuda.current_device())
                for i in range(tp_size):
                    intermediate_size_tp = config.intermediate_size // tp_size
                    gate_weight_tp = gate_weight[i * intermediate_size_tp:(i + 1) * intermediate_size_tp]
                    up_weight_tp = up_weight[i * intermediate_size_tp:(i + 1) * intermediate_size_tp]
                    new_gate_up_weight[intermediate_size_tp * 2 * i:intermediate_size_tp * 2 * (i + 1)].copy_(
                        torch.cat([gate_weight_tp, up_weight_tp], dim=0))

                tensor_chunk = torch.chunk(new_gate_up_weight, tp_size, dim=0)
                chunk_shape = tensor_chunk[0].shape
            else:
                chunk_shape = None
        else:
            chunk_shape = None

        obj_list_gate_up = [chunk_shape]
        if do_broadcast:
            dist.broadcast_object_list(obj_list_gate_up, src=0, group=mp_group)
        chunk_shape = obj_list_gate_up[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not in state_dict, skip loading")
            return

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        else:
            assert (
                    tensor.shape == chunk_shape
            ), f"rank #{torch.distributed.get_rank() == 0:} tensor {gate_name, up_name} shape {tensor.shape} != {chunk_shape}"
            sync_tensor = torch.empty_like(tensor, device=torch.cuda.current_device(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == 0:
                sync_tensor.data.copy_(tensor_chunk[i])
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    def _broadcast_tp_shard_tensor_gate_up_moe(tensor, gate_name, up_name, scale=1, do_broadcast=True) -> torch.Tensor:
        """broadcast tensor in tp shards across mp_group"""
        nonlocal state_dict
        nonlocal mp_group
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()

        if torch.distributed.get_rank() == 0:
            if gate_name in state_dict and up_name in state_dict:
                gate_weight = state_dict[gate_name]
                up_weight = state_dict[up_name]
                # print(f"\n\n+++++++++++++++++++++:{config.moe_intermediate_size}, scale: {scale}\n\n")
                new_gate_up_weight = torch.empty(config.moe_intermediate_size * 2 * scale,
                                                config.hidden_size,
                                                dtype=params_dtype,
                                                device=torch.cuda.current_device())
                for i in range(tp_size):
                    intermediate_size_tp = config.moe_intermediate_size * scale // tp_size
                    gate_weight_tp = gate_weight[i * intermediate_size_tp:(i + 1) * intermediate_size_tp]
                    up_weight_tp = up_weight[i * intermediate_size_tp:(i + 1) * intermediate_size_tp]
                    new_gate_up_weight[intermediate_size_tp * 2 * i:intermediate_size_tp * 2 * (i + 1)].copy_(
                        torch.cat([gate_weight_tp, up_weight_tp], dim=0))

                tensor_chunk = torch.chunk(new_gate_up_weight, tp_size, dim=0)
                chunk_shape = tensor_chunk[0].shape
            else:
                chunk_shape = None
        else:
            chunk_shape = None

        obj_list_gate_up_moe = [chunk_shape]
        if do_broadcast:
            dist.broadcast_object_list(obj_list_gate_up_moe, src=0, group=mp_group)

        chunk_shape = obj_list_gate_up_moe[0]
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not in state_dict, skip loading")
            return
        
        # if chunk_shape is not None:
        #     chunk_shape = torch.Size([1408, 2048])

        if tensor is None:
            sync_tensor = torch.empty(
                chunk_shape,
                dtype=params_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        else:
            assert (
                    tensor.shape == chunk_shape
            ), f"rank #{torch.distributed.get_rank()} tensor {gate_name, up_name} shape {tensor.shape} != {chunk_shape}, tp_rank:{tp_rank}, pp_rank:{pp_rank}"
            sync_tensor = torch.empty_like(tensor, device=torch.cuda.current_device(), requires_grad=False)

        for i in range(tp_size):
            if torch.distributed.get_rank() == 0:
                # print(f"++++++++++++++++++++++ chunk shape: {tensor_chunk[0].shape}")
                sync_tensor.data.copy_(tensor_chunk[i])
            # print(f"++++++++++++++++++++++ chunk shape: sync_tensor:{sync_tensor}")
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            if (i == tp_rank) and (tensor is not None):
                tensor.data.copy_(sync_tensor)

    if dp_rank == 0:
        # Embeddings
        # -------------------
        print_rank_0("loading embeddings...")
        gpt_model_module = _get_gpt_model(models[0])
        embed_tokens_weight = None
        if pp_rank == 0:
            embed_tokens_weight = gpt_model_module.model.embed_tokens.weight
        _broadcast_tp_shard_tensor_vocab(embed_tokens_weight, "model.embed_tokens.weight")

        # Transformer layers
        # -------------------
        layer_map = _megatron_calc_layer_map(config)

        # 获取当前pp段拥有哪些层
        pp_schedule = deepseekv2_lite_schedule()
        layers_pp = [sum(vp) for vp in pp_schedule]
        layer_offset = sum(layers_pp[:pp_rank])

        start_layer = 1 if layer_offset == 0 else layer_offset
        for layer in range(config.num_hidden_layers):
        # for layer in tqdm(range(start_layer, layer_offset + layers_pp[pp_rank])):
            # print_rank_0(f"==================>loading layer #{layer}..., model.layers.{layer}, pp_rank:{pp_rank}, layer_offset: {layer_offset}")

            dst_pp_rank, dst_virtual_pp_rank, dst_layer_idx = layer_map[layer]
            layer_name = f"model.layers.{layer}"

            # assert dst_layer_idx < len(gpt_model_module.model.layers), f"dst_layer_idx: {dst_layer_idx}, len(gpt_model_module.model.layers)：{len(gpt_model_module.model.layers)}， gpt_model_module.model.layers：{gpt_model_module.model.layers}， pp_rank：{pp_rank}"
            if dst_layer_idx >= len(gpt_model_module.model.layers):
                dst_layer_idx = -1
            sync_layer = gpt_model_module.model.layers[dst_layer_idx]

            gpt_model_module = _get_gpt_model(models[dst_virtual_pp_rank])
            # if torch.distributed.get_rank() == 0:
            #     print(f'\n\n=================>sync_layer:{sync_layer}, layer_name:{layer_name}')
            #     print_rank_0(f"++++++++++++++++ dst_pp_rank:{dst_pp_rank}, dst_virtual_pp_rank:{dst_virtual_pp_rank}, dst_layer_idx:{dst_layer_idx},  layer:{layer}, len_layers:{len(gpt_model_module.model.layers)}")

            # 第一层和其他层都有的，包括MLA和input，output
            _broadcast_tensor(
                sync_layer.input_layernorm.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.input_layernorm.weight"
            )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attn.q_proj.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.self_attn.q_proj.weight",
                chunk_dim=0
            )

            # _broadcast_tp_shard_tensor(
            #     sync_layer.self_attn.kv_a_proj_with_mqa.weight if dst_pp_rank == pp_rank else None,
            #     f"{layer_name}.self_attn.kv_a_proj_with_mqa.weight",
            #     chunk_dim=0
            # )
            _broadcast_tensor(
                sync_layer.self_attn.kv_a_proj_with_mqa.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.self_attn.kv_a_proj_with_mqa.weight",
            )

            _broadcast_tensor(
                sync_layer.self_attn.kv_a_layernorm.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.self_attn.kv_a_layernorm.weight"
            )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attn.kv_b_proj.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.self_attn.kv_b_proj.weight",
                chunk_dim=0
            )

            _broadcast_tp_shard_tensor(
                sync_layer.self_attn.o_proj.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.self_attn.o_proj.weight",
                chunk_dim=1
            )

            _broadcast_tensor(
                sync_layer.post_attention_layernorm.weight if dst_pp_rank == pp_rank else None,
                f"{layer_name}.post_attention_layernorm.weight"
            )

            # 第一层：MHA网络
            if 0 == layer:
                _broadcast_tp_shard_tensor_gate_up(
                    sync_layer.mlp.gate_up_proj.weight if dst_pp_rank == pp_rank else None,
                    f"{layer_name}.mlp.gate_proj.weight", f"{layer_name}.mlp.up_proj.weight")

                _broadcast_tp_shard_tensor(
                    sync_layer.mlp.down_proj.weight if dst_pp_rank == pp_rank else None,
                    f"{layer_name}.mlp.down_proj.weight",
                    chunk_dim=1
                )
            # 其他层，混合专家网络
            else:
                # 路由专家
                for expert_id in range(config.n_routed_experts):
                    expert_name = f"experts.{expert_id}"
                    # if torch.distributed.get_rank() == 0:
                    #     print(f'sync_layer:{sync_layer}, layer_name:{layer_name}')
                    # print(f"=================>>> layer_name_494: {layer_name}, dst_pp_rank:{dst_pp_rank}, pp_rank:{pp_rank}, sync_layer:{sync_layer}")
                    _broadcast_tp_shard_tensor_gate_up_moe(
                        sync_layer.mlp.experts[expert_id].gate_up_proj.weight if dst_pp_rank == pp_rank else None,
                        f"{layer_name}.mlp.{expert_name}.gate_proj.weight",
                        f"{layer_name}.mlp.{expert_name}.up_proj.weight")

                    _broadcast_tp_shard_tensor(
                        sync_layer.mlp.experts[expert_id].down_proj.weight if dst_pp_rank == pp_rank else None,
                        f"{layer_name}.mlp.{expert_name}.down_proj.weight",
                        chunk_dim=1,
                    )
                # gate
                _broadcast_tensor(
                    sync_layer.mlp.gate.weight if dst_pp_rank == pp_rank else None,
                    f"{layer_name}.mlp.gate.weight",
                )

                # 共享专家
                _broadcast_tp_shard_tensor_gate_up_moe(
                    sync_layer.mlp.shared_experts.gate_up_proj.weight if dst_pp_rank == pp_rank else None,
                    f"{layer_name}.mlp.shared_experts.gate_proj.weight",
                    f"{layer_name}.mlp.shared_experts.up_proj.weight", scale=2)

                _broadcast_tp_shard_tensor(
                    sync_layer.mlp.shared_experts.down_proj.weight if dst_pp_rank == pp_rank else None,
                    f"{layer_name}.mlp.shared_experts.down_proj.weight",
                    chunk_dim=1,
                )

        # if pp_rank + 1 == pp_size:
        # Final Layernorm
        # -------------------
        print_rank_0("loading final layernorm...")
        gpt_model_module = _get_gpt_model(models[-1])
        _broadcast_tensor(
            getattr(gpt_model_module.model.norm, "weight", None),
            "model.norm.weight",
        )

        if False:#tie_word_embeddings:
            print_rank_0("tie_word_embeddings skip load lm_head")
        else:
            print_rank_0("loading lm_head...")
            lm_head_weight = None
            if pp_rank + 1 == pp_size:
                lm_head_weight = gpt_model_module.lm_head.weight

            if is_value_model:
                # if torch.distributed.get_rank() == 0:
                if 'lm_head.weight' in state_dict and state_dict['lm_head.weight'].shape[0] == 1:
                    _broadcast_tensor(lm_head_weight, "lm_head.weight")
                elif 'reward_head.weight' in state_dict and state_dict['reward_head.weight'].shape[0] == 1:
                    _broadcast_tensor(lm_head_weight, "reward_head.weight")
                    print_rank_0('load lm_head from value_head weight')
                else:
                    _broadcast_tensor(None, "lm_head.weight")
                    print_rank_0('fail to match lm_head in value_model')
                # else:

                #     _broadcast_tensor(lm_head_weight, "lm_head.weight")

            else:
                _broadcast_tp_shard_tensor(lm_head_weight, "lm_head.weight")

    dist.barrier()
    print("--------------->> broadcast param to dp group!!!!!!!!")
    # Broadcast weights inside data parallel groups
    for wrapped_model in wrapped_models:
        broadcast_params(wrapped_model)

    torch.cuda.empty_cache()
    print_rank_0(f"loading megatron ckpt done, time elapsed {time.time() - start_time}s")
