def deepseekv2_lite_schedule():
    from megatron.core import mpu
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # 由外到内分别为：dp组，pp组，vp组
    if pp_size == 8:
        test_pp_schedule = [[4],[4],[4],[3],[3],[3],[3],[3]]
    elif pp_size == 4:
        test_pp_schedule = [[8],[7],[6],[6]]
    elif pp_size == 3:
        test_pp_schedule = [[9],[9],[9]]
    elif pp_size == 2:
        test_pp_schedule = [[14],[13]]
    elif pp_size == 1:
        test_pp_schedule = [[27]]
    else:
        raise f"PP size {pp_size} is not support for deepseekv2-lite now!"
        
    return test_pp_schedule

MODEL_PP_SCHEDULE={
    "deepseek_v2": deepseekv2_lite_schedule(),

}