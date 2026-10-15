import os


if os.environ.get("VERL_OPD_VLLM_PATCH") == "1" and os.environ.get("VERL_OPD_VLLM_PATCH_DEFER") != "1":
    from speculative_decode import apply_patches
    apply_patches()
