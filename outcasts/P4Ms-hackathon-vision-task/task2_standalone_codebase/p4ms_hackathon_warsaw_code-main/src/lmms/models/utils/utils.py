import os
from typing import List


def get_adapter(target_modules: List[str], adapter_type: str, model):
    from src.lmms.models.adaptations import AdapterConfig, get_peft_model

    peft_config = AdapterConfig(
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        adapter_type=adapter_type,
    )
    return get_peft_model(model, peft_config)


def write_model_info(training_args, model):
    with open(
        os.path.join(training_args.output_dir, "model_trainable_params.txt"), "w"
    ) as f:
        f.write("\n")
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            with open(
                os.path.join(training_args.output_dir, "model_trainable_params.txt"),
                "a",
            ) as f:
                f.write(name + "  " + str(param.shape))
                f.write("\n")
            params.append(param.numel())
    trainable_params = sum(params) / 1e6
    with open(
        os.path.join(training_args.output_dir, "model_trainable_params.txt"), "a"
    ) as f:
        f.write(f"trainable_params: {trainable_params:.3f}MB")
    print(f"trainable_params: {trainable_params:.3f}MB")

    with open(os.path.join(training_args.output_dir, "model.txt"), "w") as f:
        f.write(str(model))


def get_compute_type(training_args):
    import torch

    compute_dtype = torch.float32
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16
    return compute_dtype


def enable_gradient_checkpointing(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def pretty_print_missing_keys_get_len(keys: List[str], module_name: str):
    import torch.distributed as dist

    new_keys = [k for k in keys if k in module_name]

    is_main = True
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            is_main = False

    if is_main:
        if len(new_keys) == 0:
            print(f"No missing keys in {module_name}")
        for k in new_keys:
            print(f"Missing key in {module_name}: {k}")
    return len(new_keys)


def vpgtrans_to_moka(vpgtrans_path: str, moka_path: str, is_audio: bool = False):
    if os.path.exists(moka_path):
        print("VPGTrans Qformer already transformed to MokA-compatible version")
        return

    import torch

    prefix = "model.al_projector." if is_audio else "model.vl_projector."

    os.makedirs(os.path.dirname(moka_path), exist_ok=True)
    ckpt = torch.load(vpgtrans_path, map_location="cpu")
    new_ckpt = dict()
    for k, v in ckpt["model"].items():
        new_key = prefix + k
        new_ckpt[new_key] = v

    torch.save(new_ckpt, moka_path)
    print("VPGTrans Qformer transformed to MokA-compatible version")


def maybe_adjust_ckpt_keys(ckpt: dict) -> dict:
    new_ckpt = dict()
    for k, v in ckpt.items():
        if k.startswith("base_model.model."):
            k = k[len("base_model.model.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        new_ckpt[k] = v
    return new_ckpt
