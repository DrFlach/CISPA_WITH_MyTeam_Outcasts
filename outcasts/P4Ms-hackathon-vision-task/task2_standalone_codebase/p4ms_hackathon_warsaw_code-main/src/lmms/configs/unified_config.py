from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Tuple, Union, Dict, Any
import transformers, os, json
import yaml

method_names = [
    f.split(".")[0] for f in os.listdir("config/models") if f.endswith(".yaml")
]
configs = dict()
for method_name in method_names:
    with open(f"config/models/{method_name}.yaml", mode="r") as fd:
        cfg = yaml.safe_load(fd)
        configs[method_name] = cfg
with open(f"config/models/blueprints.yaml", mode="r") as fd:
    blueprints = yaml.safe_load(fd)


@dataclass
class GeneralArgs:
    config: str = field(default=None)
    method: str = field(default=None)

    output_hidden_states: bool = field(default=False)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llm_name: str = field(default=None)
    adapter_type: str = field(default=None)
    inputs_embeds_with_mmask: bool = field(default=False)

    speech_proj_ckpt_dir: str = field(default=None)
    visual_proj_ckpt_dir: str = field(default=None)
    pretrained_ckpt_path: str = field(default=None)

    visual_encoder_type: str = field(default=None)
    speech_encoder_type: str = field(default=None)

    d_model: int = field(default=4096)


@dataclass
class DataArguments:
    video_frame_nums: int = field(default=8)
    data_image_size: int = field(default=None)
    multi_frames: bool = field(default=False)

    tasks: str = field(default=None)
    mel_size: int = field(default=128)

    is_test: bool = field(default=False)
    pack_conversations: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # Training Data Arguments
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=32, metadata={"help": "How many bits to use."})

    lora_trainable: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"

    visual_branch: bool = field(default=False)
    speech_branch: bool = field(default=False)

    save_modules: str = field(default="vl_projector,al_projector,lora")

    exp_desc: str = field(default="exp")
    global_adaptive_clipping: bool = field(default=False)


def baseline_instances():
    bm = ModelArguments()
    bd = DataArguments()
    bt = TrainingArguments(output_dir="runs/tmp")
    return bm, bd, bt


def overwrite_args(cli, baseline, config_args: dict):
    for f in fields(cli):
        v_cli = getattr(cli, f.name)
        v_base = getattr(baseline, f.name)
        if v_cli != v_base:
            if v_cli == "None":
                v_cli = None
            setattr(cli, f.name, v_cli)
        else:
            if f.name in config_args:
                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    print(
                        f"Overwriting {f.name} from {v_base} to {config_args[f.name]}"
                    )
                setattr(cli, f.name, config_args[f.name])


def overwrite_all_args(clis, baselines, config_args):
    if "model_args" in config_args:
        overwrite_args(clis[0], baselines[0], config_args["model_args"])
    if "data_args" in config_args:
        overwrite_args(clis[1], baselines[1], config_args["data_args"])
    if "training_args" in config_args:
        overwrite_args(clis[2], baselines[2], config_args["training_args"])


def recursive_dict_update(child, parent) -> dict:
    # This is basically to update the final config with the parent default/blueprint config, with inheritance support.
    # The child config has higher priority than the parent config, but if the child config does not have a key that the parent config has, then we will use the value from the parent config.
    for k in parent:
        if k not in child:
            child[k] = parent[k]
    for k, v in child.items():
        if isinstance(v, dict) and k in parent:
            recursive_dict_update(v, parent[k])
        else:
            parent[k] = v
    return parent


def maybe_overwrite_args_with_default(method_cfg: dict, specific_cfg: dict):
    # This is to overwrite the specific config with the default config if the specific config has a "default" key, which indicates which default config to use.
    # The default config is defined in the "defaults" key of the method config, and is a "parent" config to the final config.
    if "defaults" not in method_cfg:
        return
    default_cfg_name = specific_cfg.get("default", None)
    if default_cfg_name is None:
        return
    specific_cfg = recursive_dict_update(
        specific_cfg, method_cfg["defaults"][default_cfg_name]
    )


def maybe_overwrite_args_with_blueprint(specfic_cfg: dict):
    # This is to overwrite the specific config with the blueprint config if the specific config has a "blueprint" key, which indicates which blueprint config to use.
    blueprint_name = specfic_cfg.get("blueprint", None)
    if blueprint_name is None:
        return
    blueprint_cfg = blueprints[blueprint_name]
    specfic_cfg = recursive_dict_update(specfic_cfg, blueprint_cfg)


def update_everything(
    original_cli_args: tuple,
    baseline_args: tuple,
    method: str,
    cfg: dict,
):
    method_cfg = configs[method]
    maybe_overwrite_args_with_default(
        method_cfg,
        cfg,
    )
    maybe_overwrite_args_with_blueprint(cfg)
    overwrite_all_args(
        original_cli_args,
        baseline_args,
        cfg,
    )
    return original_cli_args


def parse_args() -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    cli_parser = transformers.HfArgumentParser(
        (GeneralArgs, ModelArguments, DataArguments, TrainingArguments)
    )
    cli_general, cli_model, cli_data, cli_training = (
        cli_parser.parse_args_into_dataclasses()
    )
    assert cli_general.config is not None, "Please provide a config name!"
    assert cli_general.method in configs, f"Config {cli_general.method} not found!"

    baseline_model, baseline_data, baseline_training = baseline_instances()

    cfg: dict = configs[cli_general.method][cli_general.config]
    cli_model, cli_data, cli_training = update_everything(
        (cli_model, cli_data, cli_training),
        (baseline_model, baseline_data, baseline_training),
        cli_general.method,
        cfg,
    )
    if cli_training.local_rank == 0:
        print("model_args:", cli_model)
        print("data_args:", cli_data)
        print("training_args:", cli_training)
        save_args(cli_model, cli_data, cli_training)
    return cli_model, cli_data, cli_training


def args_from_config(
    method: str,
    config_name: str,
) -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    default_config = json.loads(open("config/default_args.json", "r").read())
    model_args = ModelArguments(**default_config["model_args"])
    data_args = DataArguments(**default_config["data_args"])
    training_args = TrainingArguments(**default_config["training_args"])

    cfg: dict = configs[method][config_name]
    model_args, data_args, training_args = update_everything(
        (model_args, data_args, training_args),
        (model_args, data_args, training_args),
        method,
        cfg,
    )

    return model_args, data_args, training_args


def save_args(model_args, data_args, training_args):
    saved_config = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "saved_config.json"), "w") as f:
        f.write(json.dumps(saved_config, indent=4))
