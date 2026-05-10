import json
from pathlib import Path

paths = [
    Path("target_lmm/saved_config.json"),
    Path("target_lmm/config.json"),
]

def patch_obj(obj):
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            lk = str(k).lower()

            # Убираем flash attention / deepspeed из любых вложенных конфигов
            if "deepspeed" in lk:
                print("Removing key:", k)
                obj.pop(k, None)
                continue

            if k in [
                "attn_implementation",
                "_attn_implementation",
                "attn_implementation_internal",
            ]:
                print("Changing", k, "from", obj[k], "to eager")
                obj[k] = "eager"
                continue

            if isinstance(obj[k], str) and "flash_attention_2" in obj[k]:
                print("Changing", k, "from", obj[k], "to eager")
                obj[k] = "eager"
                continue

            patch_obj(obj[k])

    elif isinstance(obj, list):
        for x in obj:
            patch_obj(x)

for path in paths:
    if not path.exists():
        print("Missing:", path)
        continue

    backup = path.with_suffix(path.suffix + ".bak2")
    if not backup.exists():
        backup.write_text(path.read_text())

    data = json.loads(path.read_text())
    patch_obj(data)

    # На всякий случай добавляем прямо в корень
    data["attn_implementation"] = "eager"
    data["_attn_implementation"] = "eager"

    path.write_text(json.dumps(data, indent=2))
    print("Patched:", path)
