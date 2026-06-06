"""Patch slime's Qwen3.5 HF conversion for bridge checkpoint names.

The Qwen3.6-35B-A3B bridge checkpoint stores Megatron keys below a
``language_model.`` namespace, while slime's Qwen3.5 converter expects the
namespace-less Megatron names. It can also carry unused vision placeholders from
the VLM bridge. Normalize these names before conversion.

Executed at image-build time via ``python3 <this file>``.
"""

from pathlib import Path

TARGET = Path("/root/slime/slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py")
TOOL_TARGET = Path("/root/slime/tools/convert_torch_dist_to_hf.py")
MARKER = "PATCHED_QWEN35_BRIDGE_NAMES"
TOOL_MARKER = "PATCHED_QWEN35_SKIP_VISION_NAMES"

if not TARGET.exists():
    print(f"WARNING: {TARGET} not found, skipping Qwen3.5 conversion-name patch")
else:
    src = TARGET.read_text()
    needle = (
        '    """Convert Qwen3.5 model parameters from Megatron to HuggingFace format.'
    )
    if MARKER in src:
        print("qwen3_5.py already patched for bridge checkpoint names")
    elif needle in src:
        insert = (
            f"    # {MARKER}\n"
            '    if name.startswith("module.module.language_model."):\n'
            '        name = "module.module." + name[len("module.module.language_model.") :]\n'
            '    if name.startswith("module.module.vision_model."):\n'
            "        return []\n"
            "\n"
        )
        src = src.replace(needle, insert + needle, 1)
        linear_attn_needle = 'raise ValueError(f"Unknown parameter name: {name}")'
        linear_attn_insert = (
            f"        # {MARKER}: bridge linear-attention names\n"
            "        qwen35_direct = {\n"
            "            'self_attention.A_log': 'linear_attn.A_log',\n"
            "            'self_attention.dt_bias': 'linear_attn.dt_bias',\n"
            "            'self_attention.in_proj.layer_norm_weight': 'input_layernorm.weight',\n"
            "            'self_attention.in_proj.weight.alpha': 'linear_attn.in_proj_a.weight',\n"
            "            'self_attention.in_proj.weight.beta': 'linear_attn.in_proj_b.weight',\n"
            "            'self_attention.in_proj.weight.z': 'linear_attn.in_proj_z.weight',\n"
            "            'self_attention.out_norm.weight': 'linear_attn.norm.weight',\n"
            "            'self_attention.out_proj.weight': 'linear_attn.out_proj.weight',\n"
            "        }\n"
            "        if rest in qwen35_direct:\n"
            "            return [(f'{prefix}.{qwen35_direct[rest]}', param)]\n"
            "        qwen35_cat = {\n"
            "            'self_attention.in_proj.weight.query': ('linear_attn.in_proj_qkv.weight', 'query'),\n"
            "            'self_attention.in_proj.weight.key': ('linear_attn.in_proj_qkv.weight', 'key'),\n"
            "            'self_attention.in_proj.weight.value': ('linear_attn.in_proj_qkv.weight', 'value'),\n"
            "            'self_attention.conv1d.weight.query': ('linear_attn.conv1d.weight', 'query'),\n"
            "            'self_attention.conv1d.weight.key': ('linear_attn.conv1d.weight', 'key'),\n"
            "            'self_attention.conv1d.weight.value': ('linear_attn.conv1d.weight', 'value'),\n"
            "        }\n"
            "        if rest in qwen35_cat:\n"
            "            target, part = qwen35_cat[rest]\n"
            "            return [(f'__qwen35_cat__.{prefix}.{target}.{part}', param)]\n"
        )
        raise_pos = src.rfind(linear_attn_needle)
        if raise_pos >= 0:
            line_start = src.rfind("\n", 0, raise_pos) + 1
            src = src[:line_start] + linear_attn_insert + src[line_start:]
        else:
            print("WARNING: Could not find Qwen3.5 final raise for linear-attn patch")
        TARGET.write_text(src)
        print(f"Patched {TARGET}: normalized Qwen3.5 bridge names")
    else:
        print("WARNING: Could not find Qwen3.5 converter insertion point")

if not TOOL_TARGET.exists():
    print(f"WARNING: {TOOL_TARGET} not found, skipping Qwen3.5 tool-name patch")
else:
    src = TOOL_TARGET.read_text()
    needle = "def get_named_params(args, state_dict):\n    for name, param in state_dict.items():\n"
    if TOOL_MARKER in src:
        print("convert_torch_dist_to_hf.py already patched to skip vision names")
    elif needle in src:
        replacement = (
            needle
            + f'        if name.startswith("vision_model."):  # {TOOL_MARKER}\n'
            + "            continue\n"
        )
        src = src.replace(needle, replacement, 1)
        init_needle = (
            "    current_size = 0\n    total_size = 0\n    modeltensors = [{}]\n"
        )
        init_insert = (
            "    current_size = 0\n"
            "    total_size = 0\n"
            "    modeltensors = [{}]\n"
            "    qwen35_cat_tensors = {}\n\n"
            "    def add_converted_tensor(converted_name, converted_param):\n"
            "        nonlocal current_size, total_size\n"
            "        if converted_name.startswith('__qwen35_cat__.'):\n"
            "            target_and_part = converted_name[len('__qwen35_cat__.'):]\n"
            "            target, part = target_and_part.rsplit('.', 1)\n"
            "            qwen35_cat_tensors.setdefault(target, {})[part] = converted_param\n"
            "            return\n"
            "        tensor_size = converted_param.numel() * converted_param.element_size()\n"
            "        if tensor_size + current_size > chunk_size:\n"
            "            modeltensors.append({})\n"
            "            current_size = 0\n"
            "        modeltensors[-1][converted_name] = converted_param\n"
            "        current_size += tensor_size\n"
            "        total_size += tensor_size\n"
        )
        if init_needle in src:
            src = src.replace(init_needle, init_insert, 1)
        else:
            print(
                "WARNING: Could not find save_tensors init block for Qwen3.5 cat patch"
            )
        loop_needle = (
            "            tensor_size = converted_param.numel() * converted_param.element_size()\n"
            "            if tensor_size + current_size > chunk_size:\n"
            "                modeltensors.append({})\n"
            "                current_size = 0\n"
            "            modeltensors[-1][converted_name] = converted_param\n"
            "            current_size += tensor_size\n"
            "            total_size += tensor_size\n"
        )
        if loop_needle in src:
            src = src.replace(
                loop_needle,
                "            add_converted_tensor(converted_name, converted_param)\n",
                1,
            )
        else:
            print("WARNING: Could not find save_tensors add loop for Qwen3.5 cat patch")
        metadata_needle = '    metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}\n'
        metadata_insert = (
            "    for converted_name, parts in qwen35_cat_tensors.items():\n"
            "        missing = {'query', 'key', 'value'} - set(parts)\n"
            "        if missing:\n"
            "            raise ValueError(f'Missing Qwen3.5 linear-attn pieces for {converted_name}: {sorted(missing)}')\n"
            "        add_converted_tensor(\n"
            "            converted_name,\n"
            "            torch.cat([parts['query'], parts['key'], parts['value']], dim=0),\n"
            "        )\n"
        )
        if metadata_needle in src:
            src = src.replace(metadata_needle, metadata_insert + metadata_needle, 1)
        else:
            print("WARNING: Could not find metadata block for Qwen3.5 cat patch")
        TOOL_TARGET.write_text(src)
        print(f"Patched {TOOL_TARGET}: skipped unused Qwen3.5 vision params")
    else:
        print("WARNING: Could not find convert_torch_dist_to_hf get_named_params block")
