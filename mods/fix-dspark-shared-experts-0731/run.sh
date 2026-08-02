#!/bin/bash
# Mod: map the DSpark draft's shared-expert gate_up_proj weights (0731 checkpoint).
#
# vLLM's DSpark draft loader renames only ".shared_experts.w2" -> down_proj, so
# the w1/w3 shared-expert projections match no parameter and fall through to
# logger.debug("Skipping unknown DSpark weight") -- silent at INFO. On
# DeepSeek-V4-Flash-0731 that drops 12 tensors
# (model.layers.{43,44,45}.ffn.shared_experts.gate_up_proj.{weight,weight_scale_inv})
# and every draft stage runs with its always-on shared expert uninitialised:
# output quality is unchanged, but draft acceptance collapses (~60% -> ~26%)
# and throughput roughly halves.
#
# Fix (2 rows in _STACKED_PARAM_NAME_MAPPING) from
# tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
# patches/0004-dspark-shared-expert-gate-up-proj.patch. The target model's own
# loader already carries these rows; the draft loader lost them when the
# mapping was narrowed to avoid the markov_w1 collision.
#
# The file's location moved across vLLM versions, so we locate the module by
# its _STACKED_PARAM_NAME_MAPPING table instead of hardcoding a path.
set -e

VLLM_DIR=/usr/local/lib/python3.12/dist-packages

python3 - "$VLLM_DIR" <<'EOF'
import pathlib
import sys

vllm_dir = pathlib.Path(sys.argv[1]) / "vllm"
prefix = "--- [fix-dspark-shared-experts-0731]"

candidates = [
    p for p in vllm_dir.rglob("*dspark*.py")
    if "_STACKED_PARAM_NAME_MAPPING" in p.read_text()
]
if not candidates:
    print(f"{prefix} no DSpark loader with _STACKED_PARAM_NAME_MAPPING found; "
          "this image has no DSpark support, skipping.")
    sys.exit(0)
if len(candidates) > 1:
    print(f"{prefix} multiple DSpark loader candidates: {candidates}")
    sys.exit(1)

path = candidates[0]
text = path.read_text()

if "shared_experts.w1" in text:
    print(f"{prefix} shared-expert rows already present in {path}, skipping.")
    sys.exit(0)

anchor = '    ("attn.fused_wqa_wkv", ".attn.wkv", 1),\n'
if anchor not in text:
    print(f"{prefix} anchor row not found in {path}; loader has diverged, "
          "refusing to patch blind.")
    sys.exit(1)

rows = (
    '    ("shared_experts.gate_up_proj", ".shared_experts.w1", 0),\n'
    '    ("shared_experts.gate_up_proj", ".shared_experts.w3", 1),\n'
)
path.write_text(text.replace(anchor, anchor + rows, 1))
print(f"{prefix} added shared-expert gate_up_proj rows to {path}")
EOF

echo "=== OK"
