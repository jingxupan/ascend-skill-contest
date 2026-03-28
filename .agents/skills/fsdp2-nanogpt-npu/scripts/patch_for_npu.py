#!/usr/bin/env python3
"""
Patch FSDP2 nanoGPT example for Ascend NPU.

Usage:
    cd pytorch-examples/distributed/FSDP2
    python3 /path/to/patch_for_npu.py

Patches applied:
  1. example.py — inject torch_npu + transfer_to_npu
  2. model.py   — replace F.scaled_dot_product_attention with npu_fusion_attention
  3. utils.py   — add import torch_npu
  4. run_example.sh — change default to 2 NPUs
"""

import os
import sys
import re


def patch_file(filepath, patches, description):
    """Apply a list of (old, new) replacements to a file."""
    if not os.path.exists(filepath):
        print(f"[SKIP] {filepath} not found")
        return False

    with open(filepath, "r") as f:
        content = f.read()

    original = content
    for old, new in patches:
        if old not in content:
            print(f"[WARN] Pattern not found in {filepath}: {old[:60]}...")
            continue
        content = content.replace(old, new, 1)

    if content == original:
        print(f"[SKIP] {filepath} — already patched or no changes needed")
        return False

    with open(filepath, "w") as f:
        f.write(content)
    print(f"[OK]   {filepath} — {description}")
    return True


def main():
    base_dir = os.getcwd()

    for required in ["example.py", "model.py", "utils.py"]:
        if not os.path.exists(os.path.join(base_dir, required)):
            print(f"ERROR: {required} not found in {base_dir}")
            print("Please run this script from the FSDP2 directory:")
            print("  cd pytorch-examples/distributed/FSDP2")
            sys.exit(1)

    # 1. Patch example.py
    patch_file("example.py", [
        (
            "import argparse\nimport os\n\nimport torch\n",
            "import argparse\nimport os\n\nimport torch\nimport torch_npu\nfrom torch_npu.contrib import transfer_to_npu\n",
        ),
    ], "injected torch_npu + transfer_to_npu")

    # 2. Patch model.py
    patch_file("model.py", [
        (
            "from dataclasses import dataclass\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
            "from dataclasses import dataclass\nimport math\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torch_npu\n",
        ),
        (
            "        output = F.scaled_dot_product_attention(\n"
            "            queries,\n"
            "            keys,\n"
            "            values,\n"
            "            None,\n"
            "            self.dropout_p if self.training else 0,\n"
            "        )",
            "        scale = 1.0 / math.sqrt(self.head_dim)\n"
            "        drop_rate = self.dropout_p if self.training else 0.0\n"
            "        output = torch_npu.npu_fusion_attention(\n"
            "            queries, keys, values,\n"
            "            head_num=self.n_heads,\n"
            "            input_layout=\"BNSD\",\n"
            "            scale=scale,\n"
            "            keep_prob=1.0 - drop_rate,\n"
            "        )[0]",
        ),
    ], "replaced scaled_dot_product_attention with npu_fusion_attention")

    # 3. Patch utils.py
    patch_file("utils.py", [
        (
            "import torch\n",
            "import torch\nimport torch_npu\n",
        ),
    ], "added import torch_npu")

    # 4. Patch run_example.sh
    patch_file("run_example.sh", [
        (
            'echo "Launching ${1:-example.py} with ${2:-4} gpus"\n'
            'torchrun --nnodes=1 --nproc_per_node=${2:-4} ${1:-example.py}',
            'echo "Launching ${1:-example.py} with ${2:-2} NPUs"\n'
            'torchrun --nnodes=1 --nproc_per_node=${2:-2} ${1:-example.py}',
        ),
    ], "changed default to 2 NPUs")

    print("\n[DONE] All patches applied. Ready for NPU training:")
    print("  torchrun --nnodes=1 --nproc_per_node=2 example.py")


if __name__ == "__main__":
    main()
