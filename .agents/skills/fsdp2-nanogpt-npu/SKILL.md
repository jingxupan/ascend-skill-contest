---
name: fsdp2-nanogpt-npu-migrate
description: >
  将 PyTorch Examples 仓库中 FSDP2 nanoGPT 分布式训练示例从 GPU (CUDA) 迁移到
  华为昇腾 NPU (Ascend910 系列)。涵盖环境搭建、torch_npu 注入、
  npu_fusion_attention 融合算子替换、多卡 torchrun 分布式训练启动，
  以及 checkpoint 保存与加载验证全流程。
  当用户提到 FSDP2 nanoGPT NPU 迁移、FSDP2 昇腾训练、
  nanoGPT 分布式 NPU 训练 时触发。
metadata:
  short-description: FSDP2 nanoGPT GPU→NPU 分布式训练迁移
  category: NPU-Migration
  tags: [ascend, npu, fsdp2, nanogpt, distributed-training, torch_npu]
---

# FSDP2 nanoGPT 昇腾 NPU 迁移 Skill

将 [pytorch/examples FSDP2](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)
nanoGPT 训练任务从 GPU 迁移到昇腾 NPU，完成多卡分布式训练及 checkpoint 验证。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（≥ 2 卡） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | ≥ 8.0（推荐 8.2+） |
| Python | 3.8 – 3.10（推荐 3.10） |
| Conda | Miniconda 或 Anaconda（用于创建隔离环境） |
| PyTorch | ≥ 2.7 |
| torch_npu | 与 PyTorch 版本一致（≥ 2.7） |

## 迁移流程总览

```
1. 创建 Conda 隔离环境
→ 2. 环境准备与验证
→ 3. 克隆代码
→ 4. 代码适配（4 个文件）
→ 5. 第一次训练（保存 checkpoint）
→ 6. 第二次训练（加载 checkpoint 验证）
```

---

## 1. 创建 Conda 隔离环境

**必须新建独立 Conda 环境**，避免污染系统默认 `base` 环境。

```bash
conda create -n fsdp2_npu python=3.10 -y
conda activate fsdp2_npu
```

> 后续所有安装和训练命令均在 `fsdp2_npu` 环境中执行。
> 若终端重新连接，需先 `conda activate fsdp2_npu` 再操作。

---

## 2. 环境准备与验证

### 2.1 CANN 环境初始化

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
```

确认 NPU 设备可见且状态正常。

### 2.2 安装 PyTorch + torch_npu

```bash
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
pip install 'torch>=2.7' 'torch_npu>=2.7'
pip install numpy==1.26.4
pip install pyyaml scipy decorator attrs psutil
```

> - 版本必须 ≥ 2.7，torch 与 torch_npu 版本一致（pip 会自动匹配最新兼容版本）
> - **numpy 必须用 1.26.4**：numpy 2.x 移除了 `np.float_`，与 CANN 运行时不兼容
> - `pyyaml scipy decorator` 等为 torch_npu / CANN 运行时隐式依赖，缺失会导致初始化失败

### 2.3 验证 NPU 可用

```bash
python3 -c "
import torch
import torch_npu
print('torch version:', torch.__version__)
print('torch_npu version:', torch_npu.__version__)
a = torch.randn(3, 4).npu()
print('NPU tensor:', a.device)
print('NPU device count:', torch.npu.device_count())
"
```

输出应显示 `device='npu:0'` 且设备数 ≥ 2。

---

## 3. 克隆代码

```bash
git clone https://github.com/pytorch/examples.git pytorch-examples
cd pytorch-examples/distributed/FSDP2
pip install -r requirements.txt
```

后续所有操作在 `FSDP2/` 目录下进行。

---

## 4. 代码适配

需要修改 4 个文件：`example.py`、`model.py`、`utils.py`、`run_example.sh`。

也可直接运行 `scripts/patch_for_npu.py` 一键完成适配（见本 Skill `scripts/` 目录），
或按以下步骤手动修改。

### 4.1 example.py — 注入 torch_npu

在文件**顶部** import 区域添加：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

`transfer_to_npu` 自动将 `torch.cuda.*` API 映射到 NPU，
并将分布式 backend 从 `nccl` 映射到 `hccl`。

此文件无需其他修改 — 原代码已使用 `torch.accelerator` 抽象 API，
可自动检测 NPU 设备。

### 4.2 model.py — 接入 npu_fusion_attention

这是本次迁移的**核心改动**。将 `F.scaled_dot_product_attention` 替换为
`torch_npu.npu_fusion_attention` 融合算子。

在文件顶部添加：

```python
import math
import torch_npu
```

将 `Attention.forward` 方法中的注意力计算替换为：

```python
def forward(self, x):
    bsz, seq_len, _ = x.size()
    queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
    queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
    keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
    values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

    queries = queries.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    scale = 1.0 / math.sqrt(self.head_dim)
    drop_rate = self.dropout_p if self.training else 0.0
    output = torch_npu.npu_fusion_attention(
        queries, keys, values,
        head_num=self.n_heads,
        input_layout="BNSD",
        scale=scale,
        keep_prob=1.0 - drop_rate,
    )[0]

    output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
    return self.resid_dropout(self.wo(output))
```

**关键参数说明：**

| 参数 | 值 | 说明 |
|------|------|------|
| `input_layout` | `"BNSD"` | 对应 (Batch, NumHeads, SeqLen, HeadDim) |
| `scale` | `1/sqrt(head_dim)` | 注意力缩放因子 |
| `keep_prob` | `1 - dropout_p` | 保留概率（非丢弃概率） |
| `[0]` | 取第一个返回值 | 函数返回元组，第一个元素是注意力输出 |

### 4.3 utils.py — 添加 torch_npu import

在文件顶部添加：

```python
import torch_npu
```

确保 `torch_npu` 在使用 NPU tensor 操作前已加载。

### 4.4 run_example.sh — 调整默认卡数

将默认 GPU 数量改为 2（最小多卡要求）：

```bash
echo "Launching ${1:-example.py} with ${2:-2} NPUs"
torchrun --nnodes=1 --nproc_per_node=${2:-2} ${1:-example.py}
```

---

## 5. 第一次训练 — 保存 checkpoint

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
conda activate fsdp2_npu
export ASCEND_RT_VISIBLE_DEVICES=0,1

torchrun --nnodes=1 --nproc_per_node=2 example.py
```

**预期输出：**
- 打印模型结构和 rank 信息
- 完成 10 步训练迭代
- 在 `checkpoints/` 目录下生成 checkpoint 文件（包含模型和优化器状态）

验证 checkpoint 已保存：

```bash
find checkpoints/ -type f -name "*.pt" | head -10
```

---

## 6. 第二次训练 — 加载 checkpoint

直接再次运行相同命令：

```bash
torchrun --nnodes=1 --nproc_per_node=2 example.py
```

**预期行为：**
- 程序检测到 `checkpoints/` 中存在已保存的状态
- 加载模型权重和优化器状态（而非重新初始化）
- 继续训练 10 步并保存新 checkpoint

验证 checkpoint 目录中有**两个**时间戳子目录：

```bash
ls checkpoints/dtensor_api/
```

---

## 可选功能

### Mixed Precision 训练

```bash
torchrun --nnodes=1 --nproc_per_node=2 example.py --mixed-precision
```

使用 bf16 参数精度 + fp32 梯度归约，可减少显存占用。

### Explicit Prefetching

```bash
torchrun --nnodes=1 --nproc_per_node=2 example.py --explicit-prefetch
```

### DCP API

```bash
torchrun --nnodes=1 --nproc_per_node=2 example.py --dcp-api
```

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `No module named 'torch_npu'` | 未安装 torch_npu | `pip install 'torch_npu>=2.7'` |
| `np.float_` was removed | numpy 2.x 与 CANN 不兼容 | `pip install numpy==1.26.4` |
| `No module named 'yaml'` | torch_npu 隐式依赖缺失 | `pip install pyyaml` |
| `No module named 'scipy'` | CANN 初始化依赖缺失 | `pip install scipy decorator attrs psutil` |
| `SetPrecisionMode error 500001` | 上述依赖缺失导致 CANN 初始化失败 | 安装全部运行时依赖后重试 |
| `HCCL ... timeout` | 多卡通信超时 | 检查 `ASCEND_RT_VISIBLE_DEVICES`，确认卡间连接 |
| `npu_fusion_attention` 报错 | 输入 dtype 不匹配 | 确保 Q/K/V 为 fp16 或 bf16；fp32 下内部自动转换 |
| checkpoint 目录为空 | 仅 rank 0 保存 | 正常行为，检查 rank 0 日志 |
| `transfer_to_npu` 未生效 | import 顺序不对 | 必须在所有其他 import 之前导入 |
| OOM 内存不足 | 模型或 batch 过大 | 减小 `batch_size` 或 `n_layers` |

---

## 迁移检查清单

- [ ] 已创建并激活独立 Conda 环境 `fsdp2_npu`
- [ ] CANN 环境已 source，NPU 设备可见
- [ ] torch / torch_npu ≥ 2.7 已安装
- [ ] `example.py` 顶部注入 `torch_npu` + `transfer_to_npu`
- [ ] `model.py` 中 `F.scaled_dot_product_attention` 替换为 `npu_fusion_attention`
- [ ] `utils.py` 添加 `import torch_npu`
- [ ] 使用 `torchrun --nproc_per_node=2` 启动多卡训练
- [ ] 第一次训练产生 checkpoint 文件
- [ ] 第二次训练成功加载 checkpoint 并继续训练
