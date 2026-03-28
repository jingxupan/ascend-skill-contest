# npu_fusion_attention API 参考

## 函数签名

```python
torch_npu.npu_fusion_attention(
    query, key, value,
    head_num,
    input_layout,
    *,
    pse=None,
    padding_mask=None,
    atten_mask=None,
    scale=1.0,
    keep_prob=1.0,
    pre_tockens=2147483647,
    next_tockens=2147483647,
    inner_precise=0,
    prefix=None,
    sparse_mode=0,
    gen_mask_parallel=True,
    sync=False,
)
```

## 返回值

返回元组 `(attention_output, softmax_max, softmax_sum, softmax_out, seed, offset, numels)`，
通常只需取 `[0]` 即注意力输出 tensor。

## 常用参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `query/key/value` | Tensor | Q/K/V 张量，dtype 为 fp16 或 bf16（fp32 内部自动转换） |
| `head_num` | int | 注意力头数 |
| `input_layout` | str | 张量布局：`"BSH"`, `"BNSD"`, `"SBH"`, `"BSND"` |
| `scale` | float | 缩放因子，通常为 `1/sqrt(head_dim)` |
| `keep_prob` | float | Dropout 保留概率（1.0 = 不丢弃） |
| `atten_mask` | Tensor/None | 注意力掩码（因果掩码等） |

## 布局说明

| Layout | 维度含义 | 典型场景 |
|--------|----------|----------|
| `"BNSD"` | (Batch, NumHeads, SeqLen, HeadDim) | 标准多头注意力 |
| `"BSH"` | (Batch, SeqLen, HiddenSize) | 合并头维度 |
| `"SBH"` | (SeqLen, Batch, HiddenSize) | sequence-first 布局 |
| `"BSND"` | (Batch, SeqLen, NumHeads, HeadDim) | 未转置的多头布局 |

## FSDP2 nanoGPT 中的用法

原始代码中 Q/K/V shape 为 `(bsz, n_heads, seq_len, head_dim)`，
对应 `"BNSD"` 布局，替换方式：

```python
# 原始
output = F.scaled_dot_product_attention(queries, keys, values, None, dropout_p)

# NPU 替换
output = torch_npu.npu_fusion_attention(
    queries, keys, values,
    head_num=self.n_heads,
    input_layout="BNSD",
    scale=1.0 / math.sqrt(self.head_dim),
    keep_prob=1.0 - dropout_p,
)[0]
```

## 注意事项

- `keep_prob` 是**保留**概率，与 PyTorch dropout 的**丢弃**概率互补
- 推理时设 `keep_prob=1.0`（即 dropout_p=0）
- 若输入为 fp32，算子内部会转换为 fp16 计算再转回，可能有精度差异
- 因果掩码（causal）可通过 `atten_mask` 参数或 `sparse_mode` 控制
