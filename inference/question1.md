# 题目1: 推理框架(vllm/sglang/xllm/mindie)部署

## 题目概览

| 项目 | 说明 |
|------|------|
| 难度 | ⭐ 初等 |
| 预估时长 | 60 分钟 |

---

## 使用场景

- 对于框架部署不熟悉的新人，用于自动化部署或者指导部署
- 在Agentic Coding场景，使用AI端到端完成搭建开发环境

## 任务描述

使用vllm/sglang/xllm/mindie等推理框架中的**其中一个**部署[Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)模型，建议使用镜像部署而不是源码构建部署。

具体要求：

| 项目 | 说明 |
|------|------|
| Prompt | 使用vllm/sglang/xllm/mindie部署Qwen3-0.6B模型 |
| 执行时间 | 30 分钟以内 |

## 格式要求

参赛者需按照以下目录结构提交Skill：
```
skill-name/
├── SKILL.md        # 必须
├── reference/      # 可选
└── scripts/        # 可选
```

## 评分标准

参考 [Agent Skill 创作最佳实践](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)：

| 维度 | 权重 | 说明 |
|------|------|------|
| 功能完整性 | 60% | 是否能成功完成 Qwen3-0.6B 部署，覆盖关键步骤 |
| 易用性 | 15% | 能够以尽可能简短的prompt完成部署，非必要的参数如端口、容器名可自动配置默认值，镜像自动拉取最新版本等 |
| 指令与结构 | 15% | SKILL.md 是否简洁（建议 500 行以内）；指令步骤是否清晰可执行；是否合理使用渐进式披露 |
| 错误处理 | 10% | 错误处理是否清晰，能够以交互的方式提醒用户缺失的信息，如模型名称/路径 |


## PR模板

### 题目1: 推理框架(vllm/sglang/xllm/mindie)部署

#### 推理框架
vllm/sglang/xllm/mindie选一个

#### Prompt

#### 测试结果（截图）
