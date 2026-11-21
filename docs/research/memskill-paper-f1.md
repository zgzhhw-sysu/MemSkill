# MemSkill 论文 F1 对照协议

正式测试分别报告 LoCoMo 的 314 题和 LongMemEval-S 的 101 题。每道题使用完整对话历史建库后的答案，先算原生逐题 F1，再对同一数据集的题等权平均并乘以 100。论文的 L-J 为独立的模型裁判指标，表 1 的 Avg. L-J 也按裁判分计算。ALFWorld 报告成功率和步数。协议固定 arXiv v3 和作者代码 `f65b797`。[论文表 1](https://arxiv.org/pdf/2602.02474v3#page=7) · [作者的评测入口](https://github.com/ViktorAxelsen/MemSkill/blob/f65b797338033f4f9c34822f1810f9c2747e6022/main.py#L524-L650) · [机器可读配置](../../experiments/controller/protocols/memskill_paper_f1.json)

## 测试题和分母

| 数据集 | 作者的 test 划分 | 过滤规则 | 正式计分 |
| --- | --- | --- | ---: |
| LoCoMo | 10 段对话按原序分为训练 6 段、验证 2 段、测试 2 段。测试取零起算索引 8、9 | 两段共 400 题，去掉类别 5 的 86 道对抗题 | `conv-49` 156 题，`conv-50` 158 题，共 **314** 题 |
| LongMemEval-S | 对原始 `longmemeval_s_cleaned.json` 的 500 条记录应用作者公布的索引，test 命中 105 条 | 去掉 `question_id` 以 `_abs` 结尾的 4 条，每条历史附 1 题 | **101** 题，101 段完整历史 |

论文附录 B.1 将 LongMemEval-S 写为分层抽取约 100 条。这里的精确题数来自公开代码和数据。LoCoMo 两段对话共 55 个 session，四类有效题依次为多跳 69、时间 65、开放域 20、单跳 160。LongMemEval-S 的 101 条历史共 4,830 个 session，含时间推理 28、跨 session 26、知识更新 15、用户信息 14、助手信息 12、偏好 6。额外弃答诊断可另报。训练与选型使用训练和验证集，LongMemEval-S 直接迁移 LoCoMo 的技能和控制器，允许在验证集确定数据集配置。[论文附录 B.1](https://arxiv.org/pdf/2602.02474v3#page=17) · [官方划分](https://github.com/ViktorAxelsen/MemSkill/blob/f65b797338033f4f9c34822f1810f9c2747e6022/data/longmemeval_s_splits.json) · [官方切分实现](https://github.com/ViktorAxelsen/MemSkill/blob/f65b797338033f4f9c34822f1810f9c2747e6022/main.py#L64-L95)

评分沿用作者的文本规范化、Porter 词干和词频重叠 F1。LoCoMo 类别 1 将预测和参考按逗号拆分，每个参考子答案取最佳预测匹配，再求平均。类别 3 取参考答案第一个分号前的内容。LongMemEval-S 使用普通 token F1。空预测保留在分母中。LoCoMo 两段对话以题数加权，两段对话的 F1 可另行列出。[原生评分函数](https://github.com/ViktorAxelsen/MemSkill/blob/f65b797338033f4f9c34822f1810f9c2747e6022/eval_utils.py) · [LoCoMo 类别处理](https://github.com/ViktorAxelsen/MemSkill/blob/f65b797338033f4f9c34822f1810f9c2747e6022/src/eval/locomo.py)

## 论文中的 MemSkill 参照值

| 原文位置 | 写入与问答模型 | LoCoMo F1 | LongMemEval-S F1 |
| --- | --- | ---: | ---: |
| 表 1 | Llama-3.3-70B-Instruct | 44.21 | 31.12 |
| 表 1 | Qwen3-Next-80B-A3B-Instruct | 42.08 | 25.29 |
| 附表 7 | Llama-3.1-8B-Instruct | 20.59 | 18.13 |

当前项目使用 Meta-Llama-3-8B-Instruct。报告改动带来的增益时，同一模型、checkpoint 起点、推理配置和测试题上分别运行原始 MemSkill 和候选。可在旁边列论文值，并标明模型版本、权重精度和服务差异。开发集前缀、固定技能数为 3 的整 session 实验各自保留配置标签。[论文表 1](https://arxiv.org/pdf/2602.02474v3#page=7) · [论文附表 7](https://arxiv.org/pdf/2602.02474v3#page=16)

论文第 4.1 节给出的对话测试默认值是 512 token 分块、每块选 7 个技能、Qwen3-Embedding-0.6B 最多检索 20 条记忆。本项目论文对照入口对两个数据集统一采用这些默认值，并按发布脚本固定相邻块重叠 64 token。上游 LongMemEval 脚本写的是 5 个技能，LoCoMo 脚本注释建议 Qwen 用 256 token。公开材料没有给出能逐行核验表 1 每次运行的配置。如需复现脚本变体，单列其结果和参数。[正文第 4.1 节](https://arxiv.org/pdf/2602.02474v3#page=6) · [作者 LoCoMo 脚本](https://github.com/ViktorAxelsen/MemSkill/blob/f65b797338033f4f9c34822f1810f9c2747e6022/eval_locomo.sh) · [作者 LongMemEval 脚本](https://github.com/ViktorAxelsen/MemSkill/blob/f65b797338033f4f9c34822f1810f9c2747e6022/eval_longmemeval.sh)

## 核验与运行

在仓库根目录设置 `LOCOMO_DATA` 和 `LONGMEMEVAL_DATA` 指向完整原始数据文件。下列命令只核验原始文件、划分与有序题目 ID 的 SHA-256。

```bash
python -m experiments.controller.paper_f1 locomo "$LOCOMO_DATA"
python -m experiments.controller.paper_f1 longmemeval "$LONGMEMEVAL_DATA"
```

已有逐题预测的策略可在项目 Python 环境直接复算。输入沿用原生 `--eval-f1-only` 的 JSON，每条 `answers` 记录包含 `sample_id`、`qa_index`、`prediction`。LoCoMo 使用原始对话内的题号，LongMemEval 使用原始 `question_id` 和题号 0。可以同时传入同一次测试中互不重叠的文件。工具从原始数据读取参考答案，忽略报告自带的分数。漏题、重题或额外题会终止汇总。输出的 `f1_percent` 与论文表格同尺度。

```bash
python -m experiments.controller.paper_f1 locomo "$LOCOMO_DATA" --answers "$LOCOMO_RESULT"
python -m experiments.controller.paper_f1 longmemeval "$LONGMEMEVAL_DATA" --answers "$LONGMEMEVAL_RESULT"
```

启动原始 MemSkill 对照时，在现有 GPU 租约内使用下列命令。服务模型和 checkpoint 由实验指定，输出目录取新路径以隔离记忆缓存。入口先核验题目集合，调用原生 `main.py` 建库和逐题问答，最后从预测重新计算 F1。将同一冻结 checkpoint 迁移到 LongMemEval-S。运行记录须注明服务实际加载的模型和精度。

```bash
python -m experiments.controller.paper_f1 locomo "$LOCOMO_DATA" \
  --run-baseline "$RUN_DIR/locomo" --load-checkpoint "$CHECKPOINT" \
  --model "$MODEL" --api --api-base "$API_BASE" --api-key "$API_KEY" --device cuda

python -m experiments.controller.paper_f1 longmemeval "$LONGMEMEVAL_DATA" \
  --run-baseline "$RUN_DIR/longmemeval" --load-checkpoint "$CHECKPOINT" \
  --model "$MODEL" --api --api-base "$API_BASE" --api-key "$API_KEY" --device cuda
```
