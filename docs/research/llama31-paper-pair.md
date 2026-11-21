# Llama-3.1-8B 的原始技能与日期写入对照

本轮使用 MemSkill 附表 7 的 Meta-Llama-3.1-8B-Instruct。作者报告 LoCoMo 20.59、LongMemEval-S 18.13。历史 Llama-3-8B 实验中，512 token、K=7 的原版得到 16.14。整 session、K=3 的原版与日期修订分别得到 16.18、18.06。本轮重新测量 3.1 版本的原版和修订，两组共用一次冻结的模型、Controller 和技能嵌入。[论文附表 7](https://arxiv.org/pdf/2602.02474v3#page=16) · [固定测试协议](memskill-paper-f1.md)

## 看结果前固定的比较

- 写入和回答都用 Meta 发布版本 `0e9e39f249a16976918f6564b8830bc894c89659` 的 BF16 权重和对应 tokenizer。在 251 上使用同一个 vLLM 0.10.2 服务，贪心解码，seed 42，上下文上限 8192。实验保存权重、tokenizer 和配置的摘要。论文使用的服务端实现可能与本地不同。[模型发布记录](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659)
- 公开 conversation Controller checkpoint 的 SHA-256 为 `fe784612b0d4028f431379e7cae618d743761175532f0f8feb14959ed37d0226`。先运行原始技能库，再只给 `capture_temporal_context` 启用已冻结的 `dated_relative_turns`。各自从头建库，共用只读模型、原始描述和嵌入。缓存只复用同一服务版本、完全相同请求的确定性回答。记录实际新增日期证据的 span 数。
- 两组均为固定 512 token 分块、重叠 64、每块 7 个技能、Qwen3-Embedding-0.6B 检索 20 条。LoCoMo 两段完整历史共 314 题，每段和整体各报一次原生 F1。LongMemEval-S 为 101 题迁移测试。该日期规则只识别 LoCoMo 的日期和发言格式，预期 LongMemEval-S 的实际触发数为零。两个数据集都保留原版和修订的逐题结果。
- 假设为 LoCoMo 中保留下来的带日期源句能提高时间问题的 F1。若总体提升集中在单段对话、其他题型退化或实际日期写入为零，按实报告。论文分数作为外部参照，本地方法增益取同一数据集修订版减原版。测试集已有历史评测记录，候选在本次新模型测试前已冻结。

251 的评测脚本通过 `gpu_guard.py run` 启动，生成服务使用 7 号卡，检索和 Controller 使用另一张获配 GPU。运行目录为 `~/memskill-work/runs/llama31-paper-v1`，逐题输出按数据集、原版或日期版、历史 ID 保存。完成一个历史和一组技能后核对题号并原子发布结果，重启沿用已完成的历史。完整数据集通过题号验证后才写出 `comparison.json`。[运行入口](../../scripts/experiments/run_llama31_paper_pair_251.sh)

## Forward-only 论文能检验的假设

Huang、Balashankar 和 Aminifar 用小批输入前向计算激活的角度集中程度，找出小奇异值方向贡献较分散的权重矩阵，再在验证集上选矩阵和保留的秩比例。执行时保留较大的奇异值。文章表 2 的 36.7 倍是 GPT-J 层级搜索相对 LASER 穷举的平均计算加速。它没有给出 MemSkill 的问答 F1，也不能据此估计本项目的增益。[算法 1 与表 2](https://arxiv.org/html/2611.10610v1)

可把这个思路做成一个独立的 Reader 实验。固定已建好的记忆、每题检索到的 20 条和提示词，用训练对话的 Reader 输入做前向筛选，再用独立验证对话比较原始 Reader、筛中的单矩阵截秩、随机矩阵同秩截断。记录原生 F1 和推理耗时，并单独统计“检索文本已有答案但 Reader 答错”的题。这样可以检验提升是否来自模型使用证据的能力。若验证集支持，随后把冻结的 Reader 与原版／日期写入两组交叉测试，区分写入增益和模型权重适配增益。
