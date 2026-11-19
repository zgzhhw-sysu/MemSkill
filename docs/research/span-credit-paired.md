# 用单槽换技检验 span 信用是否提高问答 F1

研究问题是独立对话中的问答能否改善。每个 span 先由公开 Controller 对九项技能排序，再比较两个有序的三技能组合。两个组合只有一项不同。一个组合包含 `capture_activity_details`，另一个用原策略排序最高的可替代技能。其余两项的位置保持固定。

本实验给 `capture_activity_details` 配置了 `attributed_turn_pairs`。它在成功写入后附加带日期和说话人的相邻发言，以便检索短回答时能看到上文。所有对照组都拥有同一版本的九项技能。写入时可见当前 span、检索记忆和选中技能。问题、标准答案和标注证据不进入选技或写入输入。该技能可能增加写入量，评估同时统计写入次数及模型调用 token 数。

## 训练与对照

- 训练只使用 LoCoMo 前六条训练对话，各取 session 1–2 和 3–4，形成十二个两步 episode。在每个真实 span，以公平硬币采样一个三技能组合，并记录状态、选中组合、行为概率和另一个组合。对反事实后缀使用原硬币，重新计算该分支状态上的候选组合。
- 局部信用沿用证据规则 2。检查原写入及问答能否重现，再检查只换一个技能是否可重复地获得或失去答案。保持技能组合补写、补检也能解释变化时，该问题不给选择信用。每个 span 最多探查四道关联训练题。
- 条件策略的初始对数几率等于两个候选技能的原始 logit 差。训练在该值上加一个带正则化的线性校正，输入是常数项和归一化后的原生状态向量。独立评估只用冻结权重及当前状态，按校正后的概率选择三技能组合。
- `native` 使用原生贪心排序。`terminal` 从同一批采样动作学习，每步使用 episode 终局 F1 减去其他 episode 的均值。`credit` 使用相同终局项，再加入有独立选择证据的 span 差值。对原采样动作按差值的符号及绝对值训练。额外项在有非零信用的 span 上取均值，单步差值裁剪到 ±0.5。`always` 每步选带来源发言的组合，用于检验始终使用该技能的效果。
- 报告实际部署的二选一概率，并在同一份已记录状态上分别重算 `terminal` 和 `credit` 的概率。这是给定这两种组合时的条件概率。另计实际发生的三技能组合切换数及变动的问答。

验证集还运行 `rate` 对照。它保留 `terminal` 的全部状态系数，只调整一个截距，使其在完整训练窗口上的平均选技概率与 `credit` 一致。比较两组在相同验证状态上的贪心选择和实际逐题 F1，可检查收益是否仅靠整体提高某项技能的使用概率就能复现。概率匹配不保证贪心使用次数相同。

251 上的实验在打开最终测试集之前，对两个验证前缀统一执行五组比较。信用策略相对于每个对照都需要有正的平均 F1 差值，并且至少在一个对照状态上改变贪心选择。只统计实际选技和读入问答的提示都发生变化的题时，配对 F1 差值之和也须为正。完整报告在 `validation/selection.json`。验证集只用于选择策略，正式收益仍由冻结后的测试集估计。

第一轮若无法通过上述验证，第二轮只扩充同六条训练对话，加入 session 5–6 和 7–8。这样可从 12 个训练窗口、24 个 span 增到 24 个窗口、48 个 span。窗口、种子和每个 span 的四题探查预算提前固定，验证和测试的对话及题目沿用第一轮。第二轮使用独立的实验目录，前四个 session 的训练轨迹可在核对协议和训练记录后复用。

## 固定策略的验证结果

251 上使用同一套技能、BF16 生成服务和读取器，完成了以下两种固定选技策略的配对比较。F1 是对每道题分别计分后的平均值。

| 验证对话 | 题数 | 原始贪心策略 | 始终选带来源发言的技能 | F1 差值 | 改善／退步／持平 |
| --- | ---: | ---: | ---: | ---: | ---: |
| conv-47，前六个 session | 30 | 0.2300 | 0.3074 | +0.0774 | 12／7／11 |
| conv-48，前六个 session | 45 | 0.2202 | 0.3290 | +0.1089 | 17／13／15 |
| 按题合计 | 75 | 0.2241 | 0.3204 | +0.0963 | 29／20／26 |

两组在 12 个 span 都选择了不同的技能组合。逐题从最终记忆库重建读取提示后，49 道得分发生变化的题均读到了不同的提示。这个差值衡量固定策略本身的收益。局部信用的独立贡献需用 `credit` 与 `terminal`、`rate` 和 `always` 的结果判断。逐题运行记录在 251 的 `~/memskill-work/study/validation/{native,always}/`，聚合审计在 `~/memskill-work/runs/paired-251/fixed-controls.json`。

在 8/12 个训练窗口的中间检查点，16 个训练 span 中有 4 个得到非零选择信用。在同一批 12 个原始验证状态上，`terminal`、`credit`、`rate` 选择带来源发言技能的平均概率依次为 0.4160、0.6535、0.6692，贪心选择数依次为 0、12、12。用这三组临时权重逐步核对两条验证对话实际经过的状态和动作，`terminal` 在两条历史上都复现原始策略，`credit` 和 `rate` 都复现 `always`。因此在 75 题上，临时信用策略相对终局训练提高 0.0963 F1，相对只调训练平均概率的对照提高 0.0000。这个检查点支持整体选技偏好变化带来的收益，尚未显示逐 span 分配的额外收益。轨迹步数和原评分文件的 SHA-256 记录在 251 的 `~/memskill-work/runs/paired-251/interim-eight-dev.json`。

## 测试范围

| 数据 | 使用方式 | 样本 |
| --- | --- | --- |
| LoCoMo 验证 | 选择方案，检验查询和写入链路 | 两条固定验证对话，各取前六个 session，评估证据完整落在前缀内的可评分题 |
| LoCoMo 测试 | 冻结后顺序读完整历史，再逐题评估 | 两条对话、55 个 session、314 道原生可评分题。单跳 160、多跳 69、时间 65、开放域 20。原生协议排除的 86 道对抗题单列覆盖情况 |
| LongMemEval-S 迁移 | 使用 MemSkill 固定测试索引与冻结的 LoCoMo 策略，按历史顺序处理全部 session | 105 段历史、5,027 个 session。六类题分别报告。101 道可回答题用于主 F1，4 道弃答题另列 |

LoCoMo 报告每段对话的配对差值以及各题型 F1。LongMemEval-S 在同一段历史上配对，并按题型分层对历史做 bootstrap。使用 token F1 衡量回答重合度，官方 LLM judge 准确率属于另一项指标。两个基准分别汇报。[预先固定的样本、文件指纹与指标](../../experiments/controller/protocols/paired_grounding_cohort.json)

## 推理负载

训练用的 [Transformers 服务](../../scripts/experiments/serve_low_duty.py) 每次接一个请求，提示按 16 个 token 分块。每段预填充和每个解码步都等待 CUDA 完成后再休眠，默认工作时间比例为 0.40。PyTorch 的本进程分配上限为物理显存的 76%。用 `nvidia-smi` 分别记录实际显存、利用率、温度和功率。

评估也可使用相同 BF16 权重转换的 GGUF。对于所用 llama.cpp 版本 `a956a85`，编译 [计算限流器](../../scripts/experiments/llama_process_duty.cpp) 时包含该版本的 `include/`、`ggml/include/`，使用 `-std=c++17 -O2 -fPIC -shared -ldl -pthread`。仅在实验服务进程中设置 `LD_PRELOAD` 指向编译出的动态库，并设置 `MEMSKILL_DECODE_DUTY=0.45`。以 `-b 32 -ub 32 -np 1 --no-warmup --no-cache-prompt` 启动。第一次本地请求后检查服务日志出现 `Limited compute: duty=0.45`，这表明限流器实际覆盖了模型计算。

## 复现

在仓库根目录设置 `STUDY_DIR`、`LOCOMO_DATA`、`CONTROLLER_CHECKPOINT`、`MEMSKILL_QWEN_EMBEDDING_PATH` 和 `MEMSKILL_API_BASE`。`MEMSKILL_PYTHON` 可指定实验用的 Python。基础模型为同一本地 Llama-3-8B-Instruct，编码器为 Qwen3-Embedding-0.6B。为多个等价端点共用结果时，将 `MEMSKILL_BACKEND_ID` 设置为相同的模型指纹和推理配置。研究机上实验期间关闭四张卡的显存占用进程。

```bash
bash scripts/experiments/paired_grounding.sh train
bash scripts/experiments/paired_grounding.sh validation
python -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" validation --arms native terminal credit always rate
touch "$STUDY_DIR/require-selective-validation"

# 冻结方案后执行
bash scripts/experiments/paired_grounding.sh test
python -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" test

# LONGMEMEVAL_DATA 指向 longmemeval_s_cleaned.json
for arm in native terminal credit; do
  bash scripts/experiments/paired_grounding.sh transfer --study-arm "$arm"
done
python -m experiments.controller.audit_paired "$STUDY_DIR" "$LONGMEMEVAL_DATA" longmemeval
```

第二轮在新的 `STUDY_DIR` 下使用 `bash scripts/experiments/paired_grounding.sh train --study-train-sessions 8`。后续验证、测试和断点续跑从该目录的 `protocol.json` 读取训练窗口范围。

`train --study-reverse` 可供另一低负载工作进程从反向顺序领取训练 episode。每个完成的 episode 独立保存来源、轨迹、原始回答和归因记录。评估时，若同一输入、模型和技能版本的已有完整轨迹在新策略下逐步选出完全相同的动作，可复用实际回答。复用记录保存参考轨迹、核对步数和分数文件的 SHA-256。

逐题审计从写入日志重建最终记忆库，核对读取的记忆版本和顺序，并重建实际问答提示。对每道 F1 变化的题，报告两组读取的提示是否不同，以及检索结果中哪些条目最后写于选技不同的 span。某次写入也可能通过后续检索和改写间接影响结果。若两组动作完全相同或问答提示相同，该题的 F1 差值单列，不能直接归因于 span 选技。
