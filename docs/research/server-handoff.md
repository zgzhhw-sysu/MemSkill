# MemSkill 换机交接 · 2026-11-20

251 的写入格式复核与新增因果样本结果分别在 `source-compression/writer-block-repair/` 和 `late-pairs/selection-labels/`，均由 248 自动收回。后期策略加入第 9–12 个 session 的 88 题后，只在之后的 378 题上计分。[运行与恢复方法](server-251.md#写入格式复核和后续因果训练样本)

251 的后期三技能实验已固定六段训练对话的 25 个窗口、209 道题。248 备份目录为 `../runs/span_credit/251-bf16/linked-eight-gpu/logs/late-pairs/semantic-writer/`。交接时一起保留该目录、同级的 `outcomes/` 和上级 `source-compression/retrieval-competition/`。双技能和三技能结果各自按完成窗口发布。[251 启动与恢复命令](server-251.md#后期-25-个窗口的三技能实测)

11 月 21 日起继续在 251 做四路径的末段归因复核和同原文写入实验。248 在 `../runs/span_credit/251-bf16/linked-eight-gpu/logs/source-compression/` 收回 `factorial-repairs/`、`terminal-capacity/`、`semantic-writer/`。恢复实验时先核对原始四路径目录、已冻结的概率和各窗口回执。三个目录分别保存补写补检结果、固定使用率上界和关系技能的真实写入与回答。[251 服务与恢复命令](server-251.md)

跨窗口补充训练面板有 185 道题、99 处末段干预。251 的 `memskill-251-cross-prefix` 保留每次干预之前的完整对话和记忆。248 的结果在 `linked-eight-gpu/logs/late-pairs/cross-window-prefix/`，保留 `manifest.json`、各条原始轨迹、响应缓存和 `through-*.json`。进度按已经发布的 `conv-*_to*/pair.json` 计数。单条路径出分后仍需另一条路径和逐题核对。[服务与恢复](server-251.md)

251 的条件选技校准由 `memskill-251-prefix-return` 持续运行。248 在同一 `linked-eight-gpu/logs/late-pairs/` 备份目录收回 `prefix-return/` 和 `rank-return/`。每种方法均保存训练来源、概率参数、早于参数冻结的路径指纹及逐题 F1。恢复时先同步这两个目录及同级原始轨迹，再从部署脚本继续；已发布窗口会复用。当前先导的 52 个历史问题关联及 14 道完整四路径留出训练题均属于训练诊断。[实验结果](algorithm-notes.md) · [251 运行方法](server-251.md)

后续配对实验的两个用户级服务为 `memskill-251-late-pairs` 和 `memskill-251-audit-late-pairs`。数据在 `~/memskill-work/runs/paired-251-linked-eight/late-pairs/`，248 收回到 `runs/span_credit/251-bf16/linked-eight-gpu/logs/late-pairs/`。保留 `cohort.json`、`features.json`、`frozen.json`、`step-mixture/` 和完整的 `outcomes/`。重跑时逐路径校验已经落盘的分数和写入，只计算缺失路径。首轮借用现有 7 号卡读者，后续由同配置的独立读者接续。终局、原生、两个校准版本及置换对照共用同一批真实轨迹。[251 运行说明](server-251.md) · [实验判据](algorithm-notes.md)

251 的 `memskill-251-write-components` 复用主训练的 18773 限载读者，每次提交一条请求。11 个历史训练窗口的四组记忆库共对应 119 道题，计划、提示、逐题回答和策略结果在 `~/memskill-work/runs/paired-251-linked-eight/write-components/`。查看 `answers/` 可获知已完成的题，`eligible-labels.json` 包含所有能改变末段权重的问题。GPU 2 负责检索和策略拟合。达到拟合门槛时生成 `frozen.json`，否则生成 `screened-out.json`。完整读者分解写入 `summary.json`。248 每五分钟收回文件。[算法与判据](algorithm-notes.md)

251 的 `memskill-251-conditional-credit` 和 `memskill-251-query-prototypes` 会在四路径收齐后用 GPU 2 复核已冻结的候选。权重、概率、执行计划及逐题结果位于 `~/memskill-work/runs/paired-251-linked-eight/source-compression/{conditional-credit,query-prototypes}/`。六条训练对话的邻句读取筛选位于同一运行根下的 `neighbor-read/`。248 回收服务同步这些子目录。候选冻结时间、可用前瞻题数和当前判断见[算法实验笔记](algorithm-notes.md)。

251 已接续 248 的训练。两台机器的 Python 均为 3.12.12，配对实验所用的 109 个 Python 依赖逐项版本一致。目标机的代码在 `~/memskill-work/MemSkill`，训练断点和模型路径见 [251 运行说明](server-251.md)。

11 月 20 日 21:56，251 已启动新训练 `memskill-251-linked-eight`。运行目录为 `~/memskill-work/study-linked-eight-gpu`，逐卡日志在 `~/memskill-work/runs/paired-251-linked-eight/`。旧四题预算实验已完成，75 题开发集上的信用、原策略及终局奖励 F1 均为 0.41787。248 的私有备份 `runs/span_credit/251-bf16/recovered-20261120-2040/` 包含 4,292 个逐文件核对 SHA-256 的普通文件。新实验的进度按已原子发布的训练窗口计数。前瞻选技概率保存在新实验日志的 `source-compression/frozen.json`，SHA-256 为 `a9d72d31f2716b05ef4d860307a6d46a4e5eed028d38114cc0d455f103e6be5d`。路径级对照使用独立的 GPU 7 生成后端；四种选技顺序的真实回答均由同一后端采集。248 的副本位于 `runs/span_credit/251-bf16/linked-eight-gpu/logs/source-compression/`。

11 月 20 日 10:38，GPU 端已保存 `conv-26_s1` 和 `conv-44_s3` 两个完整训练窗口，四个 span 的初轮规则 2 额外信用均为零。工作进程在窗口落盘后已重启为 CUDA 拟合版本。7 号卡按相同编码指纹补探四题预算遗漏的问题。77 题的记忆分组 GPU 对照已完成。新候选在每个 span 按技能描述选择五条发言，879 道训练题上的证据覆盖筛选已完成，真实写入库上的 123 题读者实验在 `~/memskill-work/runs/paired-251/skill-budget-writer/`。两项训练筛选的代码入口和评分范围见[配对实验记录](span-credit-paired.md)。

11 月 20 日 09:55，原候选队列已结束。训练写入与部署对齐的版本收齐 12 个窗口，但在两段开发对话上，信用、固定活动技能、匹配使用率三组同为 0.32039。原队列的三个候选均未通过。另登记一轮 CPU 编码策略的独立复核，其六组开发结果和四份冻结输入已有指纹。两套正式测试沿用此前清单，共 314 道可评分 LoCoMo 题和 105 条独立 LongMemEval-S 历史。结果无论正负均报告。[CPU 复核登记](../../experiments/controller/protocols/cpu_anchor_holdout.json)

11 月 20 日 06:03，原 GPU 编码的零信用约束策略完成两个开发前缀的 75 道题。信用组 F1 为 0.28743，终局组为 0.22411，固定使用活动技能和匹配训练使用率两组均为 0.32039。该候选未达到开发门槛。训练与部署写入提示对齐的候选继续原计划。新增的等来源训练单独写入 `~/memskill-work/study-equal-sources`，通过 `memskill-251-equal-sources.service` 运行，当前使用 CPU 编码和已有的 5、6 号卡生成服务。两个实验均按窗口续跑。正式候选的优先顺序和两个基准的测试清单已冻结。

06:08，等次数开发对照也已核验，F1 为 0.29407。7 号卡交给外部开发验证，当前使用独立的 18757 端口。可用 `systemctl --user status memskill-251-transfer-validation memskill-251-external-odd` 查看两个客户端。训练窗口的双向读者复核已经覆盖八个 session 版本的 24 个窗口。对 7 个待复核的问题信用，有 4 个通过、1 个存在竞争性修复、2 个没有复现。重新拟合后的留一对话单步期望 F1 为 0.33775，匹配训练使用率对照为 0.34016。

06:20，GPU 编码环境复测 CPU 选择后，实际只替换了一步。75 道开发题的 F1 从 0.28743 升至 0.30618，固定技能对照为 0.32039。两条对话的分数和实际读写轨迹在 `study-anchor-gpu/diagnostics/validation-schedule/`。248 的私有迁移副本位于论文工作目录下的 `runs/span_credit/251-bf16/migration-snapshot/`，包含原 GPU 开发决定、交叉重放和已完成的对齐写入训练窗口。等来源实验通过 `memskill-251-equal-sources` 和 `memskill-251-equal-reverse` 从两端领取窗口，使用同一模型服务和逐窗口文件锁。

06:33，反向重放完成并已同步回 248。在 CPU 配置上，两段前缀执行 GPU 选择序列时为 0.30634，执行 CPU 选择序列时为 0.33274。两个方向的问答、实际写入和选择输入指纹已分别保存在两个研究目录中。外部验证的冻结文件及已完成请求也已备份。对齐写入训练的首对 session 覆盖六条对话，12 个 span 中有 6 个取得非零信用，旧写入为 3 个。固定到两种提示都能重现的 25 个问题与 span 配对时，直接信用策略的单步期望 F1 为 0.36807，匹配使用率对照为 0.36997。

截至 2026-11-20 05:45，扩充训练已完成 24/24 个窗口，48 个 span 中有 8 个获得非零信用。该版在两个开发前缀上与始终选活动细节、只调使用率两组的 75 个逐题答案完全一致，三组 F1 都为 0.32039。终局组 F1 为 0.29622。五组开发验证已完成，扩充版未通过筛选，原 GPU 编码环境的零信用约束版本已接管 4、7 号卡做验证。训练和部署写入提示一致的实验已完成 6/12 个窗口。此前 CPU 编码的零信用约束版本在 12 个相同状态上将平均概率从 0.4431 改为 0.5227，开发 F1 为 0.33274，等使用次数随机安排位置的对照为 0.30920。该版本在独立 LongMemEval-S 验证历史上继续运行。完整正式测试等待候选筛选。[实验方法和开发集结果](span-credit-paired.md) · [固定测试清单](../../experiments/controller/protocols/paired_grounding_cohort.json)

## 从最初的迁移包恢复

下表是 248 创建迁移包时保留的六个窗口。已有 251 工作目录时，以目标机的实际断点为准。

| 训练窗口 | 已完成，可续用 | 迁移后补跑 |
| --- | --- | --- |
| session 1–2 | conv-26、30、41、42 | conv-43、44 |
| session 3–4 | conv-43、44 | conv-26、30、41、42 |

源机 GPU 7 报过 `GPU is lost`，原因未定位。源机实验已停止。源机空闲守护仅配置 4、5、6 号卡，每卡上限 2048 MiB；交接时驱动初始化返回错误 999，三卡均为 `retry_wait`、实际占用 0 MiB。CPU 验证中断于首条对话，须从验证阶段起点重跑。迁移包只收录原子落盘的完整训练窗口，每个窗口包含采样、问答索引、调用用量和反事实轨迹。迁移演练已用真实训练入口读取新目录下的 checkpoint、模型和协议，按正序跳过已有窗口后，在首个待跑窗口 `conv-43_s1` 前停下。

源机迁移目录为 `/home/lixian/Documents/研究生大论文/runs/handoff/span-credit-20261120/`。整目录约 17 GiB。它含本次发布的 Git bundle、Controller checkpoint、两份原始数据、Llama 3 8B BF16、Qwen3-Embedding-0.6B、tiktoken 缓存和训练快照。模型、数据及对话轨迹随私有目录转移；GitHub 提供代码和文档。需要在目标机上另备 Python 环境及匹配的 NVIDIA 驱动。

在源机用可断点续传的命令复制。将目标改为自己的 SSH 登录地址。

```bash
NEW_SERVER='user@new-server'
rsync -a --partial --info=progress2 \
  '/home/lixian/Documents/研究生大论文/runs/handoff/span-credit-20261120/' \
  "$NEW_SERVER:span-credit-20261120/"
```

以下命令在新机的 Bash 中运行。校验和覆盖模型分片、数据、训练窗口、代码包及交接材料。保留原始快照，训练在独立工作目录续写。

```bash
export HANDOFF_DIR="$(realpath ~/span-credit-20261120)"
(cd "$HANDOFF_DIR" && sha256sum --quiet -c SHA256SUMS) || exit 1
mkdir -p ~/memskill-work
cd ~/memskill-work
git clone --branch main "$HANDOFF_DIR/MemSkill.bundle" MemSkill
cd MemSkill
git bundle verify "$HANDOFF_DIR/MemSkill.bundle"
git switch -c research/llama8b-designer --track origin/research/llama8b-designer
git remote set-url origin https://github.com/zgzhhw-sysu/MemSkill.git
export STUDY_DIR="$HOME/memskill-work/study"
if [[ ! -e "$STUDY_DIR" ]]; then cp -a "$HANDOFF_DIR/study" "$STUDY_DIR"; fi
python3 -m experiments.controller.resume_status "$STUDY_DIR"
```

首次应显示 `6/12 episodes, 12 spans, 4 nonzero span credits`。以后可重复运行最后一条命令查看进度。校验器会把已保存的采样动作及信用与相应写入轨迹逐 span 核对。

## 准备运行环境

源机使用 Python 3.12.12、PyTorch 2.10.0+cu128、Transformers 4.53.3、sentence-transformers 5.6.0。下例适用于驱动支持 CUDA 12.8 的目标机。其他驱动环境需安装兼容的 PyTorch，并记录实际软硬件版本。Ubuntu x86-64 上可按两台服务器实际核对过的 109 项版本安装。[251 配对实验的完整版本清单](../../scripts/experiments/requirements-paired-cu128.txt)

```bash
python3.12 -m venv ~/memskill-work/venv
source ~/memskill-work/venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
python -m pip install -r scripts/experiments/requirements-paired-cu128.txt
python -m pip check
export MEMSKILL_PYTHON="$(command -v python)"
export LOCOMO_DATA="$HANDOFF_DIR/assets/locomo10.json"
export LONGMEMEVAL_DATA="$HANDOFF_DIR/assets/longmemeval_s_cleaned.json"
export CONTROLLER_CHECKPOINT="$HANDOFF_DIR/assets/conversation_controller.pt"
export MEMSKILL_QWEN_EMBEDDING_PATH="$HANDOFF_DIR/models/Qwen3-Embedding-0.6B"
export TIKTOKEN_CACHE_DIR="$HANDOFF_DIR/assets/tiktoken"
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=4
export MEMSKILL_API_BASE=http://127.0.0.1:18752/v1 MEMSKILL_API_TIMEOUT_S=600
export MEMSKILL_DEVICE=cpu
export MEMSKILL_BACKEND_ID="hf-llama3-8b-bf16-prefill16-duty40-cpuencoder-$(hostname -s)"
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests
```

## 限流与断点续跑

在调度器分配的一张健康卡上启动推理。源机 24 GiB 卡的生成进程占约 16 GiB。下面默认用 CPU 编码，避免生成与编码挤占同一卡。源机的六个训练窗口使用 GPU 编码，换硬件或编码设备可能产生数值差异。需要整批同硬件复现时，把 `STUDY_DIR` 指到新的空目录，从头采集 12 个窗口。所有正式评测组应使用相同的生成服务、编码设备与 `MEMSKILL_BACKEND_ID`。

如果目标机已有显存占用守护，先通过该机的实验租约入口释放占用，或在实验期间暂停守护，等模型服务退出后再恢复。沿用目标机获配的 GPU 编号。源机的 4–7 号和 systemd 服务名仅适用于源机。空闲守护只持有限量显存，源机设置为每卡最高 2048 MiB。

在同一终端，先将 `CUDA_VISIBLE_DEVICES` 设为本次分配的单卡，或保留调度器设置的单卡环境，再启动下列服务。服务每次处理一个请求，预填充块为 16 token，计算间隔休眠，目标工作时间比例 0.40。用 `nvidia-smi` 观察实际负载；该比例不是瞬时利用率上限。

```bash
: "${CUDA_VISIBLE_DEVICES:?请先选择目标机获配的一张 GPU}"
mkdir -p "$STUDY_DIR"
python scripts/experiments/serve_low_duty.py \
  --model "$HANDOFF_DIR/models/Meta-Llama-3-8B-Instruct" --port 18752 --duty .40 \
  >"$STUDY_DIR/server.log" 2>&1 &
MODEL_PID=$!
trap 'kill "$MODEL_PID" 2>/dev/null || true' EXIT
for ((i=0; i<120; i++)); do
  kill -0 "$MODEL_PID" 2>/dev/null || { tail "$STUDY_DIR/server.log"; exit 1; }
  curl -fsS --max-time 2 "$MEMSKILL_API_BASE/models" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "$MEMSKILL_API_BASE/models" || exit 1
export CUDA_VISIBLE_DEVICES=''
bash scripts/experiments/paired_grounding.sh train
python -m experiments.controller.resume_status "$STUDY_DIR"
```

训练成功后应有 12 个窗口、24 个 span，以及 `terminal.json`、`credit.json` 两份策略权重，以及匹配训练平均选技概率的对照权重 `rate.json`。中途退出后，重启服务并重跑相同命令即可。仅发布成功的完整窗口会被复用。服务端口或后端配置发生变化时，核实客户端连接地址和后端标识。

## 验证、冻结与正式测试

先在两段验证对话的前六个 session 比较五组。环境和工作目录沿用上文。

```bash
touch "$STUDY_DIR/require-selective-validation"
bash scripts/experiments/paired_grounding.sh validation &&
python -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" validation \
  --select-for-test
```

读取 `validation/selection.json`。先比较 `credit_minus.terminal` 的同状态概率变化、贪心切换数、实际三技能组合变化和逐题 F1。再检查 `changed_question_evidence` 中两组读到的提示及记忆来源。与 `always` 相同的轨迹要单列，这种结果不足以证明策略会按 span 选择性地使用该技能。若验证退化，保留该版，用训练和验证对话继续迭代。

选定方案后，从仓库根目录运行完整评估。入口先核对五组验证结果，补齐等次数对照，再冻结三份策略权重和实验协议。它在每个阶段分别执行六组比较，续跑时校验已保存的指纹和逐段结果。

```bash
bash scripts/experiments/paired_holdout.sh
```

迁移运行器默认从 Qwen 编码器的同级目录 `Meta-Llama-3-8B-Instruct` 读取写入模型的 tokenizer 和 `config.json`。模型在其他位置时，设置 `MEMSKILL_WRITER_TOKENIZER_PATH` 指向实际服务端权重目录的本地副本。长 session 分多次写入并记录原文范围，各段共享一次选技和总计 256 个生成 token。六组轨迹使用相同的 tokenizer 指纹。

LoCoMo 主测试含两段完整对话、55 个 session、314 道可评分题。预设目标为相对 `terminal` 和 `native` 的逐题平均 F1 均至少提高 **0.02**，且两段对话分别为正。另有 86 道无普通参考答案的对抗题按原协议单列。LongMemEval-S 迁移测试含 105 段完整历史、5,027 个 session，覆盖六类问题；101 道可回答题用于主 F1，4 道弃答题另报。报告逐类结果和按历史配对、按题型分层的 95% 区间。两个数据集分别报告，保留逐题回退和无法由选技解释的变化。

六组正式报告在 `test/comparison-with-count.json` 和 `longmemeval/comparison-with-count.json`，冻结清单及运行日志在 `holdout/`。每个已完成的评估历史保存实际问题编号、分数、决策、轨迹及用量；完整历史间可续跑。实验结束后在启动服务的终端执行 `kill "$MODEL_PID"; wait "$MODEL_PID"`，并按目标机约定释放租约或恢复限量空闲守护。

### 确定性评估请求的复用

运行器在 `validation`、`test` 和 `transfer` 阶段复用同一研究、同一阶段、同一本地后端的贪心回答。请求指纹包含完整提示、模型、生成参数和读写用途。记录保存首次生成的回答、服务端输入和输出 token 数及校验值。拷贝可审计的评估结果时，同时迁移 `$STUDY_DIR/response-cache/`、各组的 `usage.jsonl` 和轨迹目录。审计逐次核对实际提示、回答和 token 数，`executor_tokens` 与 `reader_tokens` 表示实际请求模型的用量，命中的用量分别列为 `executor_cached_tokens` 与 `reader_cached_tokens`。

两条 CPU 编码开发对话中，整段动作相同的轨迹已先行复用。余下实际执行的 48 次写入和 300 次问答仍有 12 次写入和 33 次问答完全重复，涉及 32,961 个输入和输出 token。各对重复调用的实际回复和 token 数逐一相同。训练、换技重放和重复性检查继续发出独立模型请求。

LongMemEval-S 的六组评估按同一条完整历史依次执行。对于冻结的 Qwen 编码器，只有编码实例、文本顺序、批大小和向量归一化设置全部一致时才复用该批向量。每条历史结束后释放缓存，运行日志记录命中和重新计算的批数。选技、记忆写入、读取的证据和逐题分数仍分别保存。

在 251 用真实 Qwen 和原始 Controller 重建了 `conv-47` 的两个不同策略。原生策略首次计算 19 批向量，信用策略的 19 批中有 7 批与前一组完全相同，重新执行原生策略时 19 批均可复用。每组 6 次选技的完整状态、概率和动作、全部记忆修改、30 题的前五条检索、答案和 F1 均与历史产物一致。此次重放的读写回答取自相同请求的实测归档。原生和信用组 F1 分别为 0.1969 和 0.3411。

### 251 上的相邻发言检索实验

在 251 检查 `systemctl --user status memskill-251-turn-key-screen memskill-251-turn-key-reader`。数据目录为 `~/memskill-work/runs/paired-251-linked-eight/source-compression/turn-keys/`，先看 `screen-summary.json` 的证据覆盖，再看最新 `reader-through-N.json` 的配对题数和 F1。比较 `policy` 与 `new_native` 可度量仅移动选技概率的增益，比较 `policy` 与 `matched_rate` 可度量针对具体窗口选择的增益。`reader/<窗口>/jobs.json` 保留原始分支未能复现的原因。248 镜像位于项目父目录下的 `runs/span_credit/251-bf16/linked-eight-gpu/logs/source-compression/turn-keys/`。[实验协议](../../experiments/controller/protocols/turn_key_credit.json)

检索竞争策略与首段归因核查都在 251 上运行。248 定期镜像到论文工作目录的 `runs/span_credit/251-bf16/linked-eight-gpu/logs/source-compression/`。在 `retrieval-competition` 中并列读取最新 `through-NN.json` 的 `all` 和 `prospective`；在 `root-write-repairs` 中查同一问题在两个后续动作条件下的独立核查。跨窗口、跨模型的数字分别保留原分母和后端。[251 服务与产物说明](server-251.md)

251 重启、瞬时服务消失时，可从部署目录重新启动本轮三个任务。脚本复用已落盘的不可变回执与确定性模型响应。服务仍存在时可直接用 `systemctl --user status` 查看它。

```bash
source ~/memskill-work/deploy/env-251.sh
cd "$work/run-linked-questions-v1"
systemd-run --user --unit=memskill-251-retrieval-competition bash "$PWD/scripts/experiments/run_retrieval_competition_251.sh"
systemd-run --user --unit=memskill-251-root-write-repairs bash "$PWD/scripts/experiments/run_root_write_repairs_251.sh"
systemd-run --user --unit=memskill-251-root-overlap bash "$PWD/scripts/experiments/run_root_overlap_251.sh"
```

### 在 251 接手自然回复实验

在 248 的论文目录下，查看 `runs/span_credit/251-bf16/linked-eight-gpu/logs/source-compression/reply-placement/`。先读 `calibration.json` 的 91 题四路径结果，再对照最新 `through-NN.json` 中的 `all`、`prospective` 和 `session13_plus`。其中 `reply` 使用真实相邻回复，`sham` 使用同一说话者的另一条发言，`raw` 使用相同特征和原始终局奖励，`rate` 保留拟合策略在训练样本中的平均使用率。`robust` 是上一版使用解析复核信用的策略。训练历史内的结果单独记录。独立验证与正式测试仍按原登记的完整历史评价。[服务和恢复命令](server-251.md)

### 案例提示实验的状态

在 248 的同步目录 `runs/span_credit/251-bf16/linked-eight-gpu/logs/source-compression/case-credit/` 查看 `choices/`、`joint/choices/` 了解已完成多少组实际决策。核对 `joint/later/through-*.json` 的 `session13_plus` 和 `strict_session13_plus`、对话覆盖及分类型得分。模型在同一概率下的实际读写 token 与额外的选技推断开销分别记录。完整独立测试仍遵循原协议。[251 服务与恢复步骤](server-251.md)

### 逐题证据匹配控制器

251 是计算机，248 是源码与结果镜像机。接手时从 248 同步目录 `runs/span_credit/251-bf16/linked-eight-gpu/logs/source-compression/evidence-transport/` 开始。先导策略已冻结 64 个窗口的数值预测。91 题的选技概率平均移动 4.16 个百分点，期望 F1 比原生低 0.00120。先看 `pilot/suffix-diagnostic.json` 了解首段训练目标对末段动作的敏感性，再检查 `late-pairs/selection-labels/through-*.json` 与 `late-pairs/donor-factorial/` 的原路径核验和互补路径进度。扩展策略须集齐 88 道训练题的原始核验记录，其冻结后评估范围是预先保留的 378 题。训练诊断与正式完整历史分别汇报。[详细实验记录](algorithm-notes.md) [251 查看、复算与恢复](server-251.md)

### 首段收益估计与跨对话方向

在 248 的镜像目录下读取 `logs/source-compression/assisted-span-backup/masked-policy.json`。相同四路径题给出了两种不同检验：能否估准供样题上的首段收益，以及所得策略能否把正确方向迁移到其他对话。首个候选已冻结 58 个窗口，91 题期望 F1 为 0.46137，原生为 0.46189。正式判断查 `expanded/later/` 和 `resolved/later/` 的覆盖、严格前瞻结果、固定基准与等频对照；仅在既有条件全部通过后进入完整历史验证。[251 运行与恢复](server-251.md)

### 查验时间修订技能的实际影响

在 248 镜像的 `logs/source-compression/temporal-update-writer/` 先查 `writes.json` 中的 `older_updates`，再对照同窗口 `pair.json` 的逐题答案、F1 和读出的记忆版本。已有真实首例修改了一条旧记忆。候选动作覆盖 91 道早期题和 209 道较晚训练题，全历史测试仍按原保留集执行。后期窗口全部回收后比较原生、等频、错配前文和固定技能。归因候选需另做规则 2 对照。[251 运行、回执与恢复](server-251.md)

首个 9 题窗口已收回，来源策略 0.46852、原生条件策略 0.45066。两处分数变化中一升一降，额外信用均为零。第二个登记队列从同一训练历史的 session 9–10 开始，与早期窗口交替执行。检查 `reader-inputs.json` 可看到 GPU 1 归还前冻结的检索索引、版本与读者提示，`reader.log` 记录其后由 GPU 7 完成的问答。

已用首个窗口核对分开执行编码与问答的结果。9 题的答案、F1、检索索引及记忆版本与原记录逐项相同，其中 5 题的修订分支检索上下文有变化，5 次读者请求都命中此前实测的同一提示。核验回执为 `reader-equivalence-first.json`。

### 交叉库核验和新策略

在 248 镜像的 `logs/source-compression/temporal-update-audit/`，首个 9 题窗口的两项来源效应已经拆开。旧项贡献为 0，新增条目贡献了原策略观察到的全部 +0.01786 期望 F1。另一个 8 题窗口在两种固定技能下均为 0.46429。两者来自同一条训练对话，独立基准仍封存。查看 `controller/*/forecast.json` 的供样窗口和各状态选技概率，并用对应的 `through-*.json` 核对比单独预测总收益、等频策略、原生策略多得多少 F1。只有通过登记的筛选后才进入独立开发与完整测试。[251 的运行入口和回执](server-251.md)

### 继续核验截断写入

251 已在 conv-43 第 112 题实测出同技能修复使 F1 从 0 变为 1。查看 `logs/source-compression/capped-source-credit/conv-43_s11/01/questions/112.json` 的原答案和重测答案，邻近的 `inputs.json` 和 `writes.json` 保存检索输入、记忆版本及写入记录。检查同级 `complete.json` 确认 36 次分支问答全部到齐，再读取 `policy/repaired/` 的策略结果。策略对照的主要指标是目标 span 选择概率及其加权 F1，同时检查按对话的得分、等频对照、终局奖励对照和置信区间。独立开发和完整测试的准入条件见登记协议。[运行与断点恢复](server-251.md) · [控制器实验协议](../../experiments/controller/protocols/capped_source_credit.json)

conv-43 这条路径已完成全部 10 题，固定技能重测后均分增加 0.15714，换技能的原始平均优势从 0.21464 缩到 0.05750。临时将受影响的训练比较置零后，88 题回算的期望 F1 为 0.47455，原生为 0.47451，终局策略为 0.47981。正式目标仍待跨对话和独立数据检验。

后续晋级依据使用 `policy/joint-repaired` 和 `rank-allocation/joint-repaired`。两者都逐题应用已测出的写入格式和截断修复。旧目录 `policy/mask` 中 conv-43 的两条历史负信用后来已被同技能修复否决。新的 `fit.json` 包含每个外层目标对话、内层供样历史和阈值，`calibration/summary.json` 是 88 题回算，`later/through-*.json` 中的 `strict` 只统计数值冻结后才出分的窗口。[251 恢复命令](server-251.md)

排序版会继续处理 `rank-allocation/{joint-mask,joint-repaired}/long-prefix/`。这里的 `registration.json` 绑定 99 个端点、185 个跨 session 的问题和冻结时已经有分数的分支，每个端点有独立回执。88 题上的平均选技概率变化已达到 6.14 个百分点，F1 为 0.47301；信用相对只用回报增加 0.00071，整体仍低于原生 0.47451。后续要检查新出分窗口及跨 session 问题上的收益能否重复。

### 最早和最后证据的比较

251 正在为历史训练面板的 185 道跨 session 问题补测最早证据位置的技能干预。248 镜像的 `logs/late-pairs/first-evidence/` 中，`first-writes/` 收到的写入和每组 `forecast.json` 表示中间步骤，完整延迟回答以该组 `pair.json` 为准。比较同一批问题上的 `first_f1` 和 `last_f1` 各自相对原生的差；跨历史区间以对话为单位。四路径诊断已说明在最初 91 道训练题上，单靠在两种前置记忆之间重新分配末段用技概率，最多提高 0.00421。独立开发和完整测试仍待候选达到原晋级标准。[251 服务和恢复命令](server-251.md)

已实测 conv-26 第 2 个 session 和 conv-30 第 1 个 session 的早段写入。活动技能换成时间技能后，成功动作数分别从 19 到 28、从 22 到 35。可从 `first-writes/conv-26_s02/` 追到两次提示与新模型响应。判断是否已经测到回答，请看对应终点目录下的 `pair.json`。有些问题共用相同的最早决策而最后证据位于不同 session。

### 从读取位置回查信用

在 248 结果镜像的 `logs/source-compression/reader-slot-credit/` 中，先看 `verified-cases.json` 里的七次已核验信用及其双向交换结果。conv-42 第 90 题的两次新读者调用已经完成。全体 179 道不同训练题对应 452 次比较，`cohort.json` 可追到每个原始有序提示、实际响应和训练对照。查看 `aliases/later/` 和 `complete/later/` 比较选技概率与 F1；后者需要 291 个互异的交换提示全部完成。每份报告都区分全部窗口和冻结后才开始出分的窗口，`long-prefix/` 单列 185 题面板。独立基准的大规模结论仍待实际测试。[算法记录](algorithm-notes.md) · [251 服务和恢复命令](server-251.md)

第一阶段已回算的后期 31 题 F1 为 0.583784，原生为 0.583735；跨 session 的 21 题为 0.231985，原生为 0.232003。前者的 `possible_completed_f1` 上界为 0.583957，补齐混合回答也无法在这批题上达到 0.02 的增益。接手时优先看 `single-rank/rank-*.json` 中“拿掉会丢失正确答案”和“加入可恢复答案”的实测记录，以更具体的训练记忆文本探索选技排序。完整 378 题和独立测试的要求照原协议执行。

`single-rank/rank-00.json` 已核对 conv-42 第 90 题的第一个名次。替走后 F1 从 1 到 0，反方向加入后仍为 0.2。选用未来的训练示例时保留背景方向，因为在一个分支里必要的文本，在另一分支中未必足够。

### 251 的文本匹配选技与四路径试验

当前模型计算位于 251；248 的用户服务 `memskill-248-pull-linked-eight` 定期收回逐题结果。新策略按目标对话留一，用训练时实际影响回答的读入文本匹配当前发言。已冻结的组级分支覆盖 128 个 session。五个后期窗口已收回 43 题，组级策略期望 F1 为 0.52568，原生为 0.51893。首次严格前瞻的 12 题上分别为 0.36041 和 0.35151，仅用回报为 0.36415，旋转标签为 0.40387。完整单名次干预和 378 题四路径面板尚在收集。本阶段读数仅用于训练诊断。[算法和分母](algorithm-notes.md)

在 `source-compression/witness-passages/` 下，`group/forecast.json` 和 `witness/forecast.json` 分别保存两个阶段的预测；后者只在全部名次完成后生成。`group/later/` 是原随机两臂下的局部选技估计，`four-paths/cells/` 是另一套限载后端实测的四条完整路径，`four-paths/{group,witness}/` 是其整段策略估计。交接时保留原时间戳和各自的响应目录，恢复时对照注册文件；读取 `strict.questions` 可区分已有分数和预测冻结后的题数。[251 服务与恢复](server-251.md)

### 按供样证据和案例可靠性选择技能

251 新增了证据发言匹配、案例权重和长前缀核算。原版记忆匹配在 43 道后期题的期望 F1 为 0.52568，附加证据发言匹配得 0.51783，再按其他训练对话的实测收益衰减案例权重得 0.52263。完整前缀已分别冻结全部 99 个端点的预测，代表 185 道与局部面板不重复的问题。首个 10 题四路径复测窗口已完成，其原随机两臂的 20 个回答跨后端一致。仍在累积数据，尚未进入正式测试。[结果、对照与具体误判](algorithm-notes.md)

在 `source-compression/witness-passages/evidence-advantage/` 下，`group/` 存放全部合格名次的结果，`witness/` 等待单名次干预完成，`case-reliability/{group,witness}/` 存放按案例加权的独立预测。各阶段的 `early`、`donor`、`replica` 给出实测四路径的整段期望收益，`calibration`、`later` 给出随机焦点收益，`long-prefix` 给出完整前缀最后一次选技的收益。阅读时同时查看 `strict.questions`、`all.questions` 和预测的 `exposure.json`。[运行入口](server-251.md)

### 自然问答监督的交接位置

当前新增的自然问答扩充实验在 251 的物理卡 3、5、6 上运行。248 的镜像位于 `../runs/span_credit/251-bf16/linked-eight-gpu/logs/source-compression/exchange-credit/`。先看 `registration.json` 的 234 个已登记来源，再看 `questions/` 的拒绝原因与 `outcomes/` 的真实选技和修补对照。允许单行引言的分析存于 `framed/`，与根目录共用教师响应。两套策略按阶段冻结，只有完整后期诊断过线才进入独立开发集，之后再运行锁定的两套正式测试。[251 的恢复和查数命令](server-251.md#训练对话内的问答与选技对照) · [算法和首轮诊断](algorithm-notes.md#用自然问答补充-span-选技监督)

### 写入前的问题向量预测和较弱分支复核

在同一个 248 镜像根目录下，`source-compression/query-displacement/full/` 保存两个完整冻结预测、各动作概率、实测 F1 与源问题排名核验；`source-compression/legacy-terminal-symmetry/` 保存前四个 session 的独立同技能复核。看后期成绩时，使用同一批窗口的 `through-*.json` 对齐题数，再比较 `raw`、`native` 和 `credit`。`strict` 为策略冻结后才首次出分的窗口。每个窗口的事后最佳得分是机会界，不能作为策略所得 F1。[251 启动和数据位置](server-251.md#问题向量迁移与早期标签复核)

连续权重的冻结预测和逐题结果在 `source-compression/query-displacement/support-mass/full/`。第一批后期 61 题中，活动技能概率平均绝对变化 23.07 个百分点，期望 F1 从仅用回报的 0.55108 降至 0.54200，暂未达到独立开发条件。较弱动作复核的第一份回执为 `legacy-terminal-symmetry/outcomes/conv-26_s1_p1_q85.json`。五条末段候选全部完成后，`summary.json` 将列出每条的修补方向、竞争解释、最终信用以及按对话的分母。[训练诊断和机会界](algorithm-notes.md#用其他对话的问题预测选技后的证据排名)

### 同一对话内的 span 收益排序

251 的五条旧末段标签已全部复核，三条保留，两条有同技能修补的竞争解释。新结果在 248 镜像的 `logs/source-compression/matched-span-rank/`。查看 `consensus/original/later/through-08.json` 的同一批 74 题，核验信用投票 F1 为 0.529637，同种投票只用回报为 0.528751，原生为 0.515726，错配信用为 0.530109。信用相对同种回报投票的增益区间仍跨零。完整后期 378 题还在收集，正式测试继续留出。两条早期首段的后续轨迹需要在较弱技能上重放，当前位置为 `logs/source-compression/legacy-root-symmetry/`。[逐版本配对数据](algorithm-notes.md#同一对话内的-span-收益排序) · [251 恢复命令](server-251.md#同对话配对排序残差校准与删除供样对话投票)

`legacy-root-symmetry/writers-complete.json` 已收回，两条首段替换均重现原解析动作。固定后续技能的修补与读者问答仍待测量。三技能可达收益在 `logs/late-pairs/semantic-writer/three-writer/capacity/through-09.json`，其中 6 个匹配窗口有 37 题，3 个执行异常的窗口保留原因。[三技能机会界及分母](algorithm-notes.md#同一对话内的-span-收益排序)

### 在 251 接续首段重放与关系技能实验

248 的镜像根目录为 `../runs/span_credit/251-bf16/linked-eight-gpu/logs/source-compression/`。在 `legacy-root-symmetry/` 中，两条候选的后续写入已复现，四份原技能或替换技能的最终记忆库及问答检索顺序均与归档一致。两条较弱首段补入缺失证据之后，最终读者看到的前五条记忆均与各自的较弱原路径一致。另两种修补改变了上下文，逐条判定见 `outcomes/*.json` 中真实回答和 `summary.json` 为准。

关系技能的原时间前缀面板共 12 个训练窗口、91 题，其中 10 个写入匹配窗口覆盖 74 题，40 题的证据位于干预段。仅 conv-41 第 7–8 场的原题 80 满足局部完整参考答案恢复，时间技能 F1 为 1，关系技能为 0.75。`relation-skill-repairs/0/outcomes/conv-41_s7.json` 记录修补后的归因。活动前缀复测的相同题目继续使用原题号，在 `semantic-writer/activity-prefix/contrasts/` 查看两前缀同时匹配的实际分母。

`matched-span-rank/root-augmented/` 保存纳入较早首段后的策略。读表时用 `credit` 对照 `zero_roots` 衡量新增标签的作用，再与先前版本 `expanded` 对照。完整发布条件涉及后期 378 题、四路径复核和独立于这些问题的长前缀 185 题。正式测试继续留出。[251 恢复命令与各目录说明](server-251.md#在-251-重放首段写入和两个前缀的关系技能) · [实验假设与配对方式](algorithm-notes.md#首段写入如何影响下一次选技)

首条已收回的最终判定为 `legacy-root-symmetry/outcomes/conv-41_s1_p0_q67.json`。原技能、替换技能的 F1 分别为 1.00、0.50，三种同技能修补均为 0.50，该题首段信用确认为 +0.50。conv-42 第 103 题在较弱技能下补检后恢复了参考短语，F1 从 0 升至 0.40，首段选择信用为零。两条根节点裁决及时间前缀关系技能第 80 题的 −0.25 信用均已形成最终回执。

首版加入首段信用的策略在 `matched-span-rank/root-augmented/later/through-09.json` 已有 9 窗 81 题，5 个实际干预位置改变了选技概率。其配对期望 F1 为 0.511809，同样本置零为 0.513554，上版为 0.515111，原生为 0.512328。本轮未晋级。另有 7 窗 61 题的四路径回执。首个关系技能双前缀窗口已完成，同一批 9 题分别得到 +0.02183、+0.02924 的关系减时间差，首选技能方向相同。两项任务均在 251 按登记窗口继续。

### 独立供样确认和三技能对照

最新代码以 251 为计算机，248 镜像在 `../runs/span_credit/251-bf16/linked-eight-gpu/logs/`。双技能局部规则的 81 题回执在 `source-compression/matched-span-rank/selective-rules/later/through-09.json`。原生 0.51233，局部回报规则与加入核验信用均为 0.51990，这 9 个实际干预位置的信用修正没有触发。三技能回执在 `late-pairs/semantic-writer/three-writer/margin/through-09.json`；后期仅 13 题，核验信用比原生高 0.02585，比相同原始回报模型低 0.00238。当前都未晋级正式测试。

早期 123 题补测与活动/关系直接对照在 251 后台继续。交接时查看 `source-compression/earlier-terminal-returns/outcomes/`、`source-compression/activity-relation-repairs/0/outcomes-complete.json` 以及 `late-pairs/semantic-writer/three-writer/direct-allocation/time/forecast.json`。最后一项需等时间前缀 12 窗都完成较弱技能复核；另一个双前缀版本还要等第二批 12 窗。原题在不同技能对中可能重复，不能合计为独立测试题。[本轮结果和分母](algorithm-notes.md#在独立供样对话上确认选技规则) · [251 恢复命令](server-251.md#双技能局部规则早期回报补测与三技能概率分配)

后期最新的双技能数据为 10 窗 92 题，原生 0.52409，核验信用与基础回报规则同为 0.53102；其中冻结后首次出分的 11 题已单列。分别查看 `selective-rules/later/through-10.json` 的 `all` 和 `strict`，以及 `selective-rules/opportunity/later-10.json` 的逐窗机会界。三技能直接对照已冻结 25 个窗口的原始回报预测，13 道后期题的初始分数为 0.55359，文件在 `direct-allocation/raw-reference/through-09.json`。三个直接复核候选仍按原题号追踪。[逐项指标及解释](algorithm-notes.md#局部规则在后期窗口还有多少改进空间)

首条活动/关系直接信用已收回。conv-26 的原题 20 在活动写入的三种修补下 F1 保持 0，关系写入为 0.85714，逐项回执在 `source-compression/activity-relation-repairs/0/outcomes/conv-26_s5.json`。此前缺少末段替换的 conv-41 四题也已全部复现，活动减时间的窗均 F1 差为 +0.07857。四题证据均位于前一场，详见 `source-compression/earlier-terminal-returns/outcomes/conv-41_s3.json`。

### 三技能直接干预的当前交接

251 时间前缀下的活动／关系三题已经全部裁决。原题 20、110、80 的关系减活动核验信用分别为 +0.85714、−1.00、−0.25。248 镜像下的 `late-pairs/semantic-writer/three-writer/direct-allocation/time/through-10.json` 记录完整模型。后期已测 3 窗、24 题，核验信用 0.577821，原始回报 0.577281，原生 0.564505；两段对话都已在冻结前产生物理答案。原登记后期共 18 窗、161 题，当前只有两类题有返回值。

`skill-graph/complete/` 按同一道题合并多条技能边。`skill-graph/attribution/verified-complete.json` 已核对 25 个决策的单题贡献，`through-10-111.json` 给出整队列的可达范围。`direct-step/` 记录只在内层供样对话选出的放大步长。扩大供样的原题 85 在 `skill-graph/extra-donors/` 中逐项记录，后续策略在 `skill-graph/expanded/`。完整正式测试尚待候选通过预先登记的验证条件。[本轮度量与候选出处](algorithm-notes.md#同一道题上合并三种技能的核验信用) · [251 恢复命令](server-251.md#三种技能在同一道题上的信用)

新增供样的原题 85 已裁决。保留关系技能时三种修补均把 F1 从 0.19355 恢复到 1，故额外选技信用为零。`skill-graph/expanded/through-10.json` 中 24 题的 F1 为 0.573458，等于 `zero_new`。在 `direct-step/decisions/evaluation-10.json` 中，核验信用与原始回报在同一组随机数下选到相同技能，单次回放 F1 都是 0.566282；同目录每窗文件保留实际技能顺序、版本、读写 token 和各策略的换技概率。早期两技能全窗口补测另有两处归档检索顺序不符，排除信息保存在 `source-compression/earlier-terminal-returns/prepared/`。conv-26 原题 89 的控制重放与原生轨迹读到的相同前五项次序不同，conv-30 原题 7 发生在替换分支。

### 原始状态三技能实验及新策略

模型计算继续在 251 进行，248 的结果镜像为 `../runs/span_credit/251-bf16/linked-eight-gpu/logs/`。先看 `source-compression/earliest-three-skills/` 的 `forecast.json`、`pairs/`、`judged/` 和最新的 `evaluation/snapshot-*.json`。早期队列共 123 道不同原题，有 91 题通过旧分支的检索重放。直接信用在共同抽样下使 conv-43 第 3–4 场由活动技能换为关系技能。首个完成实测的窗口是 conv-41 第 3–4 场，其 4 题全部涉及前一场，原题答案及三技能分数已对齐。

新状态策略的后期结果在 `late-pairs/semantic-writer/three-writer/checkpoint-odds/through-10.json`。已有的 24 道后期题中，期望 F1 为 0.578094，同设计原始回报为 0.577469；在共同抽样中两者选技相同。独立测试仍留出。等待早期所有窗口完成后，`earliest-credit-update/` 会将新增的经核验信用用于下一次拟合，并与保留新增原始回报、清零新增信用的版本对照。[算法及逐项指标](algorithm-notes.md#用写入前的原生技能偏好调节-span-信用) · [251 恢复命令](server-251.md#原始轨迹的第三种技能与状态条件概率)

后期队列又收回 conv-43 的 12 题。最新的 `checkpoint-odds/through-11.json` 含 4 个匹配评估窗口、36 题，状态信用 0.553433，同设计原始回报 0.553411，关闭状态特征 0.553919。对话配对区间跨零，未晋级。查 `questions/evaluation-11.json` 可分题型查看，查 `decisions/evaluation-11.json` 可查看 36 题上的共同抽样 F1 0.552774 和读写成本。251 的完整自动化测试共 276 项，全部通过。

### 在 251 接续写入指令和分组信用的实验

本轮权重拟合、检索和问答运行在 251，248 收回回执。三道训练题的四条反事实路径均已复核。第 20 题的 +0.85714 全来自逐条分组，第 80 题的 −0.25 全来自写入指令，第 110 题的 −1 分为指令 −0.81579 和分组 −0.18421。后期 4 窗 36 题上，分项信用改变了四窗概率；相对同设计清零，期望 F1 从 0.553839 变为 0.553770，固定抽样换技 0 次。当前版本保留为开发对照。后期登记分母为 161 题，正式测试仍留出。

接手后先看 248 的 `../runs/span_credit/251-bf16/linked-eight-gpu/logs/source-compression/writer-format-credit/`。三条归因在 `questions/`，整批 74 题进度在 `readers/` 与 `identification/`。策略的 `verified`、`transfer`、`dense` 结果分别对应真实信用、增加旧分组观测、增加当前交叉回报。在同一后期窗口比较各自的 `credit` 和 `raw`，再核对换技数、冻结前暴露和匹配分母。[逐题分解与当前评估](algorithm-notes.md#区分写入指令与原文分组对-span-信用的贡献) · [251 恢复命令](server-251.md#交叉复测写入指令和原文分组)
