# MemSkill 换机交接 · 2026-11-20

251 已接续 248 的训练。两台机器的 Python 均为 3.12.12，配对实验所用的 109 个 Python 依赖逐项版本一致。目标机的代码在 `~/memskill-work/MemSkill`，训练断点和模型路径见 [251 运行说明](server-251.md)。

11 月 20 日 06:03，原 GPU 编码的零信用约束策略完成两个开发前缀的 75 道题。信用组 F1 为 0.28743，终局组为 0.22411，固定使用活动技能和匹配训练使用率两组均为 0.32039。该候选未达到开发门槛。训练与部署写入提示对齐的候选继续原计划。新增的等来源训练单独写入 `~/memskill-work/study-equal-sources`，通过 `memskill-251-equal-sources.service` 运行，当前使用 CPU 编码和已有的 5、6 号卡生成服务。两个实验均按窗口续跑。正式候选的优先顺序和两个基准的测试清单已冻结。

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
