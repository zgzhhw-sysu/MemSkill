# MemSkill 换机交接 · 2026-11-20

接手任务是补完 LoCoMo 的配对训练，冻结 Controller，再检验 span 级信用能否改变实际选技并提高独立测试 F1。当前已保存 **6/12 个训练窗口、12 个 span，其中 4 个有非零信用**。还没有最终策略权重、完整验证分数或测试分数。训练窗口中的局部 F1 差值不能作为测试收益。[实验方法](span-credit-paired.md) · [固定测试清单](../../experiments/controller/protocols/paired_grounding_cohort.json)

## 从哪里接

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

源机使用 Python 3.12.12、PyTorch 2.10.0+cu128、Transformers 4.53.3、sentence-transformers 5.6.0。下例适用于驱动支持 CUDA 12.8 的目标机。其他驱动环境需安装兼容的 PyTorch，并记录实际软硬件版本。独立环境只需安装配对实验依赖。

```bash
python3.12 -m venv ~/memskill-work/venv
source ~/memskill-work/venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
python -m pip install -r scripts/experiments/requirements-paired.txt
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

训练成功后应有 12 个窗口、24 个 span，以及 `terminal.json`、`credit.json` 两份权重。中途退出后，重启服务并重跑相同命令即可。仅发布成功的完整窗口会被复用。服务端口或后端配置发生变化时，核实客户端连接地址和后端标识。

## 验证、冻结与正式测试

先在两段验证对话的前六个 session 比较四组。环境和工作目录沿用上文。

```bash
bash scripts/experiments/paired_grounding.sh validation &&
python -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" validation \
  --arms native terminal credit always
```

读取 `validation/comparison.json`。先比较 `credit_minus.terminal` 的同状态概率变化、贪心切换数、实际三技能组合变化和逐题 F1。再检查 `changed_question_evidence` 中两组读到的提示及记忆来源。与 `always` 相同的轨迹要单列，这种结果不足以证明策略会按 span 选择性地使用该技能。若验证退化，保留该版，用训练和验证对话继续迭代。

选定方案后记录权重和协议指纹，再一次性跑完整留出集。以下命令使用同一组冻结权重。

```bash
(
  set -e
  cd "$STUDY_DIR"
  sha256sum protocol.json terminal.json credit.json > frozen.sha256
)
(
  set -e
  bash scripts/experiments/paired_grounding.sh test
  python -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" test \
    --arms native terminal credit always
  for arm in native terminal credit always; do
    bash scripts/experiments/paired_grounding.sh transfer --study-arm "$arm"
  done
  python -m experiments.controller.audit_paired "$STUDY_DIR" "$LONGMEMEVAL_DATA" longmemeval \
    --arms native terminal credit always
  (cd "$STUDY_DIR" && sha256sum -c frozen.sha256)
)
```

LoCoMo 主测试含两段完整对话、55 个 session、314 道可评分题。预设目标为相对 `terminal` 和 `native` 的逐题平均 F1 均至少提高 **0.02**，且两段对话分别为正。另有 86 道无普通参考答案的对抗题按原协议单列。LongMemEval-S 迁移测试含 105 段完整历史、5,027 个 session，覆盖六类问题；101 道可回答题用于主 F1，4 道弃答题另报。报告逐类结果和按历史配对、按题型分层的 95% 区间。两个数据集分别报告，保留逐题回退和无法由选技解释的变化。

最终报告在 `test/comparison.json` 和 `longmemeval/comparison.json`。每个已完成的评估历史保存实际问题编号、分数、决策、轨迹及用量；完整历史间可续跑。实验结束后在启动服务的终端执行 `kill "$MODEL_PID"; wait "$MODEL_PID"`，并按目标机约定释放租约或恢复限量空闲守护。
