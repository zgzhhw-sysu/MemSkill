# 251 上的配对实验

CPU 编码的冻结信用组另做 LongMemEval-S 独立验证。服务 `memskill-251-transfer-validation` 使用原 40% 占空比的生成服务，实时进度在 `~/memskill-work/runs/paired-251-external-validation/client.log`，历史断点在 `~/memskill-work/study-anchor/longmemeval-validation/`。正式候选选定并运行时，这组补充验证让出生成资源。17 条验证历史均按完整会话执行，首次运行前记录权重、协议、验证决定及样本清单的 SHA-256。

研究工作区是 `~/memskill-work/MemSkill`，实验断点在 `~/memskill-work/study`，迁入的原始快照在 `~/span-credit-20261120`。独立环境 `~/memskill-work/venv` 使用 Python 3.12.12、PyTorch 2.10.0+cu128、Transformers 4.53.3 和 sentence-transformers 5.6.0。248 实际使用的配对实验依赖及其传递依赖共 109 个，准确版本在目标机 `~/memskill-work/deploy/requirements-248.txt`。研究数据的口径见[配对协议](span-credit-paired.md)。

## GPU 占用和释放

251 的 0–7 号卡均已获准使用。空闲守护通过 CUDA 驱动分配显存，工作进程随后休眠。当前配置每卡申请 512 MiB，另有 CUDA 上下文开销。新建占用时额外预留 512 MiB 上下文余量，设备至少保留 8192 MiB 空闲显存。余量不足时跳过，已运行的占用在下一轮检测到低于预留值时退出。此设置允许已有作业共存。它不提供设备独占。

下列命令在 251 执行。守护接受物理编号，以 UUID 限制子进程可以看到的卡。任意实验租约生效期间，所有占用 CUDA 进程都会退出。监督进程保留在 CPU，以便最后一个租约释放后自动恢复占用。

```bash
guard="$HOME/memskill-work/deploy/gpu_guard.py"
python3 "$guard" status
# 服务未启动、需要按本机配置启动时使用
python3 "$guard" start --shared --gpus 0,1,2,3,4,5,6,7 --hold-mib 512 --reserve-mib 8192
# 通过租约启动单卡任务；替换命令末尾的程序
python3 "$guard" run --gpus 4 -- nvidia-smi -L
```

守护程序的版本化源码在 [gpu_guard.py](../../scripts/resources/gpu_guard.py)。`--shared` 仅影响显式启用该选项的服务；共享模式按实际剩余显存决定是否创建或释放本项目的占用进程。

## 当前运行

生成服务使用物理 7 号卡，每次处理一个请求，预填充每 16 token 休息一次，每个解码步也同步休息，工作时间比例为 0.40。启动时 PyTorch 显存上限取物理显存的 76% 和当前可用显存减 2048 MiB 两者中的较小值。训练编码使用物理 4 号卡，与生成进程分开。整卡利用率含同卡其他进程，按实际监控记录解释。

```bash
systemctl --user status memskill-251-iteration.service --no-pager
tail -n 15 ~/memskill-work/runs/paired-251/train.log
cd ~/memskill-work/MemSkill
python3 -m experiments.controller.resume_status ~/memskill-work/study
```

运行日志和每三秒的显卡采样在 `~/memskill-work/runs/paired-251/`。模型服务随实验租约退出。每个完整训练窗口、每条验证／测试对话以及每段迁移测试历史分别落盘，换机或重启后可从已落盘的完整记录续跑。验证阶段检查同状态选技概率、实际选技、读到的记忆和逐题 F1；最终留出集沿用已锁定的完整历史和计分规则。

训练可用 5、6 号卡上的第二个生成服务分担反事实回放。这个副本按层均分同一份 BF16 权重，端口为 18753，设置 `--devices 2 --duty .28 --reserve-mib 4608`。每个分块和解码步都会等两张卡完成计算后休息。在 251 上用主服务实际跑过的训练对话 `conv-43` 核对了 2 次写入与 8 次问答，两端回复文字和输入、输出 token 数逐条一致。比对在 `runs/paired-251/aux-equivalence.json`，运行目录以 `~/memskill-work` 为起点。反向训练进程取得 3 号编码卡的租约后按相反顺序领取窗口；新采集窗口的 `backend.json` 记录推理后端。正式基准的各组评估都沿用主服务。

验证包含两条互相独立的对话，分别有 30 和 45 道可评分题。251 在实验目录创建 `require-selective-validation` 后，统一启动脚本会在读取最终测试题前复核五组验证结果，并将决定写入 `validation/selection.json`。若第一轮发展出的策略未能在验证时胜过整体概率对照等基线，后续任务 `memskill-251-followup` 可从已完成的十二个窗口开始，在 `~/memskill-work/study-eight` 采集额外十二个训练窗口。第二轮日志在 `~/memskill-work/runs/paired-251-eight/`。

第一轮已完成。扩充训练窗口的任务使用 `memskill-251-followup.service`。另一个独立任务 `memskill-251-aligned.service` 在 3 号卡编码，5、6 号卡以 28% 工作时间比例生成，重新采集训练和反事实写入提示都与部署相同的十二个窗口。对应的断点和日志位于 `~/memskill-work/study-aligned-writer/` 和 `~/memskill-work/runs/paired-251-aligned/`。用 `systemctl --user status memskill-251-followup memskill-251-aligned` 可查看两组当前进程。

`memskill-251-anchor.service` 从第一轮十二个经过校验的训练窗口重新拟合策略，断点在 `~/memskill-work/study-anchor/`。五个验证组都用 CPU 编码，以及同一个 18752 端口的限载 BF16 生成服务。首次发生选技分歧后，各组按实际写入独立运行。查看 `~/memskill-work/runs/paired-251-anchor/validation-credit.log` 可跟踪信用组，五组完成后的逐题核对和选择决定在 `study-anchor/validation/selection.json`。

在 `conv-47` 上，CPU 编码的信用组前三次实际选择活动细节技能，第四次切换到时间上下文技能。原 GPU 后端在已保存的固定策略轨迹上预计第三次就会切换。`memskill-251-anchor-gpu.service` 已校验两个基线的原始轨迹、回答和冻结权重，并等待扩充训练释放 4、7 号卡，再用原编码配置复核。该组使用 `~/memskill-work/study-anchor-gpu/` 和 `~/memskill-work/runs/paired-251-anchor-gpu/`，生成服务使用端口 18754，限载设置与 18752 相同。

正式测试的候选顺序在三组验证决定产生之前固定。先考虑扩充到八个 session 的策略，其次是原 GPU 编码环境下的零信用约束策略，最后是训练写入提示与部署一致的策略。选取顺序中第一个通过五组验证标准的候选。`memskill-251-official.service` 在等待期间只使用 CPU。选定后，`~/memskill-work/runs/official-251/selected.json` 记录验证决定及冻结权重的指纹。完整评估沿用该候选的模型、编码和限载配置，按历史补齐六组结果，包括在每条历史保持技能使用次数相同的对照。每个基准的逐题结果和中断恢复入口见[服务器交接文档](server-handoff.md)。

训练归因的补充任务可用以下命令查看。原始训练窗口、各项探查与完整历史评估分别写入各自目录。前三个写入粒度窗口的逐题回答、等来源发言数量复核和第 85 题较差分支的两种修复记录都在 `~/memskill-work/runs/paired-251/` 下。

```bash
systemctl --user status memskill-251-weaker-six memskill-251-omitted-strict memskill-251-omitted-weaker --no-pager
systemctl --user status memskill-251-granularity-placement memskill-251-duration-aligned --no-pager
systemctl --user status memskill-251-transfer-validation memskill-251-official --no-pager
```

`memskill-251-omitted-weaker` 等待遗漏题记录收齐后，用同一限载后端逐条复核较差分支。`memskill-251-duration-aligned` 等待十二个对齐写入窗口全部完成，再用 CPU 重新计算排除数词写法影响后的留一对话结果。

第一轮真实写入的发言分组对照由 `memskill-251-writer-granularity` 在 CPU 建库，再请求 5、6 号卡已有的 28% 限载服务。逐题记录在 `~/memskill-work/runs/paired-251/writer-granularity/`，完整结果写入该目录的 `summary.json`。在 251 复跑已发布版本可执行以下命令。

```bash
source ~/memskill-work/deploy/env-251.sh
export CUDA_VISIBLE_DEVICES="" MEMSKILL_DEVICE=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty28-gpu56-qwen-cpu-fp16
export MEMSKILL_API_BASE=http://127.0.0.1:18753/v1
"$MEMSKILL_PYTHON" -m experiments.controller.writer_granularity "$STUDY_DIR" "$LOCOMO_DATA" "$run_dir/writer-granularity"
```
