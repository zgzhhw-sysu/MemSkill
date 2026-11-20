# 251 上的配对实验

## 11 月 21 日的两个训练实验

`memskill-251-factorial-repairs` 用物理 GPU 2 重放四路径窗口的末段。它复用历史写入动作，核对两条原始读者上下文，再分别补入遗漏原文或替换一条检索结果。每个已完成窗口对应 `source-compression/factorial-repairs/conv-*_s*.json`。同目录的 `usage.jsonl` 记载真实调用和缓存命中。

`memskill-251-semantic-writer` 用物理 GPU 0 编码，在相同原文下比较关系与时间技能。24 个早期训练 session 用于校准来源相似度。12 个目标 span 的概率修正保存在 `semantic-writer/source-forecast.json`。各窗口的 `choice.json` 在请求新写入之前保存，`write.json` 保存实际动作及记忆，`pair.json` 保存匹配状态和原始问题结果。首个窗口结束后等原四路径面板齐备再继续。这两个实验与四路径任务共用物理 GPU 7 上目标占空比 22% 的 Llama 3 8B BF16 服务。编码占空比为 30%，每批最多两条。

平衡关系技能先验的追加计分由 `memskill-251-audit-semantic-writer` 完成。它用 GPU 0 冻结全部 12 个概率，此后按已有逐题答案汇总。恢复入口为 `scripts/experiments/audit_semantic_writer_251.sh`。

`memskill-251-outcome-credit` 在补检窗口和随机配对窗口发布后，用 GPU 0 重算动作优势策略。参数、冻结时已出分的路径和逐题结果保存在 `source-compression/outcome-credit/pilot-*/`，12 个补检窗口齐备后另存 `main/`。恢复入口为 `scripts/experiments/audit_outcome_credit_251.sh`。

在 251 查看或恢复服务时运行以下命令。已运行的服务可直接查看其日志。

```bash
source ~/memskill-work/deploy/env-251.sh
cd "$work/run-linked-questions-v1"
systemctl --user status memskill-251-factorial-repairs memskill-251-semantic-writer --no-pager
base="$work/runs/paired-251-linked-eight/source-compression"
tail -n 8 "$base/factorial-repairs/client.log" "$base/semantic-writer/client.log"
python3 "$guard" status
# 对已退出的服务，分别恢复对应的实验
systemd-run --user --collect --unit=memskill-251-factorial-repairs bash "$PWD/scripts/experiments/run_factorial_repairs_251.sh"
systemd-run --user --collect --unit=memskill-251-semantic-writer bash "$PWD/scripts/experiments/run_semantic_writer_251.sh"
# 复算当前完整窗口在固定末段用技率下的 F1 上界
python3 "$guard" run --gpus 0 -- "$MEMSKILL_PYTHON" -m experiments.controller.terminal_capacity "$base"
```

## 11 月 20 日晚的运行

`memskill-251-linked-eight.service` 使用 GPU 3、4 执行两个训练进程，逐个从训练窗口的两端取任务。GPU 5、6 共同承载一个 Llama 3 8B BF16 服务，端口为 18773，逐卡分配约 7658 MiB。编码器每批最多两条，目标占空比为 35%。生成服务逐 16 个输入 token 预填充并逐 token 解码，目标占空比为 28%。调度脚本位于 `~/memskill-work/deploy/run-linked-251.sh`。

```bash
systemctl --user status memskill-251-linked-eight
cd ~/memskill-work
tail -n 5 runs/paired-251-linked-eight/{writer,train-3,train-4}.log
find study-linked-eight-gpu/train -maxdepth 2 -name rows.json | wc -l
python3 deploy/gpu_guard.py status
```

训练集有 24 个两段 session 的窗口。每个窗口完整写入后发布 `rows.json`。两路训练退出后，同一服务评估开发集及各对照，开发筛选通过才运行两套正式测试。GPU 租约关闭后守护按显存余量恢复有限占用。

独立的 `memskill-251-source-compression` 服务读取 22:06 冻结的选技概率，等第 5 至 8 个 session 的 12 个窗口全部完成后，使用 GPU 2 对照逐 span 的换技结果。`memskill-251-training-factorial` 在 GPU 1 上以单个 Qwen 模型、每批两条编码，GPU 7 的 Llama BF16 以 22% 目标占空比分别对四种选技顺序建库并回答问题。该单卡服务端口为 18774，结果保存在 `~/memskill-work/runs/paired-251-linked-eight/source-compression/factorial/`。248 的回收服务每五分钟同步完整窗口及反馈，等主训练、来源概率核对、四路径、条件选技、问题原型、写入分解、后续配对、配对复核、条件概率校准及跨窗口全前缀干预十项服务全部退出后再做最后一次同步。

第 9 个 session 以后的 52 个已登记窗口位于 `~/memskill-work/runs/paired-251-linked-eight/late-pairs/`。`memskill-251-late-pairs` 先在原 18774 串行服务上运行第一对完整干预。原 12 个四路径窗口收齐并归还 7 号卡后，它按相同模型、精度和 22% 目标占空比继续剩余窗口。真实轨迹由 2 号卡编码，每批最多两条，保留 2048 MiB 显存余量；窗口间交还编码租约。`memskill-251-audit-late-pairs` 在 0 号卡上复核实际记忆选择、概率、逐题 F1 和抽样权重。原方案与重采样方案分别在第一条新路径出分前冻结。调度脚本和复核脚本随代码发布。[配对调度](../../scripts/experiments/run_late_pairs_251.sh) · [逐窗口复核](../../scripts/experiments/audit_late_pairs_251.sh)

```bash
systemctl --user status memskill-251-late-pairs memskill-251-audit-late-pairs
cd ~/memskill-work/runs/paired-251-linked-eight/late-pairs
find outcomes -name pair.json | wc -l
ls step-mixture/randomized-*.json
```

用户服务 `memskill-251-prefix-return` 读取同一批真实轨迹，在 GPU 0 的短租约上拟合两种选技规则。结果分别写入 `late-pairs/prefix-return/` 和 `late-pairs/rank-return/`。每个目录的 `pilot/` 保存两条对话拟合的参数、冻结时间及当时已出分的文件。四路径收满 12 个窗口后，服务自动生成 `main/forecast.json`，含六套留一对话参数及一套全训练参数。查看 `progress-*.json` 的 `four_path_prospective` 和 `paired_prospective` 可区分数值冻结后的窗口。248 的回收服务将结果随主作业一同回收。[运行入口](../../scripts/experiments/run_prefix_return_251.sh)

为尽早取得冻结后第一条先导留出对话的配对结果，一次性任务 `memskill-251-prefix-early-pairs` 在同一串行 18774 服务上，按既定次序补跑第 2 和第 3 个窗口。它只申请 GPU 2 的编码租约，生成端沿用 22% 目标占空比。三窗合计 18 题。主六对话交叉拟合在后面冻结时会将已经出分的前缀排除出严格前瞻估计，现有物理分配与提问顺序均可从 `late-pairs/cohort.json` 复核。

跨窗口干预服务 `memskill-251-cross-prefix` 使用同一 GPU 2 编码器配置和 GPU 7 生成服务。它在启动前核对监听进程、模型路径、物理卡和 22% 目标占空比。首先采集一个跨窗口位置，然后给原有 12 个四路径窗口优先权。后续每个完整前缀单独申请编码租约，原配对任务结束后，如仍有前缀待做，服务以相同配置接续生成。位置和题号在 `late-pairs/cross-window-prefix/manifest.json` 登记；`through-NNN.json` 给出已经完成的配对，`held_histories` 只包含未参加两个先导拟合的对话。六套主策略就绪后，条件概率复核服务在同一目录的 `main-crossfit/` 记录严格前瞻的逐题配对。[运行脚本](../../scripts/experiments/run_cross_prefix_251.sh) · [跨窗口协议](../../experiments/controller/protocols/cross_window_prefix.json)

## 前序实验记录

自 11 月 20 日 10:13 起，模型实验统一走 CUDA。等来源训练由 GPU 3、4 编码和检索，GPU 5、6 做写入与问答。训练、开发及通过筛选后的完整测试均使用这套后端。配对启动脚本默认选择 CUDA，有限显存时在 GPU 租约处等待。CPU 负责 I/O、分词、调度和 F1 统计。

旧 CPU 编码正式复核和两路外部开发作业已经停止，已有断点保留。等来源 CPU 训练保存了四个完整窗口，工作进程已暂停；其服务内的 18759 端口仍向 GPU 训练提供 5、6 号卡上的 Llama 生成。GPU 作业退出后一起释放该共享服务。查看当前工作使用 `systemctl --user status memskill-251-equal-gpu` 和 `python3 ~/memskill-work/deploy/gpu_guard.py status`。

此前，CPU 编码的冻结信用组曾在 LongMemEval-S 独立开发集上运行。服务 `memskill-251-transfer-validation` 使用原 40% 占空比的生成服务，日志在 `~/memskill-work/runs/paired-251-external-validation/client.log`，历史断点在 `~/memskill-work/study-anchor/longmemeval-validation/`。原计划含 17 条完整历史，首次运行前已记录权重、协议、验证决定及样本清单的 SHA-256。

研究工作区是 `~/memskill-work/MemSkill`，实验断点在 `~/memskill-work/study`，迁入的原始快照在 `~/span-credit-20261120`。独立环境 `~/memskill-work/venv` 使用 Python 3.12.12、PyTorch 2.10.0+cu128、Transformers 4.53.3 和 sentence-transformers 5.6.0。248 实际使用的配对实验依赖及其传递依赖共 109 个，已核对的版本在 [109 项环境清单](../../scripts/experiments/requirements-paired-cu128.txt)，目标机副本保存在 `~/memskill-work/deploy/requirements-248.txt`。研究数据的口径见[配对协议](span-credit-paired.md)。

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

扩充训练结束后由五组开发验证决定是否交接 GPU 4、7。通过时只由 `memskill-251-official` 执行六组正式测试，并优先于 `memskill-251-anchor-gpu` 使用这两张卡。旧版扩充服务在同一进程里排了四组迁移评估。本轮运行中的一次性交接服务 `memskill-251-expanded-handoff` 会核对原训练任务的 systemd 调用标识、75 道开发题的选择结果、正式服务选中的实验目录和四个冻结文件的 SHA-256，再结束该旧任务以释放租约。验证未通过时，扩充任务自行退出，原编码环境的候选接续验证。再次启动时，部署目录中的 `run-expanded.sh` 已在开发验证和冻结后结束。

本轮扩充版五组开发比较已完成，选择文件记录 `advance=false`。原编码环境的零信用约束组已启用 18754 端口。两个开发目录中的原始组和固定技能组输入相同，已向后一组接续提供 85 条具有相同后端、请求内容与 token 记录的开发缓存。外部验证在第一条历史完成 47 次写入后遇到端口交接，已在 18754 端口重启该历史，可从逐请求缓存重用已完成的生成。

在 `study-eight/diagnostics/weaker-branch/<backend-hash>/`，常用的逐 span 结果及 `through-8.json` 会在逐题复核完成后更新。新版收据带有 `reader_match_version=2`，副本位于 `per-question/`。与早期判断不同的共享写入或读取情况可用 `all-readers-v1/` 下归档的旧收据对照。两道训练题对原始检索、替换末位及调到首位的重复读取在 `runs/paired-251/opposite-credit-retrieval/` 和 `opposite-credit-rank/`。

整窗口训练审计的结果在 `~/memskill-work/runs/paired-251/forward-stochastic.json`。同一末段两种真实写入的逐题复测保存在相邻的 `forward-last-switches/`，早期快照含 `progress`，完整快照含 `summary.json`。复测由 CPU 执行 Qwen 检索，调用 5、6 号卡上已有的 28% 占空比 Llama 生成服务。

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
### 等来源选技的训练与开发验证

`memskill-251-equal-sources.service` 在独立工作目录运行 12 个真实训练窗口，再验证两段开发前缀。编码在 CPU 上，生成复用 5、6 号卡上占空比 28% 的 BF16 服务。该服务退出后，监督脚本等待已排定的正式任务和对齐写入任务结束，再按相同参数为这两张卡申请租约。新实验的逐窗口轨迹在 `~/memskill-work/study-equal-sources/train/`，日志位于 `~/memskill-work/runs/paired-251-equal-sources/`。目标机的启动脚本为 `~/memskill-work/deploy/run-equal-source-study.sh`。

```bash
source ~/memskill-work/deploy/env-251.sh
systemctl --user status memskill-251-equal-sources --no-pager
"$MEMSKILL_PYTHON" -m experiments.controller.resume_status "$work/study-equal-sources"
tail -c 1200 "$work/runs/paired-251-equal-sources/train.log"
```

迁移时拷贝原实验目录、模型、数据及脚本，并保留首轮的 `protocol.json`。另一台机器准备好同一推理后端后，将 `STUDY_DIR` 设为拷贝的研究目录。通过 `bash scripts/experiments/paired_grounding.sh train --study-equal-sources --study-deployment-writer --study-anchor-abstentions` 续跑。训练断点会复核被冻结的技能及其在两种写入中的位置。开发结果包括固定逐条记的 `plain` 组和固定两条合记的 `always` 组。
### 外部开发集的双进程评估

已经通过 LoCoMo 开发筛选的 CPU 编码策略，在固定的 17 条 LongMemEval-S 开发历史上复核。运行 `MEMSKILL_EXTERNAL_WORKERS=2 bash scripts/experiments/paired_transfer_validation.sh` 可在同一个限载服务前使用两个 CPU 客户端。固定清单按从零开始的序号分配，偶数组含 9 条历史、453 个 session，奇数组含 8 条历史、363 个 session。每个进程都在同一历史上跑齐所有策略，全部结束后统一复核 17 条历史的逐题结果。默认使用一个客户端。日志在 `$STUDY_DIR/longmemeval-validation/worker-even.log` 和 `worker-odd.log`。

251 的原 GPU 编码候选完成 75 题及等次数对照后，信用组为 0.28743，等次数组为 0.29407，固定活动技能组为 0.32039。该候选及优先级更高的扩充候选均已记录 `advance=false`。剩余正式候选只需 3、5、6 号卡，外部验证因此可在 7 号卡上申请独立租约，通过 18757 端口使用 40% 占空比生成服务。此次交接沿用已冻结的协议、权重与逐请求缓存。已启动的主进程按原清单顺序运行，一个补充进程先处理奇数序号的历史，每个完成的策略与历史组合按锁和输入指纹复用。
### 跨编码配置重放开发动作

`memskill-251-encoder-schedule-cross` 依次在两个编码环境执行另一个环境的信用组选技。GPU 方向只在建库和问答期间申请 4 号编码卡，两边的 Llama 请求都经 7 号卡已有的 18757 限载服务。逐向日志保存在 `~/memskill-work/runs/paired-251/encoder-schedule-cross/`。固定检查点技能向量的训练分析只需 CPU。

```bash
source ~/memskill-work/deploy/env-251.sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 "$MEMSKILL_PYTHON" -m experiments.controller.audit_skill_embeddings \
    "$work/study-eight" "$CONTROLLER_CHECKPOINT" --development "$work/study-anchor" "$work/study-anchor-gpu"
```
### 已暂停的冻结 CPU 策略完整测试

09:55 登记的 CPU 编码复核从原六组开发结果出发。运行器在读取测试对话前校验权重、开发报告、后端和两套测试清单，并将登记原文写入 `$STUDY_DIR/holdout/registration.json`。LoCoMo 六组完成后转入 LongMemEval-S，按照完整历史保存各组结果。两段 LoCoMo 对话共享各自的写入和检索历史，逐题差异须与逐对话结果一起解释。

251 的服务为 `memskill-251-cpu-anchor-confirmation`，日志在 `~/memskill-work/runs/cpu-anchor-confirmation/` 和 `~/memskill-work/study-anchor/holdout/`。它复用 7 号卡的 18757 限载服务。该服务完成外部开发验证并退出后，复核程序可申请同一张卡，在 18760 端口以原 40% 占空比续跑。CPU 编码使用六线程。换机时先恢复开发快照和已完成的正式轨迹，准备好匹配的生成服务后运行以下命令。

```bash
source ~/memskill-work/deploy/env-251.sh
export STUDY_DIR="$work/study-anchor" MEMSKILL_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty40-qwen-cpu-fp16
export MEMSKILL_API_BASE=http://127.0.0.1:18757/v1
export MEMSKILL_HOLDOUT_REGISTRATION="$repo/experiments/controller/protocols/cpu_anchor_holdout.json"
bash scripts/experiments/paired_holdout.sh
```
### GPU 3、4 上的等来源选技实验

`memskill-251-equal-gpu` 通过 GPU 租约在 3、4 号卡分别执行正序和逆序训练窗口，Qwen 状态编码、技能编码及检索均使用对应的 GPU，生成请求共用 5、6 号卡的 BF16 服务。该实验以 `~/memskill-work/study-equal-sources-gpu` 保存独立的样本与权重。训练和开发时设置 `MEMSKILL_ENCODER_DUTY=.35` 和 `MEMSKILL_ENCODER_MAX_BATCH=2`，每批 Transformer 前向结束后同步当前设备，并按实际运行耗时休息。5、6 号卡生成服务继续使用 28% 占空比。每张卡的瞬时利用率会随前向和休息交替变化。

启动前在两张卡上用相同的真实训练 span、同一检查点和九种技能分别做了 25% 和 35% 占空比的编码。四次运行的状态、技能向量、检索向量、两种有序动作及活动技能概率均一致。每卡的核对结果在 `~/memskill-work/runs/paired-251-equal-gpu/encoder-{3,4}.json`，工作日志为 `train-3.log` 和 `train-4.log`。原 CPU 策略的复核已单独归档。

### 按已完成训练窗口补探与五条发言预算实验

`memskill-251-live-omitted-gpu` 在 GPU 7 录得的编码、检索、技能向量及选技概率与 3、4 号卡核对一致后，补查末 span 关联、但原四题预算遗漏的全部训练题。使用生成该训练窗口时相同的双卡写入后端和规则 2。每个窗口单独落盘，进行中写 `diagnostics/omitted-final/<后端指纹>/progress.json`，待所有预定窗口都完成后再写 `summary.json`。已有实验可通过 GPU 租约运行 `bash scripts/experiments/paired_grounding.sh probe-omitted --study-probe-completed`。

7 号卡的训练记忆库分组对照已完成，完整回答与对齐结果在 `~/memskill-work/runs/paired-251/writer-granularity-gpu/`。GPU 轮流运行 Qwen 检索和 Llama 问答，两者占空比分别为 30% 和 28%。另一个受五条发言预算约束的技能实验在 `semantic-turn-budget.json` 保存每个训练 span 的两个选择及覆盖率。服务 `memskill-251-budget-readers` 为 123 道训练问题构建真实写入库对照，在 `skill-budget-writer/` 保存选择、检索计划、逐题回答、实测 token 数和分数。2 号卡上的 Qwen 批大小为 2，使用当时可用显存扣除 2048 MiB 后的分配上限。生成使用现有的 5、6 号卡 28% 占空比服务。服务 `memskill-251-budget-policy` 等全部回答完成后在 GPU 上运行留一对话的概率比较，结果写入同目录的 `crossfit.json`。

### 相邻发言索引与末段选技概率

251 上的 `memskill-251-turn-key-screen` 运行 `scripts/experiments/run_turn_keys_251.sh`。GPU 0 上的 Qwen 编码批大小为 2，占空比为 30%，按当时的空闲显存保留至少 2048 MiB。六条完整训练对话都处理完之后，`~/memskill-work/runs/paired-251-linked-eight/source-compression/turn-keys/` 下的 `forecast.json` 保存留一对话拟合与选技修正，`screen.json` 和 `screen-summary.json` 保存分对话及分题型的检索统计。

`memskill-251-turn-key-reader` 运行 `scripts/experiments/run_turn_key_reader_251.sh`。准备任务通过 GPU 1 租约重建真实记忆库，与原始四路径归档使用同一张编码卡。源端概率拟合固定在 GPU 0 的已有结果；回执分别记录两个后端，逐题核对原始检索排名、答案和 token 数。每个窗口的 `reader/<窗口>/jobs.json` 原子落盘后，评分任务即通过已有的 7 号卡 Llama 服务生成变化的提示，服务占空比为 22%。`pair.json` 保存逐题读入内容的指纹、答案、F1 和用量，同目录的 `usage.jsonl` 记录新调用。`reader-through-N.json` 保存按预登记顺序完成的前 N 个窗口，全部完成才产生 `reader-summary.json`。248 的 `memskill-248-pull-linked-eight` 定期收回整个目录。

### 检索竞争与首段写入重放

`memskill-251-retrieval-competition` 使用 GPU 1 的 30% 占空比 Qwen 编码器重建选技前的检索特征，GPU 上拟合留一对话的两阶段策略，按已落盘的随机配对窗口增量计分。结果在 `~/memskill-work/runs/paired-251-linked-eight/source-compression/retrieval-competition/`。`fit.json`、`preexisting.json` 分别保存模型和冻结前已有分数的窗口。`calibration.json` 对应 91 道训练题，`through-NN.json` 中的 `prospective` 给出严格前瞻成绩，`diagnostics-NN.json` 记录分阶段概率、逻辑读写 token 和晋级判据。各组原来的选技顺序与逐题回答可按存档窗口和路径位回查。`capacity.json` 使用已测四路径，给出保持两阶段用技次数的可行事后策略以及放宽次数约束的上界。

`memskill-251-root-write-repairs` 使用原归档的 GPU 1 编码器和 GPU 7 的 22% 占空比生成服务。启动程序为 `scripts/experiments/run_root_write_repairs_251.sh`。结果位于同级的 `root-write-repairs/<窗口>/`，每个 `q<原始题号>-suffix<动作>.json` 保存两条事实重放、补写后重跑的写入轨迹与补检后回答。相同的本地贪心请求复用有校验的原归档；新请求和 token 写到独立响应目录及窗口用量日志。两个任务都按 251 的 GPU 租约运行。

`memskill-251-root-overlap` 等待全部首段修复回执，再在 GPU 上拟合仅改变首段的附加信用。`source-compression/root-overlap/` 单独保存首次规则登记、数值冻结、此前已出过分的窗口、四路径校准和滚动结果。该项目的 52 个后续训练窗口中，27 个窗口的干预点在首段（257 题），25 个在末段（209 题）；末段焦点概率沿用原策略，首段概率也会通过路径到达率影响两段回报估计。

`retrieval-competition/design-variance.json` 保存有完整路径的训练窗口中四种随机分配下的方差核查。`retrieval-competition/return-control/` 保存对照常数、各自的数值冻结时点及后续面板原始 HT 与校正读数。两个冻结时点之前已产生过路径分数的窗口分别记录。

`root-overlap/question-influence.json` 在全部首段核查和主策略冻结之后生成。它为每一道产生首段额外信用的训练题保存独立留出后的 F1 和首段概率改变量，主策略继续使用预登记的全部训练题。

### 后期 25 个窗口的三技能实测

251 在 GPU 1 上限载运行 Qwen 编码，共用 GPU 7 上端口 18774 的 BF16 Llama。编码计算占空比为 0.30，生成占空比为 0.22。两个服务按窗口申请租约。模型计算等待可用 GPU。251 的结果目录为 `~/memskill-work/runs/paired-251-linked-eight/late-pairs/semantic-writer/`，248 在 `../runs/span_credit/251-bf16/linked-eight-gpu/logs/late-pairs/semantic-writer/` 保存副本，路径相对仓库根目录。

```bash
cd ~/memskill-work/run-linked-questions-v1
systemctl --user status memskill-251-late-semantic-writer memskill-251-three-writer
# 服务退出后，若仍有缺失的窗口，可从已保存的回执继续。
systemd-run --user --collect --unit=memskill-251-late-semantic-writer bash "$PWD/scripts/experiments/run_late_semantic_writer_251.sh"
systemd-run --user --collect --unit=memskill-251-three-writer bash "$PWD/scripts/experiments/audit_three_writer_251.sh"
```

`panel.json` 固定问题、首段动作和窗口顺序，`forecast.json` 保存两种关系概率。一个窗口的 `choice.json`、`write.json` 和 `pair.json` 分别代表已选技、已建库和已完成该组问答。三路结果保存在 `three-writer/through-*.json`，等待该窗口原始活动路径及其冻结策略预测一并到齐才发布。两个目录都有最终的 `summary.json`。恢复时还需同一父目录中的 `outcomes/` 和 `../source-compression/retrieval-competition/` 的原始回执。

### 写入格式复核和后续因果训练样本

251 的 `memskill-251-writer-block-repair` 在 GPU 0 重建末段记忆，并共用 GPU 7 的限载读者。结果在 `source-compression/writer-block-repair/`。`census.json` 给出受影响条目及实际检索次数，`continuous-envelope.json` 给出连续信用范围的上界，`credit/` 和 `population/` 分别保存原有信用的复核和全部题目的双路径回执。`memskill-251-writer-block-policy` 在 GPU 1 拟合修正后的策略。

`memskill-251-late-selection-labels` 在原 GPU 2 编码后端重放 52 个后期随机窗口。`late-pairs/selection-labels/split.json` 固定 88 题新增训练块和 378 题后续评价块。`memskill-251-late-credit-kernel` 在 GPU 1 拟合，`pilot/` 和 `expanded/` 分别保留仅有较早轨迹和加入新增训练轨迹的数值冻结及逐题结果。路径均相对于 `~/memskill-work/runs/paired-251-linked-eight/`。248 的同步目录为 `../runs/span_credit/251-bf16/linked-eight-gpu/logs/`。

```bash
cd ~/memskill-work/run-linked-questions-v1
systemctl --user status memskill-251-writer-block-repair memskill-251-writer-block-policy memskill-251-late-selection-labels memskill-251-late-credit-kernel
# 服务已退出且仍有缺失窗口时，恢复对应任务。
systemd-run --user --collect --unit=memskill-251-writer-block-repair bash "$PWD/scripts/experiments/run_writer_block_repair_251.sh"
systemd-run --user --collect --unit=memskill-251-writer-block-policy bash "$PWD/scripts/experiments/audit_writer_block_251.sh"
systemd-run --user --collect --unit=memskill-251-late-selection-labels bash "$PWD/scripts/experiments/run_late_selection_labels_251.sh"
systemd-run --user --collect --unit=memskill-251-late-credit-kernel bash "$PWD/scripts/experiments/audit_late_credit_kernel_251.sh"
```

### 自然回复与写入前选技

251 的 `memskill-251-reply-probes` 在物理 GPU 1 上用 Qwen 编码，批大小为 2，编码占空比上限为 30%。`memskill-251-reply-placement` 在同一卡的独立租约内拟合策略，读取前序受控实验的实际写入和读者得分。两个任务可断点恢复。恢复时在 251 的运行目录执行以下命令。

```bash
cd ~/memskill-work/run-linked-questions-v1
systemd-run --user --collect --unit=memskill-251-reply-probes --property=Restart=on-failure --property=RestartSec=60 bash "$PWD/scripts/experiments/run_reply_probes_251.sh"
systemd-run --user --collect --unit=memskill-251-reply-placement --property=Restart=on-failure --property=RestartSec=45 bash "$PWD/scripts/experiments/audit_reply_placement_251.sh"
```

运行产物位于 `~/memskill-work/runs/paired-251-linked-eight/source-compression/reply-placement/`。每个 session 的 `sources/` 回执含来源指纹、可用回复数、实际目标与同说话者错配目标的四个特征。`fit.json` 固定按对话留出的权重和训练输入指纹，`preexisting.json` 登记拟合时已经落盘的原始问答路径。`calibration.json` 和 `through-NN.json` 保存逐题 F1、各节点选技概率与期望读写 token。查看 `prospective_session13_plus` 可得到冻结后在较晚 session 中的覆盖和 F1。248 定期把该目录拉回论文工作区。

### 案例提示的选技试验

251 的物理 GPU 7 串行运行现有 22% 占空比 BF16 Llama，两个客户端复用同一个服务。GPU 1 在单独的限额租约中计算概率、枚举可行分数区间并核对已观测的写入和问答轨迹。每条推断原子保存，客户端可从已有响应恢复。主输出目录为 `~/memskill-work/runs/paired-251-linked-eight/source-compression/case-credit/`。两阶段追加登记和结果在其 `joint/` 子目录。

```bash
cd ~/memskill-work/run-linked-questions-v1
systemctl --user status memskill-251-case-credit memskill-251-case-credit-joint memskill-251-audit-case-credit
# 瞬时服务已退出、仍有未完成的窗口时，按需恢复对应任务。
systemd-run --user --collect --unit=memskill-251-case-credit --property=Restart=on-failure --property=RestartSec=45 bash "$PWD/scripts/experiments/run_case_credit_251.sh"
systemd-run --user --collect --unit=memskill-251-case-credit-joint --property=Restart=on-failure --property=RestartSec=45 bash "$PWD/scripts/experiments/run_case_credit_joint_251.sh"
systemd-run --user --collect --unit=memskill-251-audit-case-credit --property=Restart=on-failure --property=RestartSec=45 bash "$PWD/scripts/experiments/audit_case_credit_251.sh"
```

`prompts.json` 保存全部目标与提示，`preexisting.json` 保存登记时已有的输出指纹，`responses/` 和 `choices/` 保存实际推断及数值偏移。`factorial/`、`later/` 保存逐题实测问答所得的策略得分。`bounds/` 与 `joint/capacity-*.json` 是只供诊断的事后区间。两个原有 GPU 7 生成服务结束后，如推断尚未完成且端口空闲，客户端可按相同模型和占空比在租约内启动服务。

### 逐题证据匹配与后期训练轨迹补采

251 的 GPU 1 用 Qwen3-Embedding-0.6B 编码证据发言，编码占空比 30%，批大小 2。拟合与统计在同一卡的单独租约运行。GPU 2 的补采器一次租约最多新增一条真实轨迹，读取原随机试验的两条已测路径，共用 GPU 7 上占空比 22% 的 BF16 写入与问答服务。248 定期拉回数值预测、实测问答、成本和日志。实验期间，守护器暂停所有项目空闲占用。

在 251 查询进度或在相应瞬时单元退出后恢复未完成的工作。编码完成时编码单元正常退出。

```bash
cd ~/memskill-work/run-linked-questions-v1
systemctl --user status memskill-251-evidence-turns memskill-251-evidence-transport memskill-251-donor-factorial
systemd-run --user --collect --unit=memskill-251-evidence-turns --property=Restart=on-failure --property=RestartSec=45 bash "$PWD/scripts/experiments/run_evidence_turns_251.sh"
systemd-run --user --collect --unit=memskill-251-evidence-transport --property=Restart=on-failure --property=RestartSec=45 bash "$PWD/scripts/experiments/audit_evidence_transport_251.sh"
systemd-run --user --collect --unit=memskill-251-donor-factorial --property=Restart=on-failure --property=RestartSec=45 bash "$PWD/scripts/experiments/run_donor_factorial_251.sh"
```

产物根目录为 `~/memskill-work/runs/paired-251-linked-eight/`。进入 `source-compression/evidence-transport/`，查看 `turns/` 的 128 份来源、`pilot/predictions/` 的 64 份冻结预测、`pilot/factorial/summary.json` 的 91 题结果与 `pilot/sign-capacity.json`、`pilot/suffix-diagnostic.json` 的事后方向诊断。全部 88 题获得原路径与规则 2 回执后，`expanded/` 自动冻结剩余 40 个窗口的预测。查看最新 `later/through-*.json` 的 `strict_session13_plus` 和 `eligible_for_development`。独立补采位于 `late-pairs/donor-factorial/`，各窗口的 `complete.json` 含四条真实路径的问答、读写成本指纹和逐题精确期望。原验证标准及登记划分继续适用。

复算先导策略的步长上限与随机末段误差时运行以下命令。

```bash
source ~/memskill-work/deploy/env-251.sh
cd "$work/run-linked-questions-v1"
python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.evidence_transport_capacity "$work/runs/paired-251-linked-eight"
```

### 首段训练回报的模型辅助校正

在 251 的 `~/memskill-work/runs/paired-251-linked-eight/source-compression/assisted-span-backup/` 中，`registration.json` 固定了实验方案及当时已有的物理分支，`masked-policy.json` 包含 64 种遮蔽分配下的实测策略结果。各阶段的 `donors.json`、`predictions/`、`factorial/` 和 `later/` 分别保存训练与代码指纹、数值预测、完整路径计分和随机计分。阶段名分别为 `interim`（45 道新增训练题）、`expanded`（88 道新增训练题）和 `resolved`（同 88 题的互补路径齐备）。各阶段只读取其指定训练题。守护由 251 的物理 GPU 1 计算，248 自动拉回。可用下列命令查询、恢复或复核。

```bash
source ~/memskill-work/deploy/env-251.sh
cd "$work/run-linked-questions-v1"
systemctl --user status memskill-251-assisted-span
# 单元退出且任务未完成时恢复
systemd-run --user --collect --unit=memskill-251-assisted-span --property=Restart=on-failure --property=RestartSec=60 bash "$PWD/scripts/experiments/run_assisted_span_251.sh"
python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.assisted_trial diagnose "$work/runs/paired-251-linked-eight" "$work/study-linked-eight-gpu" "$LOCOMO_DATA"
```

### 原生时间修订技能的写入对照

`memskill-251-temporal-update` 在 GPU 1 上限速编码，复用 GPU 7 的 BF16 生成。早期和后期两个登记队列交替取下一个已经具备原始对照的窗口。每次租约准备一个窗口，保存真实写入、检索结果与读者输入后归还 GPU 1。问答客户端随后将新提示发到 GPU 7，按核验过的提示复用相同回答，两个阶段都可断点恢复。完整输出在 `~/memskill-work/runs/paired-251-linked-eight/source-compression/temporal-update-writer/`。交接时核对 `forecast.json` 的 37 个目标，以及各窗口的 `choice.json`、`writes.json`、`pair.json` 与 `recovery-candidates.json`。后两项分别记录逐题实际回答与待排查的归因。按窗口完成顺序查看 `early-*.json` 和 `later-*.json` 的可匹配题数、旧记忆修改、逻辑读写 token、按对话和题型的 F1。只有完整后期评价才计算是否通过该技能的筛选。原始物理轨迹结果、执行故障和不匹配原因均保留。

```bash
source ~/memskill-work/deploy/env-251.sh
cd "$work/run-linked-questions-v1"
systemctl --user status memskill-251-temporal-update
# 单元退出且仍有等待采集的窗口时恢复
systemd-run --user --collect --unit=memskill-251-temporal-update --property=Restart=on-failure --property=RestartSec=60 bash "$PWD/scripts/experiments/run_temporal_update_251.sh"
# 重算扩大动作范围的依据，结果保存在 assisted-span-backup/history-capacity.json
python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.evidence_transport_capacity "$work/runs/paired-251-linked-eight" --by-history
```
