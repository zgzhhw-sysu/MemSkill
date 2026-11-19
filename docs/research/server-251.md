# 251 上的配对实验

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
