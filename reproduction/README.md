# 已完成的314题测试代码存档

本存档按用户要求保存并发布三个Git版本。原项目目录和已完成实验保持原样。

## 三个版本

| 分支 | 内容 | 本次314题F1 |
| --- | --- | --- |
| `memskill-original` | 作者原始提交 `f65b797338033f4f9c34822f1810f9c2747e6022`，原始文件树未改动 | 原生算法复现结果25.250262，见下述说明 |
| `memskill-memory10-raw` | 原版写入；回答读取前10条问题检索记忆和1段问题检索原文 | 34.318242 |
| `memskill-best` | 上述混合读取，加K=1、insert仅读当前原文、块内精确INSERT去重 | 40.665413 |

三个分支都有日期标签 `snapshot/20261126-original`、`snapshot/20261126-memory10-raw`、`snapshot/20261126-best`。

纯原版分支保持作者Git提交原样。25.25是使用本项目GPU/后端适配与审计驱动得到的原生算法结果，不能说是刚刚在未经适配的上游默认配置上重新跑出的分数。对应实际运行源代码也一并保存。

## 代码位置与运行方式

- 根目录：实际测试使用的MemSkill写入运行时源代码，按冻结快照原样保存。
- `reproduction/frozen/snapshot/run-linked-questions-v1/`：完整写入运行时快照。
- `reproduction/frozen/snapshot/deploy/sources/locomo-credit-attribution-v1/`：完整读取运行时快照。`experiments/controller/passage_prefix_sources.py`保存原文片段构造与预算适配，`reader_slot_credit.py`保存读取请求和响应核验。
- `reproduction/frozen/fair_test_mixed.py`：第二组实际使用的10条记忆＋1段原文答题入口。
- `reproduction/frozen/fair_test_writer.py`：原生对照和最佳组合实际使用的写入入口。`FAIR_TEST_ARM=native`恢复K7原生对照；`FAIR_TEST_ARM=best`启用K1和仅当前原文写入，并调用去重。
- `reproduction/frozen/fair_test_common.py`：`unique_actions`按同一块解析后的INSERT正文做字符串精确去重；`test_reader`使用前10条记忆与固定原文，并按原生预算裁剪。
- `reproduction/frozen/prepare_fair_test.py`：冻结原文排名、片段、提示词与逐题输入上限。原文排名来自先前已核验的GPU检索，不使用参考答案检索。
- `reproduction/frozen/run_fair_test.sh`、`fair_test_queue.py`：原始环境和调度入口；`score_fair_test.py`、`audit-author/`：固定作者评分器与完整对照核验。
- `reproduction/ACTIVE_VERSION.json`明确本分支采用哪一组参数和入口。为了保留完整对照和审计链，两条改进分支都带有同一套未经重写的冻结驱动；不是新实现，也没有重跑实验。
- `reproduction/source-manifest.json`记录每个归档文件的原始路径和SHA256。运行 `python reproduction/verify_snapshot.py`只核验源文件并显示本分支配置，不启动模型。

冻结脚本保留原来的目录约定。原始完整运行目录为 `/home/lixian/Documents/研究生大论文/review-stage/hybrid-controller-20261126`；完整结果在其 `fair-test-three-v1/` 下。原调用为 `run_fair_test.sh writer native conv-49`、`run_fair_test.sh mixed mixed conv-49`、`run_fair_test.sh writer best conv-49`，conv-50同理。运行writer必须通过父目录AGENTS.md规定的GPU guard入口。已完成目录禁止覆盖；再次实验必须准备独立输出目录及相应注册文件，不能直接对这份已完成存档重复执行。

模型权重、Controller权重、数据集、完整记忆库、逐题输出和缓存没有装入Git；其位置和指纹见 `external-artifacts.json`和原始注册文件。仅克隆代码不等于已具备这些运行依赖。

## 配置和结果边界

Llama-3.1-8B-Instruct BF16/TP2，Qwen3-Embedding-0.6B；同一Controller checkpoint；512 tokens分块、重叠64；温度0、seed42、答题最多32 tokens。两段对话156＋158=314题。每题输入预算上限相同，不要求实际用量相同。后两组使用完全相同的原文片段。

最佳组合实际115块均选择insert；去重拦截803个重复INSERT动作，跨块重复保留。最佳组合相对混合读取增加6.347171个F1点，相对原版增加15.415150点；这是整套组合的收益，不能单独归因于去重或Controller学习。F1不是正确率；测试对话此前做过实验，不是全新未接触的独立测试。完整语义核验尚未完成。

原注册、配置和完整三组汇总保存在 `reproduction/evidence/`，保持原始字节。三组分数和既有答案没有因本次Git保存而改变。

## 当前分支

本分支采用原版K7写入，只修改答题阶段的10＋1读取。
