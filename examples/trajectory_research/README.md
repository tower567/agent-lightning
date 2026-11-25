# 轨迹采集与离线分析（小规模研究）

本功能只增加观察与离线统计，不修改 Agent 的提示词、工具协议、决策循环、重试次数/退避、奖励公式或训练数据处理规则。所有记录均由 `AGL_RESEARCH_RUN_DIR` 显式启用；未启用时不写研究数据。CLI 没有训练入口。

本分支基于当前仓库 `2070bad`（包版本 1.0.3，Agent Lightning v1 系列），不是声称逐项复现论文 v1.0.0。每次采集保存实际 commit、工作区状态、diff 哈希及源码哈希。

## 安装与快速验证

在 WSL 中使用 micromamba 隔离环境，例如：

```bash
micromamba create -n agl-research -c conda-forge python=3.12 pip -y
micromamba run -n agl-research python -m pip install -e '.[research]'
micromamba run -n agl-research python -m agentlightning.research demo /tmp/agl-fixture-01
```

`demo` 只产生 **14 个合成生命周期样例**，不连接模型或检索服务，不申请 GPU、不运行 RL。样例包括成功、正常失败、缺失奖励、关联重试并恢复、超时、外部中断，以及仅排队/仍在执行。样例费用也是合成 usage 与演示价格的估计，不是实际账单。

```bash
python -m agentlightning.research analyze /tmp/agl-fixture-01 \
  --output /tmp/agl-analysis --bootstrap 1000 --seed 42
# 支持多个独立 run，并按共有 task_id 比较条件：
python -m agentlightning.research analyze /data/run-control /data/run-fault --output /data/comparison
# 不安装 matplotlib 时，可使用纯标准库统计：
python -m agentlightning.research analyze /data/run-control --output /data/tables --no-plots
```

同一 run_id 不可作为两个输入，以免重复计数。原始事件分析只按 event_id 去除相同导出记录的重复输入；相同请求文本、命令、prompt 或响应绝不作为事件去重键。重复记录数写入完整性报告。

## 真实小规模采集

1. 复制并填写 `metadata.example.json`；模型/数据 revision、抽样 ID 清单和实际参数需由实验启动者提供。未知字段填 `null`，不要把示例配置当成自动读取的实际配置。
2. 初始化一个新目录，然后把同一个目录作为环境变量传给 **server、local controller 及 Agent 子进程**：

```bash
python -m agentlightning.research init /data/research/run-001 --metadata my-metadata.json
export AGL_RESEARCH_RUN_DIR=/data/research/run-001
# 此后按现有方式启动 server/controller，只调度少量目标样例。
```

3. 调度时在 rollout input 中传入稳定的 `task_id` / `data_id` / `instance_id` / `id`；或在 metadata 中显式传 `task_id`。在 rollout metadata 中传 `repetition` 和 `condition`，例如：

```json
{"input":{"task_id":"hotpot-123","question":"...","golden_answers":["..."]},
 "is_train":false,
 "metadata":{"task_id":"hotpot-123","repetition":0,"condition":"control"},
 "config":{"timeout_seconds":60,"local":{"agent_class":"examples.search_r1.agents.search_r1_agent:SearchR1Agent",
 "env_map":{"QUESTION":"input.question","GOLDEN_ANSWERS":"input.golden_answers"}}}}
```

自动记录不会替代或改动现有调度请求。调用者未提供稳定任务 ID 或重复编号时记为缺失；不把相同文本哈希或原框架临时生成的分组 UUID 冒充数据集 task_id。`AGL_RESEARCH_ROLLOUT_ID` 由 LocalReconciler 传给实际子进程。

真实 Search-R1 仍需原有模型服务和真实检索环境。本功能的样例与自动测试使用明确标注的本地模拟后端，不能据此声称已采集真实模型轨迹或完成论文实验。

## 接入位置与覆盖范围

| 入口 | 本次记录 | 所处阶段 |
|---|---|---|
| `server/routes/rollouts.py` enqueue/patch/delete | 调度请求、稳定 ID、状态变更、删除通知 | enqueue 幂等分支之前也保留请求，状态拒绝请求不丢失 |
| `controller/local_reconciler.py` worker/spawn/finish/timeout/shutdown | 实际子进程开始、退出、错误栈；控制器超时与关停原因 | 不将正常退出解释为解题成功 |
| `server/routes/proxy.py` 与 `server/proxy.py` | gateway、每个真实上游尝试、重试关联、退避耗时、请求/响应、服务权重版本 | `_pop_routed_experts`、事件裁剪、triplet 去重之前 |
| `server/routes/events.py::record_event` | 全部原始框架事件、原始奖励 | route tensor 丢弃、reward 裁剪和 trace 去重之前 |
| `examples/search_r1/agents/search_r1_agent.py` | 真实进入的轮次、搜索 query/完整 JSON 结果、EM 评估、LLM 客户端请求 | EM 公式与调用循环不变；仅增加关联 HTTP header |
| `verl/rollout_adapter.py` | CompletedRollout 原始输入、实际输出行数/样本奖励、精确 token 前缀比较 | 原始输入在排序、补奖励和合并之前保存；输出另记 |

Search-R1 两种 Agent（chat / completion）均接入。LLM SDK 每次逻辑调用使用独立 `X-AGL-Research-Call-ID`；SDK 重发与 gateway 内重试均可关联。其他 Agent 经过代理时也能采集每个上游尝试，但其 SDK 级重发若没有显式关联 header，标记为未知关联，不能按文本推断。其他工具/评估器需在原调用边界使用下面的观察接口。

```python
from agentlightning.research.events import observed_call, observe

with observed_call(
    "tool",
    rollout_id,
    tool_type="shell",
    request={"command": command},
    retry_group_id=logical_id,
    retry_of=previous_call_id,
) as call:
    result = existing_tool(command)  # 原有执行、重试及异常处理保持原样
    if call:
        call.result.update(response=result)
# 只有任务评估器能定义 success，不能泛化成 reward > 0。
observe("reward_observed", rollout_id, value=raw_reward, reward_valid=True, success=task_success, source="my_evaluator")
```

每一次重试单独进入 context，并传同一 retry_group_id、前一个 call_id。async 调用也可在同步 `with` 内 `await`。仅记录/重新抛出异常，不执行重试，不吞异常。

当前未自动覆盖：任意第三方工具内部重试、同步 SDK 自带重试、Kubernetes Pod 内实际开始/停止、外部未监控进程被 SIGKILL 的准确原因、没有显式轮次事件的逐轮风险。K8s 的 controller RUNNING 不冒充实际 Agent 启动；需在 Pod 内补相同观察接口并导出事件。本机多进程 writer 使用文件锁；分布式节点应分别写目录再按 run/task 配对分析，不建议多节点写不支持 flock 的共享文件系统。

## 产物与统计口径

- `events.jsonl`：只追加原始事件，包含失败、未结束调用及所有奖励观察，不因框架后续过滤/删除而删除。
- `metadata.json`：运行身份与版本，初始化后禁止覆盖；源文件哈希包含未提交代码。
- `calls.csv`：逐调用的标识、错误、关联、耗时与 token 字段覆盖情况；完整载荷仍在原始事件中。
- `rollouts.csv`：一行一个 `(run_id, rollout_id)`，保留尚未启动/结束的记录。
- `metrics.json` / `metrics.csv`：数量、比例及分母、缺失覆盖、按 task 聚类 bootstrap CI。
- `distributions.csv`：均值、中位数、P90、P95，另列实际观测数及缺失数。
- `length_reward_missing.csv`、`round_interruption_risk.csv`、`by_task_id.csv`、`by_tool_types.csv`、`by_termination_reason.csv`。
- `attempts_by_task.csv`、`first_fault_and_recovery.csv`、`paired_tasks.csv`、`paired_conditions.csv`。
- `associations.csv`、`by_fault_status.csv`、`by_retry_status.csv`：描述性相关与故障/重试分组；不输出因果结论。
- `adapter_reward_baselines.csv` 与 Adapter 汇总；没有输出时 `unavailable`，不重新 tokenize 或模拟 Adapter 输出。
- 同名 PNG/SVG 图：长度与缺失奖励、分组成功/成本、首次故障与恢复、同题尝试、配对差异、逐轮风险。

有效评估率 = 有限数值原始奖励的 rollout 数 / 已启动数；bool、null、NaN、非数值均无效。奖励缺失率是未取得有效奖励数 / 已启动数。`reward=0` 是有效评估。每条原始奖励都保留，CSV 采用最后一次观察并标出冲突；本地评估已完成但回传失败与完全未评估可由 source/框架事件区分。

已确认成功比例 = `success is True` 数 / 已启动数，未知仍未知；已评估成功率 = 成功数 / 有效评估数。泛用 reward 未携带任务成功标签时 success 仍为 null；只针对已知 Search-R1 EM 原因解释成功。

启动以实际 `rollout_started` 为准；终止可来自控制器终态或实际 worker 退出。正常进程退出、框架状态、任务成功是不同字段。未结束轨迹保留并作为右删失观察，不能当成已失败或成功。

逐轮风险分母为**明确记录进入该轮的轨迹**，分子是该轮已观察到的外部中断/超时/预算中断。正在该轮执行的轨迹单列 right_censored_here，不认定中断。未知原因单列 unknown，不靠错误文字猜测。只有最终轮数而无轮次进入事件时，逐轮风险 unavailable。

最终长度被中断截短，分桶图仅表示观测关联。配对比较按共同 task_id 聚合各条件的全部已启动重复尝试，再比较 task 均值；不把不同重复编号强配成同一次实验。CI 按 task 聚类/配对 task bootstrap（seed 可复现），不足两个 task 时不报告 CI。所有输出都不能自动解释为因果效应。

预算耗尽仅统计明确记录的预算原因；未知终止使精确耗尽率 unavailable，同时报告已确认下界与可能上界。模型某轮 `finish_reason=length` 保留原响应，不能自动认定整个 Agent 的 token 预算耗尽。

成本分开报告：总墙钟/排队、各类调用时长之和、同一时钟域内调用区间并集、实际上报的生成 token、按显式价格估算的 LLM 美元费用。不同 clock_id 不能相减；并行调用和不能作总墙钟。每 run 另报从首条到末条捕获事件的观察时长，未结束时仅为下界。调用耗时用开始记录落盘后的单调时钟测量，记录落盘开销单列；任务墙钟仍包含观察开销。

重试成本定义为关联组中第一次以外的额外尝试成本，另报 backoff；所有尝试总成本仍包含首次失败。重试恢复率分母为首次失败且确实重试过的组，分子为随后已观察到成功的组；未结束重试不视作已恢复。工具错误率分母是已启动工具调用，未结束调用单列。

未声明完整工具采集的 Harness，其工具总次数记为 unavailable，并保留 observed 次数；无 SDK 关联 ID 时不把重发次数冒充决策轮数。Search-R1 在实际启动时声明其已接入的工具、评估和轮次能力。

token 和费用未知时不补 0：超时或 5xx 的已消耗 token 可能不可得，只报告已观测下界。没有价格就不估美元费用；价格估计不冒充账单或完整基础设施成本。所有尝试总成本 / 成功数在无成功或成本不完整时不可计算。工具类型分组中同一 rollout 可属于多个工具，分组成本不可再次相加。

## 字段定义

| 字段 | 含义 |
|---|---|
| run_id / rollout_id / task_id / repetition / condition | 运行、尝试、稳定任务、重复编号、实验条件；缺失不推断 |
| event_id / event_seq | UUID 原始事件身份、文件内全局顺序（允许进程崩溃造成序号间隙） |
| utc / monotonic_ns / clock_id / pid / host | UTC 时间、单调纳秒与主机启动时钟域、来源进程；跨域不求差 |
| event_type / data | 原始事件类型与完整载荷；非有限数值用带类型对象保存，仍判为无效奖励 |
| call_id / call_index / kind / tool_type | 物理调用身份、全 run 调用开始事件序号、类型与工具名；每 rollout 可按序号排序 |
| retry_group_id / retry_of / parent_call_id | 明确的重试组、上一尝试、SDK 逻辑调用；不是文本相似关联 |
| request / response / error / status | 完整请求、响应、原错误类型/栈/分类，及调用状态 |
| started_utc / started_monotonic_ns / finished_monotonic_ns / duration_seconds | 实际调用区间及单调时钟耗时；start 事件与实际区间通过 start_event_id 关联 |
| usage / tokens | 上游真实 usage；实际 prompt/response IDs、采样 log probabilities、权重版本的值与 availability |
| value / reward_valid / success / source | 原始奖励、有限数值有效性、任务解决标签、来源；null 与 0 区分 |
| execution_status / process_exited_normally / termination_* | 框架/进程状态与明确原因；不等于 success |
| *_observed_lower_bound / n_unavailable / unmeasured_llm_attempts | 部分观测成本下界与覆盖情况，不能当成完整成本 |

原始 token 仅来自响应扩展或 Completions 实际 IDs；不以重新编码文本补齐。后端未返回 logprobs、token IDs、usage 或权重版本时标 `unavailable`。注册的权重版本是 registry 观察值，不额外声称权重文件内容哈希。元数据中的期望模型版本与实际服务版本分别保存。

观察失败采用 fail-open，以免改变任务逻辑；发出 RuntimeWarning，并在后续成功事件标出该进程丢失事件数。磁盘完全失效且没有后续记录时，必须查看进程 stderr，不能声称事件完整。JSONL 最后一行被进程终止截断时，分析保留文件并报告 truncated_tail；中间损坏直接报错，不静默跳过。

## 验证边界

自动测试覆盖统计分母、0/null、无成功、删失、重叠并行耗时、按 task 的 CI、配对比较、重试恢复、原始 token 缺失、并发追加、截断文件、观察故障；还执行实际代理重试循环、官方多轮 SearchR1Agent + EM（本地模拟 HTTP 后端）以及真实子进程 SIGTERM。未启动真实模型训练、真实 GPU 压测或大规模采样。可选 Adapter 原生 VERL 路径若环境未装训练栈，需在已有 VERL 环境单独验证，不以合成输出冒充真实 Adapter 数据。

补充统计口径：`retry_count` 分层合计可观察到的上游、SDK 网关重发与工具/评估重试，嵌套重试不重复计数；网关拒绝而未发起模型请求的 SDK 重试也计入。`sdk_attempt_coverage` 说明到达网关前的网络重试仍不可见。`retry_gateway_extra_seconds` 与模型调用耗时可能重叠，不可相加；首次故障时间按错误被观察到的调用结束时间记录，另保留该调用开始时间。模型调用耗时是端到端延迟，包含网络与服务队列；没有后端分项遥测时不估算纯 GPU 计算时间。分析产物记录分析脚本自身 SHA256。
