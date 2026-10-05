# Qwen3.6 决策训练微批次对照

目标是减少 FSDP2 CPU 卸载训练中一次优化器更新需要的微批次、参数搬运和梯度同步次数。比较每卡 batch/累积次数为 `1/4`、`2/2`、`4/1` 的配置，四卡全局 batch 始终为 16。语言主干全参训练、冻结视觉、数据、学习率、600步调度和卸载策略保持一致。不把小模型耗时当作真实35B性能。

另一个优化是多样本的决策位置筛选：基础版SWIFT在batch大于1时保留从最早目标位置到末尾的整段logits，造成不参与损失的词表输出。`decision_sparse_logits.py`显式启用后，对本任务模板取各样本目标位置的并集，只计算前一位置的预测并保持因果标签对齐；交叉熵仍覆盖完整词表，不缩成候选类别分类。batch1沿用原逻辑。先核对损失、hidden梯度、输出层梯度，再做完整模型更新检查。

## 环境与基础版本

- 基础镜像：`quay.nju.edu.cn/ascend/ms-swift:v4.5.2-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12`，无需导出。
- 镜像 digest：`sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1`。
- 已有训练环境：CANN 9.1、Torch 2.10、torch_npu 2.10.0.post2、Transformers 5.15.1、Accelerate 1.14、PEFT 0.20、NumPy 1.26.4。
- 训练基础代码：个人 MS-SWIFT `qwen36_35b_decision_npu`，`e6667c9276186ed0dfe769058f9d1a8e93f16696`。
- 本目录由任务19独立微批次优化分支交付，任务归档记录完整交付 SHA。运行前沿用基础分支的数据准备、模型下载及容器挂载。训练数据和权重不随代码发布。

## 单个对照运行

在已有容器内设置路径。每个 trial 使用不存在的独立输出目录，恢复点不可被试验写入。`TOTAL_STEPS=600` 保留原调度；回调在恢复后的4次更新结束时停止，不通过缩短 max_steps 改变学习率曲线。

```bash
export TRAIN_WORKSPACE=/workspace
export MODEL_PATH=/models/Qwen3.6-35B-A3B
export RESUME_FROM=/workspace/outputs/train/checkpoint-300
export PROBE_PLUGIN=/workspace/throughput-tuning/throughput_probe.py
export OUTPUT_DIR=/workspace/throughput-tuning/batch2
export MICROBATCH=2 TOTAL_STEPS=600 DECISION_PROBE_UPDATES=4
export DECISION_SPARSE_LOGITS=1
export MASTER_PORT=29661 HCCL_NPU_SOCKET_PORT_RANGE=auto
bash /workspace/throughput-tuning/run_throughput.sh
```

依次把 `MICROBATCH` 设为 1、2、4，每次重新从同一完整断点恢复，并更换输出目录。不要同时运行真实35B候选。小模型并行检查也会暂时竞争资源，不能用于整网计时。[HCCL 官方端口说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0144.html)：多进程共卡时须避免默认端口冲突。

## 计时、数值与恢复检查

`throughput_probe.py` 记录各rank每次更新的同步端到端时间、分配/保留内存峰值和去掉padding后的输入/标签哈希。首个更新作为预热排除；其后三次更新分别取最慢rank耗时，再取中位数。该窗口较短，最终仍需观察长跑稳定性。

```bash
python check_throughput.py
python check_sparse_logits.py --device npu:0
python - <<'PY'
from tune_microbatch import summarize, admissible
baseline = summarize('/work/tuning/baseline', start=300)
candidate = summarize('/work/tuning/batch2', start=300)
print(admissible(baseline, candidate))
print(baseline['median_seconds'], candidate['median_seconds'])
PY
```

准入检查要求每次更新的全局16条样本完全一致，loss和梯度范数有限，并且相对基线的逐步loss差不超过5%、梯度范数差不超过10%。这些只是短跑异常筛查，不是最终模型精度标准。至少快10%的合格候选才参与自动选用；无合格候选则恢复原batch。

先用同架构小模型做四卡训练与完整保存恢复，并保留恢复后的模型状态，检查更新差异：

```bash
python compare_batch_weights.py \
  --initial /workspace/tiny-base/checkpoint-2 \
  --baseline /workspace/tiny-batch1/checkpoint-6 \
  --candidate /workspace/tiny-batch2/checkpoint-6 \
  --output /workspace/tiny-weight-comparison.json
```

要求候选与基线的权重差异L2不超过基线相对初始断点更新L2的10%，且确实产生非零更新。重复检查batch4。小模型通过不表示真实35B已通过，正式对照必须重新验证真实模型恢复、内存、输入和数值。

## 完整保存后的自动接续

宿主运行 `python3 tune_microbatch.py --plan /secure/local-plan.json`。计划中的路径、容器名、原训练launcher及四个rank的PID/启动身份仅保存在本地。驱动等到目标断点的模型、优化器四个分片、元数据、scheduler、四rank随机数和trainer状态都完整且大小/mtime稳定，再向精确匹配的原torchrun发送TERM。只有原进程退出后才依次执行对照。不会终止其他服务或整台机器上的训练进程。

计划文件结构如下；所有尖括号字段必须按当前机器填写。`owned_pids`包含launcher及四个rank，其启动身份来自`/proc/PID/stat`的starttime字段（正确处理括号内进程名后读取），禁止复用过期PID。运行前确认目标checkpoint尚未被轮换删除，核对所有路径和小模型检查结果。

```json
{
  "container": "<owned-training-container>",
  "host_output": "/work/task19/throughput-tuning/real-trials",
  "container_output": "/workspace/throughput-tuning/real-trials",
  "host_checkpoint": "/work/task19/outputs/train/checkpoint-300",
  "container_checkpoint": "/workspace/outputs/train/checkpoint-300",
  "checkpoint_step": 300,
  "continuation_output": "/workspace/outputs/train",
  "container_launcher": "/workspace/throughput-tuning/run_throughput.sh",
  "launcher_pid": "<launcher-pid>",
  "original_master_port": "29651",
  "owned_pids": {"<launcher-pid>": "<starttime>", "<rank0-pid>": "<starttime>", "<rank1-pid>": "<starttime>", "<rank2-pid>": "<starttime>", "<rank3-pid>": "<starttime>"},
  "wait_seconds": 86400,
  "environment": {
    "TRAIN_WORKSPACE": "/workspace",
    "MODEL_PATH": "/models/Qwen3.6-35B-A3B",
    "PROBE_PLUGIN": "/workspace/throughput-tuning/throughput_probe.py",
    "MASTER_PORT": "29661",
    "TOTAL_STEPS": "600",
    "SAVE_STEPS": "150",
    "DECISION_SPARSE_LOGITS": "1"
  }
}
```

每个真实候选最多两小时。失败候选不写入恢复点；发现遗留进程时拒绝重叠启动。通过样本/数值/速度筛查后，从未被试验修改的断点以选中配置继续至600步。正式接续恢复正常保存和验证，关闭逐输入哈希。`state.json`、各trial日志及每rank JSONL用于追踪，只有完成状态才能证明试验已结束。训练最终完成后，仍需按任务19已有独立业务、49项和官方七项评测入口验收；服务压测不适用。

此驱动不会自动清理或迁移checkpoint。启动前须核对下一次完整保存所需容量；跨盘分片挂载和后续轮换按任务19存储说明处理。正式模型精度、吞吐结果和原始证据只本地保存。
