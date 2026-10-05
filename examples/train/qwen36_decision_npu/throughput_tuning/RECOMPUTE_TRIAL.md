# 章节一 模型简介

本流程用于 Qwen3.6-35B-A3B 的全参数语言主干决策训练；视觉模块冻结。训练数据、损失、全局 batch 16、优化器及完整恢复点均保持一致。

# 章节二 性能和精度数据

增加关闭 FSDP `activation_checkpointing` 的独立四次更新试验。命令行 `gradient_checkpointing=false` 不会自动关闭 FSDP 层面的重计算。试验结果与最终精度保存在本地；未完成真实试验前不宣称提速。

第一步作为预热，其余三步按每步最慢 rank 计时后取中位数。要求相同全局样本及标签哈希、有限数值、loss 相对差不超过 5%、梯度范数差不超过 10%，并相对原始基线至少提速 10% 才选用。这些诊断阈值不替代独立精度验收。

# 章节三 复现分支和指导

代码分支 `task19_qwen36_recompute_trial`，基于 `task19_qwen36_resume_pinfix` 的 `f1e684d5244281531e9800fb0253f405612a66e0`。

基础镜像 `quay.nju.edu.cn/ascend/ms-swift:v4.5.2-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12`，digest `sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1`。在已核验的原训练容器中执行，使用父分支提供的插件和配置。

# 章节四 启动和配置参数

从原 FSDP JSON 生成仅修改一项的候选配置：

```bash
python - "$ORIGINAL_FSDP_CONFIG" "$NO_RECOMPUTE_CONFIG" <<'PY'
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
assert config['fsdp_config']['activation_checkpointing'] is True
config['fsdp_config']['activation_checkpointing'] = False
with open(sys.argv[2], 'x') as out:
    json.dump(config, out, indent=2)
PY
```

单次真实计时对照（在容器内执行；变量均为当前实例的显式路径）：

```bash
export TRAIN_WORKSPACE MODEL_PATH PROBE_PLUGIN RESUME_FROM OUTPUT_DIR
export FSDP_CONFIG="$NO_RECOMPUTE_CONFIG"
export MICROBATCH=1 TOTAL_STEPS=600 SAVE_STRATEGY=no EVAL_STRATEGY=no
export DECISION_PROBE_UPDATES=4 DECISION_PROBE_AUDIT=1 DECISION_SPARSE_LOGITS=1
bash run_throughput.sh
```

自动对照和接续：

```bash
python check_throughput.py
python tune_microbatch.py --plan "$PLAN_JSON"
```

计划沿用父分支的容器、完整 checkpoint、独占设备、输出和环境字段，新增 `no_recompute_config`（容器内候选 JSON 路径）。输出必须独立且不存在旧 state.json。可选 `reuse_trials` 对象以 `baseline`、`batch2` 为键，每项包含 `host_output`、`container_output` 和 `controller_state`，仅复用同一 checkpoint 的完整四 rank 记录及最终 runtime 记录。复用记录的 SHA256 会写入新状态。

若接管尚未完成的探针，先核验并停止旧 host 控制器本身，保留其 docker/torchrun 子任务；计划 `handoff` 必须包含旧 `controller_pid`、`controller_identity`、`child_pid`、`child_identity` 和 `container_output`。identity 是 /proc/PID/stat 的 starttime。新控制器只等待该任务退出，不杀训练 rank；旧控制器存活、设备不空闲或记录不完整时拒绝新启动。`previous_outputs` 必须覆盖所有旧输出。存储轮换守护程序需同步到新状态路径，避免读取过期状态。

通过候选诊断后从同一个保存点正式继续，候选更新不保存，不累计为正式训练进度。选中的 FSDP 配置传入正式接续命令；候选异常且其所有进程退出后采用已通过的原配置，禁止重叠任务。

# 章节五 问题列表

CPU 参数/梯度卸载和 FSDP 重计算均可能增加耗时。去除重计算增加显存需求，应以真实峰值及恢复后更新结果判断。保存空间与最终评测仍由原归档中的存储轮换和验收流程负责。无需导出基础镜像。

# 章节六 总结

本改动提供可回退的重计算消融试验与已完成探针的可核验复用。CPU 检查覆盖存活控制器拒绝、记录及 checkpoint 一致性、候选配置传入接续；不能替代真实大模型训练验收。
