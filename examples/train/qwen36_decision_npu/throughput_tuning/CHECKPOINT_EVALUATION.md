# 第300步决策精度检查

本任务的训练集为4800条单问题记录，全局batch16；300次优化器更新相当于一轮。600步是两轮的初始预算，不是官方要求或已证明的最优步数。

完整checkpoint-300保存后，在与吞吐试验相同的四卡环境中顺序评测初始35B权重和第300步权重：

| 数据 | 用途 | 口径 |
| --- | --- | --- |
| 600条独立验证记录 | 判断后续训练收益 | 合法候选答案argmax准确率 |
| 2000条业务测试记录 | 与既定业务验收口径对照 | 仅报告，不用于调参或选择checkpoint |

不把训练token_acc或loss当作业务准确率。此检查不是官方七项平均或扩展49项评测；这些是单独的最终验收口径。是否需要继续第二轮应参考验证集，测试结果不能成为早停或超参数选择依据。本次增加评测，不自动改变既定600步训练计划。

## 实现与隔离

`checkpoint_eval.py`通过SWIFT外部回调，在框架完成FSDP模型、优化器和trainer状态恢复后、第一次训练更新前执行评测。评测期间不计算训练梯度、不执行optimizer.step、不保存模型。结束后通过专用异常退出已准备好的训练循环，返回正常评测完成状态。候选logits取自每个`<decision>`标记前的位置，softmax只覆盖题目给出的合法答案符号；输入保留完整证据、选项次序和单问题格式，禁止截断和隐藏失败记录。

逐rank记录预测、概率和输入哈希，合并检查完整覆盖；训练、验证、测试案例ID必须隔离。两轮评测必须具有相同数据哈希、输入哈希及条数。原始预测与最终数值只留本地。

## 环境、启动与比较

沿用本目录[吞吐优化复现说明](THROUGHPUT_TUNING.md)的基础镜像、固定训练依赖和工作区。无需导出镜像、重编译或安装新包。先配置已有训练数据、模型、四卡以及FSDP分片断点路径。

本地评测计划：

```json
{
  "training_data": "/workspace/decision_data/train.jsonl",
  "batch_size": 4,
  "suites": [
    {"name": "validation", "role": "validation", "data": "/workspace/decision_data/validation.jsonl", "expected_rows": 600},
    {"name": "business_test", "role": "test_report_only", "data": "/workspace/decision_data/test.jsonl", "expected_rows": 2000}
  ]
}
```

容器内独立命令（先确认训练/其他候选已退出）：

```bash
export TRAIN_WORKSPACE=/workspace
export MODEL_PATH=/models/Qwen3.6-35B-A3B
export PROBE_PLUGIN=/workspace/throughput-tuning/throughput_probe.py
export OUTPUT_DIR=/workspace/throughput-tuning/eval-checkpoint
export RESUME_FROM=/workspace/outputs/train/checkpoint-300
export MICROBATCH=4 TOTAL_STEPS=600
export DECISION_EVAL_ONLY=1
export DECISION_EVAL_CONFIG=/workspace/throughput-tuning/evaluation-config.json
export MASTER_PORT=29661 HCCL_NPU_SOCKET_PORT_RANGE=auto
bash /workspace/throughput-tuning/run_throughput.sh
```

初始权重对照把`RESUME_FROM`设为空并更换OUTPUT_DIR。完整结果在各自`evaluation.json`和逐rank JSON中。自动流程在`real-plan.json`增加`evaluation_config`（容器内路径）后，顺序执行初始权重评测、第300步评测、吞吐对照、按既定计划接续训练。不要在已有驱动执行期间另起评测抢卡；仅在驱动尚未接管、明确处于等待保存状态时替换驱动。

```bash
python3 tune_microbatch.py --plan /secure/local-plan.json
```

`state.json`的`evaluations`和`evaluation_comparison`记录两组结果及差值，`evaluating`阶段不代表完成。框架对分片模型的实际恢复、小模型评测冒烟、完整35B评测和最终质量验收分别记录；小模型检查不能替代35B精度结果。服务压测不适用，训练吞吐计时仍使用同目录的独立对照流程。
