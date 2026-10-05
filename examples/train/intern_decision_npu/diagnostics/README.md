# 4B决策模型推理数值诊断

## 章节一 模型简介

针对已完成训练的Qwen3.5-4B决策权重，固定输入、标签、候选答案与权重，对比三种NPU推理配置：融合GDN/BF16、PyTorch参考GDN/BF16、PyTorch参考GDN/FP32。参考路径仍在同一NPU执行，不是CPU/GPU金标准。此工具不修改训练框架或模型权重，不执行训练、校准、阈值拟合或模型选择。

## 章节二 性能和精度数据

采用JevBench-Hard完整111条，batch=1、SDPA、温度1、最大长度8192，超长拒绝而非截断。三种模式要求输入token哈希和数据哈希相同、111条全覆盖。记录逐题结果及正确性配对变化。FP32模式同时改变模型计算dtype，不能把它与BF16的差异全部归为GDN算子误差。

最终精度、性能及原始预测只存本地；本说明不发布数值。本轮不做服务压测，不用参考路径耗时代表部署性能。即使推理预测接近，也不能证明训练反向数值一致或充分解释训练数据覆盖问题。

## 章节三 复现分支和指导

本工具基于个人MS-SWIFT的4B分支`intern_decision_4b_npu`，基础提交`e815cb65efcb27c00c3cc5524603d6916ab5a112`；任务分支`task19_4b_precision_diagnosis`。工具所在提交请使用交付任务书中核验的完整SHA固定检出。

推理源码固定为个人Intern-Decision的`intern_decision`分支，提交`c025a7d1a516d792b6f32df0e14cbf7c1471d41c`。该仓库未修改。运行时核查schema、engine、HF/NPU backend与评分器的SHA256与此提交一致。

基础镜像：`quay.io/ascend/vllm-ascend:nightly-main`，拉取时固定registry digest `sha256:2c8aac4281e56953764a7fa60773cec734342d32d9d6d76c0c8f3a8660a64b85`；本地image ID为`sha256:4f7bc48083e5efc9f506bba85c1de16361e3f40cf5dc0220ae54712863ed7d18`。本实验只调用torch_npu，不运行服务引擎。不导出镜像。

已有环境为910B4-1、CANN9.1.0、Python3.12.13、Torch2.10.0+cpu、torch_npu2.10.0.post4、Transformers5.14.1。额外虚拟环境和推理源码不包含在基础镜像中，准备方式见固定推理仓库的`docs/NPU.md`：

```bash
git clone --branch intern_decision https://github.com/Liuchenbing-2026/Intern-Decision.git inference
git -C inference checkout c025a7d1a516d792b6f32df0e14cbf7c1471d41c
cd inference
uv venv --python 3.12 --system-site-packages .venv
uv pip install --python .venv/bin/python -r requirements-npu.txt -r requirements-eval.txt
```

保留平台匹配的Torch/torch_npu，勿用GPU版requirements覆盖。准备包含训练tokenizer的完整HF权重，以只读方式挂载。上述安装步骤为复现指南，本轮复用既有匹配环境，未从零重装。

## 章节四 启动和配置参数

在已确认空闲的单卡独立容器执行，勿复用正在训练的容器。实际独立容器不开放网络，使用私有/dev并仅保留指定davinci节点、设置ASCEND_RT_VISIBLE_DEVICES，运行前检查torch.npu.device_count()==1；驱动目录只读挂载，输出单独可写。示例宿主启动：

```bash
export DEVICE_INDEX=5
export IMAGE_REF=quay.io/ascend/vllm-ascend@sha256:2c8aac4281e56953764a7fa60773cec734342d32d9d6d76c0c8f3a8660a64b85
# SOURCE_DIR、TOOLS_DIR、MODELS_DIR、OUTPUT_DIR均指向准备好的宿主目录。
docker run -d --name decision-numerics --network none --privileged --shm-size 16g   -e ASCEND_RT_VISIBLE_DEVICES="$DEVICE_INDEX" -e OMP_NUM_THREADS=4   -e PYTHONPATH=/inference -e HF_HUB_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false   -v "$SOURCE_DIR:/inference:ro" -v "$TOOLS_DIR:/diagnostics:ro"   -v "$MODELS_DIR:/models:ro" -v "$OUTPUT_DIR:/outputs"   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro   -v /usr/local/dcmi:/usr/local/dcmi:ro   --entrypoint bash "$IMAGE_REF" -c   'for i in 0 1 2 3 4 5 6 7; do if [ "$i" != "$ASCEND_RT_VISIBLE_DEVICES" ]; then rm -f /dev/davinci$i; fi; done; exec sleep infinity'
```

直接评测命令（输出路径必须全新）：

```bash
docker exec -w /inference decision-numerics /inference/.venv/bin/python   /diagnostics/evaluate_numerics.py --checkpoint /models/checkpoint   --data /inference/benchmarks/accuracy-v1/jevbench/hard.jsonl   --mode fused-bf16 --output /outputs/fused-bf16.jsonl
# 依次使用reference-bf16、reference-fp32及对应新输出名。
```

宿主顺序调度支持多个固定checkpoint，私有plan示例：

```json
{
  "container": "decision-numerics",
  "source": "/inference",
  "python": "/inference/.venv/bin/python",
  "script": "/diagnostics/evaluate_numerics.py",
  "host_output": "/local/diagnostic-output",
  "container_output": "/outputs",
  "data": "/inference/benchmarks/accuracy-v1/jevbench/hard.jsonl",
  "models": {"checkpoint_a": "/models/checkpoint_a", "checkpoint_b": "/models/checkpoint_b"},
  "expected_rows": 111,
  "stop_exclusive_container": true
}
```

```bash
python3 run_numerical_diagnosis.py --plan /local/private-plan.json
```

仅对本实验专用容器设置stop_exclusive_container。程序失败即停止后续阶段，保留日志；正常结束或失败后释放专用容器设备。每阶段最多一小时，未完成不得算作通过。输出包括state.json、逐阶段日志/预测/metrics/audit与最终comparison.json。

## 章节五 问题列表

- 原有单题与联合输入差距属于协议影响，不能由本次Hard测试关闭。
- 原有训练覆盖不足仍需独立训练数据和验证集的干预对照，本轮不改变任何训练样本。
- 无CPU/GPU金标准、无训练反向逐层对齐，本轮结论仅限固定权重的NPU推理路径及指定111条。
- 不根据该测试调整超参数、选择权重或把拆题分数替换官方联合输入分数。

## 章节六 总结

本工具用于量化推理实现与计算dtype对现有权重预测的影响，为训练精度定位提供一个受控检查。完成范围以本地state、完整覆盖和同输入核验为准；启动成功不能写成精度问题已修复。
