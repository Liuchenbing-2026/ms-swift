# 任务19：4B多任务覆盖干预训练

## 章节一 模型简介

在已完成的4B联合字段权重基础上继续训练语言主干，冻结视觉和projector。采用相同masked causal全词表交叉熵、FP32参数与CPU AdamW状态、BF16计算；不是LoRA，也不是原优化器断点续训。保留旧权重和旧评测。此独立实验补工具选择与安全判断，业务案例仍保留全部联合字段。

## 章节二 性能和精度数据

固定240次更新、每次4个案例；三个来源各320个唯一训练案例，按业务/工具/安全交错，安全内部safe/unsafe各半。比例指案例数，不是监督token数。独立验证保留各来源全部冻结验证记录；测试不参与采样、步数或选模。训练原始数据与完整数值只保存在本地。

先执行真实tokenizer审计、同架构小模型两步训练和保存重载检查，再执行真实4B的基线验证、固定训练和重载验证。统计真实进度JSON和日志的端到端更新时间，不能把CPU检查当作模型验收。服务压测未开展；最终业务、扩展49项、官方七项另行验收。该数据干预不保证达到官方综合精度。

## 章节三 复现分支和指导

个人MS-SWIFT分支`task19_4b_multitask_training`，基于联合训练与验收分支`c36ce5a139653225c0d44626cd35cbfde42112c5`。共享同一提交树中`../joint_training`的模板、schema、CPU AdamW与buffer保留导出辅助函数。当前目录提供独立数据混合、可变字段验证、调度、训练入口、CPU检查及本文；框架内部和原生算子未修改，无需重新编译。

基础镜像：`quay.nju.edu.cn/ascend/ms-swift:v4.5.2-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12`，固定registry digest `sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1`；实际镜像ID `sha256:ff2ea9131d5fdf8cf390171dd7af28fd3420646b2e010faf75dc004f58f8dc97`。实用依赖为SWIFT `ff5128777dd3dc5538f1a95fbcc29442c327d7d2`、Python3.12.13、torch2.10.0、torch_npu2.10.0.post2、Transformers5.15.1、CANN9.1。镜像无需导出；镜像中的既有依赖直接复用。

只占一张已核实空闲的910B 64GB卡，使用新容器和新输出目录。容器创建、CANN环境变量和只读驱动挂载沿用[固定环境说明](../joint_training/README.md)，将新代码/数据目录挂载`/workspace`，共享辅助源码复制到`/workspace/joint_training`，旧联合权重/小模型目录只读挂载`/joint`，旧业务数据只读挂载`/prior`。保留实际容器命令和镜像检查记录到私有运行清单，不在公开仓库写地址/凭据。环境不再安装或升级额外依赖。

多任务源与许可、固定revision、词面去重和测试输入排除见独立数据准备分支`task19_4b_multitask_data`提交`459dfcf5cad5700cd936331055f71c438bbd3bd7`：ToolACE Apache-2.0、Aegis2.0 CC-BY-4.0。必须使用已通过token审计的原始记录；保留人类标签，不按测试答案重标注。不使用受限数据或benchmark标签构建训练。

## 章节四 启动和配置参数

数据准备与CPU检查：

```bash
python check_mix.py
python check_runner.py
python prepare_mix.py --business-train "$BUSINESS_TRAIN" \
  --business-validation "$BUSINESS_VALIDATION" \
  --multitask-train "$AUDITED_MULTITASK_TRAIN" \
  --multitask-validation "$AUDITED_MULTITASK_VALIDATION" \
  --schema-dir "$JOINT_CODE_DIR" --per-source 320 --seed 42 --output "$NEW_DATA_DIR"
```

容器内直接启动（完整调度使用下面host入口）：

```bash
export TRAIN_WORKSPACE=/workspace JOINT_CODE_DIR=/workspace/joint_training
export PYTHONPATH=/workspace:/workspace/joint_training
export MULTITASK_DATA=/workspace/data MODEL_PATH=/joint/experiment/checkpoint-120
export OUTPUT_DIR=/workspace/experiment TOTAL_STEPS=240 ACCUMULATION=4
export TRITON_CACHE_DIR=/workspace/triton-cache TMPDIR=/workspace/tmp
python audit_mix_tokens.py --data "$MULTITASK_DATA" --tokenizer "$MODEL_PATH"
python multitask_validation.py --checkpoint "$MODEL_PATH" \
  --data-dir "$MULTITASK_DATA" --output /workspace/baseline
bash /workspace/run_multitask.sh
python multitask_validation.py --checkpoint /workspace/experiment/checkpoint-240 \
  --data-dir "$MULTITASK_DATA" --output /workspace/reload
```

正式host调度：`python run_multitask.py --plan "$PRIVATE_PLAN"`。计划包含`host_workspace`、`container`、`prior_acceptance`、`checkpoint`、`checkpoint_shards`（宿主只读原权重分片列表）、`smoke_checkpoint`、`steps=240`、`reserve_gib=1.5`、`pinned_files`（训练数据和源码文件SHA256）、`free_device_check`命令，以及`environment`中的PYTHONPATH、JOINT_CODE_DIR、TRITON_CACHE_DIR、TMPDIR、HF_HOME。调度仅等待前一轮完整验收成功，不根据其分数选择模型；之后核对源文件、剩余磁盘和空闲设备，启动专用容器。等待、运行和失败分开记录；失败或结束自动停止本容器，不终止其他任务。仅保存最终BF16权重，不声称可恢复优化器状态。每步日志提供loss/梯度与时间；先检查真实有限更新再判断训练已启动。

训练固定学习率1e-6、seed42、梯度累积4、最大长度8192、无截断/无packing、梯度检查点、关闭DataLoader锁页内存、仅训练进程禁用THP。不热修改其他训练任务。最终独立评测调用[已有三组评测入口](../joint_training/README.md)，固定新checkpoint及完整覆盖，重新创建结果目录。

## 章节五 问题列表

- 数据源的安全政策与目标测试分布不同，人工标签也可能有噪声；不把格式转换当作质量保证。
- 本轮尚未增加复杂多跳推理或视觉训练。先通过独立验证区分覆盖收益和业务遗忘，不能承诺一次训练解决全部差距。
- 可变字段验证按真实questions数量核对覆盖；不是强行要求每条五字段。
- 依赖先前评测完成后才占用该卡；完整成功状态是验证完成，最终测试仍需另行执行。
- 数据和源码控制的四项CPU检查通过，真实小模型及真实4B执行状态以私有日志为准，准备完成不等于训练完成。

## 章节六 总结

该分支提供一次预先固定的均衡数据干预，保留旧模型与失败证据。待完成独立训练、保存重载、三来源验证及最终业务/通用评测后，再报告收益与未达到的目标。
