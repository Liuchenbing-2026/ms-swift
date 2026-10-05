# 4B联合字段继续训练试验

## 章节一 模型简介

从已有Qwen3.5-4B决策checkpoint初始化，继续全参数训练语言主干，冻结视觉与projector。不使用LoRA，不宣称从基础模型独立重训，也不是恢复旧optimizer状态。

当前阶段检验单题训练与联合字段输入的协议差异。沿用冻结案例划分：960个训练案例，每例五字段，共4800个监督决策；验证集120例、600决策。compile_row仅将证据、问题与完整decision骨架写入输入，答案符号仅用于labels。标签在marker位置，因果CE仅由框架移位一次。训练顺序固定随机种子42，禁止再次打乱。

## 章节二 性能和精度数据

首轮固定120次更新、单卡microbatch1、累积4，共480个案例、2400个监督决策，覆盖半轮联合样本。它是固定预算的改进试验，不是官方训练轮数或综合精度验收。

训练前以BF16推理评测已有权重的单题和联合验证准确率；训练结束后导出BF16权重、重载后重复同一验证集并检查完整覆盖及输入哈希。旧权重不覆盖；不读取测试金标做训练、调参、早停或选模。prepare_joint仅对test/calibration执行案例ID隔离审计，不将其编译为训练输入。

最终精度、性能与原始预测只存本地。联合训练调度器只检查validation；独立验收调度器等待固定120步产物保存和重载检查结束，再顺序执行业务专项、扩展49项和官方七项。它不按测试分数选择或再次训练权重。相较旧权重的收益包含额外训练预算影响，缺少等预算单题继续训练对照时，不把收益全部归因于联合格式。本轮没有服务压测。

## 章节三 复现分支和指导

任务分支：个人MS-SWIFT `task19_4b_joint_training`，基于4B分支`intern_decision_4b_npu`的`e815cb65efcb27c00c3cc5524603d6916ab5a112`；交付完整SHA记录在任务书。复用decision_schema.py和decision_plugin.py，新增数据准备、联合/单题验证、CPU AdamW、数值检查、单卡启动器、持久调度器及最终验收调度/CPU检查；不修改训练框架内部或原生算子。

基础镜像：`quay.nju.edu.cn/ascend/ms-swift:v4.5.2-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12`。
registry digest：`sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1`；本地image ID：`sha256:ff2ea9131d5fdf8cf390171dd7af28fd3420646b2e010faf75dc004f58f8dc97`。不用导出镜像。本轮复用镜像内MS-SWIFT `ff5128777dd3dc5538f1a95fbcc29442c327d7d2`，外部插件从本任务分支取得。采用CANN9.1、Torch2.10、torch_npu2.10.0.post2及镜像匹配的Transformers环境，不新增编译依赖。

单卡保存FP32训练参数与梯度；CPUAdamW在CPU持有FP32参数副本、梯度及一二阶矩，更新后同步回NPU，避免Adam状态占用NPU容量。复用CPU梯度缓冲，避免每步重新分配大块内存；启动器启用`DECISION_DISABLE_THP=1`，通过Linux prctl仅关闭当前训练进程的透明大页，不修改宿主全局策略。未设置此变量时保持系统默认。BF16用于计算；训练结束才转换模型为BF16推理产物，节省磁盘空间。仅保存最终模型，**不保存optimizer/RNG/调度器恢复点**，中断后不可宣称精确接续本轮。该短试验明确接受这一恢复限制；旧完整断点保持只读。

五步合成FP32更新检查涵盖分组weight decay、LR变化、无梯度参数、状态字典往返及梯度缓冲地址复用，要求NPU参数回传结果与CPU标准AdamW逐元素一致。另需同架构小模型实际训练、保存与重载通过，不能以CPU检查替代真实NPU训练。

## 章节四 启动和配置参数

先准备旧实验的prepared数据目录、完整checkpoint及任务代码。数据准备、全量token长度/marker检查通过后才能训练；所有训练输入必须不超过8192，记录实际有效条数，不能依赖delete静默丢弃样本。

宿主容器示例（确认所选卡空闲；旧权重和数据只读挂载）：

```bash
export DEVICE_INDEX=5
export TRAIN_IMAGE=quay.nju.edu.cn/ascend/ms-swift@sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1
docker run -d --name decision-joint-training --network none --privileged --shm-size 8g   -e ASCEND_RT_VISIBLE_DEVICES="$DEVICE_INDEX" -e PYTHONPATH=/workspace   -e OMP_NUM_THREADS=4 -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1   -v "$EXPERIMENT_DIR:/workspace" -v "$PRIOR_DIR:/prior:ro"   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro   -v /usr/local/dcmi:/usr/local/dcmi:ro   -v /etc/ascend_install.info:/etc/ascend_install.info:ro   --entrypoint bash "$TRAIN_IMAGE" -c   'for i in 0 1 2 3 4 5 6 7; do if [ "$i" != "$ASCEND_RT_VISIBLE_DEVICES" ]; then rm -f /dev/davinci$i; fi; done; exec sleep infinity'
```

不要挂载宿主/dev后删除设备。保留镜像的CANN运行环境；运行前确认仅一张NPU可见。程序和数据占用独立目录；启动后不覆盖运行脚本。容器内准备：

```bash
cd /workspace
python prepare_joint.py --prepared /prior/prepared --output /workspace/data
python check_cpu_adamw.py
# 核对data/manifest.json及token长度；冻结train/validation/test案例身份。
export TRAIN_WORKSPACE=/workspace
export MODEL_PATH=/prior/outputs/your-checkpoint
export OUTPUT_DIR=/workspace/experiment
export JOINT_VALIDATION_DATA=/prior/prepared/validation.jsonl
export TOTAL_STEPS=120 ACCUMULATION=4
bash /workspace/run_joint.sh
```

真实任务采用宿主持久调度器，顺序执行BF16基线、训练、BF16重载验证。私有plan示例：

```json
{
  "host_workspace": "/local/experiment",
  "container": "decision-joint-training",
  "checkpoint": "/prior/outputs/your-checkpoint",
  "validation_data": "/prior/prepared/validation.jsonl",
  "steps": 120
}
```

```bash
mkdir -p "$EXPERIMENT_DIR/launches"
cp "$EXPERIMENT_DIR/run_joint.sh" "$EXPERIMENT_DIR/launches/real.sh"
python3 run_pipeline.py --plan /local/private-plan.json
```

需要至少14GiB可用空间容纳最终BF16产物并留余量；不得占用其他训练已规划的保存空间。宿主RAM还需容纳CPU参数副本、梯度与Adam状态，按实际参数量核算。TRITON_CACHE_DIR只影响编译缓存，可设独立目录；复用相同环境的缓存不等于复用权重或训练结果。

独立精度复测与训练计时：

```bash
python joint_validation.py --checkpoint /workspace/experiment/checkpoint-120   --data /prior/prepared/validation.jsonl --output /workspace/new-validation.json
python - <<'END'
import json
from pathlib import Path
rows = [json.loads(x) for x in Path('/workspace/experiment/logging.jsonl').read_text().splitlines()]
for row in rows:
    if 'loss' in row:
        print({k: row.get(k) for k in ('global_step/max_steps', 'elapsed_time', 'train_speed(s/it)', 'loss', 'grad_norm')})
END
```

首步可能含编译，不用它估计稳定吞吐。完整端到端还包括初始验证、导出和重载验证。输出为pipeline-state.json、阶段日志、experiment/progress.json、BF16 checkpoint、validation-after.json、reload-validation.json及comparison.json。程序仅停止plan指定的本实验专用容器，不管理其他训练任务；不重复启动已有pipeline-state目录。

### 固定120步产物的完整验收

`run_acceptance.py --plan "$PRIVATE_ACCEPTANCE_PLAN"`在宿主等待训练状态`validation_complete`且训练容器已停止，核对120步BF16导出、保存权重SHA256，才依次执行下面三组命令。业务与49项使用训练镜像；七项复用此前固定的NPU评测环境，镜像为`quay.nju.edu.cn/ascend/vllm-ascend@sha256:2c8aac4281e56953764a7fa60773cec734342d32d9d6d76c0c8f3a8660a64b85`，image ID `sha256:4f7bc48083e5efc9f506bba85c1de16361e3f40cf5dc0220ae54712863ed7d18`，实际Torch2.10.0、torch_npu2.10.0.post4、Transformers5.14.1。这里只调用原生NPU backend。推理核心沿用个人Intern-Decision `intern_decision`的`c025a7d1a516d792b6f32df0e14cbf7c1471d41c`；各源文件及评测数据hash另由私有plan固定，逐阶段核验。

```bash
# 在原训练环境内；EVAL_CODE提供原4B分支的评测脚本与decision_schema。
python "$EVAL_CODE/evaluate_decision.py" --checkpoint "$FINAL_CHECKPOINT" \
  --data "$PREPARED_TEST" --output "$RESULTS/business.json" --batch-size 4
python "$EVAL_CODE/evaluate_laya_suite.py" --checkpoint "$FINAL_CHECKPOINT" \
  --dataset "$LAYA_SUITES" --typed-data "$PREPARED_TEST" \
  --typed-result "$RESULTS/business.json" --output "$RESULTS/broad49" --batch-size 8
# 在固定独立NPU评测环境和Intern-Decision源码目录中：
python -m src.eval.jev --backend npu --checkpoint "$FINAL_CHECKPOINT" \
  --suite --batch-size 8 --output "$RESULTS/official7"
python -m unittest discover -s . -p 'check_acceptance.py'
```

私有plan需填写`host_workspace`、`training_container`、`checkpoint_step:120`、`wait_seconds`、`pinned_files`（源码及数据绝对路径→SHA256）、`free_device_check`（只读确认目标设备空闲的命令数组），以及顺序为business/broad49/official7的`stages`。每阶段包含`name/container/command/result/timeout_seconds`；command为实际docker exec参数数组，result为对应JSON的宿主路径。容器须预先创建并停止，使用同一张授权设备，将checkpoint和评测输入只读挂载、结果写入独立目录；确认容器私有/dev后仅保留该设备。不得写入主机/IP/凭据至公开plan。

调度器不启动新训练、不选择测试最优模型。设备被占用时失败留证，不终止其他进程；每阶段最多4小时，结束只停止自身启动的专用容器。训练失败、验证未完成、源码/数据漂移或覆盖数不符均不得标记评测完成。固定覆盖分别为2000决策、49项17416有效决策、七项10751行/12351决策；49项原协议的500个超候选上限决策单列排除，不混入有效分母。`final-evaluation/state.json`记录实际状态；完成仅表示固定checkpoint评测结束，不代表达到目标精度。

## 章节五 问题列表

- 联合字段继续训练是否提高验证准确率，须以本轮真实结果判断。
- 数据仍仅四类业务，尚未补齐复杂推理、安全判断和工具选择覆盖，不能保证七项达标。
- 实际首轮在DataLoader锁页内存分配处失败，发生于首个前向之前、零次更新。启动器显式设置`--dataloader_pin_memory false`；这只改变主机数据传输配置，不改变样本、标签或优化器数学定义。保留失败日志，在独立输出中重试并重新验证。
- 已完成真实更新后，为减少CPU分配开销引入缓冲复用和进程级THP控制；重新执行数值检查后从原始权重与固定种子重启短实验。因为未保存optimizer断点，不称作从中断步数恢复；各次日志单独保留。内存微测收益不等于整网训练收益，须检查后续真实步耗时。
- CPU优化器更新减少NPU显存，但引入CPU计算与传输；真实速度待测，不能套用小模型速度。
- 不保留完整optimizer断点；BF16成品可用于推理或新一轮warm start，不是完整状态恢复。
- 导出前后使用相同验证数据和输入哈希，重载预测须一致；失败保留证据，不标记验收完成。

## 章节六 总结

这是4B精度提升的联合字段训练阶段，独立分支、独立容器和输出，旧最优权重及测试集保留。实测范围与进展更新到任务书，未完成训练或综合验收时不得宣称模型已达标。

## Export precision recovery

A completed fixed-step run can fail the in-memory versus reload check because a whole-model `to(bfloat16)` also rounds nonpersistent FP32 rotary frequency buffers. Canonical loading regenerates those buffers in FP32. The export helper now preserves buffer values/dtypes while converting parameters; it does not change the training objective or optimizer. Existing saved weights are preserved.

Run an independent diagnostic in the same pinned training environment:

```bash
python check_export_runtime.py --checkpoint "$FIXED_CHECKPOINT" \
  --data "$VALIDATION_JSONL" --output "$NEW_EXPORT_AUDIT_DIR"
python check_acceptance.py
python run_acceptance.py --plan "$PRIVATE_ACCEPTANCE_PLAN"
```

The diagnostic checks every saved tensor against the loaded model, binds all checkpoint files by SHA256, repeats canonical validation, and compares explicit whole-model cast and autocast paths. Results are private artifacts. In a recovery plan, set `export_runtime_audit` to this completed audit directory and use a fresh `output_directory`. The acceptance gate requires exact canonical prediction/probability repeatability, exact reproduction of the prior in-memory output by the buffer cast, unchanged checkpoint hashes, and only nonpersistent FP32-to-BF16 inverse-frequency buffer changes. It retains the original pipeline failure and explicitly records canonical reload as the assessment runtime. This path rejects other pipeline errors; it does not weaken the original zero-change check or rerun training.

Keep the predeclared checkpoint step, three test suites and all pinned evaluator/data hashes unchanged. Stop the owned diagnostic container after its process exits, then let the acceptance driver acquire the reserved free device. No test score selects a model or training setting. Four CPU acceptance tests cover the ordinary path, failure/device isolation and recovery rejection after weight or prediction changes. New export-helper validation is separate from the previously completed training run.
