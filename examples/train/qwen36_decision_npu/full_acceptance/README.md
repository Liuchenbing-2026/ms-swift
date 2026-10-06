# 章节一 模型简介

本目录对固定完成的 Qwen3.6-35B-A3B 决策训练断点执行只读综合验收。通过原SWIFT/FSDP环境恢复权重，在首个优化器更新前进入评测；不训练、不改变保存点，不以测试成绩选择模型。

# 章节二 性能和精度数据

官方七项包含10,751条记录、12,351次决策，报告七项准确率等权平均；Typed Decision保留一条输入的所有字段。扩展49项包含17,416次决策，每个问题单独输入；预先排除500条超过62候选的记录，并保存排除清单。两套数据不合并成一个总分。最终数值、权重及逐题预测仅本地。

官方七项每rank batch8，扩展集每rank batch1。BF16/FSDP算术与其他运行时存在差别，不宣称逐位等价。本流程只验收准确率，不替代温度校准、概率质量或服务性能测试。

# 章节三 复现分支和指导

个人MS-SWIFT分支 `task19_qwen36_full_acceptance`，基于 `task19_qwen36_recompute_trial` 固定提交 `16b7399199b813cf5acf794ea3563ecaa1641466`。使用父分支的训练插件、完整断点读取和设备空闲检查。

基础镜像：`quay.nju.edu.cn/ascend/ms-swift:v4.5.2-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12`，digest `sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1`。沿用已核验训练依赖，不导出镜像、不修改C++，无需额外编译框架。

# 章节四 启动和配置参数

提供本地固定官方数据目录和历史扩展数据，编译器及训练数据目录来自原训练分支。准备流程保持字段/选项次序，目标标签单独保存，禁止进入消息：

```bash
PYTHONPATH="$TRAIN_WORKSPACE" python prepare_acceptance.py \
  --official-root "$OFFICIAL_ACCURACY_ROOT" --broad "$BROAD_SUITE_JSON" \
  --output "$PREPARED_DATA"
PYTHONPATH="$TRAIN_WORKSPACE" python check_inputs.py
```

以生成的manifest构建评测JSON：`scope`为`official7_and_broad49`，`training_data`为原训练JSONL，`suites`各项包含`name`、`data`、`sha256`、`expected_rows`、`expected_decisions`、`batch_size`。路径均是容器内路径。每个suite写入分rank原始结果，合并时检查完整覆盖、样本身份、有限候选分数，并记录输入哈希。

单次启动（先确认四卡空闲、原训练及保存已完成）：

```bash
export TRAIN_WORKSPACE MODEL_PATH PROBE_PLUGIN RESUME_FROM OUTPUT_DIR FSDP_CONFIG
export FULL_EVAL_PLUGIN="$ACCEPTANCE_CODE/checkpoint_eval.py"
export DECISION_EVAL_CONFIG="$FULL_CONFIG_JSON" DECISION_EVAL_ONLY=1
export MICROBATCH=4 DECISION_PROBE_UPDATES=0 SAVE_STRATEGY=no EVAL_STRATEGY=no
bash "$ACCEPTANCE_CODE/run_evaluation.sh"
```

先小模型执行检查，再完整35B验收的宿主调度命令：

```bash
PYTHONPATH="$THROUGHPUT_CONTROLLER_CODE" python run_acceptance.py --plan "$PLAN_JSON"
```

计划沿用父分支最终评测中的容器、设备/进程身份、完整断点、源码/data SHA256、环境及全新输出目录字段；增加`model`、`smoke_model`、`smoke_config`、`full_config`，设置`container_launcher`为本目录启动脚本，`environment.FULL_EVAL_PLUGIN`指向本目录回调。小模型无resume，覆盖单字段、多字段、choice/noul/score；只有零更新、完整输出通过才加载固定600步断点。执行前后确认断点文件大小和mtime不变。进程/设备被占用、数据变化或任一阶段失败均停止，不终止其他工作负载。

# 章节五 问题列表

业务专项达标不等于七项平均达标。小模型只验证执行路径，不能证明35B质量。扩展集所有有效问题重新推理，不复用其他模型的预测。训练/评测标签隔离同时检查案例ID及完全相同的已编译消息；这不声称覆盖所有语义重复。图片输入不在此文本验收范围。

# 章节六 总结

完整验收完成后才报告结果，状态文件与分套覆盖共同构成证据。Twinkle真实35B训练是独立后续工作，不能用SWIFT结果代替。

官方ToolACE文件存在重复原始ID，按物理行身份保留全部310条；输出同时保留原始ID和独立行身份，不去重、不改变输入或标签。案例隔离检查仍使用原始ID。首次准备因将原始ID当唯一键而拒绝执行，修正后重新准备；失败文件保留，仅用于诊断。
