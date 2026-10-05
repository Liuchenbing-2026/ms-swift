# 任务19：4B多任务决策训练数据准备

## 章节一 模型简介

面向Qwen3.5-4B语言主干的决策监督数据准备，作为联合字段继续训练之后的独立数据覆盖实验。本文不是训练完成报告，不复用评测金标构建训练样本。输出为原始decision-schema记录及仅含输入messages/监督符号的训练记录，可交由任务分支的masked causal CE模板处理。

## 章节二 性能和精度数据

本阶段只执行数据准备、隔离审计和token检查，不训练权重，不报告模型精度或速度收益。记录原始条数、解析排除、冲突去重、词面近似重叠排除、有效条数、监督数量及token长度；最终清单、数据和任何后续性能/精度只存本地。测试集只提供id/text用于排除污染，不提供目标标签。

## 章节三 复现分支和指导

个人MS-SWIFT分支`task19_4b_multitask_data`，基于原4B分支`e815cb65efcb27c00c3cc5524603d6916ab5a112`。新增数据准备、token审计、CPU检查及本文四个文件；不修改框架内部。与运行中的`task19_4b_joint_training`分支隔离。

训练源及许可：

- [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)，revision `6bda777c88d21e5a204703c1ee45597a8fa4f734`，Apache-2.0；使用该revision的`data.json`。仅保留首个用户请求对应唯一合法单工具调用且至少两个候选的子集；不把多工具调用或无工具回复强行改成单工具标签。函数调用只以AST解析，绝不执行。
- [NVIDIA Aegis AI Content Safety Dataset 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)，revision `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c`，[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)；仅使用`train.json`、`validation.json`的人工`prompt_label`。不读取response作为模型输入，不以response_label替代prompt_label，不使用test文件。输出为格式转换后的衍生记录，原始数据和许可权利归原提供者。

下载固定revision到私有SOURCES目录，分别命名为`toolace-data.json`、`aegis-train.json`、`aegis-validation.json`。不要将原始或处理后数据提交代码仓库。WildJailBreak训练集受访问控制，本实现没有绕过访问限制，也不依赖其训练文件。

基础镜像复用`quay.nju.edu.cn/ascend/ms-swift@sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1`，无需导出镜像。准备器/检查用标准Python3.9+即可；token审计使用已固定训练环境中的Transformers及本地4B tokenizer，不加载模型权重、无需NPU。

## 章节四 启动和配置参数

```bash
python -m unittest discover -s . -p 'test_*.py'
python prepare_multitask.py \
  --sources "$SOURCES" --heldout-inputs "$HELDOUT_INPUTS" \
  --output "$PREPARED"
python audit_tokens.py \
  --prepared "$PREPARED" --schema-dir "$DECISION_SCHEMA_DIR" \
  --tokenizer "$CHECKPOINT" --output "$COMPILED" --max-tokens 8192
```

`DECISION_SCHEMA_DIR`提供原4B分支的decision_schema.py。`CHECKPOINT`必须是实际将训练的tokenizer目录，包含已注册的`<decision>`单token；输出目录必须不存在。HELDOUT_INPUTS仅为`{"suite": [{"id": "...", "text": "..."}]}`，由冻结评测输入生成；接口拒绝额外字段。测试目标标签不进入准备器。

先保留Aegis官方validation；ToolACE按规范化请求哈希固定约一成validation。同一请求的不同标签或不同候选工具上下文整组排除；验证先于训练入库，避免相似请求落入两边。规范化为NFKC、casefold和词序列，精确重叠直接排除；较短一侧至少8个word-8-gram且覆盖达到80%时排除词面近似重叠。依次对冻结测试输入、已接受的验证与训练记录做同一筛查。

token审计使用与训练模板一致的chat template、enable_thinking=false；检查标签不进入messages、每字段一个marker和单token监督。超长记录明确排除并保存ID和长度，不截断；保留样本哈希和来源统计。该阶段没有服务启动和压测命令，因为不启动推理或训练进程。独立模型训练/验收沿用任务19相应训练分支，不能把本工具执行成功写成训练完成。

## 章节五 问题列表

- 词面筛查不保证消除语义相似或未知预训练污染；不得据此宣称绝对无污染。
- Aegis与WildJailBreak的安全政策和分布不同，应在独立验证集确定任务映射，不能假定两个标签体系完全等价。
- ToolACE转换只覆盖唯一单工具选择，尚未覆盖多工具、缺参数和无工具决策。
- 本阶段尚未覆盖复杂多跳推理、新闻分类或视觉任务；增加数据不保证官方七项达标。
- 后续须固定多任务采样比例、保留业务回放与联合字段样本；不能直接将所有安全数据堆入训练，或依据测试分数反复挑选数据/模型。

## 章节六 总结

本分支提供独立、可审计的数据准备路径。先完成标签、去重和token审计，再在冻结训练/验证协议下做数据干预实验；旧最优权重不覆盖，最终验收仍需单独执行。
