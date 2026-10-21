# 框架开发说明

DrugDesign 采用“无状态分子特征 + scikit-learn 模型 + 配置工作流”的结构。开发预测脚本时，优先使用公共 API `PipelineConfig`、`run_pipeline`、`train_model`、`predict_candidates` 和 `QSARModel`。

## 数据与模型约定

1. 每个任务只训练一个终点。按靶点、实验类型、测量终点和实验条件组织数据；IC50、Ki、EC50 不能仅因为都能取负对数就混为一个标签。
2. 数据接口输出 `smiles` 和指定目标列。浓度转换采用 `-log10(浓度[M])`。ChEMBL 的 `standard_units` 和 `standard_relation` 自动识别，非精确测量排除。未知单位报错；一般 CSV 仍可使用固定 `unit`。
3. 默认去盐、取最大碎片、中和并生成规范 SMILES；不枚举互变异构体，也不作 pH 依赖的质子化。这些设置随模型保存，训练、推理、适用域和类药性使用同一分子表示。结果保留原始 `smiles`，并输出 `model_smiles`。
4. 回归重复结构按配置汇总；分类重复结构标签不一致会报错。分类目标必须为整数标签；二分类为 0/1 时才提供 `probability_active`。其他分类输出各类 `probability_class_<标签>`，不推断哪一类意味着“活性”。
5. 特征提取失败或非有限目标会被排除，不能用零值替代失败的描述符。预测保留无效输入行并输出 `valid=False`、NaN 预测值。

## 评估与复现

先剔除无效输入，再划分留出集。在留出集的训练部分进行交叉验证，然后用全部有效数据拟合交付模型。缩放器必须位于 estimator 的 sklearn Pipeline 内，不能先对全数据拟合。特征器为无状态接口；需要学习的转换请放入 estimator Pipeline。

- `random`：相同规范结构属于同一分组，避免重复结构跨集合。
- `scaffold`：按不含手性的 Bemis–Murcko 骨架分组；所有无环分子作为同一个空骨架分组。`test_size` 对骨架组比例生效，实际行数比例可能不同；骨架样本不足时显式报错。
- `cv_folds=0`、`test_size=0` 分别关闭交叉验证和留出评估。请求的评估无法执行时不会悄悄跳过。
- JSON 报告区分留出训练行数 `n_train`、测试行数 `n_test` 和最终拟合行数 `n_final_fit`。记录框架/依赖版本、种子、分组策略、数据文件 SHA-256、完整配置及清洗前后行数；无法定义的指标保存为 JSON `null`。

相同种子和数据在兼容依赖版本下可重复运行。持久化模型包含 RDKit 指纹、特征器和 estimator，需要相容的 RDKit/scikit-learn 环境。只加载可信的 joblib 文件，且只运行可信 Python 插件，因为二者能够执行代码。

`max_train_similarity` 是固定 Morgan 指纹的最大训练集 Tanimoto 相似度，不是校准概率或预测置信区间。留出集/CV 用于评估；最终模型包含全部有效训练数据，因此重新预测这些分子属于训练集预测。

## 扩展脚本、特征与模型

可直接修改 `examples/predict_script.py` 的配置。注册新特征器时继承 `Featurizer`，提供 `n_features`、`featurize_one`（一维向量）及 `get_params`。提供的类应可序列化；RDKit generator 等不能序列化的成员通过 `__getstate__` / `__setstate__` 重建。

模型工厂用 `@MODELS.register("my_model")` 注册，接受 `random_state` 和模型参数，返回可被 sklearn `clone` 的回归器、分类器或 Pipeline。CLI `--seed` / 配置 `random_state` 是统一种子入口，不能重复放入 `params`。

文件插件使用文件名和内容哈希生成模块名，同名不同内容可区分，失败导入会撤销注册。跨进程重载时必须先加载同一内容、同一文件名的插件；大型项目建议将插件制作成可安装模块并用模块名加载。不要在导入时运行训练或操作外部服务。

## 候选筛选与对接

批量预测的 `batch_size` 限制每次特征矩阵大小；输入和结果表仍保存在内存中。候选元数据按行保留，和系统结果冲突的列加 `input_` 前缀。类药性计算使用模型分子表示。

对接默认按预测终点排序：p 活性越高越优，IC50/Ki/Kd/EC50 原始值越低越优；0/1 分类优先按 `probability_active` 排序。其他终点通过 `docking.ranking` 明确方向。`top_n` 仅限制实际对接数量，结果中保留全部候选。每个候选记录 `docking_status`、`docking_error` 和亲和力。

Vina/Open Babel/Meeko 为可选工具。CLI Vina 支持 timeout，Python Vina 无法实施该时限，设置时会明确报错。RDKit 添加显式氢和 3D 嵌入不等于受体准备或生理 pH 质子化；真实对接需要独立检查受体、辅因子、水分子、配体状态和盒子。Vina 分数和 QED/PAINS 规则仅为筛选指标，不构成经过验证的 ADMET 或临床有效性预测。
