# DrugDesign

基于 Python 的药物设计预测脚本开发框架：从活性数据清洗、分子特征化、QSAR 活性预测模型，到类药性过滤和 AutoDock Vina 分子对接，提供可复用的模块、命令行工具和插件机制。

0.3 版本统一训练与推理的分子标准化，提供防重复结构泄漏的随机/骨架划分、严格配置校验、批量预测和可追溯 JSON 训练报告。开发指南见 [框架结构与扩展约定](docs/architecture.md)，可直接修改 [预测脚本模板](examples/predict_script.py) 开始开发。

```
活性数据 (ChEMBL CSV) ──► 清洗/标准化 ──► 特征化 ──► QSAR 模型训练与评估 ──► 保存模型
                                                                     │
候选分子 (.smi/.csv) ─────────────────────────────────────────────────┴──► 活性预测
                                                                           + 适用域相似度
                                                                           + 类药性 (Lipinski/Veber/QED/PAINS)
                                                                           + 分子对接 (可选)
```

## 安装

```bash
pip install -e .            # 安装框架及命令行工具 `drugdesign`
pip install -e ".[dev]"     # 额外安装 pytest
```

分子对接为可选功能，需要另外安装：

- [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina/releases)（`vina` 可执行文件，或 `pip install vina`）
- 配体准备：[Meeko](https://github.com/forlilab/Meeko)（`pip install meeko`，推荐）或 [Open Babel](https://openbabel.org/)
- 受体准备：ADFR suite 的 `prepare_receptor` 或 Open Babel（`obabel`）

## 快速开始

```bash
# 查看所有已注册的特征化方法和模型
drugdesign list

# 训练模型（Morgan 指纹 + MACCS 组合特征，随机森林）
drugdesign train --data examples/data/demo_activity.csv \
    --featurizer morgan maccs --model random_forest --split scaffold --out outputs/model.joblib
# 同时生成 outputs/model.report.json；可用 --report 指定路径

# 预测新分子（输出预测值、训练集最大相似度、类药性指标）
drugdesign predict --model outputs/model.joblib --smiles "CCOc1ccccc1NC(=O)C"
drugdesign predict --model outputs/model.joblib --input examples/data/candidates.smi --out outputs/pred.csv

# 分子对接（盒子可由坐标或参考配体确定）
drugdesign dock --receptor target.pdb --smiles "CCOc1ccccc1NC(=O)C" --center 10 12.5 -3 --size 20 20 20
drugdesign dock --receptor target.pdb --smiles "..." --reference-ligand ligand_in_pocket.sdf

# 通过配置文件运行完整流程
drugdesign run examples/config.toml
drugdesign init-config --out my_config.json   # 生成默认配置

# 自定义 Python 脚本入口
python examples/predict_script.py
```

Windows PowerShell 下将上面的多行训练命令写为一行：

```powershell
drugdesign train --data examples/data/demo_activity.csv --featurizer morgan maccs --model random_forest --split scaffold --out outputs/model.joblib
```

> `examples/data/demo_activity.csv` 中的分子是真实药物结构，但 pIC50 是**合成的演示数据**，仅用于测试流程，不具备任何生物学意义。

## 作为 Python 库使用

```python
from drugdesign import QSARModel, load_activity_data, clean_dataset, druglikeness_table

data = clean_dataset(load_activity_data("chembl_data.csv", value_col="standard_value"))
model = QSARModel(
    featurizer=[{"name": "morgan", "radius": 2}, "basic_descriptors"],
    estimator="random_forest",
    estimator_params={"n_estimators": 300},
)
report = model.train_and_evaluate(
    data["smiles"], data["pIC50"], test_size=0.2, cv_folds=5, split_strategy="scaffold"
)
print(report.summary())
report.save("training_report.json")

model.save("model.joblib")
model = QSARModel.load("model.joblib")
print(model.predict_frame(["CCOc1ccccc1NC(=O)C"]))
```

`standardize=True` 默认去盐、中和并规范化 SMILES，设置随模型保存。预测保留原始输入，另输出 `model_smiles`。`predict_candidates(..., batch_size=1024)` 可保留候选表元数据并按批提取特征。无效分子保留其行，预测为 NaN。

评估先清洗再划分；CV 仅使用留出训练部分，最后用全部有效分子重拟合交付模型。`split_strategy="scaffold"` 按 Murcko 骨架分组，所有无环分子属于同一组；不足以划分时明确报错，可调整比例/折数，或显式使用随机划分。`cv_folds=0` 和 `test_size=0` 分别关闭相应评估。

ChEMBL 导入自动读取 `standard_units` 的逐行单位，并仅保留 `standard_relation="="` 的精确测量。可用 `unit_col`、`relation_col` 指定其他列名。使用同一靶点、同一终点和可比实验条件的数据训练；未知单位及冲突分类标签报错。分类需显式提供整数标签，不能直接将连续 pIC50 当成类别。

## 模块说明

| 模块 | 功能 |
| --- | --- |
| `drugdesign.data` | 读取 CSV/TSV/ChEMBL 导出文件；IC50/Ki (nM, µM…) → pIC50；SMILES 标准化（去盐、中和电荷）；重复化合物合并 |
| `drugdesign.features` | `morgan`、`maccs`、`rdkit_descriptors`、`basic_descriptors` 特征化器，可组合使用 |
| `drugdesign.models` | `QSARModel`：直接以 SMILES 为输入的训练/评估（留出集 + K 折交叉验证）/预测/保存；内置回归与分类模型 |
| `drugdesign.filters` | Lipinski、Veber、QED、PAINS 结构警报 |
| `drugdesign.docking` | 受体/配体 PDBQT 准备、对接盒子、Vina 运行与结果解析 |
| `drugdesign.pipeline` | 端到端工作流与插件加载 |
| `drugdesign.config` | TOML/JSON 配置（相对路径以配置文件所在目录为基准） |
| `drugdesign.cli` | 命令行入口 |

内置模型：`random_forest`、`extra_trees`、`gradient_boosting`、`svr`、`ridge`、`knn`、`mlp`（回归），`random_forest_classifier`、`logistic_regression`（分类）。

## 扩展：开发自己的预测脚本

特征化器和模型都通过注册表管理，新增方法无需修改框架代码。参考 `examples/plugins/custom_featurizer.py`：

```python
from drugdesign.features import FEATURIZERS, Featurizer
from drugdesign.models import MODELS


@FEATURIZERS.register("atom_pair")
class AtomPairFingerprint(Featurizer): ...


@MODELS.register("hist_gb")
def hist_gradient_boosting(random_state=None, **params):
    return HistGradientBoostingRegressor(random_state=random_state, **params)
```

使用插件：

```bash
drugdesign --plugin examples/plugins/custom_featurizer.py train --data ... --featurizer atom_pair --model hist_gb
```

或在配置文件中写入 `plugins = ["plugins/custom_featurizer.py"]`。加载用插件训练的模型时，需要同样传入 `--plugin`。

文件插件通过内容哈希区分模块；加载模型需要保留插件文件名与内容。大型插件可制作成可安装 Python 模块。模型使用 joblib/pickle，只加载可信模型和插件。旧版内置模型保留原来的推理方式；建议重新训练以采用统一标准化及新评估流程，旧版文件插件模型也建议重新训练。

## 测试

```bash
pytest
python -m ruff check .
python -m ruff format --check .
python -m build
```

对接的端到端测试会在检测到 Vina 和 Open Babel 时自动运行，否则跳过。

仓库包含 Windows/Linux 与 Python 3.9/3.11/3.14 的 GitHub Actions 测试配置。真实 Vina 对接依赖另外安装，`timeout` 仅适用于 Vina CLI。受体准备、pH 状态、构象及对接盒子需要按具体体系检查；RDKit 添加氢不等同于生理 pH 质子化。

示例模型验证的是软件流程；演示数据不能用于判断实际活性。最大 Tanimoto 相似度是适用域参考，QED/PAINS 和 Vina 分数是筛选指标，框架目前未包含经过验证的 ADMET 或临床预测模型。

## 数据资源

- 蛋白结构：[RCSB PDB](https://www.rcsb.org/)
- 化合物活性数据：[ChEMBL](https://www.ebi.ac.uk/chembl/)
- ADMET 预测：[SwissADME](http://www.swissadme.ch/)

## License

本项目以 [GNU GPL v3](LICENSE) 协议发布。
