# DrugDesign

基于 Python 的药物设计预测脚本开发框架：从活性数据清洗、分子特征化、QSAR 活性预测模型，到类药性过滤和 AutoDock Vina 分子对接，提供可复用的模块、命令行工具和插件机制。

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
    --featurizer morgan maccs --model random_forest --out outputs/model.joblib

# 预测新分子（输出预测值、训练集最大相似度、类药性指标）
drugdesign predict --model outputs/model.joblib --smiles "CCOc1ccccc1NC(=O)C"
drugdesign predict --model outputs/model.joblib --input examples/data/candidates.smi --out outputs/pred.csv

# 分子对接（盒子可由坐标或参考配体确定）
drugdesign dock --receptor target.pdb --smiles "CCOc1ccccc1NC(=O)C" --center 10 12.5 -3 --size 20 20 20
drugdesign dock --receptor target.pdb --smiles "..." --reference-ligand ligand_in_pocket.sdf

# 通过配置文件运行完整流程
drugdesign run examples/config.toml
drugdesign init-config --out my_config.json   # 生成默认配置
```

> `examples/data/demo_activity.csv` 中的分子是真实药物结构，但 pIC50 是**合成的演示数据**，仅用于测试流程，不具备任何生物学意义。

## 作为 Python 库使用

```python
from drugdesign import QSARModel, load_activity_data, clean_dataset, druglikeness_table

data = clean_dataset(load_activity_data("chembl_data.csv", value_col="standard_value"))
model = QSARModel(featurizer=[{"name": "morgan", "radius": 2}, "basic_descriptors"],
                  estimator="random_forest", estimator_params={"n_estimators": 300})
report = model.train_and_evaluate(data["smiles"], data["pIC50"], test_size=0.2, cv_folds=5)
print(report.summary())

model.save("model.joblib")
model = QSARModel.load("model.joblib")
print(model.predict_frame(["CCOc1ccccc1NC(=O)C"]))
```

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
class AtomPairFingerprint(Featurizer):
    ...

@MODELS.register("hist_gb")
def hist_gradient_boosting(random_state=None, **params):
    return HistGradientBoostingRegressor(random_state=random_state, **params)
```

使用插件：

```bash
drugdesign --plugin examples/plugins/custom_featurizer.py train --data ... --featurizer atom_pair --model hist_gb
```

或在配置文件中写入 `plugins = ["plugins/custom_featurizer.py"]`。加载用插件训练的模型时，需要同样传入 `--plugin`。

## 测试

```bash
pytest
```

对接的端到端测试会在检测到 Vina 和 Open Babel 时自动运行，否则跳过。

## 数据资源

- 蛋白结构：[RCSB PDB](https://www.rcsb.org/)
- 化合物活性数据：[ChEMBL](https://www.ebi.ac.uk/chembl/)
- ADMET 预测：[SwissADME](http://www.swissadme.ch/)

## License

本项目以 [GNU GPL v3](LICENSE) 协议发布。
