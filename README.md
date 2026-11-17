# PS3 — Claims modelling

本仓库供小组共同完成 PS3 的三个 exercise。上游为
[cambridge-FDS/ps3_claims](https://github.com/cambridge-FDS/ps3_claims)。

- [正式题单](https://languid-citipati-d0d.notion.site/PS-3-Feature-engineering-modelling-2ac32dea82c1814dad25e6d3c2af3b4e)
- [A–E 分工、前置条件与共享接口](TEAM_ASSIGNMENTS.md)（待认领）
- [老师指定的 glum 教程](https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html)

截止：2026 年 11 月 20 日中午 12 点；由一名组员向 Vahan
（vg400@cam.ac.uk）发送公开仓库链接。

## 安装环境

沿用老师的 `environment.yml`，主要入口是 Conda，不需要另建 Pixi 或 uv 环境。
先安装 [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
或已有的 Mamba，再执行：

```bash
git clone https://github.com/starries-creator/ps3_claims.git
cd ps3_claims
conda env create -f environment.yml
conda activate ps3
python -m pip install --no-build-isolation -e .
pre-commit install
```

使用 Mamba 时可将创建命令替换为 `mamba env create -f environment.yml`。
如果本机已有同名 `ps3` 环境，请先查看其用途；新建时可用
`conda env create -f environment.yml -n ps3-coursework`，随后激活该名字。
不要删除或覆盖其他课程环境。

`environment.yml` 包含数据、模型、绘图、ipykernel 和 pre-commit 所需依赖。
它沿用上游的版本范围，没有锁定全部依赖版本；不同平台由 Conda 解析兼容依赖。
在 VS Code 中为 `.py` 的 `# %%` interactive cells 选择这个环境的 Python／kernel。

## 安装后的基本入口

```bash
python -c "from ps3.data import load_transform, create_sample_split; from ps3.preprocessing import Winsorizer; print('imports OK')"
python -m pytest tests/test_core.py
pre-commit run --all-files
```

每位组员须在**自己的 clone** 执行 `pre-commit install`，该本机 hook 不会随 clone
自动安装。当前 hook 检查 YAML、TOML、Python 语法、冲突标记和新增大文件，revision
固定；它不会删除 notebook 输出。检查通过不代表题目已经完成。

## 数据和作业入口

`ps3.data.load_transform()` 直接读取 OpenML 的两个公开数据文件，无需账号：

- [freMTPL2freq](https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff)
- [freMTPL2sev](https://www.openml.org/data/get_csv/20649149/freMTPL2sev.arff)

```python
from ps3.data import load_transform
df = load_transform()
```

首次读取需要联网；现有 loader 每次调用都会读取远端，没有本地缓存。
`data/` 仅供个人下载／缓存时使用，当前 loader 不自动从该目录读取。不要把原始数据、
虚拟环境、模型文件或大型逐保单预测提交到仓库；小型指标和必要图表可放在 `results/`。

主要分析入口为：

```bash
python analyses/ps3_script.py
```

**该命令需在作业实现完成后运行。** 当前脚手架尚未创建 `sample`，Winsorizer 的
方法留空，pipeline 和搜索仍有 TODO；现在全量执行会在未完成部分停止。可以先在
VS Code 逐 cell 运行 imports／数据读取，再按分工接入后续阶段。

Ex2 的题目测试入口为：

```bash
python -m pytest tests/preprocessing/test_winsorizer.py
```

其中 `assert False` 是老师留下的待完成测试，B 实现 Transformer 时一并完成；当前
不能用全套 pytest 作为环境是否安装成功的判断标准。

## 目录与协作

| 位置 | 用途 |
| --- | --- |
| `ps3/data/` | 老师的数据 loader 与 Ex1 稳定划分 |
| `ps3/preprocessing/` | Ex2 Winsorizer 及共享预处理 |
| `analyses/ps3_script.py` | 老师的分析脚手架，后续串联模型 |
| `tests/` | 基础测试与待完成的题目测试 |
| `TEAM_ASSIGNMENTS.md` | 各 example 的 A–E 责任、依赖和交付物 |

Ex1 的正式题名使用 `create_sample_column`，现有仓库导出 `create_sample_split`，
由 A 在实现时衔接并保留兼容入口。Ex3 的特征要求与脚本注释也有差异，详见分工文件。

从最新 `main` 建自己的分支并提交 PR；前置改动合并后同步 `main`。分开模型模块和
输出，减少同时编辑主脚本。预处理及调参只使用训练数据，所有模型保留一致的数据
划分、目标、权重及评估口径。完成各自工作后更新运行方法、结果及必要解释。
