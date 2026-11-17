# PS3 — Feature engineering & modelling：A–E 分工

- [正式题单](https://languid-citipati-d0d.notion.site/PS-3-Feature-engineering-modelling-2ac32dea82c1814dad25e6d3c2af3b4e) · [团队仓库](https://github.com/starries-creator/ps3_claims) · [安装与运行](README.md)
- [老师指定的 glum 教程](https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html)
- 提交截止：**2026 年 11 月 20 日（周五）中午 12 点**。一名组员向 Vahan（vg400@cam.ac.uk）发送公开 GitHub 仓库链接。
- **A–E 待认领，不对应既定姓名。** 三道题共用本仓库；以下是小组协作安排，题目要求以正式题单为准。

## 共同约定

先固定 Ex1 的划分接口、主键和 training fraction，所有模型使用同一 `sample` 列。接口约定为 `create_sample_column(df, id_columns, training_frac=0.8)`，支持单列或多列，返回带 `sample`（`train` / `test`）的 dataframe；A 同时保留现有 `create_sample_split` 入口供脚手架使用，明确参数对应关系。

Ex3 沿用脚手架的 pure premium 目标、Exposure 权重及 Tweedie power=1.5，统一报告 exposure 加权、按总权重归一的 train/test deviance。如需变更，先同步全组，重跑所有对比模型。调参只使用训练部分的交叉验证；预处理在各训练折内拟合，测试集不用于选择超参数。

每个模型保存小型指标文件 `results/<model>_metrics.json`，包含模型名、数据划分约定、train/test 行数及 deviance、目标/权重/power、参数与选择方法；C 汇总为 `results/metrics.csv`。各自保存必要图表及解释，不把大型逐保单预测、原始数据或模型文件推入 Git。

## Ex1 — Create fixed train/test split

[题目](https://languid-citipati-d0d.notion.site/Ex-1-Create-fixed-train-test-split-2ac32dea82c181e082a6c7b76a79f71a)

**目标：** 按一个或多个主键进行稳定划分，使同一观测始终进入同一集合。

**跨 example 关系：** 可立即开始；完成后为 Ex3 全部模型提供固定划分。Ex2 可同时独立进行。Ex3 可先准备数据、模型代码和评估接口，正式比较须等划分合并。

| 阶段／顺序 | 任务及题号 | 负责人 | 开始条件 | 交付物 |
| --- | --- | --- | --- | --- |
| 1，可立即开始 | Ex1 全部：实现单／多主键、training fraction 和稳定分桶；衔接题单与脚手架的函数命名 | A | README 环境可用 | `ps3/data/_sample_split.py`、公开导出；保留兼容入口 |
| 1，同一工作包 | 验证重复调用、行重排及新增观测时已有观测的归属不变；多主键、比例和边界输入；说明分桶选择及比例含义 | A | 与实现同步推进 | `tests/data/test_sample_split.py`，接口和限制说明 |
| 2，接入 Ex3 | Ex3 第 2 项：让统一数据入口使用正式划分，记录主键及 fraction | A | Ex1 合并，C 的数据入口确定 | 可供所有模型复用的 split 调用；数据划分说明 |

## Ex2 — Write your own Transformer

[题目](https://languid-citipati-d0d.notion.site/Ex-2-Write-your-own-Transformer-2ac32dea82c181679356c02e5fc4436e)

**目标：** 实现并测试 sklearn 兼容的 `Winsorizer`，供后续 model pipeline 使用。

**跨 example 关系：** 与 Ex1 并行。单元测试不依赖真实保险数据；B 可先与 D/E 确定输入形状及返回格式，合并后供 Ex3 接入。

| 阶段／顺序 | 任务及题号 | 负责人 | 开始条件 | 交付物 |
| --- | --- | --- | --- | --- |
| 1，可立即开始 | Ex2 实现：`__init__(lower_quantile, upper_quantile)`、`fit` 保存 `lower_quantile_` / `upper_quantile_`、`transform` 裁剪 | B | README 环境可用 | 完成 `ps3/preprocessing/_winsorizer.py` |
| 1，同一工作包 | Ex2 测试：完成老师留空的 pytest；检查不同分位数、fit/transform 行为、sklearn 接口及 pipeline 兼容性 | B | 与实现同步推进 | `tests/preprocessing/test_winsorizer.py`，测试命令及必要解释 |
| 2，可先约接口 | 负责自定义 Winsorizer 的 Ex3 接入：与 D/E 确认在哪个 numeric 分支使用、训练折拟合规则和特征输出接口 | B | 接口可先约定；正式联调等 Ex2 与对应 pipeline 就绪 | 可复用预处理组件及使用说明；由 D/E 在各自模型入口调用 |

## Ex3 — Improve the model

[题目](https://languid-citipati-d0d.notion.site/Ex-3-Improve-the-model-2ac32dea82c18123a6e1e926a62721a9)

**目标：** 在固定划分与一致评估口径下，比较 Tweedie benchmark、改进的参数模型和调优后的 LightGBM。

**跨 example 关系：** 正式运行依赖 Ex1；使用自定义 Winsorizer 的 pipeline 还依赖 Ex2。C 可立即确认数据入口；A/D/E 可提前写独立模型模块，最终接入统一数据及评估函数。

| 阶段／顺序 | 任务及题号 | 负责人 | 开始条件 | 交付物 |
| --- | --- | --- | --- | --- |
| 1，可立即开始 | 第 1 项：用 `load_transform` 取得数据，确认所需列及数据获取方式；约定公共数据／评估接口 | C | 环境可用、OpenML 可访问 | 主脚本的数据入口，公共接口说明；不另改老师的数据转换口径 |
| 1，可提前准备；2，正式运行 | 第 3 项：运行初始 Tweedie benchmark，回答为何除以 Exposure；提供统一评估及结果输出 | C | 可先读教程／准备代码；正式结果等 A 的划分 | `analyses/ps3_script.py` 的公共入口、benchmark 指标与解释 |
| 1，可提前准备；2，正式运行 | 第 4 项：对 **BonusMalus 和 VehPower** 用 polynomial 或 spline 增加灵活性；比较 deviance，解释如何发现过拟合 | D | 可先准备变换；正式对比等公共数据／评估接口和 Ex1；Winsorizer 联调等 B | 独立参数模型模块、指标、必要图表和解释 |
| 2，与参数模型并行 | 第 5 项的初始 LGBM：确定适合任务的初始模型与预测入口，交付给 E 调参 | A | Ex1 及公共接口确定；可提前准备模型接口 | `analyses/lgbm_baseline.py`、初始模型指标、参数和运行方法 |
| 1，可先准备搜索；3，调优与解释 | 第 5 项的调优：在训练集内合理选择参数与控制过拟合，和初始 LGBM／GLM 比较，说明选择依据 | E | 可先设计搜索／评估；正式运行等 A 的初始 LGBM 和公共接口；预处理与 B 联调 | `analyses/lgbm_tuning.py`、CV 结果、最终参数、调优模型指标与解释 |
| 4，汇总分析成果 | 汇总 benchmark、参数模型、初始及调优 LGBM；串联可运行入口，整理对比结论及脚手架中的 Lorenz/Gini 图 | C | 各模型模块及指标就绪；汇总模板可先准备 | `results/metrics.csv`、小型图表、统一结果说明、README 最终运行方法 |

**题单与脚手架的差异：** 第 4 项正式题单要求 `BonusMalus` 和 `VehPower`，脚本注释却写了 `BonusMalus` 和 `Density`，且初始代码把 `VehPower` 当 categorical。D 需明确调整列的归属，避免同一特征被重复输入；Density 的额外实验可自行说明，但不能替代题单指定特征。

**文件边界：** C 主改 `analyses/ps3_script.py` 及最终结果说明；D 主改独立参数模型模块，A/E 分别主改初始 LGBM／调优模块；B 主改 `ps3/preprocessing/` 及对应测试。公共接口先在 PR 中写清函数参数、返回值及 sample weight 传递方式，主脚本再调用。修改他人负责的共享文件前先沟通。

**选做单列：** 第 6 项 frequency + severity LGBM 及对比为选做，不计入上述必做负担。必做完成后再认领，届时指定一位主负责人并沿用相同划分和评估口径。

五个角色均包含实现、对自己结果的必要测试和解释；A 的稳定划分完成后转做初始 LGBM，B 完成 Transformer 后做预处理接入，C 的负担包含公共评估与最终分析汇总，D/E 各负责较重的模型实验。实际认领后，承担 PR 审核／合并的同学还需计入这部分时间，再核对其工作量；当前不假定该同学是 A。

大家先按 README 配好环境并在自己的 clone 安装 hook，使用个人分支提交 PR，写明完成的任务和必要验证。PR 审核／合并通常由仓库负责人统筹，不另派常规 reviewer 或 merger。前置成果合并后同步 `main`，再接入下游工作；A–E 的姓名和账号对应关系待认领后补充。
