# skd-backtest

面向指数增强研究的日频 Python 回测框架。已实现完整回测链路：
市场数据、独立异步推理、运行缓存、组合构建、交易、费用、估值、标签、评价及审计输出。
组件通过固定缓存协议交换非市场数据，代码按组件独占文件。

默认分别使用后台线程预取数据和顺序执行模型推理，账户在需要分数时等待；
可用 `async_inference=False` 单独关闭推理异步。默认显示交易日进度，并在回测成功后打印指标表；
`friendly_output=False` 时，`engine.run()` 不主动打印进度或结果，命令行评测向标准输出打印指标 JSON。
`engine.run()` 返回 15 项原始指标，
并保留最终账户和七张审计表。指定输出目录时生成指标 JSON、七张 CSV 和运行日志。
优化器只提供 Top-K 和 Barra 约束优化器两种选择；
支持单股、主动权重、行业、风格及换手约束。RankIC / RankICIR 始终按当日合法股票池全截面计算。
所有数据统一从 `data_dir` 读取，按实际目录和文件字段检查可用性。
自动读取其下 `HS300_weight`、`HS300_industry` 年度中文文件；权重按日归一化，行业按历史行业代码分组。
权重名单必须与当日成分一致，原始数据不变。
后复权回测强制读取 `MarketDataRawOpen`，结合历史后复权收盘价及 `Factor33_winsor.is_st` 推算涨跌停价，
执行涨停不买、跌停不卖；无需开关。该临时路径的舍入规则和特殊交易安排边界见[数据约定](docs/usage.md#数据约定)。

## 快速开始

需要 Python 3.11 及以上。发行包的 pip 和 uv 安装方式见[使用说明](docs/usage.md)。
安装后可通过 `skd_backtest.__version__` 或 `skd-backtest-evaluate --version` 查询版本。

有两种独立的使用方式：

**Python API（个人单模型回测）**：用户对自己的单个模型进行回测时，建议参考 [examples/basic_usage.py](examples/basic_usage.py) 使用 Python API，将已初始化模型的 `model.predict` 作为 `inference` 传给 `BacktestEngine`。

**命令行评测（评测平台批量回测）**：`skd-backtest-evaluate` 是本包提供的命令行评测入口，推荐用于评测平台批量回测时使用。每次调用处理一个标准提交，由平台调度多个调用，例如 `skd-backtest-evaluate --submission submission --config config.toml`。`skd-backtest-evaluate --help` 同时提供这两种使用方式的说明和完整单模型示例。

从源码开发或运行仓库示例时，在项目根目录执行：

```powershell
python -m pip install -e .
python examples/basic_usage.py
```

[示例](examples/basic_usage.py) 通过 `engine.run()` 运行回测流程，默认读取 `D:\Data`，运行 2016–2022 年区间；
naive 模型为当日全部沪深300成分股统一输出零分，`top_k=300` 生成每只股票 `1/300` 的等权目标。
每5个交易日生成调仓信号，下一交易日开盘执行；实际持仓受停牌等交易限制影响。
运行前请将示例中的 `data_dir` 改为实际数据目录。

使用现有 `D:\Data` 的 Barra 约束示例见 [examples/barra_usage.py](examples/barra_usage.py)。
Barra 所需的 SciPy 随包默认安装；运行该示例即可使用已有风格暴露、权重和行业数据构建组合。
`OptimizerConfig(method="barra")` 统一执行约束优化，没有内部模式选择。
每次回测统一从 `data_dir/HS300_index` 读取沪深300指数日线，
将原始 `涨跌幅`（百分数）除以 100，计算年化超额收益率、跟踪误差和信息比率。
指数数据是回测必需输入，缺少目录、日期或有效收益时直接报错；无需额外配置。

`BacktestEngine` 也支持传入 `submission_dir`，由平台加载标准提交并管理每次评测的模型实例。
已完成预测回放时，可传入 `precomputed_scores`（含 `date/code/score` 的 DataFrame）复用冻结分数。
它与 `inference/submission_dir` 互斥，沿原信号日校验股票池和分数；只读取执行行情和当日Barra，
保留原交易、费用、账户、标签、指标和七张审计表。此路径的性能验收针对本地源码，尚未发布新发行包。
完整配置、参考数据与可复现性约定见[使用说明](docs/usage.md)。

## 文档

- [使用说明](docs/usage.md)：模型接入、数据格式、API、配置和返回结果。
- [实现设计与开发](docs/design.md)：数据流、模块边界、当前进度、测试与打包。
- [组件接口协议](docs/interfaces.md)：缓存主题、共享数据包、固定入口和并行文件归属。
- [性能基线](docs/benchmarks.md)：复测方法、吞吐与内存结果、指标口径。
- [开发规格](BACKTEST_PLATFORM_SPEC_v2.md)：只读需求文档。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可。
