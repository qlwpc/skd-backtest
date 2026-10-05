# 使用说明

[项目首页](../README.md) · [使用说明](usage.md) · [实现设计](design.md) · [性能基线](benchmarks.md)

完整回测链路已实现；示例通过 engine.run() 计算交易、账户、标签和评价指标。源码开发和仓库示例命令在项目根目录执行。

## 安装与运行

需要 Python 3.11 及以上。当前源码版本为 `1.0.1`，对应 wheel 的安装命令如下。
GitHub 下载链接在对应 Release 发布后生效；源码安装方式见下文。

使用 pip：

```powershell
python -m pip install "https://github.com/HenryZ16/skd-backtest/releases/download/v1.0.1/skd_backtest-1.0.1-py3-none-any.whl"
skd-backtest-evaluate --help
```

使用 uv（Windows / PowerShell；示例采用 Python 3.13，框架支持 Python 3.11 及以上）：

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv "https://github.com/HenryZ16/skd-backtest/releases/download/v1.0.1/skd_backtest-1.0.1-py3-none-any.whl"
.\.venv\Scripts\skd-backtest-evaluate.exe --help
```

已有符合版本要求的 `.venv` 时，可跳过创建环境。macOS / Linux 的命令入口位于 `.venv/bin/skd-backtest-evaluate`。

**Python API（个人单模型回测）**：用户对自己的单个模型进行回测时，建议参考 [examples/basic_usage.py](../examples/basic_usage.py) 使用 Python API，将已初始化模型的 `model.predict` 作为 `inference` 传给 `BacktestEngine`。

**命令行评测（评测平台批量回测）**：`skd-backtest-evaluate` 是本包提供的命令行评测入口，推荐用于评测平台批量回测时使用。每次调用通过 `--submission` 指定一个标准模型提交目录、通过 `--config` 指定 JSON/TOML 配置，由平台调度多个调用。该命令由 `pyproject.toml` 的 `[project.scripts]` 声明，安装时自动生成，实际调用 `skd_backtest.evaluate.main()`，无需另装工具。在安装包的同一 Python 环境中，`python -m skd_backtest.evaluate` 提供相同入口。`-h`、`-help` 和 `--help` 均展示这两种独立使用方式的说明和完整 `basic_usage.py` 示例代码。

附件和版本说明见 [GitHub Release](https://github.com/HenryZ16/skd-backtest/releases/tag/v1.0.1)。数据集需单独准备。
若要开发或运行仓库中的示例，在项目根目录执行：

```powershell
python -m pip install -e .
python examples/basic_usage.py
```

使用 uv 也可以：

```powershell
uv venv .venv
uv pip install -e .
.venv\Scripts\python.exe examples/basic_usage.py
```

```python
import pandas as pd
from skd_backtest import BacktestEngine, CostConfig, OptimizerConfig


def inference(as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # data 只包含截至当日、最多 lookback 个交易日的研究历史。
    day = int(as_of_date.replace("-", ""))
    codes = data["Barra_factor"].loc[lambda frame: frame["日期"] == day, "代码"]
    return pd.DataFrame({"date": as_of_date, "code": codes, "score": 0.0})


engine = BacktestEngine(
    data_dir=r"D:\Data",             # 必须显式传入，不硬编码机器路径
    start_date="2016-01-01",         # YYYY-MM-DD，区间两端包含
    end_date="2022-12-31",
    inference=inference,
    initial_cash=1_000_000,
    rebalance_interval=5,            # 调仓间隔，交易日
    holding_period=5,                # RankIC 标签持有期，交易日
    lookback=252,                    # 模型可见历史窗口，交易日
    read_batch_months=12,            # 每批最多 12 个自然月
    prefetch=True,                   # 后台预取下一批行情
    async_inference=True,            # 独立推理线程；False 使用同步推理
    friendly_output=True,            # 默认显示进度和结果表；False 时引擎不主动打印
    price_mode="adjusted_return",
    optimizer_config=OptimizerConfig(top_k=300),  # 沪深300全部成分股等权目标
    cost_config=CostConfig(slippage=0.0),
)
metrics = engine.run()               # 自动展示结果；仍返回 15 项原始指标，未定义值为 None
print(engine.trading_dates)          # 实际遍历的全部交易日
print(engine.tables["predictions"]) # 模型分数与独立计算的 future_return
print(engine.performance)           # 读取、播放、推理及框架耗时
```

已有设计文件中的模型时，只加载一次，再传入其方法句柄：

```python
model = InferenceModel("./submission/model/")
engine = BacktestEngine(
    data_dir=r"D:\Data",
    start_date="2016-01-01",
    end_date="2022-12-31",
    inference=model.predict,
)
metrics = engine.run()
```

推理接口如下；平台按 as_of_date 和 data 两个关键字调用：

```python
def predict(self, as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ...
```

传入绑定方法 `model.predict` 时不传 self；普通函数省略 self。
类型要求统一声明在框架内部，由 Typeguard 装饰器自动检查；用户无需继承类、添加装饰器或编写检查代码，用户函数省略类型标注也能使用。
每次调用前检查 as_of_date 为 str、data 为 dict，且所有键为 str、所有值为 DataFrame；
模型返回时检查实际结果为 DataFrame，类型不符直接抛出 typeguard.TypeCheckError。
构造阶段不执行推理或检查用户函数的标注；参数名或调用方式不兼容时，由 Python 在实际调用时抛出 TypeError。
直接传入 inference 与 submission_dir 加载的模型，以及同步和异步推理，共用同一检查入口。
使用普通 Python 运行模式；python -O / -OO 会关闭 Typeguard 的装饰器检查。
首个信号另检查 date/code/score 列结构。
Runner 会验证当日日期、有限数值分数、代码唯一及完整覆盖当日合法池，
然后按代码排序并发布缓存。传入回调时，模型实例由调用者创建与管理；平台不执行训练。
示例使用 naive 模型，为当日 Barra 表中的全部沪深300成分股统一返回零分；
`top_k=300` 为300只成分股生成各 `1/300` 的等权目标，实际持仓受停牌等交易限制影响。
历史不足时提供已有数据，模型自行处理短窗口。

默认模型在独立线程按信号日期顺序调用，可在账户处理当前调仓期间计算下一信号日。
模型始终只接收对应日期的研究数据；账户在 Optimizer 需要当日分数时等待，缓存只由主线程访问。
最多提前缓冲一个调仓间隔的日包，研究窗口按引用交给任务，不新增深拷贝。
`async_inference=False` 恢复同步推理，`prefetch=False` 只关闭行情读取预取；两者独立。
用户回调必须正常返回或抛出异常才能结束正在运行的线程。无需修改 model.predict 签名或 basic_usage 调用方式。

## 查询版本

Python API 从已安装发行包的元数据读取版本，与 wheel 的版本保持一致：

```python
import skd_backtest

print(skd_backtest.__version__)  # 1.0.1
```

命令行不需要提供提交目录、配置或市场数据：

```powershell
skd-backtest-evaluate --version
# skd-backtest-evaluate 1.0.1
python -m skd_backtest.evaluate --version
```

从源码开发时先执行 `python -m pip install -e .`；修改项目版本后需重新安装以更新包元数据。

## 标准参赛提交与可复现性

用 `submission_dir="./submission"` 替代 `inference=model.predict` 即可由平台加载
`submission/inference.py`，调用 `InferenceModel(model_dir)`；model/ 可以有多个固化模型文件。
模型路径的两个参数必须且只能传一个。每次评测只初始化一次实例，各信号日复用；
同一个 Engine 再次 run 时，标准提交会重新加载模型，避免上次评测的内部状态影响结果。
推理文件可用 `from .helper import ...` 加载提交目录内的辅助模块，不同队伍的相对导入互不混用。

`random_seed` 默认 0，控制加载模型与推理期间的 Python random 和 NumPy 传统全局随机流；
每个 Runner 持有独立状态，调用结束恢复宿主进程随机状态，模型调用之间继续推进自己的流。
可复现评测需固定 submission、数据、配置与数值环境。自行创建的 `default_rng()`、第三方随机生成器、
外部熵及非确定性硬件算法由参赛者固定种子/配置；平台不自动改写参赛代码。
直接传 callable 时，平台重置上述随机流，但其自有模型内部状态仍由调用者复位或重新创建。

已完成逐日预测的研究可使用源码中的`precomputed_scores`路径（尚未发布到新发行包）：

```python
scores = pd.read_csv("predictions.csv", dtype={"date": str, "code": str})
engine = BacktestEngine(
    data_dir="./data", start_date="2019-01-02", end_date="2019-12-31",
    precomputed_scores=scores, holding_period=1, rebalance_interval=1,
)
metrics = engine.run()
```

`precomputed_scores/inference/submission_dir`必须且只能提供一个。引擎复制冻结分数，按原信号日推进并经原Runner校验
当日完整股票池、唯一编码和有限分数；缺少信号日立即报错，无下一执行日的末日仍不生成目标。
冻结路径读取当日Barra及执行行情，省去三表研究窗口，行情预取仍生效，推理线程关闭。
撮合、持仓、费用、历史参考价格、独立前向标签及审计输出保持同一实现。
这条路径只评价分数，不验证生成分数的模型时间边界；完整标准提交验收仍须先逐日执行模型入口。
`engine.performance.score_source`标记`precomputed`或`inference`，读取统计反映实际减少的数据量。

评测平台批量回测推荐使用 `skd-backtest-evaluate`；一次调用处理一个标准提交，批量任务由平台调度。统一入口使用 JSON 或 TOML 配置：

~~~powershell
python evaluate.py --submission submissions/team_001 --config configs/private.toml
# 安装包也提供同样的命令：
skd-backtest-evaluate --submission submissions/team_001 --config configs/private.toml
# 关闭进度和结果表，向标准输出打印指标 JSON：
skd-backtest-evaluate --submission submissions/team_001 --config configs/private.toml --no-friendly-output
~~~

最小 TOML：

~~~toml
[backtest]
data_dir = "D:/Data"
start_date = "2016-01-01"
end_date = "2022-12-31"
random_seed = 0
friendly_output = true
output_dir = "../result/team_001"

[optimizer]
method = "top_k"
top_k = 50
~~~

**配置段与参数说明：**

| TOML 配置段 | 参数说明 |
|---|---|
| `[backtest]` | [回测区间、数据目录、资金及运行选项](#backtest-config) |
| `[optimizer]` | [Top-K / Barra 选择与组合约束参数](#optimizer-config) |
| `[costs]`、`[[costs.fee_schedule]]` | [佣金、滑点和历史费率](#costs-config) |

配置段仅为 backtest、optimizer、costs，字段与对应数据类同名。
相对数据和输出路径相对于配置文件目录解析。
统一入口未指定 output_dir 时使用 `result/<提交目录名>`，必定生成审计文件。
费用表使用 costs.fee_schedule 条目，包含 effective_date、stamp_tax_rate、transfer_fee_rate。
直接使用 Engine 时仍可 output_dir=None，不写结果文件。
`--friendly-output` / `--no-friendly-output` 优先于配置文件中的 `backtest.friendly_output`。

## 数据约定

所有路径以构造函数传入的 `data_dir` 为根：

```text
<data_dir>/Factor33_winsor/<YYYY>/<MM>/<YYYYMM>.parquet
<data_dir>/Barra_factor/<YYYY>/<MM>/<YYYYMM>.parquet
<data_dir>/MarketData/<YYYY>/<MM>/<YYYYMM>.parquet
<data_dir>/MarketDataRawOpen/<YYYY>/<MM>/<YYYYMM>.parquet  # 后复权开盘执行必需的原始 open
<data_dir>/HS300_weight/<YYYY>/hs300_weight_<YYYY>.csv
<data_dir>/HS300_industry/<YYYY>/hs300_industry_<YYYY>.csv
<data_dir>/HS300_index/<YYYY>/hs300_index_<YYYY>.csv    # 必需指数日线，GBK / UTF-8
```

`data` 是以 `Factor33_winsor`、`Barra_factor`、`MarketData` 为键的字典，
各表列名保留源数据格式：`日期/代码/名称` 及各自特征列。
源表 `日期` 为整数 YYYYMMDD，代码保留 `SH`/`SZ` 前缀，按日期、代码排序；
NaN 保持不变，不补值、不重新缩尾或拟合变换。推理结果的 `date` 使用
YYYY-MM-DD 字符串，`code` 保留市场前缀。

引擎专用行情保留年度并集，以便后续继续处理已调出成分的持仓：
开盘列为 `date/code/adjusted_open/is_suspended/is_missing/previous_close/previous_close_date/upper_limit/lower_limit`；
其中上下限与 adjusted_open 同为后复权口径。
收盘列为 `date/code/adjusted_close/is_suspended/is_missing/has_valid_close/previous_close/`
`previous_close_date/reference_close/reference_date/is_stale`。

- `previous_close`：严格早于当日、明确非停牌记录中的最近有限正收盘价；首条记录之前没有已知价格则为空。
- `reference_close`：截至当日、明确非停牌记录中的最近有限正收盘价；停牌、状态缺失、缺价、NaN、Inf、非正价格不覆盖历史参考价。
- 参考价格日期使用整数 YYYYMMDD，可空；`date` 是 YYYY-MM-DD 字符串。
- `is_missing`：源表当天没有这只股票的记录；`has_valid_close`：当日明确非停牌且包含有限正收盘价。
- `is_stale`：有参考价格，但其有效源记录日期早于当日。停牌行即使携带有限报价也不会更新参考价；
  跨批次和回测前历史初始化使用同一规则，`is_suspended` 独立保留。

引擎行情会保留已出现但当前缺失的股票；当日开盘/收盘原值保持空缺，历史参考价单独提供，
不填成可成交价格、不把缺失当零收益。股票跨年退出源数据覆盖时也不会从行情输入中消失。
新出现的股票不会因为批次预取而提前出现在开盘/收盘输入中。
开盘接口不携带当日 high/low/close/volume/amount；后复权价也不命名为 raw_open。
基准权重和行业数据缺失时，通过 Dataset(status="unavailable", data=None, reason=...) 表达，
不使用空表或等权假冒真实数据。DailyData.portfolio 仅承载当日 Barra，其他参考输入由 Reference Data 发布。

当前数据约定：

- Barra 是每日真实沪深300成分；Factor33 和 MarketData 是当年度成分并集。
  研究输入已按逐日 Barra 成分过滤，当前合法股票池由当日 Barra 定义。
- 财务因子的披露日 PIT 和源数据预处理依赖数据生产方；本层只按已有日期和成分过滤，
  不从日频因子文件重新构建财报披露版本。
- MarketData 的 OHLC 是后复权价；MarketDataRawOpen 独立提供未复权开盘价，字段仅为
  `日期、代码、open`。当前文件未提供完整真实 OHLC、复权因子和涨跌停限价；默认交易模式为 `adjusted_return`。
- `raw_price` 已支持真实股数、T+1、整手买入和涨跌停单边限制；须补齐以下数据，
  不从 `amount/volume` 推造开盘价。

真实价格模式在同一月度 MarketData 中额外读取：

| 实际文件字段 | 读取方式 |
|---|---|
| `raw_open/raw_high/raw_low/raw_close` 完整存在 | 直接使用真实价格 |
| 缺少完整原价，但有 `adjustment_factor` 和复权 OHLC | 用复权价格除以因子恢复原价 |
| `upper_limit/lower_limit` | raw_price 执行所需的历史限价 |

复权因子约定为 `adjusted_price = raw_price * adjustment_factor`，因子必须为有限正值。
按每个月文件的实际字段选择来源，同时提供原价和因子时优先原价；没有可用来源时指出文件与缺少的字段。
真实开盘/收盘接口分别使用 raw_open/raw_close，
previous_close 和 reference_close 也使用原价体系。研究窗口始终只含原 SOURCE_COLUMNS，
不会把原价、因子或限价附加列传给模型。股票停牌或当日价格缺失时不成交，持仓继续按最近可用参考价估值。

`label_price_basis` 独立选择 adjusted_open 或 raw_open；后者逐文件识别 raw_open 或 open + adjustment_factor，
但不要求将交易模式切换为 raw_price。

1.0.x 的后复权开盘执行强制使用 MarketDataRawOpen 推算涨跌停价，没有开关。
该数据只用于后复权交易限制，不进入模型、标签计算或真实价格回测；这条临时推算路径后续可能移除。
每个执行月份分别读取 MarketDataRawOpen 的 `日期、代码、open` 和 Factor33_winsor 的 `日期、代码、is_st`，
按日期和代码对齐，包含已调出当日成分的持仓；仅缓存一个月。`limit_files` 单独统计这两类辅助读取，
`data_files` 仍统计原三套完整数据。独立的收盘估值 API 不读取这些辅助数据。

常规涨跌幅按股票代码与日期确定：沪深主板（SH60、SZ00）为 10%，其中 `is_st=1` 为 5%；
科创板（SH688）为 20%；创业板（SZ30）自 2020-08-24 起为 20%，此前按 10%／ST 5% 处理。
is_st 使用原字段的 0/1，不按股票名称推断，也不根据当日 high/low/close 推断交易状态。
创业板切换日期见[深交所说明](https://www.szse.cn/aboutus/trends/news/t20200821_580924.html)。

推算顺序如下，所有金额舍入均用十进制 ROUND_HALF_UP：

1. `f = adjusted_open / raw_open`，比例保持精度。
2. 用 `previous_close / f` 估计当天原价口径的前收盘参考价，并四舍五入恢复到分。
3. 分别乘以 `1+r` 和 `1-r`，四舍五入到 0.01 元；若与参考价相差不足一分钱，至少增减一分钱，最低价不低于 0.01 元。
4. 两个原价限价乘以 `f`，得到后复权上下限；这一步不再按原价的 0.01 元取整。

涨跌停的舍入与最小变动规则见[深交所说明](https://investor.szse.cn/knowledge/stock/deal/t20180801_553961.html)。
例如参考价 10.05 元、涨跌幅 10% 时，原价上限为 11.06、下限为 9.05；不能使用 Python round 的银行家舍入，
也不能将跌停价算成参考价除以 1.10。

Broker 用后复权开盘价与后复权限价比较：涨停拒绝买入、跌停拒绝卖出，另一方向正常判断；
比较只容忍换算产生的浮点误差，不使用半分钱等宽容差。卖出失败保留持仓，后续买入只使用实际可用现金。
缺少月文件、可交易行 raw open 缺失/非正/非有限、is_st 无效或日期代码重复均明确报错；
缺少历史前收盘参考价时上下限为空，需要成交的订单以 `MISSING_PRICE_LIMIT` 拒绝，不静默放行。

该路径按现有比例后复权数据估计常规限价，不等同于交易所原始限价数据。
它依赖源价格与 is_st 的准确性，未重建新股无涨跌幅期、重新上市、退市整理等特殊交易安排。

## 独立数据 API

`engine.data_provider.playback()` 不调用模型、Broker、Accounting、Metrics 或 ResultWriter，
按真实交易日依次交付 `DailyData`。消费者可以选择需要的字段：

```python
from contextlib import closing

with closing(engine.data_provider.playback()) as days:
    for day in days:
        open_prices = day.open_market
        close_prices = day.close_market
        # 非信号日以及末日为 None；不在数据 API 内自动调用 inference。
        research = day.research
        portfolio = day.portfolio
        next_date = day.execution_date
```

完整遍历时自动回收线程；提前 break 或消费者可能抛出异常时使用 `closing`，确保及时回收。
独立调用从头播放；若已调用 prepare，则复用该准备结果而不重复启动读取。
关闭后下一次重新准备。不支持在同一个 DataProvider 上交错运行两个播放迭代器。

独立加载整个回测区间的估值输入：

```python
valuation_data = engine.data_provider.valuation_inputs()
selected = engine.data_provider.valuation_inputs(codes=["SZ000001", "SH600000"])
```

一次调用返回逐日收盘输入表，不计算收益。只读取 MarketData，不要求因子和 Barra 文件；
使用独立读数状态，不干扰当前播放或修改其统计。传入 codes 时，未知或缺失股票也保留每日一行，
价格未知则为空，不补造历史价格。省略 codes 时返回源数据中已出现过的股票。
无交易日区间返回带完整列名的空表。
此 API 会物化整个区间的结果，内存随日期数和股票数增长；播放 API 的分批缓存限制不适用于返回表。

<a id="backtest-config"></a>

## 构造配置

### `[backtest]` 回测参数

| 参数 | TOML 类型 | 默认值 | 含义与取值要求 |
|---|---|---|---|
| `data_dir` | 字符串 | 必填 | 数据根目录，例如 `"D:/Data"`；参考数据也从其固定子目录读取 |
| `start_date` | 字符串 | 必填 | 回测开始日期，格式 `"YYYY-MM-DD"`，包含当天 |
| `end_date` | 字符串 | 必填 | 回测结束日期，格式 `"YYYY-MM-DD"`，包含当天；不得早于开始日期 |
| `initial_cash` | 数值 | `1000000.0` | 初始资金，必须有限且大于 0 |
| `rebalance_interval` | 整数 | `5` | 调仓信号间隔，单位为交易日，必须为正整数 |
| `holding_period` | 整数 | `5` | 预测评价标签的持有期，单位为交易日，必须为正整数 |
| `lookback` | 整数 | `252` | 模型研究窗口长度，单位为交易日，必须为正整数 |
| `price_mode` | 字符串 | `"adjusted_return"` | `"adjusted_return"` 按复权价格推进持仓价值；`"raw_price"` 按真实价格和股数交易、估值 |
| `trading_days_per_year` | 整数 | `252` | 绩效年化使用的每年交易日数，必须为正整数 |
| `risk_free_rate` | 数值 | `0.0` | 年化无风险利率，按小数填写，必须有限且大于 -1 |
| `output_dir` | 字符串 | 命令行：`result/<提交目录名>`；Python API：不写文件 | 审计结果输出目录；显式填写的相对路径以 TOML 所在目录为基准，命令行默认目录以当前工作目录为基准 |
| `read_batch_months` | 整数 | `12` | 每批读取的月份数，必须为正整数 |
| `prefetch` | 布尔值 | `true` | 是否在后台预取下一批数据 |
| `async_inference` | 布尔值 | `true` | 是否使用后台线程执行模型推理 |
| `random_seed` | 整数 | `0` | 随机种子，范围为 0 至 4294967295 |
| `label_price_basis` | 字符串 | `"adjusted_open"` | 预测标签价格口径，可选 `"adjusted_open"` 或 `"raw_open"` |
| `friendly_output` | 布尔值 | `true` | 显示进度和结果表；命令行 `--friendly-output` / `--no-friendly-output` 可覆盖此值 |

TOML 模型目录由命令行 `--submission` 指定；优化器和费用分别填写在 `[optimizer]`、`[costs]`。
直接使用 Python API 时，另传 `inference` / `submission_dir` 二选一，以及可选的 `optimizer_config`、`cost_config`；这四项不写入 `[backtest]`。
TOML 中省略可选字段即使用默认值，不写 `None` 或 `null`。

数据配置统一为 `data_dir`，不再另设路径或能力声明。平台按实际目录、文件和字段判断可用性。
所有回测都需要 HS300_index；Barra 需要 HS300_weight 和 Barra_factor，行业约束另需 HS300_industry。
参考目录递归读取 CSV / Parquet；既支持标准列名，也支持下述指数、权重和行业的原始中文格式。
缺少必需目录、文件或字段时明确报错。
按精确日期读取，不前向填充；指数收益评价统一执行，优化参考权重按组合构建需求读取。
完整字段及其关系见[接口协议](interfaces.md)。

<a id="optimizer-config"></a>

### `[optimizer]` 优化器参数

| 参数 | TOML 类型 | 默认值 | 含义与取值要求 |
|---|---|---|---|
| `method` | 字符串 | `"top_k"` | 仅支持 `"top_k"` 和 `"barra"` |
| `top_k` | 整数 | `50` | 仅 Top-K 使用；选择分数最高的 K 只股票并等权分配，必须为正整数；股票不足 K 时使用全部合法股票 |
| `long_only` | 布尔值 | `true` | 仅做多；当前现货引擎不支持 `false` |
| `fully_invested` | 布尔值 | `true` | 目标权重合计为 1；`false` 允许剩余现金；要求满仓却不可行时会报错 |
| `single_name_weight_limit` | 数值 | 未设置 | 两种方法均支持；每只股票目标权重上限，范围为 `(0, 1]`，例如 `0.10` 表示 10% |
| `active_weight_limit` | 数值 | 未设置 | 仅 Barra；每只股票相对指数权重偏离 `abs(w-b)` 的上限，有限且非负；`0.01` 表示 1 个百分点 |
| `industry_exposure_limit` | 数值 | 未设置 | 仅 Barra；每个行业主动权重之和的绝对值上限，有限且非负；启用时需要历史行业数据 |
| `barra_style_exposure_limit` | 数值 | 未设置 | 仅 Barra；源表全部因子各自的主动暴露 `abs(Xᵀ(w-b))` 上限，有限且非负；单位沿用输入因子 |
| `turnover_limit` | 数值 | 未设置 | 仅 Barra；双边权重换手 `sum(abs(w-current_weights))` 上限，有限且非负，包含调出股票归零的卖出部分，不除以 2 |

“未设置”表示省略该字段；TOML 不写 `None` 或 `null`。四项 Barra 专用上限写 `0` 表示严格零偏离或零换手，Top-K 收到这些限制时会报错。
Top-K 分数并列时按代码升序选择。Barra 在全部合法股票上优化，不使用 `top_k`。
股票调出合法池时生成零目标，实际能否卖出由 Broker 判断。

### Barra 约束优化器

Barra 所需的 SciPy 随包默认安装，在 `[optimizer]` 中选择 `method = "barra"` 即可。
统一使用当日风格暴露 X、归一化参考权重 b 和启用行业约束时的历史行业分类，
在组合约束下最大化 `alphaᵀ w`，没有内部模式开关。完整运行示例为
[examples/barra_usage.py](../examples/barra_usage.py)。

alpha 为全部合法股票分数的平均并列百分位减 0.5，不截取 Top-K。
本地风格数据为 0–1 截面百分位；风格限制使用该输入单位，例如 0.02 表示主动暴露差不超过 0.02。
所有输入必须在信号时点已知，不使用之后的收益或快照。

满仓与换手限制必须共同可行，
例如初始全现金到满仓的双边股票权重换手至少为 1。实际成交受交易限制影响，可偏离优化目标。

参考数据放在 `data_dir` 下的固定目录：

| 目录 | 必需字段 |
|---|---|
| HS300_weight | date、code、benchmark_weight；或下述原始中文格式 |
| HS300_industry（启用行业限制时） | date、code、industry；或下述原始中文格式 |
| HS300_index（所有回测必需） | 日期、代码、涨跌幅；也支持标准 date、benchmark_return |

每次回测统一计算年化超额收益率、跟踪误差和信息比率，无需设置额外参数。
HS300_index 原始日期为 YYYYMMDD，代码必须为 SH000300；
`benchmark_return = 涨跌幅 / 100`，例如 -7.02 表示 -7.02%，转换为 -0.0702。
直接使用每条记录的涨跌幅，保留回测首日和跨年度首日的真实收益；不使用对数收益，也不将首日填零。
全部回测交易日必须各有一条有效记录，不补齐缺失日期。指数目录缺失或数据无效时直接报错。
年化超额收益率为组合与指数各自年化收益率之差；跟踪误差和信息比率使用日超额收益的样本标准差，
按 trading_days_per_year 年化（默认 252），不根据数据文件说明自动改成 243。
优化权重不会被当作真实指数收益。缺少必需指数数据时明确报错，不能跳过指数评价。

风格暴露直接使用当日 `Barra_factor` 中除日期、代码、名称外的全部因子列，无需配置因子名单；数据必须覆盖全部合法股票，且各因子值完整、有限。缺少必需来源或不可行约束明确失败。
Barra 使用 SciPy [HiGHS 线性规划](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html)。
结果再次验证全部约束，容差为 1e-8。不自动放宽约束或回退简单算法。
线性目标存在多个最优解时由固定代码顺序与求解器确定结果，不保证最接近基准的解。

### 原始权重与历史行业数据

传入 `data_dir="D:/Data"` 即会发现其下的 HS300_weight 和 HS300_industry。
目录内递归读取 CSV / Parquet，忽略 README 等其他文件；CSV 支持 UTF-8 和 GBK 编码。
每个目录可以包含一个或多个文件，已有年度目录无需搬动或改名。
原始权重核心列为 `日期、代码、权重、指数成份日`；行业核心列为 `日期、代码、行业代码`。
日期使用 YYYYMMDD。`权重来源`、`行业名称`、证券名称等附加列保留在读取表中。
标准 CSV/Parquet 接口仍使用 `date, code, benchmark_weight` 或 `date, code, industry`，日期为 YYYY-MM-DD。

权重按日期除以当天合计，归一化为 1；目标表保存归一化后的权重。输入名单必须与当日合法池完全一致。
新调入成分直接使用数据表提供的权重；已调出股票目标为零，执行受限时实际持仓继续保留。
程序不补零、不剔除错误成分、不估算缺失权重，发现名单不一致时直接报错。

行业约束使用对应日期的行业代码。相同行业代码的名称变化不产生新行业；股票历史行业代码变化则使用当日值。
所有合法成分必须有行业标签。通用行业文件可包含更大的股票池，读取时选择当日合法成分。
HS300_industry 存在时自动读取，启用行业约束只需配置
`OptimizerConfig(method="barra", industry_exposure_limit=...)`。
[完整示例](../examples/barra_usage.py) 同时启用行业、风格、主动权重和换手约束。

权重必须有限、非负，每日合计有限且大于零。日期/代码重复、整日缺失、未来快照均报错。
带 snapshot_date 的标准文件同样校验快照日期。来源按精确日期读取，不回填未来数据。
原始文件不修改；输入数据应满足信号时点的历史可得性。

<a id="costs-config"></a>

### `[costs]` 费用参数

| 参数 | TOML 类型 | 默认值 | 含义与取值要求 |
|---|---|---|---|
| `commission_rate` | 数值 | `0.0` | 佣金率，按小数填写，必须有限且非负，买卖双边收取 |
| `minimum_commission` | 数值 | `0.0` | 每笔成交最低佣金金额，与账户资金同单位，必须有限且非负 |
| `slippage` | 数值 | `0.0` | 滑点比例，按小数填写，范围为 `[0, 1)`；买入提高成交对价，卖出降低成交对价 |
| `fee_schedule` | 表数组 | `[]` | 历史印花税、过户费表，使用下面的 `[[costs.fee_schedule]]` 条目；按生效日期严格升序排列 |

#### `[[costs.fee_schedule]]` 历史费率条目

| 参数 | TOML 类型 | 默认值 | 含义与取值要求 |
|---|---|---|---|
| `effective_date` | 字符串 | 每条必填 | 生效日期，格式 `"YYYY-MM-DD"`；自当天起适用，直至下一条费率生效；日期不得重复 |
| `stamp_tax_rate` | 数值 | 每条必填 | 印花税率，按小数填写，必须有限且非负，仅卖出时收取 |
| `transfer_fee_rate` | 数值 | 每条必填 | 过户费率，按小数填写，必须有限且非负，买卖双边收取 |

省略 `fee_schedule` 或填写 `fee_schedule = []` 表示印花税、过户费均为零。
程序不内置或自动获取历史税率；非空费率表未覆盖某交易日期时会报错。
佣金按成交现金对价乘费率与最低佣金的较大者收取，印花税仅卖出，过户费双边收取。
买入滑点提高现金对价，卖出滑点降低现金对价；transaction_cost 只汇总显式费用，避免重复计算滑点。
现金不足时缩量并重新报价，只有最终接受的报价计入成交。

## 返回指标与审计输出

默认 `friendly_output=True`：准备数据时显示阶段提示，回放时显示已完成/总交易日、百分比、
最新完成日期、已用时间及预计剩余回放时间；回放后提示评价和文件保存阶段。
进度写入标准错误，终端内原位刷新，重定向时改用低频换行记录。回放达到 100% 后仍需完成评价、输出和资源清理。
成功结束后向标准输出打印全部 15 项指标的中文表格、交易日数、耗时及已配置的结果目录。
收益率、风险和换手等以百分比展示，费用保留两位小数，未定义值显示 N/A；原始返回值与 JSON/CSV 不做展示性舍入。
关闭时，`engine.run()` 不主动打印进度或结果，统一评测入口向标准输出打印指标 JSON；调用方及模型自身的打印不受影响。
性能测试脚本显式关闭友好输出，保留 JSON 解析和性能测量方式。

`engine.run()` 返回以下扁平字典，`engine.metrics` 保存该结果。
None 表示没有足够有效样本或比率未定义。示例的模型分数全部相同，至少两只有效配对的信号日 RankIC 记为 0；
存在有效日时 Mean RankIC 和正值比例为 0，至少两个有效日时 RankIC 标准差为 0，RankICIR 因分母为零仍为 None。
数据、模型或协议发生真实错误时仍会抛出异常，并保持 metrics=None、tables/account 为空。

| 键 | 规格指标 |
|---|---|
| `mean_rankic` | Mean RankIC |
| `rankic_std` | RankIC Std |
| `rankic_ir` | RankICIR（不年化） |
| `positive_rankic_ratio` | Positive RankIC Ratio |
| `total_return` | Total Return |
| `annualized_return` | Annualized Return |
| `annualized_excess_return` | Annualized Excess Return |
| `annualized_volatility` | Annualized Volatility |
| `maximum_drawdown` | Maximum Drawdown |
| `tracking_error` | Tracking Error |
| `information_ratio` | Information Ratio |
| `sharpe_ratio` | Sharpe Ratio |
| `turnover` | Turnover |
| `transaction_cost` | Transaction Cost |
| `failed_orders` | Failed Orders |

`engine.tables` 包含规格的七张审计表：`predictions`、`rankic`、`target_weights`、
`orders`、`trades`、`positions`、`equity_curve`。缓存已按唯一生产者归集这些表：分数和标签由 Evaluator 合并，目标来自 Optimizer，
订单/成交来自 Broker，持仓/净值来自 Accounting。Engine 只在完整运行及资源清理成功后
接收输出对象的所有权，每次 run 重置运行状态。内部缓存使用只读约定下的共享引用，关闭时不修改导出的对象。

predictions 保留所有信号日分数与独立计算的未来收益；缺价、停牌或数据集尾部不足时，
future_return 为空，不删除预测记录。每个信号日的 RankIC 面向当日合法股票池的全截面，
使用全部有效 score / future_return 配对的平均并列排名计算；不按 Top-K、目标权重或实际持仓筛选，
也不受是否成交影响。top_k 只决定组合选股数量，不改变 RankIC 或 RankICIR。
rankic 表中的 n_stocks 记录该日全截面的有效配对数。至少两只有效配对且 score 全部相等时，
RankIC 记为 0 并参与汇总；不足两只，或 score 有差异但标签全部相等时，RankIC 为空。
标签只在事后读取，可越过回测 end_date 取得持有期终点，模型不会收到未来价格。

target_weights 记录目标，orders/trades 记录真实执行结果，positions/equity_curve 记录实际持仓与逐日账户。
目标不等于实际持仓；受限订单不会自动顺延。后复权模式用资产金额记账，股数与真实成交价字段为空；
真实模式按原价和股数核算，并维护可卖股数和 T+1 锁定。

标准差采用样本标准差，RankICIR 不年化。年化收益以全部回测交易日数计算；
最大回撤包含初始 NAV=1，按非负损失比例表示；年化超额收益为组合年化收益减基准年化收益。
信息比率 `information_ratio = annualized_excess_return / tracking_error`；
tracking_error 为日主动收益的样本标准差乘以年化因子，分母为零或所需值无效时信息比率为 None。
turnover 是每日双边成交现金对价 / 当日开盘交易前权益的合计；failed_orders 只计完全拒绝，
部分成交由订单表记录。完整公式见[接口协议](interfaces.md)。

Result Writer 在指定 output_dir 后，从运行开始记录配置、阶段及错误。
成功时输出 metrics.json、七张 CSV 和 run.log；空表也保留完整列，JSON 空值为 null。
已有任一协议输出文件的目录会被拒绝，需选择新目录；output_dir=None 只保留内存结果。
CSV 先写入临时文件，全部成功后最后发布 metrics.json 作为完成标记。
写入失败不发布成功回执，也不保留本次完成标记；日志和已写 CSV 可用于排错。
