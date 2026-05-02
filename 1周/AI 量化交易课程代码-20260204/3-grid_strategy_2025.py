# -*- coding: utf-8 -*-
"""
网格交易策略回测 - 使用本地CSV数据
针对贵州茅台股票（600519.SH）
回测区间：2025年1月1日到12月31日

策略逻辑（等差网格）：
- 震荡区间：1300-1700，中心位置：1500
- 每下跌50元，买入100股（1450买、1400买、1350买、1300买，低于1300不买）
- 每上涨50元，卖出100股（1550卖、1600卖、1650卖、1700卖，高于1700不卖）
- 初始100万现金，无持仓

网格交易原理：
- 在预设的价格区间内设置多个网格线
- 价格触及下方网格时买入，触及上方网格时卖出
- 通过高抛低吸赚取震荡区间内的差价

注意：运行此脚本前，请确保 data/600519_SH_daily.csv 文件存在
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ── 策略参数 ──────────────────────────────────────────────
STOCK_CODE      = '600519.SH'
STOCK_NAME      = '贵州茅台'
CENTER_PRICE    = 1500
GRID_SHARES     = 100
BUY_GRIDS       = sorted([1450, 1400, 1350, 1300], reverse=True)  # 高→低
SELL_GRIDS      = sorted([1550, 1600, 1650, 1700])                 # 低→高
LOWER_LIMIT     = min(BUY_GRIDS)
UPPER_LIMIT     = max(SELL_GRIDS)
INIT_SHARES     = 0
INIT_CASH       = 1_000_000.0
START_DATE      = '2025-01-01'
END_DATE        = '2025-12-31'
COMMISSION_RATE = 0.0003
DATA_FILE       = os.path.join(os.getcwd(), 'data', '600519_SH_daily.csv')
OUTPUT_DIR      = os.path.join(os.getcwd(), 'outputs')


# ── 数据加载 ──────────────────────────────────────────────
def load_stock_data(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"错误：数据文件不存在：{path}"); return None
    df = pd.read_csv(path, encoding='utf-8-sig', parse_dates=['date'])
    if 'close' not in df.columns:
        print("错误：数据文件中没有 close 列"); return None
    return df.sort_values('date').reset_index(drop=True)


# ── 指标计算 ──────────────────────────────────────────────
def calc_max_drawdown(nav: np.ndarray) -> float:
    return float(((nav / np.maximum.accumulate(nav)) - 1).min())


# ── 网格策略 ──────────────────────────────────────────────
class GridStrategy:
    """
    层级制等差网格。
    层级 n 表示已累计买入 n 手：
      - 可触发 buy_grids[n]  （第 n+1 个买入网格，价格更低）
      - 可触发 sell_grids[n-1]（第 n   个卖出网格，价格更高）
    买入后 level+1，卖出后 level-1，确保高抛低吸配对。
    """

    def __init__(self):
        self.cash   = INIT_CASH
        self.shares = INIT_SHARES
        self.level  = INIT_SHARES // GRID_SHARES   # 初始层级
        self.trades: list[dict] = []

        print(f"网格策略初始化：中心={CENTER_PRICE} 每次={GRID_SHARES}股 "
              f"买入网格={BUY_GRIDS} 卖出网格={SELL_GRIDS} "
              f"初始现金={INIT_CASH:,.0f} 初始层级={self.level}")

    @property
    def nav(self) -> float:
        """用最近成交价估算净值（调用前需传入当前价）"""
        return self._last_price and self.cash + self.shares * self._last_price or self.cash

    def get_nav(self, price: float) -> float:
        return self.cash + self.shares * price

    def execute(self, date: pd.Timestamp, price: float, prev_price: float) -> dict | None:
        """
        O(1) 网格触发：直接用 level 作为索引定位目标网格，无需遍历。
        同一天只触发一次（买入优先于卖出，与原逻辑一致）。
        """
        # ── 买入检查 ──────────────────────────────
        if self.level < len(BUY_GRIDS):
            target = BUY_GRIDS[self.level]                # O(1) 直接定位
            if prev_price > target >= price:
                cost       = GRID_SHARES * target
                commission = cost * COMMISSION_RATE
                if self.cash >= cost + commission:
                    self.cash   -= cost + commission
                    self.shares += GRID_SHARES
                    self.level  += 1
                    return self._record('买入', date, target, price)

        # ── 卖出检查 ──────────────────────────────
        if self.level > 0:
            target = SELL_GRIDS[self.level - 1]           # O(1) 直接定位
            if prev_price < target <= price and self.shares >= GRID_SHARES:
                proceeds   = GRID_SHARES * target
                commission = proceeds * COMMISSION_RATE
                self.cash   += proceeds - commission
                self.shares -= GRID_SHARES
                self.level  -= 1
                return self._record('卖出', date, target, price)

        return None

    def _record(self, action: str, date, grid_price: float, exec_price: float) -> dict:
        """构造并追加交易记录（抽取买卖共用的字典结构）"""
        amount     = GRID_SHARES * grid_price
        commission = amount * COMMISSION_RATE
        rec = {
            'action':    action,
            'date':      date,
            'grid_price':grid_price,
            'exec_price':exec_price,
            'shares':    GRID_SHARES,
            'amount':    amount,
            'commission':commission,
            'cash':      self.cash,
            'total_shares': self.shares,
            'level':     self.level,
            'nav':       self.get_nav(exec_price),
        }
        self.trades.append(rec)
        return rec


# ── 回测核心 ──────────────────────────────────────────────
def run_backtest(df: pd.DataFrame) -> tuple[pd.Series, GridStrategy]:
    """逐日撮合，返回 (nav_series, strategy)"""
    strategy = GridStrategy()
    close  = df['close'].values
    dates  = pd.DatetimeIndex(df['date'])
    nav    = np.empty(len(close))
    nav[0] = strategy.get_nav(close[0])

    for i in range(1, len(close)):
        strategy.execute(dates[i], close[i], close[i - 1])
        nav[i] = strategy.get_nav(close[i])

    return pd.Series(nav, index=dates), strategy


# ── 输出：打印报告 ─────────────────────────────────────────
def print_report(nav: pd.Series, strategy: GridStrategy):
    trades      = strategy.trades
    buy_trades  = [t for t in trades if t['action'] == '买入']
    sell_trades = [t for t in trades if t['action'] == '卖出']
    total_comm  = sum(t['commission'] for t in trades)
    total_ret   = nav.iloc[-1] / nav.iloc[0] - 1
    max_dd      = calc_max_drawdown(nav.values)

    sep = "=" * 70
    print(f"\n{sep}\n回测结果\n{sep}")
    print(f"股票代码  ：{STOCK_CODE} ({STOCK_NAME})")
    print(f"回测区间  ：{START_DATE} 至 {END_DATE}")
    print(f"网格参数  ：中心={CENTER_PRICE}，每次={GRID_SHARES}股")
    print(f"买入网格  ：{BUY_GRIDS}")
    print(f"卖出网格  ：{SELL_GRIDS}")
    print("-" * 70)
    print(f"初始资金  ：{nav.iloc[0]:>14,.2f} 元")
    print(f"期末资金  ：{nav.iloc[-1]:>14,.2f} 元")
    print(f"总收益率  ：{total_ret:>+.4%}")
    print(f"最大回撤  ：{max_dd:>.4%}")
    print("-" * 70)
    print(f"总交易次数：{len(trades)} 次（买 {len(buy_trades)}，卖 {len(sell_trades)}）")
    print(f"累计手续费：{total_comm:>14,.2f} 元")
    if buy_trades:
        print(f"平均买入价：{np.mean([t['exec_price'] for t in buy_trades]):.2f} 元")
    if sell_trades:
        print(f"平均卖出价：{np.mean([t['exec_price'] for t in sell_trades]):.2f} 元")
    print("-" * 70)
    print(f"期末持股  ：{strategy.shares} 股  层级：{strategy.level}")
    print(f"期末现金  ：{strategy.cash:>14,.2f} 元")
    print(sep)

    if trades:
        print("\n交易记录：")
        header = f"{'日期':<12}{'操作':<5}{'网格价':>8}{'成交价':>8}{'股数':>6}{'金额':>14}{'手续费':>10}{'持股':>7}{'层级':>5}{'净值':>14}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for t in trades:
            print(f"{t['date'].strftime('%Y-%m-%d'):<12}{t['action']:<5}"
                  f"{t['grid_price']:>8.0f}{t['exec_price']:>8.0f}"
                  f"{t['shares']:>6}{t['amount']:>14,.0f}"
                  f"{t['commission']:>10,.0f}{t['total_shares']:>7}"
                  f"{t['level']:>5}{t['nav']:>14,.0f}")
        print("-" * len(header))
    else:
        print("\n交易记录：无交易（价格未触及任何网格线）")


# ── 输出：保存文件 ─────────────────────────────────────────
def save_results(nav: pd.Series, strategy: GridStrategy):
    """统一保存 nav / trades / summary 三份文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trades     = strategy.trades
    total_ret  = nav.iloc[-1] / nav.iloc[0] - 1
    max_dd     = calc_max_drawdown(nav.values)
    total_comm = sum(t['commission'] for t in trades)
    buy_trades = [t for t in trades if t['action'] == '买入']
    sell_trades= [t for t in trades if t['action'] == '卖出']

    # 净值曲线
    pd.DataFrame({'date': nav.index, 'nav': nav.values,
                  'return': nav.values / nav.iloc[0] - 1}
                 ).to_csv(f'{OUTPUT_DIR}/grid_strategy_2025_nav.csv',    index=False, encoding='utf-8-sig')

    # 交易记录
    if trades:
        pd.DataFrame(trades).to_csv(f'{OUTPUT_DIR}/grid_strategy_2025_trades.csv', index=False, encoding='utf-8-sig')

    # 汇总报告
    lines = [
        "网格交易策略回测报告", "=" * 60,
        f"股票代码：{STOCK_CODE} ({STOCK_NAME})",
        f"回测区间：{START_DATE} 至 {END_DATE}",
        f"网格参数：中心={CENTER_PRICE}，每次={GRID_SHARES}股",
        f"买入网格：{BUY_GRIDS}", f"卖出网格：{SELL_GRIDS}", "-" * 60,
        f"初始资金：{nav.iloc[0]:,.2f} 元", f"期末资金：{nav.iloc[-1]:,.2f} 元",
        f"总收益率：{total_ret:.4%}",         f"最大回撤：{max_dd:.4%}", "-" * 60,
        f"总交易：{len(trades)} 次（买 {len(buy_trades)}，卖 {len(sell_trades)}）",
        f"累计手续费：{total_comm:,.2f} 元", "-" * 60,
        f"期末持股：{strategy.shares} 股",   f"期末现金：{strategy.cash:,.2f} 元",
    ]
    with open(f'{OUTPUT_DIR}/grid_strategy_2025_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"结果已保存至：{OUTPUT_DIR}/")


# ── 输出：绘图 ─────────────────────────────────────────────
def plot_results(date_index: pd.DatetimeIndex, close: np.ndarray,
                 nav: pd.Series, trades: list[dict]):
    """双联图：股价+网格线+买卖点 / 资金曲线"""
    buy_trades  = [t for t in trades if t['action'] == '买入']
    sell_trades = [t for t in trades if t['action'] == '卖出']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{STOCK_NAME}({STOCK_CODE}) 网格交易策略回测 - 2025年',
                 fontsize=16, fontweight='bold')

    # 子图1：股价 + 网格线
    ax1.plot(date_index, close, 'b-', linewidth=1.5, label='收盘价')
    ax1.axhline(CENTER_PRICE, color='gray', linewidth=2, alpha=0.8, label=f'中心线 {CENTER_PRICE}')
    for p in BUY_GRIDS:
        ax1.axhline(p, color='green', linestyle='--', linewidth=1, alpha=0.6)
        ax1.text(date_index[0], p, f' 买 {int(p)}', va='center', fontsize=9, color='green')
    for p in SELL_GRIDS:
        ax1.axhline(p, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax1.text(date_index[0], p, f' 卖 {int(p)}', va='center', fontsize=9, color='red')
    if buy_trades:
        ax1.scatter([t['date'] for t in buy_trades],  [t['exec_price'] for t in buy_trades],
                    marker='^', c='green', s=120, zorder=5, label='买入点', edgecolors='darkgreen', linewidths=1)
    if sell_trades:
        ax1.scatter([t['date'] for t in sell_trades], [t['exec_price'] for t in sell_trades],
                    marker='v', c='red',   s=120, zorder=5, label='卖出点', edgecolors='darkred',   linewidths=1)
    ax1.set(ylabel='股价 (元)', title='股价走势与网格交易点',
            ylim=(min(min(close), LOWER_LIMIT) * 0.95, max(max(close), UPPER_LIMIT) * 1.05))
    ax1.legend(loc='upper right', fontsize=10); ax1.grid(alpha=0.3)

    # 子图2：资金曲线
    nav_万  = nav.values / 10000
    init_万 = nav.iloc[0] / 10000
    ax2.plot(date_index, nav_万, 'purple', linewidth=1.5, label='资金曲线')
    ax2.axhline(init_万, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='初始资金')
    ax2.fill_between(date_index, nav_万, init_万, where=(nav_万 >= init_万), color='lightgreen', alpha=0.3)
    ax2.fill_between(date_index, nav_万, init_万, where=(nav_万 <  init_万), color='lightcoral', alpha=0.3)
    for t in trades:
        if t['date'] in date_index:
            idx = date_index.get_loc(t['date'])
            ax2.scatter(t['date'], nav_万[idx],
                        marker='^' if t['action']=='买入' else 'v',
                        color='green' if t['action']=='买入' else 'red', s=60, zorder=5)
    final_ret = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    ax2.annotate(f'最终资金: {nav_万[-1]:.2f}万 ({final_ret:+.2f}%)',
                 xy=(date_index[-1], nav_万[-1]), xytext=(-150, 20), textcoords='offset points',
                 fontsize=11, fontweight='bold', color='purple',
                 arrowprops=dict(arrowstyle='->', color='purple'))
    ax2.set(ylabel='资金 (万元)', xlabel='日期', title='资金曲线')
    ax2.legend(loc='upper left', fontsize=10); ax2.grid(alpha=0.3)

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout(); plt.subplots_adjust(top=0.93)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chart_file = f'{OUTPUT_DIR}/grid_strategy_2025_chart.png'
    plt.savefig(chart_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"策略图表已保存至：{chart_file}")
    plt.show()


# ── 主入口 ────────────────────────────────────────────────
def grid_strategy_backtest(data_file=DATA_FILE) -> dict | None:
    print("=" * 70)
    print(f"网格交易策略回测  {STOCK_NAME}({STOCK_CODE})  {START_DATE}~{END_DATE}")
    print("=" * 70)

    df = load_stock_data(data_file)
    if df is None:
        return None

    # ✅ 先转为局部变量，query 的 @ 才能正确引用
    start = pd.Timestamp(START_DATE)
    end   = pd.Timestamp(END_DATE)
    df_bt = df.query('@start <= date <= @end').reset_index(drop=True)

    if df_bt.empty:
        print(f"回测区间 {START_DATE}~{END_DATE} 无数据"); return None
    print(f"成功加载 {len(df)} 条，回测区间 {len(df_bt)} 个交易日")

    # 2. 回测
    nav, strategy = run_backtest(df_bt)

    # 3. 打印 + 保存 + 绘图
    print_report(nav, strategy)
    save_results(nav, strategy)
    plot_results(pd.DatetimeIndex(df_bt['date']), df_bt['close'].values, nav, strategy.trades)

    trades = strategy.trades
    return {
        'total_return':    float(nav.iloc[-1] / nav.iloc[0] - 1),
        'max_drawdown':    calc_max_drawdown(nav.values),
        'init_nav':        float(nav.iloc[0]),
        'final_nav':       float(nav.iloc[-1]),
        'trades_count':    len(trades),
        'buy_count':       sum(1 for t in trades if t['action'] == '买入'),
        'sell_count':      sum(1 for t in trades if t['action'] == '卖出'),
        'total_commission':sum(t['commission'] for t in trades),
        'final_shares':    strategy.shares,
        'final_cash':      strategy.cash,
        'trades':          trades,
    }


if __name__ == "__main__":
    result = grid_strategy_backtest()
    if result:
        print(f"\n回测完成！总收益率：{result['total_return']:+.4%}  "
              f"最大回撤：{result['max_drawdown']:.4%}  "
              f"交易次数：{result['trades_count']}（买{result['buy_count']}卖{result['sell_count']}）")
    else:
        print("\n回测失败，请检查错误信息。")