# -*- coding: utf-8 -*-
"""
MACD交易策略回测 - 使用本地CSV数据
针对贵州茅台股票（600519.SH）
回测区间：2025年1月1日到12月31日
全仓买入和全仓卖出策略
初始资金：100万

策略逻辑：
- 当MACD的DIF线上穿DEA线（金叉）时，满仓买入
- 当MACD的DIF线下穿DEA线（死叉）时，清仓卖出

注意：运行此脚本前，请先运行 6a-qmt_download_data.py 下载数据
      或者手动准备CSV数据文件
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 策略参数配置
STOCK_CODE = '600519.SH'  # 贵州茅台股票代码
STOCK_NAME = '贵州茅台'
SHORT_PERIOD = 12       # MACD快线周期
LONG_PERIOD = 26        # MACD慢线周期
SIGNAL_PERIOD = 9       # MACD信号线周期
START_DATE = '2025-01-01'  # 回测开始日期
END_DATE = '2025-12-31'    # 回测结束日期
INIT_CASH = 1000000.0       # 初始资金（100万）
LOT_SIZE = 100             # 最小交易单位（一手=100股）
COMMISSION_RATE = 0.0003    # 手续费率（万分之三，买入和卖出都收取）
DATA_FILE      = os.path.join(os.getcwd(), 'data', '600519_SH_daily.csv')
OUTPUT_DIR     = os.path.join(os.getcwd(), 'outputs')

# 数据文件路径（相对于当前目录）
DATA_FILE = os.path.join(os.getcwd(), 'data', '600519_SH_daily.csv')

# ─────────────────────────────── 数据加载 ───────────────────────────────
def load_stock_data(data_file):
    """
    从CSV文件加载股票数据
    参数:
        data_file: CSV文件路径
    返回:
        DataFrame: 包含日期和收盘价的数据框
    """
    if not os.path.exists(data_file):
        print(f"错误：数据文件不存在：{data_file}")
        return None
    df = pd.read_csv(data_file, encoding='utf-8-sig', parse_dates=['date'])
    if 'close' not in df.columns:
        print("错误：数据文件中没有 close 列")
        return None
    return df.sort_values('date').reset_index(drop=True)


def calc_macd(close, short=12, long=26, signal=9):
    """
    计算MACD指标
    参数:
        close: 收盘价序列
        short: 快线周期，默认12
        long: 慢线周期，默认26
        m: 信号线周期，默认9
    返回:
        dif: DIF线（快线）
        dea: DEA线（信号线）
        macd_bar: MACD柱状图（DIF-DEA）*2
    """
    # 计算EMA
    s = pd.Series(close)
    
    # DIF = EMA(12) - EMA(26)
    dif = s.ewm(span=short, adjust=False).mean() - s.ewm(span=long,  adjust=False).mean()
    
    # DEA = EMA(DIF, 9)
    dea = dif.ewm(span=signal, adjust=False).mean()
    
    return dif.to_numpy(), dea.to_numpy(), ((dif - dea) * 2).to_numpy()


def calc_max_drawdown(nav_series):
    """
    计算最大回撤
    参数:
        nav_series: 净值序列（pandas Series）
    返回:
        max_drawdown: 最大回撤值（负数）
    """
    return float(((nav_series / np.maximum.accumulate(nav_series)) - 1).min())


def run_backtest(close: np.ndarray, dif: np.ndarray, dea: np.ndarray,
                 date_index: pd.DatetimeIndex) -> tuple[np.ndarray, list[dict]]:
    """
    向量化生成信号，仅在交易日当天循环执行撮合，避免全量逐日循环。
    返回 (nav_array, trades_list)。
    """
    # 向量化生成信号（1=金叉买入，-1=死叉卖出，0=无信号）
    golden = (dif[:-1] <= dea[:-1]) & (dif[1:] > dea[1:])   # shape = (n-1,)
    death  = (dif[:-1] >= dea[:-1]) & (dif[1:] < dea[1:])
    signal = np.zeros(len(close), dtype=int)
    signal[1:][golden] =  1
    signal[1:][death]  = -1

    # 只在有信号的 index 处执行撮合，其余用向量推进净值
    nav    = np.full(len(close), np.nan)
    nav[0] = INIT_CASH
    cash, shares = INIT_CASH, 0
    trades = []

    for i in range(1, len(close)):
        price = close[i]
        sig   = signal[i]

        if sig == 1 and shares == 0:
            # 满仓买入：用全部现金计算最大可买手数
            max_shares = int(cash / (price * (1 + COMMISSION_RATE)) / LOT_SIZE) * LOT_SIZE
            if max_shares >= LOT_SIZE:
                cost = max_shares * price
                commission = cost * COMMISSION_RATE
                cash  -= cost + commission
                shares = max_shares
                trades.append(_make_trade('买入', date_index[i], price, shares, cost, commission, cash))

        elif sig == -1 and shares > 0:
            # 清仓卖出
            proceeds   = shares * price
            commission = proceeds * COMMISSION_RATE
            trades.append(_make_trade('卖出', date_index[i], price, shares, proceeds, commission, cash + proceeds - commission))
            cash  += proceeds - commission
            shares = 0

        nav[i] = cash + shares * price

    # 补齐首日之后、第一笔交易之前的 nan（直接持有现金）
    nav = pd.Series(nav).ffill().fillna(INIT_CASH).to_numpy()
    return nav, trades


def _make_trade(action, date, price, shares, amount, commission, cash_after) -> dict:
    """构造交易记录字典（抽取重复结构）"""
    return {
        'action':     action,
        'date':       date,
        'price':      price,
        'shares':     shares,
        'amount':     amount,
        'commission': commission,
        'cash':       cash_after,
        'position':   (shares * price) / (cash_after + shares * price) if action == '买入' else 0.0,
    }


# ── 输出：打印报告 ─────────────────────────────────────────
def print_report(nav: np.ndarray, trades: list[dict], cash: float, shares: int):
    total_return   = nav[-1] / INIT_CASH - 1
    max_dd         = calc_max_drawdown(nav)
    total_comm     = sum(t['commission'] for t in trades)

    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"股票代码  ：{STOCK_CODE} ({STOCK_NAME})")
    print(f"回测区间  ：{START_DATE} 至 {END_DATE}")
    print(f"初始资金  ：{INIT_CASH:>14,.2f} 元")
    print(f"期末净值  ：{nav[-1]:>14,.2f} 元")
    print(f"总收益率  ：{total_return:>+.4%}")
    print(f"最大回撤  ：{max_dd:>.4%}")
    print(f"交易次数  ：{len(trades)} 次")
    print(f"期末持仓  ：{'空仓' if shares == 0 else f'{shares} 股'}")
    print(f"期末现金  ：{cash:>14,.2f} 元")
    print(f"累计手续费：{total_comm:>14,.2f} 元")
    print("=" * 60)

    if trades:
        print("\n交易记录：")
        for t in trades:
            print(f"  {t['date'].strftime('%Y-%m-%d')} | {t['action']:4s} | "
                  f"价格: {t['price']:>8.2f} | {t['shares']:>6}股 | "
                  f"金额: {t['amount']:>12,.2f} | 手续费: {t['commission']:>8.2f} | "
                  f"仓位: {t['position']:.1%}")


# ── 输出：保存文件 ─────────────────────────────────────────
def save_results(nav: np.ndarray, date_index: pd.DatetimeIndex,
                 return_arr: np.ndarray, position_arr: np.ndarray,
                 trades: list[dict]):
    """统一保存 nav/trades/summary 三份文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_return = nav[-1] / INIT_CASH - 1
    max_dd       = calc_max_drawdown(nav)

    # 净值曲线
    pd.DataFrame({'date': date_index, 'nav': nav, 'return': return_arr, 'position': position_arr}
                 ).to_csv(f'{OUTPUT_DIR}/macd_strategy_2025_nav.csv',    index=False, encoding='utf-8-sig')

    # 交易记录
    if trades:
        pd.DataFrame(trades).to_csv(f'{OUTPUT_DIR}/macd_strategy_2025_trades.csv', index=False, encoding='utf-8-sig')

    # 汇总报告
    lines = [
        "MACD策略回测报告", "=" * 60,
        f"股票代码：{STOCK_CODE} ({STOCK_NAME})",
        f"回测区间：{START_DATE} 至 {END_DATE}",
        f"初始资金：{INIT_CASH:,.2f} 元",
        f"期末净值：{nav[-1]:,.2f} 元",
        f"总收益率：{total_return:.4%}",
        f"最大回撤：{max_dd:.4%}",
        f"交易次数：{len(trades)} 次",
        f"MACD参数：快线={SHORT_PERIOD}, 慢线={LONG_PERIOD}, 信号线={SIGNAL_PERIOD}",
        f"手续费率：{COMMISSION_RATE * 10000:.2f} 万分之",
    ]
    with open(f'{OUTPUT_DIR}/macd_strategy_2025_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"结果已保存至：{OUTPUT_DIR}/")


# ── 输出：绘图 ─────────────────────────────────────────────
def plot_results(date_index, close, nav, dif, dea, macd_bar, trades):
    """三联图：股价+买卖点 / MACD / 资金曲线"""
    buy_trades  = [t for t in trades if t['action'] == '买入']
    sell_trades = [t for t in trades if t['action'] == '卖出']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'{STOCK_NAME}({STOCK_CODE}) MACD策略回测 - 2025年', fontsize=16, fontweight='bold')

    # 子图1：股价
    ax1.plot(date_index, close, 'b-', linewidth=1.5, label='收盘价')
    for i, t in enumerate(buy_trades):
        ax1.scatter(t['date'], t['price'], marker='^', c='red',   s=150, zorder=5, edgecolors='darkred',   linewidths=1)
        ax1.annotate(f"买{i+1}", (t['date'], t['price']), xytext=(0, 15), textcoords='offset points',
                     ha='center', fontsize=9, color='red',   fontweight='bold')
    for i, t in enumerate(sell_trades):
        ax1.scatter(t['date'], t['price'], marker='v', c='green', s=150, zorder=5, edgecolors='darkgreen', linewidths=1)
        ax1.annotate(f"卖{i+1}", (t['date'], t['price']), xytext=(0,-20), textcoords='offset points',
                     ha='center', fontsize=9, color='green', fontweight='bold')
    ax1.set(ylabel='股价 (元)', ylim=(min(close)*0.95, max(close)*1.05), title='股价走势与买卖点')
    ax1.legend(loc='upper left', fontsize=10); ax1.grid(alpha=0.3)

    # 子图2：MACD
    ax2.plot(date_index, dif, 'b-',      linewidth=1.2, label='DIF')
    ax2.plot(date_index, dea, 'orange',  linewidth=1.2, label='DEA')
    colors = np.where(macd_bar >= 0, 'red', 'green')
    ax2.bar(date_index, macd_bar, color=colors, alpha=0.6, width=1.5, label='MACD柱')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    for t in trades:
        ax2.axvline(t['date'], color='red' if t['action']=='买入' else 'green', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set(ylabel='MACD', title=f'MACD ({SHORT_PERIOD},{LONG_PERIOD},{SIGNAL_PERIOD})')
    ax2.legend(loc='upper left', fontsize=10); ax2.grid(alpha=0.3)

    # 子图3：资金曲线
    nav_万 = nav / 10000
    init_万 = INIT_CASH / 10000
    ax3.plot(date_index, nav_万, 'purple', linewidth=1.5, label='资金曲线')
    ax3.axhline(init_万, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='初始资金')
    ax3.fill_between(date_index, nav_万, init_万, where=(nav_万 >= init_万), color='lightgreen', alpha=0.3)
    ax3.fill_between(date_index, nav_万, init_万, where=(nav_万 <  init_万), color='lightcoral', alpha=0.3)
    for t in trades:
        idx = date_index.get_loc(t['date'])
        ax3.scatter(t['date'], nav_万[idx],
                    marker='^' if t['action']=='买入' else 'v',
                    color='red' if t['action']=='买入' else 'green', s=80, zorder=5)
    final_ret = (nav[-1] / INIT_CASH - 1) * 100
    ax3.annotate(f'最终资金: {nav_万[-1]:.2f}万 ({final_ret:+.2f}%)',
                 xy=(date_index[-1], nav_万[-1]), xytext=(-150, 20), textcoords='offset points',
                 fontsize=11, fontweight='bold', color='purple',
                 arrowprops=dict(arrowstyle='->', color='purple'))
    ax3.set(ylabel='资金 (万元)', xlabel='日期', title='资金曲线')
    ax3.legend(loc='upper left', fontsize=10); ax3.grid(alpha=0.3)

    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout(); plt.subplots_adjust(top=0.93)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chart_file = f'{OUTPUT_DIR}/macd_strategy_2025_chart.png'
    plt.savefig(chart_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"策略图表已保存至：{chart_file}")
    plt.show()


# ── 主入口 ────────────────────────────────────────────────
def macd_strategy_backtest(data_file=DATA_FILE) -> dict | None:
    print(f"开始回测：{STOCK_NAME}({STOCK_CODE}) - MACD策略")
    print(f"回测区间：{START_DATE} 至 {END_DATE}  初始资金：{INIT_CASH:,.0f} 元")
    print("-" * 60)

    # 1. 加载数据
    df = load_stock_data(data_file)
    if df is None:
        return None
    print(f"成功加载 {len(df)} 条数据 ({df['date'].iloc[0].date()} 至 {df['date'].iloc[-1].date()})")

    close_all = pd.Series(df['close'].values, index=df['date'])
    if len(close_all) < LONG_PERIOD + SIGNAL_PERIOD + 2:
        print("错误：历史数据不足以计算MACD"); return None

    # 2. 用全量数据计算 MACD（避免截断造成预热误差）
    dif_all, dea_all, bar_all = calc_macd(close_all.values, SHORT_PERIOD, LONG_PERIOD, SIGNAL_PERIOD)

    # 3. 截取回测区间
    mask       = (close_all.index >= START_DATE) & (close_all.index <= END_DATE)
    close      = close_all[mask].values
    date_index = close_all[mask].index
    dif, dea, macd_bar = dif_all[mask], dea_all[mask], bar_all[mask]

    if len(close) == 0:
        print(f"回测区间 {START_DATE}~{END_DATE} 无数据"); return None
    print(f"回测区间内共 {len(close)} 个交易日")

    # 4. 回测
    nav, trades = run_backtest(close, dif, dea, date_index)

    # 从 trades 还原期末状态（用于报告打印）
    last_buy  = next((t for t in reversed(trades) if t['action'] == '买入'),  None)
    last_sell = next((t for t in reversed(trades) if t['action'] == '卖出'), None)
    shares = last_buy['shares'] if (last_buy and (last_sell is None or last_buy['date'] > last_sell['date'])) else 0
    cash   = nav[-1] - shares * close[-1]

    # 5. 打印 + 保存
    print_report(nav, trades, cash, shares)
    return_arr   = nav / INIT_CASH - 1
    position_arr = np.where(nav > 0, (nav - cash) / nav, 0.0)   # 近似仓位
    save_results(nav, date_index, return_arr, position_arr, trades)

    # 6. 绘图
    plot_results(date_index, close, nav, dif, dea, macd_bar, trades)

    return {
        'total_return': float(nav[-1] / INIT_CASH - 1),
        'max_drawdown': calc_max_drawdown(nav),
        'final_nav':    float(nav[-1]),
        'trades_count': len(trades),
        'trades':       trades,
    }


if __name__ == "__main__":
    result = macd_strategy_backtest()
    if result:
        print(f"\n回测完成！总收益率：{result['total_return']:+.4%}  "
              f"最大回撤：{result['max_drawdown']:.4%}  交易次数：{result['trades_count']}")
    else:
        print("\n回测失败，请检查错误信息。")