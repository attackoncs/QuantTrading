# -*- coding: utf-8 -*-
"""
CASE：格雷厄姆的低PB策略（捡烟蒂选股器）

核心思想：
  格雷厄姆认为：当一家公司的市场价格低于其账面净资产（PB<1），
  就像捡别人扔掉的烟蒂，虽然只剩一口，但那一口是免费的。

筛选条件：
  - PB < 1（破净：市场给价低于公司净资产，相当于"清算价"买入）
  - ROE > 5%（仍在赚钱，排除亏损的垃圾股）

数据文件：data/stock_basic.csv, data/daily_basic_latest.csv, data/fina_indicator_pool.csv
"""
# -*- coding: utf-8 -*-
"""
格雷厄姆"捡烟蒂"选股器
纯本地 CSV 分析，无 API 调用
"""
import os
import pandas as pd

# ── 可调参数 ──────────────────────────────────────────────
PB_MAX  = 1.0   # PB 上限：低于此值为"破净"
ROE_MIN = 5.0   # ROE 下限（%）

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# ── 数据加载 ──────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[None, None, None]:
    required = {
        'stock_basic.csv':        '股票列表',
        'daily_basic_latest.csv': '估值数据',
        'fina_indicator_pool.csv':'财务指标',
    }
    missing = [f"  {f}（{d}）" for f, d in required.items()
               if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print("错误：缺少数据文件，请先运行数据下载脚本\n" + "\n".join(missing))
        return None, None, None

    stocks = pd.read_csv(os.path.join(DATA_DIR, 'stock_basic.csv'),
                         dtype={'ts_code': str}, encoding='utf-8-sig')
    daily  = pd.read_csv(os.path.join(DATA_DIR, 'daily_basic_latest.csv'),
                         dtype={'ts_code': str}, encoding='utf-8-sig')
    fina   = pd.read_csv(os.path.join(DATA_DIR, 'fina_indicator_pool.csv'),
                         dtype={'ts_code': str, 'end_date': str}, encoding='utf-8-sig')
    return stocks, daily, fina


def _best_period(fina: pd.DataFrame) -> tuple[str, str]:
    """优先 20241231，否则取数据中最大年报期"""
    if fina is None or fina.empty or 'end_date' not in fina.columns:
        y = pd.Timestamp.now().year - (1 if pd.Timestamp.now().month >= 5 else 2)
        return f'{y}1231', str(y)

    # ✅ 向量化提取 8 位日期，替代多次链式 .str 操作
    end8 = fina['end_date'].str.replace('-', '', regex=False).str[:8]
    valid = end8[end8.str.fullmatch(r'\d{8}', na=False)]

    if '20241231' in valid.values:
        return '20241231', '2024'

    period = valid.max()
    period = str(period)[:8] if pd.notna(period) else f'{pd.Timestamp.now().year - 1}1231'
    return period, period[:4]


# ── 主流程 ────────────────────────────────────────────────
def run_screener():
    stocks, daily, fina = load_data()
    if stocks is None:
        return

    period, roe_year = _best_period(fina)

    trade_date = ''
    if 'trade_date' in daily.columns and not daily.empty:
        td = str(daily['trade_date'].iloc[0])
        trade_date = f"{td[:4]}-{td[4:6]}-{td[6:8]}" if len(td) >= 8 else td

    sep = '=' * 70
    print(sep)
    print(f"筛选条件：PB < {PB_MAX}（破净）且 ROE > {ROE_MIN}%（仍盈利）")
    print(f"估值日期：{trade_date}    ROE 来源：{roe_year} 年报 {period}")
    print(sep)

    # ── 排除 ST ──────────────────────────────────────────
    st_mask      = stocks['name'].str.contains('ST', case=False, na=False)
    stocks_clean = stocks[~st_mask].copy()
    print(f"\n全市场 {len(stocks)} 只，排除 ST {st_mask.sum()} 只，剩余 {len(stocks_clean)} 只")

    # ── 合并估值 ─────────────────────────────────────────
    merged = (stocks_clean
              .merge(daily[['ts_code', 'close', 'pb', 'pe', 'total_mv']],
                     on='ts_code', how='inner')
              .dropna(subset=['pb']))
    print(f"有 PB 数据：{len(merged)} 只")

    # ── 合并 ROE（指定报告期，去重取最后一条） ───────────
    end8 = fina['end_date'].str.replace('-', '', regex=False).str[:8]
    fina_period = (fina[end8 == period]
                   .drop_duplicates('ts_code', keep='last')
                   [['ts_code', 'roe']])
    merged = merged.merge(fina_period, on='ts_code', how='inner').dropna(subset=['roe'])
    print(f"有 PB + ROE 数据：{len(merged)} 只")

    # ── 核心筛选 ─────────────────────────────────────────
    final = (merged
             .query('0 < pb < @PB_MAX and roe > @ROE_MIN')   # ✅ query 更简洁
             .sort_values('pb')
             .reset_index(drop=True))

    # ── 结果输出 ─────────────────────────────────────────
    pb_candidates = merged.query('0 < pb < @PB_MAX')
    print(f"\n{sep}")
    print(f"破净候选（PB<{PB_MAX}）：{len(pb_candidates)} 只")
    print(f"加 ROE>{ROE_MIN}% 后：{len(final)} 只")
    print('-' * 70)

    if final.empty:
        print("没有满足条件的股票，建议放宽 PB_MAX 或降低 ROE_MIN")
        return

    # ── 表格展示 ─────────────────────────────────────────
    display = (final[['ts_code', 'name', 'industry', 'close', 'pb', 'roe', 'total_mv']]
               .assign(
                   total_mv = lambda d: (d['total_mv'] / 10000).round(1),
                   pb       = lambda d: d['pb'].round(3),
                   roe      = lambda d: d['roe'].round(2),
               )
               .rename(columns={
                   'ts_code':'代码', 'name':'名称', 'industry':'行业',
                   'close':'收盘价', 'pb':'PB', 'roe':'ROE(%)', 'total_mv':'市值(亿)',
               }))

    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 200)

    show_n = min(30, len(display))
    print(f"\n前 {show_n} 只（按 PB 从低到高）：\n" + '-' * 70)
    print(display.head(show_n).to_string(index=False))
    if len(display) > show_n:
        print(f"\n... 还有 {len(display) - show_n} 只，完整结果见 CSV")

    # ── 行业分布 ─────────────────────────────────────────
    print(f"\n{'-' * 70}\n行业分布（破净股集中在哪些行业？）：\n{'-' * 70}")
    industry_counts = final['industry'].value_counts().head(15)
    max_c = industry_counts.iloc[0]
    for ind, cnt in industry_counts.items():   # ✅ .items() 替代已废弃的 .iteritems()
        print(f"  {ind:<10s} {cnt:>3d} 只  {'█' * int(cnt / max_c * 30)}")

    # ── PB 分布 ──────────────────────────────────────────
    pb = final['pb']
    print(f"\n{'-' * 70}\nPB 分布：")
    print(f"  最低：{pb.min():.3f}（{final.iloc[0]['name']}）")
    print(f"  最高：{pb.max():.3f}    平均：{pb.mean():.3f}    中位：{pb.median():.3f}")

    # ── 保存 ─────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, '11-格雷厄姆PB选股_result.csv')
    save_cols = [c for c in ('ts_code','name','industry','close','pb','roe','pe','total_mv')
                 if c in final.columns]
    final[save_cols].to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n完整结果已保存：{out_path}")


if __name__ == '__main__':
    run_screener()