# -*- coding: utf-8 -*-
"""
CASE：获取贵州茅台的财务指标

- 市值(Market Cap)：股价 * 总股本，把公司整个买下来要花多少钱
- PE(市盈率)：市值/净利润 = 股价/EPS，回本需要多少年
- PB(市净率)：股价/每股净资产，破产了还能剩多少

本脚本使用 Tushare daily_basic 接口获取真实基本面数据（需环境变量 TUSHARE_TOKEN）。
同时用本地 data/600519_SH_daily.csv 最后一日作为基准日期；若需与 Tushare 收盘价一致，
可直接采用 daily_basic 返回的 close。
"""
# -*- coding: utf-8 -*-
"""
CASE：获取贵州茅台的财务指标（AkShare 版）
pip install akshare

接口：
  ak.stock_individual_info_em(symbol)  → 总股本(股)、总市值(元)、行业、上市时间
  ak.stock_zh_a_spot_em()             → 实时最新价、市盈率-动态、市净率
"""
import akshare as ak
import pandas as pd

STOCK_NAME = '贵州茅台'
SYMBOL     = '600519'   # 纯6位代码，AkShare 不加后缀


def run_demo():
    # ── 请求1：个股基本信息（总股本、总市值、行业） ──────
    info = ak.stock_individual_info_em(symbol=SYMBOL)
    # 返回 item/value 两列，转成字典方便取值
    info_dict = dict(zip(info['item'], info['value']))

    total_shares = float(info_dict.get('总股本', 0))       # 单位：股
    total_mv_yuan = float(info_dict.get('总市值', 0))      # 单位：元
    industry = info_dict.get('行业', '-')
    list_date = str(info_dict.get('上市时间', '-'))

    # ── 请求2：全市场实时行情，取茅台一行 ────────────────
    spot = ak.stock_zh_a_spot_em()
    row  = spot[spot['代码'] == SYMBOL].iloc[0]

    price = float(row['最新价'])
    pe    = float(row['市盈率-动态']) if pd.notna(row['市盈率-动态']) else None
    pb    = float(row['市净率'])      if pd.notna(row['市净率'])      else None
    today = pd.Timestamp.today().strftime('%Y-%m-%d')

    # ── 派生指标 ──────────────────────────────────────────
    market_cap_yi = total_mv_yuan / 1e8                    # 元→亿元
    total_shares_yi = total_shares / 1e8                   # 股→亿股
    eps = price / pe if pe and pe > 0 else None            # 每股收益（反推）
    bps = price / pb if pb and pb > 0 else None            # 每股净资产（反推）

    # ── 输出 ──────────────────────────────────────────────
    print(f"{'=' * 55}")
    print(f"  {STOCK_NAME}({SYMBOL})  基本面指标  {today}")
    print(f"{'=' * 55}")
    print(f"  行业       ：{industry}（上市：{list_date}）")
    print(f"  最新价     ：{price:.2f} 元")
    print(f"  总股本     ：{total_shares_yi:.4f} 亿股")
    print(f"  总市值     ：{market_cap_yi:,.2f} 亿元")
    print(f"  验证市值   ：股价 × 总股本 = {price:.2f} × {total_shares_yi:.4f}亿 "
          f"= {price * total_shares / 1e8:,.2f} 亿元")
    print('-' * 55)
    print(f"  PE（市盈率）：{pe:.2f}  → 按当前盈利约 {pe:.0f} 年回本" if pe else "  PE：无数据")
    print(f"  PB（市净率）：{pb:.2f}  {'（低于1为破净）' if pb and pb < 1 else ''}" if pb else "  PB：无数据")
    print(f"  EPS（反推） ：{eps:.2f} 元/股" if eps else "  EPS：无法计算")
    print(f"  BPS（反推） ：{bps:.2f} 元/股" if bps else "  BPS：无法计算")
    print('-' * 55)
    print("  说明：PE 越低越便宜，成长股可容忍高 PE；PB<1 为破净，")
    print("        常见于银行、钢铁；市值=股价×总股本，是收购整家公司的成本。")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    run_demo()