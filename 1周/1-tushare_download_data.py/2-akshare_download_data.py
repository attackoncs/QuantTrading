# -*- coding: utf-8 -*-
"""使用 AkShare 下载寒武纪(688256.SH)历史数据 —— 完全免费，无需注册"""
import os
import akshare as ak
import pandas as pd

STOCK_CODE = '688256'      # AkShare 只用纯数字代码
STOCK_NAME = '寒武纪'
DATA_START  = '20240101'
DATA_END    = '20251231'


def download_stock_data():
    print(f"股票：{STOCK_NAME}({STOCK_CODE}.SH)")
    print(f"日期范围：{DATA_START} 至 {DATA_END}")
    print("-" * 60)

    # 直接调用，无需任何 token / 登录
    df = ak.stock_zh_a_hist(
        symbol     = STOCK_CODE,
        period     = "daily",
        start_date = DATA_START,
        end_date   = DATA_END,
        adjust     = "hfq",        # 后复权；前复权用 "qfq"，不复权用 ""
    )

    if df is None or df.empty:
        print("错误：无法获取历史数据")
        return None

    # AkShare 返回的列名已是中文，统一重命名为英文
    df = (
        df.rename(columns={
            '日期': 'date', '开盘': 'open', '最高': 'high',
            '最低': 'low',  '收盘': 'close', '成交量': 'volume',
        })
        .assign(date=lambda x: pd.to_datetime(x['date']))
        [['date', 'open', 'high', 'low', 'close', 'volume']]
        .sort_values('date')
        .reset_index(drop=True)
    )

    print(f"成功获取 {len(df)} 条数据")
    print(f"日期范围：{df['date'].iloc[0].date()} 至 {df['date'].iloc[-1].date()}")

    output_dir  = os.path.join(os.getcwd(), 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{STOCK_CODE}_SH_daily2.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"数据已保存至：{output_file}")
    print(f"\n收盘价范围：{df['close'].min():.2f} - {df['close'].max():.2f}")
    return output_file


if __name__ == "__main__":
    result = download_stock_data()
    print("完成！" if result else "失败，请检查网络连接。")