# -*- coding: utf-8 -*-
"""
QMT数据下载脚本
使用xtquant下载贵州茅台股票（600519.SH）的历史数据
保存到本地CSV文件，供后续策略分析使用

注意：运行此脚本需要安装QMT并配置好xtquant
"""
import os
import pandas as pd
from xtquant import xtdata

# 数据下载参数配置
STOCK_CODE = '600519.SH'   # 贵州茅台股票代码
STOCK_NAME = '贵州茅台'
DATA_START = '20240101'   # 数据开始日期（需要足够的历史数据计算MACD）
DATA_END = '20251231'     # 数据结束日期
FIELDS      = [ 'open', 'high', 'low', 'close', 'volume']

def download_stock_data():
    """
    下载股票历史数据并保存到CSV文件
    """
    print(f"开始下载股票数据")
    print(f"股票：{STOCK_NAME}({STOCK_CODE})")
    print(f"日期范围：{DATA_START} 至 {DATA_END}")
    print("-" * 60)
    
    try:
        # 步骤1：缓存历史数据到本地（同步执行，完成后自动返回，无需 sleep）
        print("步骤1：下载历史数据...")
        xtdata.download_history_data(
            stock_code = STOCK_CODE,
            period     = '1d',
            start_time = DATA_START,
            end_time   = DATA_END,
        )
        print("下载完成")

        # 步骤2：用 get_market_data_ex 一次性读取所有字段
        # 返回结构：{股票代码: DataFrame}，DataFrame 列为各字段，行为各时间点
        print("\n步骤2：读取历史数据...")
        res = xtdata.get_market_data_ex(
            field_list    = FIELDS,        # ✅ 直接指定字段，无需事后筛选
            stock_list    = [STOCK_CODE],
            period        = '1d',
            start_time    = DATA_START,
            end_time      = DATA_END,
            dividend_type = 'front',       # 前复权
            fill_data     = True,
        )

        if not res or STOCK_CODE not in res:
            print("错误：无法获取历史数据")
            return None

        # get_market_data_ex 直接返回 {code: DataFrame}，无需逐字段拼接
        df = res[STOCK_CODE].copy()         # ✅ 一行完成，替代原来十几行的字段提取

        if df.empty:
            print("错误：获取到的 DataFrame 为空，请检查本地缓存是否正常")
            print(f"  调试信息 - res keys: {list(res.keys())}")
            return None
        # 步骤3：数据预处理
        df = (
            df.reset_index()                         # 将时间 index 变为列
              .rename(columns={df.index.name or 'index': 'date'})  # 适配不同版本的索引名
              .assign(date=lambda x: pd.to_datetime(
                  x['date'].astype(str).str[:8],     # 兼容 '20240103' 和 '20240103000000' 两种格式
                  format='%Y%m%d', errors='coerce'
              ))
              .dropna(subset=['date', 'close'])
              .sort_values('date')
              .reset_index(drop=True)
        )

        print(f"成功获取 {len(df)} 条历史数据")
        print(f"数据日期范围：{df['date'].iloc[0].date()} 至 {df['date'].iloc[-1].date()}")

        # 步骤4：保存到CSV文件
        print("\n步骤3：保存数据到CSV文件...")
        output_dir  = os.path.join(os.getcwd(), 'data')
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f'{STOCK_CODE.replace(".", "_")}_daily.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"数据已保存至：{output_file}")

        # 数据预览与统计
        print("\n数据预览（前5行）：")
        print(df.head().to_string(index=False))
        print("\n数据预览（后5行）：")
        print(df.tail().to_string(index=False))

        print("\n数据统计信息：")
        print(f"  总记录数：{len(df)}")
        print(f"  收盘价范围：{df['close'].min():.2f} - {df['close'].max():.2f}")
        if 'volume' in df.columns:
            print(f"  成交量范围：{df['volume'].min():,.0f} - {df['volume'].max():,.0f}")

        return output_file

    except Exception as e:
        print(f"下载数据过程中发生错误：{e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = download_stock_data()
    
    if result:
        print("\n" + "=" * 60)
        print("数据下载完成!")
        print(f"数据文件：{result}")
        print("=" * 60)
        print("\n提示：现在可以运行 6b-macd_strategy_analysis.py 进行策略分析")
    else:
        print("\n数据下载失败，请检查错误信息。")