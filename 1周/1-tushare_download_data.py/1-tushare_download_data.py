# -*- coding: utf-8 -*-
"""
Tushare数据下载脚本
使用tushare下载寒武纪股票（688256.SH）的历史数据
保存到本地CSV文件，供后续策略分析使用

注意：运行此脚本需要安装tushare并配置好TUSHARE_TOKEN环境变量
"""
import os
import pandas as pd
import tushare as ts
from datetime import datetime


# 数据下载参数配置
STOCK_CODE = '688256.SH'  # 寒武纪股票代码
STOCK_NAME = '寒武纪'
DATA_START = '20240101'   # 数据开始日期
DATA_END = '20251231'     # 数据结束日期

# API 直接返回的字段名 → 目标字段名映射
FIELD_MAP = {
    'trade_date': 'date',
    'open':       'open',
    'high':       'high',
    'low':        'low',
    'close':      'close',
    'vol':        'volume',
}

def download_stock_data():
    """
    下载股票历史数据并保存到CSV文件
    """
    print(f"开始下载股票数据")
    print(f"股票：{STOCK_NAME}({STOCK_CODE})")
    print(f"日期范围：{DATA_START} 至 {DATA_END}")
    print("-" * 60)
    
     # 步骤1：初始化 tushare —— 直接通过 pro_api(token) 一步完成
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        print("错误：未找到环境变量 TUSHARE_TOKEN")
        print("请设置环境变量：export TUSHARE_TOKEN=your_token")
        return None

    pro = ts.pro_api(token)   # ✅ API 已提供此方式，无需再 set_token
    print(f"读取到的Token：[{token}]")   # 用方括号包住，方便看有没有空格/换行
    print("tushare初始化成功")
        
    # 步骤2：下载历史数据
    print("\n步骤2：下载历史数据...")
        # tushare的daily接口需要股票代码格式为：688256.SH
        # 日期格式：YYYYMMDD
    df = pro.query(
        'daily',
        ts_code   = STOCK_CODE,
        start_date= DATA_START,
        end_date  = DATA_END,
        fields    = ','.join(FIELD_MAP.keys()),   # ✅ 让 API 只返回需要的列
    )
        
    if df is None or df.empty:
        print("错误：无法获取历史数据")
        return None

    print(f"成功获取 {len(df)} 条历史数据")
        
       # 步骤3：数据预处理 —— rename 直接用映射字典，不再逐列检查
    df = (
        df
        .rename(columns=FIELD_MAP)                                     # ✅ 一行完成重命名
        .assign(date=lambda x: pd.to_datetime(x['date'], format='%Y%m%d', errors='coerce'))
        .dropna(subset=['date', 'close'])
        .sort_values('date')
        .reset_index(drop=True)
    )
        
    print(f"数据日期范围：{df['date'].iloc[0].date()} 至 {df['date'].iloc[-1].date()}")
        
    # 步骤4：保存到CSV文件
    print("\n步骤3：保存数据到CSV文件...")
    output_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(output_dir, exist_ok=True)
        
    # 保存完整数据
    output_file = os.path.join(output_dir, f'{STOCK_CODE.replace(".", "_")}_daily.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"数据已保存至：{output_file}")
        
    # 显示数据预览
    print("\n数据预览（前5行）：")
    print(df.head().to_string(index=False))
    print("\n数据预览（后5行）：")
    print(df.tail().to_string(index=False))
        
    # 显示数据统计
    print("\n数据统计信息：")
    print(f"  总记录数：{len(df)}")
    print(f"  收盘价范围：{df['close'].min():.2f} - {df['close'].max():.2f}")
    if 'volume' in df.columns:
        print(f"  成交量范围：{df['volume'].min():,.0f} - {df['volume'].max():,.0f}")
    
    return output_file


if __name__ == "__main__":
    result = download_stock_data()
    
    if result:
        print("\n" + "=" * 60)
        print("数据下载完成!")
        print(f"数据文件：{result}")
        print("=" * 60)
    else:
        print("\n数据下载失败，请检查错误信息。")
