# -*- coding: utf-8 -*-
"""
基本面选股 -- 数据下载脚本（AkShare 版）
pip install akshare

接口：
  股票列表  → ak.stock_zh_a_spot_em()            东方财富实时行情（含代码/名称/PE/PB/市值）
  财务指标  → ak.stock_financial_abstract(symbol)  东方财富财务摘要（ROE/BPS/EPS/净利润增速等）
  负债率    → ak.stock_balance_sheet_by_yearly_em(symbol) 资产负债表年报（资产负债率）
  现金流    → ak.stock_cash_flow_sheet_by_yearly_em(symbol) 现金流量表（经营现金流/净利润）

保存文件（data/ 目录）：
  stock_basic.csv          股票列表（代码/名称）
  daily_basic_latest.csv   最新行情估值（close/pb/pe/total_mv，万元）
  fina_indicator_pool.csv  财务指标（roe/bps/eps/debt_to_assets/ocf_to_profit/netprofit_yoy）
  | 数据                | 接口                                         | 优势                                          |
| ----------------- | ------------------------------------------ | ------------------------------------------- |
| 股票列表 + PE/PB/市值   | stock_zh_a_spot_em()                       | 一次请求同时获得行情和估值，替代原版两个步骤 cloud.tencent        |
| ROE/BPS/EPS/净利润增速 | stock_financial_abstract(symbol)           | 东方财富，列名稳定，无需按年份构造参数 developer.cloud.tencent |
| 资产负债率             | stock_balance_sheet_by_yearly_em(symbol)   | 年报资产负债表，直接取或自行计算 tessl                      |
| 经营现金流/净利润         | stock_cash_flow_sheet_by_yearly_em(symbol) | 与财务摘要同源，避免 VIP 接口限制 tessl                   |
"""
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import akshare as ak

# ── 配置 ──────────────────────────────────────────────────
NUM_ANNUAL_PERIODS = 3       # 最近 N 期年报
BATCH_SIZE         = 50      # 每批股票数，批完即存（断点续跑）
NUM_WORKERS        = 3       # 并发线程数
REQUEST_INTERVAL   = 0.6     # 节流间隔（秒）

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_akshare")
FINA_COLS = ["ts_code", "end_date", "roe", "bps", "eps",
             "debt_to_assets", "ocf_to_profit", "netprofit_yoy"]

# 目标年报期列表，如 ["2025", "2024", "2023"]
def _target_years(n: int = NUM_ANNUAL_PERIODS) -> list[str]:
    y = datetime.now().year
    if datetime.now().month < 4:
        y -= 1
    return [str(y - i) for i in range(n)]


# ── 工具 ──────────────────────────────────────────────────
def _to_float(v) -> float | None:
    if v is None: return None
    s = str(v).replace("%", "").replace(",", "").strip()
    try:    return float(s)
    except: return None

def _code_to_tscode(code: str) -> str:
    if code.startswith(("60", "68", "900")): return code + ".SH"
    if code.startswith(("00", "30", "200")): return code + ".SZ"
    return code + ".BJ"

# ── Step 1：股票列表 + 最新行情（一次请求两用） ─────────────
def download_stock_basic_and_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ak.stock_zh_a_spot_em() 一次性返回全市场实时行情。
    stock_basic 和 daily_basic_latest 都从这里生成，无需两次请求。
    注意：总市值单位为元，转换为万元与原版一致。
    """
    print("[Step 1] 获取 A 股行情 + 估值（东方财富全市场快照）...")
    try:
        raw = ak.stock_zh_a_spot_em()
    except Exception as e:
        print(f"  失败: {e}"); return pd.DataFrame(), pd.DataFrame()

    # 最新 akshare 列名
    raw = raw.rename(columns={
        "代码": "symbol", "名称": "name",
        "最新价": "close", "市盈率-动态": "pe",
        "市净率": "pb",   "总市值": "total_mv_yuan",
    })

    # 只保留沪深（过滤北交所）
    raw = raw[raw["symbol"].str.match(r"^(60|00|30|68|20|90)\d{4}$")].copy()
    raw["ts_code"] = raw["symbol"].apply(_code_to_tscode)
    for c in ("close", "pe", "pb", "total_mv_yuan"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["total_mv"] = (raw["total_mv_yuan"] / 1e4).round(2)  # 元→万元

    # stock_basic
    stock_basic = raw[["ts_code", "symbol", "name"]].copy()

    # daily_basic_latest（日期用今天）
    today = datetime.now().strftime("%Y%m%d")
    daily = raw[["ts_code", "close", "pe", "pb", "total_mv"]].copy()
    daily.insert(1, "trade_date", today)
    daily["close"] = daily["close"].round(2)

    os.makedirs(DATA_DIR, exist_ok=True)
    p1 = os.path.join(DATA_DIR, "stock_basic.csv")
    p2 = os.path.join(DATA_DIR, "daily_basic_latest.csv")
    stock_basic.to_csv(p1, index=False, encoding="utf-8-sig")
    daily.to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"  股票列表 {len(stock_basic)} 只 -> {p1}")
    print(f"  行情估值 {len(daily)} 只 -> {p2}")

    mt = daily[daily["ts_code"] == "600519.SH"]
    if not mt.empty:
        r = mt.iloc[0]
        print(f"  [校验] 茅台: close={r.close}, pb={r.pb}, pe={r.pe}, mv={r.total_mv}万")
    return stock_basic, daily


# ── Step 2：财务指标（断点续跑） ──────────────────────────
_lock    = threading.Lock()
_last_t  = [0.0]
_err_set = set()
_err_lk  = threading.Lock()

def _throttle():
    with _lock:
        wait = _last_t[0] + REQUEST_INTERVAL - time.time()
        if wait > 0: time.sleep(wait)
        _last_t[0] = time.time()


def _fetch_one(symbol: str, ts_code: str, years: list[str]) -> list[dict]:
    """
    拉取单只股票多年财务指标。
    ak.stock_financial_abstract 返回列（东方财富）：
      报告期、每股收益(元)、每股净资产(元)、净资产收益率(%)、
      净利润(万元)、净利润同比增长(%)、每股经营性现金流(元) …

    资产负债率来自 ak.stock_balance_sheet_by_yearly_em：
      报告期、资产总计、负债合计 → 自行计算，或直接取"资产负债率(%)"列（若有）
    """
    results = []
    _throttle()
    try:
        df_abs = ak.stock_financial_abstract(symbol=symbol)
        if df_abs is None or df_abs.empty:
            return results
        # 报告期格式通常为 "2024年报" / "2024三季报"
        df_abs = df_abs[df_abs["报告期"].str.contains("年报", na=False)].copy()
        df_abs["year"] = df_abs["报告期"].str[:4]
    except Exception as e:
        _log_err(str(e)); return results

    # 资产负债率（年报资产负债表）
    debt_map: dict[str, float | None] = {}
    try:
        _throttle()
        df_bs = ak.stock_balance_sheet_by_yearly_em(symbol=symbol)
        if df_bs is not None and not df_bs.empty:
            # 列名含 "资产负债率" 则直接用；否则用 负债合计/资产总计
            if "资产负债率" in df_bs.columns:
                for _, r in df_bs[["REPORT_DATE", "资产负债率"]].dropna().iterrows():
                    y = str(r["REPORT_DATE"])[:4]
                    debt_map[y] = _to_float(r["资产负债率"])
            elif {"负债合计", "资产总计", "REPORT_DATE"} <= set(df_bs.columns):
                for _, r in df_bs[["REPORT_DATE", "负债合计", "资产总计"]].dropna().iterrows():
                    y = str(r["REPORT_DATE"])[:4]
                    total = _to_float(r["资产总计"])
                    debt  = _to_float(r["负债合计"])
                    if total and total != 0:
                        debt_map[y] = round(debt / total * 100, 4)
    except Exception as e:
        _log_err(str(e))

    # 经营现金流/净利润（现金流量表）
    ocf_map: dict[str, float | None] = {}
    try:
        _throttle()
        df_cf = ak.stock_cash_flow_sheet_by_yearly_em(symbol=symbol)
        if df_cf is not None and not df_cf.empty:
            # 尝试直接取比率列；否则用 经营活动产生的现金流量净额 / 净利润
            ratio_col = next((c for c in df_cf.columns if "经营现金" in c and "净利润" in c), None)
            if ratio_col:
                for _, r in df_cf[["REPORT_DATE", ratio_col]].dropna().iterrows():
                    y = str(r["REPORT_DATE"])[:4]
                    ocf_map[y] = _to_float(r[ratio_col])
            else:
                ocf_col = next((c for c in df_cf.columns if "经营活动" in c and "现金流量" in c), None)
                net_col = next((c for c in df_cf.columns if "净利润" in c), None)
                if ocf_col and net_col:
                    for _, r in df_cf[["REPORT_DATE", ocf_col, net_col]].dropna().iterrows():
                        y   = str(r["REPORT_DATE"])[:4]
                        ocf = _to_float(r[ocf_col])
                        net = _to_float(r[net_col])
                        if net and net != 0:
                            ocf_map[y] = round(ocf / net, 4) if ocf else None
    except Exception as e:
        _log_err(str(e))

    # 组装目标年份记录
    col_map = {
        "每股收益(元)":      "eps",
        "每股净资产(元)":    "bps",
        "净资产收益率(%)":   "roe",
        "净利润同比增长(%)": "netprofit_yoy",
    }
    # 容错：兼容不同版本列名变体
    alt_map = {
        "基本每股收益":      "eps",
        "每股净资产":        "bps",
        "净资产收益率":      "roe",
        "净利润增长率":      "netprofit_yoy",
    }
    for yr in years:
        row_df = df_abs[df_abs["year"] == yr]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        rec = {"ts_code": ts_code, "end_date": f"{yr}1231"}
        for src_col, dst_col in {**col_map, **alt_map}.items():
            if dst_col not in rec and src_col in row.index:
                rec[dst_col] = _to_float(row[src_col])
        for dst_col in ("eps", "bps", "roe", "netprofit_yoy"):
            rec.setdefault(dst_col, None)
        rec["debt_to_assets"] = debt_map.get(yr)
        rec["ocf_to_profit"]  = ocf_map.get(yr)
        results.append(rec)

    return results


def _log_err(msg: str):
    with _err_lk:
        key = msg[:80]
        if key not in _err_set:
            _err_set.add(key)
            print(f"  [API] {msg[:200]}")


def _save(out_path: str, existing: pd.DataFrame, new_records: list) -> pd.DataFrame:
    """合并去重后写 CSV，返回最新全量 DataFrame"""
    if not new_records:
        return existing
    new_df = pd.DataFrame(new_records)
    result = (pd.concat([existing, new_df], ignore_index=True)
              if existing is not None and not existing.empty
              else new_df)
    result = (result[[c for c in FINA_COLS if c in result.columns]]
              .drop_duplicates(["ts_code", "end_date"], keep="last"))
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def download_fina_indicators(stock_list: list[str]) -> pd.DataFrame:
    """
    批量下载财务指标，支持断点续跑：已有 (ts_code, end_date) 自动跳过。
    """
    years    = _target_years()
    out_path = os.path.join(DATA_DIR, "fina_indicator_pool.csv")

    # 加载已有
    existing, done_keys = None, set()
    if os.path.exists(out_path):
        try:
            existing = pd.read_csv(out_path, dtype=str, encoding="utf-8-sig")
            done_keys = set(zip(existing["ts_code"], existing["end_date"]))
        except Exception:
            pass

    # 需要请求的股票（任意一年缺失即重拉该股票所有年份）
    to_fetch = [
        code for code in stock_list
        if any((code, f"{y}1231") not in done_keys for y in years)
    ]

    print(f"\n[Step 2] 财务指标（{','.join(years)} 年报）："
          f"共 {len(stock_list)} 只，需拉取 {len(to_fetch)} 只，"
          f"跳过 {len(stock_list) - len(to_fetch)} 只...")

    if not to_fetch:
        print(f"  全部已存在，跳过。当前 {len(existing)} 条 -> {out_path}")
        return existing or pd.DataFrame()

    result   = existing
    total_ok = total_fail = 0
    num_batches = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
    _err_set.clear()

    for b_idx in range(num_batches):
        batch   = to_fetch[b_idx * BATCH_SIZE:(b_idx + 1) * BATCH_SIZE]
        records, done = [], 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            # 注意：_fetch_one 内部已包含节流，不依赖外部 semaphore
            futs = {ex.submit(_fetch_one, code.split(".")[0], code, years): code
                    for code in batch}
            for fut in as_completed(futs):
                recs = fut.result()
                records.extend(recs)
                done += 1
                if done % 20 == 0 or done == len(batch):
                    print(f"  批 {b_idx+1}/{num_batches}  {done}/{len(batch)} 只完成...")

        n_ok   = sum(1 for r in records if r)
        n_fail = len(batch) - len({r["ts_code"] for r in records}) if records else len(batch)
        total_ok += len({r["ts_code"] for r in records})
        total_fail += n_fail

        # 读最新文件再合并（防多进程竞争）
        try:
            cur = pd.read_csv(out_path, dtype=str, encoding="utf-8-sig") if os.path.exists(out_path) else pd.DataFrame()
        except Exception:
            cur = result or pd.DataFrame()

        result = _save(out_path, cur, records)
        print(f"  批 {b_idx+1}/{num_batches} 保存完毕，当前共 {len(result)} 条")

    # 校验
    print(f"\n  合计成功 {total_ok} 只，无数据/失败 {total_fail} 只")
    mt = result[result["ts_code"] == "600519.SH"] if result is not None else pd.DataFrame()
    print("  [校验] 贵州茅台：")
    if not mt.empty:
        r = mt.iloc[0]
        print(f"    {r['end_date']}: roe={r.get('roe')}, debt={r.get('debt_to_assets')}, bps={r.get('bps')}")
    else:
        print("    无数据")
    return result


# ── 主流程 ────────────────────────────────────────────────
def run_download():
    sep = "=" * 60
    print(f"{sep}\n基本面选股 -- 数据下载（AkShare 版）\n{sep}")
    print(f"数据目录: {DATA_DIR}")
    print(sep)

    stock_df, daily_df = download_stock_basic_and_daily()
    if stock_df.empty:
        print("错误：无法获取股票列表"); return

    fina_df = download_fina_indicators(stock_df["ts_code"].tolist())

    print(f"\n{sep}\n下载完成\n{sep}")
    for fname in ("stock_basic.csv", "daily_basic_latest.csv", "fina_indicator_pool.csv"):
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            print(f"  {fname:<32} {os.path.getsize(p) / 1024:.0f} KB")
    print(sep)


if __name__ == "__main__":
    run_download()