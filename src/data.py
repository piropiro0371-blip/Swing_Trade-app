"""
株価データ（日足）の取得と保存。

Yahoo Finance から日足を取得し、縦長の表
  [date, code, open, high, low, close, volume]
にして parquet（高速に読める表形式ファイル）で data/cache/ に保存する。
株価は分割・配当を調整済みの値（auto_adjust=True）。

使い方:
  python -m src.data --period 10y --tag bt      # バックテスト用（全上場銘柄・10年）
  python -m src.data --period 2y  --tag daily   # 日次判定用（universe.csv・2年）
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import yfinance as yf

from .universe import ROOT, load_all_listed, load_universe, to_ticker

CACHE_DIR = os.path.join(ROOT, "data", "cache")
BENCHMARK = "1306"  # TOPIX連動ETF（市場全体の物差しとして使う）
BATCH = 150


def cache_path(tag: str) -> str:
    return os.path.join(CACHE_DIR, f"prices_{tag}.parquet")


def _download_batch(codes: list[str], period: str) -> pd.DataFrame:
    tickers = [to_ticker(c) for c in codes]
    raw = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    frames = []
    for code, tk in zip(codes, tickers):
        try:
            sub = raw[tk] if len(tickers) > 1 else raw
        except KeyError:
            continue
        if isinstance(sub.columns, pd.MultiIndex):
            sub = sub.droplevel(0, axis=1)
        sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
        sub = sub[(sub["Close"] > 0) & (sub["Open"] > 0)]
        if sub.empty:
            continue
        sub = sub.rename(columns=str.lower).reset_index()
        sub = sub.rename(columns={sub.columns[0]: "date"})
        sub["code"] = code
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download(codes: list[str], period: str) -> pd.DataFrame:
    """複数銘柄をまとめて取得。失敗したまとまりは1回だけ再試行する。"""
    out = []
    n = len(codes)
    for i in range(0, n, BATCH):
        batch = codes[i : i + BATCH]
        df = pd.DataFrame()
        for attempt in range(2):
            try:
                df = _download_batch(batch, period)
                if not df.empty:
                    break
            except Exception as e:  # noqa: BLE001
                print(f"  [警告] 取得失敗 ({attempt + 1}回目): {e}", flush=True)
            time.sleep(5)
        got = df["code"].nunique() if not df.empty else 0
        print(f"  {min(i + BATCH, n)}/{n} 銘柄処理済み（このまとまりで {got} 銘柄取得）", flush=True)
        if not df.empty:
            out.append(df)
        time.sleep(1)
    big = pd.concat(out, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"]).dt.tz_localize(None).dt.normalize()
    for col in ("open", "high", "low", "close"):
        big[col] = big[col].astype("float32")
    big["volume"] = big["volume"].astype("float64")
    return big.sort_values(["code", "date"]).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """誤データの除去。
    - 終値が前後10日の中央値から大きく外れた日（半分未満 or 2倍超）は誤りとみなす
      （値幅制限があるので、現実には1日でここまで動かない）
    - 物差しのETFはその日だけ取り除く。個別銘柄は1日でも該当したら銘柄ごと対象から外す
    """
    med = df.groupby("code")["close"].transform(lambda s: s.rolling(21, center=True, min_periods=5).median())
    ratio = df["close"] / med
    bad = (ratio < 0.5) | (ratio > 2.0)
    bad_codes = set(df.loc[bad, "code"]) - {BENCHMARK}
    out = df[~df["code"].isin(bad_codes) & ~(bad & (df["code"] == BENCHMARK))]
    return out.reset_index(drop=True)


def load(tag: str) -> pd.DataFrame:
    return clean(pd.read_parquet(cache_path(tag)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="2y")
    ap.add_argument("--tag", default="daily")
    ap.add_argument("--all", action="store_true", help="全上場銘柄を対象にする")
    args = ap.parse_args()

    listed = load_all_listed() if (args.all or args.tag == "bt") else load_universe()
    codes = sorted(set(listed["code"].tolist()) | {BENCHMARK})
    print(f"{len(codes)} 銘柄・期間 {args.period} を取得します", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    prices = download(codes, args.period)
    prices.to_parquet(cache_path(args.tag), index=False)
    print(f"保存: {cache_path(args.tag)}  {prices['code'].nunique()} 銘柄 / {len(prices):,} 行")
