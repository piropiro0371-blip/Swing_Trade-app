"""
株式分割の履歴。

取得した株価は、過去の値も「分割後の水準」に直されている（例: 1株を10株に分割した銘柄は、
分割前の株価も10分の1で記録されている）。一方、決算データの1株利益・1株純資産は、発表時点の
株数が基準。両者をそのまま比べると、分割した銘柄が実際より極端に割安に見えてしまう。

ここで分割の履歴を取得し、
  - raw_factor(): ある日の株価を「その日に実際に付いていた値段」に戻す倍率
  - since():      ある日より後に行われた分割の倍率の合計（1株あたりの数字を今の株数基準に直すのに使う）
を提供する。

使い方:  python -m src.splits      （Yahoo Financeから取得し data/cache/splits.parquet に保存）
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yfinance as yf

from .data import BATCH, CACHE_DIR
from .universe import load_universe, to_ticker

PATH = os.path.join(CACHE_DIR, "splits.parquet")


def fetch(codes: list[str], period: str = "3y") -> pd.DataFrame:
    rows = []
    for i in range(0, len(codes), BATCH):
        batch = codes[i:i + BATCH]
        raw = yf.download([to_ticker(c) for c in batch], period=period, interval="1d", auto_adjust=True,
                          actions=True, group_by="ticker", threads=True, progress=False)
        for c in batch:
            try:
                s = raw[to_ticker(c)]["Stock Splits"]
            except KeyError:
                continue
            s = s[(s.notna()) & (s != 0)]
            rows += [(c, pd.Timestamp(d).tz_localize(None).normalize(), float(v)) for d, v in s.items()]
        print(f"  {min(i + BATCH, len(codes))}/{len(codes)} 銘柄処理済み（分割 累計{len(rows)}件）", flush=True)
    return pd.DataFrame(rows, columns=["code", "date", "ratio"])


def load() -> pd.DataFrame:
    if not os.path.exists(PATH):
        return pd.DataFrame(columns=["code", "date", "ratio"])
    return pd.read_parquet(PATH)


def since(asof: pd.Series) -> pd.Series:
    """銘柄ごとに、asof（日付）より後に行われた分割の倍率を掛け合わせたもの。分割が無ければ1。"""
    sp = load()
    out = pd.Series(1.0, index=asof.index)
    for code, g in sp.groupby("code"):
        if code in out.index and pd.notna(asof[code]):
            r = g.loc[g["date"] > pd.Timestamp(asof[code]), "ratio"].prod()
            out[code] = r if r > 0 else 1.0
    return out


def raw_factor(dates: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    """日付×銘柄の表。各日の株価に掛けると、その日に実際に付いていた値段に戻る倍率
    （＝その日より後に行われた分割の倍率の合計）。"""
    f = np.ones((len(dates), len(codes)))
    sp = load()
    pos = {c: k for k, c in enumerate(codes)}
    for code, d, r in sp.itertuples(index=False):
        if code in pos and r > 0:
            f[: dates.searchsorted(pd.Timestamp(d)), pos[code]] *= r   # 分割日より前の日すべてに掛ける
    return pd.DataFrame(f, index=dates, columns=codes)


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = fetch(sorted(load_universe()["code"].tolist()))
    df.to_parquet(PATH, index=False)
    print(f"保存: {PATH}  分割 {len(df)}件 / {df['code'].nunique()}銘柄")
    print(df.sort_values("date").tail(12).to_string(index=False))
