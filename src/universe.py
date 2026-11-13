"""
銘柄一覧（ユニバース）の作成。

日本取引所グループ(JPX)が公開している「東証上場銘柄一覧」を取得し、
プライム・スタンダード・グロースの内国株式だけを config/all_listed.csv に保存する。

使い方:  python -m src.universe
"""

from __future__ import annotations

import io
import os

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "config")
ALL_LISTED = os.path.join(CONFIG_DIR, "all_listed.csv")
UNIVERSE = os.path.join(CONFIG_DIR, "universe.csv")

JPX_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xlsx"
)
MARKETS = ("プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）")


def fetch_all_listed() -> pd.DataFrame:
    """JPXの一覧を取得して 内国株式だけの表 (code, name, market, sector) を返す。"""
    res = requests.get(JPX_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    res.raise_for_status()
    raw = pd.read_excel(io.BytesIO(res.content), dtype=str)
    df = raw[raw["市場・商品区分"].isin(MARKETS)].copy()
    out = pd.DataFrame(
        {
            "code": df["コード"].str.strip(),
            "name": df["銘柄名"].str.strip(),
            "market": df["市場・商品区分"].str.replace("（内国株式）", "", regex=False),
            "sector": df["33業種区分"].str.strip(),
        }
    )
    return out.sort_values("code").reset_index(drop=True)


def load_all_listed() -> pd.DataFrame:
    return pd.read_csv(ALL_LISTED, dtype=str)


def load_universe() -> pd.DataFrame:
    """日次の判定に使う銘柄（流動性で絞り込み済み）。無ければ全上場銘柄。"""
    path = UNIVERSE if os.path.exists(UNIVERSE) else ALL_LISTED
    return pd.read_csv(path, dtype=str)


def to_ticker(code: str) -> str:
    """証券コード → Yahoo Finance のティッカー（例: 7203 → 7203.T）"""
    return f"{code}.T"


if __name__ == "__main__":
    os.makedirs(CONFIG_DIR, exist_ok=True)
    listed = fetch_all_listed()
    listed.to_csv(ALL_LISTED, index=False, encoding="utf-8")
    print(f"{len(listed)} 銘柄を保存: {ALL_LISTED}")
    print(listed["market"].value_counts().to_string())
