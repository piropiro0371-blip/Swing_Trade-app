"""
J-Quants API（日本取引所グループ公式のデータ配信）からの取得。

APIキーは .env の JQUANTS_API_KEY から読む（Git管理外。画面やログに出さない）。
Freeプランは「直近12週間を除く2年分」「1分あたり5回まで」の制限があるため、
呼び出しの間隔を空け、取得結果は data/cache/ に保存して使い回す。
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

from .universe import ROOT

BASE = "https://api.jquants.com/v2"
CACHE_DIR = os.path.join(ROOT, "data", "cache")
INTERVAL = 13.0   # Freeプランの上限（5回/分）に収まる呼び出し間隔（秒）
_last = 0.0


def _key() -> str:
    k = os.environ.get("JQUANTS_API_KEY", "")
    path = os.path.join(ROOT, ".env")
    if not k and os.path.exists(path):
        for line in open(path, encoding="utf-8-sig"):
            if line.startswith("JQUANTS_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise SystemExit("JQUANTS_API_KEY が設定されていません（.env を確認）")
    return k


def get(path: str, params: dict, interval: float = INTERVAL) -> list[dict]:
    """1つのAPIを呼び、続きがあれば最後まで取得して data の中身を返す。"""
    global _last
    out, params = [], dict(params)
    while True:
        wait = interval - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.time()
        res = requests.get(f"{BASE}{path}", params=params, headers={"x-api-key": _key()}, timeout=60)
        if res.status_code == 429:   # 呼び出し過多。少し待ってやり直す
            time.sleep(60)
            continue
        if res.status_code != 200:
            raise RuntimeError(f"J-Quants {path} {res.status_code}: {res.text[:300]}")
        body = res.json()
        rows = next((v for v in body.values() if isinstance(v, list)), [])
        out.extend(rows)
        if not body.get("pagination_key"):
            return out
        params["pagination_key"] = body["pagination_key"]


def fins_summary(code: str | None = None, date: str | None = None) -> pd.DataFrame:
    """決算短信などの要約（発表日・売上・利益・会社予想など）。code か date のどちらかを指定。"""
    params = {k: v for k, v in (("code", code), ("date", date)) if v}
    return pd.DataFrame(get("/fins/summary", params))


FINS_PATH = os.path.join(CACHE_DIR, "fins.parquet")
FINS_DONE = os.path.join(CACHE_DIR, "fins_done.txt")
FINS_COLS = ["DiscDate", "DiscTime", "Code", "DocType", "CurPerType", "CurPerEn", "CurFYEn",
             "Sales", "OP", "NP", "EPS", "FSales", "FOP", "FNP", "FEPS",
             "NxFSales", "NxFOP", "NxFNp", "NxFEPS", "DivAnn", "FDivAnn", "BPS", "Eq", "ShOutFY"]


def fetch_fins(dates: list[str]) -> pd.DataFrame:
    """指定した日付の開示を順に取得して fins.parquet に貯める。取得済みの日は飛ばす（再開できる）。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    done = set(open(FINS_DONE, encoding="utf-8").read().split()) if os.path.exists(FINS_DONE) else set()
    have = pd.read_parquet(FINS_PATH) if os.path.exists(FINS_PATH) else pd.DataFrame(columns=FINS_COLS)
    todo = [d for d in dates if d not in done]
    print(f"取得済み {len(done)}日 / 残り {len(todo)}日", flush=True)
    buf = []
    for n, d in enumerate(todo, 1):
        try:
            df = fins_summary(date=d)
        except RuntimeError as e:
            print(f"  {d}: 取得できず（{str(e)[:120]}）", flush=True)
            continue
        if len(df):
            buf.append(df.reindex(columns=FINS_COLS).astype(str))
        done.add(d)
        if n % 20 == 0 or n == len(todo):
            if buf:
                have = pd.concat([have, *buf], ignore_index=True)
                buf = []
            have.to_parquet(FINS_PATH, index=False)
            with open(FINS_DONE, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(done)))
            print(f"  {n}/{len(todo)}日 処理済み（累計 {len(have):,} 件）", flush=True)
    return have


if __name__ == "__main__":
    # Freeプランで取れる範囲: 今日から2年前 〜 12週間前
    from . import data
    today = pd.Timestamp.now().normalize()
    lo, hi = today - pd.DateOffset(years=2) + pd.Timedelta(days=1), today - pd.Timedelta(weeks=12)
    days = sorted(data.load("bt")["date"].unique())
    fetch_fins([pd.Timestamp(d).strftime("%Y-%m-%d") for d in days if lo <= pd.Timestamp(d) <= hi])
