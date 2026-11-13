"""
毎日の判定で監視する銘柄（config/universe.csv）を作り直す。週1回の実行を想定。

全上場銘柄の直近3か月の株価を取得し、
  - 株価が予算（資金÷売買単位）の1.5倍以内
  - 売買代金（20日中央値）が基準の半分以上
の銘柄だけを残す（今は条件外でも、近いうちに条件に入りそうな銘柄を含めるため余裕を持たせている）。

使い方:  python -m src.refresh_universe          （Yahoo Financeから取得）
         python -m src.refresh_universe --from-cache bt   （保存済みデータから作る）
"""

from __future__ import annotations

import argparse

from . import data
from .research import load_config
from .universe import ALL_LISTED, UNIVERSE, fetch_all_listed, load_all_listed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", default=None)
    args = ap.parse_args()
    cfg = load_config()

    if args.from_cache:
        listed = load_all_listed()
        prices = data.load(args.from_cache)
    else:
        try:
            listed = fetch_all_listed()
            listed.to_csv(ALL_LISTED, index=False, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[警告] 上場銘柄一覧の更新に失敗。前回の一覧を使います: {e}")
            listed = load_all_listed()
        prices = data.clean(data.download(listed["code"].tolist(), "3mo"))

    last = prices.sort_values("date").groupby("code").tail(20)
    last = last.assign(to=last["close"] * last["volume"])
    g = last.groupby("code").agg(close=("close", "last"), to=("to", "median"), n=("close", "size"))
    ok = g[(g["n"] >= 15) & (g["close"] <= cfg["capital"] / cfg["lot"] * 1.5) & (g["to"] >= cfg["min_turnover"] * 0.5)]
    uni = listed[listed["code"].isin(ok.index)]
    if len(uni) < 100:
        raise SystemExit(f"残った銘柄が {len(uni)} と少なすぎます。取得失敗の可能性があるため更新しません。")
    uni.to_csv(UNIVERSE, index=False, encoding="utf-8")
    print(f"{len(listed)} 銘柄中 {len(uni)} 銘柄を監視対象として保存: {UNIVERSE}")
