"""
原因分析に使う「特徴」（シグナルが出た日の、その銘柄や市場の状態）。

勝ちトレードと負けトレードを分けている要因を探すための切り口。
ここに1行足せば、原因分析・学習ループ・絞り込み条件のすべてで使えるようになる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BENCH, Panel


def _mkt(p: Panel) -> pd.Series:
    return p.c[BENCH]


# 名前: (画面に出す名前, 単位, 計算式)   単位: pct=％ / yen=円 / oku=億円 / x=倍
FEATURES = {
    "atr_pct":     ("値動きの大きさ（1日の値幅÷株価）", "pct", lambda p: p.atr(14) / p.c),
    "turnover":    ("売買代金（20日中央値）", "oku", lambda p: p.turnover(20)),
    "price":       ("株価", "yen", lambda p: p.c),
    "ret5":        ("直近5日の騰落", "pct", lambda p: p.ret(5)),
    "ret20":       ("直近20日の騰落", "pct", lambda p: p.ret(20)),
    "ret126":      ("直近半年の騰落", "pct", lambda p: p.ret(126)),
    "dist200":     ("200日移動平均からの距離", "pct", lambda p: p.c / p.sma(200) - 1),
    "vol_ratio":   ("出来高倍率（当日÷20日平均）", "x", lambda p: p.v / p.vol_avg(20)),
    "mkt_ret20":   ("市場全体の直近20日騰落", "pct", lambda p: _mkt(p) / _mkt(p).shift(20) - 1),
    "mkt_dist200": ("市場全体の200日移動平均からの距離", "pct",
                    lambda p: _mkt(p) / _mkt(p).rolling(200, min_periods=200).mean() - 1),
}


def value(p: Panel, name: str):
    """特徴の値。銘柄ごとの特徴は表（日付×銘柄）、市場全体の特徴は列（日付）で返す。"""
    k = f"feat_{name}"
    if k not in p._cache:
        p._cache[k] = FEATURES[name][2](p)
    return p._cache[k]


def at_trades(p: Panel, t: pd.DataFrame) -> pd.DataFrame:
    """各トレードに、シグナルが出た日の特徴の値を f_名前 の列として付ける。"""
    t = t.copy()
    i = t["i"].to_numpy()
    j = p.codes.get_indexer(t["code"])
    for name in FEATURES:
        v = value(p, name)
        arr = v.to_numpy(dtype="float64")
        t[f"f_{name}"] = arr[i] if arr.ndim == 1 else arr[i, j]
    return t


def mask(p: Panel, filters: list[dict]) -> np.ndarray:
    """絞り込み条件（特徴が lo 以上 hi 未満）をすべて満たす所が True の表。"""
    m = np.ones((len(p.dates), len(p.codes)), dtype=bool)
    for f in filters:
        arr = value(p, f["feature"]).to_numpy(dtype="float64")
        if arr.ndim == 1:
            arr = arr[:, None]
        ok = np.isfinite(arr)
        if f.get("lo") is not None:
            ok = ok & (arr >= f["lo"])
        if f.get("hi") is not None:
            ok = ok & (arr < f["hi"])
        m = m & ok
    return m


def fmt(name: str, v: float) -> str:
    unit = FEATURES[name][1]
    if unit == "pct":
        return f"{v * 100:+.1f}%" if name != "atr_pct" else f"{v * 100:.1f}%"
    if unit == "oku":
        return f"{v / 1e8:.1f}億円"
    if unit == "yen":
        return f"{v:,.0f}円"
    return f"{v:.1f}倍"


def describe(f: dict) -> str:
    """絞り込み条件を日本語の1文にする。"""
    label = FEATURES[f["feature"]][0]
    lo, hi = f.get("lo"), f.get("hi")
    if lo is not None and hi is not None:
        return f"{label}が {fmt(f['feature'], lo)} 以上 {fmt(f['feature'], hi)} 未満"
    if lo is not None:
        return f"{label}が {fmt(f['feature'], lo)} 以上"
    return f"{label}が {fmt(f['feature'], hi)} 未満"


def round_sig(x: float, n: int = 2) -> float:
    """しきい値を有効数字n桁に丸める（毎週わずかに変わって別物として数えられるのを防ぐ）"""
    if x == 0 or not np.isfinite(x):
        return 0.0
    return float(f"{x:.{n}g}")
