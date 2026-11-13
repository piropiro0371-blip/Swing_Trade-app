"""
売買ルール（戦略）の定義。

どの戦略も「買いシグナルの表（日付×銘柄の True/False）」「優先順位スコア」「売りルール」を返す。
パラメータ（数値の設定）は config/strategy.json と、検証用の GRID から渡される。

戦略の考え方:
- pullback : 長期では上昇トレンドの銘柄が、短期的に売られた所を買い、反発したら売る（押し目買い）
- breakout : 出来高を伴って過去N日の高値を更新した銘柄を買い、逆指値を引き上げながら伸ばす（順張り）
- gap_drift: 大きな窓を開けて上昇し、出来高急増のまま高く引けた銘柄を買う
             （好決算などの材料が出た後もしばらく上がり続けやすい、という傾向を狙う）
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .engine import ExitSpec, Panel


def _regime_mask(p: Panel, sig: pd.DataFrame, use: bool) -> pd.DataFrame:
    if not use:
        return sig
    return sig & p.regime(200).to_numpy()[:, None]


def pullback(p: Panel, base: pd.DataFrame, rsi_max=10, stop_atr=3.0, max_hold=7, regime=True, **_):
    c = p.c
    sig = base & (c > p.sma(200)) & (p.sma(50) > p.sma(200)) & (p.rsi(2) < rsi_max)
    sig = _regime_mask(p, sig, regime)
    score = p.ret(126)  # 半年の上昇率が高い銘柄を優先
    return sig, score, ExitSpec(stop_atr=stop_atr, exit_above_sma=5, max_hold=max_hold)


def breakout(p: Panel, base: pd.DataFrame, lookback=60, vol_mult=2.0, stop_atr=2.0, trail_atr=3.0,
             max_hold=30, regime=True, **_):
    c = p.c
    vr = p.v / p.vol_avg(20)
    sig = base & (c > p.prior_high(lookback)) & (vr >= vol_mult) & (c > p.sma(200))
    # 1日で上がり過ぎた銘柄（+15%超）は翌日の高値づかみになりやすいので除外
    sig = sig & (p.ret(1) < 0.15)
    sig = _regime_mask(p, sig, regime)
    return sig, vr, ExitSpec(stop_atr=stop_atr, trail_atr=trail_atr, max_hold=max_hold)


def gap_drift(p: Panel, base: pd.DataFrame, gap=0.04, vol_mult=3.0, stop_atr=2.0, trail_atr=3.0,
              max_hold=20, regime=False, **_):
    pc = p.c.shift(1)
    vr = p.v / p.vol_avg(20)
    gap_up = p.o / pc - 1
    rng = (p.h - p.l).replace(0, np.nan)
    close_pos = (p.c - p.l) / rng  # その日の値幅の中で終値がどの位置か（1に近いほど高値引け）
    sig = base & (gap_up >= gap) & (p.c >= p.o) & (close_pos >= 0.6) & (vr >= vol_mult)
    sig = sig & (p.ret(1) < 0.20)
    sig = _regime_mask(p, sig, regime)
    return sig, vr, ExitSpec(stop_atr=stop_atr, trail_atr=trail_atr, max_hold=max_hold)


STRATEGIES = {"pullback": pullback, "breakout": breakout, "gap_drift": gap_drift}

LABELS = {"pullback": "押し目買い", "breakout": "高値ブレイク", "gap_drift": "窓開け上昇の追随"}

# 検証で試すパラメータの組み合わせ（多すぎると「たまたま良い組」を拾うので絞っている）
GRID = {
    "pullback": {"rsi_max": [5, 10, 15], "stop_atr": [2.0, 3.0], "max_hold": [5, 10], "regime": [True, False]},
    "breakout": {"lookback": [20, 60, 120], "vol_mult": [1.5, 2.5], "trail_atr": [2.0, 3.0],
                 "max_hold": [20, 40], "regime": [True, False]},
    "gap_drift": {"gap": [0.03, 0.05], "vol_mult": [2.0, 4.0], "trail_atr": [2.0, 3.0],
                  "max_hold": [10, 20], "regime": [True, False]},
}


def build_signal(p: Panel, base: pd.DataFrame, name: str, params: dict, filters: list | None = None):
    """戦略の買いシグナルに、学習で見つけた絞り込み条件（あれば）を重ねる。"""
    sig, score, ex = STRATEGIES[name](p, base, **params)
    if filters:
        from . import features
        sig = sig & features.mask(p, filters)
    return sig, score, ex


def grid(name: str) -> list[dict]:
    g = GRID[name]
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]
