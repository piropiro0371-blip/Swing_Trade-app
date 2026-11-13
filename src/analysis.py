"""
銘柄分析: テクニカル指標（株価と出来高から計算する指標）と、その読み取り。

ここで出す読み取り（「上昇トレンド」「買われすぎ」など）は、指標の一般的な見方をそのまま当てはめたもの。
このシステムの検証では、これらの指標に市場平均を上回る力は確認できていない。判断の材料の一つとして使う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Panel


def _c(p: Panel, key: str, fn):
    if key not in p._cache:
        p._cache[key] = fn()
    return p._cache[key]


def ema(p: Panel, n: int) -> pd.DataFrame:
    return _c(p, f"ema{n}", lambda: p.c.ewm(span=n, adjust=False, min_periods=n).mean())


def macd_hist(p: Panel) -> pd.DataFrame:
    """MACDヒストグラム（短期と長期の平均の差が、さらにその平均より上か下か）。株価に対する割合で表す。"""
    def f():
        macd = ema(p, 12) - ema(p, 26)
        return (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / p.c
    return _c(p, "macd_hist", f)


def bb_pos(p: Panel) -> pd.DataFrame:
    """ボリンジャーバンド内の位置。+1 で上の線（+2σ）、-1 で下の線（-2σ）、0 が中心（20日平均）。"""
    return _c(p, "bb_pos", lambda: (p.c - p.sma(20)) / (2 * p.c.rolling(20, min_periods=20).std()))


def dev(p: Panel, n: int) -> pd.DataFrame:
    """n日移動平均からの乖離率（どれだけ離れているか）"""
    return _c(p, f"dev{n}", lambda: p.c / p.sma(n) - 1)


def hi52(p: Panel) -> pd.DataFrame:
    """52週（250営業日）高値からの距離。0 なら高値圏、マイナスが大きいほど高値から遠い。"""
    return _c(p, "hi52", lambda: p.c / p.h.rolling(250, min_periods=120).max() - 1)


def _days_since_cross(diff: pd.DataFrame, look: int = 20) -> pd.Series:
    """差がプラスに転じて（またはマイナスに転じて）から何日か。プラス=上抜け、マイナス=下抜け。無ければ0。"""
    sign = np.sign(diff.iloc[-(look + 1):].to_numpy())
    out = np.zeros(sign.shape[1])
    for k in range(1, look + 1):                 # k日前に符号が変わったか（新しい方を優先）
        cur, prev = sign[-k], sign[-k - 1]
        hit = (out == 0) & (cur != prev) & (cur != 0) & (prev != 0) & np.isfinite(cur) & np.isfinite(prev)
        out = np.where(hit, cur * k, out)
    return pd.Series(out, index=diff.columns)


def snapshot(p: Panel) -> pd.DataFrame:
    """全銘柄の、最新の日のテクニカル指標を1行ずつにまとめる。"""
    last = lambda w: w.iloc[-1]
    macd_line = ema(p, 12) - ema(p, 26)
    sig_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    return pd.DataFrame({
        "close": last(p.c), "sma5": last(p.sma(5)), "sma25": last(p.sma(25)),
        "sma75": last(p.sma(75)), "sma200": last(p.sma(200)),
        "rsi14": last(p.rsi(14)), "macd": last(macd_hist(p)) * 100,
        "macd_x": _days_since_cross(macd_line - sig_line, 10),
        "ma_x": _days_since_cross(p.sma(25) - p.sma(75), 20),
        "bb": last(bb_pos(p)), "hi52": last(hi52(p)) * 100,
        "vol": last(p.v / p.vol_avg(20)), "atr_pct": last(p.atr(14) / p.c) * 100,
    })


# 画面に配るときの並び（文字数を減らすため、名前ではなく順番で渡す）
SNAP_COLS = ["sma5", "sma25", "sma75", "sma200", "rsi14", "macd", "macd_x", "ma_x", "bb", "hi52", "vol", "atr_pct"]
