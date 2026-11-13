"""
バックテストの土台。

- Panel:   縦長の株価表を「日付×銘柄」の横長の表に並べ替え、指標を計算して持つ
- simulate: 売買ルール通りに1トレードずつ損益を計算する
- stats / portfolio: 成績の集計と、資金制約つきの資産推移

前提（現実の売買に合わせた決まりごと）:
- シグナルは大引け後に確定 → 翌営業日の寄り付き（始値）で成行買い
- 損切りは逆指値。寄り付きが逆指値を下回ったら始値で約定（窓開けの損失を織り込む）
- 同じ日に損切りと利確の両方に届いた場合は、損切りが先に付いたとみなす（保守的）
- 売りシグナル（終値で判定するもの）は翌営業日の寄り付きで成行売り
- コストは片道 cost（スリッページ等）を往復分差し引く
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BENCH = "1306"


# ---------------------------------------------------------------- Panel
class Panel:
    def __init__(self, prices: pd.DataFrame):
        def wide(col):
            w = prices.pivot(index="date", columns="code", values=col)
            return w.sort_index()

        self.o = wide("open").astype("float32")
        self.h = wide("high").astype("float32")
        self.l = wide("low").astype("float32")
        self.c = wide("close").astype("float32")
        self.v = wide("volume").astype("float64")
        if BENCH in self.c.columns:  # 物差しのETFは、欠けた日を前日の値で埋める
            for w in (self.o, self.h, self.l, self.c):
                w[BENCH] = w[BENCH].ffill()
        self.dates = self.c.index
        self.codes = self.c.columns
        self._cache: dict[str, pd.DataFrame] = {}

    # --- 指標（必要になった時に計算して使い回す）
    def sma(self, n: int) -> pd.DataFrame:
        k = f"sma{n}"
        if k not in self._cache:
            self._cache[k] = self.c.rolling(n, min_periods=n).mean()
        return self._cache[k]

    def atr(self, n: int = 14) -> pd.DataFrame:
        """ATR（1日の平均的な値幅。損切り幅の物差しに使う）"""
        k = f"atr{n}"
        if k not in self._cache:
            pc = self.c.shift(1)
            tr = np.maximum(self.h - self.l, np.maximum((self.h - pc).abs(), (self.l - pc).abs()))
            self._cache[k] = tr.rolling(n, min_periods=n).mean()
        return self._cache[k]

    def rsi(self, n: int = 2) -> pd.DataFrame:
        """RSI（直近の上げ下げの偏り。低いほど短期的に売られている）"""
        k = f"rsi{n}"
        if k not in self._cache:
            d = self.c.diff()
            up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
            dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
            self._cache[k] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
            self._cache[k] = self._cache[k].where(dn > 0, 100.0).where(d.notna())
        return self._cache[k]

    def prior_high(self, n: int) -> pd.DataFrame:
        """前日までのn日間の高値"""
        k = f"ph{n}"
        if k not in self._cache:
            self._cache[k] = self.h.rolling(n, min_periods=n).max().shift(1)
        return self._cache[k]

    def vol_avg(self, n: int = 20) -> pd.DataFrame:
        """前日までのn日間の平均出来高"""
        k = f"va{n}"
        if k not in self._cache:
            self._cache[k] = self.v.rolling(n, min_periods=n).mean().shift(1)
        return self._cache[k]

    def turnover(self, n: int = 20) -> pd.DataFrame:
        """売買代金（終値×出来高）のn日中央値。流動性の物差し"""
        k = f"to{n}"
        if k not in self._cache:
            self._cache[k] = (self.c * self.v).rolling(n, min_periods=n).median()
        return self._cache[k]

    def ret(self, n: int) -> pd.DataFrame:
        k = f"ret{n}"
        if k not in self._cache:
            self._cache[k] = self.c / self.c.shift(n) - 1
        return self._cache[k]

    def tradable(self, max_price: float, min_turnover: float) -> pd.DataFrame:
        """買える銘柄か（株価が予算内・売買代金が十分・上場から約1年以上）"""
        k = f"tr_{max_price}_{min_turnover}"
        if k not in self._cache:
            ok = (self.c <= max_price) & (self.turnover(20) >= min_turnover) & self.sma(200).notna()
            if BENCH in ok.columns:
                ok[BENCH] = False
            self._cache[k] = ok
        return self._cache[k]

    def regime(self, n: int = 200) -> pd.Series:
        """地合い: TOPIX連動ETFがn日移動平均より上なら True（上昇基調）"""
        b = self.c[BENCH]
        return b > b.rolling(n, min_periods=n).mean()


# ---------------------------------------------------------------- 売りルール
@dataclass
class ExitSpec:
    stop_atr: float = 2.5          # 初期の損切り幅（ATRの何倍か）
    target_atr: float | None = None  # 利確幅（ATRの何倍か）。None なら使わない
    trail_atr: float | None = None   # 終値の最高値から何ATR下に逆指値を引き上げるか
    exit_above_sma: int | None = None  # 終値がこの移動平均を上回ったら売り（押し目買い用）
    exit_below_sma: int | None = None  # 終値がこの移動平均を下回ったら売り（順張り用）
    max_hold: int = 10             # 最大保有日数（営業日）


def simulate(p: Panel, entries: pd.DataFrame, score: pd.DataFrame, ex: ExitSpec,
             cost: float = 0.001, include_open: bool = False) -> pd.DataFrame:
    """entries が True の日の翌営業日に買い、ExitSpec に従って売る。"""
    O, H, L, C = (x.to_numpy(dtype="float64") for x in (p.o, p.h, p.l, p.c))
    A = p.atr(14).to_numpy(dtype="float64")
    E = entries.to_numpy(dtype=bool)
    S = score.to_numpy(dtype="float64")
    SA = p.sma(ex.exit_above_sma).to_numpy(dtype="float64") if ex.exit_above_sma else None
    SB = p.sma(ex.exit_below_sma).to_numpy(dtype="float64") if ex.exit_below_sma else None
    n = len(p.dates)
    rows = []
    cols = np.flatnonzero(E.any(axis=0))
    for j in cols:
        o, h, l, c, a = O[:, j], H[:, j], L[:, j], C[:, j], A[:, j]
        sa = SA[:, j] if SA is not None else None
        sb = SB[:, j] if SB is not None else None
        busy_until = -1
        for i in np.flatnonzero(E[:, j]):
            if i <= busy_until or i + 1 >= n:
                continue
            e = i + 1
            entry = o[e]
            atr = a[i]
            if not (entry > 0) or not (atr > 0):
                continue
            stop = entry - ex.stop_atr * atr
            target = entry + ex.target_atr * atr if ex.target_atr else None
            hi_close = entry
            exit_px = None
            reason = "open"
            held = 0
            x = e
            k = e
            pending = None  # 終値で売りシグナル → 次の寄り付きで売る
            while k < n:
                ok_, hk, lk, ck = o[k], h[k], l[k], c[k]
                if not (ck > 0):  # 売買が無かった日は飛ばす
                    k += 1
                    continue
                if pending is not None:
                    exit_px, reason, x = ok_, pending, k
                    break
                held += 1
                if ok_ <= stop:
                    exit_px, reason, x = ok_, "損切り(窓)", k
                    break
                if lk <= stop:
                    exit_px, reason, x = stop, "損切り", k
                    break
                if target is not None and hk >= target:
                    exit_px, reason, x = max(ok_, target), "利確", k
                    break
                if ck > hi_close:
                    hi_close = ck
                if ex.trail_atr and a[k] > 0:
                    stop = max(stop, hi_close - ex.trail_atr * a[k])
                if sa is not None and ck > sa[k]:
                    pending = "反発で売り"
                elif sb is not None and ck < sb[k]:
                    pending = "トレンド終了"
                elif held >= ex.max_hold:
                    pending = "期限"
                k += 1
            if exit_px is None:
                if not include_open:
                    busy_until = n
                    continue
                x = n - 1
                exit_px = c[x] if c[x] > 0 else entry
            if reason.startswith("損切り") and exit_px >= entry:
                reason = "逆指値で利益確定"   # 引き上げた逆指値に掛かったが、買値より上で売れた
            busy_until = x
            rows.append((j, i, e, x, entry, exit_px, held, S[i, j], reason, stop, pending or ""))
    if not rows:
        return pd.DataFrame(columns=["code", "signal_date", "entry_date", "exit_date", "entry", "exit",
                                     "hold", "score", "reason", "ret", "mkt", "excess"])
    t = pd.DataFrame(rows, columns=["j", "i", "e", "x", "entry", "exit", "hold", "score", "reason",
                                    "stop_now", "pending"])
    t["code"] = p.codes[t["j"]].to_numpy()
    t["signal_date"] = p.dates[t["i"]].to_numpy()
    t["entry_date"] = p.dates[t["e"]].to_numpy()
    t["exit_date"] = p.dates[t["x"]].to_numpy()
    t["ret"] = t["exit"] / t["entry"] - 1 - 2 * cost
    bo = p.o[BENCH].ffill().to_numpy(dtype="float64")
    bc = p.c[BENCH].ffill().to_numpy(dtype="float64")
    t["mkt"] = bc[t["x"]] / bo[t["e"]] - 1   # 同じ期間に市場全体(TOPIX)を持っていた場合の損益
    t["excess"] = t["ret"] - t["mkt"]
    return t.drop(columns=["j"]).sort_values(["entry_date", "score"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------- 集計
def stats(t: pd.DataFrame) -> dict:
    """トレード一覧から成績をまとめる。t値は「月ごとの平均」で計算し、同じ日に固まった
    トレードを別々の証拠として数えすぎないようにしている。"""
    if len(t) == 0:
        return {"n": 0}
    r = t["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    m = t.groupby(t["entry_date"].dt.to_period("M"))[["ret", "excess"]].mean()

    def tval(s):
        return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 2 and s.std(ddof=1) > 0 else 0.0

    yearly = t.groupby(t["entry_date"].dt.year)["ret"].agg(["count", "mean"])
    yearly_ex = t.groupby(t["entry_date"].dt.year)["excess"].mean()
    return {
        "n": int(len(t)),
        "win_rate": float((r > 0).mean()),
        "avg_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "avg_excess": float(t["excess"].mean()),
        "avg_hold": float(t["hold"].mean()),
        "t_ret": tval(m["ret"]),
        "t_excess": tval(m["excess"]),
        "months": int(len(m)),
        "years_pos": int((yearly["mean"] > 0).sum()),
        "years_ex_pos": int((yearly_ex > 0).sum()),
        "years": int(len(yearly)),
        "yearly": {int(y): {"n": int(row["count"]), "avg_ret": float(row["mean"]),
                            "avg_excess": float(yearly_ex[y])} for y, row in yearly.iterrows()},
    }


def portfolio(p: Panel, t: pd.DataFrame, capital: float, max_positions: int,
              lot: int = 100, start=None, end=None) -> dict:
    """資金制約つきの資産推移。その日の候補をスコア順に、買える分だけ買う。"""
    if start is not None:
        t = t[t["entry_date"] >= start]
    if end is not None:
        t = t[t["entry_date"] <= end]
    dates = p.dates
    if start is not None:
        dates = dates[dates >= start]
    if end is not None:
        dates = dates[dates <= end]
    if len(t) == 0 or len(dates) == 0:
        return {"trades": 0}
    by_entry = {d: g for d, g in t.groupby("entry_date")}
    C = p.c.ffill()
    cash = capital
    open_pos: list[dict] = []
    curve = []
    taken = []
    for d in dates:
        still = []
        for pos in open_pos:
            if pos["exit_date"] <= d:
                cash += pos["shares"] * pos["entry"] * (1 + pos["ret"])
            else:
                still.append(pos)
        open_pos = still
        if d in by_entry:
            held = {x["code"] for x in open_pos}
            for row in by_entry[d].itertuples():
                if len(open_pos) >= max_positions:
                    break
                if row.code in held:
                    continue
                budget = min(cash, capital_per_slot(cash, open_pos, max_positions, C, d))
                shares = int(budget // (row.entry * lot)) * lot
                if shares <= 0:
                    continue
                cash -= shares * row.entry
                pos = {"code": row.code, "shares": shares, "entry": row.entry, "ret": row.ret,
                       "exit_date": row.exit_date}
                open_pos.append(pos)
                taken.append(row.ret)
        mtm = sum(x["shares"] * float(C.at[d, x["code"]]) for x in open_pos)
        curve.append(cash + mtm)
    eq = pd.Series(curve, index=dates)
    yrs = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    dd = (eq / eq.cummax() - 1).min()
    b = p.c[BENCH].reindex(dates).ffill()
    weekly = eq.resample("W").last().dropna()
    return {
        "trades": len(taken),
        "final": float(eq.iloc[-1]),
        "total_ret": float(eq.iloc[-1] / capital - 1),
        "cagr": float((eq.iloc[-1] / capital) ** (1 / yrs) - 1),
        "max_dd": float(dd),
        "bench_total_ret": float(b.iloc[-1] / b.iloc[0] - 1),
        "bench_cagr": float((b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1),
        "bench_max_dd": float((b / b.cummax() - 1).min()),
        "win_rate": float(np.mean([x > 0 for x in taken])) if taken else 0.0,
        "curve": [[d.strftime("%Y-%m-%d"), round(float(v))] for d, v in weekly.items()],
        "bench_curve": [[d.strftime("%Y-%m-%d"), round(float(v / b.iloc[0] * capital))]
                        for d, v in b.resample("W").last().dropna().items()],
    }


def capital_per_slot(cash, open_pos, max_positions, C, d) -> float:
    """1銘柄に使える上限 = 総資産 ÷ 最大保有銘柄数"""
    mtm = sum(x["shares"] * float(C.at[d, x["code"]]) for x in open_pos)
    return (cash + mtm) / max_positions
