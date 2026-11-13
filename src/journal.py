"""
学習ノート（docs/data/journal.json）。

- これまでに試した仮説・設定を1つずつ数える（同じものは二重に数えない）
- 試した数が増えるほど、合格に必要なt値（統計的な確かさの物差し）を引き上げる
  → たくさん試せば偶然良く見えるものが必ず出るため、その分ハードルを上げる
- 何を試して何が分かったかを、新しい順に読める記録として残す
"""

from __future__ import annotations

import json
import os
from statistics import NormalDist

import pandas as pd

from .universe import ROOT

PATH = os.path.join(ROOT, "docs", "data", "journal.json")
T_MIN, T_MAX = 2.0, 3.0   # 合格に必要なt値の下限と上限
MAX_ENTRIES = 300


def load() -> dict:
    if os.path.exists(PATH):
        with open(PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"keys": [], "extra_trials": 0, "entries": []}


def save(j: dict) -> None:
    j["entries"] = j["entries"][-MAX_ENTRIES:]
    j["trials_total"] = trials(j)
    j["t_required"] = t_required(j)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)


def key(name: str, params: dict, filters: list | None = None) -> str:
    f = ";".join(f"{x['feature']}[{x.get('lo')},{x.get('hi')})" for x in (filters or []))
    return f"{name}|{json.dumps(params, sort_keys=True)}|{f}"


def register(j: dict, keys: list[str]) -> int:
    """試した設定を登録し、新しく増えた数を返す。"""
    have = set(j["keys"])
    new = [k for k in dict.fromkeys(keys) if k not in have]
    j["keys"].extend(new)
    return len(new)


def trials(j: dict) -> int:
    return len(j["keys"]) + int(j.get("extra_trials", 0))


def t_required(j: dict) -> float:
    """試した回数 N に対し、偶然で1つでも合格してしまう確率を5%以内に抑えるt値（ボンフェローニ補正）。
    実務上の目安に合わせ 2.0〜3.0 の範囲に収める。"""
    n = max(trials(j), 1)
    z = NormalDist().inv_cdf(1 - 0.025 / n)
    return round(min(T_MAX, max(T_MIN, z)), 2)


def add(j: dict, kind: str, title: str, body: str, lessons: list[str] | None = None,
        result: str = "", date: str | None = None) -> None:
    """kind: 検証 / 学習 / 監視 / 調査"""
    j["entries"].append({
        "date": date or pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d"),
        "kind": kind, "title": title, "body": body,
        "lessons": lessons or [], "result": result,
    })
