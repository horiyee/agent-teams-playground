"""DSPy コミットプレフィックス分類器 (MIPROv2 最適化)。"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Literal

import dspy
from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.development.local"
_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "commits.json"
_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "optimized_model.json"

PrefixType = Literal[
    "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert"
]


class ClassifyCommitPrefix(dspy.Signature):
    """Classify a commit message body into a Conventional Commits prefix."""

    body: str = dspy.InputField(desc="The commit message body (without prefix)")
    prefix: PrefixType = dspy.OutputField(desc="The Conventional Commits prefix")


class CommitPrefixClassifier(dspy.Module):
    def __init__(self) -> None:
        self.predict = dspy.Predict(ClassifyCommitPrefix)

    def forward(self, body: str) -> dspy.Prediction:
        return self.predict(body=body)


def load_dataset(
    path: Path = _DATA_FILE, *, train_ratio: float = 0.8, seed: int = 42
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """commits.json を読み込み、train/dev に分割する。"""
    data = json.loads(path.read_text())
    examples = [
        dspy.Example(body=item["body"], prefix=item["prefix"]).with_inputs("body")
        for item in data
    ]
    random.seed(seed)
    random.shuffle(examples)
    split = int(len(examples) * train_ratio)
    return examples[:split], examples[split:]


def prefix_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """完全一致メトリック (大文字小文字を区別しない)。"""
    return example.prefix.lower() == prediction.prefix.lower()


def optimize(
    trainset: list[dspy.Example],
) -> dspy.Module:
    """MIPROv2 で分類器を最適化する。"""
    optimizer = dspy.MIPROv2(metric=prefix_metric, auto="medium", verbose=True)
    optimized = optimizer.compile(
        CommitPrefixClassifier(),
        trainset=trainset,
    )
    return optimized


def evaluate(module: dspy.Module, devset: list[dspy.Example]) -> float:
    """dev セットでモジュールを評価し、精度を返す。"""
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=prefix_metric,
        num_threads=4,
        display_progress=True,
        display_table=5,
    )
    return float(evaluator(module))


def main() -> None:
    load_dotenv(_ENV_FILE)

    # DSPy / MIPROv2 の内部ログを表示
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    lm = dspy.LM("gemini/gemini-2.0-flash")
    dspy.configure(lm=lm)

    print("データセットを読み込み中 ...")
    trainset, devset = load_dataset()
    print(f"  学習: {len(trainset)} 件, 評価: {len(devset)} 件")

    if len(trainset) < 100:
        print(f"\n  ⚠ 学習データが {len(trainset)} 件と少なめです。")
        print("    MIPROv2 の効果を得るには 200 件以上を推奨します。")
        print("    make collect REPOS=\"...\" でより多くのリポジトリからデータを収集してください。")

    # --- ベースライン ---
    print("\n=== ベースライン評価 ===")
    baseline = CommitPrefixClassifier()
    baseline_score = evaluate(baseline, devset)
    print(f"ベースライン精度: {baseline_score:.1f}%")

    # --- 最適化 ---
    print("\n=== MIPROv2 プロンプト最適化 ===")
    optimized = optimize(trainset)

    # 最適化後のプロンプトを表示
    print("\n--- 最適化されたプロンプト ---")
    for i, predictor in enumerate(optimized.predictors()):
        if hasattr(predictor, "demos") and predictor.demos:
            print(f"\n[Predictor {i}] Few-shot デモ ({len(predictor.demos)} 件):")
            for j, demo in enumerate(predictor.demos):
                print(f"  例 {j+1}: body={demo.body!r} → prefix={demo.prefix!r}")
        if hasattr(predictor, "signature"):
            sig = predictor.signature
            if sig.instructions:
                print(f"\n[Predictor {i}] 最適化された指示文:")
                print(f"  {sig.instructions}")

    print("\n=== 最適化モデル評価 ===")
    optimized_score = evaluate(optimized, devset)
    print(f"最適化後精度: {optimized_score:.1f}%")

    # --- 比較 ---
    print("\n=== 結果比較 ===")
    print(f"  ベースライン: {baseline_score:.1f}%")
    print(f"  最適化後:     {optimized_score:.1f}%")
    diff = optimized_score - baseline_score
    print(f"  差分:         {diff:+.1f}%")

    # --- モデル保存 ---
    optimized.save(_MODEL_PATH)
    print(f"\n最適化モデルを {_MODEL_PATH} に保存しました")
