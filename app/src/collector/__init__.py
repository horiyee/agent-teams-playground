"""git リポジトリから Conventional Commits を収集する。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

_CONVENTIONAL_RE = re.compile(
    r"^(?P<prefix>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\([^)]*\))?!?:\s*(?P<body>.+)$",
    re.IGNORECASE,
)

_DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "commits.json"


def collect_from_repo(repo_path: str, *, max_commits: int = 500) -> list[dict]:
    """単一リポジトリから git log を取得し Conventional Commits をパースする。"""
    result = subprocess.run(
        ["git", "log", f"--max-count={max_commits}", "--format=%s"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commits: list[dict] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _CONVENTIONAL_RE.match(line)
        if m:
            commits.append(
                {
                    "prefix": m.group("prefix").lower(),
                    "body": m.group("body").strip(),
                    "original": line,
                }
            )
    return commits


def collect_from_repos(
    repo_paths: list[str], *, max_commits: int = 500
) -> list[dict]:
    """複数リポジトリからコミットを収集する。"""
    all_commits: list[dict] = []
    for path in repo_paths:
        print(f"収集中: {path} ...")
        commits = collect_from_repo(path, max_commits=max_commits)
        print(f"  {len(commits)} 件の Conventional Commits を検出")
        all_commits.extend(commits)
    return all_commits


def save(commits: list[dict], output: Path) -> None:
    """コミットを JSON に保存し、プレフィックス分布を表示する。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(commits, ensure_ascii=False, indent=2))
    print(f"\n{len(commits)} 件のコミットを {output} に保存しました")

    counter = Counter(c["prefix"] for c in commits)
    print("\nプレフィックス分布:")
    for prefix, count in counter.most_common():
        print(f"  {prefix:10s} {count:4d}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="git リポジトリから Conventional Commits を収集"
    )
    parser.add_argument(
        "repo_paths", nargs="+", help="git リポジトリのパス"
    )
    parser.add_argument(
        "--max", type=int, default=500, help="リポジトリあたりの最大コミット数 (デフォルト: 500)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"出力先 JSON パス (デフォルト: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    commits = collect_from_repos(args.repo_paths, max_commits=args.max)
    if not commits:
        print("Conventional Commits が見つかりませんでした。")
        return
    save(commits, args.output)
