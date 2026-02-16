from pathlib import Path

import dspy
from dotenv import load_dotenv
from fasthtml.common import (
    Button,
    Details,
    Form,
    H1,
    H2,
    H3,
    Input,
    Label,
    Main,
    P,
    Pre,
    Summary,
    Table,
    Tbody,
    Td,
    Th,
    Thead,
    Tr,
    fast_app,
    serve,
)

from commit_prefix import CommitPrefixClassifier, _MODEL_PATH

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.development.local"

load_dotenv(_ENV_FILE)
dspy.configure(lm=dspy.LM("gemini/gemini-2.0-flash"))

app, rt = fast_app()

if _MODEL_PATH.exists():
    print(f"最適化モデルを {_MODEL_PATH} からロードします")
    _classifier = CommitPrefixClassifier()
    _classifier.load(_MODEL_PATH)
else:
    print("最適化モデルが見つかりません。ベースラインを使用します")
    _classifier = CommitPrefixClassifier()


def _model_info():
    """最適化モデルのメタデータを表示するコンポーネントを返す。"""
    is_optimized = _MODEL_PATH.exists()
    sections = [
        H2("モデル情報"),
        P(f"モデル状態: {'最適化済み' if is_optimized else 'ベースライン'}"),
        P(f"モデルパス: {_MODEL_PATH}"),
    ]

    for i, predictor in enumerate(_classifier.predictors()):
        # 指示文
        if hasattr(predictor, "signature") and predictor.signature.instructions:
            sections.append(H3(f"Predictor {i} — 指示文"))
            sections.append(Pre(predictor.signature.instructions))

        # Few-shot デモ
        if hasattr(predictor, "demos") and predictor.demos:
            rows = []
            for j, demo in enumerate(predictor.demos):
                if isinstance(demo, dict):
                    body_val, prefix_val = demo.get("body", ""), demo.get("prefix", "")
                else:
                    body_val, prefix_val = demo.body, demo.prefix
                rows.append(Tr(Td(str(j + 1)), Td(body_val), Td(prefix_val)))
            sections.append(H3(f"Predictor {i} — Few-shot デモ ({len(predictor.demos)} 件)"))
            sections.append(
                Table(
                    Thead(Tr(Th("#"), Th("body"), Th("prefix"))),
                    Tbody(*rows),
                )
            )

    return Details(Summary("最適化プロンプトの詳細を表示"), *sections)


def _classify_form(result: str | None = None):
    children = [
        H1("コミットプレフィックス分類デモ"),
    ]
    if result is not None:
        children.append(P(f"判定結果: {result}"))
    children.append(
        Form(
            Label("コミットメッセージ body:", **{"for": "body"}),
            Input(name="body", placeholder="例: add user authentication"),
            Button("分類する"),
            method="post",
            action="/classify",
        )
    )
    children.append(_model_info())
    return Main(*children)


@rt("/")
def get():
    return Main(
        H1("Hello, FastHTML!"),
        P("App is running."),
    )


@rt("/classify")
def get():
    return _classify_form()


@rt("/classify")
def post(body: str):
    prediction = _classifier(body=body)
    return _classify_form(result=prediction.prefix)


def main() -> None:
    serve(port=5001)
