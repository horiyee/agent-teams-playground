from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fasthtml.common import H1, Main, P, fast_app, serve

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.development.local"

app, rt = fast_app()


@rt("/")
def get():
    return Main(
        H1("Hello, FastHTML!"),
        P("App is running."),
    )


def main() -> None:
    load_dotenv(_ENV_FILE)
    serve(port=5001)
