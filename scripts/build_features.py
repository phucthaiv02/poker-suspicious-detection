from __future__ import annotations

import typer
from poker_collusion.config import load_config
from poker_collusion.pipeline import prepare_features

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = "configs/baseline.yaml"):
    outputs = prepare_features(load_config(config))
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


if __name__ == "__main__":
    app()
