from __future__ import annotations

import json
import typer
from poker_collusion.config import load_config
from poker_collusion.pipeline import train

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = "configs/baseline.yaml"):
    metrics = train(load_config(config))
    typer.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
