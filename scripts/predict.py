from __future__ import annotations

import typer
from poker_collusion.config import load_config
from poker_collusion.pipeline import predict_submission

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = "configs/baseline.yaml", output: str = "submission.csv"):
    path = predict_submission(load_config(config), output)
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
