from __future__ import annotations

import json
import typer
from poker_collusion.config import load_config
from poker_collusion.pipeline import prepare_features, train, predict_submission

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = "configs/baseline.yaml", output: str = "submission.csv"):
    cfg = load_config(config)
    prepare_features(cfg)
    metrics = train(cfg)
    typer.echo(json.dumps(metrics, indent=2))
    typer.echo(f"Wrote {predict_submission(cfg, output)}")


if __name__ == "__main__":
    app()
