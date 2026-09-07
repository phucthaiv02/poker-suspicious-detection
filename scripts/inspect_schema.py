from __future__ import annotations

import json
import typer
import polars as pl

from poker_collusion.config import load_config, resolve_paths
from poker_collusion.schema import detect_schema

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = "configs/baseline.yaml"):
    cfg = load_config(config)
    paths = resolve_paths(cfg)
    schema = detect_schema(paths, cfg.get("schema_overrides", {}))
    typer.echo(json.dumps(schema.to_dict(), indent=2))
    typer.echo("\nColumns:")
    for key in ["players", "hands", "seats", "actions", "development_labels", "development_evidence", "evaluation_pairs", "sample_submission"]:
        path = paths[key]
        cols = pl.scan_parquet(path).collect_schema().names() if path.suffix == ".parquet" else pl.read_csv(path, n_rows=2).columns
        typer.echo(f"- {path.name}: {cols}")


if __name__ == "__main__":
    app()
