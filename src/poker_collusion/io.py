from __future__ import annotations

from pathlib import Path
import polars as pl


def scan(path: Path) -> pl.LazyFrame:
    if path.suffix.lower() == ".parquet":
        return pl.scan_parquet(path)
    return pl.scan_csv(path)


def read(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path)


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")
