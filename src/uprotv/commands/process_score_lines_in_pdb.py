# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "click",
#     "polars",
# ]
# ///

import click

import collections

from typing import Iterator

from pathlib import Path

import polars as pl


def read_score_lines(pdb_filepath: Path) -> Iterator[str]:
    with open(pdb_filepath, "r") as f:
        for line in f:
            if "SCORE" in line:
                yield line.split(" ")


def extract_score_from_line(line: list[str]) -> tuple[str, float]:
    return line[1].strip()[: -1], float(line[2].strip())


@click.command()
@click.argument("path", type = Path)
@click.option("--glob-str", type = str, default = "*.pdb", help = "The glob string to use for search in the path")
def main(
    path: Path,
    glob_str: str,
) -> int:
    pdb_filepaths = list(path.glob(glob_str))
    all_scores = collections.defaultdict(list)
    filenames = []
    for pdb_filepath in pdb_filepaths:
        scores = [extract_score_from_line(line) for line in read_score_lines(pdb_filepath)]
        if len(scores) == 0:
            click.echo(f"Could not process {pdb_filepath.stem}")
            continue
        for score_name, score_value in scores:
            all_scores[score_name].append(score_value)
        filenames.append(pdb_filepath.stem)
    pl.LazyFrame(all_scores).with_columns(pl.Series(filenames).alias("name")).sink_parquet(path / "metrics.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())