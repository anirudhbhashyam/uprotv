# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "click",
#     "deltalake",
#     "flax",
#     "jax[cuda12]<0.6.0",
#     "omegaconf",
#     "polars",
#     "proteinbee",
#     "rich",
#     "colabdesign @ git+ssh://git@github.com/sokrypton/ColabDesign",
# ]
# ///

import click

from colabdesign import mk_afdesign_model
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.shared.utils import clear_mem

import functools

import jax
import jax.numpy as jnp

import multiprocessing as mp

import numpy as np

from omegaconf import (
    DictConfig,
    OmegaConf,
)

from pathlib import Path

import polars as pl

from proteinbee.motif import Motif

from rich import progress

import subprocess

from typing import Any


def setup_af2_params() -> None:
    path = Path.home().joinpath(".cache", "params")
    if path.exists():
        click.echo("AF2 parameters already downloaded.")
        return
    path.mkdir(parents = True)
    url = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    try:
        subprocess.check_call(
            [
                "aria2c",
                "-x",
                "4",
                url,
                "-d",
                str(path),
            ],
        )
    except subprocess.CalledProcessError:
        click.echo("`aria2c` was not found. Using wget")
        subprocess.check_call(
            [
                "wget",
                url,
                "-P",
                str(path),
            ],
        )
    subprocess.check_call(
        [
            "tar",
            "xvf",
            str(path / "alphafold_params_2022-12-06.tar"),
            "-C",
            str(path),
        ],
    )
    (path / "alphafold_params_2022-12-06.tar").unlink(missing_ok = True)


def worker(
    work: tuple[str, Path, DictConfig],
    save_path: Path,
    config: DictConfig,
    *,
    mpnn_model: mk_mpnn_model,
    af2_binder_model: mk_afdesign_model,
    af2_monomer_model: mk_afdesign_model,
) -> pl.LazyFrame | None:
    design_name, structure_filepath, design_data = work
    if not structure_filepath.exists():
        click.echo(f"Structure `{structure_filepath}` was not found.")
        return None
    motif_str = design_data.get("motif", None)
    chains = None
    fix_pos = None
    if motif_str is not None:
        motif = Motif.from_string(motif_str)
        fix_pos = ",".join(str(x) for x in motif.selector_iter())
        chains = ",".join(
            comp["protein"]["chain"]
            for comp in design_data["components"]
            if comp.get("protein") is not None
        )
        binder_seq_pos = next(
            i for i, comp in enumerate(design_data["components"])
            if comp.get("protein") is not None and comp["protein"]["name"] == "binder"
        )
    params_path = str(Path.home().joinpath(".cache"))
    mpnn_model.prep_inputs(
        pdb_filename = str(structure_filepath),
        chain = chains,
        fix_pos = fix_pos,
        rm_aa = config["proteinmpnn"].get("rm_aa", None),
        ignore_missing = False,
    )
    samples = mpnn_model.sample_parallel(
        batch = config["proteinmpnn"]["sequence"]["N"],
        temperature = config["proteinmpnn"]["sequence"]["temperature"],
    )
    binder_chain = next(
        x["protein"]["chain"]
        for x in design_data["components"]
        if x.get("protein") is not None and x["protein"]["name"] == "binder"
    )
    target_chain = next(
        x["protein"]["chain"]
        for x in design_data["components"]
        if x.get("protein") is not None and x["protein"]["name"] == "target"
    )
    af2_binder_model.prep_inputs(
        pdb_filename = str(structure_filepath),
        chain = target_chain,
        binder_chain = binder_chain,
        use_binder_template = True,
        rm_template_ic = True,
    )
    af2_monomer_model.prep_inputs(
        pdb_filename = str(structure_filepath),
        chain = binder_chain,
    )
    binder_seqs = [
        s.split("/")[binder_seq_pos] for s in samples["seq"]
    ]
    names = [f"{design_name}_seq_{i}" for i in range(len(binder_seqs))]
    results = []
    for name, seq in zip(names, binder_seqs):
        af2_binder_model.predict(
            seq,
            num_recycles = config["af2"]["num_recycles"],
            model_nums = config["af2"]["model_nums"],
            verbose = False,
        )
        af2_monomer_model.predict(
            seq,
            num_recycles = config["af2"]["num_recycles"],
            model_nums = [0, 3],
            verbose = False,
        )
        results.append(
            {
                "plddt": af2_binder_model.aux["log"]["plddt"],
                "pae": af2_binder_model.aux["log"]["pae"],
                "ptm": af2_binder_model.aux["log"]["ptm"],
                "rmsd": af2_binder_model.aux["log"]["rmsd"],
                "ipae": af2_binder_model.aux["log"]["i_pae"],
                "iptm": af2_binder_model.aux["log"]["i_ptm"],
                "binder_plddt": af2_monomer_model.aux["log"]["plddt"],
                "binder_pae": af2_monomer_model.aux["log"]["pae"],
                "binder_ptm": af2_monomer_model.aux["log"]["ptm"],
                "binder_rmsd": af2_monomer_model.aux["log"]["rmsd"],
            }
        )
        af2_binder_model.save_pdb(save_path / f"{name}.pdb")
    return pl.LazyFrame(results).with_columns(
        name = pl.Series(names),
        seq = pl.Series(samples["seq"]),
    )


@click.command(short_help = "Run AF2 binder evaluation with ProteinMPNN sequence design.")
@click.argument("config-filepath", type = Path)
def main(config_filepath: Path) -> int:
    setup_af2_params()
    config = OmegaConf.load(config_filepath)
    tracker = progress.Progress(
        progress.TextColumn("[progress.task]{task.description}"),
        progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        progress.BarColumn(),
        progress.MofNCompleteColumn(),
        progress.TextColumn("•"),
        progress.TimeRemainingColumn(),
    )
    design_tasks = config["design"]["detail"]
    n_design_tasks = len(design_tasks)
    work = (
        (
            design_name,
            Path(config["design"]["store_path"]).resolve().joinpath(design_data["filename"]),
            design_data,
        )
        for design_name, design_data in design_tasks.items()
    )
    save_path = Path(config["design"]["store_path"]).resolve().joinpath("af2_binder")
    if not save_path.exists():
        save_path.mkdir(parents = True)
    params_path = Path.home().joinpath(".cache")
    clear_mem()
    mpnn_model = mk_mpnn_model(weights = config["proteinmpnn"]["weights"])
    af2_binder_model = mk_afdesign_model(
        protocol = "binder",
        data_dir = params_path,
        use_multimer = True,
        use_initial_guess = True,
    )
    af2_monomer_model = mk_afdesign_model(protocol = "fixbb", data_dir = params_path)
    save_every_n_iterations = 10
    final_results = []
    func = functools.partial(
        worker,
        config = config,
        save_path = save_path,
        mpnn_model = mpnn_model,
        af2_binder_model = af2_binder_model,
        af2_monomer_model = af2_monomer_model,
    )
    results_iter = (func(w) for w in work)
    metrics_save_filepath = save_path / "metrics.delta"
    with tracker:
        task = tracker.add_task("MPNN and AF2 evaluation", total = n_design_tasks)
        for i, result in enumerate(results_iter):
            if result is not None:
                final_results.append(result)
            if i % save_every_n_iterations == 0:
                pl.concat(final_results).collect().write_delta(metrics_save_filepath, mode = "append")
                final_results = []
            tracker.update(task, advance = 1)
            af2_binder_model.restart()
            af2_monomer_model.restart()
        if final_results:
            click.echo(f"Saving final results")
            pl.concat(final_results).collect().write_delta(metrics_save_filepath, mode = "append")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
