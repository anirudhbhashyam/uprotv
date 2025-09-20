# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "click",
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
from colabdesign.af.alphafold.common import residue_constants 
from colabdesign.shared.utils import clear_mem
from colabdesign.af.loss import get_pae_loss


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


def setup_af2_params() -> None:
    path = Path.home().joinpath(".cache", "params")
    if path.exists():
        click.echo("AF2 parameters already downloaded.")
        return
    path.mkdir(parents = True)
    try:
        subprocess.check_call(
            [
                "aria2c",
                "-x",
                "4",
                "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
                "-d",
                str(path),
            ],
        )
    except subprocess.CalledProcessError:
        click.echo("`aria2c` was not found. Using wget")
        subprocess.check_call(
            [
                "wget",
                "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
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


def _predict_structure(
    model: mk_afdesign_model,
    seq: jnp.ndarray,
) -> dict[str, float]:
    model._cfg.global_config = model._cfg.model.global_config
    model._params["seq"] = seq
    model._inputs["bias"] = jnp.zeros(seq.shape[1 :])
    model.set_opt(hard = True, soft = False, temp = 1, dropout = False, pssm_hard = True)
    model.set_args(shuffle_first = False)
    model._inputs["opt"] = model.opt
    loss, aux = model._model["fn"](
        model._params,
        model._model_params[0],
        model._inputs,
        model.key(),
    )
    return {
        "plddt": aux["plddt"].mean(-1),
        "pae": aux["pae"].mean(),
        "ptm": aux["ptm"],
        "rmsd": aux["losses"]["rmsd"],
        "ipae": aux["losses"].get("i_pae", None),
        "iptm": aux.get("i_ptm", None),
    }


_parallel_predict_structure = jax.vmap(_predict_structure, in_axes = (None, 0))


def _prepare_seq_for_af2(seq: str) -> jnp.ndarray:
    B, L, A = (1, len(seq), 20)
    s = jnp.array([[residue_constants.restype_order.get(aa, -1) for aa in s] for s in [seq]])
    if jnp.issubdtype(s.dtype, jnp.integer):
        s_ = jnp.eye(A)[s]
        s_ = s_.at[s == -1].set(0)
        return s_
    return s


def worker(
    work: tuple[str, Path, DictConfig],
    save_path: Path,
    config: DictConfig,
    *,
    af2_binder_model: mk_afdesign_model,
    af2_monomer_model: mk_afdesign_model,
) -> pl.LazyFrame | None:
    design_name, structure_filepath, design_data = work
    _tokenize_seq = lambda seq: [residue_constants.restype_order[aa] for aa in seq]
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
    mpnn_model = mk_mpnn_model(weights = config["proteinmpnn"]["weights"])
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
        use_binder_template = False,
    )
    af2_monomer_model.prep_inputs(
        pdb_filename = str(structure_filepath),
        chain = binder_chain,
    )

    binder_seqs = jnp.array(
        [_prepare_seq_for_af2(s.split("/")[binder_seq_pos]) for s in samples["seq"]]
    )
    names = [f"{design_name}_seq_{i}" for i in range(len(binder_seqs))]
    batch_results = {
        k: np.array(v)
        for k, v in _parallel_predict_structure(
            af2_binder_model,
            binder_seqs,
        ).items()
        if v is not None
    }
    batch_monomer_results = {
        f"binder_{k}": np.array(v)
        for k, v in _parallel_predict_structure(
            af2_monomer_model,
            binder_seqs,
        ).items()
        if v is not None
    }
    lf = (
        pl.concat(
            [
                pl.LazyFrame(batch_results),
                pl.LazyFrame(batch_monomer_results),
            ],
            how = "horizontal",
        )
        .with_columns(
            name = pl.Series(names),
            binder_seq_pos = pl.Series([binder_seq_pos] * len(names)),
            seq = pl.Series(samples["seq"]),
        )
        .select(
            "name",
            "seq",
            "binder_seq_pos",
            *batch_results.keys(),
            *batch_monomer_results.keys(),
        )
    )
    af2_binder_model.restart()
    af2_monomer_model.restart()
    return lf


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
    final_results = []
    save_path = Path(config["design"]["store_path"]).resolve().joinpath("af2_binder")
    if not save_path.exists():
        save_path.mkdir(parents = True)
    params_path = Path.home().joinpath(".cache")
    af2_binder_model = mk_afdesign_model(
        protocol = "binder",
        data_dir = params_path,
        use_multimer = True,
        use_initial_guess = True,
        use_templates = True,
    )
    af2_monomer_model = mk_afdesign_model(protocol = "fixbb", data_dir = params_path)
    with tracker:
        task = tracker.add_task("MPNN and AF2 evaluation", total = n_design_tasks)
        func = functools.partial(
            worker,
            config = config,
            save_path = save_path,
            af2_binder_model = af2_binder_model,
            af2_monomer_model = af2_monomer_model,
        )
        for result in (func(w) for w in work):
            if result is not None:
                final_results.append(result)
                tracker.update(task, advance = 1)
    pl.concat(final_results).sink_parquet(save_path / "metrics.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())