# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "biotite",
#     "click",
#     "omegaconf",
#     "proteinbee",
# ]
# ///

import biotite.structure as biostructure
import biotite.structure.io as structureio

import click

import numpy as np

from omegaconf import OmegaConf

from pathlib import Path

from proteinbee.motif import Motif

import pickle

import re

import sys


@click.command()
@click.argument("input-path", type = Path)
@click.argument("binder-chain", type = str)
@click.argument("target-chain", type = str)
@click.option("--sampled-motif-order", "-smo", type = str, default = "bt", help = "The order in which the 'contigmap' was specified b = binder t = target")
@click.option("--out-path", "-op", type = Path, default = None, help = "The path to the directory where the config will be saved.")
def main(
    input_path: Path,
    binder_chain: str,
    target_chain: str,
    sampled_motif_order: str,
    out_path: Path | None,
) -> None:
    if out_path is None:
        out_path = input_path.resolve().joinpath("relaxed")
    chain_mask = {
        "b": binder_chain,
        "t": target_chain,
    }
    config = {
        "design": {
            "store_path": str(out_path),
            "detail": {},
        },
        "proteinmpnn": {
            "weights": "soluble",
            "sequence": {
                "N": 8,
                "temperature": 0.1,
            },
        },
        "af2": {
            "model_nums": [0, 1],
            "num_recycles": 3,
        },
    }
    _re = re.compile(r"([A-Z]\d+-\d+)")
    for design_filepath in input_path.glob("design_*.pdb"):
        filename = design_filepath.name
        design_structure = structureio.load_structure(design_filepath)
        design_structure.res_id = biostructure.create_continuous_res_ids(design_structure, restart_each_chain = True)
        structureio.save_structure(design_filepath, design_structure)
        design_config = pickle.load(design_filepath.with_suffix(".trb").open("rb"))
        motif = []
        assert len(sampled_motif_order) == len(design_config["sampled_mask"])
        for comp, g in zip(sampled_motif_order, design_config["sampled_mask"]):
            s = []
            for m in g.split("/"):
                first_char = m[0]
                if first_char.isnumeric() and first_char != "0":
                    s.append(m.split("-")[0])
                elif first_char.isalpha():
                    design_motif = Motif.from_string(m).get_motif_wrt_designed_structure(chain_mask[comp])
                    s.append(str(design_motif))
            motif.append("/".join(s))
        motif = "/0/".join(motif)
        motif = Motif.from_string(motif) # Sanity check
        config["design"]["detail"].update(
            {
                f"{filename.split(".")[0]}": {
                    "filename": filename,
                    "motif": str(motif),
                    "components": [
                        {
                            "protein": {
                                "chain": binder_chain,
                                "name": "binder", 
                            },
                        },
                        {
                            "protein": {
                                "chain": target_chain,
                                "name": "target", 
                            },
                        },
                    ],
                },
            }
        )
    OmegaConf.save(config, input_path / "config.yaml")


if __name__ == "__main__":
    raise SystemExit(main())
