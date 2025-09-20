# uprotv

A collection of CLI tools in a convenient CLI that lets you do routine protein design tasks.


## Installation

### Install uv

See: https://docs.astral.sh/uv/getting-started/installation/#installation-methods or do the following

```bash
pip install uv
```
And that's it!

### Running commands
From the repostory do 

```bash
uv uprotv <command> <args> <options>
```

## Available commands

### `af2-binder-eval`

Evaluate binder designs using AF2 and ProteinMPNN. It takes in a specific config file that determines where your backbones are what they contain. Roughly:

```yaml
design:
  store_path: <path_to_structures>
  detail:
    <name>:
      filename: <filename>
      motif: <motif>
      components:
      - protein:
          chain: <chain>
          name: binder
      - protein:
          chain: <chain>
          name: target
    <name>:
      filename: <filename>
      motif: <motif>
      components:
      - protein:
          chain: <chain>
          name: binder
      - protein:
          chain: <chain>
          name: target
    .
    .
    .
```

The motif is optional and can be ommited if all residues in the binder need to be designed. The motif argument works similarly to what is seen in [`RFdiffusion`](https://github.com/RosettaCommons/RFdiffusion).


### `generate-config-from-rfdiffusion-run`

Generate a config file from a RFdiffusion run. The command will generate a `yaml` file based on a provided input directory containing backbones from RFdiffusion.






