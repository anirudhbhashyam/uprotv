import click

import os

import subprocess

from typing import (
    Any,
    Callable,
)


@click.group()
def main():
    pass


def _add_commands() -> None:
    commands_path = os.path.join(os.path.dirname(__file__), "commands")
    for filename in os.listdir(commands_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            command_name = filename[: -3].replace("_", "-")
            script_path = os.path.join(commands_path, filename)

            def command_runner_wrapper(path) -> Callable[[tuple[Any]], None]:
                @click.command(
                    name = command_name,
                    context_settings = {
                        "help_option_names": [],
                        "ignore_unknown_options": True,
                    },
                )
                @click.argument("script_args", nargs = -1, type = click.UNPROCESSED)
                def _runner(script_args: tuple[Any]) -> None:
                    try:
                        if "--help" in script_args:
                            command = ["uv", "run", path, "--help"]
                        else:
                            command = ["uv", "run", path] + list(script_args)
                        subprocess.check_call(command)
                    except FileNotFoundError:
                        click.echo("Error: 'uv' command not found. Is uv installed and discoverable?", err = True)
                        raise SystemExit(1)
                    except subprocess.CalledProcessError as e:
                        raise SystemExit(e.returncode)
                return _runner
            main.add_command(command_runner_wrapper(script_path))


_add_commands()
