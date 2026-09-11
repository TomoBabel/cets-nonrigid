"""CETS command-line interface; native direction aliases are intentionally absent."""

import json
from pathlib import Path
import click

from cets_nonrigid import __version__
from cets_nonrigid.cli.exchange import register


@click.group(name="cets-nonrigid")
@click.version_option(version=__version__)
def main():
    """Exchange sampled non-rigid alignments through CETS."""


register(main)


@main.command("inspect")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--source", type=click.Choice(["aretomo3", "warp", "relion", "mcaln", "warp-movie", "relion-motion"]))
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Native source options as JSON.")
def inspect_command(path, source, config):
    """Describe a CETS document or a native alignment."""
    from cets_nonrigid.api import inspect_alignment

    try:
        options = json.loads(config.read_text()) if config else {}
        click.echo(json.dumps(inspect_alignment(path, source=source, **options), indent=2, default=str))
    except (ValueError, OSError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("validate")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def validate_command(path):
    """Check core references, geometry, payload shapes and context digests."""
    from cets_nonrigid.api import validate_bundle

    try:
        click.echo(json.dumps(validate_bundle(path), indent=2, default=str))
    except (ValueError, OSError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("report")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Write the report JSON to a new file.")
def report_command(path, output):
    """Report recorded validation and fitting results."""
    from cets_nonrigid.api import report_bundle

    try:
        text = json.dumps(report_bundle(path), indent=2, default=str) + "\n"
        if output:
            with output.open("x") as handle:
                handle.write(text)
        else:
            click.echo(text, nl=False)
    except (ValueError, OSError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
