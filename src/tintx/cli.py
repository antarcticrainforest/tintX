"""Command line interface for tintX tracking."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional

import click
from tintx import __version__, RunDirectory, config


CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
)
PROG_NAME = "tintx"


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, "-V", "--version", prog_name=PROG_NAME)
def tintx(argv: Optional[list[str]] = None, standalone_mode: bool = True) -> None:
    """Command line interface (cli) of the tintX tracking algorithm.

    The cli offers two sub commands. One for applying the tracking algorithm,
    one for visualisation of already tracked data.
    """
    return


def _get_file_names(input_paths: tuple[Path, ...]) -> Iterator[Path]:

    suffixes = (".nc", ".nc4", ".grb", ".grib")
    for input_path in input_paths:
        if input_path.is_file() and input_path.suffix in suffixes:
            yield input_path
        elif input_path.is_dir():
            for input_file in input_path.rglob("*.*"):
                if input_file.suffix in suffixes:
                    yield input_file
        else:
            # This is a long shot, we assume that a glob pattern was given
            for input_file in input_path.parent.rglob(input_path.name):
                yield input_file


@tintx.command()
@click.argument(
    "input_file",
    nargs=1,
    type=click.Path(
        resolve_path=True, path_type=Path, exists=True, dir_okay=False
    ),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(resolve_path=True, path_type=Path, dir_okay=False),
    default=None,
    help=(
        "Path to a visualisation of the tracking. If `None` is given, no "
        "visualisation will be created. The type of visualisation will be "
        "determined from the file type. E.i for `.png` or `.jpg` files "
        "trajectory plots will be created for `.mp4`, `.gif` animations "
        "will be created."
    ),
)
@click.option("--animate", is_flag=True, help="Create animation")
@click.option(
    "--dt",
    type=click.FLOAT,
    default=0,
    help="Offset in hours from UTC. Animation only",
)
@click.option(
    "--fps",
    type=click.FLOAT,
    default=2,
    help="Play back speed of an animation. Animation only",
)
@click.option(
    "--cmap",
    type=click.STRING,
    default="Blues",
    help="Colormap used for animations.",
)
@click.option(
    "--vmin", type=click.FLOAT, default=0, help="minimum values to be plotted"
)
@click.option(
    "--vmax",
    type=click.FLOAT,
    default=3,
    help="Maximum values to display. Animation only",
)
@click.option(
    "--mintrace",
    type=click.FLOAT,
    default=2,
    help="Minimum length of a trace to be plotted: Trajectory plot only",
)
@click.option(
    "--markersize",
    "-ms",
    type=click.FLOAT,
    default=25,
    help="Help marker size of the trijectory plot",
)
@click.option(
    "--linewidth",
    "-lw",
    type=click.FLOAT,
    default=1,
    help="Line width to be plotted.",
)
def plot(
    input_file: Path,
    output: Optional[Path] = None,
    cmap: str = "Blues",
    animate: bool = False,
    **kwargs: float,
) -> None:
    """Plot/Animate existing tracking data.

    Arguments:

    input_files:
        Filename of the HDF5 file contaning the tracking data.

    """

    run_dir = RunDirectory.from_dataframe(input_file)
    time_suffix = (
        f'{run_dir.start.strftime("%Y%m%dT%H%M")}-'
        f'{run_dir.end.strftime("%Y%m%dT%H%M")}'
    )
    outf = Path(".").absolute() / f"tintx_tracks_{run_dir.var_name}_{time_suffix}"
    if animate:
        output = output or outf.with_suffix(".mp4")
        output.parent.mkdir(exist_ok=True, parents=True)
        anim = run_dir.animate(
            cmap=cmap,
            vmin=kwargs["vmin"],
            vmax=kwargs["vmax"],
            dt=kwargs["dt"],
            fps=kwargs["fps"],
            plot_style={"lw": kwargs["linewidth"]},
        )
        anim.save(output, fps=kwargs["fps"])

    else:
        output = output or outf.with_suffix(".png")
        output.parent.mkdir(exist_ok=True, parents=True)
        axes = run_dir.plot_trajectories(
            thresh=kwargs["vmin"],
            mintrace=int(kwargs["mintrace"]),
            plot_style={"lw": kwargs["linewidth"], "ms": kwargs["markersize"]},
        )
        axes.figure.savefig(output, bbox_inches="tight")


@tintx.command()
@click.argument("variable", nargs=1, type=click.STRING)
@click.argument(
    "input_files",
    nargs=-1,
    type=click.Path(resolve_path=True, path_type=Path),
)
@click.option(
    "--start",
    "-s",
    type=click.STRING,
    default=None,
    help="ISO-8601 string representation of the first tracking time step.",
)
@click.option(
    "--end",
    "-e",
    type=click.STRING,
    default=None,
    help="ISO-8601 string representation of the last tracking time step.",
)
@click.option(
    "--x-coord",
    type=click.STRING,
    default="lon",
    help="Name of the X (eastward) coordinate",
)
@click.option(
    "--y-coord",
    type=click.STRING,
    default="lat",
    help="Name of the Y (northward) coordinate",
)
@click.option(
    "--time-coord",
    type=click.STRING,
    default="time",
    help="Name of the time coordinate",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(resolve_path=True, path_type=Path),
    default=None,
    help=(
        "Output file where the tracking results pandas DataFrame is saved "
        "to. If `None` given (default) the output filename will be set from "
        "the input meta data and the variable name."
    ),
)
@click.option(
    "--field-thresh",
    type=click.FLOAT,
    default=32,
    help=(
        "Threshold used for object detection. Detected objects are "
        "connected pixels above this threshold."
    ),
)
@click.option(
    "--iso-thresh",
    type=click.FLOAT,
    default=4.0,
    help=(
        "Used in isolated cell classification. Isolated cells must not be  "
        "connected to any other cell by contiguous pixels above this "
        "threshold."
    ),
)
@click.option(
    "--min-size",
    type=click.FLOAT,
    default=8.0,
    help=("Minimum size threshold in pixels for an object to be detected."),
)
@click.option(
    "--search-margin",
    type=click.FLOAT,
    default=250.0,
    help=("Radius of the search box around the predicted object center."),
)
@click.option(
    "--flow-margin",
    type=click.FLOAT,
    default=750,
    help=("Margin size around objects to perform phase correlation."),
)
@click.option(
    "--max-disparity",
    type=click.FLOAT,
    default=999.0,
    help=("Maximum allowable disparity value."),
)
@click.option(
    "--max-flow-mag",
    type=click.FLOAT,
    default=50.0,
    help=("Maximum allowable global shift magnitude."),
)
@click.option(
    "--max-shift-disp",
    type=click.FLOAT,
    default=15.0,
    help=(
        "Maximum magnitude of difference in meters per second for two shifts"
        " to be considered in agreement."
    ),
)
@click.option(
    "--gs-alt",
    type=click.FLOAT,
    default=1500.0,
    help=(
        "Altitude in meters at which to perform phase correlation for global"
        "shift calculation. 3D data only"
    ),
)
@click.option(
    "--iso-smooth",
    type=click.FLOAT,
    default=4.0,
    help=("Gaussian smoothing parameter in peak detection preprocessing."),
)
def track(
    variable: str,
    input_files: tuple[Path, ...],
    start: Optional[str] = None,
    end: Optional[str] = None,
    x_coord: str = "lon",
    y_coord: str = "lat",
    time_coord: str = "time",
    output: Optional[Path] = None,
    **parameters: float,
) -> None:
    """Apply the tintX tracking algorithm.

    The sub command takes at least two arguments and attempts of read data
    saved in netcdf/grib format and apply the tracking to a data variable
    within the dataset.

    Arguments:

    variable:
        Variable name of the data that is tracked

    input_files:
        Filename(s) or Directory where the data is stored.
    """
    with config.set(**parameters):
        run_d = RunDirectory.from_files(
            _get_file_names(input_files),
            variable,
            start=start,
            end=end,
            x_coord=x_coord,
            y_coord=y_coord,
            time_coord=time_coord,
            use_cftime=True,
            combine="by_coords",
        )
        time_suffix = (
            f'{run_d.start.strftime("%Y%m%dT%H%M")}-'
            f'{run_d.end.strftime("%Y%m%dT%H%M")}'
        )
        outf = (
            Path(".").absolute() / f"tintx_tracks_{variable}_{time_suffix}.hdf5"
        )
        output = output or outf
        num_tracks = run_d.get_tracks()
        click.echo(f"Found and tracked {num_tracks} objects.")
        output.parent.mkdir(exist_ok=True, parents=True)
        run_d.save_tracks(output)
