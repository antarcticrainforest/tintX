"""Visualization tools for tracks objects."""

from __future__ import annotations
from functools import partial
from datetime import timedelta
from typing import Any, Optional, Iterator, Union

from cartopy import crs
from cartopy.mpl.geoaxes import GeoAxesSubplot
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import xarray as xr

from .tracks import Cell_tracks
from .types import GridType
from .helpers import convert_to_cftime


class Tracer:
    colors = ["m", "r", "lime", "darkorange", "k", "b", "darkgreen", "yellow"]
    colors.reverse()

    def __init__(self, tobj: Cell_tracks) -> None:
        self.tobj = tobj
        self.color_stack = self.colors * 10
        self.cell_color = pd.Series([], dtype=str)
        self.history: pd.DataFrame = pd.DataFrame()
        self.current: pd.DataFrame = pd.DataFrame()

    def update(self, nframe: int) -> None:
        self.history = self.tobj.tracks.loc[:nframe]
        self.current = self.tobj.tracks.loc[nframe]

    def _check_uid(self, uid: str) -> None:
        if uid not in self.cell_color.keys():
            try:
                self.cell_color[uid] = self.color_stack.pop()
            except IndexError:
                self.color_stack += self.colors * 5
                self.cell_color[uid] = self.color_stack.pop()

    def plot(self, ax: GeoAxesSubplot) -> None:
        for uid, group in self.history.groupby(level="uid"):
            self._check_uid(uid)
            tracer = group[["lon", "lat"]]
            if uid in self.current.index:
                ax.plot(tracer.lon, tracer.lat, self.cell_color[uid])


def _get_axes(
    X: Union[np.ndarray, xr.DataArray],
    Y: Union[np.ndarray, xr.DataArray],
    ax: Optional[GeoAxesSubplot],
    **kwargs: Union[float, int, str],
) -> GeoAxesSubplot:
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection=crs.PlateCarree())
    if not isinstance(ax, GeoAxesSubplot):
        raise TypeError("Ax should be a cartopy GeoAxesSubplot object")
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlim(X.min(), X.max())
    kwargs.setdefault("resolution", "10m")
    ax.coastlines(**kwargs)
    return ax


def _gen_from_grids(
    total_num: int, grid_0: GridType, grids: Iterator[GridType]
) -> Iterator[GridType]:

    with tqdm(total=total_num, desc="Animating", leave=False) as pbar:
        pbar.update()
        yield grid_0
        for grid in grids:
            pbar.update()
            yield grid


def full_domain(
    tobj: Cell_tracks,
    grids: Iterator[GridType],
    vmin: float = 0.01,
    vmax: float = 15,
    ax: Optional[GeoAxesSubplot] = None,
    cmap: Union[str, plt.cm] = "Blues",
    alt: Optional[float] = None,
    fps: float = 5,
    isolated_only: bool = False,
    tracers: bool = False,
    dt: float = 0,
    plot_style: Optional[dict[str, Union[float, int, str]]] = None,
) -> GeoAxesSubplot:
    alt = alt or tobj.params["GS_ALT"]
    plot_style = plot_style or {}
    shading = plot_style.pop("shading", "auto")
    if tracers:
        tracer = Tracer(tobj)
    nframes = tobj._tracks.index.levels[0].max()
    title = plot_style.pop("title", "")
    grid = next(grids)
    new_ax = _get_axes(grid.lon, grid.lat, ax, **plot_style)
    try:
        data = grid.data[0].filled(np.nan)
    except AttributeError:  # pragma: no cover
        data = grid.data[0]  # pragma: no cover
    im = new_ax.pcolormesh(
        grid.lon,
        grid.lat,
        data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        shading=shading,
    )

    ann: dict[str, mpl.text.Annotation] = {}

    def _update(enum: tuple[int, GridType], title: str = "") -> None:
        for annotation in ann.values():
            try:
                annotation.remove()
            except ValueError:  # pragma: no cover
                pass  # pragma: no cover
        nframe, grid = enum
        try:
            im.set_array(grid.data[0].filled(np.nan).ravel())
        except AttributeError:  # pragma: no cover
            im.set_array(grid.data[0].ravel())  # pragma: no cover
        title_text = ""
        if title:
            title_text = f"{title} at "
        time_str = (convert_to_cftime(grid.time) + timedelta(hours=dt)).strftime(
            "%Y-%m-%d %H:%M"
        )
        new_ax.set_title(f"{title_text}{time_str}")
        if nframe in tobj._tracks.index.levels[0]:
            frame_tracks = tobj._tracks.loc[nframe]
            if tracers:
                tracer.update(nframe)
                tracer.plot(new_ax)
            for ind, uid in enumerate(frame_tracks.index):
                if isolated_only and not frame_tracks["isolated"].iloc[ind]:
                    continue
                x = frame_tracks["lon"].iloc[ind]
                y = frame_tracks["lat"].iloc[ind]
                ann[uid] = new_ax.annotate(uid, (x, y), fontsize=20)

    frames = enumerate(_gen_from_grids(nframes, grid, grids))
    animation = FuncAnimation(
        new_ax.get_figure(),
        partial(_update, title=title),
        frames=frames,
        interval=1000 / fps,
    )
    plt.close()
    return animation


def plot_traj(
    traj: pd.DataFrame,
    X: Union[np.ndarray, xr.DataArray],
    Y: Union[np.ndarray, xr.DataArray],
    label: bool = False,
    ax: Optional[GeoAxesSubplot] = None,
    uids: list[str] = None,
    mintrace: int = 2,
    size: int = 50,
    thresh: float = -1.0,
    color: Optional[str] = None,
    plot_style: Optional[dict[str, Union[float, int, str]]] = None,
) -> GeoAxesSubplot:

    """This code is a fork of plot_traj method in the plot module from the
    trackpy project see http://soft-matter.github.io/trackpy fro more details

    Plot traces of trajectories for each particle.

    Parameters
    ----------
    tobj : trajectory containing the tracking object
    X : 1D array of the X vector
    Y : 1D array of the Y vector
    label : boolean, default: False
        Set to True to write particle ID numbers next to trajectories.
    cmap : colormap, Default matplotlib.colormap.winter
        Colormap used to color different tracks
    ax: cartopy.mpl.geoaxes.GeoAxesSubplot, default: None
        Axes object that is used to create the animation. If None (default)
        a new axes object will be created.
    uids : list[str], default: None
    a preset of stroms to be drawn, instead of all
        (default)
    color : str, default: None
    A pre-defined color, if None (default) each track will be assigned a
    different color
    thresh : float, default: -1
        plot only trajectories with average intensity above this value.
    mintrace : int, default 2
        Minimum length of a trace to be plotted
    plot_style : dictionary
        Keyword arguments passed through to the `Axes.plot(...)` command

    Returns
    -------
    Axes object

    """

    plot_style = plot_style or {}
    _plot_style = dict(linewidth=1)
    resolution = plot_style.pop("resolution", "10m")
    _plot_style.update(**_normalize_kwargs(plot_style, "line2d"))
    size = _plot_style.pop("markersize", 50)
    ax = _get_axes(X, Y, ax, resolution=resolution, **_plot_style)
    proj = ax.projection
    if len(traj) == 0:
        raise ValueError("DataFrame of trajectories is empty.")

    # Read http://www.scipy.org/Cookbook/Matplotlib/MulticoloredLine
    y = traj["lat"]
    x = traj["lon"]
    val = traj["mean"]
    if uids is None:
        uid = np.unique(x.index.get_level_values("uid")).astype(np.int32)
        uid.sort()
        uids = uid.astype(str)

    for particle in uids:
        x1 = x[:, particle].values
        y1 = y[:, particle].values
        mean1 = val[:, particle].values.mean()
        if x1.shape[0] > int(mintrace) and mean1 >= thresh:
            im = ax.plot(x1, y1, color=color, **_plot_style)
            sc_color = im[0].get_color()
            ax.scatter(x1[0], y1[0], marker="o", color=sc_color, s=[size])
            ax.scatter(x1[-1], y1[-1], marker="*", color=sc_color, s=[size])
            if label:
                if len(x1) > 1:
                    cx, cy = proj.transform_point(
                        x1[int(x1.size / 2)], y1[int(y1.size / 2)], proj
                    )
                    dx, dy = proj.transform_point(
                        ((x1[1] - x1[0]) / 8.0), ((y1[1] - y1[0]) / 8.0), proj
                    )
                else:
                    cx, cy = proj.transform_point(x1[0], y1[0], proj)
                    dx, dy = 0, 0
                ax.annotate(
                    "%s" % str(particle),
                    xy=(cx, cy),
                    xytext=(cx + dx, cy + dy),
                    fontsize=16,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
        else:
            im = None
    return ax


def _normalize_kwargs(
    kwargs: dict[str, Any], kind: str = "patch"
) -> dict[str, Any]:
    """Convert matplotlib keywords from short to long form."""
    # Source:
    # github.com/tritemio/FRETBursts/blob/fit_experim/fretbursts/burst_plot.py
    if kind == "line2d":
        long_names = dict(
            c="color",
            ls="linestyle",
            lw="linewidth",
            mec="markeredgecolor",
            mew="markeredgewidth",
            mfc="markerfacecolor",
            ms="markersize",
        )
    elif kind == "patch":
        long_names = dict(
            c="color",
            ls="linestyle",
            lw="linewidth",
            ec="edgecolor",
            fc="facecolor",
        )
    _ = kwargs.pop("resolution", None)
    for short_name in long_names.keys():
        if short_name in kwargs:
            kwargs[long_names[short_name]] = kwargs.pop(short_name)
    return kwargs
