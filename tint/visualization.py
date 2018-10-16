"""
tint.visualization
==================

Visualization tools for tracks objects.

"""

import gc
import os
import pandas as pd
import numpy as np
import shutil
import tempfile
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from IPython.display import display, Image
from matplotlib import pyplot as plt
from datetime import timedelta
import sys
from .grid_utils import get_grid_alt


class Tracer(object):
    colors = ['m', 'r', 'lime', 'darkorange', 'k', 'b', 'darkgreen', 'yellow']
    colors.reverse()

    def __init__(self, tobj, persist):
        self.tobj = tobj
        self.persist = persist
        self.color_stack = self.colors * 10
        self.cell_color = pd.Series()
        self.history = None
        self.current = None

    def update(self, nframe):
        self.history = self.tobj.tracks.loc[:nframe]
        self.current = self.tobj.tracks.loc[nframe]
        if not self.persist:
            dead_cells = [key for key in self.cell_color.keys()
                          if key
                          not in self.current.index.get_level_values('uid')]
            self.color_stack.extend(self.cell_color[dead_cells])
            self.cell_color.drop(dead_cells, inplace=True)

    def _check_uid(self, uid):
        if uid not in self.cell_color.keys():
            try:
                self.cell_color[uid] = self.color_stack.pop()
            except IndexError:
                self.color_stack += self.colors * 5
                self.cell_color[uid] = self.color_stack.pop()

    def plot(self, ax):
        for uid, group in self.history.groupby(level='uid'):
            self._check_uid(uid)
            tracer = group[['grid_x', 'grid_y']]
            tracer = tracer*self.tobj.grid_size[[2, 1]]
            if self.persist or (uid in self.current.index):
                ax.plot(tracer.grid_x, tracer.grid_y, self.cell_color[uid])


def full_domain(tobj, grids, tmp_dir, vmin=0.01, vmax=15, cmap=None, alt=None,
                basemap_res='f', isolated_only=False, tracers=False,
                persist=False, m=None, dt=0):

    grid_size = tobj.grid_size
    if cmap is None:
        cmap = mpl.cm.Blues
    try :
        cmap.set_under('w')
        cmap.set_bad('w')
    except:
        pass
    if alt is None:
        alt = tobj.params['GS_ALT']
    if tracers:
        tracer = Tracer(tobj, persist)
    radar_lon = tobj.radar_info['radar_lon']
    radar_lat = tobj.radar_info['radar_lat']
    #lon = np.arange(radar_lon-5, radar_lon+5, 0.5)
    #lat = np.arange(radar_lat-5, radar_lat+5, 0.5)
    nframes = tobj.tracks.index.levels[0].max() + 1
    print('Animating', nframes, 'frames')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for nframe, grid in enumerate(grids):
        sys.stdout.flush()
        sys.stdout.write('\rFrame: %s       ' %nframe)
        sys.stdout.flush()
        #display.plot_crosshairs(lon=radar_lon, lat=radar_lat)
        if nframe == 0:
            X = grid['x']
            Y = grid['y']
            m = Basemap(llcrnrlat=min(Y), llcrnrlon=min(X), urcrnrlat=max(Y),
                        urcrnrlon=max(X), resolution=basemap_res, ax=ax)
            m.drawcoastlines()
            try:
                im = m.pcolormesh(X, Y, grid['data'][0].filled(np.nan),
                        vmin=vmin, vmax=vmax, cmap=cmap, shading='gouraud')
            except AttributeError:
                im = m.pcolormesh(X, Y, grid['data'][0],
                        vmin=vmin, vmax=vmax, cmap=cmap, shading='gouraud')
        else:
            try:
                im.set_array(grid['data'][0].filled(np.nan).ravel())
            except AttributeError:
                im.set_array(grid['data'][0].ravel())
        ax.set_title('Rain-rate at %s'\
                     %((grid['time']+timedelta(hours=dt)).strftime('%Y-%m-%d %H:%M')))
        ann = []
        if nframe in tobj.tracks.index.levels[0]:
            frame_tracks = tobj.tracks.loc[nframe]

            if tracers:
                tracer.update(nframe)
                tracer.plot(ax)

            for ind, uid in enumerate(frame_tracks.index):
                if isolated_only and not frame_tracks['isolated'].iloc[ind]:
                    continue
                
                x = frame_tracks['lon'].iloc[ind]
                y = frame_tracks['lat'].iloc[ind]
                ann.append(ax.annotate(uid, (x, y), fontsize=20))
        plt.savefig(tmp_dir + '/frame_' + str(nframe).zfill(3) + '.png')
        for an in ann:
            try:
                an.remove()
            except ValueError:
                pass

        gc.collect()
    plt.close()
    del grid, ax



def get_plotly_traj(traj, label=None, thresh=('max',-1), particles=None,
                    color='grey', mintrace=2, **kwargs):
    """This method get track information and returns a plotly dict for the
        tracks.

        Parameters
        ----------
        tobj : trajectory containing the tracking object
        X : 1D array of the X vector
        Y : 1D array of the Y vector
        colorby : {'particle', 'frame'}, optional
        mpp : float, optional
            Microns per pixel. If omitted, the labels will have units of pixels.
        label : boolean, optional
            Set to True to write particle ID numbers next to trajectories.
        particles : a preset of storms (particles) to be drawn, instead of all
            (default)
        size : size of the sctter indicating start and end of the storm
        color : Color of the storm tracks (default grey)
        thresh : tuple for thresholds to be applied to the plotted objects. first
            entry of the tuple is the variable (default 'mean') second one the
            the minimum value (default -1)
        kwargs : extra arguments for plotting
        mintrace : int
            Minimum length of a trace to be plotted
        Returns
        -------
        Axes object
        

    """

    y = traj['lat']
    x = traj['lon']
    val = traj[thresh[0]]
    uid = np.unique(x.index.get_level_values('uid')).astype(np.int32)
    color_numbers = uid.max()
    uid.sort()
    paths = []
    if particles is None:
        uid = np.unique(x.index.get_level_values('uid')).astype(np.int32)
        color_numbers = uid.max()
        uid.sort()
        particles = uid.astype(str)
    else:
        color_numbers = len(particles)

    if particles is None:
        uid = np.unique(x.index.get_level_values('uid')).astype(np.int32)
        color_numbers = uid.max()
        uid.sort()
        particles = uid.astype(str)
    else:
        color_numbers = len(particles)


    for particle in uid.astype(str):
        try:
            x1 = x[:,particle].values
            y1 = y[:,particle].values
            mean1 = val[:,particle].values.mean()
        except KeyError:
            x1 = x[:,int(particle)].values
            y1 = y[:,int(particle)].values
            mean1 = val[:,int(particle)].values.mean()
        if x1.shape[0] > int(mintrace) and mean1 >= thresh[1]:
            paths.append(dict(
                            type='scattergeo',
                            lon=list(x1),
                            lat=list(y1),
                            mode='lines',
                            line=dict(width=1, color=color),
                            opacity=float(mean1/val.max()),
                            **kwargs))
    return paths



def plot_traj(traj, X, Y, mpp=None, label=False, basemap_res='i',
              superimpose=None, cmap=None, ax=None, t_column=None, particles=None,
              pos_columns=None, plot_style={}, mintrace=2, size=100,
              thresh=('mean', -1), color=None, create_map=None, **kwargs):

    """This code is a fork of plot_traj method in the plot module from the
    trackpy project see http://soft-matter.github.io/trackpy fro more details

    Plot traces of trajectories for each particle.
    Optionally superimpose it on a frame from the video.
    Parameters
    ----------
    tobj : trajectory containing the tracking object
    X : 1D array of the X vector
    Y : 1D array of the Y vector
    colorby : {'particle', 'frame'}, optional
    mpp : float, optional
        Microns per pixel. If omitted, the labels will have units of pixels.
    label : boolean, optional
        Set to True to write particle ID numbers next to trajectories.
    basemap_res: str
        Set the resolution of the basemap
    superimpose : ndarray, optional
        Background image, default None
    cmap : colormap, optional
        This is only used in colorby='frame' mode. Default = mpl.cm.winter
    ax : matplotlib axes object, optional
        Defaults to current axes
    t_column : string, optional
        DataFrame column name for time coordinate. Default is 'frame'.
    particles : a preset of stroms (particles) to be drawn, instead of all
        (default)
    size : size of the sctter indicating start and end of the storm
    color : A pre-defined color, if None (default) each track will be assigned
        a different color
    thresh : tuple for thresholds to be applied to the plotted objects. first
        entry of the tuple is the variable (default 'mean') second one the
        the minimum value (default -1)
    create_map: boolean, reate a map object, this can be useful for loops where
        a basemap object has already been created
    pos_columns : list of strings, optional
        Dataframe column names for spatial coordinates. Default is ['x', 'y'].
    plot_style : dictionary
        Keyword arguments passed through to the `Axes.plot(...)` command
    mintrace : int
        Minimum length of a trace to be plotted
    Returns
    -------
    Axes object
    
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
    if create_map is None :
        m = Basemap(llcrnrlat=min(Y), llcrnrlon=min(X), urcrnrlat=max(Y),
                urcrnrlon=max(X), resolution=basemap_res, ax=ax)
        try:
            lw=plot_style['lw']
        except KeyError:
            lw=0.5
        m.drawcoastlines(linewidth=lw)
    else:
        m = create_map

    if cmap is None:
        cmap = plt.cm.winter
    if t_column is None:
        t_column = 'scan'
    if pos_columns is None:
        pos_columns = ['lon', 'lat']
    if len(traj) == 0:
        raise ValueError("DataFrame of trajectories is empty.")
    _plot_style = dict(linewidth=1)
    _plot_style.update(**_normalize_kwargs(plot_style, 'line2d'))

    # Axes labels
    if mpp is None:
        #_set_labels(ax, '{} [px]', pos_columns)
        mpp = 1.  # for computations of image extent below
    else:
        if mpl.rcParams['text.usetex']:
            _set_labels(ax, r'{} [\textmu m]', pos_columns)
        else:
            _set_labels(ax, r'{} [\xb5m]', pos_columns)
    # Background image
    if superimpose is not None:
        ax.imshow(superimpose, cmap=plt.cm.gray,
                  origin='lower', interpolation='nearest',
                  vmin=kwargs.get('vmin'), vmax=kwargs.get('vmax'))
        ax.set_xlim(-0.5 * mpp, (superimpose.shape[1] - 0.5) * mpp)
        ax.set_ylim(-0.5 * mpp, (superimpose.shape[0] - 0.5) * mpp)
    # Trajectories
    # Read http://www.scipy.org/Cookbook/Matplotlib/MulticoloredLine
    y = traj['lat']
    x = traj['lon']
    val = traj[thresh[0]]
    if particles is None:
        uid = np.unique(x.index.get_level_values('uid')).astype(np.int32)
        color_numbers = uid.max()
        uid.sort()
        particles = uid.astype(str)
    else:
        color_numbers = len(particles)

    for particle in particles:
        try:
            x1 = x[:,particle].values
            y1 = y[:,particle].values
            mean1 = val[:,particle].values.mean()
        except KeyError:
            x1 = x[:,int(particle)].values
            y1 = y[:,int(particle)].values
            mean1 = val[:,int(particle)].values.mean()
        if x1.shape[0] > int(mintrace) and mean1 >= thresh[1]:
            alpha=float(mean1/val.max())
            if color is not None:
                im = m.plot(x1,y1, color=color, **plot_style)
            else:
                im = m.plot(x1,y1, color=color, **plot_style)
            m.scatter(x1[0], y1[0], marker='o', color=color, s=[size])
            m.scatter(x1[-1], y1[-1], marker='*', color=color, s=[size])
            if label:
                if len(x1) > 1:
                    cx, cy = m(x1[int(x1.size/2)], y1[int(y1.size/2)])
                    dx,dy =m(((x1[1]-x1[0])/8.),((y1[1]-y1[0])/8.))
                else:
                    cx, cy = m(x1[0], y1[0])
                    dx, dy = 0, 0
                ax.annotate('%s'%str(particle), xy=(cx, cy), xytext=(cx+dx,cy+dy),
                            fontsize=16, horizontalalignment='center',
                            verticalalignment='center')
        else:
            im = None
    return ax, m, im

'''def lagrangian_view(tobj, grids, tmp_dir, uid=None, vmin=-8, vmax=64,
                    cmap=None, alt=None, basemap_res='l', box_rad=25):

    if uid is None:
        print("Please specify 'uid' keyword argument.")
        return
    stepsize = 6
    title_font = 20
    axes_font = 18
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16

    field = tobj.field
    grid_size = tobj.grid_size

    if cmap is None:
        cmap = 'Blues'
    if alt is None:
        alt = tobj.params['GS_ALT']
    cell = tobj.tracks.xs(uid, level='uid')

    nframes = len(cell)
    print('Animating', nframes, 'frames')
    cell_frame = 0

    for nframe, grid in enumerate(grids):
        if nframe not in cell.index:
            continue

        print('Frame:', cell_frame)
        cell_frame += 1

        row = cell.loc[nframe]
        #display = pyart.graph.GridMapDisplay(grid)

        # Box Size
        tx = np.int(np.round(row['grid_x']))
        ty = np.int(np.round(row['grid_y']))
        tx_met = grid.x['data'][tx]
        ty_met = grid.y['data'][ty]
        lat = row['lat']
        lon = row['lon']
        box_rad_met = box_rad * 1000
        box = np.array([-1*box_rad_met, box_rad_met])

        lvxlim = (tx * grid_size[2]) + box
        lvylim = (ty * grid_size[1]) + box
        xlim = (tx_met + box)/1000
        ylim = (ty_met + box)/1000

        fig = plt.figure(figsize=(20, 15))

        fig.suptitle('Cell ' + uid + ' Scan ' + str(nframe), fontsize=22)
        plt.axis('off')

        # Lagrangian View
        ax1 = fig.add_subplot(3, 2, (1, 3))

        display.plot_grid(field, level=get_grid_alt(grid_size, alt),
                          vmin=vmin, vmax=vmax, mask_outside=False,
                          cmap=cmap,
                          ax=ax1, colorbar_flag=False, linewidth=4)

        display.plot_crosshairs(lon=lon, lat=lat,
                                line_style='k--', linewidth=3)

        ax1.set_xlim(lvxlim[0], lvxlim[1])
        ax1.set_ylim(lvylim[0], lvylim[1])

        ax1.set_xticks(np.arange(lvxlim[0], lvxlim[1], (stepsize * 1000)))
        ax1.set_yticks(np.arange(lvylim[0], lvylim[1], (stepsize * 1000)))
        ax1.set_xticklabels(np.round(np.arange(xlim[0], xlim[1], stepsize), 1))
        ax1.set_yticklabels(np.round(np.arange(ylim[0], ylim[1], stepsize), 1))

        ax1.set_title('Top-Down View', fontsize=title_font)
        ax1.set_xlabel('East West Distance From Origin (km)' + '\n',
                       fontsize=axes_font)
        ax1.set_ylabel('North South Distance From Origin (km)',
                       fontsize=axes_font)

        # Latitude Cross Section
        ax2 = fig.add_subplot(3, 2, 2)
        display.plot_latitude_slice(field, lon=lon, lat=lat,
                                    title_flag=False,
                                    colorbar_flag=False, edges=False,
                                    vmin=vmin, vmax=vmax, mask_outside=False,
                                    cmap=cmap,
                                    ax=ax2)

        ax2.set_xlim(xlim[0], xlim[1])
        ax2.set_xticks(np.arange(xlim[0], xlim[1], stepsize))
        ax2.set_xticklabels(np.round((np.arange(xlim[0], xlim[1], stepsize)),
                                     2))

        ax2.set_title('Latitude Cross Section', fontsize=title_font)
        ax2.set_xlabel('East West Distance From Origin (km)' + '\n',
                       fontsize=axes_font)
        ax2.set_ylabel('Distance Above Origin (km)', fontsize=axes_font)
        ax2.set_aspect(aspect=1.3)

        # Longitude Cross Section
        ax3 = fig.add_subplot(3, 2, 4)
        display.plot_longitude_slice('reflectivity', lon=lon, lat=lat,
                                     title_flag=False,
                                     colorbar_flag=False, edges=False,
                                     vmin=vmin, vmax=vmax, mask_outside=False,
                                     cmap=cmap,
                                     ax=ax3)
        ax3.set_xlim(ylim[0], ylim[1])
        ax3.set_xticks(np.arange(ylim[0], ylim[1], stepsize))
        ax3.set_xticklabels(np.round(np.arange(ylim[0], ylim[1], stepsize), 2))

        ax3.set_title('Longitudinal Cross Section', fontsize=title_font)
        ax3.set_xlabel('North South Distance From Origin (km)',
                       fontsize=axes_font)
        ax3.set_ylabel('Distance Above Origin (km)', fontsize=axes_font)
        ax3.set_aspect(aspect=1.3)

        # Time Series Statistic
        max_field = cell['max']
        plttime = cell['time']

        # Plot
        ax4 = fig.add_subplot(3, 2, (5, 6))
        ax4.plot(plttime, max_field, color='b', linewidth=3)
        ax4.axvline(x=plttime[nframe], linewidth=4, color='r')
        ax4.set_title('Time Series', fontsize=title_font)
        ax4.set_xlabel('Time (UTC) \n Lagrangian Viewer Time',
                       fontsize=axes_font)
        ax4.set_ylabel('Maximum ' + field, fontsize=axes_font)

        # plot and save figure
        fig.savefig(tmp_dir + '/frame_' + str(nframe).zfill(3) + '.png')
        plt.close()
        del grid, display
        gc.collect()
'''

def make_mp4_from_frames(tmp_dir, dest_dir, basename, fps, glob='*'):
    cur_dir = os.path.abspath(os.path.curdir)
    os.chdir(tmp_dir)
    os.system(" ffmpeg -framerate " + str(fps)
              + " -pattern_type glob -i '"+glob+".png'"
              + " -movflags faststart -pix_fmt yuv420p -vf"
              + " 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -y "
              + basename)
    try:
        if os.path.isfile(os.path.join(dest_dir, basename)):
            os.remove(os.path.join(dest_dir, basename))
        shutil.move(basename, dest_dir)
    except FileNotFoundError:
        print('Make sure ffmpeg is installed properly.')
    os.chdir(cur_dir)

def animate(tobj, grids, outfile_name, style='full', fps=1, keep_frames=False,
            overwrite=False, **kwargs):
    """
    Creates gif animation of tracked cells.

    Parameters
    ----------
    tobj : Cell_tracks
        The Cell_tracks object to be visualized.
    grids : iterable
        An iterable containing all of the grids used to generate tobj
    outfile_name : str
        The name of the output file to be produced.
    alt : float
        The altitude to be plotted in meters.
    vmin, vmax : float
        Limit values for the colormap.
    arrows : bool
        If True, draws arrow showing corrected shift for each object. Only used
        in 'full' style.
    isolation : bool
        If True, only annotates uids for isolated objects. Only used in 'full'
        style.
    uid : str
        The uid of the object to be viewed from a lagrangian persepective. Only
        used when style is 'lagrangian'.
    fps : int
        Frames per second for output gif.

    """

    styles = {'full': full_domain}
             # 'lagrangian': lagrangian_view}
    anim_func = styles[style]

    dest_dir = os.path.dirname(outfile_name)
    basename = os.path.basename(outfile_name)
    if len(dest_dir) == 0:
        dest_dir = os.getcwd()

    if os.path.exists(os.path.join(outfile_name)):
        if not overwrite:
            print('Filename already exists.')
            return
        else:
            os.remove(outfile_name)

    tmp_dir = tempfile.mkdtemp()

    try:
        anim_func(tobj, grids, tmp_dir, **kwargs)
        if len(os.listdir(tmp_dir)) == 0:
            print('Grid generator is empty.')
            return
        make_mp4_from_frames(tmp_dir, outfile_name, basename, fps)
        if keep_frames:
            frame_dir = os.path.join(dest_dir, basename + '_frames')
            shutil.copytree(tmp_dir, frame_dir)
            os.chdir(dest_dir)
    finally:
        shutil.rmtree(tmp_dir)


def embed_mp4_as_gif(filename):
    """ Makes a temporary gif version of an mp4 using ffmpeg for embedding in
    IPython. Intended for use in Jupyter notebooks. """
    if not os.path.exists(filename):
        print('file does not exist.')
        return

    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    newfile = tempfile.NamedTemporaryFile()
    newname = newfile.name + '.gif'
    if len(dirname) != 0:
        os.chdir(dirname)

    os.system('ffmpeg -i ' + basename + ' ' + newname)

    try:
        with open(newname, 'rb') as f:
            display(Image(f.read()))
    finally:
        os.remove(newname)


def _normalize_kwargs(kwargs, kind='patch'):
    """Convert matplotlib keywords from short to long form."""
    # Source:
    # github.com/tritemio/FRETBursts/blob/fit_experim/fretbursts/burst_plot.py
    if kind == 'line2d':
        long_names = dict(c='color', ls='linestyle', lw='linewidth',
                          mec='markeredgecolor', mew='markeredgewidth',
                          mfc='markerfacecolor', ms='markersize',)
    elif kind == 'patch':
        long_names = dict(c='color', ls='linestyle', lw='linewidth',
                          ec='edgecolor', fc='facecolor',)
    for short_name in long_names:
        if short_name in kwargs:
            kwargs[long_names[short_name]] = kwargs.pop(short_name)
    return kwargs

def _set_labels(ax, label_format, pos_columns):
    """This sets axes labels according to a label format and position column
    names. Applicable to 2D and 3D plotting.
    Parameters
    ----------
    ax : Axes object
        The axes object on which the plot will be called
    label_format : string
        Format that is compatible with ''.format (e.g.: '{} px')
    pos_columns : list of strings
        List of column names in x, y(, z) order.
    Returns
    -------
    None
    """
    ax.set_xlabel(label_format.format(pos_columns[0]))
    ax.set_ylabel(label_format.format(pos_columns[1]))
    if hasattr(ax, 'set_zlabel') and len(pos_columns) > 2:
        ax.set_zlabel(label_format.format(pos_columns[2]))
def invert_yaxis(ax):
    """Inverts the y-axis of an axis object."""
    bottom, top = ax.get_ylim()
    if top > bottom:
        ax.set_ylim(top, bottom, auto=None)
    return ax

