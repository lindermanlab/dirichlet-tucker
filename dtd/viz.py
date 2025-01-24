from typing import Literal, Sequence
import numpy as onp

import matplotlib as mpl
import matplotlib.pyplot as plt


DEFAULT_TICKLABEL_KWARGS = dict(fontsize='x-small')

def set_syllable_cluster_ticks(
    cluster_names: Sequence,
    label: bool=True,
    grid: bool=True,
    ticklabel_kwargs: dict|None=None,
    grid_kwargs: dict|None=None,
    axis: Literal['x','y']='x', 
    ax: mpl.axes.Axes=None,
):
    """Annotate syllable axis with feature similarity cluster names.
    
    Parameters
        cluster_names (sequence): Syllable cluster name, length (n_syllables,)
        label (bool): If True, label cluster groups. Else, just draw boundaries.
        grid (bool): If True, draw lines across image indicate cluster boundaries,.
        ticklabel_kwargs (dict, optional): Text keyword arguments for tick labels.
        grid_kwargs (dict, optional): Line keyword arguments for cluster boundaries.
        axis (literal): Axis to add ticks to. Default: 'x'
        ax (mpl.axes.Axes | None)
    """

    if ax is None:
        ax = plt.gca()

    if ticklabel_kwargs is None:
        ticklabel_kwargs = dict()
    ticklabel_kwargs = DEFAULT_TICKLABEL_KWARGS | ticklabel_kwargs

    if grid_kwargs is None:
        grid_kwargs = dict(color='0.6', ls='--', alpha=0.5)

    # Get unique instances of cluster names. Note that these are sorted
    # alphanumerically. Each element length (n_unique,)
    unique_names, indices, counts = \
        onp.unique(cluster_names, return_index=True, return_counts=True)

    # Sort results into the the order that they were seen
    indices = onp.argsort(indices)
    unique_names = [str(unique_names[i]) for i in indices]
    counts = onp.array([counts[i] for i in indices])

    # Calculate location of cluster edges and centers
    cluster_edges = onp.cumsum(counts)
    cluster_centers = onp.cumsum(counts) - 0.5 * counts

    # Label axis
    if axis == 'x':
        # Break long cluster names into two lines
        labels = [lbl.replace(" ", "\n") for lbl in unique_names]

        # Set boundary ticks; extend tick length
        ax.set_xticks(cluster_edges[:-1]-0.5, minor=True)
        ax.tick_params(axis='x', which='minor', length=18)

        # Set labels
        if label:
            ax.set_xticks(cluster_centers)
            ax.set_xticklabels(labels, ha='center', va='center', **ticklabel_kwargs)
            
            # Hide label tick, add padding from axis
            ax.tick_params(axis='x', which='major', length=0, pad=10)
        else:
            ax.tick_params(axis='x', which='major', bottom=False, labelbottom=False)
        
        # Add grid lines
        if grid:
            ax.grid(axis='x', which='minor', **grid_kwargs)
    
    elif axis == 'y':
        # Keep cluster names in a single line
        labels = unique_names

        # Set boundary ticks; extend tick length
        ax.set_yticks(cluster_edges[:-1]-0.5, minor=True)

        # Set labels
        if label:
            ax.tick_params(axis='y', which='minor', length=18)  # Set long cluster dividers

            ax.set_yticks(cluster_centers)
            ax.set_yticklabels(labels, ha='right', va='center', **ticklabel_kwargs)
            
            # Hide label tick, add padding from axis
            ax.tick_params(axis='y', which='major', length=0, pad=5)
        else:
            ax.tick_params(axis='y', which='minor', length=9)  # Set short cluster tick dividers
            
            ax.tick_params(axis='y', which='major', left=False, labelleft=False)
        
        if grid:
            ax.grid(axis='y', which='minor', **grid_kwargs)

    else:
        print(f"WARNING: axis {axis} not recognized. Expected one of 'x' or 'y'.")
    
    return


def set_time_within_session_ticks(
    n_bins: int,
    frames_per_bin: int,
    frames_per_sec: float,
    tick_period: float,
    tick_units: str='min',
    label: bool=True,
    include_end: bool=True,
    ticklabel_kwargs: dict|None=None,
    tick_format: str="d",
    axis: Literal['x','y']='x',
    ax: mpl.axes.Axes=None
):
    """Annotate time-within-session axis with periodic ticks and tick labels.
    
    Parameters
        n_bins (int): Total duration of a session, in units of bins.
        frames_per_bin (int): aka bin size.
        frames_per_sec (float). Frames per second.
        tick_period (float): Time between ticks, in units of `units`.
        tick_units (literal): Units of seconds, minutes, or hours. Default: minutes.
            - seconds: One of {'s', 'sec', 'second', 'seconds'}
            - minutes: One of {'m', 'min', 'minute', 'minutes'}
            - hours: One of {'h', 'hr', 'hrs', 'hour', 'hours'}
        label (bool): If True, label ticks. Else, set ticks but do not label.
        include_end (bool): If True, include the last tick indicating session length.
        ticklabel_kwargs (dict, optional): Text keyword arguments for tick labels.
        tick_format (str): f-string format
        axis (literal): Axis to add ticks to. Default: 'x'
        ax (mpl.axes.Axes): Axis object to plot on. If None, get current axis.        
    """

    if ax is None:
        ax = plt.gca()

    if ticklabel_kwargs is None:
        ticklabel_kwargs = dict()
    ticklabel_kwargs = DEFAULT_TICKLABEL_KWARGS | ticklabel_kwargs

    # ------------------------------------------------------------------------
    if tick_units.lower() in ('s', 'sec', 'second', 'seconds'):
        sec_per_unit = 1.
    elif tick_units.lower() in ('m', 'min', 'minute', 'minutes'):
        sec_per_unit = 60
    elif tick_units.lower() in ('h', 'hr', 'hrs', 'hour', 'hours'):
        sec_per_unit = 60 * 60

    # Compute the tick_labels in units specified by `tick_units`
    length = n_bins * frames_per_bin / frames_per_sec  / sec_per_unit
    tick_labels = onp.arange(0, length, tick_period)
    
    # If include_end, include the total session length to tick labels
    # regardless of the period.
    if include_end and tick_labels[-1] != length:
        tick_labels = onp.concatenate([tick_labels, [length]])

    # ------------------------------------------------------------------------
    # Compute tick locations, units of bins
    ticks = tick_labels * sec_per_unit * frames_per_sec / frames_per_bin - 0.5

    # ------------------------------------------------------------------------
    # Format tick labels
    if 'd' in tick_format:
        tick_labels = [f"{int(lbl):{tick_format}}" for lbl in tick_labels]
    else:
        tick_labels = [f"{lbl:{tick_format}}" for lbl in tick_labels]

    # ------------------------------------------------------------------------
    # Add ticks and tick labels
    if axis == 'x':
        ax.set_xticks(ticks)
        if label:
            ax.set_xticklabels(tick_labels, **ticklabel_kwargs)

    elif axis == 'y':
        ax.set_yticks(ticks)
        if label:
            ax.set_yticklabels(tick_labels, **ticklabel_kwargs)

    else:
        print(f"WARNING: axis {axis} not recognized. Expected one of 'x' or 'y'.")
    
    return


def draw_drug_class_boxes(
    drug_class: Sequence,
    bbox_x: float=-0.1,
    bbox_w: float=1.2,
    label: bool=True,
    label_kwargs: dict|None=None,
    axis: Literal['y']='y', 
    ax: mpl.axes.Axes=None,
):
    """Annotate session axis with drug class boxes.

    Sessions must be on y-axis. 

    Used for Wiltschko et al. 2022 drug dataset.

    Parameters
        drug_class (sequence): Drug class names, length (n_sessions,)
        bbox_x (flaot): Drug class bounding box left coord, in units of axis fraction.
        bbox_w (float): Drug class bounding box width, in units of axis fraction.
        label (bool): If True, label classes. Else, just draw bounding boxes.
        label_kwargs (dict, optional): Text keyword arguments for tick labels.
        axis (literal): Axis to add ticks to. Default: 'x'
        ax (mpl.axes.Axes | None)

    """

    if ax is None:
        ax = polt.gca()

    if label_kwargs is None:
        label_kwargs = dict(
            va='center', ha='right', rotation=0,
        )

    # -----------------------------------------------------------------------------------
    # Get unique instances of drug class names. Note that these results are sorted
    # alphanumerically. Each element is length (n_unique,)
    unique_names, indices, counts = \
        onp.unique(drug_class, return_index=True, return_counts=True)

    # Ensure that class labels and counts are in the original order
    indices = onp.argsort(indices)
    unique_names = [unique_names[i] for i in indices]
    counts = onp.array([counts[i] for i in indices])

    # Calculate location of cluster edges and centers
    edges = onp.cumsum(counts)
    centers = onp.cumsum(counts) - counts/2

    # -----------------------------------------------------------------------------------
    for i, color in enumerate(mpl.colormaps['Set2'].colors[:len(unique_names)]):
        # Draw bounding box
        y0 = edges[i-1] if i > 0 else 0
        y1 = edges[i]
        
        rect = mpl.patches.Rectangle(
            (bbox_x, y0+0.1), bbox_w, y1-y0-0.2,
            transform=ax.get_yaxis_transform(),
            lw=2, ec=color, fc='none', clip_on=False, zorder=20)
        ax.add_patch(rect)

        if label:
            ax.text(
                bbox_x-0.1, centers[i], unique_names[i],
                transform=ax.get_yaxis_transform(), clip_on=False,
                bbox=dict(facecolor=color, boxstyle='square,pad=0.2'),
                **label_kwargs
            )

    return