import pytest

import numpy as onp

from dtd.data import load_wiltschko22_data

# Data directory containing subdirectories "killifish/", "moseq-drugs/"
DATA_DIR = "/home/groups/swl1/eyz/data/"

def test_load_wiltschko22_data():
    filepath = DATA_DIR + "moseq-drugs/syllable_binned_1min.npz"
    counts, batch_axes, event_axes, metadata = load_wiltschko22_data(filepath)

    n_sessions, n_bins, n_syllables = 500, 20, 90
    
    assert counts.shape == (n_sessions, n_bins, n_syllables)
    
    # TODO: This assertion can be relaxed in the future, since we do not strictly
    # need all bins to have the exact same number of counts.
    assert onp.all(counts.sum(event_axes) == metadata["frames_per_bin"])