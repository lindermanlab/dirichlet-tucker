from pathlib import Path

import jax.numpy as jnp
import numpy as onp

def load_wiltschko22_data(filepath: Path | str):
    """Load binned syllable data from Wiltschko et al. 2022 paper.
    
    This data consists of behavioral syllables during a 20 minute
    open-field assay (OFA) after drug administration.

    Paramters
        filepath (Path | str): Path to .npz file containing binned syllables

    Returns
        counts (ndarray): shape (n_sessions, n_bins, n_syllables), uint dtype
            Binned syllable counts, where n_sessions = 500, n_syllables = 90.
        batch_axes (tuple[int]):
            Axes or modes of the count tensor that are treated as independent samples.
            Analogous to the use of `batch_shape` or `batch_dims` in probabilistic machine
            learning packages.
        event_axes (tuple[int]):
            Axes (or modes) of the count tensor that represent a single draw from the
            topic model. Analogous to the use of `event_shape` or `event_dims` in
            probabilistic machine learning packages.
        metadata (dict): 
            - session (dict): Consists of entries of length (n_sessions,)
                - drug_names: Name of administered drug, 4-letter abbreviation
                - drug_class: Name of drug class of administered drug
                - drug_doses: Name of drug dosage, on a scale of [0,6) from
                "Extremely Low" to "Extremely High"
            - syllable (dict): Consists of entries of length (n_syllables,)
                - cluster_names: Name of syllable cluster name. Clusters are based
                  on syllable feature cosine similarity, which differs from the
                  "functional" similarity / co-occurence of syllables highly used
                  in the same behavioral topics 
            - frames_per_bin (int): Aka `bin_size`
            - frames_per_sec (float): Recording frame rate
    """

    batch_axes = (0,1)
    event_axes = (2,)

    metadata = dict()
    with onp.load(filepath) as f:
        counts = f['session_syllable_counts']
        
        metadata['session'] = dict(
            drug_names=f['session_drug_name'],
            drug_class=f['session_drug_class'],
            drug_doses=f['session_drug_dose_0to6'],
        )

        metadata['frames_per_bin'] = f['meta_bin_size']
        metadata['frames_per_sec'] = f['meta_fps']

        metadata['syllable'] = dict(
            cluster_names=f['syllable_cluster']
        )
    
    return counts, batch_axes, event_axes, metadata

