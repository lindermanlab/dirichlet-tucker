"""Calculate classification performance vs. model type and dimensionality.

Example usage:
    python run.py \
        --file_path /path/to/binned/syllables.npz --output_dir /path/to/output/dir \
        --seed 0 --target_name drug --model_name dtd --model_dim 48
"""

from typing import Literal, Optional
from numpy.typing import ArrayLike

import argparse
from dotenv import find_dotenv, load_dotenv
import itertools
from math import prod
import os
from pathlib import Path
import time
from tqdm import tqdm
import wandb

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from dtd.data import load_wiltschko22_data
from dtd.model3d import DirichletTuckerDecomp
from dtd.utils import (
    create_block_speckled_mask,
    get_jax_rng_state,
    get_numpy_rng_state,
)

ArrayLike = ArrayLike | jax.Array
KeyArray = jax.Array  # Specific to jax PRNG KeyArray
PathLike = Path | str

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

DATA_DIR = Path(os.environ['DATA_DIR'])
OUTPUT_DIR = Path(os.environ['OUTPUT_DIR'])

# Optimal (k2, k3) parameters for given k1
# Identified via reconstruction loss cross validation, see: 
#   ../crossval_reconstruction/Analyze_cross_validation_results.ipynb
# This is called in `fit_transform_model`
DTD_TOPIC_RANK = {
      2: {'k2': 6, 'k3':  4},
      4: {'k2': 6, 'k3':  8},
      8: {'k2': 8, 'k3': 16},
     16: {'k2': 8, 'k3': 24},
     24: {'k2': 6, 'k3': 24},
     32: {'k2': 6, 'k3': 24},
     48: {'k2': 4, 'k3': 36},
     64: {'k2': 4, 'k3': 36},
     96: {'k2': 4, 'k3': 48},
    128: {'k2': 2, 'k3': 36},
    256: {'k2': 2, 'k3': 24},
}

# =======================================================================================

def get_labels(session_metadata: dict, target_name: Literal['drug', 'dose', 'class']):
    """Get specified target_name labels.

    Parameters
    ----------
    session_metadata (dict): Contains keys {'drug_names', 'drug_doses', 'drug_class'}
        and pointing to string arrays of length (n_sessions,)
    target_name (str): One of
        - 'drug': Drug name (n=16)
        - 'dose': Drug name x dosage (0-6 scale) combination (n=45)
        - 'drug_class': Drug class (n=7)
    
    Returns
    -------
    y (onp.ndarray): shape (n_sessions,)
    label_encoder (sklearn.preprocessing._label.LabelEncoder)
    label_binarizer (sklearn.preprocessing._label.LabelBinarizer)
        This will be used for calculating precision and recall by class
    """

    if target_name == 'drug':
        labels = session_metadata['drug_names']
    elif target_name == 'dose':
        drug_labels = session_metadata['drug_names']
        dose_labels = session_metadata['drug_doses']
        labels = [
            f'{drug}-{dose}' for drug, dose in zip(drug_labels, dose_labels)
        ]
    elif target_name == 'class':
        labels = session_metadata['drug_class']

    # Ensure labels are binarized in the order that they were see
    unique_labels, indices = onp.unique(labels, return_index=True)
    unique_labels = [str(unique_labels[i]) for i in onp.argsort(indices)]    

    label_encoder = LabelEncoder().fit(unique_labels)
    label_encoder.classes_ = onp.asarray(unique_labels)  # Preserve order

    label_binarizer = LabelBinarizer().fit(unique_labels)
    label_binarizer.classes_ = onp.asarray(unique_labels)  # Preserve order

    return labels, label_encoder, label_binarizer

# =======================================================================================

def _fit_and_transform_sum(data: ArrayLike, model_dim: int, **kwargs):
    """Empirical frequency of syllable usage per session; Wiltschko et al. 2020."""

    print(f"\nFitting: sum ({model_dim=})")
    freq = data.sum(axis=1) / (data.sum(axis=1)).sum(axis=-1, keepdims=True)

    return freq, dict()


def _fit_and_transform_lda(
    data: ArrayLike, model_dim: int, *, seed:int, n_inits: int=10, **kwargs
):
    """Latent Dirichlet Allocation (vanilla topic model)."""
    
    # Matricize data, shape (n_sessions, n_timebins * n_syllables)
    n_sessions = len(data)
    data_mat = data.reshape(n_sessions, -1)

    session_weights = onp.zeros((n_sessions, model_dim))
    best_score = -onp.inf
    for _ in range(n_inits):
        lda = LatentDirichletAllocation(n_components=model_dim, random_state=seed)
        lda.fit(data_mat)

        score = lda.score(data_mat)
        if score > best_score:
            best_score = score
            session_weights = lda.transform(data_mat)

    print(f"\nFitting: lda ({model_dim=}, {n_inits=})")
    freq = data.sum(axis=1) / (data.sum(axis=1)).sum(axis=-1, keepdims=True)

    aux = dict(
        topics=lda.components_ / lda.components_.sum(axis=1,keepdims=True)
    )

    return session_weights, aux


def _fit_and_transform_dtd(
    data: ArrayLike, model_dim: int, *, k2: int, k3: int, event_ndims: int,
    key: int|KeyArray, n_inits: int=10, n_epochs: int=500, **kwargs,
) -> onp.ndarray:
    """
    Model parameters
    ----------------
    k2, k3 (int).
    event_ndims (int)
    n_inits (int). Number of random initializations.
    n_epochs (int). Number of epochs / iterations to run fit.
    key (int|KeyArray). PRNG for randomly initializing random parameters.
        If int, interpreted as a seed.
    """

    if isinstance(key, int):
        key = jr.key(key)

    d1, d2, d3 = data.shape
    k1 = model_dim

    # Get total counts assuming that all bins sum to the same constant number.
    event_axes = tuple([i for i in range(data.ndim)[-event_ndims:]])
    total_counts = int((data.sum(event_axes).ravel())[0])

    # Fit (multiple instances of) model
    data = jnp.asarray(data, dtype=float)
    mask = jnp.ones(data.shape[:-event_ndims], dtype=bool)  # Fit on all data
    model = DirichletTuckerDecomp(total_counts, k1, k2, k3, alpha=1.1)

    def _fit(carry, _key):
        best_params, best_lp = carry

        # Initialize
        init_params = model.sample_params(_key, d1, d2, d3, conc=0.5)

        # Fit model to held-in data
        params, lps = model.fit(data, mask, init_params, n_epochs)

        carry = jax.lax.cond(
            lps[-1] > best_lp, lambda: (params, lps[-1]), lambda: (best_params, best_lp),
        )
        return carry, None

    print(f"\nFitting: DTD ({model_dim=}, {k2=}, {k3=}, {n_inits=})")
    init_carry = (model.sample_params(key, d1, d2, d3, conc=0.5), -jnp.inf,)
    (params, lp), _ = jax.lax.scan(_fit, init_carry, jr.split(key, n_inits))   

    G, F1, F2, F3 = params
    aux = dict(G=onp.asarray(G), F2=onp.asarray(F2), F3=onp.asarray(F3), lp=float(lp))
    return onp.asarray(F1), aux


def fit_and_transform_model(
    data: ArrayLike,
    model_name: Literal['dtd', 'lda', 'tca'],
    model_dim: int,
    model_params: dict,
) -> onp.ndarray:
    """Fit model to data and return featurizations.
    
    Parameters
    ----------
    data (ArrayLike): shape (n_sessions, n_timebins, n_syllables).
    model_name (str): Model to fit
    model_dim (int): Model size
    model_params (dict): Additional model parameters

    Returns
    -------
    features (onp.ndarray): shape (n_sessions, model_dim)
    aux (dict): Auxiliary model outputs, to save in pickle file.
    """

    if model_name == 'sum':
        features, aux = _fit_and_transform_sum(data, model_dim, **model_params)
    elif model_name == 'lda':
        features, aux = _fit_and_transform_lda(data, model_dim, **model_params)
    elif model_name == 'dtd':
        features, aux = _fit_and_transform_dtd(data, model_dim, **model_params)
    else:
        raise NotImplementedError(f"Function not implemented for {model_name=}")
        
    return features, aux


# =======================================================================================


def train_and_eval_one(X_train, y_train, X_test, y_test, C=100.0):
    """Train and evaluate a logistic regression classifier.
    
    Ported from "Load and Analyze Behavioral Fingerprints.ipynb" notebook in
        https://github.com/dattalab/moseq-drugs/blob/master/notebooks/Load%20and%20Analyze%20Behavioral%20Fingerprints.ipynb
    
    Parameters
    ----------
    X_train (onp.ndarray): shape (n_train, feature_dim)
    y_train (onp.ndarray): shape (n_train,)
    X_test (onp.ndarray): shape (n_test, feature_dim)
    y_test (onp.ndarray): shape (n_test,)
    C (float): Inverse of regularization strength. Smaller values specify stronger regularization.
                C=100 seems to work the best for across the board, but still worth varying.
    
    Returns
    -------
    true_y_test (onp.ndarray): shape (n_test,). Pass through of `y_test`.
    pred_y_test (onp.ndarray): shape (n_test,). Classifier prediction, values in [0, n_labels).
    scores (onp.ndarray): shape (n_test, n_labels). Classifier predictions scores, for each label.
    """

    model = OneVsRestClassifier(
        LogisticRegression(penalty='l2', C=C, solver='lbfgs', class_weight='balanced')
    )
    model.fit(X_train, y_train)

    true_y_test = y_test
    pred_y_test = model.predict(X_test)
    scores = model.predict_proba(X_test)

    return true_y_test, pred_y_test, scores


def main(
    file_path: PathLike,
    output_dir: PathLike,
    target_name: str,
    model_name: str,
    model_dim: int,
    model_params: dict,
    *,
    seed: int,
    n_splits: int=500,
    test_frac: float=0.1,
    use_wandb: bool=False,
):
    """Train and evaluate a given model on logistic classification over multiple splits.

    Loads data, constructs masks, then calls `train_and_eval_one` for each split.

    Parameters
    ----------
    
    """

    if use_wandb:
        wnb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", None),
            config=dict(
                target_name=target_name,
                model_name=model_name,
                model_dim=model_dim,
                model_params=model_params,
                seed=seed,
                n_splits=n_splits,
                test_frac=test_frac,
            ),
        )

    # ===================================================================================
    # Load data, fit model, and get labels
    # ===================================================================================
    data, _, event_axes, metadata = load_wiltschko22_data(file_path)
    event_ndims = len(event_axes)
    print(f"\ndata: shape={data.shape}, dtype={data.dtype}, event_ndims={event_ndims}")

    X, model_aux = fit_and_transform_model(data, model_name, model_dim, model_params)
    y_labels, label_encoder, label_binarizer = get_labels(metadata['session'], target_name)
    y = label_encoder.transform(y_labels)

    n_labels = len(label_encoder.classes_)

    # ===================================================================================
    # Train, evaluate, and log
    # ===================================================================================
    splits = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_frac, random_state=seed
    ).split(X, y)

    n_test = len(next(splits)[-1])  # split yields (train_idxs, test_idxs)
    
    all_y_true = onp.zeros((n_splits*n_test,), dtype=int)
    all_y_pred = onp.zeros((n_splits*n_test,), dtype=int)
    all_scores = onp.zeros((n_splits*n_test, n_labels))
    # all_cms = onp.zeros((n_splits, n_labels, n_labels))
    # all_f1s = onp.zeros((n_splits, n_labels))
    print(f'Classifying: {target_name} ({n_splits=}, {test_frac=})')
    for i_split, (train_idxs, test_idxs) in enumerate(tqdm(splits, total=n_splits)):
        y_true, y_pred, scores = train_and_eval_one(
            X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs],
        )

        slc = slice(i_split*n_test, (i_split+1)*n_test)
        all_y_true[slc] = y_true
        all_y_pred[slc] = y_pred
        all_scores[slc] = scores

    # ---------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------
    target_output_dir = Path(output_dir)/target_name
    target_output_dir.mkdir(parents=True, exist_ok=True)
    onp.savez_compressed(target_output_dir/f"{model_name}-{model_dim}-{seed}.npz",
                         target_name=target_name,
                         model_name=model_name,
                         model_dim=model_dim,
                         model_params=model_params,
                         n_splits=n_splits,
                         test_frac=test_frac,
                         seed=seed,
                         label_classes=label_encoder.classes_,
                         X=X,
                         model_aux=model_aux,
                         all_y_true=all_y_true,
                         all_y_pred=all_y_pred,
                         all_scores=all_scores,
    )

    if use_wandb:
        wnb_run.summary['avg_f1'] = f1_score(all_y_true, all_y_pred, average="micro")
        wnb_run.finish()

    return

# To be used in conjunction with a WandB Sweep file (.yaml)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str,
                        help='Path to file containining binned syllable data')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save results files to.')
    parser.add_argument('--seed', type=int,
                        help='Seed for reproducibly splitting data.')
    parser.add_argument('--target_name', choices=['drug', 'dose', 'class',],)
    parser.add_argument('--model_name', choices=['dtd', 'lda', 'tca', 'sum'],)
    parser.add_argument('--model_dim', type=int,
                        help="Model dimensionality. For 'sum', model_dim fixed to 90.")
    parser.add_argument('--n_splits', type=int, default=500)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--use_wandb', action='store_true',
                        help="Log outputs to WandB. If used, assumes WANDB_ENTITY and WANDB_PROJECT environment variables are set.")

    args: dict = vars(parser.parse_args())

    # ===================================================================================
    # Process arguments
    # ===================================================================================
    # Legacy numpy RandomState (used by scikit-learn) expects seed between 0 and 2**32-1
    model_seed = int(time.time()) % (2**32-1)
    
    # Configure model parameters
    if args['model_name'] == 'sum':
        args['model_dim'] = 90  # Fixed, this model has no other dimensionality
        model_params = dict()

    elif args['model_name'] == 'lda':
        model_params = dict(
            seed=model_seed,
            n_inits=10,
        )

    elif args['model_name'] == 'dtd':
        model_params = dict(
            **DTD_TOPIC_RANK[args['model_dim']],  # Use optimal k2, k3 dims, defined at top of file
            key=jr.key(model_seed),
            event_ndims=1,  # Hard-coded in, given the data
            n_epochs=500,
            n_inits=10,
        )
    
    # ===================================================================================
    # Run
    # ===================================================================================
    main(**args, model_params=model_params)