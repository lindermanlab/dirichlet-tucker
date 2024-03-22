import bson
import click
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb

import itertools as it
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix
from tensorflow_probability.substrates import jax as tfp

from dtd.model4d import DirichletTuckerDecomp

tfd = tfp.distributions

def load_data(data_dir):
    # Load te drug ids ((M,) array of ints)
    y = np.load(os.path.join(data_dir, "drug_ids.npy"))

    # read in behav data
    with open(os.path.join(data_dir, "behav_data.bson"), 'rb') as f:
        data = bson.decode_all(f.read())
    
    X = np.reshape(np.frombuffer(data[0]['Xb']['data'], dtype=np.float64), data[0]['Xb']['size'], order='F').astype(np.float32)
    
    # Behavior tensor X is mice x syllables x positions x epochs
    # Permute to mice x epochs x positions x syllables
    X = np.transpose(X, (0, 3, 2, 1))
    
    # Permute the syllables
    perm = np.array([2, 45, 32, 33, 20, 23, 31, 47, 6, 25, 10, 17, 42, 12, 30, 36, 28, 34, 49, 13, 44, 7, 16, 15, 29, 48, 37, 38, 8, 24, 39, 43, 1, 26, 11, 19, 9, 40, 27, 21, 3, 18, 35, 4, 14, 5, 41, 22, 46, 50]) - 1
    X = X[..., perm]
    return X, y


def make_mask(X, key=0, train_frac=0.8):
    """## Split the data into train and test"""
    M, N, P, S = X.shape
    key = jr.PRNGKey(key)
    mask = tfd.Bernoulli(probs=train_frac).sample(seed=key, sample_shape = (M, N)).astype(bool)
    return mask


def fit_model(key, X, mask, K_M, K_N, K_P, K_S, alpha, num_iters, tol):
    M, N, P, S = X.shape
    C = X[0,0].sum()

    # Construct a model
    model = DirichletTuckerDecomp(C, K_M, K_N, K_P, K_S, alpha=alpha)
    # Initialize the parameters randomly
    print("initializing model")
    init_params = model.sample_params(key, M, N, P, S)
    print("done")

    # Fit the model with EM
    params, lps = model.fit(X, mask, init_params, num_iters, tol)

    # Compute test LL
    print("computing pct_dev")
    test_ll = model.heldout_log_likelihood(X, mask, params)

    # Make a baseline of average syllable usage in each epoch
    # NOTE: ignores the multinomial normalizing constant, like `heldout_log_likelihood`
    baseline_probs = jnp.mean(X[mask], axis=0) + alpha
    baseline_probs /= baseline_probs.sum()
    baseline_test_ll = jnp.sum(X[~mask] * jnp.log(baseline_probs))

    # Compute the test log likelihood under the saturated model
    sat_probs = X[~mask] / X[~mask].sum(axis=(-1, -2), keepdims=True)
    sat_test_ll = jnp.nansum(X[~mask] * jnp.log(sat_probs))

    pct_dev = (test_ll - baseline_test_ll) / (sat_test_ll - baseline_test_ll)
    print("done")
    return model, params, lps, pct_dev


def plot_results(X, y, model, params):
    # Compute the reconstruction
    X_hat = model.reconstruct(params)

    # Plot the true and reconstructed data for mouse `m`
    m = 0
    n = 0
    fig, axs = plt.subplots(3, 1)
    im = axs[0].imshow(X[m, n], aspect="auto", interpolation="none")
    plt.colorbar(im)
    im = axs[1].imshow(X_hat[m, n], aspect="auto", interpolation="none")
    plt.colorbar(im)
    im = axs[2].imshow(X[m, n] - X_hat[m, n], aspect="auto", interpolation="none")
    plt.colorbar(im)

    """### Look at the factors"""
    G, Psi, Phi, Theta, Lambda = params

    # Plot the position topics
    plt.figure()
    plt.imshow(Theta, aspect="auto", interpolation="none")
    plt.xlabel("position")
    plt.ylabel("topics")
    plt.colorbar()

    # Plot the syllable topics
    syll_groups = np.array([2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]) -1
    group_bounds = np.cumsum(np.bincount(syll_groups)[:-1])

    plt.figure()
    plt.imshow(Lambda, aspect="auto", interpolation="none")
    for bound in group_bounds:
        plt.axvline(bound-.5, color='r')
    plt.xlabel("syllables")
    plt.ylabel("topics")
    plt.colorbar()

    # Plot the epoch loadings
    plt.figure()
    plt.imshow(Phi.T, aspect="auto", interpolation="none")
    plt.xlabel("epochs")
    plt.ylabel("loadings")
    plt.colorbar()

    # Plot the mouse loadings
    plt.figure()
    perm = jnp.argsort(y)
    bounds = jnp.cumsum(jnp.bincount(y)[1:-1])
    plt.imshow(Psi[perm], aspect="auto", interpolation="none")
    for bound in bounds:
        plt.axhline(bound, color='r')
    plt.xlabel("loadings")
    plt.ylabel("mice")
    plt.colorbar()


def evaluate_prediction(Psi, y):
    """### Classify drug labels"""

    # normalize weights and factors
    def normalize_weights(features):
        features -= features.mean(axis=0)
        features /= features.std(axis=0)
        return features

    parameters = {"C":10 ** np.linspace(-15,15,num=31)}
    lr = LogisticRegression()
    gridsearch = GridSearchCV(lr, parameters)
    gridsearch.fit(normalize_weights(Psi), y)

    acc = gridsearch.best_score_
    classifier = gridsearch.best_estimator_
    y_pred = cross_val_predict(classifier, normalize_weights(Psi), y=y)
    confusion_mat = confusion_matrix(y, y_pred)

    return acc, confusion_mat

def run_one(data_dir, seed, km, kn, kp, ks, alpha, num_iters, train_frac=0.8, tol=1e-4):
    key = jr.PRNGKey(seed)

    # Split the data deterministically
    X, y = load_data(data_dir)
    mask = make_mask(X, key=0, train_frac=train_frac)

    # Fit the model using the random seed provided
    model, params, lps, pct_dev = fit_model(key, X, mask, km, kn, kp, ks, alpha, num_iters, tol)
    
    G, Psi, Phi, Theta, Lambda = params
    acc, confusion_matrix = evaluate_prediction(Psi, y)

    # Plot some results
    plot_results(X, y, model, params)

    # Finish the session
    print("pct_dev:", pct_dev)
    print("acc:", acc)

    return model, params, lps, acc, confusion_matrix


@click.command()
@click.option('--data_dir', default="/home/groups/swl1/swl1", help='path to folder where data is stored.')
@click.option('--seed', default=0, help='random seed for initialization.')
@click.option('--km_min', default=2, help='number of factors along dimension 1.')
@click.option('--km_max', default=20, help='number of factors along dimension 1.')
@click.option('--kn_min', default=2, help='number of factors along dimension 2.')
@click.option('--kn_max', default=10, help='number of factors along dimension 2.')
@click.option('--kp_min', default=2, help='number of factors along dimension 3.')
@click.option('--kp_max', default=20, help='number of factors along dimension 3.')
@click.option('--ks_min', default=2, help='number of factors along dimension 4.')
@click.option('--ks_max', default=20, help='number of factors along dimension 4.')
@click.option('--k_step', default=1, help='step size for factor grid search.')
@click.option('--num_restarts', default=1, help='number of random initializations.')
@click.option('--alpha', default=1.1, help='concentration of Dirichlet prior.')
@click.option('--num_iters', default=2000, help='max number of iterations of EM')
@click.option('--tol', default=1e-4, help='tolerance for EM convergence')
def run_sweep(data_dir, seed, km_min, km_max, kn_min, kn_max, kp_min, kp_max, ks_min, ks_max, k_step, num_restarts, alpha, num_iters, tol):

    # Split the data deterministically
    X, y = load_data(data_dir)
    mask = make_mask(X, key=0)

    # Get the set of existing runs
    print("getting list of existing runs")
    api = wandb.Api(timeout=60)
    runs = api.runs("linderman-lab/serotonin-tucker-decomp-4d")
    configs = []
    for run in runs:
        configs.append((run.config['K_M'], run.config['K_N'], run.config['K_P'], run.config['K_S']))
    print("done")

    for km, kn, kp, ks in it.product(range(km_min, km_max+1, k_step),
                                     range(kn_min, kn_max+1, k_step),
                                     range(kp_min, kp_max+1, k_step),
                                     range(ks_min, ks_max+1, k_step),
                                     ):
        if (km, kn, kp, ks) in configs:
            print("found existing run with km={}, kn={}, kp={} ks={}. continuing.".format(km, kn, kp, ks))
            continue

        print("fitting model with km={}, kn={}, kp={} ks={}".format(km, kn, kp, ks))

        # start a new wandb run to track this script
        print("initializing wandb")
        wandb.init(
            # set the wandb project where this run will be logged
            project="serotonin-tucker-decomp-4d",

            # track hyperparameters and run metadata
            config={
                "K_M": km,
                "K_N": kn,
                "K_P": kp,
                "K_S": ks,
                "alpha": alpha,
                "num_restarts": num_restarts,
                "seed": seed,
                "num_iters": num_iters,
                "tol" : tol
                }
        )
        print("done")

        # Fit the model using the random seed provided
        pct_devs = []
        accs = []
        for i in range(num_restarts):
            key = jr.PRNGKey(seed)
            model, params, lps, pct_dev = fit_model(key, X, mask, km, kn, kp, ks, alpha, num_iters, tol)

            # Compute drug prediction accuracy
            G, Psi, Phi, Theta, Lambda = params
            acc, _ = evaluate_prediction(Psi, y)
            
            # Keep track of perf measures
            pct_devs.append(pct_dev)
            accs.append(acc)
            seed += 1

            # Cleanup
            del model
            del params
            del lps

        # Finish the session
        wandb.run.summary["pct_dev_mean"] = jnp.array(pct_devs).mean()
        wandb.run.summary["pct_dev_std"] = jnp.array(pct_devs).std()
        wandb.run.summary["acc_mean"] = jnp.array(accs).mean()
        wandb.run.summary["acc_std"] = jnp.array(accs).std()
        wandb.finish()


if __name__ == '__main__':
    run_sweep()
