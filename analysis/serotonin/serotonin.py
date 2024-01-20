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
from tqdm.auto import trange

from dtd.model3d import DirichletTuckerDecomp

tfd = tfp.distributions

def load_data(data_dir):
    # Load te drug ids ((M,) array of ints)
    y = np.load(os.path.join(data_dir, "drug_ids.npy"))

    # read in behav data
    with open(os.path.join(data_dir, "behav_data.bson"), 'rb') as f:
        data = bson.decode_all(f.read())
    # Xb = np.reshape(np.frombuffer(data[0]['Xb']['data'], dtype=np.float64), data[0]['Xb']['size'], order='F')
    Xb = np.reshape(np.frombuffer(data[0]['Xb']['data'], dtype=np.float64), data[0]['Xb']['size'], order='F').astype(np.float32)
    # Behavior tensor Xb is mice x syllables x positions x epochs

    # Option 1: Marginalize over positions
    X = Xb.sum(axis=2)
    X = np.transpose(X, (0, 2, 1))

    # Option 2: Combine syllables and positions
    # X = np.transpose(Xb, (0, 3, 1, 2))
    # X = X.reshape(X.shape[0], X.shape[1], -1)
    return X, y


def make_mask(X, key=0, train_frac=0.8):
    """## Split the data into train and test"""
    M, N, P = X.shape
    key = jr.PRNGKey(key)
    mask = tfd.Bernoulli(probs=train_frac).sample(seed=key, sample_shape = (M, N)).astype(bool)
    return mask


def fit_model(key, X, mask, K_M, K_N, K_P, alpha=1.1):
    M, N, P = X.shape
    S = X[0,0].sum()

    # Construct a model
    model = DirichletTuckerDecomp(S, K_M, K_N, K_P, alpha)
    # Initialize the parameters randomly
    print("initializing model")
    init_params = model.sample_params(key, M, N, P)
    print("done")

    # Fit the model with EM
    params, lps = model.fit(X, mask, init_params, 5000)

    # scale = M * N * P
    # plt.plot(jnp.array(lps) / scale)
    # plt.xlabel("iteration")
    # plt.ylabel("log joint prob (per entry)")

    # Compute test LL
    print("computing pct_dev")
    test_ll = model.heldout_log_likelihood(X, mask, params)

    # Make a baseline of average syllable usage in each epoch
    baseline_probs = jnp.mean(X[mask], axis=0)
    baseline_probs /= baseline_probs.sum(axis=-1, keepdims=True)
    baseline_test_ll = tfd.Multinomial(S, probs=baseline_probs).log_prob(X[~mask]).sum()

    # Compute the test log likelihood under the saturated model
    probs_sat = X / X.sum(axis=-1, keepdims=True)
    test_ll_sat = tfd.Multinomial(S, probs=probs_sat).log_prob(X)[~mask].sum()

    pct_dev = (test_ll - baseline_test_ll) / (test_ll_sat - baseline_test_ll)
    print("done")

    params_dict = dict(
        [(k, v) for k, v in zip(["G", "Psi", "Phi", "Theta"], params)]
    )

    return model, params_dict, lps, pct_dev


def plot_results(X, y, model, params):
    # Compute the reconstruction
    S_train = X.sum(axis=-1)
    X_hat = model.reconstruct(params, S_train)

    # Plot the true and reconstructed data for mouse `m`
    m = 0
    fig, axs = plt.subplots(3, 1)
    im = axs[0].imshow(X[m], aspect="auto", interpolation="none")
    plt.colorbar(im)
    im = axs[1].imshow(X_hat[m], aspect="auto", interpolation="none")
    plt.colorbar(im)
    im = axs[2].imshow(X[m] - X_hat[m], aspect="auto", interpolation="none")
    plt.colorbar(im)

    """### Look at the factors"""
    G, Psi, Phi, Theta = params

    # Plot the syllable topics
    plt.figure()
    plt.imshow(Theta, aspect="auto", interpolation="none")
    plt.xlabel("syllables")
    plt.ylabel("topics")
    plt.colorbar()

    # Plot the epoch loadings
    plt.figure()
    plt.imshow(Phi.T, aspect="auto", interpolation="none")
    plt.xlabel("epochs")
    plt.ylabel("factors")
    plt.colorbar()

    # Plot the mouse loadings
    plt.figure()
    perm = jnp.argsort(y)
    bounds = jnp.cumsum(jnp.bincount(y)[1:-1])
    plt.imshow(Psi[perm], aspect="auto", interpolation="none")
    for bound in bounds:
        plt.axhline(bound, color='r')
    plt.xlabel("factors")
    plt.ylabel("mice")
    plt.colorbar()


def evaluate_prediction(Psi, y):
    """### Classify drug labels"""

    # train_mask = jnp.ones(M, dtype=bool)
    # test_inds = jr.choice(jr.PRNGKey(1), jnp.arange(M,), shape=(M // 5,), replace=False)
    # train_mask = train_mask.at[test_inds].set(False)

    # lr = LogisticRegressionCV(verbose=False)
    # lr.fit(Psi[train_mask], drugs[train_mask] - 1)
    # print(lr.score(Psi[~train_mask], drugs[~train_mask] - 1))
    # pred_probs = lr.predict_proba(Psi[~train_mask])
    # print(pred_probs[jnp.arange(len(pred_probs)), drugs[~train_mask] - 1].mean())


    # ##
    # train_mask = jnp.ones(M, dtype=bool)
    # test_inds = jr.choice(jr.PRNGKey(0), jnp.arange(M,), shape=(M // 5,), replace=False)
    # train_mask = train_mask.at[test_inds].set(False)

    # Y = X.sum(axis=1)
    # Y /= Y.sum(axis=1, keepdims=True)

    # lr = LogisticRegressionCV(verbose=False)
    # lr.fit(Y[train_mask], drugs[train_mask] - 1)
    # print(lr.score(Y[~train_mask], drugs[~train_mask] - 1))
    # pred_probs = lr.predict_proba(Y[~train_mask])
    # print(pred_probs[jnp.arange(len(pred_probs)), drugs[~train_mask] - 1].mean())

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


# def model_selection():
#     """## Cross-validate the number of factors"""

#     import itertools as it

#     K_Ms = np.arange(2, 21)
#     K_Ns = np.arange(2, 11)
#     K_Ps = np.arange(2, 16)

#     all_test_lls = dict()
#     all_params = dict()
#     all_lps = dict()

#     for K_M, K_N, K_P in it.product(K_Ms, K_Ns, K_Ps):
#         print("fitting model with K_M={}, K_N={}, K_P={}".format(K_M, K_N, K_P))
#         key = (K_M, K_N, K_P)
#         if key in all_test_lls:
#             continue

#         model = DirichletTuckerDecomp(K_M, K_N, K_P, alpha)
#         init_params = model.sample_params(jr.PRNGKey(1), M, N, P)
#         params, lps = model.fit(X_train, init_params, 3000)

#         all_test_lls[key] = model.log_likelihood(X_test, params)
#         all_params[key] = params
#         all_lps[key] = lps

#     all_test_lls_tens = np.nan * np.ones((len(K_Ms), len(K_Ns), len(K_Ps)))
#     for i, K_M in enumerate(K_Ms):
#         for j, K_N in enumerate(K_Ns):
#             for k, K_P in enumerate(K_Ps):
#                 key = (K_M, K_N, K_P)
#                 if key in all_test_lls:
#                     all_test_lls_tens[i, j, k] = all_test_lls[key]

#     scale = M * N * P
#     plt.imshow(all_test_lls_tens[0] / scale, extent=(K_Ps[0], K_Ps[-1], K_Ns[-1], K_Ns[0]))
#     plt.xlabel(r"$K_P$")
#     plt.ylabel(r"$K_N$")
#     plt.title(r"test lls for $K_M=${}".format(K_Ms[0]))
#     plt.colorbar()

#     # Find the max test
#     jnp.unravel_index(jnp.argmax(all_test_lls_tens[0]), (len(K_Ns), len(K_Ps)))

@click.command()
@click.option('--seed', default=0, help='random seed for initialization.')
@click.option('--km', default=2, help='number of factors along dimension 1.')
@click.option('--kn', default=2, help='number of factors along dimension 2.')
@click.option('--kp', default=2, help='number of factors along dimension 3.')
@click.option('--alpha', default=1.1, help='concentration of Dirichlet prior.')
def run_one(seed, km, kn, kp, alpha, ):

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="serotonin-tucker-decomp",

        # track hyperparameters and run metadata
        config={
            "K_M": km,
            "K_N": kn,
            "K_P": kp,
            "alpha": alpha,
            "seed": seed,
            }
    )

    key = jr.PRNGKey(seed)

    # Split the data deterministically
    X, y = load_data()
    X_train, X_test = make_mask(X, key=0)

    # Fit the model using the random seed provided
    model, params, lps, pct_dev = fit_model(key, X_train, X_test, km, kn, kp, alpha)
    acc, confusion_matrix = evaluate_prediction(params["Psi"], y)
    wandb.run.summary["pct_dev"] = pct_dev
    wandb.run.summary["acc"] = acc

    # Plot some results
    # plot_results(X_train, y, model, params, confusion_matrix)

    # Finish the session
    wandb.finish()


@click.command()
@click.option('--data_dir', default="/home/groups/swl1/swl1", help='path to folder where data is stored.')
@click.option('--seed', default=0, help='random seed for initialization.')
@click.option('--km_min', default=2, help='number of factors along dimension 1.')
@click.option('--km_max', default=20, help='number of factors along dimension 1.')
@click.option('--kn_min', default=2, help='number of factors along dimension 2.')
@click.option('--kn_max', default=10, help='number of factors along dimension 2.')
@click.option('--kp_min', default=2, help='number of factors along dimension 3.')
@click.option('--kp_max', default=20, help='number of factors along dimension 3.')
@click.option('--k_step', default=1, help='step size for factor grid search.')
@click.option('--num_restarts', default=1, help='number of random initializations.')
@click.option('--alpha', default=1.1, help='concentration of Dirichlet prior.')
def run_sweep(data_dir, seed, km_min, km_max, kn_min, kn_max, kp_min, kp_max, k_step, num_restarts, alpha):

    # Split the data deterministically
    X, y = load_data(data_dir)
    mask = make_mask(X, key=0)

    for km, kn, kp in it.product(jnp.arange(km_min, km_max+1, step=k_step),
                                 jnp.arange(kn_min, kn_max+1, step=k_step),
                                 jnp.arange(kp_min, kp_max+1, step=k_step)):
        print("fitting model with km={}, kn={}, kp={}".format(km, kn, kp))

        # start a new wandb run to track this script
        print("initializing wandb")
        wandb.init(
            # set the wandb project where this run will be logged
            project="serotonin-tucker-decomp-masked",

            # track hyperparameters and run metadata
            config={
                "K_M": km,
                "K_N": kn,
                "K_P": kp,
                "alpha": alpha,
                "num_restarts": num_restarts,
                "seed": seed,
                }
        )
        print("done")

        # Fit the model using the random seed provided
        pct_devs = []
        accs = []
        for i in range(num_restarts):
            key = jr.PRNGKey(seed)
            model, params, lps, pct_dev = fit_model(key, X, mask, km, kn, kp, alpha)
            acc, confusion_matrix = evaluate_prediction(params["Psi"], y)

            pct_devs.append(pct_dev)
            accs.append(acc)
            seed += 1

        # Finish the session
        wandb.run.summary["pct_dev_mean"] = jnp.array(pct_devs).mean()
        wandb.run.summary["pct_dev_std"] = jnp.array(pct_devs).std()
        wandb.run.summary["acc_mean"] = jnp.array(accs).mean()
        wandb.run.summary["acc_std"] = jnp.array(accs).std()
        wandb.finish()


if __name__ == '__main__':
    run_sweep()
