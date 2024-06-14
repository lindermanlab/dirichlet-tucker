import bson
import click
import numpy as np
import os
import pickle
import wandb

import jax.numpy as jnp
import jax.random as jr

from tensorflow_probability.substrates import jax as tfp
from fastprogress import progress_bar
from dtd.model4d import DirichletTuckerDecomp
tfd = tfp.distributions


def load_data(data_dir):
    # Load the drug ids ((M,) array of ints)
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
    params, lps = model.fit(X, mask, init_params, num_iters, tol=tol)

    return model, params, lps


@click.command()
@click.option('--data_dir', default="/home/groups/swl1/swl1", help='path to folder where data is stored.')
@click.option('--results_dir', default="/home/groups/swl1/swl1/dirichlet-tucker/analysis/serotonin/results/2024_06_14-11_15-bootstrap", help='path to folder where results are stored.')
@click.option('--km', default=22, help='number of mouse factors.')
@click.option('--kn', default=4, help='number of epoch factors.')
@click.option('--kp', default=4, help='number of position factors.')
@click.option('--ks', default=22, help='number of syllable factors.')
@click.option('--alpha', default=1.1, help='concentration of Dirichlet prior.')
@click.option('--num_iters', default=2000, help='max number of iterations of EM')
@click.option('--tol', default=1e-4, help='tolerance for EM convergence')
@click.option('--num_bootstrap', default=1000, help='number of bootstrap iterations')
@click.option('--wandb_project', default="serotonin-behavior-bootstrap", help='wandb project name')
def run_sweep(data_dir, 
              results_dir, 
              km, 
              kn, 
              kp, 
              ks, 
              alpha, 
              num_iters, 
              tol, 
              num_bootstrap,
              wandb_project):
    # Load the data
    X, _ = load_data(data_dir)
    num_mice, num_epochs, _, _ = X.shape
    mask = jnp.ones((num_mice, num_epochs), dtype=bool)

    for i in progress_bar(range(num_bootstrap)):
        print(f"bootstrap sample {i}")
        result_file = os.path.join(results_dir, f"params_{i:04d}.pkl")
        if os.path.exists(result_file):
            continue
            
        # Create a bootstrapped dataset
        key = jr.PRNGKey(i)
        bootstrap_inds = jr.choice(key, num_mice, shape=(num_mice,), replace=True)
        bootstrap_X = X[bootstrap_inds]
        
        # Initialize wandb run
        run = wandb.init(
            project=wandb_project,
            job_type="train",
            config=dict(
                bootstrap_iter=i,
                bootstrap_inds=bootstrap_inds,
                km=km,
                kn=kn,
                kp=kp,
                ks=ks,
                alpha=alpha,
                tol=tol,
                max_num_iters=num_iters,
                data_dir=data_dir,
                results_dir=results_dir,
                )
            )
        
        key, init_key = jr.split(key, 2)
        
        # Fit the model using the random seed provided
        model, params, lps = fit_model(init_key, 
                                       bootstrap_X, 
                                       mask, 
                                       km, 
                                       kn, 
                                       kp, 
                                       ks, 
                                       alpha, 
                                       num_iters, 
                                       tol)

        # Save and log the results
        wandb.run.summary["final_lp"] = lps[-1]
        print("saving results")
        with open(result_file, 'wb') as f:
            pickle.dump(dict(bootstrap_inds=bootstrap_inds,
                            core_tensor=params[0],
                            loadings=params[1],
                            epoch_factors=params[2],
                            position_factors=params[3],
                            syllable_factors=params[4],
                            ), 
                        f)

        artifact = wandb.Artifact(name="params_pkl", type="model")
        artifact.add_file(local_path=result_file)
        run.log_artifact(artifact)

        # Log results to wandb
        wandb.finish()
        
        del model
        del params
        del lps
    

if __name__ == '__main__':
    run_sweep()
