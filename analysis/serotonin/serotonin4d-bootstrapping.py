
import bson
import os
import numpy as np
import pickle
import wandb
import jax.numpy as jnp
import jax.random as jr

from tensorflow_probability.substrates import jax as tfp
from fastprogress import progress_bar
from dtd.model4d import DirichletTuckerDecomp
tfd = tfp.distributions

# Setup logging and outputs
WANDB_PROJECT = "serotonin-behavior-bootstrap"
DATA_DIR = "/home/groups/swl1/swl1"
RESULTS_DIR = r"/home/groups/swl1/swl1/dirichlet-tucker/analysis/serotonin/results/2024_06_14-11_15-bootstrap"
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)


# Set hyperparameters
KM = 22
KN = 4 
KP = 4 
KS = 22 
ALPHA = 1.1
NUM_ITERS = 2000
TOL = 1e-4
NUM_BOOTSTRAP = 2


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


# Load the data
X, _ = load_data(DATA_DIR)
NUM_MICE, NUM_EPOCHS, NUM_POS, NUM_SYLLABLES = X.shape
mask = jnp.ones((NUM_MICE, NUM_EPOCHS), dtype=bool)

for i in progress_bar(range(NUM_BOOTSTRAP)):
    print(f"bootstrap sample {i}")
    result_file = os.path.join(RESULTS_DIR, f"params_{i:04d}.pkl")
    if os.path.exists(result_file):
        continue
        
    # Create a bootstrapped dataset
    key = jr.PRNGKey(i)
    bootstrap_inds = jr.choice(key, NUM_MICE, shape=(NUM_MICE,), replace=True)
    bootstrap_X = X[bootstrap_inds]
    
    # Initialize wandb run
    run = wandb.init(
        project=WANDB_PROJECT,
        job_type="train",
        config=dict(
            bootstrap_iter=i,
            bootstrap_inds=bootstrap_inds,
            km=KM,
            kn=KN,
            kp=KP,
            ks=KS,
            alpha=ALPHA,
            tol=TOL,
            max_num_iters=NUM_ITERS,
            data_dir=DATA_DIR,
            
            )
        )
    
    key, init_key = jr.split(key, 2)
    
    # Fit the model using the random seed provided
    model, params, lps = fit_model(init_key, X, mask, KM, KN, KP, KS, ALPHA, NUM_ITERS, TOL)

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
