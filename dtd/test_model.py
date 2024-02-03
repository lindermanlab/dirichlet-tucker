import pytest
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import optax

import dtd.model3d
import dtd.model4d

G_ref3d = jnp.array(
        [[[0.22888382, 0.12679227, 0.6443239 ],
        [0.6080653 , 0.26882705, 0.12310766],
        [0.34474236, 0.50688195, 0.1483757 ],
        [0.38623324, 0.1827938 , 0.4309729 ]],

       [[0.0597137 , 0.10976315, 0.83052313],
        [0.08409822, 0.49517527, 0.42072654],
        [0.399395  , 0.14816998, 0.45243502],
        [0.05969471, 0.42457888, 0.51572645]],

       [[0.81258833, 0.03440552, 0.15300614],
        [0.05181619, 0.5838028 , 0.364381  ],
        [0.3848821 , 0.46172112, 0.15339684],
        [0.38800618, 0.40774044, 0.20425339]],

       [[0.0885789 , 0.4264779 , 0.4849432 ],
        [0.42678872, 0.00795055, 0.5652607 ],
        [0.49590245, 0.09522115, 0.40887642],
        [0.9064688 , 0.05734175, 0.03618946]],

       [[0.46930948, 0.09847417, 0.43221632],
        [0.5446366 , 0.05531384, 0.4000495 ],
        [0.09186033, 0.5451099 , 0.36302978],
        [0.13099472, 0.45833832, 0.41066697]]]
)

Psi_ref3d = jnp.array(
       [[0.1629919 , 0.18183072, 0.2395989 , 0.20146915, 0.21410935],
       [0.11589337, 0.09371325, 0.32107556, 0.01414208, 0.45517582],
       [0.10069277, 0.15901922, 0.03371688, 0.46499023, 0.24158096],
       [0.23573256, 0.03423238, 0.11343041, 0.24994686, 0.3666578 ],
       [0.3906782 , 0.38704106, 0.05806748, 0.04818021, 0.11603302],
       [0.16267528, 0.04750169, 0.29558468, 0.30144304, 0.19279528],
       [0.00488283, 0.5319235 , 0.22648835, 0.07223001, 0.16447534],
       [0.40896958, 0.13896377, 0.07543833, 0.19709484, 0.17953344],
       [0.31956613, 0.24472027, 0.16860168, 0.11028862, 0.15682329],
       [0.01518635, 0.00439926, 0.13446318, 0.45742702, 0.38852417],
       [0.36385617, 0.03851783, 0.10740596, 0.31030107, 0.17991896],
       [0.03940718, 0.00435056, 0.02274767, 0.35610238, 0.5773922 ],
       [0.02984791, 0.25772396, 0.17290734, 0.35456583, 0.18495497],
       [0.21365607, 0.22828105, 0.07531224, 0.17303604, 0.3097146 ],
       [0.01970384, 0.2733329 , 0.30260247, 0.17859815, 0.2257627 ],
       [0.00730551, 0.1054559 , 0.3572269 , 0.5104085 , 0.01960316],
       [0.67160934, 0.09162226, 0.07302454, 0.04058229, 0.12316156],
       [0.00965542, 0.54336494, 0.19780669, 0.24818754, 0.00098539],
       [0.00580989, 0.4222336 , 0.32441396, 0.19215742, 0.05538515],
       [0.46920294, 0.0161285 , 0.40656146, 0.06735693, 0.0407501 ]]
)

Phi_ref3d = jnp.array(
       [[0.13136178, 0.0534343 , 0.47297212, 0.34223175],
       [0.25267345, 0.18352574, 0.5623502 , 0.00145059],
       [0.13470773, 0.47619787, 0.25461218, 0.13448225],
       [0.33707318, 0.05205541, 0.14018737, 0.47068408],
       [0.04598241, 0.3760978 , 0.1668522 , 0.41106752],
       [0.20026691, 0.30130368, 0.2208867 , 0.27754274],
       [0.26956794, 0.01181109, 0.3834261 , 0.3351949 ],
       [0.4620078 , 0.29676583, 0.15325469, 0.08797174],
       [0.5796633 , 0.13719837, 0.19329543, 0.08984292],
       [0.15861559, 0.30315536, 0.3274506 , 0.21077847]]
)

Theta_ref3d = jnp.array(
      [[0.02385255, 0.02362574, 0.18257715, 0.01686615, 0.02437016,
        0.01711842, 0.01098475, 0.04674876, 0.03147967, 0.01398002,
        0.00466638, 0.04388857, 0.00737828, 0.0673511 , 0.00065422,
        0.14076798, 0.00435977, 0.13324851, 0.04045269, 0.01744017,
        0.01988617, 0.01049244, 0.06980497, 0.02223207, 0.02577332],
       [0.0093546 , 0.0095409 , 0.01328044, 0.03143237, 0.01020836,
        0.0085911 , 0.04926695, 0.04338633, 0.07240042, 0.03016157,
        0.02167595, 0.00814663, 0.01465736, 0.02108227, 0.02711031,
        0.0183874 , 0.03164068, 0.005996  , 0.09229182, 0.17971422,
        0.03819614, 0.01116543, 0.02215276, 0.12560539, 0.10455452],
       [0.03881061, 0.01511372, 0.04741625, 0.01631909, 0.02183761,
        0.09205423, 0.02397863, 0.02495555, 0.00363545, 0.08273853,
        0.0542959 , 0.04758182, 0.07812005, 0.06836562, 0.01516082,
        0.07499385, 0.02441934, 0.07181901, 0.00244494, 0.01152093,
        0.06764752, 0.02727046, 0.07823356, 0.00526818, 0.00599832]]
)

def make_random_mask(key, shape, train_frac=0.8):
    """Make binary mask to split data into a train (1) and validation (0) sets."""
    return jr.bernoulli(key, train_frac, shape)

def make_shuffled_indices(key, shape, minibatch_size):
    """Shuffle indices and make into minibatches."""
    indices = jnp.indices(shape).reshape(len(shape), -1).T
    indices = jr.permutation(key, indices, axis=0)
    
    n_batches = len(indices) // minibatch_size
    batched_indices, remaining_indices \
                            = jnp.split(indices, (n_batches*minibatch_size,))
    batched_indices = batched_indices.reshape(n_batches, minibatch_size, -1)

    return batched_indices, remaining_indices

def test_fit_3d(shape=(20, 10, 25), 
                rank=(5, 4, 3), 
                batch_ndims=2,
                total_counts=1000,
                n_iters=5,
                key=0
                ):
    """Evaluate full-batch EM fit."""
    key = jr.PRNGKey(key)
    key_true, key_data, key_mask, key_init = jr.split(key, 4)

    # Instantiate the model
    model = dtd.model3d.DirichletTuckerDecomp(total_counts, *rank)

    # Generate observations from an underlying DTD model
    true_params = model.sample_params(key_true, *shape)
    X = model.sample_data(key_data, true_params)
    mask_shape = shape[:batch_ndims] 
    mask = make_random_mask(key_mask, mask_shape)

    # Initialize a different set of parameters and fit
    init_params = model.sample_params(key_init, *shape)

    fitted_params, lps = model.fit(X, mask, init_params, n_iters)

    # Check that log probabilities are monotonically increasing
    assert jnp.all(jnp.diff(lps) > 0), \
        f'Expected lps to be monotonically increasing.'
    
    assert jnp.allclose(G_ref3d, fitted_params[0], atol=1e-8)
    assert jnp.allclose(Psi_ref3d, fitted_params[1], atol=1e-8)
    assert jnp.allclose(Phi_ref3d, fitted_params[2], atol=1e-8)
    assert jnp.allclose(Theta_ref3d, fitted_params[3], atol=1e-8)

def test_fit_4d(shape=(10, 9, 8, 15), 
                rank=(5, 4, 3, 2), 
                batch_ndims=2,
                total_counts=1000,
                n_iters=5,
                key=0
                ):
    """Evaluate full-batch EM fit."""
    key = jr.PRNGKey(key)
    key_true, key_data, key_mask, key_init = jr.split(key, 4)

    # Instantiate the model
    model = dtd.model4d.DirichletTuckerDecomp(total_counts, *rank)
    
    # Generate observations from an underlying DTD model
    true_params = model.sample_params(key_true, *shape)
    X = model.sample_data(key_data, true_params)
    mask_shape = shape[:batch_ndims] 
    mask = make_random_mask(key_mask, mask_shape)

    # Initialize a different set of parameters and fit
    init_params = model.sample_params(key_init, *shape)

    # Fit the model
    fitted_params, lps = model.fit(X, mask, init_params, n_iters)

    # Check that log probabilities are monotonically increasing
    assert jnp.all(jnp.diff(lps) > 0), \
        f'Expected lps to be monotonically increasing.'
    
def test_get_minibatch_3d(shape=(20, 10, 25), 
                          rank=(5, 4, 3), 
                          batch_ndims=2,
                          total_counts=1000,
                          minibatch_size=30,
                          key=0
                          ):
    """Evaluate that lax functions are indexing arrays correctly."""

    key = jr.PRNGKey(key)
    key_params, key_data, key_mask, key_stats, key_shuffle = jr.split(key, 5)

    # Instantiate the model and generate parameters
    model = dtd.model3d.DirichletTuckerDecomp(total_counts, *rank)
    
    params = model.sample_params(key_params, *shape)
    
    # Generate data and mask
    X = model.sample_data(key_data, params)
    
    mask_shape = shape[:batch_ndims] 
    mask = make_random_mask(key_mask, mask_shape)
    
    # Generate random stats (ensure non-zero for evaluation)
    _stats = model._zero_rolling_stats(X, minibatch_size)
    stats = []
    for key_ss, ss in zip(jr.split(key_stats, len(_stats)), _stats):
        stats.append(jr.uniform(key_ss, ss.shape, minval=0, maxval=100))
    stats = tuple(stats)

    # Get a minibatch of indices. Ignore incomplete minibatch.
    # shape (n_batches, minibatch_size, batch_ndims) = (6,30,2)
    batched_indices, _ = make_shuffled_indices(key_shuffle,
                                               shape[:batch_ndims],
                                               minibatch_size,)    
    these_idxs = batched_indices[0]

    # Evaluate
    this_X, this_mask, these_params, these_stats \
                    = model._get_minibatch(these_idxs, X, mask, params, stats)
    
    refr_X = X[these_idxs[:,0], these_idxs[:,1]]
    assert jnp.allclose(this_X, refr_X, atol=1e-8)

    refr_mask = mask[these_idxs[:,0], these_idxs[:,1]]
    assert jnp.allclose(this_mask, refr_mask, atol=1e-8)

    refr_Psi = params[1][these_idxs[:,0], :]
    refr_Phi = params[2][these_idxs[:,1], :]
    assert jnp.allclose(these_params[1], refr_Psi, atol=1e-8)
    assert jnp.allclose(these_params[2], refr_Phi, atol=1e-8)

    refr_aPsi = stats[1][these_idxs[:,0], :]
    refr_aPhi = stats[2][these_idxs[:,1], :]
    assert jnp.allclose(these_stats[1], refr_aPsi, atol=1e-8)
    assert jnp.allclose(these_stats[2], refr_aPhi, atol=1e-8)

    # Make sure we can lax.scan through it
    def fn(carry, these_idxs):
        this_X, this_mask, these_params, these_stats \
                    = model._get_minibatch(these_idxs, X, mask, params, stats)
        
        return None, jnp.array([this_X.mean(),
                                this_mask.mean(),
                                these_params[1].mean(),
                                these_stats[2].mean()])
    
    _, outs = lax.scan(fn, None, batched_indices)
    assert len(outs) == len(batched_indices)

def test_stochastic_fit_3d(shape=(20, 10, 25), 
                rank=(5, 4, 3), 
                batch_ndims=2,
                total_counts=1000,
                n_epochs=5,
                minibatch_size=20,
                key=0
                ):
    """Evaluate full-batch EM fit."""
    key = jr.PRNGKey(key)
    key_true, key_data, key_mask, key_init, key_fit = jr.split(key, 5)

    # Instantiate the model
    model = dtd.model3d.DirichletTuckerDecomp(total_counts, *rank)

    # Generate observations from an underlying DTD model
    true_params = model.sample_params(key_true, *shape)
    X = model.sample_data(key_data, true_params)
    mask_shape = shape[:batch_ndims] 
    mask = make_random_mask(key_mask, mask_shape)

    # Initialize a different set of parameters
    init_params = model.sample_params(key_init, *shape)

    # Define a learning rate schedule function
    lr_schedule_fn = (lambda n_minibatches, n_epochs:
        optax.cosine_decay_schedule(
            init_value=1.,
            alpha=0.,
            decay_steps=n_minibatches*n_epochs,
            exponent=0.8,
        )
    )

    # Fit!
    fitted_params, lps \
        = model.stochastic_fit(X, mask, init_params, n_epochs,
                               lr_schedule_fn, minibatch_size, key_fit)

    # Check that log probabilities are monotonically increasing
    assert lps[-1,-1] - lps[0,0] > 0, \
        "Expected final lp to be more positive than initial lp, but got " \
        + f"lps[0,0]={lps[0,0]:.2f} and lps[-1,-1]={lps[-1,-1]:.2f}"
    
    avg_mono_incr = (jnp.diff(lps.ravel()) > 0).mean()
    assert avg_mono_incr >= 0.5, \
        "Expected lps to be monotically increasing at least 50% of the time, " \
        + f"but got {avg_mono_incr*100:0.0f}%. NB: This may be a sensitive metric."

