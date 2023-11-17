import warnings
import jax.numpy as jnp
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp
from jax import jit, lax
from jax.tree_util import tree_map
from tqdm.auto import trange
import itertools
import time

from dtd.utils import ShuffleIndicesIterator

tfd = tfp.distributions
warnings.filterwarnings("ignore")

class DirichletTuckerDecomp:

    def __init__(self, C, K_M, K_N, K_P, alpha=1.1):
        """Dirichlet-Tucker decomposition of a 3d tensor with 2 batch dimension (m, n)
        and 1 event dimension (p,). 

        C: total counts for each (m,n) slice of the data tensor
        K_M, K_N, K_P: dimension of factors for the M, N, and P axes, respectively.
        alpha: concentration of Dirichlet prior. Assume shared by all axes.

        """
        self.C = C
        self.K_M = K_M
        self.K_N = K_N
        self.K_P = K_P
        self.alpha = alpha

        self.batch_ndims = 2
        self.event_ndims = 1

    def sample_params(self, key, M, N, P):
        """Sample a data tensor and parameters from the model.

        Args:
        key: jr.PRNGKey
        M, N, P: dimensions of the data

        Returns:
            params = (G, Psi, Phi, Theta) where
                G: (K_M, K_N, K_P) core tensor
                Psi: (M, K_M) factor
                Phi: (N, K_N) factor
                Theta: (K_P, P) topic (note transposed!)
        """
        # TODO: This function is stupidly slow!
        K_M, K_N, K_P = self.K_M, self.K_N, self.K_P

        # Sample parameters from the prior
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        G = tfd.Dirichlet(self.alpha * jnp.ones(K_P)).sample(seed=k1, sample_shape=(K_M, K_N,))
        Psi = tfd.Dirichlet(self.alpha * jnp.ones(K_M)).sample(seed=k2, sample_shape=(M,))
        Phi = tfd.Dirichlet(self.alpha * jnp.ones(K_N)).sample(seed=k3, sample_shape=(N,))
        Theta = tfd.Dirichlet(self.alpha * jnp.ones(P)).sample(seed=k4, sample_shape=(K_P,))
        return (G, Psi, Phi, Theta)

    def sample_data(self, key, params):
        """Sample a data tensor and parameters from the model.

        key: jr.PRNGKey
        params: tuple of params
        
        returns:
            X: (M, N, P) data tensor where shapes are determined by params.
        """
        # Sample data
        probs = jnp.einsum('ijk,mi,nj,kp->mnp', *params)
        X = tfd.Multinomial(self.C, probs=probs).sample(seed=key)
        return X

    # Now implement the EM algorithm
    def _e_step_g(self, masked_X, probs, params):
        r_g = params[0] * jnp.einsum('mnp,mi,nj,kp->ijk', masked_X/probs, *params[1:])
        return r_g
    
    def _e_step_psi(self, masked_X, probs, params):
        r_psi = jnp.einsum('ijk,mi,nj,kp->imnp', *params)
        return (masked_X * (r_psi / probs)).sum(axis=(2,3)).T
    
    def _e_step_phi(self, masked_X, probs, params):
        r_phi = jnp.einsum('ijk,mi,nj,kp->jmnp', *params)
        return (masked_X * (r_phi / probs)).sum(axis=(1,3)).T
    
    def _e_step_theta(self, masked_X, probs, params):
        r_theta = jnp.einsum('ijk,mi,nj,kp->kmnp', *params)
        return (masked_X * (r_theta / probs)).sum(axis=(1,2))
    
    def e_step(self, X, mask, params):
        """Compute posterior expected sufficient statistics of parameters.
        
        X: (M, N, P) count tensor
        mask: (M,N) binary matrix specifying which epochs are held-out for
            which mice.
        """
        probs = jnp.einsum('ijk,mi,nj,kp->mnp', *params)
        masked_X = X * mask[..., None]

        # compute alpha_* given E[Z]
        alpha_G = self._e_step_g(masked_X, probs, params)
        alpha_Psi = self._e_step_psi(masked_X, probs, params)
        alpha_Phi = self._e_step_phi(masked_X, probs, params)
        alpha_Theta =  self._e_step_theta(masked_X, probs, params)
        return alpha_G, alpha_Psi, alpha_Phi, alpha_Theta

    def _m_step_g(self, alpha_G):
        """Maximize conditional distribution of core tensor.

        alpha_G: (K_M, K_N, K_P)
        """
        alpha_post = self.alpha + alpha_G
        return tfd.Dirichlet(alpha_post).mode()

    def _m_step_psi(self, alpha_Psi):
        """Maximize conditional distribution of Psi factor.
        
        alpha_Psi: (M, K_M)
        """
        alpha_post = self.alpha + alpha_Psi
        return tfd.Dirichlet(alpha_post).mode()

    def _m_step_phi(self, alpha_Phi):
        """Maximize conditional distribution of Phi factor.
        
        alpha_Phi: (N, K_N)
        """
        alpha_post = self.alpha + alpha_Phi
        return tfd.Dirichlet(alpha_post).mode()

    def _m_step_theta(self, alpha_Theta):
        """Maximize conditional distribution of Phi factor.
        
        alpha_Theta: (K_P, P)
        """
        alpha_post = self.alpha + alpha_Theta
        return tfd.Dirichlet(alpha_post).mode()

    def m_step(self, alpha_G, alpha_Psi, alpha_Phi, alpha_Theta):
        G = self._m_step_g(alpha_G)
        Psi = self._m_step_psi(alpha_Psi)
        Phi = self._m_step_phi(alpha_Phi)
        Theta = self._m_step_theta(alpha_Theta)
        return G, Psi, Phi, Theta

    def heldout_log_likelihood(self, X, mask, params):
        # TODO: Compute the log likelihood of the held-out entries in X
        probs = jnp.einsum('ijk,mi,nj,kp->mnp', *params)
        return jnp.where(~mask, tfd.Multinomial(self.C, probs=probs).log_prob(X), 0.0).sum()

    def log_prob(self, X, mask, params):
        M, N, P = X.shape
        G, Psi, Phi, Theta = params

        # log prior
        lp = tfd.Dirichlet(self.alpha * jnp.ones(self.K_P)).log_prob(G).sum()
        lp += tfd.Dirichlet(self.alpha * jnp.ones(self.K_M)).log_prob(Psi).sum()
        lp += tfd.Dirichlet(self.alpha * jnp.ones(self.K_N)).log_prob(Phi).sum()
        lp += tfd.Dirichlet(self.alpha * jnp.ones(P)).log_prob(Theta).sum()

        # log likelihood of observed data
        probs = jnp.einsum('ijk,mi,nj,kp->mnp', *params)
        lp += jnp.where(mask, tfd.Multinomial(self.C, probs=probs).log_prob(X), 0.0).sum()
        return lp

    def reconstruct(self, params):
        return self.C * jnp.einsum('ijk, mi, nj, kp->mnp', *params)
    
    def leave_one_out_log_likelihood(self, X, mask, params):
        """Calculate heldout log likelihood when withholding one factor at a time."""
        G, F1, F2, F3 = params
        K1, K2, K3 = G.shape

        def f1_step(carry, k):
            g_axis, f_axis = 0, 1

            rolled_G, rolled_F = carry
            K = rolled_G.shape[g_axis]
            
            # G[1:,:,:], shape (K1-1, K2, K3)
            G_ = lax.dynamic_slice_in_dim(rolled_G, 1, K-1, axis=g_axis)

            # F[:,1:], shape(D1, K1-1)
            F_ = lax.dynamic_slice_in_dim(rolled_F, 1, K-1, axis=f_axis)

            # Compute held-out log likelihood
            params_ = (G_, F_, F2, F3)
            ll = self.heldout_log_likelihood(X, mask, params_)

            # Roll the carried core tensor and factor for the next iteration
            rolled_G = jnp.roll(rolled_G, -1, axis=g_axis)
            rolled_F = jnp.roll(rolled_F, -1, axis=f_axis)
            return (rolled_G, rolled_F), ll
        
        def f2_step(carry, k):
            g_axis, f_axis = 1, 1

            rolled_G, rolled_F = carry
            K = rolled_G.shape[g_axis]
            
            # G[:,1:,:], shape (K1, K2-1, K3)
            G_ = lax.dynamic_slice_in_dim(rolled_G, 1, K-1, axis=g_axis)

            # F[:,1:], shape(D1, K1-1)
            F_ = lax.dynamic_slice_in_dim(rolled_F, 1, K-1, axis=f_axis)

            # Compute held-out log likelihood
            params_ = (G_, F1, F_, F3)
            ll = self.heldout_log_likelihood(X, mask, params_)

            # Roll the carried core tensor and factor for the next iteration
            rolled_G = jnp.roll(rolled_G, -1, axis=g_axis)
            rolled_F = jnp.roll(rolled_F, -1, axis=f_axis)
            return (rolled_G, rolled_F), ll
        
        def f3_step(carry, k):
            g_axis, f_axis = 2, 0

            rolled_G, rolled_F = carry
            K = rolled_G.shape[g_axis]
            
            # G[:,1:,:], shape (K1, K2, K3-1)
            G_ = lax.dynamic_slice_in_dim(rolled_G, 1, K-1, axis=g_axis)

            # F[1:], shape(K3-1, D3)
            F_ = lax.dynamic_slice_in_dim(rolled_F, 1, K-1, axis=f_axis)

            # Compute held-out log likelihood
            params_ = (G_, F1, F2, F_)
            ll = self.heldout_log_likelihood(X, mask, params_)

            # Roll the carried core tensor and factor for the next iteration
            rolled_G = jnp.roll(rolled_G, -1, axis=g_axis)
            rolled_F = jnp.roll(rolled_F, -1, axis=f_axis)
            return (rolled_G, rolled_F), ll
            
        _, lls_1 = lax.scan(f1_step, (G, F1), jnp.arange(K1))
        _, lls_2 = lax.scan(f2_step, (G, F2), jnp.arange(K2))
        _, lls_3 = lax.scan(f3_step, (G, F3), jnp.arange(K3))
        
        return lls_1, lls_2, lls_3
    
    def sort_params(self, X, mask, params):
        """Sort params by held-out log-likelihood explained.
        
        Returns sorted parameters are their log likelihoods
        """
        lls_1, lls_2, lls_3 = self.leave_one_out_log_likelihood(X, mask, params)

        # Withheld factor with greatest contribution will have the lowest LOFO
        # log likelihood. So, use argsort ordering exactly as is
        i1_sorted = jnp.argsort(lls_1)
        i2_sorted = jnp.argsort(lls_2)
        i3_sorted = jnp.argsort(lls_3)

        # Sort factors and core tensor
        G, F1, F2, F3 = params
        F1 = F1[:, i1_sorted]
        F2 = F2[:, i2_sorted]
        F3 = F3[i3_sorted, :,]

        # Sort core tensor axis-by-axis
        G = G[i1_sorted,:,:]
        G = G[:,i2_sorted,:]
        G = G[:,:,i3_sorted]

        return (G, F1, F2, F3), (lls_1[i1_sorted], lls_2[i2_sorted], lls_3[i3_sorted])

    # Fit the model!
    def fit(self, X, mask, init_params, num_iters, wnb=None):

        @jit
        def em_step(X, mask, params):
            E_params = self.e_step(X, mask, params)
            params = self.m_step(*E_params)
            lp = self.log_prob(X, mask, params)
            return lp, params

        # Initialize the recursion
        params = init_params
        lps = []
        for itr in trange(num_iters):
            epoch_start_time = time.time()
            
            lp, params = em_step(X, mask, params)
            lps.append(lp)

            epoch_elapsed_time = time.time() - epoch_start_time

            # Log metrics to WandB
            if wnb is not None:
                wnb.log({'avg_lp': lp / mask.sum(), 'epoch':itr},
                        step=itr, commit=False)
                wnb.log({'epoch_time': epoch_elapsed_time, 'epoch':itr},
                        step=itr, commit=True)

        return params, jnp.stack(lps)

    def _zero_rolling_stats(self, X, minibatch_size):
        M, N, P = X.shape
        return (jnp.zeros((self.K_M, self.K_N, self.K_P)),
                jnp.zeros((minibatch_size, self.K_M)),
                jnp.zeros((minibatch_size, self.K_N)),
                jnp.zeros((self.K_P, P)),
        )
    
    def _get_minibatch(self, these_idxs, X, mask, params, stats):
        """Fancy-index the samples associated with `these_idxs` from each input

        Parameters
            these_idxs: shape (m, batch_ndims)
            X: data tensor, shape (b1, b2, e1)
            mask: data mask, shape (b1, b2)
            params: tuple of model parameters
                G: shape (k1, k2, k3)
                Psi: shape (b1, k1)
                Phi: shape (b2, k2)
                Theta: shape (e1, k3)
            stats: tuple of expected sufficient statistics
                alpha_G: shape (k1, k2, k3)
                alpha_Psi: shape (b1, k1)
                alpha_Phi: shape (b2, k2)
                alpha_Theta: shape (k3, e1)
        
        Returns
            this_X: data tensor, shape (m, e1)
            this_mask: data mask, shape (m)
            these_params: tuple of model parameters
                G: shape (k1, k2, k3), unchanged
                Psi: shape (m, k1)
                Phi: shape (m, k2)
                Theta: shape (e1, k3), unchanged
            these_stats: tuple of expected sufficient statistics
                alpha_G: shape (k1, k2, k3), unchanged
                alpha_Psi: shape (m, k1)
                alpha_Phi: shape (m, k2)
                alpha_Theta: shape (k3, e1), unchanged
        """
        
        this_X = X[these_idxs[:,0], these_idxs[:,1]]
        this_mask = mask[these_idxs[:,0], these_idxs[:,1]]

        # Index into parameters and statistics
        G, Psi, Phi, Theta = params
        alpha_G, alpha_Psi, alpha_Phi, alpha_Theta = stats

        # Returns: (minibatch_size, K_M)
        this_Psi = Psi[these_idxs[:,0], :]
        this_alpha_Psi = alpha_Psi[these_idxs[:,0], :]
        
        # Returns: (minibatch_size, K_N)
        this_Phi = Phi[these_idxs[:,1], :]
        this_alpha_Phi = alpha_Phi[these_idxs[:,1], :]
        
        return (this_X,
                this_mask,
                (G, this_Psi, this_Phi, Theta),
                (alpha_G, this_alpha_Psi, this_alpha_Phi, alpha_Theta))
    
    def minibatched_e_step(self, X, mask, params):
        """Compute posterior expected sufficient statistics of parameters from a minibatch of data.
        
        We are given a minibatch of data, e.g. X has shape (B,P). Contrast this
        with the data input of the full-batch `e_step`, where X has shape (M,N,P).
        Additionally, only the parameters relevant to the minibatch are provided.

        Parameters
            X: (B, P) count tensor
            mask: (B,) binary matrix specifying held-out samples.
            params: tuple of arrays
                G: shape (K_M, K_N, K_P)
                Psi: shape (B, K_M)
                Phi: shape (B, K_N)
                Theta: shape (K_P, P)
        
        Returns
            alpha_G: shape (K_M, K_N, K_P)
            alpha_Psi: shape (B, K_M)
            alpha_Phi: shape (B, K_N)
            alpha_Theta: shape (K_P, P)
        """

        # Note the different einsum notation when working with minibatched data.
        # This is equivalent to calculating the full tensor associated with these
        # parameters, then taking the principle diagonal along the batch dims.
        probs = jnp.einsum('ijk,bi,bj,kp->bp', *params)
        relative_probs = jnp.einsum('ijk,bi,bj,kp->ijkbp', *params)
        relative_probs /= probs
        E_Z = X * mask[..., None] * relative_probs

        # Note how E_z has collapsed the batch dimensions into a single axis.
        # This changes the indexing and which axes are being summed over.
        alpha_G = jnp.sum(E_Z, axis=(3,4))
        alpha_Psi = jnp.sum(E_Z, axis=(1,2,4)).T
        alpha_Phi = jnp.sum(E_Z, axis=(0,2,4)).T
        alpha_Theta = jnp.sum(E_Z, axis=(0,1,3))

        return alpha_G, alpha_Psi, alpha_Phi, alpha_Theta
    
    def stochastic_fit(self, X, mask, init_params, n_epochs,
                       lr_schedule_fn, minibatch_size, key, drop_last=False, wnb=None):
        """Fit model parameters to data using stochastic expectation maximization.
        
        Parameters

            n_epochs (int):
                Number of epochs, or passes through the full dataset, to run.
            lr_schedule_fn (Callable[[n_minibatches, n_epochs], optax.Schedule]):
                Given (n_minibatches, n_epochs), returns an optax.Schedule
            minibatch_size (int):
                Number of data samples to fit on, sampled uniformly from the batch dimensions.
            key (PRNGKey):
                PRNGKey to shuffle data loading order at each epoch
            drop_last: bool
                If False (default), process the last incomplete batch. This will
                result in an additional "slot" of memory, since it has different
                shape from the the main batch size. If running into OOM issues,
                can set to True, in which the incomplete batch is skipped. NB:
                it is up to the user to ensure that the skipped batch is relatively
                small to not overly affect the infererred statistics.
            wnb (WandB instance, [optional]):
                If WandB instance passed in, log average lps and learning rates
                for every epoch.
        
        Returns
            params
            lps: (n_epoch, n_total_minibatches_per_epoch)
            
        """

        # Instantiate an iterator that produces minibatches of indices into the data
        batch_shape = X.shape[:self.batch_ndims]
        indices_iterator = ShuffleIndicesIterator(key, batch_shape, minibatch_size)
        print(f'Running {indices_iterator.n_complete} minibatches of size {indices_iterator.minibatch_size}.')
        print(f"Incomplete minibatch size of {indices_iterator.incomplete_size}. drop_last={drop_last},")

        # Define learning rate schedule and split in complete and incomplete lrs
        n_minibatches_per_epoch = indices_iterator.n_complete
        assert n_minibatches_per_epoch > 0, \
            f"Expected n_minibatches_per_epoch > 0. Got minibatch_size={minibatch_size}, perhaps misspecified?"
        
        schedule = lr_schedule_fn(n_minibatches_per_epoch, n_epochs)
        
        learning_rates = schedule(jnp.arange(n_minibatches_per_epoch*n_epochs))
        learning_rates = learning_rates.reshape(n_epochs, n_minibatches_per_epoch)

        # Minibatch scaling factor, for each sufficient statistic. See math notes.
        scaling_factor = (batch_shape[0] * batch_shape[1] / minibatch_size**2,  # M / B_M * N / B_N
                          batch_shape[1] / minibatch_size,                      # N / B_N
                          batch_shape[0] / minibatch_size,                      # M / B_M
                          batch_shape[0] * batch_shape[1] / minibatch_size**2,) # M / B_M * N / B_N

        # Define EM step
        def em_step(carry, these_inputs):
            prev_params, prev_rolling_stats = carry
            these_idxs, lr = these_inputs
            
            # Gather minibatched data
            this_X, this_mask, these_prev_params, these_prev_rolling_stats \
                    = self._get_minibatch(these_idxs, X, mask, prev_params, prev_rolling_stats)

            # Compute expected sufficient statistics of minibatch
            these_stats = self.minibatched_e_step(this_X, this_mask, these_prev_params)

            # Incorporate expected stats from minibatch into rolling statistics
            _update_fn = (
                lambda rolling_stat, this_stat, scale:
                (1-lr) * rolling_stat + lr * scale * this_stat
            )
            these_rolling_stats = tree_map(
                _update_fn, these_prev_rolling_stats, these_stats, scaling_factor
            )

            # Maximize the posterior
            G, this_Psi, this_Phi, Theta = self.m_step(*these_rolling_stats)

            # Update the Psi and Phi of params and rolling stats at specified indices
            Psi, Phi = prev_params[1], prev_params[2]
            Psi = Psi.at[these_idxs[:,0],:].set(this_Psi)
            Phi = Phi.at[these_idxs[:,1],:].set(this_Phi)
            params = (G, Psi, Phi, Theta)

            alpha_Psi, alpha_Phi = prev_rolling_stats[1], prev_rolling_stats[2]
            alpha_G, this_alpha_Psi, this_alpha_Phi, alpha_Theta = these_rolling_stats
            alpha_Psi = alpha_Psi.at[these_idxs[:,0],:].set(this_alpha_Psi)
            alpha_Phi = alpha_Phi.at[these_idxs[:,1],:].set(this_alpha_Phi)
            rolling_stats = (alpha_G, alpha_Psi, alpha_Phi, alpha_Theta)

            # Calculate log-likelihood on full data
            lp = self.log_prob(X, mask, params)

            return (params, rolling_stats), lp

        # Explicitly jit last em step
        # TODO Have not confirmed expected behavior in reducing compile time
        incomplete_em_step = jit(em_step)

        # Initialize parameters, rolling stats
        params = init_params
        rolling_stats = self._zero_rolling_stats(X, minibatch_size)
        all_lps = []
        for epoch in trange(n_epochs):
            epoch_start_time = time.time()

            batched_indices, remaining_indices = next(indices_iterator)
            lrs = learning_rates[epoch]

            # Scan through all complete minibatches
            (params, rolling_stats), lps = lax.scan(
                em_step, (params, rolling_stats), (batched_indices, lrs),
            )

            # If drop_last == False, perform one final stochastic EM step over
            # the incomplete minibatch. Reuse last learning rate for simplicity;
            # this is okay especially if incomplete minibatch has few samples.
            if (not drop_last) and (len(remaining_indices) > 0):
                (params, rolling_stats), remaining_lp = incomplete_em_step(
                    (params, rolling_stats), (remaining_indices, lrs[-1])
                )
                lps = jnp.concatenate([lps, jnp.atleast_1d(remaining_lp)])

            epoch_elapsed_time = time.time() - epoch_start_time

            # Log metrics to WandB
            if wnb is not None:
                for i, (lp, lr) in enumerate(zip(lps, lrs)):
                    wnb.log({'avg_lp': lp / mask.sum(),
                             'learning_rate': lr,
			     'epoch': epoch+(i+1)/n_minibatches_per_epoch},
                             step=epoch*n_minibatches_per_epoch+i,
                             commit=False)
                
                    wnb.log({'epoch_time [min]': epoch_elapsed_time/60,
                           'epoch': epoch},
                           step=(epoch+1)*n_minibatches_per_epoch-1,
                           commit=True)
                
            # Check for NaNs, to more quickly identify failing!
            if any(tree_map(lambda arr: jnp.any(jnp.isnan(arr)), params)):
                raise ValueError(f"Expected params to be finite, but got\n{params}")
            
            all_lps.append(lps)

        return params, jnp.array(all_lps)
