import warnings
import jax.numpy as jnp
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp
from jax import jit, lax
from tqdm.auto import trange
import itertools

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
    def e_step(self, X, mask, params):
        """Compute posterior expected sufficient statistics of parameters.
        
        X: (M, N, P) count tensor
        mask: (M,N) binary matrix specifying which epochs are held-out for
            which mice.
        """
        probs = jnp.einsum('ijk,mi,nj,kp->mnp', *params)
        relative_probs = jnp.einsum('ijk,mi,nj,kp->ijkmnp', *params)
        relative_probs /= probs
        E_Z = X * mask[..., None] * relative_probs

        # compute alpha_* given E[Z]
        alpha_G = jnp.sum(E_Z, axis=(3,4,5))
        alpha_Psi = jnp.sum(E_Z, axis=(1,2,4,5)).T
        alpha_Phi = jnp.sum(E_Z, axis=(0,2,3,5)).T
        alpha_Theta = jnp.sum(E_Z, axis=(0,1,3,4))
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

    # Fit the model!
    def fit(self, X, mask, init_params, num_iters):

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
            lp, params = em_step(X, mask, params)
            lps.append(lp)

        return params, jnp.stack(lps)

    def _zero_rolling_stats(self, X, minibatch_size):
        M, N, P = X.shape
        return (jnp.zeros((self.K_M, self.K_N, self.K_P)),
                jnp.zeros((minibatch_size, self.K_M)),
                jnp.zeros((minibatch_size, self.K_N)),
                jnp.zeros((self.K_P, P)),
        )
    
    def _fancy_get_minibatch(self, these_idxs, X, mask, params, stats):
        """Gather the samples associated with `these_idxs` from each input.

        All the lax functions seem unnecessary >_< -- fancy indexing seems
        to work just fine, see `_get_minibatch` function. Keeping here until
        absolutely confirmed that lax slicing is unnecessary.

        Parameters
            these_idxs: shape (n_samples, batch_ndims)
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
        batch_ndims = int(these_idxs.shape[-1])
        batch_shape = X.shape[:batch_ndims]
        event_shape = X.shape[batch_ndims:]
        event_ndims = len(event_shape)

        X_slice_sizes = (1,)*len(batch_shape) + event_shape
        mask_slice_sizes =(1,)*len(batch_shape)
        Psi_slice_sizes = (1, self.K_M)
        Phi_slice_sizes = (1, self.K_N)

        X_dim_nums = lax.GatherDimensionNumbers(
            offset_dims=tuple(i for i in range(1, event_ndims+1)),
            collapsed_slice_dims=tuple(i for i in range(batch_ndims)),
            start_index_map=tuple(i for i in range(batch_ndims))
        )
        mask_dim_nums = lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=tuple(i for i in range(batch_ndims)),
            start_index_map=tuple(i for i in range(batch_ndims))
        )
        factor_dim_nums = lax.GatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(0,),
            start_index_map=(0,)
        )

        this_X = lax.gather(X, these_idxs, X_dim_nums, X_slice_sizes)
        this_mask = lax.gather(mask, these_idxs, mask_dim_nums, mask_slice_sizes)

        # Index into parameters and statistics
        G, Psi, Phi, Theta = params
        alpha_G, alpha_Psi, alpha_Phi, alpha_Theta = stats

        # TODO Don't I need to lax.slice into `these_idxs`
        # Returns: (minibatch_size, K_M)
        this_Psi = lax.gather(Psi, these_idxs[:,[0]], factor_dim_nums, Psi_slice_sizes)
        this_alpha_Psi = lax.gather(alpha_Psi, these_idxs[:,[0]], factor_dim_nums, Psi_slice_sizes)
        
        # Returns: (minibatch_size, K_N)
        this_Phi = lax.gather(Phi, these_idxs[:,[1]], factor_dim_nums, Phi_slice_sizes)
        this_alpha_Phi = lax.gather(alpha_Phi, these_idxs[:,[1]], factor_dim_nums, Phi_slice_sizes)
        
        return (this_X,
                this_mask,
                (G, this_Psi, this_Phi, Theta),
                (alpha_G, this_alpha_Psi, this_alpha_Phi, alpha_Theta))
    
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
    
