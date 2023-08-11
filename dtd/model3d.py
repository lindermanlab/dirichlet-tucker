import warnings
import jax.numpy as jnp
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp
from jax import jit
from tqdm.auto import trange

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