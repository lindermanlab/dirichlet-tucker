import warnings
import jax.numpy as jnp
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp
from jax import jit
from tqdm.auto import trange

tfd = tfp.distributions
warnings.filterwarnings("ignore")

class DirichletTuckerDecomp:

    def __init__(self, C, K_M, K_N, K_P, K_S, alpha=1.1):
        """Dirichlet-Tucker decomposition of a 4d tensor with 2 batch dimension (m, n)
        and 2 event dimensions (p, s). 

        C: total counts for each (m,n) slice of the data tensor
        K_M, K_N, K_P, K_S: dimension of factors for the M, N, P, and S axes, respectively.
        alpha: concentration of Dirichlet prior. Assume shared by all axes.

        """
        self.C = C
        self.K_M = K_M
        self.K_N = K_N
        self.K_P = K_P
        self.K_S = K_S
        self.alpha = alpha

    def sample_params(self, key, M, N, P, S):
        """Sample a data tensor and parameters from the model.

        Args:
        key: jr.PRNGKey
        M, N, P, S: dimensions of the data

        Returns:
            params = (G, Psi, Phi, Theta, Lambda) where
                G: (K_M, K_N, K_P, K_S) core tensor
                Psi: (M, K_M) factor
                Phi: (N, K_N) factor
                Theta: (K_P, P) topic (note transposed!)
                Lambda: (K_S, S) topic (note transposed!)
        """
        K_M, K_N, K_P, K_S = self.K_M, self.K_N, self.K_P, self.K_S
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        
        # Using TFP: This is stupidly slow for some reason!
        # G = tfd.Dirichlet(self.alpha * jnp.ones(K_P * K_S)).sample(seed=k1, sample_shape=(K_M, K_N,)).reshape((K_M, K_N, K_P, K_S))
        # Psi = tfd.Dirichlet(self.alpha * jnp.ones(K_M)).sample(seed=k2, sample_shape=(M,))
        # Phi = tfd.Dirichlet(self.alpha * jnp.ones(K_N)).sample(seed=k3, sample_shape=(N,))
        # Theta = tfd.Dirichlet(self.alpha * jnp.ones(P)).sample(seed=k4, sample_shape=(K_P,))
        # Lambda = tfd.Dirichlet(self.alpha * jnp.ones(S)).sample(seed=k5, sample_shape=(K_S,))

        # Using jax.random
        G = jr.dirichlet(k1, self.alpha * jnp.ones(K_P * K_S), shape=(K_M, K_N,)).reshape((K_M, K_N, K_P, K_S))
        Psi = jr.dirichlet(k2, self.alpha * jnp.ones(K_M), shape=(M,))
        Phi = jr.dirichlet(k3, self.alpha * jnp.ones(K_N), shape=(N,))
        Theta = jr.dirichlet(k4, self.alpha * jnp.ones(P), shape=(K_P,))
        Lambda = jr.dirichlet(k5, self.alpha * jnp.ones(S), shape=(K_S,))
        return (G, Psi, Phi, Theta, Lambda)

    def sample_data(self, key, params):
        """Sample a data tensor and parameters from the model.

        key: jr.PRNGKey
        params: tuple of params
        
        returns:
            X: (M, N, P) data tensor where shapes are determined by params.
        """
        # Sample data
        probs = jnp.einsum('ijkl,mi,nj,kp,ls->mnps', *params)
        X = tfd.Multinomial(self.C, probs=probs).sample(seed=key)
        return X

    # Now implement the EM algorithm
    def e_step(self, X, mask, params):
        """Compute posterior expected sufficient statistics of parameters.
        
        X: (M, N, P, S) count tensor
        mask: (M,N) binary matrix specifying which epochs are held-out for
            which mice.
        """
        probs = jnp.einsum('ijkl,mi,nj,kp,ls->mnps', *params)
        relative_probs = jnp.einsum('ijkl,mi,nj,kp,ls->ijklmnps', *params)
        relative_probs /= probs
        E_Z = X * mask[..., None, None] * relative_probs

        # compute alpha_* given E[Z]
        alpha_G = jnp.sum(E_Z, axis=(4,5,6,7))
        alpha_Psi = jnp.sum(E_Z, axis=(1,2,3,5,6,7)).T
        alpha_Phi = jnp.sum(E_Z, axis=(0,2,3,4,6,7)).T
        alpha_Theta = jnp.sum(E_Z, axis=(0,1,3,4,5,7))
        alpha_Lambda = jnp.sum(E_Z, axis=(0,1,2,4,5,6))
        return alpha_G, alpha_Psi, alpha_Phi, alpha_Theta, alpha_Lambda

    def _m_step_g(self, alpha_G):
        """Maximize conditional distribution of core tensor.

        alpha_G: (K_M, K_N, K_P, K_S)
        """
        alpha_post = self.alpha + alpha_G
        # return tfd.Dirichlet(alpha_post).mode()
        G_star = alpha_post - 1.0
        G_star /= G_star.sum(axis=(2,3), keepdims=True)
        return G_star

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
        """Maximize conditional distribution of Theta factor.
        
        alpha_Theta: (K_P, P)
        """
        alpha_post = self.alpha + alpha_Theta
        return tfd.Dirichlet(alpha_post).mode()
    
    def _m_step_lambda(self, alpha_Lambda):
        """Maximize conditional distribution of Lambda factor.
        
        alpha_Lambda: (K_S, S)
        """
        alpha_post = self.alpha + alpha_Lambda
        return tfd.Dirichlet(alpha_post).mode()

    def m_step(self, alpha_G, alpha_Psi, alpha_Phi, alpha_Theta, alpha_Lambda):
        G = self._m_step_g(alpha_G)
        Psi = self._m_step_psi(alpha_Psi)
        Phi = self._m_step_phi(alpha_Phi)
        Theta = self._m_step_theta(alpha_Theta)
        Lambda = self._m_step_lambda(alpha_Lambda)
        return G, Psi, Phi, Theta, Lambda

    def heldout_log_likelihood(self, X, mask, params):
        r"""Compute the log likelihood of the held-out entries in X
        
        NOTE: Ignores the multinomial normalizing constant (C \choose x_1; x_2, ... x_N)
        """
        probs = jnp.einsum('ijkl,mi,nj,kp,ls->mnps', *params)
        return jnp.sum(X[~mask] * jnp.log(probs[~mask]))
        # M, N, P, S = X.shape
        # return jnp.where(~mask, jnp.sum(X * jnp.log(probs), axis=(2,3)), 0.0).sum()
        # return jnp.where(~mask, tfd.Multinomial(self.C, probs=probs.reshape(M, N, P*S))\
        #                  .log_prob(X.reshape(M, N, P*S)), 0.0).sum()

    def log_prob(self, X, mask, params):
        M, N, P, S = X.shape
        G, Psi, Phi, Theta, Lambda = params

        # log prior
        G_flat = G.reshape((self.K_M, self.K_N, self.K_P * self.K_S))
        lp = tfd.Dirichlet(self.alpha * jnp.ones(self.K_P * self.K_S)).log_prob(G_flat).sum()
        lp += tfd.Dirichlet(self.alpha * jnp.ones(self.K_M)).log_prob(Psi).sum()
        lp += tfd.Dirichlet(self.alpha * jnp.ones(self.K_N)).log_prob(Phi).sum()
        lp += tfd.Dirichlet(self.alpha * jnp.ones(P)).log_prob(Theta).sum()
        lp += tfd.Dirichlet(self.alpha * jnp.ones(S)).log_prob(Lambda).sum()

        # log likelihood of observed data
        probs = jnp.einsum('ijkl,mi,nj,kp,ls->mnps', *params)
        # lp += jnp.where(mask, tfd.Multinomial(self.C, probs=probs).log_prob(X), 0.0).sum()
        # lp += jnp.where(mask, jnp.sum(X * jnp.log(probs), axis=(2,3)), 0.0).sum()
        lp += jnp.sum(X[mask] * jnp.log(probs[mask]))
        return lp

    def reconstruct(self, params):
        return self.C * jnp.einsum('ijkl,mi,nj,kp,ls->mnps', *params)

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
        scale = X[mask].sum()
        for itr in trange(num_iters):
            lp, params = em_step(X, mask, params)
            lps.append(lp)
            if itr % 100 == 0:
                print("itr {:04d}: lp: {:.5f}".format(itr, lps[-1] / scale))

        return params, jnp.stack(lps)
