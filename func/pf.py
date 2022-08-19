import numpy as np
from scipy.stats import norm

class ParticleFilter(object):
    """
    This function uses the 'Generic Particle Filter' (GPF) to compute 
    the estimation of the nonlinear dynamical system given by:
    
                x_k = F(x_{k-1}, u_k, v_{k-1})	(1)
                y_k = H(x_k, n_k)               (2)
    
    The algorithm is taken from Algorithm 3 in (Arulampalam et al).
    
    Input data:
    
    - F    : Function handle in (1) (L x 1 vector) 
            (L process equations)
    - H    : Function handle in (2) (Ly x 1 vector) 
            (Lz observation equations)
    - x_k0 : Initial particles of the dynamical system (Ns x L matrix)
    - w_k0 : Initial weigths of the dynamical system (Ns x L matrix)
    - z_k  : Measurement vector (Lz x N matrix)
            (Ly observations - N time steps)
    - u_k  : Optional control input (N x g vector) 
            (N time steps - g control inputs)
    - Rv   : Process-noise discrete covariance (L x L matrix)
    - Rn   : Measurement-noise discrete covariance (Lz x Lz matrix)
    - Nt   : Limit number of effective particles.
    
    Output data:
    
    - x_k : State estimate observational update (NsxLxN array).
    - w_k : Weigth estimate observational update (NsxLxN array).
    
    Notes:
    
    - Note that 'zk' is an "Lz x N" matrix. It means that the number of 
    columns is the number of states to estimate 'k'; the number of 
    rows is the number of measured variables at time 'k'.
    - It is assumed that "v_k" and "n_k" have mean zero, and that they 
    are uncorrelated (E[v_j n_k] = 0).
    - It is assumed that q_xk_given_xkm1_yk = p_xk_given_xkm1, like in
    the equation (62)
    
    Bibliography:
    
    - Arulampalam, M. S., Maskell, S., Gordon, N., & Clapp, T. (2002). 
    "A tutorial on particle filters for online nonlinear/non-Gaussian 
    Bayesian tracking". IEEE Transactions on signal processing, 
    50(2), 174-188.
    
    -------------------------------------------------------
    | Developed by:   Sebastian Jaramillo Moreno          |
    |                 sejaramillomo@unal.edu.co           |
    |                 National University of Colombia     |
    |                 Manizales, Colombia.                |
    -------------------------------------------------------
    """
    def __init__(self, F = None, H = None, x_0 = None, P_0 = None, \
        p_0 = None, w_0 = None, Rv = None, Rn = None, Ns = None):
        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")
        
        self.Ns = 1000 if Ns is None else Ns
        self.L = p_0.shape[1] if len(p_0.shape) != 1 else 1

        self.F = F
        self.H = H

        self.Rv = np.eye(self.L) if Rv is None else Rv
        self.Rn = Rn

        self.x_k = np.zeros(self.L) if x_0 is None else x_0
        self.p_k = np.random.multivariate_normal(x_0, P_0, self.Ns) \
            if p_0 is None else p_0
        self.w_k = 1/Ns*np.ones(self.Ns) if w_0 is None else w_0

        # likelihood function
        self.p_zk_given_xk = lambda y, xk: norm.pdf(self.H(xk, 0), \
            loc=y, scale=self.Rn)
    
    def prediction(self, u = 0, y = 0):
        for i in range(0, self.Ns):
            # draw the particles
            self.p_k[i, :] = self.F(self.p_k[i, :], u, \
                np.dot(self.Rv, np.random.randn(self.L)))
            
            # assign the particle a weight, w_k**i, according to (63)
            self.w_k[i] = self.w_k[i]*np.sum(np.diag( \
                self.p_zk_given_xk(y, self.p_k[i, :])))

        # calculate total weight
        t = np.sum(self.w_k)
        # normalize w_k**i = w_k**i/t
        self.w_k = self.w_k/t

        # calculate neff_g using (51)
        neff_g = 1/np.sum(np.power(self.w_k, 2))

        if neff_g < 0.5*self.Ns:
            self.resample()

        self.x_k = np.dot(self.p_k.T, self.w_k)
    
    def resample(self):
        idx = np.random.choice(self.Ns, self.Ns, True, self.w_k)
        self.p_k = self.p_k[idx, :]
        self.w_k = np.ones_like(self.w_k)*(1/self.Ns)