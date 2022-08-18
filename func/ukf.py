import numpy as np
from .runge_kutta import rk_discrete

class UnscentedKalmanFilter(object):
    """
    This function uses the 'Unscented Kalman Filter' (UKF) to compute 
    the estimation of the nonlinear dynamical system given by:
    
                x_{k+1} = F(x_k, u_k) + v_k        (1)
                y_k     = H(x_k)      + n_k        (2)
    
    The algorithm is taken from Table 7.3 in (Wan, Van der Merwe).
    
    Input data:
    
    - F   : Lambda function in (1) (L x 1 vector) 
            (L process equations)
    - H   : Lambda function in (2) (Ly x 1 vector) 
            (Ly observation equations)
    - x_0 : Initial state of the dynamical system (L x 1 vector)(L states)
    - P_0 : Error covariance matrix of the state x_0 (L x L matrix)
    - y   : Measurement matrix (Ly x N matrix)
            (Ly observations - N time steps)
    - u   : Optional control input (N x g vector) 
            (N time steps - g control inputs)
    - Rv  : Process-noise discrete covariance (L x L matrix)
    - Rn  : Measurement-noise discrete covariance (Ly x Ly matrix)
    - dt  : Time between measurements

    Output data:

    - x_k : State estimate observational update (LxN matrix). Every 
            column is the estimation at every time step.
    - P_k : Steady-state covariance matrix (1xN cell). Every component 
            of the cell is a covariance matrix at every time step.

    Notes:

    - Note that 'y' is an "Ly x N" matrix. It means that the number of 
    columns is the number of states to estimate 'k'; the number of 
    rows is the number of measured variables at time 'k'.
    - It is assumed that "v_k" and "n_k" have mean zero, and that they 
    are uncorrelated (E[v_j n_k] = 0).

    Bibliography:

    - WAN, Eric A., VAN DER MERWE, Rudolph. "The Unscented Kalman Filter".
    In: HAYKIN, Simon. "Kalman filtering and neural networks". John 
    Wiley & Sons Inc. First edition. 2001. Ontario, Canada.
    Available in:
    https://bit.ly/3wd6UYK

    -------------------------------------------------------
    | Developed by:   Sebastian Jaramillo Moreno          |
    |                 sejaramillomo@gmail.com             |
    |                 Manizales, Colombia.                |
    -------------------------------------------------------
    """
    def __init__(self, F = None, H = None, x_0 = None, P_0 = None, Rv = None, \
                Rn = None, dt = None):
        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.L = len(x_0)
        self.Ly = 1 if Rn is None else Rn.shape[0]
        
        self.F = F
        self.H = H
        self.dt = dt
        self.Rv = np.eye(self.L) if Rv is None else Rv
        self.Rn = np.eye(self.Ly) if Rn is None else Rn

        # filter parameters
        self.alpha = 0.001                      # \in [1e-4, 1]
        self.kappa = 3 - self.L                 # 0 or 3-L
        self.beta = 2                           # 2 for gaussian distributions

        self.lam = (self.alpha**2)*(self.L + self.kappa) - self.L #scaling parameter
        self.gamma = np.sqrt(self.L + self.lam) # composite scaling parameter

        # mean weights (eq. 7.34)
        self.Wm = np.zeros(2*self.L + 1)
        self.Wm[0] = self.lam/(self.L + self.lam)
        self.Wm[1:] = 1/(2*(self.L + self.lam))

        # covariance weights (eq. 7.34)
        self.Wc = np.copy(self.Wm)
        self.Wc[0] = (self.lam/(self.L + self.lam)) + (1 - self.alpha**2 + self.beta)

        # data initialization
        self.x_k = np.zeros(self.L) if x_0 is None else x_0      # (eq 7.35)
        self.P_k = np.eye(self.L) if P_0 is None else P_0
    
    def prediction(self, u = 0):
        ## Calculate sigma points
        # (eq. 7.52)
        sqrt_Pkm1 = np.linalg.cholesky(self.P_k)

        Xkm1 = np.vstack((self.x_k, np.vstack(( \
            np.tile(self.x_k, (self.L, 1)) + self.gamma*sqrt_Pkm1,
            np.tile(self.x_k, (self.L, 1)) - self.gamma*sqrt_Pkm1))))
        
        ## Time update
        # (eq. 7.53)
        X_k_km1_ast = np.zeros((2*self.L+1, self.L))
        for i in range(0, 2*self.L+1):
            X_k_km1_ast[i, :] = rk_discrete(self.F, Xkm1[i, :], u, self.dt)
        
        # (eq. 7.54)
        self.x_k = np.dot(X_k_km1_ast.T, self.Wm)
        
        # (eq. 7.55)
        self.P_k = self.Rv
        for i in range(0, 2*self.L+1):
            tmp = X_k_km1_ast[i, :] - self.x_k
            tmp = tmp.reshape(self.L, 1)
            self.P_k = self.P_k + self.Wc[i]*(np.dot(tmp, tmp.T))
        
        return self.x_k, self.P_k
    
    def update(self, y):
        ## Redraw sigma points
        sqrt_Pkb = np.linalg.cholesky(self.P_k)
        
        # (eq. 7.56)
        # here the method explained in the footnote is used:
        # "Here we augment the sigma points with additional points derived from
        # the matrix square root of the process noise covariance. This requires
        # setting L -> 2L and recalculating the various weights Wi accordingly.
        # Alternatively, we may redraw a complete new set of sigma points, 
        # i.e.,
        # Xk_km1 = [xh_kb xh_kb + gamma*sqrt_Pkb xh_kb - gamma*sqrt_Pkb]. 
        # This alternative approach results in fewer sigma points being used, 
        # but also discards any odd-moments information captured by the 
        # original propagated sigma points"
        
        Xk_km1 = np.vstack((self.x_k, np.vstack(( \
            np.tile(self.x_k, (self.L, 1)) + self.gamma*sqrt_Pkb,
            np.tile(self.x_k, (self.L, 1)) - self.gamma*sqrt_Pkb))))
        
        Yk_km1 = self.H(Xk_km1.T).T
        
        # (eq. 7.57)
        yh_kb = np.dot(Yk_km1.T, self.Wm).squeeze()
        
        ## Measurement-update
        # (eq. 7.58)
        P_yhyh = self.Rn
        for i in range(0, 2*self.L+1):
            tmp = Yk_km1[i, :] - yh_kb
            tmp = tmp.reshape(self.Ly, 1)
            P_yhyh = P_yhyh + self.Wc[i]*(np.dot(tmp, tmp.T))
        
        # (eq. 7.59)
        P_xy = np.zeros((self.L, self.Ly))
        for i in range(0, 2*self.L+1):
            tmp1 = Xk_km1[i, :] - self.x_k
            tmp1 = tmp1.reshape(self.L, 1)
            tmp2 = Yk_km1[i, :] - yh_kb
            tmp2 = tmp2.reshape(self.Ly, 1)
            P_xy = P_xy + self.Wc[i]*(np.dot(tmp1, tmp2.T))
        
        # (eq. 7.60)
        K_k = np.dot(P_xy, np.linalg.inv(P_yhyh))
        
        # (eq. 7.61)
        self.x_k = self.x_k + (K_k*(y - yh_kb)).squeeze()
        
        # (eq. 7.62)
        self.P_k = self.P_k - np.dot(K_k, np.dot(P_yhyh, K_k.T))
        
        return self.x_k, self.P_k