import numpy as np

class KalmanFilter(object):
    """
    This function uses the 'Kalman Filter' (KF) to compute the 
    estimation of the linear dynamical system given by:
    
            x_k = A_km1*x_km1 + B_km1*x_km1 + q_km1        (1)
                    y_k = H_k*x_k     + r_k        (2)
    
    The algorithm is taken from Theorem 4.2 in (Sarkka, S).
    
    Input data:
    
    - A_km1 : Transition matrix of the dynamic model
    - B_km1 : Transition matrix of the dynamic input
    - H_k   : Measurement model matrix
    - m_0   : Initial state of the dynamical system (L x 1 vector)(L states)
    - P_0   : Error covariance matrix of the state x_0 (L x L matrix)
    - y     : Measurement matrix (Ly x N matrix)
                (Ly observations - N time steps)
    - u     : Optional control input (N x g vector) 
                (N time steps - g control inputs)
    - Q     : Process-noise discrete covariance (L x L matrix)
    - R     : Measurement-noise discrete covariance (Ly x Ly matrix)
    
    Output data:
    
    - x_k : State estimate observational update (LxN matrix). Every 
            column is the estimation at every time step.
    - P_k : Steady-state covariance matrix (1xN cell). Every component 
            of the cell is a covariance matrix at every time step.
    
    Notes:
    
    -   Note that 'y' is an "Ly x N" matrix. It means that the number of 
        columns is the number of states to estimate 'k'; the number of 
        rows is the number of measured variables at time 'k'.
    -   It is assumed that "q_k" and "r_k" have mean zero, and that they 
        are uncorrelated (E[v_j n_k] = 0).
    
    Bibliography:
    
    - Sarkka, S. (2013). Bayesian filtering and smoothing (Vol. 3). 
        Cambridge University Press.
    
    -------------------------------------------------------
    | Developed by:   Sebastian Jaramillo Moreno          |
    |                 sejaramillomo@unal.edu.co           |
    |                 National University of Colombia     |
    |                 Manizales, Colombia.                |
    -------------------------------------------------------
    """
    def __init__(self, A_km1 = None, B_km1 = None, H_k = None, x_k = None, P_k = None, Q = None, R = None):
        if(A_km1 is None or H_k is None):
            raise ValueError("Set proper system dynamics.")
        
        N = A_km1.shape[1]

        self.A_km1 = A_km1
        self.H_k = H_k
        self.B_km1 = 0 if B_km1 is None else B_km1
        self.Q = np.eye(N) if Q is None else Q
        self.R = np.eye(N) if R is None else R
        self.x_k = np.zeros(N) if x_k is None else x_k
        self.P_k = np.eye(N) if P_k is None else P_k

    def prediction(self, u = 0):    # the prediction step is (equation 4.20)
        self.x_k = np.dot(self.A_km1, self.x_k) + np.dot(self.B_km1, u)
        self.P_k = np.dot(self.A_km1, np.dot(self.P_k, self.A_km1.T)) + self.Q
        return self.x_k, self.P_k
    
    def update(self, y):            # the update step is (equation 4.21)
        v_k = y - np.dot(self.H_k, self.x_k)
        S_k = np.dot(self.H_k, np.dot(self.P_k, self.H_k.T)) + self.R
        K_k = np.dot(self.P_k, np.dot(self.H_k.T, np.linalg.inv(S_k)))
        self.x_k = self.x_k + np.dot(K_k, v_k)
        self.P_k = self.P_k - np.dot(K_k, np.dot(S_k, K_k.T))
        return self.x_k, self.P_k