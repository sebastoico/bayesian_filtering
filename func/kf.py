import numpy as np

class KalmanFilter(object):
    def __init__(self, A_km1 = None, B_km1 = None, H_k = None, x_k = None, P_k = None, Q = None, R = None):
        if(A_km1 is None or H_k is None):
            raise ValueError("Set proper system dynamics.")
        
        self.N = A_km1.shape[1]

        self.A_km1 = A_km1
        self.H_k = H_k
        self.B_km1 = 0 if B_km1 is None else B_km1
        self.Q = np.eye(self.N) if Q is None else Q
        self.R = np.eye(self.N) if R is None else R
        self.x_k = np.zeros((self.N, 1)) if x_k is None else x_k
        self.P_k = np.eye(self.N) if P_k is None else P_k

    def prediction(self, u = 0):
        self.x_k = np.dot(self.A_km1, self.x_k) + np.dot(self.B_km1, u)
        self.P_k = np.dot(self.A_km1, self.P_k) + self.Q
        return self.x_k, self.P_k
    
    def update(self, y):
        v_k = y - np.dot(self.H_k, self.x_k)
        S_k = np.dot(np.dot(self.H_k, self.P_k), self.H_k.T) + self.R
        K_k = np.dot(np.dot(self.P_k, self.H_k.T), np.linalg.pinv(S_k))
        self.x_k = self.x_k + np.dot(K_k, v_k)
        self.P_k = self.P_k - np.dot(np.dot(K_k, S_k), K_k.T)
        return self.x_k, self.P_k