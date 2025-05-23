# kalman_filter.py
import numpy as np

class KalmanFilter:
    def __init__(self, A, B, C, Q, R, P0, x0):
        self.A = A      # Sistem geçiş matrisi
        self.B = B      # Giriş matrisi
        self.C = C      # Ölçüm matrisi
        self.Q = Q      # Süreç gürültüsü kovaryans matrisi
        self.R = R      # Ölçüm gürültüsü kovaryans matrisi
        self.P = P0     # Başlangıç kovaryans
        self.x = x0     # Başlangıç durumu

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        y_pred = self.C @ self.x
        self.x = self.x + K @ (y - y_pred)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.C) @ self.P

    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x.copy()