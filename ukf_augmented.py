# ukf_augmented.py
import numpy as np
from scipy.linalg import cholesky

class UnscentedKalmanFilter:
    def __init__(self, f_func, h_func, Q, R, x0, P0, alpha=1e-3, beta=2, kappa=0):
        self.f = f_func
        self.h = h_func
        self.Q = Q
        self.R = R
        self.x = x0.copy()
        self.P = P0.copy()
        self.n = len(x0)

        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)

        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    def _sigma_points(self, x, P):
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = x
        sqrt_P = cholesky((self.n + self.lambda_) * P)
        for i in range(self.n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[self.n + i + 1] = x - sqrt_P[i]
        return sigma_points

    def predict(self, u):
        X = self._sigma_points(self.x, self.P)
        X_pred = np.array([self.f(xi, u) for xi in X])
        self.x = np.sum(self.Wm[:, None] * X_pred, axis=0)
        dx = X_pred - self.x
        self.P = dx.T @ (self.Wc[:, None] * dx) + self.Q
        self.X_pred = X_pred

    def update(self, y):
        Z = np.array([self.h(xi) for xi in self.X_pred])
        z_pred = np.sum(self.Wm * Z)
        dz = Z - z_pred
        Pxz = np.sum(self.Wc[:, None] * (self.X_pred - self.x) * dz[:, None], axis=0)
        Pzz = np.sum(self.Wc * dz**2) + self.R
        K = Pxz / Pzz
        self.x = self.x + K * (y - z_pred)
        self.P = self.P - np.outer(K, K) * Pzz

    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x

# UKF-A sistem fonksiyonu (parametre tahmini için 5D durum)
def f_aug(x, u):
    theta, omega, b1, b2, b3 = x
    dt = 0.01
    omega_dot = b3 * u - b1 * omega - b2 * theta
    theta_dot = omega
    return np.array([
        theta + theta_dot * dt,
        omega + omega_dot * dt,
        b1,  # sabit parametreler
        b2,
        b3
    ])

# Ölçüm fonksiyonu: sadece pozisyon ölçülüyor
def h_aug(x):
    return x[0]