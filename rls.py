import numpy as np

class RecursiveLeastSquares:
    def __init__(self, n_params=3, forgetting_factor=0.99, delta=1e3):
        self.n = n_params
        self.lambda_ = forgetting_factor
        self.P = delta * np.eye(n_params)
        self.theta = np.zeros((n_params, 1))

    def step(self, u, meas, omega_f_prev):
        # Kesinlikle skaler hale getir
        u = float(np.squeeze(u))
        meas = float(np.squeeze(meas))

        phi = np.array([[-omega_f_prev], [-meas], [u]])

        P_phi = self.P @ phi
        gain_denominator = self.lambda_ + phi.T @ P_phi
        K = P_phi / gain_denominator

        error = meas - phi.T @ self.theta
        self.theta = self.theta + K * error
        self.P = (self.P - K @ phi.T @ self.P) / self.lambda_

        omega_f = omega_f_prev + K[0,0] * error

        return self.theta.flatten(), omega_f
