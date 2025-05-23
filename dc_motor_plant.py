# dc_motor_plant.py xx
import numpy as np

class DCMotorPlant:
    def __init__(self, x0, Q, R, dt=0.01):
        self.x = x0.copy()  # [theta, omega, b1, b2, b3]
        self.Q = Q
        self.R = R
        self.dt = dt

    def step(self, u):
        theta, omega, b1, b2, b3 = self.x
        omega_dot = b3 * u - b1 * omega - b2 * theta
        theta_dot = omega

        # Gürültü eklenmeden sistem durumu güncelleniyor
        self.x[0] += theta_dot * self.dt
        self.x[1] += omega_dot * self.dt

    def get_measurement(self):
        return self.x[0] + np.random.normal(0, np.sqrt(self.R))

    def get_true_kerem(self):
        return self.x.copy()
