import numpy as np
import matplotlib.pyplot as plt

from dc_motor_plant import DCMotorPlant
from kalman_filter import KalmanFilter
from ukf import UnscentedKalmanFilter as UKF, f_func, h_func
from ukf_augmented import UnscentedKalmanFilter as UKFA, f_aug, h_aug
from rls import RecursiveLeastSquares

# Giriş sinyalini yükle
u_array = np.load("input_signal.npy")

# Simülasyon parametreleri
dt = 0.01
steps = len(u_array)

# Gürültü ve başlangıç parametreleri
x0_kf = np.array([[2.0], [0.0]])
P0_kf = np.eye(2)
Q_kf = 1e-1 * np.eye(2)
R_kf = np.array([[1]])

x0_ukf = np.array([2.0, 0.0])
P0_ukf = np.eye(2)
Q_ukf = np.diag([1e-1, 1e-1])
R_ukf = 1.0

x0_ukfa = np.array([2.0, 0.0, 10.0, 0.0, 1.0])
P0_ukfa = np.diag([0.5, 0.5, 10.0, 10.0, 10.0])
Q_ukfa = np.diag([1e-1, 1e-1, 1.0, 1.0, 1.0])
R_ukfa = 1.0

true_params = np.array([10.0, 0.0, 1.0])
plant = DCMotorPlant(np.concatenate(([2.0, 0.0], true_params)), Q_ukfa, R_ukfa, dt)

# Filtreleri oluştur
A = np.array([[1, dt], [-(0.01/0.01)*dt, 1 - (0.1/0.01)*dt]])
B = np.array([[0], [(0.01/0.01)*dt]])
C = np.array([[1, 0]])

kf = KalmanFilter(A, B, C, Q_kf, R_kf, P0_kf, x0_kf)
ukf = UKF(f_func, h_func, Q_ukf, R_ukf, x0_ukf, P0_ukf)
ukfa = UKFA(f_aug, h_aug, Q_ukfa, R_ukfa, x0_ukfa, P0_ukfa)

# RLS tanımı
rls = RecursiveLeastSquares(n_params=3, forgetting_factor=0.99, delta=1e3)
omega_f = 0.0  # Başlangıç için filtrelenmiş omega

# Örnekleme oranları
kf_rate = 1
ukf_rate = 1
ukfa_rate = 1
rls_rate = 1

# Kayıt dizileri
true_states, measurements = [], []
kf_estimates, ukf_estimates, ukfa_estimates, rls_estimates = [], [], [], []
param_hist = []

for i in range(steps):
    u = u_array[i]
    plant.step(u)
    y = plant.get_measurement()
    true_state = plant.get_true_state()

    true_states.append(true_state)
    measurements.append(y)

    # KF
    if i % kf_rate == 0:
        kf_est = kf.step(np.array([[u]]), np.array([[y]]))
    kf_estimates.append(kf_est.flatten())

    # UKF
    if i % ukf_rate == 0:
        ukf_est = ukf.step(u, y)
    ukf_estimates.append(ukf_est)

    # UKF-A
    if i % ukfa_rate == 0:
        ukfa_est = ukfa.step(u, y)
    ukfa_estimates.append(ukfa_est)
    param_hist.append(ukfa_est[2:].copy())

    # RLS
    if i % rls_rate == 0:
        rls_theta, omega_f = rls.step(float(np.squeeze(u)), float(np.squeeze(y)), omega_f)
    rls_estimates.append(rls_theta)

# Verileri kaydet
np.save("true_states.npy", true_states)
np.save("kf_estimates.npy", kf_estimates)
np.save("ukf_estimates.npy", ukf_estimates)
np.save("ukfa_estimates.npy", ukfa_estimates)
np.save("param_hist.npy", np.array(param_hist))
np.save("rls_estimates.npy", rls_estimates)

print("Simülasyon tamamlandı ve veriler kaydedildi.")
