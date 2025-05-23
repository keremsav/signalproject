import numpy as np
import matplotlib.pyplot as plt

# .npy dosyalarını yükle
true_states = np.load("true_states.npy", allow_pickle=True)
kf_estimates = np.load("kf_estimates.npy", allow_pickle=True)
ukf_estimates = np.load("ukf_estimates.npy", allow_pickle=True)
ukfa_estimates = np.load("ukfa_estimates.npy", allow_pickle=True)
param_hist = np.load("param_hist.npy", allow_pickle=True)
rls_estimates = np.load("rls_estimates.npy", allow_pickle=True)

# Yardımcı fonksiyon: RMSE hesaplama
def compute_rmse(true_vals, est_vals):
    true_vals = np.array(true_vals)
    est_vals = np.array(est_vals)
    return np.sqrt(np.mean((true_vals - est_vals) ** 2))

# θ (pozisyon) için veriler
true_theta = [s[0] for s in true_states]
kf_theta = [s[0] for s in kf_estimates]
ukf_theta = [s[0] for s in ukf_estimates]
ukfa_theta = [s[0] for s in ukfa_estimates]
rls_theta = [s[0] for s in rls_estimates]

# θ için RMSE
rmse_kf = compute_rmse(true_theta, kf_theta)
rmse_ukf = compute_rmse(true_theta, ukf_theta)
rmse_ukfa = compute_rmse(true_theta, ukfa_theta)
rmse_rls = compute_rmse(true_theta, rls_theta)

# θ zaman serisi çizimi
plt.figure(figsize=(10, 5))
plt.plot(true_theta, label="Gerçek θ")
plt.plot(kf_theta, label="KF")
plt.plot(ukf_theta, label="UKF")
plt.plot(ukfa_theta, label="UKF-A")
plt.plot(rls_theta, label="RLS")
plt.xlabel("Zaman adımı")
plt.ylabel("θ (pozisyon)")
plt.title("Tüm Filtreler θ Karşılaştırması")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ω (hız) için veriler
true_omega = [s[1] for s in true_states]
kf_omega = [s[1] for s in kf_estimates]
ukf_omega = [s[1] for s in ukf_estimates]
ukfa_omega = [s[1] for s in ukfa_estimates]
# RLS hız tahmini yok, bu yüzden ω çizimi yok

# ω için RMSE
rmse_kf_omega = compute_rmse(true_omega, kf_omega)
rmse_ukf_omega = compute_rmse(true_omega, ukf_omega)
rmse_ukfa_omega = compute_rmse(true_omega, ukfa_omega)

# ω zaman serisi çizimi
plt.figure(figsize=(10, 5))
plt.plot(true_omega, label="Gerçek ω")
plt.plot(kf_omega, label="KF")
plt.plot(ukf_omega, label="UKF")
plt.plot(ukfa_omega, label="UKF-A")
plt.xlabel("Zaman adımı")
plt.ylabel("ω (hız)")
plt.title("Tüm Filtreler ω Karşılaştırması")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# RMSE Sonuçlarını Yazdır
print(f"KF RMSE (theta): {rmse_kf:.4f}")
print(f"UKF RMSE (theta): {rmse_ukf:.4f}")
print(f"UKF-A RMSE (theta): {rmse_ukfa:.4f}")
print(f"RLS RMSE (theta): {rmse_rls:.4f}")
print(f"KF RMSE (omega): {rmse_kf_omega:.4f}")
print(f"UKF RMSE (omega): {rmse_ukf_omega:.4f}")
print(f"UKF-A RMSE (omega): {rmse_ukfa_omega:.4f}")

# RMSE Bar Chart (theta ve omega)
plt.figure()
bar_width = 0.2
index = np.arange(4)
plt.bar(index, [rmse_kf, rmse_ukf, rmse_ukfa, rmse_rls], bar_width, label="Theta")
plt.bar(index + bar_width, [rmse_kf_omega, rmse_ukf_omega, rmse_ukfa_omega, 0], bar_width, label="Omega (RLS yok)")
plt.xticks(index + bar_width / 2, ["KF", "UKF", "UKF-A", "RLS"])
plt.ylabel("RMSE")
plt.title("Filtre Performans Karşılaştırması (RMSE)")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# UKF-A parametre tahmini grafiği
true_params = [10.0, 0.0, 1.0]  # C/J, K/J, Kv/J
titles = ["C/J", "K/J", "Kv/J"]
colors = ["blue", "orange", "green"]

plt.figure(figsize=(10, 5))
for i in range(3):
    plt.plot(param_hist[:, i], label=f"UKF-A Estimated {titles[i]}", color=colors[i])
    plt.axhline(true_params[i], linestyle="--", color="gray", label=f"True {titles[i]}" if i == 0 else None)
plt.xlabel("Time Step")
plt.title("UKF-A Parameter Estimation")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# RLS parametre tahmini grafiği
plt.figure(figsize=(10, 5))
for i in range(3):
    plt.plot(rls_estimates[:, i], label=f"RLS Estimated {titles[i]}", linestyle='--')
    plt.axhline(true_params[i], linestyle="--", color="gray", label=f"True {titles[i]}" if i == 0 else None)
plt.xlabel("Time Step")
plt.title("RLS Parameter Estimation")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
