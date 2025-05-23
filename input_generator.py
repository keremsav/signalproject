# input_generator.py
import numpy as np


def generate_prbs(length, amplitude=50, switch_interval=200):
    """
    PRBS sinyali 
    length: toplam adım sayısı
    amplitude: + veya - değer seviyesi
    switch_interval: kaç adımda bir değer değişsin
    """
    prbs = np.zeros(length)
    state = amplitude
    for i in range(length):
        if i % switch_interval == 0:
            state = amplitude if np.random.rand() > 0.5 else -amplitude
        prbs[i] = state
    return prbs


if __name__ == "__main__":
    total_time = 10  # saniye
    dt = 0.01        # zaman adımı
    steps = int(total_time / dt)

    u_array = generate_prbs(steps, amplitude=50, switch_interval=200)
    np.save("input_signal.npy", u_array)
    print("Giris sinyali 'input_signal.npy' dosyasina kaydedildi.")
