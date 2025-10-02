import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, bode, impulse, tf2zpk

f_rp = 300
A_rp = np.sqrt(10)

# Define component values
R = 1
R3 = 1
C = 1
R4 = 3000
R5 = 1

# Define constants
G = R5 / R4
#K = G / (R * C)
K = 1/(2*np.pi*f_rp)
print(A_rp**-2 - (1 - K**2 * 4*np.pi**2 * f_rp**2)**2)
R2 = R3/(K * 2 * np.pi * f_rp) * np.sqrt(A_rp**-2 - (1 - K**2 * 4*np.pi**2 * f_rp**2)**2)

# Define transfer functions using scipy.signal.TransferFunction (numerator, denominator)
# H3(s) = s^2 / (s^2 + R2*K*s/R3 + K)
num_H3 = [1, 0, 0]  # s^2
den_H3 = [1, R2*K/R3, K**2]
H3 = TransferFunction(num_H3, den_H3)

# H2(s) = (K/s) * H3(s)
# Multiply H3 by K/s -> Equivalent to multiplying by K and convolving with 1/s
num_H2 = np.polymul([K], num_H3)
den_H2 = np.polymul([1, 0], den_H3)  # Multiply denominator by s
H2 = TransferFunction(num_H2, den_H2)

# H1(s) = (K/s) * H2(s)
num_H1 = np.polymul([K], num_H2)
den_H1 = np.polymul([1, 0], den_H2)
H1 = TransferFunction(num_H1, den_H1)

# Function to plot pole-zero map
def plot_pzmap(tf, title):
    zeros, poles, _ = tf2zpk(tf.num, tf.den)
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', label='Zeros')
    plt.scatter(np.real(poles), np.imag(poles), marker='x', label='Poles')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.title(title)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True)
    plt.legend()

def plot_impulse(t, y, title):
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

def plot_bode(w, magnitude, phase, title, index):
    plt.subplot(2, 3, index)

    plt.semilogx(w, magnitude)
    plt.title(f"{title} - Magnitude")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)

    plt.subplot(2, 3, index+3)

    plt.semilogx(w,phase)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [°]")
    plt.title(f"{title} - Phase")
    plt.grid(True)

# --- Plot Pole-Zero Maps ---
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plot_pzmap(H1, 'H1 Pole-Zero Map')

plt.subplot(3, 1, 2)
plot_pzmap(H2, 'H2 Pole-Zero Map')

plt.subplot(3, 1, 3)
plot_pzmap(H3, 'H3 Pole-Zero Map')

plt.tight_layout()

# --- Impulse Responses ---
t1, y1 = impulse(H1)
t2, y2 = impulse(H2)
t3, y3 = impulse(H3)

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plot_impulse(t1,y1, "Impulse Response - H1")

plt.subplot(3, 1, 2)
plot_impulse(t2,y2, "Impulse Response - H2")

plt.subplot(3, 1, 3)
plot_impulse(t3, y3, "Impulse Response - H3")
plt.tight_layout()

# --- Bode Plots ---
plt.figure(figsize=(15, 5))

w1, mag1, phase1 = bode(H1)
w1 /= 2*np.pi # Hz
plot_bode(w1,mag1,phase1, 'H1', 1)

w2, mag2, phase2 = bode(H2)
w2 /= 2*np.pi # Hz 
plot_bode(w2,mag2,phase2,'H2', 2)

w3, mag3, phase3 = bode(H3)
w3 /= 2*np.pi # Hz
plot_bode(w3,mag3,phase3, 'H3', 3)

plt.tight_layout()
plt.show()
