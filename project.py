import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm      
from matplotlib.widgets import Slider
from scipy.signal import TransferFunction, bode, impulse, tf2zpk, lsim, square

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

def plot_all_pzmap(H1, H2, H3):
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    plt.sca(axs[0])
    plot_pzmap(H1, 'H1 Pole-Zero Map')
    plt.sca(axs[1])
    plot_pzmap(H2, 'H2 Pole-Zero Map')
    plt.sca(axs[2])
    plot_pzmap(H3, 'H3 Pole-Zero Map')
    
    plt.tight_layout()
    return fig


def plot_impulse(t, y, title):
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

def plot_all_impulse(H1, H2, H3):
    t1, y1 = impulse(H1)
    t2, y2 = impulse(H2)
    t3, y3 = impulse(H3)

    fig = plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plot_impulse(t1,y1, "Impulse Response - H1")

    plt.subplot(3, 1, 2)
    plot_impulse(t2,y2, "Impulse Response - H2")

    plt.subplot(3, 1, 3)
    plot_impulse(t3, y3, "Impulse Response - H3")
    plt.tight_layout()
    return fig


def plot_bode(w, magnitude, phase, title, index):
    plt.subplot(2, 3, index)

    plt.semilogx(w, magnitude)
    if index == 1: # 300 Hz point plot for LP
        closest_f_index = np.argmin(np.abs(w - 300))
        plt.plot(w[closest_f_index], magnitude[closest_f_index], ls="", marker="o", color="red")
        plt.annotate(
            f"{magnitude[closest_f_index]:.2f} dB at {np.round(w[closest_f_index])} Hz",                         
            xy=(w[closest_f_index], magnitude[closest_f_index]),             
            xytext=(50, -10),                                                
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            fontsize=9,
            color="black"
        )
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


def plot_all_bode(H1,H2,H3):
    fig = plt.figure(figsize=(15, 10))
    frequencies_hz = np.concatenate([
        np.logspace(0, 6, 1000),  
        [300]                      # ensure 300 hz
    ])
    frequencies_hz = np.unique(np.sort(frequencies_hz))  
    w = 2 * np.pi * frequencies_hz
    
    w1, mag1, phase1 = bode(H1, w=w)
    w1 /= 2*np.pi # Hz
    
    plot_bode(w1,mag1,phase1, 'H1', 1)

    w2, mag2, phase2 = bode(H2,  w=w)
    w2 /= 2*np.pi # Hz 
    plot_bode(w2,mag2,phase2,'H2', 2)

    w3, mag3, phase3 = bode(H3, w=w)
    w3 /= 2*np.pi # Hz
    plot_bode(w3,mag3,phase3, 'H3', 3)

    plt.tight_layout()


def define_H(R2, R3, K):
    # H3(s) = s^2 / (s^2 + R2*K*s/R3 + K^2)
    num_H3 = [1, 0, 0]  # s^2
    den_H3 = [1, R2*K/R3, K**2]
    H3 = TransferFunction(num_H3, den_H3)

    # H2(s) = (K/s) * H3(s)
    num_H2 = np.polymul([K], num_H3)
    den_H2 = np.polymul([1, 0], den_H3) 
    H2 = TransferFunction(num_H2, den_H2)

    # H1(s) = (K/s) * H2(s)
    num_H1 = np.polymul([K], num_H2)
    den_H1 = np.polymul([1, 0], den_H2)
    H1 = TransferFunction(num_H1, den_H1)
    return H1, H2, H3


f_rp = 300
omega_rp = 2*np.pi*f_rp
A_rp = np.sqrt(10)

# Define component values
R = 1
R3 = 1
C = 1
R4 = 1
R5 = 2000

# Define constants
G = R5 / R4
K = G / (R * C) #K between 1643 and 2280
R2 = R3/(1/K * omega_rp) * np.sqrt(1/A_rp**2 - (1 - 1/K**2 * omega_rp**2)**2)

H1, H2, H3 = define_H(R2, R3, K)

# make a figure containing the different changable variables
# when we update a variable clear old figures and update the plots

def redraw():
    H1, H2, H3 = define_H(R2, R3, K)
    plot_all_pzmap(H1, H2, H3)
    plot_all_impulse(H1, H2, H3)
    plot_all_bode(H1, H2, H3)

def rect():
    t = np.linspace(0,0.03,10000)
    y = square(t*2*np.pi*100)
    plt.plot(t, y, 'r', alpha=0.5, linewidth=1, label='input')
    tout, yout, xout = lsim(H1, T=t, U=y)
    plt.plot(tout, yout, 'k', linewidth=1.5, label='output')
    plt.legend()
redraw()
#rect()
plt.show()
