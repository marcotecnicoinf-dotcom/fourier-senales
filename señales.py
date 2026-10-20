import numpy as np
import matplotlib.pyplot as plt

# --- Definición de la señal en el dominio del tiempo ---
fs = 1000                       # muestras por segundo
t = np.arange(0, 1, 1/fs)       # tiempo de 0 a 1 segundo
senoidal = np.sin(2 * np.pi * 5 * t)   # senoidal de 5 Hz

# --- Visualización ---
plt.plot(t, senoidal)
plt.title("Señal senoidal de 5 Hz")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
# --- Cálculo de la Transformada de Fourier ---
N = len(senoidal)                       # número de muestras
X = np.fft.fft(senoidal)                # transformada
frecuencias = np.fft.fftfreq(N, 1/fs)   # eje de frecuencias en Hz

magnitud = np.abs(X) / N                # magnitud (normalizada)
fase = np.angle(X)                      # fase en radianes

# --- Visualización del espectro ---
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(frecuencias[:N//2], magnitud[:N//2])
plt.title("Magnitud del espectro")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.xlim(0, 20)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(frecuencias[:N//2], fase[:N//2])
plt.title("Fase del espectro")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Fase (rad)")
plt.xlim(0, 20)
plt.grid(True)

plt.tight_layout()
plt.show()
# --- Pulso rectangular y escalón en el dominio del tiempo ---
pulso = np.where((t >= 0.4) & (t < 0.6), 1.0, 0.0)   # vale 1 entre 0.4 y 0.6 s
escalon = np.where(t >= 0.5, 1.0, 0.0)               # vale 0 antes de 0.5 s y 1 después

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, pulso)
plt.title("Pulso rectangular")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, escalon)
plt.title("Función escalón")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Transformada de Fourier de ambas señales ---
X_pulso = np.fft.fft(pulso)
X_escalon = np.fft.fft(escalon)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(frecuencias[:N//2], np.abs(X_pulso[:N//2]) / N)
plt.title("Magnitud del espectro: pulso rectangular")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.xlim(0, 50)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(frecuencias[:N//2], np.abs(X_escalon[:N//2]) / N)
plt.title("Magnitud del espectro: escalón")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.xlim(0, 50)
plt.grid(True)

plt.tight_layout()
plt.show()
# =====================================================
# PROPIEDAD 1: LINEALIDAD
# FFT(a*x1 + b*x2) = a*FFT(x1) + b*FFT(x2)
# =====================================================
a, b = 2, 3
lado_izq = np.fft.fft(a * senoidal + b * pulso)
lado_der = a * np.fft.fft(senoidal) + b * np.fft.fft(pulso)

print("Linealidad se cumple:", np.allclose(lado_izq, lado_der))

plt.figure(figsize=(10, 4))
plt.plot(frecuencias[:N//2], np.abs(lado_izq[:N//2]) / N, label="FFT(a·x1 + b·x2)")
plt.plot(frecuencias[:N//2], np.abs(lado_der[:N//2]) / N, "--", label="a·FFT(x1) + b·FFT(x2)")
plt.title("Propiedad de linealidad")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.xlim(0, 50)
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# PROPIEDAD 2: DESPLAZAMIENTO EN EL TIEMPO
# Retrasar la señal NO cambia la magnitud, solo la fase
# =====================================================
pulso_desplazado = np.where((t >= 0.6) & (t < 0.8), 1.0, 0.0)  # mismo pulso, 0.2 s después

X_orig = np.fft.fft(pulso)
X_desp = np.fft.fft(pulso_desplazado)

print("Magnitud igual tras desplazar:", np.allclose(np.abs(X_orig), np.abs(X_desp)))

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, pulso, label="Original")
plt.plot(t, pulso_desplazado, label="Desplazado 0.2 s")
plt.title("Pulso original y desplazado (tiempo)")
plt.xlabel("Tiempo (s)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(frecuencias[:N//2], np.abs(X_orig[:N//2]) / N, label="Original")
plt.plot(frecuencias[:N//2], np.abs(X_desp[:N//2]) / N, "--", label="Desplazado")
plt.title("Magnitud (idéntica)")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, 50)
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(frecuencias[:N//2], np.angle(X_orig[:N//2]), label="Original")
plt.plot(frecuencias[:N//2], np.angle(X_desp[:N//2]), label="Desplazado")
plt.title("Fase (distinta)")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, 50)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =====================================================
# PROPIEDAD 3: ESCALAMIENTO
# Comprimir en el tiempo = ensanchar en frecuencia
# =====================================================
pulso_ancho = np.where((t >= 0.4) & (t < 0.6), 1.0, 0.0)      # dura 0.2 s
pulso_angosto = np.where((t >= 0.45) & (t < 0.55), 1.0, 0.0)  # dura 0.1 s (comprimido a la mitad)

X_ancho = np.abs(np.fft.fft(pulso_ancho)) / N
X_angosto = np.abs(np.fft.fft(pulso_angosto)) / N

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t, pulso_ancho, label="Ancho (0.2 s)")
plt.plot(t, pulso_angosto, label="Angosto (0.1 s)")
plt.title("Pulsos en el tiempo")
plt.xlabel("Tiempo (s)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(frecuencias[:N//2], X_ancho[:N//2], label="Espectro del ancho")
plt.plot(frecuencias[:N//2], X_angosto[:N//2], label="Espectro del angosto")
plt.title("Espectros: el angosto se ensancha")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, 50)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
