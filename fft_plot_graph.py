import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq  
from scipy.signal import find_peaks    
from scipy.fft import rfft, rfftfreq

path = "https://raw.githubusercontent.com/mmozumder467/Signal-Detection-/refs/heads/main/mystery_signal_data1.csv"

data = np.genfromtxt(path, delimiter=',')
time_ms = data[:, 0]
signal_adc = data[:, 1]

time_s = time_ms / 1_000.0

print(f"Loaded {len(time_s)} data points.")
print(f"Total duration: {time_s[-1]:.2f} seconds.")


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_s, signal_adc)
ax.set_title('Raw Signal vs. Time')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Signal (ADC Value)')
ax.grid(True)
plt.show()

dt = np.mean(np.diff(time_s))

N = len(signal_adc)

print(f"Calculated sample interval (dt): {dt:.8f} seconds")

yf = rfft(signal_adc)
xf = rfftfreq(N, dt) 
fig, ax = plt.subplots()
ax.plot(xf, np.abs(yf),color='blue')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude (Arbitrary Units)')
ax.set_title('FFT Spectrum')
ax.set_xlim(0,15)
ax.set_ylim(0,np.max(np.abs(yf)[xf > 0.5]) * 1.1)
ax.grid()
plt.show()



peaks, _ = find_peaks(np.abs(yf), height=100000) 
print(f"Found {len(peaks)} peaks at frequencies (Hz):")
print(xf[peaks])

fig, ax = plt.subplots()
ax.plot(xf, np.abs(yf),color='blue')
ax.plot(xf[peaks[0]],np.abs(yf[peaks[0]]) ,"x",color='red')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude (Arbitrary Units)')
ax.set_title('FFT Spectrum with Peaks')
ax.set_xlim(0,15)
ax.set_ylim(0, np.max(np.abs(yf)[xf > 0.5]) * 1.1)
ax.grid()
plt.show()