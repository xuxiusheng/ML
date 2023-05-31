from scipy import fft
from scipy.io import wavfile
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt

(sample_rate, X) = wavfile.read(r'genres\blues\converted\blues.00000.au.wav')
print(sample_rate, X.shape)

plt.figure()
ax = plt.subplot(1, 1, 1)
ax.set_xlabel('time')
ax.set_ylabel('frequency')
plt.grid(True, linestyle='-', color='0.75')
specgram(X, Fs=sample_rate, xextent=(0, 30))
plt.show()

def plotSpec(g, n):
    sample_rate, X = wavfile.read('genres/' + g + '/converted/' + g + '.' + n + '.au.wav')
    specgram(X, Fs=sample_rate, xextent=(0, 30))
    plt.title(g + '-' + n[-1])

plt.figure(num=None, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot(6, 3, 1)
plotSpec('classical', '00001')

plt.subplot(6, 3, 2); plotSpec('classical', '00002')
plt.subplot(6, 3, 3); plotSpec('classical', '00003')
plt.subplot(6, 3, 4); plotSpec('jazz', '00001')
plt.subplot(6, 3, 5); plotSpec('jazz', '00002')
plt.subplot(6, 3, 6); plotSpec('jazz', '00003')
plt.subplot(6, 3, 7); plotSpec('country', '00001')
plt.subplot(6, 3, 8); plotSpec('country', '00002')
plt.subplot(6, 3, 9); plotSpec('country', '00003')
plt.subplot(6, 3, 10); plotSpec('pop', '00001')
plt.subplot(6, 3, 11); plotSpec('pop', '00002')
plt.subplot(6, 3, 12); plotSpec('pop', '00003')
plt.subplot(6, 3, 13); plotSpec('rock', '00001')
plt.subplot(6, 3, 14); plotSpec('rock', '00002')
plt.subplot(6, 3, 15); plotSpec('rock', '00003')
plt.subplot(6, 3, 16); plotSpec('metal', '00001')
plt.subplot(6, 3, 17); plotSpec('metal', '00002')
plt.subplot(6, 3, 18); plotSpec('metal', '00003')
plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)
plt.show()

plt.figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2, 2, 1)
sample_rate, X = wavfile.read(r'.\genres\classical\converted\classical.00000.au.wav')
plt.xlabel('time')
plt.ylabel('frequency')
specgram(X, Fs=sample_rate, xextent=(0, 30))

plt.subplot(2, 2, 2)
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.xlim((0, 3000))
plt.plot(fft(X, sample_rate))

plt.subplot(2, 2, 3)
sample_rate, X = wavfile.read(r'.\genres\pop\converted\pop.00000.au.wav')
plt.xlabel('time')
plt.ylabel('frequency')
specgram(X, Fs=sample_rate, xextent=(0, 30))

plt.subplot(2, 2, 4)
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.xlim((0, 3000))
plt.plot(fft(X, sample_rate))

print(fft(X, sample_rate))
plt.show()
