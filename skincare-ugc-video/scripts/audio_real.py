import numpy as np, wave, sys
from numpy.fft import rfft, irfft, rfftfreq

SR = 48000
A = '/tmp/claude-0/-home-user-programming-courses-7912mq/cce7b8df-2aa2-5add-8441-8f6af982964f/scratchpad/aud/'
DUR = 544 / 30
rng = np.random.default_rng(7)
out = np.zeros(int(DUR * SR) + SR)


def load(n):
    w = wave.open(A + n + '.wav')
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768


def band(x, lo=None, hi=None, slope=None):
    X = rfft(x); f = rfftfreq(len(x), 1 / SR); g = np.ones_like(f)
    if lo: g *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1)) ** 4)
    if hi: g *= 1 / np.sqrt(1 + (f / hi) ** 4)
    return irfft(X * g, len(x))


def db(v):
    return 10 ** (v / 20)


def norm(x, peak_db):
    return x / (np.abs(x).max() + 1e-12) * db(peak_db)


def place(x, t, fade=0.004):
    n = int(fade * SR)
    if n and len(x) > 2 * n:
        x = x.copy(); r = np.linspace(0, 1, n); x[:n] *= r; x[-n:] *= r[::-1]
    i = int(t * SR); out[i:i + len(x)] += x


def env(n, attack, decay):
    t = np.arange(n) / SR
    return (1 - np.exp(-t / max(attack, 1e-4))) * np.exp(-t / decay)


# 2) rotation ticks: the jar's own recorded tap from IMG_3061 (~2.3-2.8s), high-passed, on each angle cut
s61 = load('3061')
seg = s61[int(2.3 * SR):int(2.8 * SR)]
pk = np.argmax(np.abs(seg)); tap = seg[max(0, pk - int(0.004 * SR)):pk + int(0.09 * SR)]
tap = band(tap, 180, 9000) * np.exp(-np.arange(len(tap)) / (0.035 * SR))
for t, g in ((2.0, -13), (2.7, -14), (3.4, -12)):
    place(norm(tap, g), t - 0.004)

# 4) lid set down: real recorded contact from IMG_3065 (src 1.8-3.4 -> out 5.9-7.5), rumble removed
s65 = load('3065')
lid = s65[int(1.8 * SR):int(3.4 * SR)]
lid = band(lid, 160, 10000)
# keep the contact click (src 2.50s) prominent; the surrounding hand rustle sits ~14 dB under it
tt = np.arange(len(lid)) / SR + 1.8
focus = np.exp(-((tt - 2.505) / 0.03) ** 2)
lid = lid * (db(-14) + (1 - db(-14)) * focus)
lid = lid / (np.abs(lid).max() + 1e-12) * db(-13)
lid *= np.clip(np.minimum(np.arange(len(lid)), len(lid) - np.arange(len(lid))) / (0.05 * SR), 0, 1)
place(lid, 5.9, 0)

out = out[:int(DUR * SR)]
out[-int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))
print('peak dBFS %.1f' % (20 * np.log10(np.abs(out).max())))
st = np.stack([out, out], 1)
w = wave.open(sys.argv[1], 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((np.clip(st, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()
