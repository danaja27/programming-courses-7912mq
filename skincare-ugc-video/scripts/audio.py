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


# 1) hook: soft airy swell rising with the jar lighting up (0 -> 2.0s), gone before the first cut
n = int(2.0 * SR)
air = band(rng.standard_normal(n), 1800, 7000)
e = np.linspace(0, 1, n) ** 2.2; e[-int(0.12 * SR):] *= np.linspace(1, 0, int(0.12 * SR))
place(norm(air * e, -30), 0.0, 0)

# 2) rotation ticks: the jar's own recorded tap from IMG_3061 (~2.3-2.8s), high-passed, on each angle cut
s61 = load('3061')
seg = s61[int(2.3 * SR):int(2.8 * SR)]
pk = np.argmax(np.abs(seg)); tap = seg[max(0, pk - int(0.004 * SR)):pk + int(0.09 * SR)]
tap = band(tap, 180, 9000) * np.exp(-np.arange(len(tap)) / (0.035 * SR))
for t, g in ((2.0, -13), (2.7, -14), (3.4, -12)):
    place(norm(tap, g), t - 0.004)

# 3) lid twist (out 4.2-5.0): fine plastic thread friction, then a soft release as the lid comes free (~5.05)
n = int(0.8 * SR)
grain = np.zeros(n)
pos = 0
while pos < n:
    pos += int(SR * rng.uniform(0.018, 0.04))
    if pos < n: grain[pos] = rng.uniform(0.5, 1.0)
k = np.exp(-np.arange(int(0.006 * SR)) / (0.0015 * SR)) * rng.standard_normal(int(0.006 * SR))
fr = np.convolve(grain, k)[:n] + 0.15 * rng.standard_normal(n)
fr = band(fr, 1500, 6000) * np.sin(np.linspace(0, np.pi, n)) ** 0.7
place(norm(fr, -27), 4.2)
n = int(0.12 * SR)
rel = band(rng.standard_normal(n), 400, 2200) * env(n, 0.0015, 0.018)
rel += 0.6 * np.sin(2 * np.pi * np.linspace(520, 380, n).cumsum() / SR) * env(n, 0.001, 0.02)
place(norm(rel, -17), 5.05)

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

# 5) inner seal coming free (~8.05): a small, soft suction tick
n = int(0.09 * SR)
pop = band(rng.standard_normal(n), 250, 1400) * env(n, 0.002, 0.014)
pop += 0.5 * np.sin(2 * np.pi * np.linspace(700, 300, n).cumsum() / SR) * env(n, 0.001, 0.012)
place(norm(pop, -20), 8.05)

# 6) fingertip meets cream (12.33) and lifts away (13.13): soft, low, short
for t, d, g in ((12.33, 0.28, -27), (13.13, 0.22, -31)):
    n = int(d * SR)
    sq = band(rng.standard_normal(n), 150, 1300) * env(n, 0.03, d / 3)
    place(norm(sq, g), t)

# 7) cream rubbed between fingertips (13.7-14.6): faint, gently modulated
n = int(0.9 * SR)
rub = band(rng.standard_normal(n), 700, 3200)
rub *= (0.55 + 0.45 * np.sin(2 * np.pi * 4.5 * np.arange(n) / SR)) * np.sin(np.linspace(0, np.pi, n))
place(norm(rub, -32), 13.7)

# 8) hero: faint air bookend fading to silence
n = int(2.6 * SR)
tail = band(rng.standard_normal(n), 2000, 7000) * np.exp(-np.arange(n) / (0.8 * SR))
tail[:int(0.3 * SR)] *= np.linspace(0, 1, int(0.3 * SR))
place(norm(tail, -36), 14.7, 0)

out = out[:int(DUR * SR)]
out[-int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))
print('peak dBFS %.1f' % (20 * np.log10(np.abs(out).max())))
st = np.stack([out, out], 1)
w = wave.open(sys.argv[1], 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((np.clip(st, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()
