import subprocess, sys, glob, numpy as np, cv2

UP = '/root/.claude/uploads/cce7b8df-2aa2-5add-8441-8f6af982964f/'
W, H, FPS = 1080, 1920, 30
OUT = sys.argv[1]
PREVIEW = len(sys.argv) > 2  # stills only


def clip(n):
    return glob.glob(UP + '*IMG_%s.mov' % n)[0]


def load(n, t0, t1):
    """Decode [t0,t1] at native 30fps as float32 RGB 0..1 (HLG decoded as-is, see notes)."""
    cmd = ['ffmpeg', '-v', 'error', '-ss', '%.3f' % t0, '-i', clip(n), '-t', '%.3f' % (t1 - t0),
           '-map', '0:v:0', '-vf', 'fps=30', '-pix_fmt', 'rgb48le', '-f', 'rawvideo', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint16).reshape(-1, H, W, 3)
    return fr, t0


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def easeout(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 2


def grade_params(img, warm=0.0, target=0.93, sat=1.06, contrast=0.14, despill=0.0):
    """White balance from bright, low-saturation pixels; exposure so the bright end sits at target."""
    f = img.astype(np.float32) / 65535
    lum = f.mean(2)
    mx, mn = f.max(2), f.min(2)
    lo, hi = np.percentile(lum, 75), np.percentile(lum, 99.0)
    m = (lum >= lo) & (lum <= hi) & ((mx - mn) < 0.12)
    rgb = f[m].mean(0)
    g = rgb.mean() / rgb
    g = np.clip(g, 0.85, 1.18)
    g = g * np.array([1 + warm, 1, 1 - warm])
    p = np.percentile((f * g).mean(2), 99.3)
    gain = target / max(p, 1e-3)
    gain = float(np.clip(gain, 0.85, 1.35))
    return dict(g=(g * gain).astype(np.float32), sat=sat, contrast=contrast, despill=despill)


def apply_grade(f, P):
    x = f * P['g']
    x = np.clip(x, 0, 1)
    a = P['contrast']
    x = x * (1 - a) + a * (x * x * (3 - 2 * x))
    if P['despill'] > 0:
        # pull yellow-green (backlit plastic) toward the cream seen in the other shots; skin (a*>6) untouched
        lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
        a, b = lab[..., 1], lab[..., 2]
        w = np.clip((b - 7) / 10, 0, 1) * np.clip((6 - a) / 6, 0, 1) * P['despill']
        lab[..., 2] = b - w * (b - 6)
        lab[..., 1] = a - w * (a - 1.5)
        x = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    if P['sat'] != 1.0:
        l = x.mean(2, keepdims=True)
        x = l + (x - l) * P['sat']
    return np.clip(x, 0, 1)


def sample(frames, t0, src_t):
    fi = (src_t - t0) * FPS
    i = int(np.floor(fi)); a = fi - i
    i = max(0, min(i, len(frames) - 1)); j = min(i + 1, len(frames) - 1)
    A = frames[i].astype(np.float32) / 65535
    if a < 0.04 or j == i:
        return A
    if a > 0.96:
        return frames[j].astype(np.float32) / 65535
    return A * (1 - a) + (frames[j].astype(np.float32) / 65535) * a


def warp(img, s, src_anchor=(W / 2, H / 2), dst_anchor=(W / 2, H / 2)):
    if abs(s - 1) < 1e-4 and src_anchor == dst_anchor:
        return img
    ax, ay = src_anchor; bx, by = dst_anchor
    M = np.float32([[s, 0, bx - s * ax], [0, s, by - s * ay]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def min_cover(s_anchor, d_anchor):
    ax, ay = s_anchor; bx, by = d_anchor
    return max(bx / ax, (W - bx) / (W - ax), by / ay, (H - by) / (H - ay))


# ---------------- shot list (approved plan) ----------------
# each shot: renders list of frames of length round(dur*30)
def shot1():  # 3057 hook: dark -> lit, 1.25x, slow push
    fr, t0 = load('3057', 0.30, 3.05)
    P = grade_params(fr[-3], warm=0.01, despill=0.8)
    n = 60
    for k in range(n):
        t = k / FPS
        img = apply_grade(sample(fr, t0, 0.40 + 1.25 * t), P)
        s = 1.0 + 0.06 * ease(t / 2.0)
        yield warp(img, s, (540, 1010), (540, 1010))


# rotation: align jar (center x, base y, height) from 3061 measurements to a shared target
ROT = [  # src in, src center x, base y, jar height
    (0.20, 559, 1114, 400),
    (4.70, 573, 1162, 412),
    (9.30, 534, 1204, 426),
]


def shot2():
    tgt = (540, 1150)
    k_need = max(min_cover((cx, by), tgt) * 426 / 426 for _, cx, by, h in ROT)
    total = 63
    idx = 0
    P = None
    for si, (tin, cx, by, h) in enumerate(ROT):
        fr, t0 = load('3061', tin - 0.05, tin + 0.80)
        if P is None:
            P = grade_params(load('3061', 9.6, 9.7)[0][0])
        for k in range(21):
            t = k / FPS
            img = apply_grade(sample(fr, t0, tin + t), P)
            rel = 412.0 / h
            base = max(min_cover((cx, by), tgt) / rel for _, cx, by, h in ROT) * 1.005
            push = 1.0 + 0.05 * (idx / (total - 1))
            s = base * rel * push
            idx += 1
            yield warp(img, s, (cx, by), tgt)


def simple(n, tin, dur, z0=1.0, z1=1.0, anchor=(540, 960), dst=None, map_t=None, warm=0.0, ref=None, eas=ease, sat=1.06, despill=0.0):
    span = map_t(dur) - tin if map_t else dur
    fr, t0 = load(n, tin - 0.05, tin + span + 0.1)
    rf = load(n, ref, ref + 0.05)[0][0] if ref is not None else fr[len(fr) // 2]
    P = grade_params(rf, warm=warm, sat=sat, despill=despill)
    nfr = int(round(dur * FPS))
    for k in range(nfr):
        t = k / FPS
        st = map_t(t) if map_t else tin + t
        img = apply_grade(sample(fr, t0, st), P)
        s = z0 + (z1 - z0) * eas(t / dur)
        yield warp(img, s, anchor, dst or anchor)


def ramp5(t):  # 3066: 1x, 0.8x through the seal release, 1x
    if t < 0.6:
        return 2.0 + t
    if t < 1.35:
        return 2.6 + (t - 0.6) * 0.8
    return 3.2 + (t - 1.35)


SHOTS = [
    ('1 hook 3057', shot1),
    ('2 rotation 3061', shot2),
    ('3 twist 3063', lambda: simple('3063', 1.90, 1.8, 1.0, 1.025, (540, 1000), ref=4.5)),
    ('4 lid down 3065', lambda: simple('3065', 1.80, 1.6, 1.02, 1.035, (540, 1000), ref=4.5)),
    ('5 seal 3066', lambda: simple('3066', 2.00, 2.55, 1.0, 1.04, (540, 900), map_t=ramp5, ref=0.3)),
    ('6 macro 3071', lambda: simple('3071', 0.50, 2.0, 1.10, 1.17, (520, 760), ref=1.5)),
    ('7 scoop 3078', lambda: simple('3078', 4.50, 1.8, 1.0, 1.03, (540, 1000), ref=3.0, despill=0.8)),
    ('8 texture 3081', lambda: simple('3081', 3.30, 1.0, 1.02, 1.04, (600, 900), ref=7.5, despill=0.8)),
    ('9 hero 3074', lambda: simple('3074', 0.10, 3.5, 1.08, 1.14, (200, 1920), ref=1.5, eas=easeout)),
]
DISSOLVE_AFTER = {4: 6}  # shot index 4 (seal) -> 6-frame dissolve into shot 5 (macro)


def to16(x):
    # tiny dither before quantizing to keep the smooth background free of banding
    x = x + (np.random.rand(*x.shape).astype(np.float32) - 0.5) / 4096
    return (np.clip(x, 0, 1) * 65535).astype(np.uint16)


if PREVIEW:
    for name, fn in SHOTS:
        frs = list(fn())
        for tag, f in (('a', frs[0]), ('b', frs[len(frs) // 2]), ('c', frs[-1])):
            cv2.imwrite(OUT + '/%s_%s.jpg' % (name.split()[0], tag), cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)[::2, ::2])
        print(name, len(frs), flush=True)
    sys.exit()

enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-',
                        '-vf', 'scale=out_color_matrix=bt709:out_range=tv:sws_dither=ed,format=yuv420p',
                        '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-profile:v', 'high', '-tune', 'film',
                        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv',
                        OUT], stdin=subprocess.PIPE)
pending = []  # tail frames waiting for a dissolve
total = 0
for i, (name, fn) in enumerate(SHOTS):
    frs = fn()
    if pending:
        d = len(pending)
        for k in range(d):
            a = (k + 1) / (d + 1)
            enc.stdin.write(to16(pending[k] * (1 - a) + next(frs) * a).tobytes()); total += 1
        pending = []
    buf = []
    hold = DISSOLVE_AFTER.get(i, 0)
    for f in frs:
        buf.append(f)
        if len(buf) > hold:
            enc.stdin.write(to16(buf.pop(0)).tobytes()); total += 1
    pending = buf
    print(name, 'done, frames so far', total, flush=True)
for f in pending:
    enc.stdin.write(to16(f).tobytes()); total += 1
enc.stdin.close(); enc.wait()
print('total frames', total, total / FPS, 's')
