"""v4: same graded shots as render.py, joined with smooth transitions; writes a clean and a captioned video."""
import sys, subprocess, collections, numpy as np, cv2

SCR = '/tmp/claude-0/-home-user-programming-courses-7912mq/cce7b8df-2aa2-5add-8441-8f6af982964f/scratchpad/'
sys.path.insert(0, SCR)
_src = open(SCR + 'render.py').read()
sys.argv = [sys.argv[0], '/dev/null'] + sys.argv[1:]
exec(_src.split('def shot1')[0])  # load, grade_params, apply_grade, sample, warp, min_cover, ease, easeout, ROT
import overlays as OV

OUT_CLEAN, OUT_TEXT = sys.argv[2], sys.argv[3]

ROT = [  # 3061 hands-free angles: src in, jar center x, base y, jar height (measured)
    (0.20, 559, 1114, 400),
    (4.70, 573, 1162, 412),
    (9.30, 534, 1204, 426),
]


def shot1(pre, post):  # 3057 hook: dark -> lit, 1.25x, slow push
    fr, t0 = load('3057', 0.30, 3.40)
    P = grade_params(fr[int((2.95 - 0.30) * FPS)], warm=0.01, despill=0.8)
    for k in range(-pre, 60 + post):
        t = k / FPS
        img = apply_grade(sample(fr, t0, 0.40 + 1.25 * t), P)
        yield warp(img, 1.0 + 0.06 * ease(t / 2.0), (540, 1010), (540, 1010))


def shot2(pre, post):  # hands-free stepped rotation, aligned, one continuous push
    tgt = (540, 1150)
    P = grade_params(load('3061', 9.6, 9.7)[0][0])
    base = max(min_cover((cx, by), tgt) / (412.0 / h) for _, cx, by, h in ROT) * 1.005
    idx = -pre
    for si, (tin, cx, by, h) in enumerate(ROT):
        k0 = -pre if si == 0 else 0
        k1 = 21 + (post if si == len(ROT) - 1 else 0)
        fr, t0 = load('3061', max(0.0, tin + k0 / FPS - 0.05), tin + k1 / FPS + 0.1)
        for k in range(k0, k1):
            img = apply_grade(sample(fr, t0, tin + k / FPS), P)
            s = base * (412.0 / h) * (1.0 + 0.05 * (idx / 62))
            idx += 1
            yield warp(img, s, (cx, by), tgt)


def simple(n, tin, dur, z0=1.0, z1=1.0, anchor=(540, 960), map_t=None, ref=None, eas=ease, despill=0.0):
    def gen(pre, post):
        mt = map_t or (lambda t: tin + t)
        a, b = mt(-pre / FPS), mt(dur + post / FPS)
        fr, t0 = load(n, max(0.0, a - 0.05), b + 0.1)
        P = grade_params(load(n, ref, ref + 0.05)[0][0], despill=despill)
        for k in range(-pre, int(round(dur * FPS)) + post):
            t = k / FPS
            img = apply_grade(sample(fr, t0, mt(t)), P)
            yield warp(img, z0 + (z1 - z0) * eas(t / dur), anchor, anchor)
    return gen


def ramp5(t):  # 3066: 1x, 0.8x through the seal release, 1x
    if t < 0.6:
        return 2.0 + t
    if t < 1.35:
        return 2.6 + (t - 0.6) * 0.8
    return 3.2 + (t - 1.35)


# (name, generator, main frames); cut points stay where v1-v3 had them, so total length is unchanged (544 frames)
SHOTS = [
    ('1 hook 3057', shot1, 60),
    ('2 rotation 3061', shot2, 63),
    ('3 twist 3063', simple('3063', 1.90, 1.8, 1.0, 1.025, (540, 1000), ref=4.5), 54),
    ('4 lid down 3065', simple('3065', 1.80, 1.6, 1.02, 1.035, (540, 1000), ref=4.5), 48),
    ('5 seal 3066', simple('3066', 2.00, 70 / 30, 1.0, 1.04, (540, 900), map_t=ramp5, ref=0.3), 70),
    ('6 macro 3071', simple('3071', 0.50, 2.0, 1.10, 1.17, (520, 760), ref=1.5), 60),
    ('7 scoop 3078', simple('3078', 4.50, 1.8, 1.0, 1.03, (540, 1000), ref=3.0, despill=0.8), 54),
    ('8 texture 3081', simple('3081', 3.30, 1.0, 1.02, 1.04, (600, 900), ref=7.5, despill=0.8), 30),
    ('9 hero 3074', simple('3074', 0.34, 3.5, 1.08, 1.14, (200, 1920), ref=1.5, eas=easeout), 105),
]
# transition after shot i: (style, frames). Centered on the cut; each side lends half the frames as a handle.
TRANS = {0: ('bloom', 12), 1: ('zoom', 10), 2: ('dissolve', 10), 3: ('blur', 10),
         4: ('zoom', 12), 5: ('blur', 10), 6: ('dissolve', 8), 7: ('bloom', 14)}


def blur(img, sigma):
    if sigma < 0.3:
        return img
    small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 2)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def zoom_c(img, s):
    return warp(img, s, (W / 2, H / 2), (W / 2, H / 2))


def transition(style, A, B, k, n):
    p = (k + 0.5) / n
    a = p * p * (3 - 2 * p)
    bell = np.sin(np.pi * p)
    if style == 'dissolve':
        return A * (1 - a) + B * a
    if style == 'blur':
        sg = 11 * bell
        return blur(A, sg) * (1 - a) + blur(B, sg) * a
    if style == 'zoom':  # push through: A grows, B settles in from slightly larger, motion softened by blur
        sg = 7 * bell
        return blur(zoom_c(A, 1 + 0.14 * a), sg) * (1 - a) + blur(zoom_c(B, 1.14 - 0.14 * a), sg) * a
    if style == 'bloom':  # soft light bloom through near-white
        x = blur(A, 4 * bell) * (1 - a) + blur(B, 4 * bell) * a
        w = 0.6 * bell ** 1.5
        return x * (1 - w) + w
    raise ValueError(style)


def encoder(path):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-',
                             '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', '-map', '0:v', '-map', '1:a', '-shortest',
                             '-vf', 'scale=out_color_matrix=bt709:out_range=tv:sws_dither=ed,format=yuv420p',
                             '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-profile:v', 'high', '-tune', 'film',
                             '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv',
                             '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', path], stdin=subprocess.PIPE)


def to16(x):
    x = x + (np.random.rand(*x.shape).astype(np.float32) - 0.5) / 4096
    return (np.clip(x, 0, 1) * 65535).astype(np.uint16)


TOTAL = sum(n for _, _, n in SHOTS)
items = OV.build(TOTAL / FPS)
e_clean, e_text = encoder(OUT_CLEAN), encoder(OUT_TEXT)
frame_no = 0


def emit(f):
    global frame_no
    e_clean.stdin.write(to16(f).tobytes())
    e_text.stdin.write(to16(OV.composite(f, frame_no / FPS, items)).tobytes())
    frame_no += 1


tail = None  # (style, n, frames of the outgoing side)
for i, (name, gen, main) in enumerate(SHOTS):
    pre = TRANS[i - 1][1] // 2 if i > 0 else 0
    post = TRANS[i][1] // 2 if i in TRANS else 0
    frames = gen(pre, post)
    if tail:
        style, n, A = tail
        for k in range(n):  # incoming side: pre handle + first n/2 main frames
            emit(transition(style, A[k], next(frames), k, n))
        skip = n - pre
    else:
        skip = 0
    hold = TRANS[i][1] - post if i in TRANS else 0  # main frames that belong to the next transition
    buf = collections.deque()
    for j, f in enumerate(frames):
        buf.append(f)
        if len(buf) > hold + post:
            emit(buf.popleft())
    tail = (TRANS[i][0], TRANS[i][1], list(buf)) if i in TRANS else None
    print(name, 'frames so far', frame_no, flush=True)
e_clean.stdin.close(); e_text.stdin.close(); e_clean.wait(); e_text.wait()
print('total frames', frame_no, frame_no / FPS)
