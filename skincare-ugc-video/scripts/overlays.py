"""CapCut-style animated Arabic captions, drawn with PIL (raqm shaping) as RGBA layers."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FD = '/tmp/claude-0/-home-user-programming-courses-7912mq/cce7b8df-2aa2-5add-8441-8f6af982964f/scratchpad/fonts/'
W, H, FPS = 1080, 1920, 30
CAPTION_FONT = FD + 'f4.ttf'   # Readex Pro Bold
TITLE_FONT = FD + 'f2.ttf'     # Cairo Black

DARK = (24, 21, 19)
CREAM = (247, 241, 233)
GOLD = (214, 184, 138)


def _pill(text, font_path, size, fg, bg, bg_alpha, pad=(46, 22), radius=None, check=False):
    font = ImageFont.truetype(font_path, size)
    box = font.getbbox(text, direction='rtl', language='ar')
    tw, th = box[2] - box[0], box[3] - box[1]
    cw = int(size * 1.05) if check else 0
    w, h = tw + 2 * pad[0] + cw, th + 2 * pad[1]
    r = radius if radius is not None else h // 2
    sh = 28  # room for the soft shadow
    im = Image.new('RGBA', (w + 2 * sh, h + 2 * sh), (0, 0, 0, 0))
    shadow = Image.new('RGBA', im.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((sh, sh + 6, sh + w, sh + h + 6), r, fill=(0, 0, 0, 70))
    im = Image.alpha_composite(im, shadow.filter(ImageFilter.GaussianBlur(12)))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((sh, sh, sh + w, sh + h), r, fill=bg + (int(255 * bg_alpha),))
    tx = sh + pad[0] - box[0]
    d.text((tx, sh + pad[1] - box[1]), text, font=font, fill=fg, direction='rtl', language='ar')
    if check:  # gold disc with a check mark, on the right (start of an RTL line)
        cx, cy, cr = sh + w - pad[0] // 2 - cw // 2, sh + h // 2, int(size * 0.42)
        d.ellipse((cx - cr, cy - cr, cx + cr, cy + cr), fill=GOLD + (255,))
        s = cr * 0.55
        d.line([(cx - s, cy + s * 0.05), (cx - s * 0.25, cy + s * 0.7), (cx + s * 0.9, cy - s * 0.6)], fill=DARK + (255,), width=max(4, size // 11), joint='curve')
    return np.asarray(im).astype(np.float32) / 255.0


def ease_out_back(x, k=1.6):
    x = min(max(x, 0.0), 1.0)
    return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Item:
    def __init__(self, layer, cy, t_in, t_out, anim='pop', cx=W // 2):
        self.layer, self.cy, self.t_in, self.t_out, self.anim, self.cx = layer, cy, t_in, t_out, anim, cx

    def state(self, t):
        """scale, alpha, dx, dy at time t, or None when hidden."""
        if t < self.t_in or t > self.t_out:
            return None
        a_in = (t - self.t_in) * FPS
        a_out = (self.t_out - t) * FPS
        alpha = min(smooth(a_in / 6), smooth(a_out / 7))
        dy = -14 * (1 - smooth(a_out / 7))
        if self.anim == 'pop':
            s = 0.82 + 0.18 * ease_out_back(a_in / 10)
            return s, alpha, 0, dy
        if self.anim == 'slide':  # RTL slide-in from the right
            return 1.0, alpha, 60 * (1 - smooth(a_in / 9)), dy
        return 1.0, alpha, 0, dy


def build(end):
    cap = lambda txt: _pill(txt, CAPTION_FONT, 62, (255, 255, 255, 255), DARK, 0.72)
    items = [
        Item(cap('خلّوني فرجيكن كريمي المفضّل'), 470, 0.25, 1.72),
        Item(_pill('كريم الكولاجين', TITLE_FONT, 104, DARK + (255,), CREAM, 0.94, pad=(58, 18)), 470, 2.30, 3.88),
        Item(cap('يلّا نفتحه'), 1490, 4.35, 5.68),
        Item(cap('شوفوا القوام'), 1490, 7.80, 9.55),
        Item(cap('ناعم وكريمي'), 1510, 10.15, 11.58),
        Item(cap('بيتوزّع بسهولة'), 440, 12.05, 14.38),
    ]
    lines = ['شدّ البشرة', 'التخلّص من التجاعيد', 'غلق المسام الواسعة', 'توحيد لون البشرة']
    for i, txt in enumerate(lines):
        lay = _pill(txt, CAPTION_FONT, 56, (255, 255, 255, 255), DARK, 0.74, pad=(40, 20), check=True)
        items.append(Item(lay, 370 + i * 134, 15.0 + i * 0.42, end + 1, anim='slide'))
    return items


def composite(img, t, items):
    """img: float32 RGB 0..1 (H, W, 3); returns a new frame with every visible caption drawn."""
    out = img
    for it in items:
        st = it.state(t)
        if st is None:
            continue
        s, alpha, dx, dy = st
        lay = it.layer
        if abs(s - 1) > 1e-3:
            import cv2
            lay = cv2.resize(lay, (max(1, int(lay.shape[1] * s)), max(1, int(lay.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
        lh, lw = lay.shape[:2]
        x0 = int(round(it.cx - lw / 2 + dx)); y0 = int(round(it.cy - lh / 2 + dy))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + lw), min(H, y0 + lh)
        if xa >= xb or ya >= yb:
            continue
        L = lay[ya - y0:yb - y0, xa - x0:xb - x0]
        a = L[..., 3:4] * alpha
        if out is img:
            out = img.copy()
        out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - a) + L[..., :3] * a
    return out
