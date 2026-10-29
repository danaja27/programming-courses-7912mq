# Collagen cream — premium UGC edit

`collagen-cream-ugc-1080x1920.mp4`: 18.1 s, 1080×1920 (9:16), 30 fps, H.264 High + AAC 256k, Rec.709 SDR.
No music; sound design only (-24.8 LUFS integrated, -1 dBFS peak).

| Out (s) | Source clip | Source in → out (s) | Shot |
|---|---|---|---|
| 0.00–2.00 | IMG_3057 | 0.40 → 2.90 (1.25×) | Hook: jar lights up from silhouette |
| 2.00–4.10 | IMG_3061 | 0.20–0.90 / 4.70–5.40 / 9.30–10.00 | Hands-free stepped rotation, aligned + continuous push |
| 4.10–5.90 | IMG_3063 | 1.90 → 3.70 | Lid twist and lift |
| 5.90–7.50 | IMG_3065 | 1.80 → 3.40 | Lid set down |
| 7.50–10.03 | IMG_3066 | 2.00 → 4.40 (0.8× ramp 2.6–3.2) | Inner seal reveal |
| 9.83–11.83 | IMG_3071 | 0.50 → 2.50 | Cream macro (6-frame dissolve in) |
| 11.83–13.63 | IMG_3078 | 4.50 → 6.30 | Fingertip scoop |
| 13.63–14.63 | IMG_3081 | 3.30 → 4.30 | Texture between fingertips |
| 14.63–18.13 | IMG_3074 | 0.10 → 3.60 | Hero: open jar, slow push |

`scripts/render.py` (video: HLG decode, per-shot white balance, selective yellow-green despill,
sub-pixel push-ins, rotation alignment) and `scripts/audio.py` (sound design) reproduce the edit
from the original iPhone clips.

## v2: library sound effects

`collagen-cream-ugc-1080x1920-v2-library-sfx.mp4`: same picture; synthesized foley replaced by recorded
library sounds from AI Producer (OpusClip), mixed with the clips' own recorded taps and lid contact
(`scripts/audio_real.py`). -23.4 LUFS, -1 dBFS peak.

| Out (s) | Sound | Gain |
|---|---|---|
| 1.38 | Whoosh - Airy Smooth 02 (reveal) | -10 dB |
| 1.99 / 2.69 / 3.39 | Gear - Middle 01 (rotation ticks) | -8 / -9 / -8 dB |
| 4.20 | Gear - Smooth and Premium 01 (lid twist) | -6 dB |
| 8.02 | Pop - Warm 01 (inner seal) | -12 dB |
| 9.55 | Whoosh - Airy Smooth 01 (dissolve to macro) | -15 dB |
| 14.21 | Swoosh - Warm Slow 01 (hero) | -15 dB |

## v3: middle/end sounds reworked

`collagen-cream-ugc-1080x1920-v3.mp4`: opening kept from v2 (reveal whoosh, rotation ticks + the jar's own taps,
lid-twist ratchet). From 5.6 s on, a sound plays only on a visible physical action:
Click - Airy 01 on the lid touching down (6.59 s, -9 dB), Click - Airy 02 as the inner seal comes free
(7.86 s in / 8.05 s hit, -12 dB). Recorded lid rustle, Pop, and the macro/hero whooshes removed; cream
moments and the hero shot are silent. -23.3 LUFS, -1 dBFS peak.
