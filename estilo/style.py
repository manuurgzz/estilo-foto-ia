"""Análisis de estilo y aplicación de receta. Solo numpy+Pillow, sin tokens, determinista."""
import numpy as np
from PIL import Image
from .recipe import Recipe

_M = np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]])
_W = np.array([.95047, 1., 1.08883])


def load(path, max_side=None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max_side:
        im.thumbnail((max_side, max_side))
    return np.asarray(im, dtype=np.float32) / 255


def save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)).save(path, quality=95)


def _lin(c): return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
def _gam(c): return np.where(c <= .0031308, c * 12.92, 1.055 * np.maximum(c, 0) ** (1 / 2.4) - .055)
def _f(t): return np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116)
def _finv(t): return np.where(t > 6 / 29, t ** 3, 3 * (6 / 29) ** 2 * (t - 4 / 29))


def to_lab(srgb):
    f = _f((_lin(srgb) @ _M.T) / _W)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def from_lab(lab):
    fy = (lab[..., 0] + 16) / 116
    xyz = np.stack([_finv(fy + lab[..., 1] / 500), _finv(fy), _finv(fy - lab[..., 2] / 200)], -1) * _W
    return np.clip(_gam(np.clip(xyz @ np.linalg.inv(_M).T, 0, 1)), 0, 1)


def skin_mask(srgb):
    # ponytail: caja YCbCr dura, sin suavizado; sustituir por segmentación si falla con pieles raras
    r, g, b = (srgb[..., i] * 255 for i in range(3))
    cb = 128 - .168736 * r - .331264 * g + .5 * b
    cr = 128 + .5 * r - .418688 * g - .081312 * b
    return ((cb > 77) & (cb < 127) & (cr > 133) & (cr < 173) & (r > g) & (r > 60)).astype(np.float32)


def features(srgb) -> dict:
    """Huella de estilo de una imagen (~100 números). Es lo único que ve el LLM en lugar de píxeles."""
    lab = to_lab(srgb[::4, ::4])  # submuestreo: suficiente para estadísticas
    L, ab = lab[..., 0], lab[..., 1:]
    sh, hi = L <= np.percentile(L, 33), L >= np.percentile(L, 66)  # <=/>=: imagen uniforme o 1x1 no deja máscara vacía
    return {
        "L_q": np.percentile(L, np.arange(101)).tolist(),
        "ab_mean": ab.reshape(-1, 2).mean(0).tolist(),
        "ab_std": ab.reshape(-1, 2).std(0).tolist(),
        "shadow_ab": ab[sh].mean(0).tolist(), "high_ab": ab[hi].mean(0).tolist(),
        "chroma": float(np.hypot(ab[..., 0], ab[..., 1]).mean()),
    }


def summary(f: dict) -> dict:
    """Versión compacta para el prompt (≈60 tokens)."""
    r = lambda v: [round(x, 1) for x in v]
    return {"L_p1_10_50_90_99": r([f["L_q"][i] for i in (1, 10, 50, 90, 99)]), "ab_mean": r(f["ab_mean"]),
            "ab_std": r(f["ab_std"]), "shadow_ab": r(f["shadow_ab"]), "high_ab": r(f["high_ab"]),
            "chroma": round(f["chroma"], 1)}


def distance(a: dict, b: dict) -> float:
    """Distancia de estilo 0..~1 entre dos huellas. 0 = mismo estilo. Sustituye al 'crítico' LLM."""
    d = [abs(a["L_q"][i] - b["L_q"][i]) / 100 for i in (1, 10, 50, 90, 99)]
    for k in ("ab_mean", "ab_std", "shadow_ab", "high_ab"):
        d += [abs(x - y) / 40 for x, y in zip(a[k], b[k])]
    d.append(abs(a["chroma"] - b["chroma"]) / 40)
    return float(np.mean(d))


def apply(srgb, r: Recipe, ref: dict | None = None) -> np.ndarray:
    h, w = srgb.shape[:2]
    skin = skin_mask(srgb)
    lin = _lin(srgb) * 2 ** r.exposure
    lin = lin * np.array([1 + .2 * r.temp, 1 - .1 * r.tint, 1 - .2 * r.temp], dtype=np.float32)
    lab = to_lab(_gam(np.clip(lin, 0, 1)))
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    if ref is not None and r.match_strength > 0:  # curva tonal: casar cuantiles de L
        sq = np.maximum.accumulate(np.percentile(L[::4, ::4], np.arange(101))) + np.arange(101) * 1e-6
        L = L + r.match_strength * (np.interp(L, sq, ref["L_q"]) - L)
    x = np.clip(L / 100, 0, 1)
    x = x + r.contrast * (x * x * (3 - 2 * x) - x)
    L = x * 100 + 20 * r.shadows * (1 - x) ** 3 + 20 * r.highlights * x ** 3
    x = np.clip(L / 100, 0, 1)

    if ref is not None and r.color_strength > 0:  # paleta: transferencia Reinhard en (a,b)
        ab = lab[..., 1:]
        mu, sd = ab.reshape(-1, 2).mean(0), ab.reshape(-1, 2).std(0) + 1e-3
        scale = np.clip(np.array(ref["ab_std"]) / sd, .5, 2)  # acotado: evita explotar fotos casi grises
        moved = (ab - mu) * scale + np.array(ref["ab_mean"])
        k = (r.color_strength * (1 - r.skin_protect * skin))[..., None]
        a, b = (ab + k * (moved - ab)).transpose(2, 0, 1)

    c = np.hypot(a, b)
    f = np.maximum(0, 1 + r.saturation) * np.maximum(0, 1 + r.vibrance * (1 - np.clip(c / 80, 0, 1)))
    a, b = a * f, b * f
    a = a + r.split_shadow_a * (1 - x) ** 2 + r.split_high_a * x ** 2
    b = b + r.split_shadow_b * (1 - x) ** 2 + r.split_high_b * x ** 2

    out = from_lab(np.stack([np.clip(L, 0, 100), a, b], -1))
    if r.vignette:
        yy, xx = np.mgrid[:h, :w]
        d = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / 1.414
        out = out * (1 - .5 * r.vignette * d ** 2)[..., None]
    if r.grain:
        out = out + np.random.default_rng(0).normal(0, .03 * r.grain, (h, w, 1)).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)
