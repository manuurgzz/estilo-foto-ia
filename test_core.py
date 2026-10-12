"""Comprobaciones del núcleo: `python test_core.py` o `pytest test_core.py`."""
import time

import numpy as np

from estilo import style
from estilo.recipe import Recipe

rng = np.random.default_rng(1)
g = np.linspace(0, 1, 128, dtype=np.float32)
src = np.clip(np.stack([np.tile(g, (128, 1))] * 3, -1) * .5 + rng.normal(0, .02, (128, 128, 3)), 0, 1).astype(np.float32)
ref = np.clip(src * [1.3, 1.0, .6] + [.05, 0, 0], 0, 1).astype(np.float32)  # referencia cálida
ref = np.clip((ref - .5) * 1.4 + .5, 0, 1).astype(np.float32)               # y contrastada
sf, rf = style.features(src), style.features(ref)


def test_identidad():
    out = style.apply(src, Recipe(match_strength=0, color_strength=0), rf)
    assert np.abs(out - src).max() < 2 / 255, "receta vacía debe ser identidad"


def test_se_acerca_a_la_referencia():
    out = style.apply(src, Recipe(), rf)
    d0, d1 = style.distance(sf, rf), style.distance(style.features(out), rf)
    assert d1 < d0 * .6, f"debe acercarse al estilo de la referencia ({d0:.3f}->{d1:.3f})"


def test_receta_acota_y_descarta_basura():
    assert Recipe.from_dict({"exposure": 99, "x": 1, "temp": "hola"}).exposure == 2
    r = Recipe.from_dict({"temp": float("nan"), "tint": float("inf"), "exposure": True})
    assert (r.temp, r.tint, r.exposure) == (0, 0, 0), "NaN/inf/bool descartados"
    assert Recipe.from_dict(None).temp == 0


def test_bandas_no_cambian_el_resultado():
    """El procesado por bandas debe dar lo mismo con cualquier altura de banda (vignette y paleta incluidas)."""
    r = Recipe(contrast=.3, shadows=.2, vignette=.4, saturation=.1)
    antes, style._BAND = style._BAND, 7
    try:
        a = style.apply(src, r, rf)
    finally:
        style._BAND = antes
    assert np.abs(a - style.apply(src, r, rf)).max() < 1e-5


def test_imagenes_de_borde():
    for img in (np.full((1, 1, 3), .5, np.float32), np.full((3, 3, 3), .5, np.float32), np.zeros((128, 128, 3), np.float32)):
        o = style.apply(img, Recipe(vignette=.5), style.features(img))
        assert not np.isnan(o).any() and 0 <= o.min() and o.max() <= 1, "imagen mínima/uniforme: sin NaN y en 0..1"
    gray = src.mean(-1, keepdims=True).repeat(3, -1)
    assert not np.isnan(style.apply(gray, Recipe(), rf)).any(), "origen gris + referencia de color"


def test_rendimiento_4000x3000():
    big = rng.random((3000, 4000, 3), dtype=np.float32)
    t = time.perf_counter()
    style.apply(big, Recipe(), rf)
    assert time.perf_counter() - t < 15, "4000x3000 debe procesarse en <15 s"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK")
