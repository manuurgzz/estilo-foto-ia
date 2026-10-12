"""Tests de la API: `pytest test_server.py` (necesita pytest y httpx, ver requirements-dev.txt)."""
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from estilo import edit, llm
from estilo.recipe import Recipe


def _jpeg(color=(120, 90, 60), size=(64, 48)) -> bytes:
    g = np.linspace(0, 1, size[0], dtype=np.float32)
    px = np.stack([np.tile(g, (size[1], 1)) * c / 255 for c in color], -1)  # degradado: evita imágenes uniformes
    buf = io.BytesIO()
    Image.fromarray((px * 255).astype(np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def _post(client, photo=None, reference=None, **form):
    files = {"photo": ("foto.jpg", photo or _jpeg(), "image/jpeg"),
             "reference": ("ref.jpg", reference or _jpeg((250, 160, 80)), "image/jpeg")}
    return client.post("/edit", files=files, data=form)


@pytest.fixture(autouse=True)
def sin_llm(monkeypatch):
    """Por defecto no hay servidor de IA local: los tests no deben depender del entorno."""
    monkeypatch.setattr(llm, "detect", lambda *a, **k: None)


@pytest.fixture
def client():
    return TestClient(server.app)


def test_health_sin_llm(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"llm": None}


def test_health_con_llm(client, monkeypatch):
    monkeypatch.setattr(llm, "detect", lambda *a, **k: ("http://localhost:11434/v1", "modelo"))
    assert client.get("/health").json() == {"llm": ["http://localhost:11434/v1", "modelo"]}


def test_edit_devuelve_jpeg_y_cabeceras(client):
    r = _post(client, text="más cálido")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).size == (64, 48)
    recipe = json.loads(r.headers["x-recipe"])
    assert set(recipe) == set(Recipe().__dict__)
    assert r.headers["x-note"] == "sin IA local (receta determinista)"
    d0, d1 = float(r.headers["x-dist-before"]), float(r.headers["x-dist-after"])
    assert d1 < d0, "la edición debe acercar la foto a la referencia"


def test_edit_cors_expone_las_cabeceras(client):
    """Sin Access-Control-Expose-Headers el navegador oculta las X-* a un origen distinto (p. ej. file://)."""
    r = client.post("/edit", headers={"Origin": "null"}, files={
        "photo": ("a.jpg", _jpeg(), "image/jpeg"), "reference": ("b.jpg", _jpeg(), "image/jpeg")})
    exposed = r.headers["access-control-expose-headers"]
    for h in ("X-Recipe", "X-Note", "X-Dist-Before", "X-Dist-After"):
        assert h in exposed


def test_intensidad_se_acota_a_0_1(client):
    # fuera de rango no debe fallar ni producir recetas fuera de límites
    for s in ("-5", "7"):
        r = _post(client, strength=s)
        assert r.status_code == 200
        rec = json.loads(r.headers["x-recipe"])
        assert 0 <= rec["match_strength"] <= .7 and 0 <= rec["color_strength"] <= .8


def test_intensidad_cero_no_cambia_la_foto(client):
    src = _jpeg()
    r = _post(client, photo=src, strength="0")
    a = np.asarray(Image.open(io.BytesIO(src)).convert("RGB"), dtype=int)
    b = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=int)
    assert np.abs(a - b).max() <= 6  # solo ruido de recompresión JPEG


def test_imagen_corrupta_da_400(client):
    r = _post(client, photo=b"esto no es una imagen")
    assert r.status_code == 400 and "imagen no procesable" in r.json()["detail"]


def test_faltan_campos_da_422(client):
    assert client.post("/edit", files={"photo": ("a.jpg", _jpeg(), "image/jpeg")}).status_code == 422


def test_archivo_demasiado_grande_da_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_MB", 0)
    r = _post(client)
    assert r.status_code == 413 and "supera" in r.json()["detail"]


def _tmp_jpgs() -> set:
    return set(Path(tempfile.gettempdir()).glob("tmp*.jpg"))


@pytest.mark.parametrize("caso", ["ok", "corrupta", "grande"])
def test_no_deja_temporales(client, monkeypatch, caso):
    antes = _tmp_jpgs()
    if caso == "grande":
        monkeypatch.setattr(server, "MAX_MB", 0)
    _post(client, photo=b"basura" if caso == "corrupta" else None)
    assert _tmp_jpgs() - antes == set()


def test_nota_con_emoji_no_rompe_la_respuesta(client, monkeypatch):
    """Las cabeceras viajan en latin-1: un emoji en el nombre del modelo no debe provocar un 500."""
    monkeypatch.setattr(llm, "detect", lambda *a, **k: ("http://x/v1", "modelo🙂"))
    monkeypatch.setattr(llm, "suggest", lambda text, s, r, base, *srv: Recipe(exposure=.5))
    r = _post(client, text="más luz")
    assert r.status_code == 200
    assert r.headers["x-note"].startswith("IA local modelo")
    assert json.loads(r.headers["x-recipe"])["exposure"] == .5


def test_fallo_del_llm_cae_a_receta_determinista(tmp_path, monkeypatch):
    """Se prueba en edit.edit y no por HTTP: la nota lleva tildes (latin-1 en cabeceras) y TestClient las decodifica como UTF-8."""
    def roto(*a, **k):
        raise ConnectionError("caído")
    monkeypatch.setattr(llm, "detect", lambda *a, **k: ("http://x/v1", "m"))
    monkeypatch.setattr(llm, "suggest", roto)
    for n in ("foto", "ref"):
        (tmp_path / f"{n}.jpg").write_bytes(_jpeg())
    res = edit.edit(str(tmp_path / "foto.jpg"), str(tmp_path / "ref.jpg"), "", str(tmp_path / "out.jpg"))
    assert "falló" in res["note"] and (tmp_path / "out.jpg").exists()


def test_la_web_se_sirve_en_la_raiz(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>" in r.text
