"""API local: POST /edit (foto + referencia -> JPEG) y GET /health. Sin BD ni autenticación."""
import json
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from estilo import edit, llm

app = FastAPI(title="estilo-foto-ia")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_MB = 30  # techo por archivo: por encima no es una foto de cámara


@app.get("/health")
def health():
    return {"llm": llm.detect()}


def _h(v) -> str:
    """Starlette codifica las cabeceras en latin-1: un emoji en la nota del LLM rompería la respuesta."""
    return str(v).encode("latin-1", "replace").decode("latin-1")


async def _save(u: UploadFile, limit: int) -> str:
    """Copia el subido a un .jpg temporal en trozos de 1 MB; 413 antes de escribir de más."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    size = 0
    try:
        while data := await u.read(1 << 20):
            size += len(data)
            if size > limit:
                raise HTTPException(413, f"{u.filename or 'archivo'} supera {limit >> 20} MB")
            tmp.write(data)
    finally:
        tmp.close()
        if size > limit:
            Path(tmp.name).unlink(missing_ok=True)  # en Windows hay que cerrar antes de borrar
    return tmp.name


@app.post("/edit")
async def edit_photo(photo: UploadFile = File(...), reference: UploadFile = File(...),
                     text: str = Form(""), strength: float = Form(1.0)):
    """Aplica el estilo de `reference` a `photo`; la receta viaja en las cabeceras X-*."""
    limit, paths = MAX_MB << 20, []
    try:
        src, ref = await _save(photo, limit), await _save(reference, limit)
        paths += [src, ref]
        out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        out.close()  # PIL deduce el formato de la extensión: fuera .jpg o falla style.save
        paths.append(out.name)
        res = edit.edit(src, ref, text, out.name, min(1.0, max(0.0, strength)))
        jpeg = Path(out.name).read_bytes()
    except (OSError, ValueError) as e:  # foto corrupta o no imagen: frontera de confianza
        raise HTTPException(400, f"imagen no procesable: {e}") from e
    finally:
        for p in paths:
            Path(p).unlink(missing_ok=True)
    return Response(jpeg, media_type="image/jpeg", headers={
        "X-Recipe": _h(json.dumps(res["recipe"].__dict__)),
        "X-Note": _h(res["note"]),
        "X-Dist-Before": _h(f'{res["dist_before"]:.6f}'),
        "X-Dist-After": _h(f'{res["dist_after"]:.6f}'),
    })


app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True))  # al final: no tapa /edit ni /health


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
