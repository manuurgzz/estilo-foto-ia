"""IA local (API compatible OpenAI: Ollama, LM Studio, llama.cpp). Una sola llamada por edición, solo texto."""
import json
import re
import urllib.request
from .recipe import LIMITS, Recipe
from .style import summary

OLLAMA, LMSTUDIO = "http://localhost:11434/v1", "http://localhost:1234/v1"


def detect(urls=(OLLAMA, LMSTUDIO)):
    """Devuelve (base_url, primer_modelo) del primer servidor local vivo, o None."""
    for u in urls:
        try:
            with urllib.request.urlopen(u + "/models", timeout=1.5) as r:
                return u, json.load(r)["data"][0]["id"]
        except Exception:
            continue


def suggest(text, src_f, ref_f, base: Recipe, url, model, timeout=120) -> Recipe:
    params = ", ".join(f"{k}[{lo},{hi}]" for k, (lo, hi, _) in LIMITS.items())
    prompt = (
        "Eres un retocador de fotos. Ajusta la receta base para que la foto ORIGEN adopte el estilo de la REFERENCIA "
        f"y cumpla la petición del usuario.\nPetición: {text or '(solo copiar el estilo)'}\n"
        f"ORIGEN: {json.dumps(summary(src_f))}\nREFERENCIA: {json.dumps(summary(ref_f))}\n"
        f"Receta base: {json.dumps(base.__dict__)}\n"
        f"Parámetros y rangos: {params}\nLab: L 0-100, a/b ±128. Devuelve SOLO un JSON con los parámetros que cambies."
    )
    body = json.dumps({"model": model, "temperature": 0.2, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url + "/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)  # los modelos pequeños envuelven el JSON en texto/markdown
    return Recipe.from_dict(json.loads(m.group(0)) if m else {}, base)
