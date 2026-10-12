"""Orquestador: foto + referencia + petición -> foto editada.
CLI: python -m estilo.edit foto.jpg ref.jpg "más cálido" out.jpg"""
import argparse
from . import llm, style
from .recipe import Recipe


def edit(src_path, ref_path, text="", out_path=None, strength=1.0, use_llm=True):
    src, ref = style.load(src_path), style.load(ref_path, 1024)
    src_f, ref_f = style.features(src), style.features(ref)
    base = Recipe(match_strength=.7 * strength, color_strength=.8 * strength)
    recipe, note = base, "sin IA local (receta determinista)"
    if use_llm and (srv := llm.detect()):
        try:
            recipe, note = llm.suggest(text, src_f, ref_f, base, *srv), f"IA local {srv[1]}"
        except Exception as e:  # servidor caído / JSON roto: la edición nunca depende del LLM
            note = f"IA local falló ({type(e).__name__}); receta determinista"
    out = style.apply(src, recipe, ref_f)
    d0, d1 = style.distance(src_f, ref_f), style.distance(style.features(out), ref_f)
    if out_path:
        style.save(out, out_path)
    return {"recipe": recipe, "note": note, "dist_before": d0, "dist_after": d1}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m estilo.edit", description="Aplica el estilo de una referencia a una foto.")
    ap.add_argument("foto")
    ap.add_argument("referencia")
    ap.add_argument("peticion", nargs="?", default="")
    ap.add_argument("salida", nargs="?", default="out.jpg")
    ap.add_argument("--intensidad", type=float, default=1.0, help="0..1 (por defecto 1)")
    ap.add_argument("--sin-ia", action="store_true", help="no consultar al LLM local")
    a = ap.parse_args(argv)
    res = edit(a.foto, a.referencia, a.peticion, a.salida, min(1.0, max(0.0, a.intensidad)), use_llm=not a.sin_ia)
    print(f"{a.salida}: {res['note']} · distancia {res['dist_before']:.3f} -> {res['dist_after']:.3f}")


if __name__ == "__main__":
    main()
