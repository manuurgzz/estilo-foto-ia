"""Orquestador: foto + referencia + petición -> foto editada.
CLI: python -m estilo.edit foto.jpg ref.jpg "más cálido" out.jpg"""
import sys
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


if __name__ == "__main__":
    a = sys.argv[1:]
    print(edit(a[0], a[1], a[2] if len(a) > 2 else "", a[3] if len(a) > 3 else "out.jpg"))
