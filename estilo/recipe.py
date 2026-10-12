"""Contrato único: receta de edición. La IA local solo puede proponer estos números, acotados."""
from dataclasses import dataclass, fields

# nombre: (mín, máx, por defecto). Fuente de verdad para validar y para el prompt del LLM.
LIMITS = {
    "exposure": (-2, 2, 0),          # EV
    "temp": (-1, 1, 0),              # azul <-> amarillo
    "tint": (-1, 1, 0),              # verde <-> magenta
    "contrast": (-1, 1, 0),
    "highlights": (-1, 1, 0),        # <0 recupera altas luces
    "shadows": (-1, 1, 0),           # >0 abre sombras
    "saturation": (-1, 1, 0),
    "vibrance": (-1, 1, 0),
    "match_strength": (0, 1, 0.7),   # cuánto copiar la curva tonal (L) de la referencia
    "color_strength": (0, 1, 0.8),   # cuánto copiar la paleta (a,b) de la referencia
    "skin_protect": (0, 1, 0.7),     # reduce la transferencia de color en piel
    "split_shadow_a": (-20, 20, 0), "split_shadow_b": (-20, 20, 0),
    "split_high_a": (-20, 20, 0), "split_high_b": (-20, 20, 0),
    "grain": (0, 1, 0),
    "vignette": (-1, 1, 0),
}


@dataclass
class Recipe:
    exposure: float = 0
    temp: float = 0
    tint: float = 0
    contrast: float = 0
    highlights: float = 0
    shadows: float = 0
    saturation: float = 0
    vibrance: float = 0
    match_strength: float = 0.7
    color_strength: float = 0.8
    skin_protect: float = 0.7
    split_shadow_a: float = 0
    split_shadow_b: float = 0
    split_high_a: float = 0
    split_high_b: float = 0
    grain: float = 0
    vignette: float = 0

    @classmethod
    def from_dict(cls, d: dict, base: "Recipe | None" = None) -> "Recipe":
        """Valida la salida del LLM: ignora claves ajenas y valores no numéricos, acota el resto."""
        out = {f.name: getattr(base or cls(), f.name) for f in fields(cls)}
        for k, v in (d or {}).items():
            if k in LIMITS and isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf"):
                lo, hi, _ = LIMITS[k]
                out[k] = min(hi, max(lo, float(v)))
        return cls(**out)
