# estilo-foto-ia

Reajusta tus fotos a partir de una referencia y una petición en lenguaje natural.

## Uso

1. Instala las dependencias con `pip install -r requirements.txt`.
2. Arranca Ollama o LM Studio con un modelo de texto (opcional: sin él se usa la receta determinista).
3. Arranca el servidor con `uvicorn server:app --port 8000`.
4. Abre `web/index.html` o `http://localhost:8000`.
5. Uso por CLI: `python -m estilo.edit foto.jpg referencia.jpg "petición" salida.jpg [--intensidad 0.8] [--sin-ia]`.
6. Tests: `pip install -r requirements-dev.txt` y `pytest` (el núcleo también se comprueba con `python test_core.py`).

La IA solo propone números y nunca ve la imagen: no recibe ningún píxel, únicamente el texto de la petición, y devuelve parámetros de ajuste que el código aplica localmente.
