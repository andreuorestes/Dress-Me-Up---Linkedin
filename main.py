import os
import json
import base64
import logging
from datetime import timedelta
from typing import List, Tuple, Optional

from flask import Flask, render_template, request, jsonify
from google.cloud import storage, secretmanager
from google.oauth2 import service_account

# Gemini
from google import genai
from google.genai import types

app = Flask(__name__)

# ---- LOGGING ----
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# ---- CONFIG ----
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
SECRET_ID = os.getenv("GCP_SECRET_ID", "") 

_storage_client = None


def _require_config():
    missing = []
    if not PROJECT_ID:
        missing.append("GCP_PROJECT_ID")
    if not BUCKET_NAME:
        missing.append("GCS_BUCKET_NAME")
    if not SECRET_ID:
        missing.append("GCP_SECRET_ID")
    if missing:
        raise RuntimeError("Faltan variables de entorno: " + ", ".join(missing))


def get_storage_client_from_secret():
    """
    Obtiene un cliente de Google Cloud Storage usando el JSON
    de la service account guardado en Secret Manager.
    """
    global _storage_client
    if _storage_client is not None:
        return _storage_client

    _require_config()
    app.logger.info("Creando cliente de Storage desde Secret Manager...")

    sm_client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID}/versions/latest"

    response = sm_client.access_secret_version(request={"name": secret_name})
    key_json = response.payload.data.decode("utf-8")

    info = json.loads(key_json)
    creds = service_account.Credentials.from_service_account_info(info)

    _storage_client = storage.Client(project=PROJECT_ID, credentials=creds)
    app.logger.info("Cliente de Storage creado correctamente.")
    return _storage_client


def _get_last_blob_for_user(user_name: str):
    """Devuelve el último blob (objeto) del usuario o None."""
    storage_client = get_storage_client_from_secret()
    prefix = f"{user_name}/"
    app.logger.info("Buscando última imagen en GCS. bucket=%s prefix=%s", BUCKET_NAME, prefix)

    blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=prefix))

    if not blobs:
        app.logger.warning("No se encontraron blobs para usuario=%s", user_name)
        return None

    last_blob = max(blobs, key=lambda b: b.time_created)
    app.logger.info("Último blob encontrado: %s", last_blob.name)
    return last_blob


def get_last_blob_for_user(user_name: str) -> Optional[str]:
    """Devuelve una URL firmada de la última imagen del usuario o None."""
    last_blob = _get_last_blob_for_user(user_name)
    if last_blob is None:
        return None

    url = last_blob.generate_signed_url(
        expiration=timedelta(hours=1),
        method="GET",
    )
    return url


def get_last_image_bytes_for_user(user_name: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Devuelve (bytes, mime_type) de la última imagen del usuario o (None, None)."""
    last_blob = _get_last_blob_for_user(user_name)
    if last_blob is None:
        return None, None

    img_bytes = last_blob.download_as_bytes()
    mime_type = last_blob.content_type or "image/jpeg"
    app.logger.info(
        "Descargada imagen base de GCS. usuario=%s blob=%s mime=%s",
        user_name,
        last_blob.name,
        mime_type,
    )
    return img_bytes, mime_type


def data_url_to_bytes(data_url: str) -> Tuple[bytes, str]:
    """
    Convierte un data URL (data:image/png;base64,xxxx) en (bytes, mime_type).
    """
    if not data_url.startswith("data:"):
        raise ValueError("Data URL inválido")

    header, b64_data = data_url.split(",", 1)
    mime_part = header.split(";")[0]  # "data:image/png"
    mime_type = mime_part.split(":")[1]  # "image/png"

    raw_bytes = base64.b64decode(b64_data)
    return raw_bytes, mime_type


def extract_image_from_gemini_response(response) -> Tuple[bytes, str]:
    """
    Busca la primera parte de tipo imagen en la respuesta de Gemini
    y devuelve (bytes, mime_type).
    """
    app.logger.info("Extrayendo imagen de la respuesta de Gemini...")
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                blob = part.inline_data
                app.logger.info("Imagen encontrada en respuesta de Gemini.")
                return blob.data, (blob.mime_type or "image/png")

    app.logger.error("La respuesta de Gemini no contiene imagen.")
    raise RuntimeError("La respuesta de Gemini no contiene imagen.")


def generate_tryon_image(person: str, clothing_data_urls: List[str]) -> Tuple[bytes, str]:
    """
    Llama a Gemini con:
      - Imagen base = última del Storage para 'person'
      - Imágenes de ropa = data URLs que vienen del front
    Devuelve (bytes_imagen_generada, mime_type).
    """
    _require_config()

    app.logger.info(
        "Iniciando generate_tryon_image. person=%s num_prendas=%d",
        person,
        len(clothing_data_urls),
    )

    # ---- GEMINI ----
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="europe-west1",
    )

    # Imagen 1: persona (desde GCS)
    base_bytes, base_mime = get_last_image_bytes_for_user(person)
    if base_bytes is None:
        raise RuntimeError(f"No se encontró imagen base para {person} en Storage.")

    parts: List[types.Part] = [
        types.Part.from_bytes(
            data=base_bytes,
            mime_type=base_mime,
        )
    ]

    # Imágenes 2..n: ropa
    for idx, data_url in enumerate(clothing_data_urls, start=2):
        img_bytes, mime_type = data_url_to_bytes(data_url)
        app.logger.info("Añadiendo imagen de ropa %d con mime=%s", idx - 1, mime_type)
        parts.append(
            types.Part.from_bytes(
                data=img_bytes,
                mime_type=mime_type,
            )
        )

    prompt_text = """
Use la persona de la imagen 1 — mantén cara, pelo, forma del cuerpo, proporciones, pose y dirección de la mirada exactamente como en la imagen 1.
Sustituye únicamente la ropa por las prendas de las imágenes siguientes (2, 3, 4, etc. si hay varias).
No cambies la pose, expresión, proporciones del cuerpo, iluminación, sombras ni el fondo original.
Coloca a la persona sobre un fondo blanco liso tipo estudio, con iluminación realista y tono de piel natural.
Imagen final en alta resolución, fotorealista, sin texto, sin marcas de agua, sin objetos extra.
"""

    parts.append(types.Part.from_text(text=prompt_text))

    contents = [
        types.Content(
            role="user",
            parts=parts,
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=32768,
        response_modalities=["TEXT", "IMAGE"],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF",
            ),
        ],
        image_config=types.ImageConfig(
            aspect_ratio="1:1",
        ),
    )

    app.logger.info("Llamando a Gemini model=gemini-2.5-flash-image...")
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=contents,
        config=generate_content_config,
    )
    app.logger.info("Respuesta recibida de Gemini.")

    img_bytes, mime_type = extract_image_from_gemini_response(response)
    app.logger.info("Imagen generada por Gemini. mime=%s tamaño=%d bytes", mime_type, len(img_bytes))
    return img_bytes, mime_type


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/last-image")
def api_last_image():
    person = request.args.get("person", "Andreu")
    if person not in ("Andreu", "Claudia"):
        return jsonify({"error": "persona inválida"}), 400

    try:
        image_url = get_last_blob_for_user(person)
        return jsonify({"imageUrl": image_url})
    except Exception as e:
        app.logger.exception("Error en /api/last-image")
        return jsonify({"imageUrl": None, "error": str(e)}), 500


@app.route("/api/try-on", methods=["POST"])
def api_try_on():
    """
    Recibe:
      {
        "person": "Andreu" | "Claudia",
        "images": [ "data:image/..;base64,...", ... ]  # ropa
      }
    Devuelve:
      { "generatedImage": "data:image/png;base64,..." }
    """
    try:
        data = request.get_json(force=True) or {}

        person = data.get("person", "Andreu")
        clothing_images = data.get("images", [])
        app.logger.info("Request /api/try-on person=%s images=%d", person, len(clothing_images) if isinstance(clothing_images, list) else -1)

        if person not in ("Andreu", "Claudia"):
            app.logger.warning("Persona inválida en /api/try-on: %s", person)
            return jsonify({"error": "persona inválida"}), 400

        if not clothing_images:
            app.logger.warning("Sin imágenes de ropa en /api/try-on")
            return jsonify({"error": "no hay imágenes de ropa"}), 400

        img_bytes, mime_type = generate_tryon_image(person, clothing_images)

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        app.logger.info("Devolviendo imagen generada a frontend.")
        return jsonify({"generatedImage": data_url})
    except Exception as e:
        app.logger.exception("Error en /api/try-on")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("PORT", "8080"))
    app.run(debug=debug, host="0.0.0.0", port=port)
