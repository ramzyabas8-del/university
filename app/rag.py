"""RAG utilities for University Regulations AI.

Gemini version with:
- PDF/DOCX/TXT extraction
- Gemini embeddings
- Hybrid lexical + semantic retrieval
- Gemini answer generation
- Automatic retry for temporary 503/429 API failures
- Safe logging without exposing secrets
"""

import io
import json
import logging
import math
import re
import time
from typing import Optional

from docx import Document
from pypdf import PdfReader

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
    TOP_K,
)

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None
    types = None

logger = logging.getLogger(__name__)

# Temporary Gemini failures should be retried, but not forever.
MAX_GEMINI_RETRIES = 4
INITIAL_RETRY_DELAY = 2.0
MAX_RETRY_DELAY = 12.0


def norm(text: str) -> str:
    """تنظيف النص مع الحفاظ على محتوى اللوائح."""
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """تقسيم النص إلى مقاطع مناسبة للبحث."""
    text = norm(text)
    if not text:
        return []

    size = max(300, int(size))
    overlap = max(0, min(int(overlap), size - 1))

    out: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end]

        if end < len(text):
            cut = max(
                piece.rfind(". "),
                piece.rfind("؛"),
                piece.rfind("،"),
                piece.rfind(" "),
            )
            if cut > size * 0.55:
                end = start + cut + 1
                piece = text[start:end]

        piece = piece.strip()
        if piece:
            out.append(piece)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return out


def extract_file(name: str, data: bytes):
    """استخراج النص من PDF أو DOCX أو TXT."""
    filename = (name or "").lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return [
            (index + 1, norm(page.extract_text() or ""))
            for index, page in enumerate(reader.pages)
        ]

    if filename.endswith(".docx"):
        document = Document(io.BytesIO(data))
        text = norm("\n".join(p.text for p in document.paragraphs))
        return [(1, text)]

    if filename.endswith(".txt"):
        return [(1, norm(data.decode("utf-8", "ignore")))]

    raise ValueError("الصيغ المدعومة: PDF, DOCX, TXT")


def lexical(query: str, text: str) -> float:
    """درجة تشابه نصية بسيطة تعمل حتى بدون Gemini."""
    a = set(re.findall(r"\w+", (query or "").lower(), re.UNICODE))
    b = set(re.findall(r"\w+", (text or "").lower(), re.UNICODE))
    return len(a & b) / len(a) if a else 0.0


def cosine(a: list[float], b: list[float]) -> float:
    """حساب cosine similarity بين متجهين."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _client():
    """إنشاء عميل Gemini بدون كشف المفتاح في السجلات."""
    if genai is None:
        logger.error("google-genai SDK is not installed or could not be imported.")
        return None

    if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
        logger.error("GEMINI_API_KEY is empty or missing from the environment.")
        return None

    try:
        return genai.Client(api_key=GEMINI_API_KEY.strip())
    except Exception:
        logger.exception("Failed to initialize the Gemini client.")
        return None


def _is_retryable_gemini_error(exc: Exception) -> bool:
    """تحديد أخطاء Gemini المؤقتة التي تستحق إعادة المحاولة."""
    text = str(exc).upper()
    retry_codes = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")
    return any(code in text for code in retry_codes)


def _retry_delay(attempt: int) -> float:
    """Exponential backoff مع حد أعلى."""
    return min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)


def _extract_embedding_values(item) -> Optional[list[float]]:
    """قراءة قيم embedding من كائن SDK أو dict."""
    values = getattr(item, "values", None)
    if values is None and isinstance(item, dict):
        values = item.get("values")

    if values is None:
        return None

    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def _embed_batch(client, batch: list[str]) -> list[Optional[list[float]]]:
    """إرسال دفعة embeddings مع إعادة المحاولة للأخطاء المؤقتة."""
    last_error = None

    for attempt in range(MAX_GEMINI_RETRIES):
        try:
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch,
            )

            embeddings = getattr(response, "embeddings", None)
            if embeddings is None and isinstance(response, dict):
                embeddings = response.get("embeddings")
            embeddings = embeddings or []

            values = [_extract_embedding_values(item) for item in embeddings]
            while len(values) < len(batch):
                values.append(None)
            return values[: len(batch)]

        except Exception as exc:
            last_error = exc
            if not _is_retryable_gemini_error(exc) or attempt == MAX_GEMINI_RETRIES - 1:
                break

            delay = _retry_delay(attempt)
            logger.warning(
                "Gemini embedding temporary error; retry %d/%d in %.1fs. model=%s status=%s",
                attempt + 1,
                MAX_GEMINI_RETRIES - 1,
                delay,
                GEMINI_EMBEDDING_MODEL,
                type(exc).__name__,
            )
            time.sleep(delay)

    logger.error(
        "Gemini embedding failed after retries. model=%s error=%s",
        GEMINI_EMBEDDING_MODEL,
        str(last_error)[:500] if last_error else "unknown",
    )
    return [None] * len(batch)


def embed(texts: list[str]) -> list[Optional[list[float]]]:
    """إنشاء embeddings باستخدام Gemini مع retry للأخطاء المؤقتة."""
    if not texts:
        return []

    client = _client()
    if client is None or not GEMINI_EMBEDDING_MODEL:
        if not GEMINI_EMBEDDING_MODEL:
            logger.error("GEMINI_EMBEDDING_MODEL is empty.")
        return [None] * len(texts)

    result: list[Optional[list[float]]] = []
    batch_size = 50

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        result.extend(_embed_batch(client, batch))

    return result[: len(texts)]


def retrieve(db, query: str):
    """استرجاع أفضل المقاطع من لوائح الجامعة."""
    from .models import Chunk, Regulation

    rows = (
        db.query(Chunk, Regulation)
        .join(Regulation, Chunk.regulation_id == Regulation.id)
        .filter(Regulation.active.is_(True))
        .all()
    )

    if not rows:
        return []

    query_vector = None
    if GEMINI_API_KEY and genai is not None:
        try:
            query_embeddings = embed([query])
            query_vector = query_embeddings[0] if query_embeddings else None
        except Exception:
            logger.exception("Unexpected error while embedding the user query.")

    scored = []

    for chunk, regulation in rows:
        lexical_score = lexical(query, chunk.text)
        semantic_score = 0.0

        if query_vector and chunk.embedding_json:
            try:
                semantic_score = cosine(
                    query_vector,
                    json.loads(chunk.embedding_json),
                )
            except Exception:
                logger.warning(
                    "Invalid stored embedding for chunk id=%s",
                    getattr(chunk, "id", "unknown"),
                )

        score = (
            0.45 * lexical_score + 0.55 * semantic_score
            if query_vector
            else lexical_score
        )
        scored.append((score, chunk, regulation))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            "score": round(float(score), 4),
            "text": chunk.text,
            "page": chunk.page_number,
            "title": regulation.title,
            "regulation_id": regulation.id,
        }
        for score, chunk, regulation in scored[:TOP_K]
        if score > 0
    ]


def _generate_answer_once(client, prompt: str, system_instruction: str):
    """محاولة واحدة لتوليد الإجابة."""
    return client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=1200,
        ),
    )


def ai_answer(query: str, sources: list[dict]):
    """توليد إجابة Gemini اعتماداً على المقاطع المسترجعة فقط.

    يعيد None إذا كانت خدمة Gemini غير متاحة بعد إعادة المحاولة،
    حتى يستطيع main.py عرض رسالة مناسبة للمستخدم بدلاً من HTTP 500.
    """
    client = _client()

    if client is None or not sources:
        return None

    if not GEMINI_MODEL:
        logger.error("GEMINI_MODEL is empty.")
        return None

    context = "\n\n".join(
        f"[المصدر {index + 1}] {item['title']} - الصفحة {item['page']}\n"
        f"{item['text']}"
        for index, item in enumerate(sources)
    )

    system_instruction = (
        "أنت مساعد أكاديمي لنظام لوائح جامعة البطانة. "
        "أجب اعتماداً على النصوص المسترجعة المرفقة فقط. "
        "إذا لم توجد الإجابة بوضوح في النصوص فقل إن المعلومات غير موجودة "
        "في المقاطع المسترجعة. لا تخترع أرقام مواد أو شروطاً أو تواريخ. "
        "اذكر اسم اللائحة ورقم الصفحة عند توفرهما. "
        "أجب بالعربية إذا كان السؤال بالعربية، وبالإنجليزية إذا كان السؤال بالإنجليزية."
    )

    prompt = (
        "النصوص المسترجعة من قاعدة لوائح الجامعة:\n\n"
        f"{context}\n\n"
        f"السؤال:\n{query}"
    )

    last_error = None

    for attempt in range(MAX_GEMINI_RETRIES):
        try:
            response = _generate_answer_once(
                client,
                prompt,
                system_instruction,
            )

            answer = getattr(response, "text", None)
            if answer:
                return answer.strip()

            logger.error(
                "Gemini returned no text. model=%s response_type=%s",
                GEMINI_MODEL,
                type(response).__name__,
            )
            return None

        except Exception as exc:
            last_error = exc

            if not _is_retryable_gemini_error(exc) or attempt == MAX_GEMINI_RETRIES - 1:
                break

            delay = _retry_delay(attempt)
            logger.warning(
                "Gemini answer temporary error; retry %d/%d in %.1fs. model=%s status=%s",
                attempt + 1,
                MAX_GEMINI_RETRIES - 1,
                delay,
                GEMINI_MODEL,
                type(exc).__name__,
            )
            time.sleep(delay)

    logger.error(
        "Gemini generate_content failed after retries. model=%s error=%s",
        GEMINI_MODEL,
        str(last_error)[:700] if last_error else "unknown",
    )
    return None
