"""RAG utilities for the University Regulations AI project.

This module uses Google's current ``google-genai`` SDK for embeddings and
answer generation.  Gemini/API failures are logged without exposing the API
key and do not turn a document upload or question page into HTTP 500 errors.
"""

import io
import json
import logging
import math
import re
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
except Exception:  # pragma: no cover - depends on installed package
    genai = None
    types = None


logger = logging.getLogger(__name__)


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


def _extract_embedding_values(item) -> Optional[list[float]]:
    """قراءة قيم embedding من كائن SDK مع دعم الاستجابة الحالية."""
    values = getattr(item, "values", None)
    if values is None and isinstance(item, dict):
        values = item.get("values")

    if values is None:
        return None

    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def embed(texts: list[str]) -> list[Optional[list[float]]]:
    """إنشاء embeddings باستخدام Gemini.

    عند فشل Gemini نرجع None للمقاطع بدلاً من إيقاف رفع الملف بالكامل.
    """
    if not texts:
        return []

    client = _client()
    if client is None:
        return [None] * len(texts)

    if not GEMINI_EMBEDDING_MODEL:
        logger.error("GEMINI_EMBEDDING_MODEL is empty.")
        return [None] * len(texts)

    result: list[Optional[list[float]]] = []
    batch_size = 50

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch,
            )

            embeddings = getattr(response, "embeddings", None)
            if embeddings is None and isinstance(response, dict):
                embeddings = response.get("embeddings")
            embeddings = embeddings or []

            batch_values = [_extract_embedding_values(item) for item in embeddings]
            result.extend(batch_values)

            while len(result) < start + len(batch):
                result.append(None)

        return result[: len(texts)]

    except Exception:
        logger.exception(
            "Gemini embedding request failed. model=%s text_count=%d",
            GEMINI_EMBEDDING_MODEL,
            len(texts),
        )
        return [None] * len(texts)


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

        if query_vector:
            score = 0.45 * lexical_score + 0.55 * semantic_score
        else:
            score = lexical_score

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


def ai_answer(query: str, sources: list[dict]):
    """توليد إجابة Gemini اعتماداً على المقاطع المسترجعة فقط."""
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

    try:
        # نستخدم الإعدادات الأساسية فقط لتجنب تغيير سلوك Gemini 3.x الافتراضي.
        # توثيق Google الحالي يوضح أن GenerateContentConfig يقبل
        # system_instruction و max_output_tokens.
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=1200,
            ),
        )

        answer = getattr(response, "text", None)
        if answer:
            return answer.strip()

        logger.error(
            "Gemini returned no text. model=%s response=%r",
            GEMINI_MODEL,
            response,
        )
        return None

    except Exception:
        # لا نسجل GEMINI_API_KEY. الخطأ الكامل سيظهر في Render Logs
        # لتحديد السبب الحقيقي (quota/key/model/request/etc.).
        logger.exception(
            "Gemini generate_content failed. model=%s source_count=%d",
            GEMINI_MODEL,
            len(sources),
        )
        return None
    
