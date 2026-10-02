import io
import json
import math
import os
import re
from typing import Iterable

from pypdf import PdfReader
from docx import Document

from .config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_EMBEDDING_MODEL,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


def norm(text: str) -> str:
    """تنظيف النص مع الحفاظ على محتوى اللوائح."""
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """تقسيم النص إلى مقاطع مناسبة للبحث الدلالي."""
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


def cosine(a, b) -> float:
    """حساب cosine similarity بين متجهين."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    return dot / (na * nb) if na and nb else 0.0


def _client():
    """إنشاء عميل Gemini عند توفر المفتاح والمكتبة."""
    if not GEMINI_API_KEY or genai is None:
        return None

    try:
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        return None


def embed(texts: list[str]) -> list[list[float] | None]:
    """
    إنشاء embeddings باستخدام Gemini.
    عند حدوث مشكلة في API لا يفشل رفع الملف بالكامل؛
    يتم إرجاع None للمقاطع حتى يستمر البحث النصي.
    """
    if not texts:
        return []

    client = _client()
    if client is None:
        return [None] * len(texts)

    result: list[list[float] | None] = []

    # دفعات صغيرة لتقليل حجم الطلب الواحد.
    batch_size = 50

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch,
            )

            embeddings = getattr(response, "embeddings", None) or []

            for item in embeddings:
                values = getattr(item, "values", None)
                result.append(list(values) if values is not None else None)

            # إذا أعاد الخادم عدداً أقل من المطلوب.
            while len(result) < start + len(batch):
                result.append(None)

        return result[: len(texts)]

    except Exception:
        # مهم: لا نعيد 500 عند نفاد الحصة أو خطأ مفتاح Gemini.
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

    query_vector = None

    if GEMINI_API_KEY and genai is not None:
        try:
            query_vector = embed([query])[0]
        except Exception:
            query_vector = None

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
                semantic_score = 0.0

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
    """
    توليد الإجابة بواسطة Gemini اعتماداً على المقاطع المسترجعة فقط.
    """
    client = _client()

    if client is None or not sources:
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
        f"{system_instruction}\n\n"
        f"النصوص المسترجعة:\n{context}\n\n"
        f"السؤال:\n{query}"
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=1200,
            ),
        )

        answer = getattr(response, "text", None)
        return answer.strip() if answer else None

    except Exception:
        # يمنع ظهور Internal Server Error عند فشل Gemini.
        return None
