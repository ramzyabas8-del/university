# نظام لوائح الجامعة AI — Web/RAG/Render

مشروع Python 3.11 + FastAPI + SQLAlchemy + PostgreSQL/SQLite + OpenAI Responses API.

## الوظائف
- تسجيل دخول وصلاحيات Admin/Viewer
- لوحة تحكم وإحصاءات
- رفع PDF/DOCX/TXT من المتصفح
- استخراج النص وتقسيمه إلى Chunks
- Embeddings عند توفر OPENAI_API_KEY
- RAG هجين: بحث نصي + cosine similarity
- إجابة AI مع مصادر وصفحات
- إدارة اللوائح والمستخدمين
- سجل الأسئلة
- عربي/English
- Dark/Light
- Health check

## التشغيل المحلي
Python 3.11 ثم:
`python -m venv .venv`
ثم فعّل البيئة وثبّت:
`pip install -r requirements.txt`
انسخ `.env.example` إلى `.env` ثم شغّل:
`uvicorn app.main:app --reload`
افتح `http://127.0.0.1:8000`.

## Render
الأفضل رفع المشروع إلى GitHub ثم في Render اختيار New > Blueprint، واختيار المستودع الذي يحتوي render.yaml. سيُنشأ Web Service وقاعدة Postgres. أدخل ADMIN_PASSWORD وOPENAI_API_KEY من Environment.

إذا اخترت Web Service يدوياً:
Build: `pip install -r requirements.txt`
Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
Health: `/health`

لا تضع API key داخل GitHub. استخدم Environment Variables في Render.

## ملاحظة اللوائح
النظام لا يخترع اللوائح. ارفع الوثائق الرسمية للجامعة فقط. ملفات PDF المصورة تحتاج OCR قبل الفهرسة في هذه النسخة.
