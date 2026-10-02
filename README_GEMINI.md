# نظام لوائح جامعة البطانة AI — Gemini / Render

هذه النسخة تستخدم Google Gemini بدل OpenAI.

## متغيرات Render المعتمدة

- ADMIN_USERNAME
- ADMIN_PASSWORD
- DATABASE_URL
- GEMINI_API_KEY
- GEMINI_MODEL
- GEMINI_EMBEDDING_MODEL
- PYTHON_VERSION
- SECRET_KEY

## القيم المقترحة

- `ADMIN_USERNAME`: مثل `admin`
- `ADMIN_PASSWORD`: كلمة مرور قوية من اختيارك
- `DATABASE_URL`: رابط PostgreSQL الذي يوفره Render
- `GEMINI_API_KEY`: مفتاح Gemini من Google AI Studio
- `GEMINI_MODEL`: النموذج الذي تختاره في حساب Gemini؛ الافتراضي في المشروع `gemini-3.8-flash`
- `GEMINI_EMBEDDING_MODEL`: الافتراضي `gemini-embedding-001` ويمكن تغييره من Render إذا اخترت نموذج embeddings آخر مدعوم
- `PYTHON_VERSION`: استخدم إصدار Python 3.11 المطلوب للمشروع؛ في `render.yaml` تم تثبيته على `3.11.11`
- `SECRET_KEY`: مفتاح سري طويل؛ ملف `render.yaml` يجعله يُولّد تلقائياً

## تشغيل Render

Build Command:
`pip install -r requirements.txt`

Start Command:
`uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Health Check:
`/health`

لا تضع المفاتيح السرية داخل GitHub. ضعها في Render Environment Variables.
