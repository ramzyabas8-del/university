import json
from pathlib import Path

from fastapi import FastAPI, Request, Depends, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .db import Base, engine, get_db
from .models import User, Regulation, Chunk, ChatLog
from .config import ADMIN_USERNAME, ADMIN_PASSWORD
from .security import hash_password, verify_password, make_session, read_session
from .rag import extract_file, chunk_text, embed, retrieve, ai_answer


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="University Regulations AI",
    version="1.0",
)

templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)


def ensure_admin():
    """إنشاء حساب المدير الافتراضي إذا لم يكن موجوداً."""
    db = next(get_db())
    try:
        user = db.query(User).filter_by(username=ADMIN_USERNAME).first()
        if not user:
            db.add(
                User(
                    username=ADMIN_USERNAME,
                    password_hash=hash_password(ADMIN_PASSWORD),
                    role="admin",
                )
            )
            db.commit()
    finally:
        db.close()


ensure_admin()


def user_of(req: Request, db: Session):
    """إرجاع المستخدم الحالي اعتماداً على Session Cookie."""
    session_id = req.cookies.get("session")
    if not session_id:
        return None

    user_id = read_session(session_id)
    if not user_id:
        return None

    return db.get(User, user_id)


def ctx(req: Request, db: Session, **kw):
    """تجهيز البيانات المشتركة لقوالب Jinja2."""
    user = user_of(req, db)

    lang = req.cookies.get("lang", "ar")
    if lang not in ("ar", "en"):
        lang = "ar"

    translations = {
        "app": "نظام لوائح الجامعة AI" if lang == "ar" else "University Regulations AI",
        "dashboard": "لوحة التحكم" if lang == "ar" else "Dashboard",
        "ask": "اسأل عن اللوائح" if lang == "ar" else "Ask Regulations",
        "regulations": "اللوائح" if lang == "ar" else "Regulations",
        "upload": "رفع لائحة" if lang == "ar" else "Upload",
        "users": "المستخدمون" if lang == "ar" else "Users",
        "history": "السجل" if lang == "ar" else "History",
        "logout": "خروج" if lang == "ar" else "Logout",
        "login": "دخول" if lang == "ar" else "Login",
    }

    return {
        "request": req,
        "user": user,
        "lang": lang,
        "tr": lambda key: translations.get(key, key),
        **kw,
    }


def render_template(
    template_name: str,
    request: Request,
    db: Session,
    status_code: int = 200,
    **context,
):
    """عرض قالب Jinja2 بالطريقة المتوافقة مع Starlette الحديثة."""
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context=ctx(request, db, **context),
        status_code=status_code,
    )


def admin(req: Request, db: Session):
    """إرجاع المدير الحالي إذا كان الحساب فعالاً."""
    user = user_of(req, db)
    if user and user.active and user.role == "admin":
        return user
    return None


def active_user(req: Request, db: Session):
    """إرجاع المستخدم الحالي إذا كان الحساب فعالاً."""
    user = user_of(req, db)
    if user and user.active:
        return user
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(req: Request, db: Session = Depends(get_db)):
    if active_user(req, db):
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(req: Request, db: Session = Depends(get_db)):
    return render_template("login.html", req, db, error=None)


@app.post("/login", response_class=HTMLResponse)
def login(
    req: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    username = username.strip()

    user = db.query(User).filter_by(username=username).first()

    if (
        not user
        or not user.active
        or not verify_password(password, user.password_hash)
    ):
        return render_template(
            "login.html",
            req,
            db,
            status_code=401,
            error="بيانات الدخول غير صحيحة.",
        )

    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie(
        "session",
        make_session(user.id),
        httponly=True,
        samesite="lax",
        max_age=43200,
    )
    return response


@app.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session")
    return response


@app.get("/language/{v}")
def language(v: str, req: Request):
    if v not in ("ar", "en"):
        v = "ar"

    response = RedirectResponse(
        req.headers.get("referer") or "/dashboard",
        status_code=303,
    )
    response.set_cookie(
        "lang",
        v,
        max_age=31536000,
        samesite="lax",
    )
    return response


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(req: Request, db: Session = Depends(get_db)):
    if not active_user(req, db):
        return RedirectResponse("/login", status_code=303)

    recent = (
        db.query(Regulation)
        .order_by(Regulation.created_at.desc())
        .limit(5)
        .all()
    )

    return render_template(
        "dashboard.html",
        req,
        db,
        regulations=db.query(Regulation).count(),
        chunks=db.query(Chunk).count(),
        questions=db.query(ChatLog).count(),
        users=db.query(User).filter_by(active=True).count(),
        recent=recent,
    )


@app.get("/ask", response_class=HTMLResponse)
def ask_page(req: Request, db: Session = Depends(get_db)):
    if not active_user(req, db):
        return RedirectResponse("/login", status_code=303)

    return render_template(
        "ask.html",
        req,
        db,
        result=None,
        question="",
    )


@app.post("/ask", response_class=HTMLResponse)
def ask(
    req: Request,
    question: str = Form(...),
    db: Session = Depends(get_db),
):
    user = active_user(req, db)

    if not user:
        return RedirectResponse("/login", status_code=303)

    question = question.strip()

    if not question:
        return render_template(
            "ask.html",
            req,
            db,
            result=None,
            question="",
        )

    sources = retrieve(db, question)
    answer = ai_answer(question, sources) if sources else None

    if not answer:
        if sources:
            answer = (
                "تم العثور على النصوص التالية من اللوائح، "
                "لكن خدمة Gemini غير متاحة حالياً. تأكد من ضبط GEMINI_API_KEY في Render."
            )
        else:
            answer = "لم أجد نصاً مناسباً في قاعدة اللوائح."

    db.add(
        ChatLog(
            username=user.username,
            question=question,
            answer=answer,
            sources_json=json.dumps(sources, ensure_ascii=False),
        )
    )
    db.commit()

    return render_template(
        "ask.html",
        req,
        db,
        result={"answer": answer, "sources": sources},
        question=question,
    )


@app.get("/regulations", response_class=HTMLResponse)
def regs(req: Request, db: Session = Depends(get_db)):
    if not active_user(req, db):
        return RedirectResponse("/login", status_code=303)

    rows = (
        db.query(Regulation)
        .order_by(Regulation.created_at.desc())
        .all()
    )

    return render_template(
        "regulations.html",
        req,
        db,
        rows=rows,
    )


@app.get("/admin/upload", response_class=HTMLResponse)
def upload_page(req: Request, db: Session = Depends(get_db)):
    if not admin(req, db):
        return RedirectResponse("/dashboard", status_code=303)

    return render_template(
        "upload.html",
        req,
        db,
        error=None,
        success=None,
    )


@app.post("/admin/upload", response_class=HTMLResponse)
async def upload(
    req: Request,
    title: str = Form(...),
    category: str = Form(""),
    language: str = Form("ar"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not admin(req, db):
        return RedirectResponse("/dashboard", status_code=303)

    try:
        filename = file.filename or "uploaded"
        pages = extract_file(filename, await file.read())
    except Exception as exc:
        return render_template(
            "upload.html",
            req,
            db,
            error=str(exc),
            success=None,
        )

    regulation = Regulation(
        title=title.strip(),
        category=category.strip(),
        language=language,
        filename=file.filename or "uploaded",
        page_count=len(pages),
    )

    db.add(regulation)
    db.flush()

    pending = []
    for page_number, text in pages:
        for piece in chunk_text(text):
            pending.append((page_number, piece))

    embeddings = embed([item[1] for item in pending]) if pending else []
    semantic_indexing = bool(
        pending
        and embeddings
        and any(item is not None for item in embeddings)
    )

    for index, (page_number, text) in enumerate(pending):
        embedding = embeddings[index] if index < len(embeddings) else None

        db.add(
            Chunk(
                regulation_id=regulation.id,
                chunk_index=index,
                page_number=page_number,
                text=text,
                embedding_json=(
                    json.dumps(embedding, ensure_ascii=False)
                    if embedding
                    else None
                ),
            )
        )

    db.commit()

    if semantic_indexing:
        success = f"تمت الفهرسة بنجاح باستخدام Gemini: {len(pending)} مقطعاً."
    elif pending:
        success = (
            f"تم حفظ {len(pending)} مقطعاً، لكن لم يتم إنشاء الفهرسة الدلالية. "
            "سيعمل البحث النصي، وتأكد من GEMINI_API_KEY في Render."
        )
    else:
        success = "تم رفع الملف، لكن لم يتم العثور على نص قابل للفهرسة."

    return render_template(
        "upload.html",
        req,
        db,
        error=None,
        success=success,
    )


@app.post("/admin/regulations/{rid}/toggle")
def toggle(
    rid: int,
    req: Request,
    db: Session = Depends(get_db),
):
    if admin(req, db):
        regulation = db.get(Regulation, rid)
        if regulation:
            regulation.active = not regulation.active
            db.commit()

    return RedirectResponse("/regulations", status_code=303)


@app.get("/users", response_class=HTMLResponse)
def users(req: Request, db: Session = Depends(get_db)):
    if not admin(req, db):
        return RedirectResponse("/dashboard", status_code=303)

    rows = (
        db.query(User)
        .order_by(User.created_at.desc())
        .all()
    )

    return render_template(
        "users.html",
        req,
        db,
        rows=rows,
        error=None,
    )


@app.post("/users")
def create_user(
    req: Request,
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("viewer"),
    db: Session = Depends(get_db),
):
    if not admin(req, db):
        return RedirectResponse("/dashboard", status_code=303)

    username = username.strip()

    if db.query(User).filter_by(username=username).first():
        rows = (
            db.query(User)
            .order_by(User.created_at.desc())
            .all()
        )
        return render_template(
            "users.html",
            req,
            db,
            rows=rows,
            error="اسم المستخدم موجود بالفعل.",
        )

    if role not in ("admin", "viewer"):
        role = "viewer"

    db.add(
        User(
            username=username,
            password_hash=hash_password(password),
            role=role,
        )
    )
    db.commit()

    return RedirectResponse("/users", status_code=303)


@app.post("/users/{uid}/toggle")
def toggle_user(
    uid: int,
    req: Request,
    db: Session = Depends(get_db),
):
    current_admin = admin(req, db)
    user = db.get(User, uid)

    if (
        current_admin
        and user
        and user.id != current_admin.id
    ):
        user.active = not user.active
        db.commit()

    return RedirectResponse("/users", status_code=303)


@app.get("/history", response_class=HTMLResponse)
def history(req: Request, db: Session = Depends(get_db)):
    if not active_user(req, db):
        return RedirectResponse("/login", status_code=303)

    rows = (
        db.query(ChatLog)
        .order_by(ChatLog.created_at.desc())
        .limit(100)
        .all()
    )

    return render_template(
        "history.html",
        req,
        db,
        rows=rows,
    )
