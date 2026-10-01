import json
from pathlib import Path
from fastapi import FastAPI,Request,Depends,Form,UploadFile,File
from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .db import Base,engine,get_db
from .models import User,Regulation,Chunk,ChatLog
from .config import ADMIN_USERNAME,ADMIN_PASSWORD
from .security import hash_password,verify_password,make_session,read_session
from .rag import extract_file,chunk_text,embed,retrieve,ai_answer
Base.metadata.create_all(bind=engine)
app=FastAPI(title='University Regulations AI',version='1.0')
templates=Jinja2Templates(directory=str(Path(__file__).parent/'templates'))
def ensure_admin():
 db=next(get_db());u=db.query(User).filter_by(username=ADMIN_USERNAME).first()
 if not u:db.add(User(username=ADMIN_USERNAME,password_hash=hash_password(ADMIN_PASSWORD),role='admin'));db.commit()
 db.close()
ensure_admin()
def user_of(req,db):
 uid=read_session(req.cookies.get('session'));return db.get(User,uid) if uid else None
def ctx(req,db,**kw):
 u=user_of(req,db); lang=req.cookies.get('lang','ar'); lang=lang if lang in ('ar','en') else 'ar'
 t={'app':'نظام لوائح الجامعة AI' if lang=='ar' else 'University Regulations AI','dashboard':'لوحة التحكم' if lang=='ar' else 'Dashboard','ask':'اسأل عن اللوائح' if lang=='ar' else 'Ask Regulations','regulations':'اللوائح' if lang=='ar' else 'Regulations','upload':'رفع لائحة' if lang=='ar' else 'Upload','users':'المستخدمون' if lang=='ar' else 'Users','history':'السجل' if lang=='ar' else 'History','logout':'خروج' if lang=='ar' else 'Logout','login':'دخول' if lang=='ar' else 'Login'}
 return {'request':req,'user':u,'lang':lang,'tr':lambda k:t.get(k,k),**kw}
def admin(req,db):
 u=user_of(req,db);return u if u and u.active and u.role=='admin' else None
def active_user(req,db):
 u=user_of(req,db);return u if u and u.active else None
@app.get('/health')
def health():return {'status':'ok'}
@app.get('/',response_class=HTMLResponse)
def home(req:Request,db:Session=Depends(get_db)):return RedirectResponse('/dashboard' if active_user(req,db) else '/login',303)
@app.get('/login',response_class=HTMLResponse)
def login_page(req:Request,db:Session=Depends(get_db)):return templates.TemplateResponse('login.html',ctx(req,db,error=None))
@app.post('/login',response_class=HTMLResponse)
def login(req:Request,username:str=Form(...),password:str=Form(...),db:Session=Depends(get_db)):
 u=db.query(User).filter_by(username=username).first()
 if not u or not u.active or not verify_password(password,u.password_hash):return templates.TemplateResponse('login.html',ctx(req,db,error='بيانات الدخول غير صحيحة.'),status_code=401)
 r=RedirectResponse('/dashboard',303);r.set_cookie('session',make_session(u.id),httponly=True,samesite='lax',max_age=43200);return r
@app.get('/logout')
def logout():r=RedirectResponse('/login',303);r.delete_cookie('session');return r
@app.get('/language/{v}')
def language(v,req:Request):r=RedirectResponse(req.headers.get('referer') or '/dashboard',303);r.set_cookie('lang',v if v in ('ar','en') else 'ar',max_age=31536000,samesite='lax');return r
@app.get('/dashboard',response_class=HTMLResponse)
def dashboard(req:Request,db:Session=Depends(get_db)):
 if not active_user(req,db):return RedirectResponse('/login',303)
 return templates.TemplateResponse('dashboard.html',ctx(req,db,regulations=db.query(Regulation).count(),chunks=db.query(Chunk).count(),questions=db.query(ChatLog).count(),users=db.query(User).filter_by(active=True).count(),recent=db.query(Regulation).order_by(Regulation.created_at.desc()).limit(5).all()))
@app.get('/ask',response_class=HTMLResponse)
def ask_page(req:Request,db:Session=Depends(get_db)):return templates.TemplateResponse('ask.html',ctx(req,db,result=None,question='')) if active_user(req,db) else RedirectResponse('/login',303)
@app.post('/ask',response_class=HTMLResponse)
def ask(req:Request,question:str=Form(...),db:Session=Depends(get_db)):
 u=active_user(req,db)
 if not u:return RedirectResponse('/login',303)
 src=retrieve(db,question.strip()); ans=ai_answer(question,src) if src else None
 ans=ans or ('تم العثور على النصوص التالية من اللوائح، لكن مفتاح OpenAI غير مفعّل لتوليد إجابة لغوية.' if src else 'لم أجد نصاً مناسباً في قاعدة اللوائح.')
 db.add(ChatLog(username=u.username,question=question,answer=ans,sources_json=json.dumps(src,ensure_ascii=False)));db.commit()
 return templates.TemplateResponse('ask.html',ctx(req,db,result={'answer':ans,'sources':src},question=question))
@app.get('/regulations',response_class=HTMLResponse)
def regs(req:Request,db:Session=Depends(get_db)):return templates.TemplateResponse('regulations.html',ctx(req,db,rows=db.query(Regulation).order_by(Regulation.created_at.desc()).all())) if active_user(req,db) else RedirectResponse('/login',303)
@app.get('/admin/upload',response_class=HTMLResponse)
def upload_page(req:Request,db:Session=Depends(get_db)):return templates.TemplateResponse('upload.html',ctx(req,db,error=None,success=None)) if admin(req,db) else RedirectResponse('/dashboard',303)
@app.post('/admin/upload',response_class=HTMLResponse)
async def upload(req:Request,title:str=Form(...),category:str=Form(''),language:str=Form('ar'),file:UploadFile=File(...),db:Session=Depends(get_db)):
 if not admin(req,db):return RedirectResponse('/dashboard',303)
 try:pages=extract_file(file.filename or 'file.txt',await file.read())
 except Exception as e:return templates.TemplateResponse('upload.html',ctx(req,db,error=str(e),success=None))
 r=Regulation(title=title.strip(),category=category.strip(),language=language,filename=file.filename or 'uploaded',page_count=len(pages));db.add(r);db.flush();pending=[]
 for p,txt in pages:
  for piece in chunk_text(txt):pending.append((p,piece))
 embs=embed([x[1] for x in pending]) if pending else []
 for i,(p,txt) in enumerate(pending):db.add(Chunk(regulation_id=r.id,chunk_index=i,page_number=p,text=txt,embedding_json=json.dumps(embs[i]) if embs[i] else None))
 db.commit();return templates.TemplateResponse('upload.html',ctx(req,db,error=None,success=f'تمت الفهرسة بنجاح: {len(pending)} مقطعاً.'))
@app.post('/admin/regulations/{rid}/toggle')
def toggle(rid:int,req:Request,db:Session=Depends(get_db)):
 if admin(req,db):
  r=db.get(Regulation,rid)
  if r:r.active=not r.active;db.commit()
 return RedirectResponse('/regulations',303)
@app.get('/users',response_class=HTMLResponse)
def users(req:Request,db:Session=Depends(get_db)):return templates.TemplateResponse('users.html',ctx(req,db,rows=db.query(User).order_by(User.created_at.desc()).all(),error=None)) if admin(req,db) else RedirectResponse('/dashboard',303)
@app.post('/users')
def create_user(req:Request,username:str=Form(...),password:str=Form(...),role:str=Form('viewer'),db:Session=Depends(get_db)):
 if not admin(req,db):return RedirectResponse('/dashboard',303)
 if db.query(User).filter_by(username=username).first():return templates.TemplateResponse('users.html',ctx(req,db,rows=db.query(User).all(),error='اسم المستخدم موجود بالفعل.'))
 db.add(User(username=username.strip(),password_hash=hash_password(password),role=role if role in ('admin','viewer') else 'viewer'));db.commit();return RedirectResponse('/users',303)
@app.post('/users/{uid}/toggle')
def toggle_user(uid:int,req:Request,db:Session=Depends(get_db)):
 a=admin(req,db);u=db.get(User,uid)
 if a and u and u.id!=a.id:u.active=not u.active;db.commit()
 return RedirectResponse('/users',303)
@app.get('/history',response_class=HTMLResponse)
def history(req:Request,db:Session=Depends(get_db)):return templates.TemplateResponse('history.html',ctx(req,db,rows=db.query(ChatLog).order_by(ChatLog.created_at.desc()).limit(100).all())) if active_user(req,db) else RedirectResponse('/login',303)
