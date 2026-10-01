# University Regulations AI - combined core source

# ===== app/config.py =====
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
BASE_DIR=Path(__file__).resolve().parent.parent
def env(k,d=''): return os.getenv(k,d)
DATABASE_URL=env('DATABASE_URL',f'sqlite:///{BASE_DIR}/university_regulations.db')
if DATABASE_URL.startswith('postgres://'): DATABASE_URL=DATABASE_URL.replace('postgres://','postgresql+psycopg://',1)
elif DATABASE_URL.startswith('postgresql://'): DATABASE_URL=DATABASE_URL.replace('postgresql://','postgresql+psycopg://',1)
SECRET_KEY=env('SECRET_KEY','dev-secret-change-me')
ADMIN_USERNAME=env('ADMIN_USERNAME','admin'); ADMIN_PASSWORD=env('ADMIN_PASSWORD','admin123')
OPENAI_API_KEY=env('OPENAI_API_KEY',''); OPENAI_MODEL=env('OPENAI_MODEL','gpt-5.6-luna'); OPENAI_EMBEDDING_MODEL=env('OPENAI_EMBEDDING_MODEL','text-embedding-3-small')
TOP_K=int(env('TOP_K','6')); CHUNK_SIZE=int(env('CHUNK_SIZE','1800')); CHUNK_OVERLAP=int(env('CHUNK_OVERLAP','250'))

# ===== app/db.py =====
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase,sessionmaker
from .config import DATABASE_URL
args={'check_same_thread':False} if DATABASE_URL.startswith('sqlite') else {}
engine=create_engine(DATABASE_URL,pool_pre_ping=True,connect_args=args)
SessionLocal=sessionmaker(bind=engine,autoflush=False,autocommit=False)
class Base(DeclarativeBase): pass
def get_db():
 db=SessionLocal()
 try: yield db
 finally: db.close()

# ===== app/models.py =====
from datetime import datetime,timezone
from sqlalchemy import String,Text,Integer,Boolean,DateTime,ForeignKey
from sqlalchemy.orm import Mapped,mapped_column,relationship
from .db import Base
def now(): return datetime.now(timezone.utc)
class User(Base):
 __tablename__='users'; id:Mapped[int]=mapped_column(primary_key=True); username:Mapped[str]=mapped_column(String(80),unique=True,index=True); password_hash:Mapped[str]=mapped_column(String(255)); role:Mapped[str]=mapped_column(String(30),default='viewer'); active:Mapped[bool]=mapped_column(Boolean,default=True); created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)
class Regulation(Base):
 __tablename__='regulations'; id:Mapped[int]=mapped_column(primary_key=True); title:Mapped[str]=mapped_column(String(300)); category:Mapped[str]=mapped_column(String(150),default=''); language:Mapped[str]=mapped_column(String(20),default='ar'); filename:Mapped[str]=mapped_column(String(300)); page_count:Mapped[int]=mapped_column(Integer,default=0); active:Mapped[bool]=mapped_column(Boolean,default=True); created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now); chunks=relationship('Chunk',back_populates='regulation',cascade='all, delete-orphan')
class Chunk(Base):
 __tablename__='chunks'; id:Mapped[int]=mapped_column(primary_key=True); regulation_id:Mapped[int]=mapped_column(ForeignKey('regulations.id'),index=True); chunk_index:Mapped[int]=mapped_column(Integer); page_number:Mapped[int]=mapped_column(Integer,default=0); text:Mapped[str]=mapped_column(Text); embedding_json:Mapped[str|None]=mapped_column(Text,nullable=True); regulation=relationship('Regulation',back_populates='chunks')
class ChatLog(Base):
 __tablename__='chat_logs'; id:Mapped[int]=mapped_column(primary_key=True); username:Mapped[str]=mapped_column(String(80)); question:Mapped[str]=mapped_column(Text); answer:Mapped[str]=mapped_column(Text); sources_json:Mapped[str]=mapped_column(Text,default='[]'); created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)

# ===== app/security.py =====
import base64,hashlib,hmac,os
from itsdangerous import URLSafeSerializer
from .config import SECRET_KEY
ser=URLSafeSerializer(SECRET_KEY,salt='university-regulations-session')
def hash_password(p):
 salt=os.urandom(16); rounds=310000; d=hashlib.pbkdf2_hmac('sha256',p.encode(),salt,rounds)
 return f'pbkdf2_sha256${rounds}${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(d).decode()}'
def verify_password(p,e):
 try:
  _,r,s,d=e.split('$'); salt=base64.urlsafe_b64decode(s); exp=base64.urlsafe_b64decode(d); act=hashlib.pbkdf2_hmac('sha256',p.encode(),salt,int(r)); return hmac.compare_digest(act,exp)
 except Exception:return False
def make_session(uid):return ser.dumps({'user_id':uid})
def read_session(v):
 try:return ser.loads(v).get('user_id') if v else None
 except Exception:return None

# ===== app/rag.py =====
import io,json,math,re
from pypdf import PdfReader
from docx import Document
from .config import OPENAI_API_KEY,OPENAI_MODEL,OPENAI_EMBEDDING_MODEL,TOP_K,CHUNK_SIZE,CHUNK_OVERLAP
try: from openai import OpenAI
except Exception: OpenAI=None
def norm(t):return re.sub(r'\s+',' ',t or '').strip()
def chunk_text(text,size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):
 text=norm(text); out=[]; start=0
 while start<len(text):
  end=min(start+size,len(text)); piece=text[start:end]
  if end<len(text):
   cut=max(piece.rfind('. '),piece.rfind('؛'),piece.rfind(' '))
   if cut>size*.55:end=start+cut+1;piece=text[start:end]
  out.append(piece.strip())
  if end>=len(text):break
  start=max(end-overlap,start+1)
 return out
def extract_file(name,data):
 n=name.lower()
 if n.endswith('.pdf'):
  r=PdfReader(io.BytesIO(data));return [(i+1,norm(p.extract_text() or '')) for i,p in enumerate(r.pages)]
 if n.endswith('.docx'):
  d=Document(io.BytesIO(data));return [(1,norm('\n'.join(p.text for p in d.paragraphs)))]
 if n.endswith('.txt'):return [(1,norm(data.decode('utf-8','ignore')))]
 raise ValueError('الصيغ المدعومة: PDF, DOCX, TXT')
def lexical(q,t):
 a=set(re.findall(r'\w+',q.lower(),re.UNICODE));b=set(re.findall(r'\w+',t.lower(),re.UNICODE));return len(a&b)/len(a) if a else 0
def cosine(a,b):
 if not a or not b or len(a)!=len(b):return 0
 x=sum(i*j for i,j in zip(a,b));na=math.sqrt(sum(i*i for i in a));nb=math.sqrt(sum(i*i for i in b));return x/(na*nb) if na and nb else 0
def embed(texts):
 if not OPENAI_API_KEY or OpenAI is None:return [None]*len(texts)
 r=OpenAI(api_key=OPENAI_API_KEY).embeddings.create(model=OPENAI_EMBEDDING_MODEL,input=texts);return [x.embedding for x in r.data]
def retrieve(db,q):
 from .models import Chunk,Regulation
 rows=db.query(Chunk,Regulation).join(Regulation,Chunk.regulation_id==Regulation.id).filter(Regulation.active.is_(True)).all(); qv=None
 if OPENAI_API_KEY and OpenAI:
  try:qv=embed([q])[0]
  except Exception:qv=None
 scored=[]
 for c,r in rows:
  lx=lexical(q,c.text); sem=cosine(qv,json.loads(c.embedding_json)) if qv and c.embedding_json else 0; s=.45*lx+.55*sem if qv else lx;scored.append((s,c,r))
 scored.sort(key=lambda x:x[0],reverse=True)
 return [{'score':round(float(s),4),'text':c.text,'page':c.page_number,'title':r.title,'regulation_id':r.id} for s,c,r in scored[:TOP_K] if s>0]
def ai_answer(q,src):
 if not OPENAI_API_KEY or OpenAI is None:return None
 context='\n\n'.join(f"[المصدر {i+1}] {x['title']} - الصفحة {x['page']}\n{x['text']}" for i,x in enumerate(src))
 instructions='أجب فقط من النصوص المرفقة. إذا لم توجد الإجابة بوضوح فقل إن المعلومات غير موجودة في المقاطع المسترجعة. لا تخترع أرقام مواد أو شروطاً أو تواريخ. اذكر المصدر والصفحة.'
 r=OpenAI(api_key=OPENAI_API_KEY).responses.create(model=OPENAI_MODEL,input=f'{instructions}\n\n{context}\n\nالسؤال: {q}')
 return r.output_text.strip()

# ===== app/main.py =====
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
