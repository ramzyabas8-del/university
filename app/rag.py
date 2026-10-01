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
