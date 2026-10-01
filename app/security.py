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
