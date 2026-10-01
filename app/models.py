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
