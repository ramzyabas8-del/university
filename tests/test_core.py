from app.rag import chunk_text, lexical

def test_chunk_text():
    x=chunk_text('جامعة '*1000,size=200,overlap=30)
    assert len(x)>1 and all(x)

def test_lexical():
    assert lexical('قبول الطلاب','شروط قبول الطلاب في الجامعة')>0
