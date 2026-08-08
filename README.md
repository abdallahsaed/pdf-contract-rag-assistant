# PDF RAG MVP

أول مشروع RAG — نظام سؤال وجواب بسيط على ملفات PDF.

## الفكرة العامة (Pipeline)

```
PDF → Load → Chunk → Embed → Store (ChromaDB) → Retrieve → LLM (Ollama) → إجابة
```

## الخطوات لما يرجع النت

### 1. تثبيت Ollama
حمّل من https://ollama.com/download/windows وثبته.
تأكد إنه اتثبت:
```
ollama --version
```

نزّل الموديل:
```
ollama pull llama3.2:3b
```

### 2. تجهيز بيئة Python
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. تجربة الموديولات لوحدها (قبل الدمج)
```
cd src
python chunker.py     # يجرب الـ chunking بنص وهمي
python loader.py       # يجرب قراءة PDF حقيقي (لازم تحط PDF في data/uploads/ الأول)
```

## خطة الخطوات الجاية (هنبنيها مع بعض بالترتيب)

- [x] بنية المشروع
- [x] loader.py — قراءة الـ PDF
- [x] chunker.py — تقسيم النص
- [ ] embedder.py — تحويل الـ chunks لـ embeddings
- [ ] vectorstore.py — تخزين واسترجاع من ChromaDB
- [ ] retriever.py — منطق الاسترجاع (top-k, similarity)
- [ ] llm.py — الاتصال بـ Ollama وتوليد الإجابة
- [ ] main.py — ربط كل حاجة مع بعض في pipeline واحد
- [ ] تجربة كاملة end-to-end بـ PDF حقيقي

## ملاحظات مهمة

- **chunk_size و chunk_overlap** في `chunker.py` قيم مبدئية (800/100) —
  هنظبطهم بعد ما نجرب على PDF حقيقي ونشوف جودة الإجابات.
- المشروع ده مقصود يكون **بسيط ومنفصل** عن ContractAI (المشروع الأكبر
  اللي هيستخدم multi-agent architecture لاحقًا).
