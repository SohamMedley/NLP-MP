# DocuRAG — Presentation Script (Team of 4)

**Project:** RAG-Based Document Question Answering System Using NLP
**Subject:** Natural Language Processing (Mini Project)
**Branch:** CSE (AI & ML)
**Total time:** about 12–15 minutes (3 minutes per member, plus demo and Q&A)

> Replace *Member 1–4* with your names. Speak naturally. You don't need to read word for word.

---

## Work division at a glance

| Member | Role | Files they own | NLP concepts they explain |
|---|---|---|---|
| **Member 1** | Introduction, problem statement & document processing | `rag/pdf_processor.py`, upload validation in `app.py` | Text extraction, preprocessing, normalisation |
| **Member 2** | Chunking & embeddings | `rag/chunker.py`, `rag/embeddings.py` | Sentence segmentation, text chunking, sentence embeddings |
| **Member 3** | Vector search, retrieval & LLM generation | `rag/vector_store.py`, `rag/retriever.py`, `rag/generator.py` | Semantic similarity, information retrieval, grounded generation, hallucination control |
| **Member 4** | Frontend, API, deployment & live demo | `templates/`, `static/`, `app.py` routes, `render.yaml` | Source attribution, context-aware chat, evaluation |

---

## 🎤 MEMBER 1 — Introduction, Problem & Document Processing (≈3 min)

**Opening**

"Good morning, Ma'am. We are a team of four from CSE AI & ML, and our NLP mini project is **DocuRAG, a RAG-Based Document Question Answering System**.

In simple words, you upload your own documents, like lecture notes, research papers or company handbooks, and ask questions in plain English. The system answers **only from those documents** and shows the exact **document and page** each answer came from."

**Problem statement**

"We noticed two problems.

First, **keyword search (Ctrl+F) is not enough**. If the notes say 'term weighting' and I search for 'importance of words', keyword search finds nothing, even though the meaning is the same.

Second, **chatbots like ChatGPT don't know our private documents**, and sometimes they **hallucinate**: they give confident answers that are wrong.

So our objective was a system that **understands meaning**, **searches our own documents**, and **answers only from them, with proof**."

**What RAG is**

"The technique we used is **Retrieval-Augmented Generation (RAG)**. We don't send the whole document to the AI. We first **retrieve** only the most relevant passages, then give just those to the language model to **generate** the answer. That makes it more accurate, cheaper and verifiable."

**My part — document processing**

"My part was the first stage of the pipeline: **getting clean text out of documents**.

1. **Upload and validation.** Users can upload **PDF and TXT** files. We don't trust the file extension alone. For PDFs we check the first bytes of the file, called the *magic number* `%PDF-`. For TXT files we check that it's really text and not a binary file renamed to .txt. We also sanitise filenames to prevent *path traversal* attacks, and we limit uploads to 50 MB.

2. **Text extraction.** For PDFs we use the **PyMuPDF** library and extract text **page by page**, so every piece of text remembers its **page number**. That's what makes citations possible later. For TXT files, which have no pages, we split the text into **sections** on paragraph boundaries, and we support different text encodings such as UTF-8 and Windows-1252.

3. **Text preprocessing (NLP).** Raw PDF text is messy, so we apply **text normalisation**:
   - We join words broken across lines, so 'embed-' plus 'ding' becomes 'embedding'.
   - We remove page-number lines and invisible control characters.
   - We fix ligature characters, where 'ﬁ' becomes 'fi'.
   - We normalise extra spaces and line breaks, but **keep paragraph boundaries**.

   We deliberately **do not** remove stop words or punctuation here, because modern embedding models need full sentences to understand meaning. Removing words like 'not' would change the meaning completely.

With that, I'll hand over to *Member 2*, who will explain how this clean text is converted into something a computer can search by meaning."

---

## 🎤 MEMBER 2 — Chunking & Embeddings (≈3 min)

"Thank you. My part covers two important NLP steps: **text segmentation (chunking)** and **embeddings**."

**Chunking**

"A document can have thousands of words, but we can't search or send the whole thing at once. So we split it into small passages called **chunks**.

We didn't just cut the text every 1000 characters, because that could cut a sentence in half and destroy its meaning. Instead, we do **sentence-aware chunking**:

1. First we split the text into **sentences** using punctuation rules. That is **sentence segmentation**, a basic NLP task.
2. Then we group whole sentences into chunks of about **1000 characters**.
3. Neighbouring chunks **overlap** by about 150 characters. The last one or two sentences of a chunk are repeated at the start of the next, so context isn't lost at the boundary.
4. A chunk never crosses a page boundary, so every chunk has an **exact page number**.

Each chunk stores its **document name, page number, chunk ID and text**.

Why 1000 characters? Our embedding model reads at most **256 tokens**, which is about 200 words. Longer chunks would be silently cut off, so we sized chunks to fit the model."

**Embeddings**

"Next, the computer has to understand the *meaning* of each chunk. For that we use **sentence embeddings**.

An embedding converts text into a **list of numbers, a vector**, and our model produces **384 numbers per chunk**. Texts with similar meaning get vectors that are **close together** in this 384-dimensional space.

For example, 'How are words weighted?' and 'TF-IDF assigns importance to terms' use different words, but their vectors end up close, because the meaning is similar. That's what beats keyword search.

We use **all-MiniLM-L6-v2**, a Sentence-Transformer model based on BERT's architecture, trained on over a billion sentence pairs. Important points:
- It runs **locally on our server**, so no document text is sent to any paid API. It's **free and private**.
- We run it with **FastEmbed and ONNX Runtime** instead of PyTorch. That uses much less memory, so it fits on a free cloud server.
- The model is loaded **only once** when the server starts, not on every request.
- We **normalise** every vector to length 1, so comparing vectors directly gives **cosine similarity**.

Now *Member 3* will explain how we search these vectors and generate answers."

---

## 🎤 MEMBER 3 — Vector Search, Retrieval & Grounded Generation (≈3.5 min)

"Thank you. My part is the core of RAG: **retrieval** and **generation**, plus how we **control hallucination**."

**Vector index (FAISS)**

"All the chunk vectors are stored in **FAISS**, Facebook AI Similarity Search, a free library for fast vector search.

We use an **inner-product index**. Because our vectors are normalised, inner product equals **cosine similarity**:

> cosine similarity = (A · B) / (|A| × |B|)

A value near **1** means very similar meaning, and near **0** means unrelated.

Each vector has an **ID** that maps back to its **document, page and text**. When a user deletes a document, we remove **both** its vectors and its metadata, so the system never cites a deleted file."

**Retrieval**

"When a user asks a question:
1. We convert the **question** into an embedding with the same model.
2. FAISS finds the **top 5 most similar chunks** (top-K retrieval).
3. We apply a **relevance threshold**. Chunks with similarity below 0.25 are dropped.

We also handle **follow-up questions**. If the user asks 'What is BERT?' and then 'How is it pre-trained?', the word 'it' is ambiguous. So for short or pronoun-based questions, we combine the previous question with the new one before searching. This is a simple form of **reference resolution**.

For **overview questions** like 'Summarize this document' or 'What is the conclusion?', no single passage matches well, so we also include representative passages from the beginning, middle and end of each document."

**Generation with Groq**

"The retrieved chunks are formatted as a **context block** with labels like `[Source 1] Document: NLP_Notes.pdf, Page: 2`, and sent with the question to a **large language model through the Groq API**. We use the open-weight model **GPT-OSS 20B**. If that model is unavailable, the system **automatically falls back** to another one.

The API key is stored as an **environment variable** on the server and is never visible in the browser."

**Hallucination control — the most important part**

"We reduce hallucination with **four layers**:
1. **Threshold gate.** If no chunk is relevant enough, we **don't call the LLM at all** and reply 'I couldn't find sufficiently relevant information'.
2. **Strict system prompt.** The model is instructed to *answer ONLY from the supplied context, not use outside knowledge, not invent facts, and cite sources*.
3. **Low temperature (0.1)**, so answers stay factual rather than creative.
4. **Answer classification.** We label each answer **Grounded**, **Partially answered** or **Not found**, so the user always knows how trustworthy it is.

We also send only the **last 3 question–answer pairs** as chat memory, which keeps the context bounded.

Now *Member 4* will show the application and the live demo."

---

## 🎤 MEMBER 4 — Frontend, API, Deployment & Live Demo (≈4 min)

"Thank you. My part was the **user interface**, the **backend API**, and **deployment**."

**Tech stack**

"- The **frontend** is pure **HTML, CSS and vanilla JavaScript**, with no framework like React.
- The **backend** is **Python Flask**, which provides REST APIs such as `/api/upload`, `/api/ask`, `/api/documents` and `/api/health`.
- In production we use **Gunicorn** as the web server.
- It is deployed on **Render** using a `render.yaml` **Blueprint** file, which is infrastructure as code.
- The whole project runs at **zero cost**. The only external service is the Groq API, which has a free tier."

**UI walkthrough (show on screen)**

"The interface has three panels:
- **Sources** on the left, for uploading documents and seeing stats: documents, pages and chunks.
- **Conversation** in the middle, where we chat.
- **Evidence** on the right, which shows the exact passages used for each answer, with page numbers and **similarity scores**. That is **source attribution**.

When a file is uploaded, the progress stepper shows each real processing stage: *Read → Extract → Chunk → Embed → Index*. Processing runs in a **background thread**, so the interface never freezes.

For security, the answer text is **HTML-escaped** before formatting, so the AI's output can't inject malicious scripts (XSS protection)."

**🖥️ LIVE DEMO (follow this order)**

1. **Upload 3 files** at once: `NLP_Notes.pdf`, `NovaTech_Employee_Handbook.pdf`, `Research_Paper_SolarSense.pdf` (or a `.txt` file).
   *Say:* "You can see each stage live. We now have 3 documents and N chunks indexed."
2. **Ask:** *"What is TF-IDF and how is it calculated?"*
   *Say:* "The answer is grounded and cites NLP_Notes.pdf, page 2. Clicking the citation number highlights the exact passage in the Evidence panel."
3. **Ask:** *"What is the notice period at NovaTech?"*
   *Say:* "NovaTech is a fictional company we made up, so the AI can't know this from its training. The correct answer, 60 days, proves it read our document."
4. **Follow-up:** *"What is BERT?"*, then *"How is it pre-trained?"*
   *Say:* "It understood that 'it' means BERT. That's our bounded chat memory."
5. **Out-of-scope:** *"Who won the 2011 Cricket World Cup?"*
   *Say:* "It refuses, because this isn't in our documents. That's hallucination control."
6. **Turn on Retrieval debug.**
   *Say:* "Here you can see every retrieved chunk with its cosine similarity score, and which ones were below the threshold and not sent to the LLM."
7. **Delete a document**, then ask about it again.
   *Say:* "Its embeddings were removed, so it no longer answers from that file."

**Limitations (be honest, since examiners appreciate it)**

"- On Render's free plan, storage is **temporary**, so documents must be re-uploaded after the server restarts.
- **Scanned PDFs** (images) aren't supported yet, because we don't do OCR.
- All users of one deployment share the same document library."

**Future scope**

"Hybrid search that combines BM25 keywords with embeddings, OCR for scanned PDFs, a re-ranking model, streamed answers, and per-user libraries."

**Conclusion (Member 4 closes)**

"To conclude, DocuRAG shows a complete NLP pipeline: **text preprocessing → segmentation → embeddings → semantic similarity → information retrieval → grounded language generation → source attribution**. It solves a real problem: getting trustworthy answers from your own documents, with proof, at zero cost.

Thank you, Ma'am. We're happy to take questions."

---

## ❓ Likely viva questions & who answers

| Question | Who | Short answer |
|---|---|---|
| What is RAG? | M1 | Retrieve relevant passages first, then generate an answer from them. It reduces hallucination and works on private data. |
| Why not send the whole PDF to the LLM? | M1 | Token limits, cost and lower accuracy (the model gets distracted), and there would be no precise citation. |
| Stemming vs lemmatization — did you use them? | M1 | Stemming chops suffixes; lemmatization uses a dictionary. We didn't apply them, because embedding models need natural full sentences. |
| What is tokenization in your project? | M2 | The embedding model uses a WordPiece tokenizer (from BERT) that splits text into sub-words. Max 256 tokens. |
| Why chunk overlap? | M2 | So information at a chunk boundary isn't lost. The next chunk repeats the last sentences. |
| What is an embedding? Dimension? | M2 | A dense vector representing meaning. all-MiniLM-L6-v2 gives 384 dimensions. |
| Why MiniLM and not OpenAI embeddings? | M2 | Free, local, private, fast, and small enough for a free server. |
| What is cosine similarity? | M3 | The angle between vectors: (A·B)/(‖A‖‖B‖). 1 = same meaning, 0 = unrelated. |
| Why FAISS? | M3 | Free, fast, local vector search from Meta; no paid vector database needed. |
| What is Top-K? Threshold? | M3 | K = 5 most similar chunks; threshold 0.25 drops weak matches, and the LLM isn't called if none pass. |
| How do you prevent hallucination? | M3 | Threshold gate, strict prompt, low temperature, and Grounded / Partial / Not found labels with citations. |
| What is temperature? | M3 | It controls randomness. 0.1 gives more deterministic, factual output. |
| Which LLM? Why Groq? | M3 | GPT-OSS 20B via Groq: very fast inference, free tier, with automatic model fallback. |
| How is the API key secured? | M4 | It's an environment variable on the server and never sent to the browser. |
| How is XSS prevented? | M4 | The AI's output is HTML-escaped before Markdown formatting. |
| What is Gunicorn? Why 1 worker? | M4 | A production WSGI server. One worker, because the vector index lives in memory and must be shared. |
| Difference between TF-IDF and embeddings? | M2/M3 | TF-IDF is sparse and keyword-based, with no synonyms. Embeddings are dense and semantic, so they understand meaning. |

---

## ✅ Pre-demo checklist
- [ ] Render service is awake (open the URL 2 minutes early, because free servers sleep)
- [ ] `/api/health` shows `"groq_configured": true`
- [ ] Test PDFs/TXT are ready on the desktop
- [ ] Each member has read their section at least once
- [ ] Backup: screenshots or a screen recording, in case the internet fails
