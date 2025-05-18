#!/usr/bin/env python
# simple_rag_vs_norag.py
#   ・質問ごとに RAG 答え & 非 RAG 答え を生成
#   ・標準出力へ表示 + results_rag_vs_norag.csv に保存

import os, glob, csv, time, torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# 1. ドキュメント読み込み（chunk_id,text CSV）
DOC_DIR = "Database_csv"
texts = []
for path in glob.glob(os.path.join(DOC_DIR, "**/*.csv"), recursive=True):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"].strip())
print(f"Loaded {len(texts)} documents.")

# ─────────────────────────────────────────────
# 2. Embedding & FAISS
embedder  = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
db        = FAISS.from_texts(texts, embedder)
retriever = db.as_retriever(search_kwargs={"k": 4})

# ─────────────────────────────────────────────
# 3. LLM
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL  = "elyza/Llama-3-ELYZA-JP-8B"

tok = AutoTokenizer.from_pretrained(MODEL)
llm = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16 if "cuda" in DEVICE else torch.float32
).to(DEVICE)

# ▼ pad/eos を必ず設定
tok.eos_token_id = tok.eos_token_id or 2          # fallback 2
tok.pad_token_id = tok.pad_token_id or tok.eos_token_id
llm.config.eos_token_id = tok.eos_token_id
llm.config.pad_token_id = tok.pad_token_id

# ─────────────────────────────────────────────
# 4. QA 実行
QA_CSV      = "QA/QA.csv"
OUT_CSV     = "results_rag_vs_norag.csv"
SYS_PROMPT  = "参考情報を元に質問に答えてください。"
NORAG_SYS_PROMPT = "質問に回答してください。"

header = [
    "question",
    "generated_answer_rag",
    "generated_answer_no_rag",
    "reference_answer",
    "retrieved_chunks"
]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(header)

with open(QA_CSV, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for idx, (question, ref_answer, *_rest) in enumerate(reader, start=1):
        # ========== RAG あり ==========
        docs = retriever.invoke(question)
        context   = "\n".join(d.page_content for d in docs)
        user_rag  = f"参考情報：{context}\n質問：{question}"

        prompt_rag = tok.apply_chat_template(
            [{"role":"system","content":SYS_PROMPT},
             {"role":"user",  "content":user_rag}],
            tokenize=False, add_generation_prompt=True
        )
        ids_rag = tok(prompt_rag, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        ids_rag.pop("token_type_ids", None)
        out_rag = llm.generate(
            **ids_rag, max_new_tokens=256, do_sample=True,
            top_k=40, temperature=0.7,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
        )
        ans_rag = tok.decode(out_rag[0], skip_special_tokens=True).split("assistant")[-1].strip()

        # ========== RAG なし ==========
        user_plain = f"質問：{question}"
        prompt_plain = tok.apply_chat_template(
            [{"role":"system","content":NORAG_SYS_PROMPT},
             {"role":"user",  "content":user_plain}],
            tokenize=False, add_generation_prompt=True
        )
        ids_plain = tok(prompt_plain, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        ids_plain.pop("token_type_ids", None)
        out_plain = llm.generate(
            **ids_plain, max_new_tokens=256, do_sample=True,
            top_k=40, temperature=0.7,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
        )
        ans_plain = tok.decode(out_plain[0], skip_special_tokens=True).split("assistant")[-1].strip()

        # ========== 出力 ==========
        print(f"\n=== Q{idx} ===")
        print("【Question】", question)
        print("── RAG answer ──")
        print(ans_rag)
        print("── No-RAG answer ──")
        print(ans_plain)
        print("── Reference ──")
        print(ref_answer)
        print("── Retrieved chunks ──")
        for i, c in enumerate(docs, 1):
            preview = c.page_content[:120] + ("…" if len(c.page_content) > 120 else "")
            print(f"<Chunk {i}> {preview}")

        # ========== CSV 追記 ==========
        with open(OUT_CSV, "a", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow([
                question,
                ans_rag,
                ans_plain,
                ref_answer,
                "\n".join(d.page_content for d in docs)
            ])

print(f"\n✅ 完了。結果を {OUT_CSV} に保存しました。")
