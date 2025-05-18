#!/usr/bin/env python
# ================================================================
# chunk_wiki.py — 句点優先 + 文単位オーバーラップ版
#   ・ docs/*.txt → Database_csv/chunks_by_char_and_delim.csv
# ================================================================

import re, csv
from pathlib import Path

# -------- パラメータ --------
INPUT_DIR    = Path("docs")                    # .txt の保存場所
OUTPUT_DIR   = Path("Database_csv")            # ⬅ 追加: 出力ディレクトリ
OUTPUT_NAME  = "chunks_by_char_and_delim.csv"
OUTPUT_PATH  = OUTPUT_DIR / OUTPUT_NAME        # フルパス
MAX_LEN      = 400
OVERLAP_CH   = 100

# -------- 1. 文分割 --------
def sentence_split(text: str):
    pat = r"(?<=[。．\.!?！?])\s+"
    return [s.strip() for s in re.split(pat, text) if s.strip()]

# -------- 2. チャンキング --------
def chunk_sentences(sents):
    chunk, length = [], 0
    for sent in sents:
        if length + len(sent) > MAX_LEN and chunk:
            yield " ".join(chunk)

            # ▼ 文単位オーバーラップ
            overlap, total = [], 0
            for s in reversed(chunk):
                if total + len(s) > OVERLAP_CH:
                    break
                overlap.insert(0, s); total += len(s)

            chunk, length = overlap[:], sum(len(s) for s in overlap)

        chunk.append(sent); length += len(sent)
    if chunk:
        yield " ".join(chunk)

# -------- 3. メイン --------
def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"{INPUT_DIR} が見つかりません")

    OUTPUT_DIR.mkdir(exist_ok=True)            # ⬅ 出力フォルダを確保

    rows, cid = [], 0
    for txt in INPUT_DIR.glob("*.txt"):
        text = txt.read_text(encoding="utf-8")
        for ch in chunk_sentences(sentence_split(text)):
            rows.append([cid, ch]); cid += 1

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["chunk_id", "text"], *rows])

    print(f"✓ {len(rows)} チャンクを書き出しました → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
