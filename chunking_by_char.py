#!/usr/bin/env python
# ================================================================
# char_chunk.py — 文字数ベースの単純チャンキング
#   ・ docs/*.txt → Database_csv/chunks_by_char.csv
# ================================================================

import csv
from pathlib import Path

# ---------------------
# パラメータ
# ---------------------
INPUT_DIR   = Path("docs")               # Wikipedia テキストを置くフォルダ
OUTPUT_DIR  = Path("Database_csv")       # ⬅ 追加: 出力ディレクトリ
OUTPUT_NAME = "chunks_by_char.csv"
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_NAME

CHUNK_SIZE  = 400                        # 1 チャンクあたりの文字数
OVERLAP     = 100                        # オーバーラップ文字数

assert 0 <= OVERLAP < CHUNK_SIZE, "OVERLAP は 0 以上 CHUNK_SIZE 未満にしてください"

# ---------------------
# ヘルパ
# ---------------------
def char_chunks(text: str, size: int, overlap: int):
    """固定文字幅スライドでテキストを yield"""
    step = size - overlap
    for start in range(0, len(text), step):
        yield text[start : start + size]

# ---------------------
# メイン
# ---------------------
def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"{INPUT_DIR} が見つかりません。")

    OUTPUT_DIR.mkdir(exist_ok=True)      # ⬅ 出力フォルダを確保

    rows, cid = [], 0
    for fp in INPUT_DIR.glob("*.txt"):
        txt = fp.read_text(encoding="utf-8").replace("\u3000", " ")
        for chunk in char_chunks(txt, CHUNK_SIZE, OVERLAP):
            rows.append((cid, chunk)); cid += 1

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "text"])
        writer.writerows(rows)

    print(f"✓ {len(rows)} チャンクを書き出しました → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
