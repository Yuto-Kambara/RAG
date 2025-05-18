import wikipedia
from pathlib import Path

wikipedia.set_lang("ja")                 # 日本語版を使う
titles = [
    "ドラゴンボール",
]

out_dir = Path("docs")
out_dir.mkdir(exist_ok=True)

for title in titles:
    page = wikipedia.page(title)
    text = page.content           # 本文（Markup 無しのプレーンテキスト）

    (out_dir / f"{title}.txt").write_text(text, encoding="utf-8")
    print(f"✓ saved {title}.txt")
