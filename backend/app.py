from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import faiss
import re
import os
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# -------------------------
# Base Paths
# -------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_PATH = os.path.join(BASE_DIR, "data", "marketing_sample_for_amazon_com-amazon_fashion_products__2023-04-11.json")

# -------------------------
# Serve Frontend
# -------------------------

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_json(DATA_PATH, lines=True)

df = df.rename(columns={
    "product_name": "title",
    "sales_price": "price"
})

df["image"] = df["large"].apply(
    lambda x: x.split("|")[0] if isinstance(x, str) else None
)

df["price"] = pd.to_numeric(df["price"], errors="coerce")

df = df[["title", "brand", "price", "product_url", "image"]]

df = df.dropna(subset=["title"])

df["brand"] = df["brand"].fillna("Unknown")

df = df.reset_index(drop=True)

# -------------------------
# Clean Title (remove brand words)
# -------------------------

df["title_clean"] = df["title"].str.lower()

df["title_clean"] = df.apply(
    lambda x: x["title_clean"].replace(x["brand"].lower(), "")
    if isinstance(x["brand"], str) else x["title_clean"],
    axis=1
)

# -------------------------
# Embedding Model
# -------------------------

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

texts = df["title_clean"].tolist()

embeddings = model.encode(texts).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# -------------------------
# Attribute dictionaries
# -------------------------

colors = [
    "white", "black", "blue", "red", "green", "yellow", "grey", "pink", "brown"
]

categories = {
    "tshirt": ["tshirt", "t-shirt", "tee"],
    "shirt": ["shirt"],
    "shoes": ["shoe", "shoes", "sneaker", "footwear"],
    "jeans": ["jeans", "denim"],
    "pants": ["pant", "trouser"]
}

# -------------------------
# Query Parser
# -------------------------

def parse_query(query):
    q = query.lower()
    filters = {
        "color": None,
        "category": None,
        "price": None
    }

    for c in colors:
        if c in q:
            filters["color"] = c

    for cat, words in categories.items():
        for w in words:
            if w in q:
                filters["category"] = cat

    price = re.search(r'\d+', q)
    if price:
        filters["price"] = int(price.group())

    return filters

# -------------------------
# Keyword Score
# -------------------------

def keyword_score(query, title, brand):
    q_words = set(query.lower().split())
    t_words = set(title.lower().split())
    b_words = set(brand.lower().split()) if isinstance(brand, str) else set()

    title_matches = len(q_words.intersection(t_words))
    brand_only_matches = len(q_words.intersection(b_words) - t_words)

    return title_matches - (0.8 * brand_only_matches)

# -------------------------
# Search API
# -------------------------

@app.route("/search", methods=["POST"])
def search():
    query = request.json["query"]
    filters = parse_query(query)
    filtered_df = df

    if filters["color"]:
        filtered_df = filtered_df[
            filtered_df["title_clean"].str.contains(filters["color"], case=False, regex=True)
        ]

    if filters["category"]:
        words = categories[filters["category"]]
        pattern = "|".join(words)
        filtered_df = filtered_df[
            filtered_df["title_clean"].str.contains(pattern, case=False, regex=True)
        ]

    if filters["price"]:
        filtered_df = filtered_df[
            filtered_df["price"] <= filters["price"]
        ]

    if len(filtered_df) < 5:
        filtered_df = df

    texts = filtered_df["title_clean"].tolist()
    emb = model.encode(texts).astype("float32")
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)

    query_emb = model.encode([query]).astype("float32")
    distances, indices = idx.search(query_emb, 8)

    results = []
    for rank, i in enumerate(indices[0]):
        p = filtered_df.iloc[i]
        semantic_score = 1 / (1 + distances[0][rank])
        keyword = keyword_score(query, p["title"], p["brand"])
        final_score = 0.7 * semantic_score + 0.3 * keyword
        results.append((final_score, p))

    results = sorted(results, key=lambda x: x[0], reverse=True)

    output = []
    for score, p in results:
        output.append({
            "title": p["title"],
            "brand": p["brand"],
            "price": int(p["price"]) if not pd.isna(p["price"]) else "N/A",
            "url": p["product_url"],
            "image": p["image"]
        })

    return jsonify(output)

# -------------------------

if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:10000")
    app.run(debug=False)
