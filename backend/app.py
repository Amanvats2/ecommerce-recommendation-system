from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import faiss
import re
import os
from sentence_transformers import SentenceTransformer

print("🚀 App starting...")

app = Flask(__name__)
CORS(app)

# -------------------------
# Paths
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
# Lazy Dataset Load
# -------------------------

df = None

def get_data():
    global df
    if df is None:
        try:
            print("📦 Loading dataset...")
            df = pd.read_json(DATA_PATH, lines=True)

            df.rename(columns={
                "product_name": "title",
                "sales_price": "price"
            }, inplace=True)

            df["image"] = df["large"].apply(
                lambda x: x.split("|")[0] if isinstance(x, str) else None
            )

            df["price"] = pd.to_numeric(df["price"], errors="coerce")

            df = df[["title", "brand", "price", "product_url", "image"]]
            df.dropna(subset=["title"], inplace=True)
            df["brand"] = df["brand"].fillna("Unknown")
            df.reset_index(drop=True, inplace=True)

            df["title_clean"] = df["title"].str.lower()
            df["title_clean"] = df.apply(
                lambda x: x["title_clean"].replace(x["brand"].lower(), ""),
                axis=1
            )

            print("✅ Dataset loaded")

        except Exception as e:
            print("❌ Dataset error:", e)
            df = pd.DataFrame(columns=["title", "brand", "price", "product_url", "image"])

    return df

# -------------------------
# Lazy Model Load (LIGHT)
# -------------------------

model = None

def get_model():
    global model
    if model is None:
        print("🤖 Loading model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight
    return model

# -------------------------
# Filters
# -------------------------

colors = ["white", "black", "blue", "red", "green", "yellow", "grey", "pink", "brown"]

categories = {
    "tshirt": ["tshirt", "t-shirt", "tee"],
    "shirt": ["shirt"],
    "shoes": ["shoe", "shoes", "sneaker", "footwear"],
    "jeans": ["jeans", "denim"],
    "pants": ["pant", "trouser"]
}

def parse_query(query):
    q = query.lower()
    filters = {"color": None, "category": None, "price": None}

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
# API
# -------------------------

@app.route("/search", methods=["POST"])
def search():
    try:
        df = get_data()
        model = get_model()

        query = request.json.get("query", "")
        filters = parse_query(query)
        filtered_df = df

        if not df.empty:

            if filters["color"]:
                filtered_df = filtered_df[
                    filtered_df["title_clean"].str.contains(filters["color"], case=False)
                ]

            if filters["category"]:
                pattern = "|".join(categories[filters["category"]])
                filtered_df = filtered_df[
                    filtered_df["title_clean"].str.contains(pattern, case=False)
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
                score = 1 / (1 + distances[0][rank])
                results.append((score, p))

            results = sorted(results, key=lambda x: x[0], reverse=True)

            return jsonify([
                {
                    "title": p["title"],
                    "brand": p["brand"],
                    "price": int(p["price"]) if not pd.isna(p["price"]) else "N/A",
                    "url": p["product_url"],
                    "image": p["image"]
                }
                for score, p in results
            ])

        return jsonify([])

    except Exception as e:
        print("❌ Search error:", e)
        return jsonify({"error": "Something went wrong"}), 500

# -------------------------
# Local Run
# -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)