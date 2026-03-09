import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import re

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="E-Commerce Personalized Product Recommendation System", layout="wide")
st.title("E-Commerce Personalized Product Recommendation System")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_json(
        "data/marketing_sample_for_amazon_com-amazon_fashion_products__2023-04-11.json",
        lines=True
    )

    df = df.rename(columns={
        "product_name": "title",
        "sales_price": "price"
    })

    df["image"] = df["large"].apply(
        lambda x: x.split("|")[0] if isinstance(x, str) else None
    )

    df = df[[
        "title", "brand", "price", "product_url", "image"
    ]]

    df = df.dropna(subset=["title"])
    df["brand"] = df["brand"].fillna("Unknown")

    # Convert price to numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df.reset_index(drop=True)

# -------------------------------
# Keyword-based filtering
# -------------------------------
def keyword_filter(df, query):
    query = query.lower()

    shoe_keywords = ["shoe", "shoes", "footwear", "sneaker", "formal shoes", "loafer"]
    shirt_keywords = ["shirt", "tshirt", "t-shirt", "tee"]
    pant_keywords = ["pant", "jeans", "trouser"]

    if any(k in query for k in shoe_keywords):
        return df[df["title"].str.contains(
            "shoe|shoes|footwear|sneaker|loafer", case=False, regex=True)]

    if any(k in query for k in shirt_keywords):
        return df[df["title"].str.contains(
            "shirt|t-shirt|tshirt|tee", case=False, regex=True)]

    if any(k in query for k in pant_keywords):
        return df[df["title"].str.contains(
            "pant|jeans|trouser", case=False, regex=True)]

    return df

# -------------------------------
# Build Model
# -------------------------------
@st.cache_resource
def build_model(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index

# -------------------------------
# Load Everything
# -------------------------------
df = load_data()

# -------------------------------
# UI Inputs
# -------------------------------
query = st.text_input(
    "🔍 Search product",
    placeholder="e.g. formal shoes, men's t-shirt"
)

min_price, max_price = st.slider(
    "💰 Select price range",
    min_value=0,
    max_value=int(df["price"].max()),
    value=(0, int(df["price"].max()))
)

# -------------------------------
# Recommendation Logic
# -------------------------------
if query:
    # Apply keyword filter
    filtered_df = keyword_filter(df, query)

    # Apply price filter
    filtered_df = filtered_df[
        (filtered_df["price"] >= min_price) &
        (filtered_df["price"] <= max_price)
    ]

    if filtered_df.empty:
        st.warning("No products found for this query and price range.")
    else:
        texts = (filtered_df["title"] + " " + filtered_df["brand"]).tolist()
        model, index = build_model(texts)

        query_embedding = model.encode([query]).astype("float32")
        _, indices = index.search(query_embedding, 5)

        st.subheader("✨ Recommended Products")

        for i in indices[0]:
            product = filtered_df.iloc[i]

            col1, col2 = st.columns([1, 3])

            with col1:
                if product["image"]:
                    st.image(product["image"], width=180)

            with col2:
                st.markdown(f"""
                ### 🛒 {product['title']}
                **Brand:** {product['brand']}  
                **Price:** ₹{int(product['price']) if not pd.isna(product['price']) else "N/A"}  
                🔗 [View on Amazon]({product['product_url']})
                """)

            st.divider()
