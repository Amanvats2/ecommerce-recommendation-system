E-Commerce Personalized Product Recommendation System
 Project Overview
This project builds an AI-powered product recommendation system for e-commerce platforms.
It recommends similar products based on a user's search query using semantic similarity.
The system uses Natural Language Processing (NLP) with Sentence Transformers and FAISS for fast similarity search.

Features
* Semantic product search
* Personalized product recommendations
* Price range filtering
* Keyword-based filtering
* Product image display
* Amazon product links

 Technologies Used
* Python
* Streamlit
* Pandas
* NumPy
* FAISS
* Sentence Transformers
* Machine Learning / NLP




?? Project Structure
ecommerce-recommendation-system/
?
??? app.py
??? requirements.txt
??? README.md
?
??? data/
?   ??? marketing_sample_for_amazon_com-amazon_fashion_products.json

?? Installation & Setup
1?? Clone the repository
git clone https://github.com/yourusername/ecommerce-product-recommendation-system.git
2?? Install dependencies
pip install -r requirements.txt
3?? Run the application
streamlit run app.py

 How the Recommendation System Works
1. Product data is loaded from the dataset.
2. Product titles and brands are converted into embeddings using Sentence Transformers.
3. FAISS builds a similarity index for fast search.
4. When a user enters a query:
o The query is converted into an embedding.
o FAISS finds the most similar products.
5. The system displays the top recommended products.

?? Dataset
The dataset contains Amazon fashion product information including:
* Product title
* Brand
* Price
* Product URL
* Product images

?? Future Enhancements
* User login and personalization
* Collaborative filtering
* Hybrid recommendation system
* Deep learning recommendation models
* Deployment on cloud platforms

 Author
Aman Vats

