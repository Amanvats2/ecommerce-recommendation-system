from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ App is working!"

@app.route("/test")
def test():
    return jsonify({"message": "API working"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)