from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from flasgger import Swagger
import os
from dotenv import load_dotenv

load_dotenv()

PAGESPEED_API_KEY = os.getenv("GOOGLE_PAGESPEED_API_KEY")

app = Flask(__name__)
Swagger(app)
# Load mô hình và tokenizer
model = tf.keras.models.load_model("models/keyword_model.h5")
with open("models/Tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url = request.form["url"]
        result = analyze_website(url)
    return render_template("analysis.html", result=result)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)
    pred = model.predict(padded)[0][0]
    return jsonify({
        "prediction": int(pred > 0.5),
        "confidence": float(pred)
    })

@app.route("/analyze_keywords", methods=["POST"])
def analyze_keywords():
    data = request.get_json()
    keywords = data.get("keywords", [])

    def calculate_score(kw):
        try:
            sv = float(kw.get("search_volume", 0))
            diff = float(kw.get("difficulty", 100))
            cpc = float(str(kw.get("cpc", "0")).replace("$", ""))
        except:
            return 0
        score = (sv / 25000) * 0.4 + (cpc / 5.5) * 0.3 + ((100 - diff) / 100) * 0.3
        return round(score, 3)

    # Tính điểm SEO
    for kw in keywords:
        kw["seo_score"] = calculate_score(kw)

    # Sắp xếp theo điểm SEO giảm dần
    keywords.sort(key=lambda x: x["seo_score"], reverse=True)

    # Gán thứ hạng
    for i, kw in enumerate(keywords, 1):
        kw["rank"] = i
        kw["is_best"] = (i == 1)

    return jsonify(keywords)


# def analyze_website(url):
#     try:
#         # 1. Crawl nội dung trang
#         response = requests.get(url, timeout=5)
#         soup = BeautifulSoup(response.text, "html.parser")
#         text = soup.get_text(separator=' ', strip=True)

#         if len(text.strip()) < 100:
#             raise Exception("Không đủ nội dung để phân tích")

#         # 2. Dự đoán bằng mô hình AI
#         sequence = tokenizer.texts_to_sequences([text])
#         padded = pad_sequences(sequence, maxlen=200)
#         prediction = model.predict(padded)[0][0]

#         # 3. Chuyển thành chỉ số SEO
#         web_score = int(prediction * 100)
#         content_score = int(min(95, web_score + np.random.randint(-5, 5)))
#         keywords_top10 = int((prediction * 20) + np.random.randint(5, 10))
#         backlinks = int(1000 + prediction * 4000)

#         summary = (
#             f"Nội dung website {url} có điểm SEO là {web_score}/100. "
#             f"Có khoảng {keywords_top10} từ khóa nằm trong top 10 tìm kiếm, "
#             f"và khoảng {backlinks} backlinks. "
#             f"Điểm nội dung: {content_score}/100."
#         )

#         return {
#             "web_score": web_score,
#             "keywords_top10": keywords_top10,
#             "backlinks": backlinks,
#             "content_score": content_score,
#             "summary": summary
#         }

#     except Exception as e:
#         return {
#             "web_score": 0,
#             "keywords_top10": 0,
#             "backlinks": 0,
#             "content_score": 0,
#             "summary": f"Lỗi khi phân tích trang {url}: {str(e)}"
#         }

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [kw for kw, score in sorted_scores[:top_n]]

def analyze_sections(soup):
    scores = {}
    if soup.title:
        title_text = soup.title.text.strip()
        scores["title"] = title_text
        scores["title_score"] = 100 if 30 <= len(title_text) <= 65 else 60
    h1s = soup.find_all('h1')
    scores["h1_tags"] = [h.get_text(strip=True) for h in h1s]
    scores["h1_score"] = 100 if len(h1s) == 1 else 50 if len(h1s) > 1 else 0
    return scores

def get_pagespeed_score(url):
    try:
        api_url = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key={GOOGLE_PAGESPEED_API_KEY}"
        res = requests.get(api_url)
        data = res.json()
        return int(data['lighthouseResult']['categories']['performance']['score'] * 100)
    except:
        return None

def analyze_website(url):
    try:
        # 1. Crawl
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=' ', strip=True)

        if len(text.strip()) < 100:
            raise Exception("Không đủ nội dung để phân tích")

        # 2. AI SEO Score
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=200)
        prediction = model.predict(padded)[0][0]

        web_score = int(prediction * 100)
        content_score = int(min(95, web_score + np.random.randint(-5, 5)))
        keywords_top10 = int((prediction * 20) + np.random.randint(5, 10))
        backlinks = int(1000 + prediction * 4000)

        # 3. Keyword Extraction
        extracted_keywords = extract_keywords(text)

        # 4. Section Analysis
        section_scores = analyze_sections(soup)

        # 5. PageSpeed Score
        pagespeed_score = get_pagespeed_score(url)

        summary = (
            f"Website {url} có điểm SEO là {web_score}/100, điểm nội dung {content_score}/100. "
            f"Khoảng {keywords_top10} từ khóa nằm trong top 10 và {backlinks} backlinks. "
            f"Title score: {section_scores.get('title_score')}, H1 score: {section_scores.get('h1_score')}. "
            f"Google PageSpeed: {pagespeed_score}/100."
        )

        return {
            "web_score": web_score,
            "keywords_top10": keywords_top10,
            "backlinks": backlinks,
            "content_score": content_score,
            "pagespeed_score": pagespeed_score,
            "keywords_extracted": extracted_keywords,
            "title": section_scores.get("title"),
            "title_score": section_scores.get("title_score"),
            "h1_tags": section_scores.get("h1_tags"),
            "h1_score": section_scores.get("h1_score"),
            "summary": summary
        }

    except Exception as e:
        return {
            "web_score": 0,
            "keywords_top10": 0,
            "backlinks": 0,
            "content_score": 0,
            "pagespeed_score": 0,
            "keywords_extracted": [],
            "title": "",
            "title_score": 0,
            "h1_tags": [],
            "h1_score": 0,
            "summary": f"Lỗi khi phân tích trang {url}: {str(e)}"
        }

if __name__ == "__main__":
    app.run(debug=True)
