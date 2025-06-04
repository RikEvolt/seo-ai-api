# AI SEO Keyword Classifier

A simple AI-powered project to classify whether an article is related to SEO using a neural network trained on keyword features.

---

## 📁 Project Structure

```
ai_setup/
├── data/
│   ├── data_with_labels.csv      # Labeled training dataset
│   └── seo_articles.csv          # Raw SEO articles
├── models/                       # Saved trained model
├── templates/                    # Flask HTML templates (if any)
├── tf-seo-env/                   # Virtual environment (not included in Git)
├── utils/                        # Utility scripts (optional)
├── app.py                        # Flask API web server
├── train_model.py                # Main training script
├── data_prep.py                  # Data preprocessing utilities
├── convert_seo_csv.py           # CSV conversion tool (optional)
├── seo_keyword_extractor.py     # Keyword extraction logic
├── train_basic.py               # Basic training baseline (optional)
```

---

## ✅ Prerequisites

* Python 3.8 or 3.9
* pip

---

## ⚙️ Installation

### 1. Create Virtual Environment

```bash
python -m venv tf-seo-env
```

### 2. Activate Environment

#### On Windows:

```bash
.\tf-seo-env\Scripts\activate
```

#### On Mac/Linux:

```bash
source tf-seo-env/bin/activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following:

```
flask
pandas
tensorflow
scikit-learn
```

Then run:

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

Run this command to train the neural network and save the model:

```bash
python train_model.py
```

* Output: Model saved at `models/keyword_model.h5`

---

## 🚀 Run the Flask Web Server

```bash
python app.py
```

Visit your local server:

```
http://localhost:5000
```

---

## 📌 Notes

* You must train the model at least once before running `app.py`
* If you see errors about missing modules, install them with `pip install` manually
* To export the full environment dependencies:

```bash
pip freeze > requirements.txt
```

---

## 📄 License

This project is for educational purposes.

---

Feel free to fork and improve!
