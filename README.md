# Twitter Sentiment Analysis

Streamlit app for classifying tweets as offensive or not offensive with a
trained machine learning model.

## Project Files

```text
.
├── app.py
├── requirements.txt
├── trained_model.pkl
├── vectorizer.pkl
└── save_vectorizer.py
```

## Quick Start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Notes

- Keep `trained_model.pkl` and `vectorizer.pkl` next to `app.py`.
- The app also checks common vectorizer filenames such as `tfidf_vectorizer.pkl`
  and `vectorizer.sav`.
- Use the sample buttons in the sidebar to test the prediction flow quickly.
