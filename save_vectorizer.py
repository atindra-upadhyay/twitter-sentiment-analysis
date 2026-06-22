"""
Run this in the same notebook/script where you trained the model.
Replace `vectorizer` with the variable name you used (often tfidf or tfidf_vectorizer).
"""

import pickle

# Example after training:
# vectorizer = TfidfVectorizer(...)
# vectorizer.fit(X_train)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Saved vectorizer.pkl")
