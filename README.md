# Lab1

Sentiment classification on Amazon product reviews using TF-IDF and multiple ML classifiers.

## Project Structure

```
Lab1/
├── src/
│   ├── train_models.py      # Train TF-IDF + classifier pipelines
│   ├── evaluate_models.py   # Evaluate saved models on test data
│   └── flag_for_response.py # Flag reviews for human response
├── data/
│   └── Reviews.csv          # Amazon product reviews dataset
├── models/                  # Trained pipelines (.joblib)
├── reports/                 # Confusion matrix plots
├── notebooks/               # Jupyter notebooks
├── requirements.txt
└── README.md
```

## Dataset

[Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) — product reviews with scores 1–5. Sentiment mapping:

- **Negative**: Score 1–2  
- **Neutral**: Score 3  
- **Positive**: Score 4–5  

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train Models

Trains Logistic Regression, Random Forest, and MLP classifiers with a TF-IDF vectorizer. Saves pipelines to `models/` and confusion matrices to `reports/`.

```bash
python src/train_models.py
```

### Evaluate Models

Loads saved pipelines and reports accuracy, F1 (macro/weighted), and classification reports on the test set.

```bash
python src/evaluate_models.py
```

### Flag for Response

Use `flag_for_response()` to mark reviews that may need a human response:

```python
from src.flag_for_response import flag_for_response

# Short negative review → flagged
flag_for_response("Worst product ever.", "negative", length_threshold=120)
# True

# Keyword match → flagged
flag_for_response("Broken after one use.", "neutral", keywords=["broken", "refund"])
# True
```

## Jupyter Notebook

A notebook is available at `notebooks/amazon_reviews_analysis.ipynb` for exploratory analysis, model loading, and `flag_for_response` examples.

**Kernel**: Use the **Python (Lab1)** kernel. To register it:

```bash
python -m ipykernel install --user --name=lab1 --display-name="Python (Lab1)"
```

## Dependencies

- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- joblib  
- ipykernel (for Jupyter)

See `requirements.txt` for versions.
