# Fake News Classification with BERT

This project implements a BERT-based fake news classification system.

## Project Structure

```
fake_news_classifier/
│
├── data/                      # Processed datasets
│   ├── train.csv             # Training data (80%)
│   └── test.csv              # Test data (20%)
│
├── models/                    # Saved models (created after training)
│   └── bert_fake_news_classifier/
│
├── preprocess_data.py        # Data preprocessing script
├── bert_classifier.py        # BERT model training script
├── predict.py                # Prediction script for new articles
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
└── README.md                 # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the preprocessing script:
```bash
python preprocess_data.py
```

This will:
- Load the True.csv and Fake.csv files
- Add labels (1 for true news, 0 for fake news)
- Combine the datasets
- Split into 80% train and 20% test
- Save to `data/train.csv` and `data/test.csv`

## Data Format

The processed CSV files contain the following columns:
- `Training the BERT Model

After preprocessing, train the BERT classifier:

```bash
python bert_classifier.py
```

This will:
- Load the pre-trained BERT model (bert-base-uncased)
- Fine-tune on your training data
- Evaluate on the test set
- Save the best model to `models/bert_fake_news_classifier/`

### Training Configuration

- **Model**: BERT-base-uncased
- **Max Sequence Length**: 512 tokens
- **Batch Size**: 16
- **Epochs**: 3
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW with linear warmup

The script will show:
- Training progress with loss and accuracy
- Test set performance after each epoch
- Final metrics: accuracy, precision, recall, F1-score
- Confusion matrix

## Making Predictions

Use the trained model to classify new articles:

```bash
python predict.py
```

Or use it in your own code:

```python
from predict import FakeNewsPredictor

# Initialize predictor
predictor = FakeNewsPredictor()

# Classify an article
text = "Breaking news: Scientists make major discovery..."
result = predictor.predict(text)

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- pandas, numpy, scikit-learn
- CUDA-compatible GPU (optional, but recommended for faster training)

After preprocessing, you can:
1. Build a BERT model for classification
2. Fine-tune BERT on the training data
3. Evaluate on the test set
4. Deploy the model

## Dataset Source

Original datasets:
- True.csv: `C:\Users\juliu\Downloads\archive\True.csv`
- Fake.csv: `C:\Users\juliu\Downloads\archive\Fake.csv`
