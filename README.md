# DSAI2_Misinformation
# Calibrating Truth: Misinformation Detection (BERT + LLM)
Misinformation detection in news articles using BERT and LLMs.

## Goal of the project:
In the current information economy, fake news is very easily produced and difficult to verify manually. Because of this, it is very important to have an automated fake news detection for supporting more reliable information systems. For this reson we leverage BERT and LLMs to create such reliable classificaion pipilines to via training a BERT-based model and refining AI intructions and classification criteria, respectively.

## Architecture
The project has two halves, each contained in a single notebook that can be run in combanation with only the data sets, both trained/evaluated on the KAGGLE ISOT fake and real news datasets.

- **`BERTfinal.ipynb`: supervised classifier.** Fine-tunes a `bert-base-cased` model to label each article Real or Fake. It reports objective metrics (accuracy, precision, recall, F1, confusion matrix), a cross-dataset generalisation test on a separate Kaggle set, an LLM-as-a-judge error analysis, and LIME/SHAP explainability.
- **`LLMfinal.ipynb`: training-free LLM pipeline.** Each article passes through three stages that share one rulebook of evaluation criteria: a **GPT-4o mini** classifier labels it TRUE/FALSE, **RAGAS** scores the quality of that decision, and a **Gemini** judge gives an independent second opinion. An orchestrator writes results (per article) to JSON and computes the similar objective metrics as the BERT half for comparability.

Both notebooks use the same label convention (Real = 1, Fake = 0) and the same cleaning steps, adn results are comprarable

## Setup

1. **Environment:** Python 3 with Jupyter or Google Colab. A GPU is recommended for the BERT notebook.
2. **Dependencies:** each notebook installs its own libraries (BERT: `torch`, `transformers`, `shap`, `lime`, `scikit-learn`, …; LLM: `openai`, `ragas`, `langchain`, `scikit-learn`, `matplotlib`, `seaborn`, …).
3. **Data:** For the BERT notebook, step 3 in the notebook itself uploads the two ISOT dataset files, True.csv and Fake.csv directly into the notebook. The code automatically idenfities which uploaded file contains real news and which contains fake news then reads both files into separate DataFrames. It also prints the number of articles and column names to quickly confirm that the files were loaded correctly.
5. For the LLM notebook, place `True.csv` and `Fake.csv` (included in this repo) in a `data/` folder next to the notebooks. The LLM notebook's preprocessing step builds the combined `data/articles.csv` it evaluates on.
6. **API keys (LLM notebook only):** set `OPENAI_API_KEY` (classifier + RAGAS) and `GEMINI_API_KEY` (judge) as environment variables or in the `Config` cell. Without keys/to avoid making API calls, set `PIPELINE_MOCK=1` to run the full pipeline offline with placeholder outputs.
7. **Run** the cells top to bottom; imports are spread across cells, so run them in order.

## Results:

The BERT classifier reaches very high scores on the ISOT test set and stays strong (~98% accuracy) on the unseen Kaggle set, indicating it generalises rather than overfitting. The LLM pipeline reaches ~95% accuracy (F1 ≈ 0.95) on a stratified 150-article sample, with RAGAS scores and judge agreement providing diagnostics on its reasoning in addition to objective metrics, which can be used for further instruction calibration to improve results.


