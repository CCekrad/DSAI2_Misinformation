DSAI2_Misinformation
# Calibrating Truth: Misinformation Detection

Misinformation detection in news articles using BERT and LLMs.

## Goal of the project:

In the current information economy, fake news is very easily produced and difficult to verify manually. Because of this, it is very important to have an automated fake news detection for supporting more reliable information systems. For this reson, we leverage BERT and LLMs to create such reliable classificaion pipilines to via training a BERT-based model and refining LLM intructions and classification criteria, respectively.

## Architecture:

The project has two halves, each contained in a single notebook that can be run in combanation with the data sets: both halves trained/instructed and evaluated on the KAGGLE ISOT fake and real news datasets. More detail on the notebooks' contents:

- **The notebook `BERTfinal.ipynb` trains and evaluated a supervised classifier.** Fine-tunes a `bert-base-cased` model to label each article Real or Fake. It reports objective metrics (accuracy, precision, recall, F1, confusion matrix), a cross-dataset generalisation test on a separate Kaggle set, an LLM-as-a-judge error analysis, and LIME/SHAP explainability.
- **The notebook `LLMfinal.ipynb` created a training-free (but iteratively fine-tanuable) LLM pipeline.** Each article passes through three stages that share the same evaluation criteria: 1) a GPT classifier labels it TRUE/FALSE, 2) RAGAS scores the quality of that decision, and 3) a Gemini judge gives an independent second opinion. An orchestrator saves results (per article) in JSON from and computes similar objective metrics as the BERT exercise for comparability.

Both notebooks use the same label convention (Real = 1, Fake = 0) and the same data cleaning / pre-processing steps, with outcome metrics also being comparable.

## Setup:

1. **Environment:** Python 3 with Jupyter or Google Colab. A GPU is recommended for the BERT notebook. High AI rate limits are recommended for running a large portion of the data through the LLM pipeline.
2. **Dependencies:** each notebook installs its own libraries in the first step. The list of libraties used is as follows (in alphabetical order): csv, google.colab, IPython, ipywidgets, json, langchain_openai, lime, matplotlib, numpy, openai, os, pandas, ragas, random, re, scikit-learn (sklearn), seaborn, shap, shutil, subprocess, sys, torch, tqdm, transformers 
3. **Data:**
   - For the BERT notebook, step 3 in the notebook itself uploads the two ISOT dataset files, `True.csv` and `Fake.csv` directly into the notebook. The code automatically idenfities which uploaded file contains real news and which contains fake news then reads both files into separate DataFrames. It also prints the number of articles and column names to quickly confirm that the files were loaded correctly.
   - For the LLM notebook, place `True.csv` and `Fake.csv` in a `data/` folder next to the notebooks. The notebook's preprocessing step builds the combined `data/articles.csv` it uses for its evaluation.
6. **API keys (LLM notebook only):** set `OPENAI_API_KEY` (classifier + RAGAS) and `GEMINI_API_KEY` (judge) - or preferred alternatives - as environment variables or in the `Config` cell. When lacking API key or to avoid making API calls, set `PIPELINE_MOCK=1` to run the full pipeline offline with placeholder outputs.
7. Cells are to be run in order. Thr notebooks are further commentated step by step to provide more context.

## Results:

The BERT classifier reaches very high scores on the ISOT test set and stays strong (~98% accuracy) on the unseen Kaggle set, indicating it generalises rather than overfitting. The LLM pipeline reaches ~95% accuracy (F1 ≈ 0.95) on a stratified 150-article sample, with RAGAS scores and judge agreement providing diagnostics on its reasoning in addition to objective metrics, which can be used for further instruction calibration to improve results.


