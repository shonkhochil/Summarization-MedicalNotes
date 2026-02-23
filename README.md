Clinical NLP Pipeline: Classification & Summarization
Project Overview
This project implements a comprehensive Natural Language Processing (NLP) pipeline tailored for the medical domain. It utilizes synthetic clinical data to perform two critical tasks:

Clinical Sequence Classification: Categorizing medical notes into types like HPI, Progress Notes, and Discharge Summaries.
Abstractive Summarization: Generating concise clinical summaries from structured and unstructured medical text.
The notebook demonstrates fine-tuning strategies for domain-specific Transformer models (Bio_ClinicalBERT and BioBART) and evaluates their performance on both short-form and high-fidelity long-form synthetic data.

Notebook Structure
Phase 1: Clinical Sequence Classification
Objective: Classify clinical text into 3 categories: History of Present Illness (HPI), Progress Notes, and Discharge Summaries.
Model: emilyalsentzer/Bio_ClinicalBERT
Data: Synthetic dataset of medical shorthand.
Key Results: Achieved 1.0 Accuracy and 1.0 Weighted F1-Score after optimizing the training pipeline (10 epochs, learning rate 5e-5).
Phase 2: Generative Summarization (Short-Form)
Objective: Generate summaries for structured clinical notes (e.g., "Patient presents with [SYMPTOM]...").
Model: GanjinZero/biobart-base
Data: 10,000 synthetic note-summary pairs generated via template filling.
Key Results: The model successfully learned the summarization patterns for short, structured inputs.
Phase 3: High-Fidelity Long-Form Evaluation
Objective: Stress-test the summarization model on complex, noisy, long-form documentation (300-500 words).
Data: Expanded clinical scenarios with realistic "medical filler" text (nursing reports, lab reconciliations, etc.).
Key Findings:
ROUGE-2 Score: ~0.0 (Near zero)
Insight: The base BioBART model struggled to distinguish critical clinical signals from administrative noise in long contexts, highlighting the need for specific "de-noising" preprocessing or intermediate fine-tuning tasks.
Getting Started
Prerequisites
This notebook is designed to run in Google Colab with a T4 GPU runtime.

Required libraries:

!pip install transformers[torch] datasets peft accelerate evaluate scikit-learn nltk rouge_score
Usage
Open in Colab: Upload the .ipynb file to Google Colab.
Runtime: Change runtime type to T4 GPU.
Run All: Execute cells sequentially. The notebook handles data generation, training, and evaluation automatically.
Models Used
Bio_ClinicalBERT: A BERT model pre-trained on the MIMIC-III dataset.
BioBART: A BART model pre-trained on PubMed abstracts and PMC full-text articles.
License
This project uses synthetic data and open-source models available under their respective licenses (typically Apache 2.0 or MIT).
