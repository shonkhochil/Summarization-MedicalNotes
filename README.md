# Clinical Note Summarization with T5

This project demonstrates how to generate a large-scale synthetic dataset of clinical notes and fine-tune a Transformer-based model (T5-small) to automatically summarize them. The system handles three distinct types of medical documentation: **History of Present Illness (H&P)**, **Progress Notes**, and **Discharge Summaries**.

## 📌 Project Overview

In the healthcare domain, summarizing patient encounters is critical but time-consuming. This project automates the process by:
1.  **Generating Synthetic Data**: Creating realistic clinical narratives using structured templates and a clinical knowledge base.
2.  **Fine-Tuning a Seq2Seq Model**: Training a T5 model to map long clinical texts to concise, physician-style summaries.
3.  **Evaluation**: Achieving high performance (ROUGE-1: ~0.81) on a held-out test set.

## 🏗️ Dataset Generation

Since real patient data (PHI) is sensitive and hard to access, we engineered a synthetic dataset generator.

-   **Volume**: 10,000 unique patient encounters (50,000 total notes).
-   **Conditions Covered**: Pneumonia, Heart Failure, Cellulitis.
-   **Note Types**:
    -   **H&P**: Initial admission notes detailing symptoms, history, and physical exam.
    -   **Progress Notes**: Daily updates (Day 1-3) on patient status, vitals, and labs.
    -   **Discharge Summaries**: Final summary of the hospital course and follow-up plan.
-   **Gold Standard**: Each note is paired with a rule-based "ground truth" summary mimicking a senior physician's style.

## 🧠 Model Architecture

-   **Base Model**: `t5-small` (Text-to-Text Transfer Transformer).
-   **Task**: Abstractive Summarization.
-   **Input**: Clinical Note Text (Truncated to 512 tokens).
-   **Output**: Summary Text (Max 128 tokens).

## ⚙️ Training Configuration

-   **Framework**: Hugging Face `transformers` & `datasets`.
-   **Hardware**: Trained on NVIDIA T4 GPU (Google Colab).
-   **Hyperparameters**:
    -   Learning Rate: `2e-4`
    -   Batch Size: `8`
    -   Epochs: `5`
    -   Optimizer: AdamW
    -   Precision: Mixed Precision (FP16)

## 📊 Results

The model achieved exceptional results on the synthetic test set, effectively learning the template logic:

| Metric | Score |
| :--- | :--- |
| **ROUGE-1** | **0.8126** |
| **ROUGE-2** | **0.7989** |
| **ROUGE-L** | **0.8126** |

*Note: High scores reflect the structured nature of the synthetic data. Performance on unstructured real-world data would require domain adaptation.*

## 🚀 Quick Start

### Prerequisites
```bash
pip install transformers datasets accelerate evaluate rouge_score torch
Usage
Generate Data: Run the data generation script to create clinical_notes_dataset.csv.

Train Model: Run the training script to fine-tune T5.

Inference:

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your saved model
model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

text = "HISTORY OF PRESENT ILLNESS: Patient is a 65 year old..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = model.generate(inputs.input_ids, max_length=128)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))


📂 Repository Structure
notebook.ipynb: Complete source code for data generation, training, and evaluation.
clinical_notes_dataset.csv: (Generated) The synthetic dataset.
README.md: Project documentation.

📝 Future Improvements
Expand the Knowledge Base to include more diseases (e.g., Diabetes, COPD).
Introduce Noise (typos, abbreviations) to simulate real-world EHR data.
Experiment with larger models like T5-Base or Bart-Large-CNN.
