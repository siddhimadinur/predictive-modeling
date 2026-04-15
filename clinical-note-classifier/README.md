# Clinical Note Classifier

Fine-tune Bio_ClinicalBERT on MTSamples medical transcriptions to predict medical specialty categories.

## Problem Statement

Manual classification of clinical notes by medical specialty is time-consuming and inconsistent. Automating this process with NLP can improve clinical workflow routing and enable better analytics across healthcare systems.

## Architecture

```
                    Clinical Note (text)
                           |
                    [ Tokenizer ]
                           |
              +---------------------------+
              |     Bio_ClinicalBERT      |
              |   (pretrained, 12 layers) |
              +---------------------------+
                           |
                    pooler_output (768)
                           |
                    [ Dropout(0.3) ]
                           |
                  [ Linear(768, N) ]
                           |
                    logits -> softmax
                           |
              Medical Specialty Prediction
```

The model uses [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT), a BERT model pretrained on clinical text from MIMIC-III, with a single linear classification head that predicts the medical specialty. Only specialties with at least 50 samples are kept to avoid tiny classes.

## Dataset

**MTSamples** is a free collection of ~5000 de-identified medical transcription samples spanning ~40 medical specialties. Each sample includes a transcription, medical specialty label, description, and keywords.

**Access:** Download from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) or [mtsamples.com](https://mtsamples.com).

**Required file:**
- `mtsamples.csv` — place in `data/raw/`

If the CSV is not available, the pipeline automatically generates a synthetic dataset of 1000 fake clinical notes for end-to-end testing.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess data

```bash
python data/preprocess.py
```

### 2. Train the model

```bash
python training/train.py
```

### 3. Evaluate on test set

```bash
python training/evaluate.py
```

### 4. Predict on a new note

```bash
python predict.py --note "Patient is a 65 year old male who presented with chest pain and shortness of breath. History of coronary artery disease and hypertension."
```

### 5. Run the API server

```bash
uvicorn api:app --reload
```

Then send requests to `POST /predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"note": "65 year old male with chest pain", "top_k": 3}'
```

## Results

| Model | Macro-F1 | Weighted-F1 | Accuracy |
|-------|----------|-------------|----------|
| Bio_ClinicalBERT (synthetic) | — | — | — |
| Bio_ClinicalBERT (MTSamples) | — | — | — |

## Model Card

- **Base model:** `emilyalsentzer/Bio_ClinicalBERT`
- **Task:** Multi-class classification of medical transcriptions by specialty
- **Training data:** MTSamples medical transcriptions (or synthetic notes for testing)
- **Input:** Raw clinical note text (truncated to 2000 chars, tokenized to 256 tokens)
- **Output:** Probability distribution over filtered medical specialty categories

### Known Limitations

- Only includes specialties with 50+ samples, so rare specialties are excluded
- Assigns a single specialty per note (no multi-label support)
- Performance on synthetic data is not indicative of real-world performance
- Truncation to 256 tokens may lose important information in longer transcriptions
- Class imbalance in MTSamples (Surgery and Orthopedic dominate) may bias predictions
