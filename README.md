# Lab 6A — NER Pipeline

Module 6 Week A lab for AI.SPIRE Applied AI & ML Systems.

Build and compare Named Entity Recognition (NER) pipelines using spaCy and Hugging Face on climate-related text data.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

This installs spaCy and Hugging Face transformers for the first time in this course. The spaCy model download is ~12MB. The first run of the HF NER model will download ~250MB of model weights — this is a one-time download.

## Tasks

Complete the six functions in `ner_pipeline.py`:
1. `load_data` — Load the climate articles dataset
2. `preprocess_text` — Normalize, tokenize, and lemmatize text with spaCy
3. `extract_spacy_entities` — Extract entities using spaCy NER
4. `extract_hf_entities` — Extract entities using Hugging Face NER
5. `compare_ner_outputs` — Compare entity counts between systems
6. `evaluate_ner` — Compute entity-level precision, recall, F1

## Submission

1. Create a branch: `lab-6a-ner-pipeline`
2. Complete `ner_pipeline.py`
3. Open a PR to `main`
4. Paste your PR URL into TalentLMS → Module 6 Week A → Lab 6A

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
