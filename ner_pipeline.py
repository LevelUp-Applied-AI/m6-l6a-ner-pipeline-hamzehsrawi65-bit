"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    # TODO: Load the CSV and return the DataFrame
    pass


def preprocess_text(text):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using spaCy.

    Args:
        text: Raw text string.

    Returns:
        List of cleaned, lemmatized token strings.
    """
    # TODO: Process text with spaCy, filter punctuation and whitespace,
    #       return lowercased lemmas
    pass


def extract_spacy_entities(texts):
    """Extract named entities from texts using spaCy NER.

    Args:
        texts: List of (text_id, text_string) tuples.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Process each text with spaCy, collect entities into rows,
    #       return as a DataFrame
    pass


def extract_hf_entities(texts):
    """Extract named entities from texts using Hugging Face NER.

    Uses the dslim/bert-base-NER model.

    Args:
        texts: List of (text_id, text_string) tuples.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Create an HF NER pipeline with dslim/bert-base-NER,
    #       process each text, reconstruct entity spans from subword
    #       tokens, return as a DataFrame
    pass


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label → count for spaCy
          'hf_counts': dict of entity_label → count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
    """
    # TODO: Count entities per label for each system, compute totals
    pass


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0–1).
    """
    # TODO: Match predicted entities to gold entities by text_id +
    #       entity_text + entity_label, compute precision/recall/F1
    pass


if __name__ == "__main__":
    # Load data
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} articles, {df['language'].value_counts().to_dict()}")

        # Filter English texts
        en_df = df[df["language"] == "en"]
        texts = list(zip(en_df["id"], en_df["text"]))
        print(f"Processing {len(texts)} English articles")

        # Preprocess sample
        sample = preprocess_text(en_df["text"].iloc[0])
        if sample is not None:
            print(f"Sample preprocessed tokens: {sample[:10]}")

        # spaCy NER
        spacy_entities = extract_spacy_entities(texts)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER
        hf_entities = extract_hf_entities(texts)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nComparison: {comparison}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")
