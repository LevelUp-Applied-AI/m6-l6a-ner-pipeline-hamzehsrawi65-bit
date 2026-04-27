"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import unicodedata

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
    # Read the dataset from the provided CSV file
    df = pd.read_csv(filepath)

    return df


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    # Count words in each text safely
    word_counts = df["text"].fillna("").apply(lambda text: len(str(text).split()))

    summary = {
        "shape": df.shape,
        "lang_counts": df["language"].value_counts().to_dict(),
        "category_counts": df["category"].value_counts().to_dict(),
        "text_length_stats": {
            "mean": float(word_counts.mean()),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
        },
    }

    return summary


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    # Normalize Unicode using NFC normalization
    normalized_text = unicodedata.normalize("NFC", str(text))

    # Process the normalized text with spaCy
    doc = nlp(normalized_text)

    cleaned_tokens = []

    for token in doc:
        # Remove punctuation and whitespace tokens
        if token.is_punct or token.is_space:
            continue

        # Use lemma, lowercase it, and keep it
        lemma = token.lemma_.lower()
        cleaned_tokens.append(lemma)

    return cleaned_tokens


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    rows = []

    # spaCy English model should only process English rows
    english_df = df[df["language"] == "en"]

    for _, row in english_df.iterrows():
        text_id = row["id"]
        text = row["text"]

        doc = nlp(str(text))

        for ent in doc.ents:
            rows.append(
                {
                    "text_id": text_id,
                    "entity_text": ent.text,
                    "entity_label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"],
    )


def _clean_hf_label(label):
    """Remove IOB prefix from Hugging Face labels.

    Example:
        B-ORG -> ORG
        I-LOC -> LOC
    """
    if "-" in label:
        return label.split("-", 1)[1]
    return label


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    rows = []

    # Hugging Face model here is English NER, so filter English rows only
    english_df = df[df["language"] == "en"]

    for _, row in english_df.iterrows():
        text_id = row["id"]
        text = str(row["text"])

        raw_entities = ner_pipeline(text)

        current_entity = None

        for item in raw_entities:
            word = item["word"]
            label = _clean_hf_label(item["entity"])
            start = item["start"]
            end = item["end"]

            # If this is a subword token, merge it with the current entity.
            # Example: IP + ##CC => IPCC
            if word.startswith("##") and current_entity is not None:
                current_entity["entity_text"] += word[2:]
                current_entity["end_char"] = end
                continue

            # If the label continues the same entity and the token is close,
            # merge it into the current entity.
            if (
                current_entity is not None
                and current_entity["entity_label"] == label
                and start <= current_entity["end_char"] + 1
            ):
                gap_text = text[current_entity["end_char"]:start]

                # Preserve normal spacing between words
                if gap_text.strip() == "":
                    current_entity["entity_text"] += gap_text + word
                    current_entity["end_char"] = end
                    continue

            # Save the previous entity before starting a new one
            if current_entity is not None:
                rows.append(current_entity)

            current_entity = {
                "text_id": text_id,
                "entity_text": word,
                "entity_label": label,
                "start_char": start,
                "end_char": end,
            }

        # Add the final entity for this text
        if current_entity is not None:
            rows.append(current_entity)

    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"],
    )


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    spacy_counts = spacy_df["entity_label"].value_counts().to_dict()
    hf_counts = hf_df["entity_label"].value_counts().to_dict()

    total_spacy = len(spacy_df)
    total_hf = len(hf_df)

    # Compare by text_id and entity_text only, as required
    spacy_set = set(zip(spacy_df["text_id"], spacy_df["entity_text"]))
    hf_set = set(zip(hf_df["text_id"], hf_df["entity_text"]))

    both = spacy_set.intersection(hf_set)
    spacy_only = spacy_set.difference(hf_set)
    hf_only = hf_set.difference(spacy_set)

    comparison = {
        "spacy_counts": spacy_counts,
        "hf_counts": hf_counts,
        "total_spacy": total_spacy,
        "total_hf": total_hf,
        "both": both,
        "spacy_only": spacy_only,
        "hf_only": hf_only,
    }

    return comparison


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
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    # Only compare using text_id, entity_text, and entity_label
    predicted_set = set(
        zip(
            predicted_df["text_id"],
            predicted_df["entity_text"],
            predicted_df["entity_label"],
        )
    )

    gold_set = set(
        zip(
            gold_df["text_id"],
            gold_df["entity_text"],
            gold_df["entity_label"],
        )
    )

    true_positives = len(predicted_set.intersection(gold_set))
    false_positives = len(predicted_set.difference(gold_set))
    false_negatives = len(gold_set.difference(predicted_set))

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")
            print("spaCy counts by label:")
            print(spacy_entities["entity_label"].value_counts())

        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"\nHF entities: {len(hf_entities)} total")
            print("HF counts by label:")
            print(hf_entities["entity_label"].value_counts())

        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")

        if spacy_entities is not None:
            spacy_metrics = evaluate_ner(spacy_entities, gold)
            if spacy_metrics is not None:
                print(f"\nspaCy evaluation: {spacy_metrics}")

        if hf_entities is not None:
            hf_metrics = evaluate_ner(hf_entities, gold)
            if hf_metrics is not None:
                print(f"HF evaluation: {hf_metrics}")