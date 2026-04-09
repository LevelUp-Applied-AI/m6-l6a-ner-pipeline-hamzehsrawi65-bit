"""Autograder tests for Lab 6A — NER Pipeline."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ner_pipeline import (load_data, preprocess_text, extract_spacy_entities,
                          extract_hf_entities, compare_ner_outputs, evaluate_ner)


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "climate_articles.csv")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gold_entities.csv")


@pytest.fixture
def df():
    data = load_data(DATA_PATH)
    assert data is not None, "load_data returned None"
    return data


@pytest.fixture
def en_texts(df):
    en_df = df[df["language"] == "en"]
    return list(zip(en_df["id"], en_df["text"]))


SAMPLE_NER_TEXT = (
    "The IPCC released a report on climate change impacts in Jordan "
    "and the broader Middle East region in March 2024."
)


def test_data_loaded(df):
    """Dataset loads with expected shape and columns."""
    assert df.shape[0] > 50, f"Expected >50 rows, got {df.shape[0]}"
    required_cols = {"id", "text", "source", "language", "category"}
    assert required_cols.issubset(set(df.columns)), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )


def test_preprocess_text_basic():
    """preprocess_text returns lowercase, no punctuation, lemmatized tokens."""
    result = preprocess_text("The IPCC released its latest report.")
    assert result is not None, "preprocess_text returned None"
    assert isinstance(result, list), "Must return a list"
    assert len(result) > 0, "Returned empty list"

    # All tokens should be lowercase strings
    for token in result:
        assert isinstance(token, str), f"Token must be str, got {type(token)}"
        assert token == token.lower(), f"Token '{token}' is not lowercase"

    # Punctuation-only tokens should be removed
    for token in result:
        assert not all(c in ".,;:!?()-'\"" for c in token), (
            f"Punctuation token '{token}' should be removed"
        )


def test_preprocess_handles_unicode():
    """preprocess_text handles non-ASCII characters without crashing."""
    result = preprocess_text("Café résumé naïve Zürich — 2°C target")
    assert result is not None, "preprocess_text returned None on Unicode input"
    assert isinstance(result, list), "Must return a list"
    assert len(result) > 0, "Returned empty list on Unicode input"


def test_spacy_ner_returns_dataframe():
    """extract_spacy_entities returns a DataFrame with required columns."""
    texts = [(1, SAMPLE_NER_TEXT)]
    result = extract_spacy_entities(texts)
    assert result is not None, "extract_spacy_entities returned None"
    assert isinstance(result, pd.DataFrame), "Must return a DataFrame"
    required_cols = {"text_id", "entity_text", "entity_label", "start_char", "end_char"}
    assert required_cols.issubset(set(result.columns)), (
        f"Missing columns: {required_cols - set(result.columns)}"
    )


def test_spacy_ner_finds_entities():
    """spaCy NER finds at least one entity in text with known entities."""
    texts = [(1, SAMPLE_NER_TEXT)]
    result = extract_spacy_entities(texts)
    assert result is not None, "extract_spacy_entities returned None"
    assert len(result) > 0, "No entities found in text containing IPCC, Jordan, etc."

    entity_texts = result["entity_text"].str.lower().tolist()
    assert any(term in " ".join(entity_texts) for term in
               ["ipcc", "jordan", "middle east", "march"]), (
        f"Expected known entities. Found: {result['entity_text'].tolist()}"
    )


def test_hf_ner_returns_dataframe():
    """extract_hf_entities returns a DataFrame with required columns."""
    texts = [(1, SAMPLE_NER_TEXT)]
    result = extract_hf_entities(texts)
    assert result is not None, "extract_hf_entities returned None"
    assert isinstance(result, pd.DataFrame), "Must return a DataFrame"
    required_cols = {"text_id", "entity_text", "entity_label", "start_char", "end_char"}
    assert required_cols.issubset(set(result.columns)), (
        f"Missing columns: {required_cols - set(result.columns)}"
    )


def test_entity_comparison():
    """compare_ner_outputs produces comparison dict with entity counts."""
    # Create minimal test DataFrames
    spacy_df = pd.DataFrame({
        "text_id": [1, 1, 1],
        "entity_text": ["IPCC", "Jordan", "March 2024"],
        "entity_label": ["ORG", "GPE", "DATE"],
        "start_char": [4, 50, 80],
        "end_char": [8, 56, 90],
    })
    hf_df = pd.DataFrame({
        "text_id": [1, 1],
        "entity_text": ["IPCC", "Jordan"],
        "entity_label": ["ORG", "LOC"],
        "start_char": [4, 50],
        "end_char": [8, 56],
    })
    result = compare_ner_outputs(spacy_df, hf_df)
    assert result is not None, "compare_ner_outputs returned None"
    assert isinstance(result, dict), "Must return a dictionary"
    assert "spacy_counts" in result, "Missing 'spacy_counts' key"
    assert "hf_counts" in result, "Missing 'hf_counts' key"
    assert "total_spacy" in result, "Missing 'total_spacy' key"
    assert "total_hf" in result, "Missing 'total_hf' key"
    assert result["total_spacy"] == 3, f"Expected 3 spaCy entities, got {result['total_spacy']}"
    assert result["total_hf"] == 2, f"Expected 2 HF entities, got {result['total_hf']}"


def test_evaluate_ner_metrics():
    """evaluate_ner computes precision, recall, F1 in 0-1 range."""
    predicted = pd.DataFrame({
        "text_id": [1, 1, 1],
        "entity_text": ["IPCC", "Jordan", "March 2024"],
        "entity_label": ["ORG", "GPE", "DATE"],
    })
    gold = pd.DataFrame({
        "text_id": [1, 1, 1],
        "entity_text": ["IPCC", "Jordan", "Middle East"],
        "entity_label": ["ORG", "GPE", "LOC"],
    })
    result = evaluate_ner(predicted, gold)
    assert result is not None, "evaluate_ner returned None"
    assert isinstance(result, dict), "Must return a dictionary"
    for key in ["precision", "recall", "f1"]:
        assert key in result, f"Missing '{key}' key"
        assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} not in [0, 1]"
    # 2 matches (IPCC+ORG, Jordan+GPE) out of 3 predicted, 3 gold
    assert result["precision"] > 0, "Precision should be > 0 with matching entities"
    assert result["recall"] > 0, "Recall should be > 0 with matching entities"
