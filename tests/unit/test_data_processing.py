"""
test_data_processing.py
"""

import pandas as pd
import torch
from transformers import AutoTokenizer
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
from src.data_processing import (
    clean_text,
    add_sentiment_label,
    clean_dataset,
    split_data,
    tokenize_data,
    ReviewDataset
)

# ---------- CLEAN TEXT ----------

def test_clean_text_lowercase():
    assert clean_text("HELLO WORLD!") == "hello world"

def test_clean_text_removes_urls_mentions_symbols():
    text = "Visit http://test.com @user #wow!!!"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "!" not in cleaned


# ---------- ADD SENTIMENT LABEL ----------

def test_add_sentiment_label_values():
    df = pd.DataFrame({"score": [1, 3, 5]})
    df = add_sentiment_label(df)
    assert list(df["sentiment"]) == ["negative", "neutral", "positive"]


# ---------- CLEAN DATASET ----------

def test_clean_dataset_structure():
    df = pd.DataFrame({
        "content": ["Great app!!!", "Terrible update :(", None],
        "score": [5, 1, 3]
    })
    cleaned = clean_dataset(df)
    assert "content" in cleaned.columns
    assert "sentiment" in cleaned.columns
    assert not cleaned["content"].isnull().any()
    assert cleaned["content"].iloc[0] == "great app"
    assert len(cleaned) <= len(df)


# ---------- SPLIT DATA ----------

def test_split_data_stratified():
    df = pd.DataFrame({
        "content": ["a", "b", "c", "d", "e", "f"],
        "sentiment": ["positive", "positive", "neutral", "neutral", "negative", "negative"]
    })
    train_df, val_df = split_data(df, test_size=0.33, random_state=0)
    assert "content" in train_df.columns
    assert "sentiment" in val_df.columns
    assert abs(len(train_df) - len(val_df)) > 0  # different sizes


# ---------- TOKENIZATION ----------

def test_tokenize_data_returns_tensors():
    df = pd.DataFrame({"content": ["this app is good", "bad performance"]})
    tokens = tokenize_data(df, tokenizer_name="bert-base-cased")
    assert isinstance(tokens["input_ids"], torch.Tensor)
    assert tokens["input_ids"].shape[0] == 2


# ---------- REVIEW DATASET ----------

def test_review_dataset_output_shape():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    texts = ["good app", "bad app"]
    labels = ["positive", "negative"]
    ds = ReviewDataset(texts, labels, tokenizer, max_len=16)
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape[0] == 16
    assert isinstance(sample["labels"], torch.Tensor)
