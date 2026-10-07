"""Exploratory analysis of answer length and writing-style features by label."""

import pandas as pd

from data_cleaning import get_answer_data


def add_features(data):
    """Add word_count and the 4 writing-style features to the answer-level data."""
    data = data.copy()
    data["word_count"] = data["answer"].str.split().str.len()
    data["sentence_count"] = data["answer"].str.count(r"[.!?]+") + 1
    data["avg_sentence_len"] = data["word_count"] / data["sentence_count"]
    data["first_person"] = data["answer"].str.count(r"\b(I|me|my)\b")
    data["question_marks"] = data["answer"].str.count(r"\?")
    return data


if __name__ == "__main__":
    data = add_features(get_answer_data())

    print("Word count by label:")
    print(data.groupby("label")["word_count"].describe())

    print("\nStyle feature means by label:")
    print(data.groupby("label")[["word_count", "sentence_count", "avg_sentence_len",
                                  "first_person", "question_marks"]].mean())

    print("\nMedian word count per source and label:")
    print(data.groupby(["source", "label"])["word_count"].median())
