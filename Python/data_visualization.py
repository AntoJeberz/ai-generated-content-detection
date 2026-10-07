"""Generate the 4 EDA charts and save them to Visualizations/."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_cleaning import get_answer_data
from exploratory_analysis import add_features

LABEL_NAMES = {0: "Human", 1: "ChatGPT"}


def plot_distribution(data):
    plt.figure()
    plt.hist(data[data["label"] == 0]["word_count"].clip(upper=600), bins=60, alpha=0.5, label="Human", density=True)
    plt.hist(data[data["label"] == 1]["word_count"].clip(upper=600), bins=60, alpha=0.5, label="ChatGPT", density=True)
    plt.xlabel("Word count")
    plt.ylabel("Density")
    plt.title("Human vs ChatGPT Answer Length")
    plt.legend()
    plt.savefig("Visualizations/distribution_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_category(data):
    counts = data.groupby(["source", "label"]).size().unstack(fill_value=0)
    sources = counts.index.tolist()
    x = np.arange(len(sources))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, counts[0], width, label="Human")
    plt.bar(x + width / 2, counts[1], width, label="ChatGPT")
    plt.xticks(x, sources, rotation=30, ha="right")
    plt.xlabel("Source")
    plt.ylabel("Number of answers")
    plt.title("Answers per Source: Human vs ChatGPT")
    plt.legend()
    plt.savefig("Visualizations/category_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_trend(data):
    features = ["sentence_count", "avg_sentence_len", "first_person", "question_marks"]
    means = data.groupby("label")[features].mean()
    x = np.arange(len(features))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, means.loc[0], width, label="Human")
    plt.bar(x + width / 2, means.loc[1], width, label="ChatGPT")
    plt.xticks(x, features, rotation=20, ha="right")
    plt.xlabel("Writing-style feature")
    plt.ylabel("Average value per answer")
    plt.title("Writing-Style Features: Human vs ChatGPT")
    plt.legend()
    plt.savefig("Visualizations/trend_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation(data):
    cols = ["word_count", "sentence_count", "avg_sentence_len", "first_person", "question_marks", "label"]
    corr = data[cols].corr()

    plt.figure(figsize=(7, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Between Features and Label")
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.savefig("Visualizations/correlation_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    data = add_features(get_answer_data())

    plot_distribution(data)
    plot_category(data)
    plot_trend(data)
    plot_correlation(data)

    print("Saved 4 charts to Visualizations/")
