"""Restructure the raw HC3 data into one row per answer and build the final balanced dataset."""

import json

import pandas as pd

from data_loading import load_raw_data


def get_answer_data():
    """Parse the JSON answer lists, explode to one row per answer and label them.

    Returns "data": columns question, source, answer, label (0 = Human, 1 = ChatGPT),
    with null and blank answers removed.
    """
    df = load_raw_data()

    df["human_answers"] = df["human_answers"].apply(json.loads)
    df["chatgpt_answers"] = df["chatgpt_answers"].apply(json.loads)

    human = df[["question", "source", "human_answers"]].explode("human_answers")
    human = human.rename(columns={"human_answers": "answer"})
    human["label"] = 0

    ai = df[["question", "source", "chatgpt_answers"]].explode("chatgpt_answers")
    ai = ai.rename(columns={"chatgpt_answers": "answer"})
    ai["label"] = 1

    data = pd.concat([human, ai], ignore_index=True)

    data = data.dropna(subset=["answer"])
    data["answer"] = data["answer"].str.strip()
    data = data[data["answer"] != ""]

    return data


def build_final_dataset(data):
    """Clean answer text, filter short/duplicate answers and balance the two classes.

    Returns "final" with 50/50 Human vs ChatGPT answers (random_state=42).
    """
    data = data.copy()
    data["word_count"] = data["answer"].str.split().str.len()

    data["answer"] = data["answer"].str.replace(r"\s+([?.!,])", r"\1", regex=True)
    data["answer"] = data["answer"].str.replace(r"\s+n't", "n't", regex=True)
    data["answer"] = data["answer"].str.replace("As an AI language model, ", "", case=False, regex=False)

    data = data[data["word_count"] >= 20]
    data = data.drop_duplicates(subset=["answer"])

    human = data[data["label"] == 0]
    ai = data[data["label"] == 1]

    n = min(len(human), len(ai))
    final = pd.concat([human.sample(n, random_state=42), ai.sample(n, random_state=42)])
    final = final.sample(frac=1, random_state=42)

    return final[["answer", "label", "source", "question"]]


if __name__ == "__main__":
    data = get_answer_data()
    final = build_final_dataset(data)

    final.to_csv("Dataset/cleaned_dataset.csv", index=False)

    check = pd.read_csv("Dataset/cleaned_dataset.csv")
    print("Final shape:", check.shape)
    print("\nLabel counts:")
    print(check["label"].value_counts())
