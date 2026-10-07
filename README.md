# AI-Generated Content Detection Pattern

## Objective

Detect patterns that distinguish human-written text from ChatGPT-generated text. The project cleans and restructures a raw question/answer corpus into a balanced, answer-level dataset and explores word-count and writing-style differences between Human and ChatGPT answers.

## Dataset

[HC3 — Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) (the `all` subset), via Hugging Face. Each row pairs a question with one or more human answers and one or more ChatGPT answers, drawn from sources such as Reddit ELI5, finance, medicine, open QA, and Wikipedia CS/AI.

## Folder Structure

```
Data-Analysis-Python-Project/
│
├── README.md
├── Dataset/
│   ├── raw_dataset.csv          ← original HC3 "all" data
│   └── cleaned_dataset.csv      ← output of the cleaning pipeline
├── Notebook/
│   └── Data_Analysis_EDA.ipynb  ← full, annotated analysis notebook
├── Python/
│   ├── data_loading.py
│   ├── data_cleaning.py
│   ├── exploratory_analysis.py
│   └── data_visualization.py
├── Visualizations/
│   ├── distribution_analysis.png
│   ├── trend_analysis.png
│   ├── category_analysis.png
│   └── correlation_analysis.png
└── Documentation/
    └── Project_Report.pdf
```

## Tools Used

- Python 3
- pandas
- matplotlib
- Jupyter Notebook

## How to Run

Install dependencies:

```
pip install pandas matplotlib
```

**Option A — run the scripts** (from the project root, in order):

```
python Python/data_loading.py
python Python/data_cleaning.py
python Python/exploratory_analysis.py
python Python/data_visualization.py
```

`data_cleaning.py` writes `Dataset/cleaned_dataset.csv`; `data_visualization.py` writes the 4 PNG charts to `Visualizations/`.

**Option B — open the notebook:**

```
jupyter notebook Notebook/Data_Analysis_EDA.ipynb
```

## Cleaning Steps Summary

1. Load `raw_dataset.csv`, drop the unused `index` column and 509 duplicate rows → **23,813** question rows.
2. Parse the `human_answers` / `chatgpt_answers` JSON lists and explode them into one row per answer, labeling each row (0 = Human, 1 = ChatGPT). Remove null/blank answers → **83,400** answers (57,019 Human, 26,381 ChatGPT).
3. Compute word count and writing-style features (sentence count, average sentence length, first-person usage, question marks).
4. Fix spacing artifacts and remove the giveaway phrase "As an AI language model".
5. Drop answers under 20 words and duplicate answer text → 49,018 Human / 26,173 ChatGPT.
6. Balance the two classes (random_state=42) → **52,346** answers, **26,173 per label**, saved as `cleaned_dataset.csv`.

## Visualizations

| Chart | Description |
|---|---|
| `distribution_analysis.png` | Histogram comparing Human vs ChatGPT answer length (word count). |
| `category_analysis.png` | Grouped bar chart of answer counts per source, split by Human vs ChatGPT. |
| `trend_analysis.png` | Grouped bar chart comparing average writing-style features (sentence count, average sentence length, first-person usage, question marks) between Human and ChatGPT. |
| `correlation_analysis.png` | Correlation heatmap between word count, writing-style features, and the label. |

## Key Findings

- ChatGPT answers are longer (median 174 words vs 83 for human) and much more consistent in length.
- Humans use first-person words (I, me, my) about 4.5 times more often (0.95 vs 0.21 per answer).
- Humans ask more questions inside answers (0.22 vs 0.07 question marks per answer).
- ChatGPT writes longer sentences (19.1 vs 16.1 words per sentence).
- Final dataset: 52,346 answers, balanced 50/50, saved as `cleaned_dataset.csv`.
