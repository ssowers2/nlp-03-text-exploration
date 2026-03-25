# nlp-03-text-exploration

[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue?logo=python)](#)
[![MIT](https://img.shields.io/badge/license-see%20LICENSE-yellow.svg)](./LICENSE)

> Professional Python project for Web Mining and Applied NLP.

Web Mining and Applied NLP focus on retrieving, processing, and analyzing text from the web and other digital sources.
This course builds those capabilities through working projects.

In the age of generative AI, durable skills are grounded in real work:
setting up a professional environment,
reading and running code,
understanding the logic,
and pushing work to a shared repository.
Each project follows a similar structure based on professional Python projects.
These projects are **hands-on textbooks** for learning Web Mining and Applied NLP.

## This Project

This project focuses on **exploratory analysis of text data**.

The goal is to analyze a small, structured corpus and observe how
patterns emerge from token distributions, category comparisons,
and contextual relationships.

You will:

- tokenize and clean text data
- build frequency distributions
- compare token usage across categories
- examine co-occurrence (context windows)
- analyze bigrams (local structure)
- visualize results and interpret patterns

This project illustrates how **structure appears in text before any machine learning is applied**.
These patterns support later pipelines, embeddings, and retrieval.

You'll work with just these files as you update authorship and experiment:

- **notebooks/nlp_corpus_explore_case.ipynb** - notebook version
- **src/nlp/nlp_corpus_explore_case.py** - Python script
- **pyproject.toml** - project configuration and dependencies
- **zensical.toml** - project metadata

## First: Follow These Instructions

Follow the [step-by-step workflow guide](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/) to complete:

1. Phase 1. **Start & Run**
2. Phase 2. **Change Authorship**
3. Phase 3. **Read & Understand**

## What to Look For

As you run the script and notebook, focus on:

- which tokens dominate each category
- how categories differ in vocabulary
- which words appear in similar contexts
- how local structure (bigrams) appears in text

These observations are the foundation for later modules.

## Success

After running the script successfully, you will see:

```shell
========================
Pipeline executed successfully!
========================
```

You will also see:

- frequency tables printed to the console
- visualizations of token distributions
- examples of co-occurrence and bigram patterns

A file named `project.log` will appear in the project folder.

## Command Reference

The commands below are used in the workflow guide above.
They are provided here for convenience.

Follow the guide for the **full instructions**.

<details>
<summary>Show command reference</summary>

### In a machine terminal (open in your `Repos` folder)

After you get a copy of this repo in your own GitHub account,
open a machine terminal in your `Repos` folder:

```shell
# Replace username with YOUR GitHub username.
git clone https://github.com/ssowers2/nlp-03-text-exploration
cd nlp-03-text-exploration
code .
```

### In a VS Code terminal

```shell
uv self update
uv python pin 3.14
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
git add -A
uvx pre-commit run --all-files

# Later, we install spacy data model and
# en_core_web_sm = english, core, web, small
# It's big: spacy+data ~200+ MB w/ model installed
#           ~350–450 MB for .venv is normal for NLP
# uv run python -m spacy download en_core_web_sm

# First, run the module
# IMPORTANT: Close each figure after viewing so execution continues
uv run python -m nlp.nlp_corpus_explore_case

# Then, open the notebook.
# IMPORTANT: Select the kernel and Run All:
# notebooks/nlp_corpus_explore_case.ipynb

uv run ruff format .
uv run ruff check . --fix
uv run zensical build

git add -A
git commit -m "update"
git push -u origin main
```

</details>

## Notes

- Use the **UP ARROW** and **DOWN ARROW** in the terminal to scroll through past commands.
- Use `CTRL+f` to find (and replace) text within a file.

## Terminology

In preparation for large language models (LLM) and related methods,
our analysis does not begin with semantic interpretation.
Instead, we focus on **proximity** and observable **patterns** in the text.

We evaluate **co-occurrence (context windows)**, that is, _which words tend to appear near each other_.

The full collection of text is called a **corpus** (a set of documents).
For this analysis, each document is represented as a single line of text.

# Corpus Exploration – Bakery Text Analysis

## Technical Modification

For this assignment, I modified the original corpus exploration notebook to use a custom bakery-themed corpus instead of the example animal and vehicle corpus. I replaced the original categories with cupcake, pastry, cookie, cake, and pie. This changed the input data and made the analysis more relevant to a bakery and pastry theme.

I also added a stopword filtering step after tokenization. This removed common filler words such as “the,” “is,” and “with” from the token DataFrame before calculating token frequencies. This improved the quality of the most frequent token results by highlighting more meaningful category-specific words.

In addition, I updated the visualization section to display the top seven tokens in the cupcake category using a horizontal bar chart with a teal color scheme. This made the output more readable and visually distinct from the original example.

---

## Why I Made the Change

I made these changes to create a more interesting and customized corpus while also improving the quality of the text analysis. The bakery categories provided a clear theme for comparing vocabulary across groups. Adding stopword filtering made the frequency results more useful because common filler words no longer dominated the token counts. Updating the chart also made the output easier to interpret.

---

## What I Observed After Running the Project

After running the modified notebook, the bakery corpus produced category-specific token patterns that were easy to compare.

The global token frequency results showed that words such as `icing`, `fresh`, `pie`, `cake`, and `bakery` appeared often across the corpus.

The top tokens by category showed clear vocabulary differences:

- **Cake** was associated with words like `cake`, `decorated`, `filled`, `pink`, and `cream`.
- **Pie** was associated with `pie`, `crust`, `baked`, `sugar`, and `golden`.
- **Cookie** was associated with `cookies`, `baking`, `chocolate`, `pastel`, and `chip`.
- **Cupcake** was associated with `cupcake`, `cream`, `frosting`, `creamy`, and `buttercream`.
- **Pastry** was associated with `pastry`, `apple`, `croissant`, `morning`, and `baked`.

The co-occurrence results showed which words appeared near the target bakery words. For example, `cupcake` appeared near words like `vanilla`, `creamy`, `red`, `velvet`, and `topped`, while `pie` appeared near `strawberry`, `flaky`, `apple`, and `blueberry`. This helps show how words appear in similar contexts. The bigram results also showed common word pairs such as `the bakery`, `decorated with`, `warm apple`, and `bakery serves`. The cupcake visualization showed that `cupcake` was the most frequent token in that category, followed by tokens such as `sells`, `topped`, `icing`, `velvet`, `vanilla`, and `cupcakes`.

---

## Analytical Questions

### What tokens dominate each category?

Each category is dominated by words tied to its baked item. Cupcake is dominated by `cupcake`, `cream`, `frosting`, and `buttercream`.

- Cake is dominated by `cake`, `decorated`, `filled`, and `cream`.
- Pie is dominated by `pie`, `crust`, `baked`, and `sugar`.
- Cookie is dominated by `cookies`, `chocolate`, `chip`, and `pastel`.
- Pastry is dominated by `pastry`, `croissant`, `apple`, and `baked`.

### How do categories differ in vocabulary?

The categories differ by the descriptive words associated with each baked good.

- Cupcake vocabulary emphasizes frosting and decoration.
- Cake vocabulary emphasizes filling and decoration.
- Pie vocabulary emphasizes crust and fruit flavors.
- Cookie vocabulary emphasizes ingredients and baking.
- Pastry vocabulary emphasizes flaky textures and breakfast-style items like croissants and danishes.

### What words appear in similar contexts?

Words appear in similar contexts when they are surrounded by descriptive bakery terms. For example, `cupcake` appears near `vanilla`, `creamy`, `red`, and `velvet`, while `pie` appears near `strawberry`, `flaky`, `apple`, and `blueberry`. These context words show that desserts are often described by flavor and texture.

### What structure is visible before using any models?

Before using any models, the corpus already shows patterns through simple exploration.

- Token frequency reveals dominant terms
- Category frequency reveals vocabulary differences
- Co-occurrence shows nearby context relationships
- Bigrams show common word pairings such as `the bakery` and `decorated with`.

## What possible improvements can be made?

One future improvement would be to apply the same stopword filtering to the co-occurrence and bigram sections so that filler words such as `the` and `has` do not appear in the results.
