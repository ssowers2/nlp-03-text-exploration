"""
nlp_corpus_explore_case.py - Module 3 Script

Purpose

  Perform exploratory analysis of a small, controlled text corpus.
  Demonstrate how structure emerges from token distributions,
  category comparisons, and co-occurrence patterns.

Analytical Questions

- What tokens dominate each category?
- How do categories differ in vocabulary?
- What words appear in similar contexts?
- What structure is visible before using any models?
- What possible improvements can be made?

Notes

- This module focuses on exploratory analysis (EDA), not modeling.
- Results here prepare for later work with pipelines and embeddings.

Run from root project folder with:

  uv run python -m nlp.nlp_corpus_explore_case
"""

# ============================================================
# Section 1. Setup and Imports
# ============================================================

from collections import defaultdict
import logging
from pathlib import Path

from datafun_toolkit.logger import get_logger, log_header, log_path
import matplotlib.pyplot as plt
import polars as pl

print("Imports complete.")


# ============================================================
# Configure Logging
# ============================================================

LOG: logging.Logger = get_logger("CI", level="DEBUG")

ROOT_PATH: Path = Path.cwd()
NOTEBOOKS_PATH: Path = ROOT_PATH / "notebooks"
SCRIPTS_PATH: Path = ROOT_PATH / "scripts"

log_header(LOG, "MODULE 3: CORPUS EXPLORATION")
LOG.info("START script.....")

log_path(LOG, "ROOT_PATH", ROOT_PATH)
log_path(LOG, "NOTEBOOKS_PATH", NOTEBOOKS_PATH)
log_path(LOG, "SCRIPTS_PATH", SCRIPTS_PATH)

LOG.info("Logger configured.")

# ============================================================
# Section 2. Define Corpus (Labeled Text Documents)
# ============================================================

# A corpus is a collection of documents.
# Each document includes a category and text.

corpus: list[dict[str, str]] = [
    # Cupcakes
    {
        "category": "cupcake",
        "text": "A vanilla cupcake has creamy buttercream frosting.",
    },
    {
        "category": "cupcake",
        "text": "The bakery sells chocolate cupcakes with rainbow sprinkles.",
    },
    {
        "category": "cupcake",
        "text": "A red velvet cupcake is topped with cream cheese icing.",
    },
    # Pastry
    {"category": "pastry", "text": "A flaky croissant is baked fresh every morning."},
    {
        "category": "pastry",
        "text": "The pastry chef prepares fruit danishes and turnovers.",
    },
    {
        "category": "pastry",
        "text": "Buttery puff pastry surrounds a warm apple filling.",
    },
    # Cookies
    {"category": "cookie", "text": "The bakery serves warm chocolate chip cookies."},
    {"category": "cookie", "text": "A sugar cookie is decorated with pastel icing."},
    {"category": "cookie", "text": "Fresh oatmeal cookies cool on the baking rack."},
    # Cake
    {"category": "cake", "text": "A birthday cake is decorated with pink frosting."},
    {"category": "cake", "text": "The layer cake is filled with strawberry cream."},
    {"category": "cake", "text": "A lemon cake is glazed with sweet icing."},
    # Pie
    {"category": "pie", "text": "A strawberry pie has a flaky buttery crust."},
    {"category": "pie", "text": "A warm apple pie is baked with cinnamon and sugar."},
    {
        "category": "pie",
        "text": "The bakery serves fresh blueberry pie with a golden crust.",
    },
]

# Show results
print(f"Corpus contains {len(corpus)} documents.")


# ============================================================
# Section 3. Tokenize and Clean Text
# ============================================================

# Tokenization splits text into word-like units.


# Define a function to tokenize text by lowercasing, splitting on whitespace,
# and stripping common punctuation. We also filter out very short tokens (length <= 2).
# This simple tokenizer is sufficient for our small, controlled corpus.
# Use the string strip() method to remove punctuation from the beginning and end of each token.
def tokenize(text: str) -> list[str]:
    tokens = text.lower().split()
    return [t.strip(".,:;!?()[]\"'") for t in tokens if len(t) > 2]


# Define a new empty list to hold the token records we will create.
records_list: list[dict[str, str]] = []

# Loop through each document, tokenize the text,
# and create a record for each token with its category and
# add it to our list of records.
for doc in corpus:
    # Call our function to tokenize the text of the current document.
    tokens = tokenize(doc["text"])
    # Loop through each token produced by the tokenizer and
    # create a record that includes the category of the document and the token itself.
    # Append this record to our list of records.
    for token in tokens:
        records_list.append({"category": doc["category"], "token": token})

# Create a Polars DataFrame from the list of token records for easier analysis.
token_df: pl.DataFrame = pl.DataFrame(records_list)

# Show results
print("Tokenization complete.")
print(token_df.head(10))


# This step removes common stopwords (such as "the", "is", and "with") and very short tokens. These words do not add meaningful information.

STOP_WORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

# Filter tokens in DataFrame
token_df = token_df.filter(
    (pl.col("token").str.len_chars() > 2) & (~pl.col("token").is_in(STOP_WORDS))
)

print("Stopword filtering complete.")
# ============================================================
# Section 4. Compute Global Token Frequencies
# ============================================================

# Frequency distribution = how often each token appears.

# Create a DataFrame that groups the tokens by their text and
# counts how many times each token appears across the entire corpus.
global_freq_df: pl.DataFrame = (
    token_df.group_by("token").len().sort("len", descending=True)
)

# Show results
print("Top global tokens:")
print(global_freq_df.head(10))


# ============================================================
# Section 5. Compute Token Frequencies by Category
# ============================================================

# Compare token usage across categories.

# Create a new DataFrame that groups the tokens by both their category and text,
# counts how many times each token appears within each category,
# and sorts the results first by category and then by frequency in descending order.
# This shows which tokens are most common within each category.
category_freq_df: pl.DataFrame = (
    token_df.group_by(["category", "token"])
    .len()
    .sort(["category", "len"], descending=True)
)

# Show results
print("Top tokens by category:")
print(category_freq_df.head(12))


# ============================================================
# Section 6. Identify Top Tokens per Category
# ============================================================

# Show top tokens per category.


# Define a new empty dictionary to store the top tokens for each category.
top_per_category_dict: dict[str, list[str]] = {}

# Loop through each unique category in the token DataFrame,
# filter the category frequency DataFrame to get the top 5 tokens for that category,
# and store the list of top tokens in the dictionary.
# Also, print the top tokens for each category.
for category in token_df["category"].unique().to_list():
    subset_df = category_freq_df.filter(pl.col("category") == category).head(5)
    top_tokens_list = subset_df["token"].to_list()
    top_per_category_dict[category] = top_tokens_list

    # Show results for this category
    print(f"{category.upper()} top tokens: {top_tokens_list}")


# ============================================================
# Section 7. Analyze Co-occurrence (Context Windows)
# ============================================================

# Co-occurrence examines which tokens appear near each other.

# Define how many tokens on each side of a target token we include as context.
# A window size of 2 means:
#   - up to 2 tokens before the target token
#   - up to 2 tokens after the target token
# The target token itself is not included in its context list.
WINDOW_SIZE: int = 2

# Define a new empty dictionary to store the co-occurrence information.
# The keys will be target tokens,
# and the values will be lists of context tokens that appear near the target token.
co_occurrence_dict: dict[str, list[str]] = defaultdict(list)

# Loop through each document in the corpus, tokenize the text,
# and for each token, determine its context tokens based on the defined window size.
for doc in corpus:
    tokens = tokenize(doc["text"])

    # Use tokenized text to find nearby context words
    for i, token in enumerate(tokens):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)
        context = tokens[start:end]
        for ctx in context:
            if ctx != token:
                co_occurrence_dict[token].append(ctx)

# Show results
print("Co-occurrence analysis complete.")
for target in ["cupcake", "pastry", "cookie", "cake", "pie"]:
    print(f"\nTop context words for '{target}':")
    print(co_occurrence_dict[target][:10])

# ============================================================
# Section 8. Create Bigrams (Local Word Pairs) and Compute Frequencies
# ============================================================

# Bigrams combine each word with the next word in the text.
# This helps us capture local structure: how words are used together,
# not just which words appear individually.

# Bigrams capture pairs of consecutive tokens.

# Define a new empty list to hold the bigram tuples we will create.
bigrams_list: list[tuple[str, str]] = []

# Loop through each document in the corpus, tokenize the text,
# and create bigrams by pairing each token with the next token in the list.
for doc in corpus:
    tokens = tokenize(doc["text"])
    for i in range(len(tokens) - 1):
        bigrams_list.append((tokens[i], tokens[i + 1]))

# Create a DataFrame from the list of bigram tuples,
# where each bigram is represented as a single string with the two tokens separated by a space.
bigram_df: pl.DataFrame = pl.DataFrame(
    {"bigram": [f"{a} {b}" for a, b in bigrams_list]}
)

# Create a new DataFrame that groups the bigrams by their text
# and counts how many times each bigram appears,
# then sorts the results by frequency in descending order.
bigram_freq_df: pl.DataFrame = (
    bigram_df.group_by("bigram").len().sort("len", descending=True)
)

# Show results
print("Top bigrams:")
print(bigram_freq_df.head(10))


# ============================================================
# Section 9. Visualize Token Frequencies
# ============================================================

# Define a new DataFrame that filters the category frequency DataFrame
# to get the top 7 tokens for the "cupcake" category.
bakery_df = category_freq_df.filter(pl.col("category") == "cupcake").sort("len").tail(7)


# Create a figure that is 8 inches wide and 4 inches tall
plt.figure(figsize=(8, 4))

# Add a bar chart to the figure using the tokens as the x-axis and their frequencies as the y-axis.
# Horizontal bar chart with teal color instead
plt.barh(bakery_df["token"], bakery_df["len"], color="teal")

# Define the x-axis tick parameters to rotate the labels by 45 degrees for better readability.
# The gca() function gets the current axes of the plot, and tick_params() is used to set the rotation of the x-axis labels.
ax = plt.gca()
ax.tick_params(axis="x", labelrotation=45)

# Set the title and labels for the axes of the plot.
plt.title("Top 7 Tokens in Cupcake Category", fontsize=14, fontweight="bold")
plt.xlabel("Frequency")
plt.ylabel("Token")

# Adjust the layout of the plot to prevent overlap and ensure everything fits well.
plt.tight_layout()

plt.xticks([0, 1, 2])

# Display the plot on the screen.
# The execution of the script will pause until the plot window is closed.
plt.show()

# ============================================================
# Section 10. Interpret Results and Identify Patterns
# ============================================================

print("\nSabriya Sowers' Observations on Bakery Text Patterns:")

print("- Tokens cluster by category (cupcake, pastry, cookie, cake, pie).")
print(
    "- Words that appear in similar contexts behave similarly (e.g., dessert items described by flavor and texture)."
)
print(
    "- Co-occurrence reveals contextual relationships between ingredients and baked items."
)
print(
    "- Bigrams capture local structure such as common phrases used in baking descriptions."
)
print(
    "- Patterns start to show before any machine learning is applied through frequency and context analysis."
)

print("\nSabriya Sowers' Specific Observations:")

print(
    "- Cupcake tokens emphasize frosting and decoration (e.g., cream, frosting, buttercream, velvet)."
)
print(
    "- Cake tokens focus on preparation and presentation (e.g., decorated, filled, cream, pink)."
)
print(
    "- Pie tokens highlight ingredients and texture (e.g., crust, apple, strawberry, flaky, baked)."
)
print(
    "- Cookie tokens emphasize ingredients and baking context (e.g., chocolate, chip, pastel, baking)."
)
print(
    "- Pastry tokens focus on texture and type (e.g., croissant, apple, buttery, morning)."
)
print(
    "- Co-occurrence shows that dessert words often appear near flavor or texture descriptors (e.g., cupcake near vanilla and creamy)."
)
print(
    "- Bigrams show common word pairs like 'the bakery', 'decorated with', and 'warm apple'."
)

# ============================================================
# END
# ============================================================

LOG.info("========================")
LOG.info("Pipeline executed successfully!")
LOG.info("========================")
LOG.info("END main()")
