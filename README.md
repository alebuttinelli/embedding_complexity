# Italian Text Complexity Analyzer

This is a command-line Python script for performing a deep linguistic complexity analysis on a corpus of Italian texts. It uses `spaCy` for advanced NLP processing and `scikit-learn` for statistical analysis.

The script reads a CSV or Parquet file, analyzes the text in a specified column, and outputs a new file (CSV or Parquet) with a wide range of calculated complexity metrics appended as new columns.



---

## Features: Calculated Metrics

This tool calculates the following metrics for each text:

### Lexical Metrics
* **`word_length`**: The average length of words (in characters).
* **`type_token_ratio`**: The ratio of unique lemmas to the total number of tokens, measuring lexical diversity.
* **`word_freq`**: The percentage of words that belong to the "Basic Italian Vocabulary" (a combination of *fondamentali*, *alto_uso*, and *alta_disponibilità* lists).

### Syntactic & Structural Metrics
* **`avg_sentence_length`**: The average number of tokens per sentence.
* **`calculate_avg_depth`**: The average depth of the syntactic dependency tree for each sentence.
* **`calculate_avg_width`**: The average maximum width (nodes at a single level) of the syntactic dependency tree.
* **`clause_density`**: The average number of clauses (finite verbs) per sentence.

### Readability & Information Metrics
* **`gulpease_index`**: The standard Gulpease readability score for the Italian language.
* **`conceptual_density`**: The density of unique concepts (nouns + named entities) per sentence.
* **`calculate_coherence`**: A measure of local coherence, calculated as the average cosine similarity between adjacent sentences (using TF-IDF).
* **`lda_thematic_progression`**: A measure of topic flow, calculated as the average cosine similarity between the LDA topic distributions of adjacent sentences.
* **`shannon_entropy`**: The Shannon entropy of the text based on word frequencies, measuring information unpredictability.

---

## Setup and Installation

### 1. Install Python Dependencies
First, clone this repository and install the required libraries from the `requirements.txt` file.

```bash
# (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
# Install all required libraries
```bash
pip install -r requirements.txt
```

### 2. Download the spaCy Model
This script requires the it_core_news_lg model from spaCy. After installing the requirements, you must run this command from your terminal:
```bash
python -m spacy download it_core_news_lg
```
If you forget this step, the script will produce an error.

### 3. How to Run
The script is run from the command line. You must provide:

An input file (--input_file)

A path for the output file (--output_file)

The path to your vocabulary folder (--data_dir)

```bash
python analyze_complexity.py \
    --input_file "./path/to/my_texts.csv" \
    --output_file "./analysis/results.csv" \
    --data_dir "./data" \
    --text_column "document_text"
```

### 4. Output
The script will produce a new file (e.g., results.csv) which is a copy of your input file but with new columns appended, one for each metric calculated (e.g., word_length, gulpease_index, clause_density, etc.).

### 5. Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
