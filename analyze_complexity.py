# -*- coding: utf-8 -*-
"""
Script for textual complexity analysis.
"""

import spacy
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import DBSCAN

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def word_length(doc):
    """Calculates the average word length."""
    if not doc or len(doc) == 0:
        return 0.0
    total_length = sum(len(token.text) for token in doc)
    return round(total_length / len(doc), 2)

def type_token_ratio(doc):
    """Ratio between the total number of tokens and the total number of unique lemmas."""
    if not doc or len(doc) == 0:
        return 0.0
    tokens_count = len(doc)
    types_count = len(set(token.lemma_ for token in doc))
    if types_count == 0:
        return 0.0
    return round(tokens_count / types_count, 2)

def word_freq(doc, data_dir):
    """
    Calculates the frequency of the basic vocabulary.
    Requires a directory containing the vocabulary .txt files.
    """
    if not doc or len(doc) == 0:
        return 0.0
        
    # Load word lists from files
    try:
        with open(data_dir / "parole_fondamentali.txt", 'r', encoding='utf-8') as f:
            p_fondamentali = set(f.read().split())
        with open(data_dir / "parole_alto_uso.txt", 'r', encoding='utf-8') as g:
            p_alto_uso = set(g.read().split())
        with open(data_dir / "parole_alta_disponibilita.txt", 'r', encoding='utf-8') as e:
            p_alta_disponibilita = set(e.read().split())
    except FileNotFoundError as e:
        logging.warning(f"Vocabulary file not found: {e}. 'word_freq' function will return 0.")
        return 0.0

    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    if not lemmas:
        return 0.0

    vocab_base_count = 0
    for token in lemmas:
        if token in p_fondamentali or token in p_alto_uso or token in p_alta_disponibilita:
            vocab_base_count += 1

    return round(vocab_base_count / len(lemmas), 2)

def avg_sentence_length(doc):
    """Calculates the average sentence length."""
    if not doc or len(doc) == 0:
        return 0.0
    sentences = list(doc.sents)
    sentences_count = len(sentences)
    if sentences_count == 0:
        return 0.0
    return round(len(doc) / sentences_count, 2)

def compute_depth(node):
    """Recursively calculates the depth of the syntactic tree."""
    if not list(node.children):
        return 0
    else:
        return 1 + max(compute_depth(child) for child in node.children)

def count_nodes_per_level(node, level=0, levels=None):
    """Counts nodes per level in the syntactic tree."""
    if levels is None:
        levels = defaultdict(int)
    levels[level] += 1
    for child in node.children:
        count_nodes_per_level(child, level + 1, levels)
    return levels

def calculate_avg_depth(doc):
    """Calculates the average depth of sentences in the document."""
    all_depths = []
    for sent in doc.sents:
        if sent.root:
            tree_depth = compute_depth(sent.root)
            all_depths.append(tree_depth)

    return round(np.mean(all_depths), 2) if all_depths else 0.0

def calculate_avg_width(doc):
    """Calculates the average width (max nodes at one level) of sentences."""
    all_widths = []
    for sent in doc.sents:
        if sent.root:
            levels = count_nodes_per_level(sent.root)
            if levels:
                tree_width = max(levels.values())
                all_widths.append(tree_width)

    return round(np.mean(all_widths), 2) if all_widths else 0.0

def count_clauses(doc_sent):
    """Counts clauses (finite verbs) in a sentence."""
    return sum(1 for token in doc_sent if token.pos_ == "VERB" and token.morph.get("VerbForm") == ['Fin'])

def clause_density(doc):
    """Calculates the average density of clauses per sentence."""
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    if num_sentences == 0:
        return 0.0

    total_clauses = sum(count_clauses(sentence) for sentence in sentences)
    return round(total_clauses / num_sentences, 2)

def gulpease_index(doc):
    """Calculates the Gulpease index, a readability metric for Italian."""
    if not doc or len(doc) == 0:
        return 0.0
        
    letters_count = sum(len(token.text) for token in doc if not token.is_punct and not token.is_space)
    words = [token for token in doc if not token.is_punct and not token.is_space]
    words_count = len(words)
    sentences_count = len(list(doc.sents))

    if words_count == 0 or sentences_count == 0:
        return 0.0

    LP = (letters_count / words_count) * 100
    FR = (sentences_count / words_count) * 100
    return round(89 - (LP / 10) + (FR * 3), 2)

def extract_concepts(doc):
    """Extracts concepts (nouns + named entities)."""
    concepts = set(token.text.lower() for token in doc if token.pos_ == "NOUN")
    concepts.update(ent.text.lower() for ent in doc.ents)
    return list(concepts)

def deduplicate_concepts(concepts):
    """Deduplicates concepts using TF-IDF and DBSCAN."""
    if not concepts:
        return []

    vectorizer = TfidfVectorizer()
    try:
        embeddings = vectorizer.fit_transform(concepts).toarray()
    except ValueError:
        return concepts

    if embeddings.shape[0] == 0:
        return []

    # Clustering to reduce duplicates
    try:
        clustering = DBSCAN(eps=0.3, min_samples=1, metric="cosine").fit(embeddings)
        unique_indices = np.unique(clustering.labels_, return_index=True)[1]
        deduplicated = [concepts[i] for i in sorted(unique_indices)]
        return deduplicated
    except Exception as e:
        logging.warning(f"DBSCAN clustering failed: {e}. Returning non-deduplicated concepts.")
        return concepts

def conceptual_density(doc):
    """Calculates conceptual density (unique concepts / sentences)."""
    concepts = extract_concepts(doc)
    deduplicated_concepts = deduplicate_concepts(concepts)
    num_sentences = len(list(doc.sents))
    
    return round(len(deduplicated_concepts) / num_sentences, 2) if num_sentences > 0 else 0.0

def calculate_coherence(doc):
    """Calculates coherence (average cosine similarity between adjacent sentences)."""
    sentences = [sent.text.strip() for sent in doc.sents]
    if len(sentences) < 2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer()
        sentence_embeddings = vectorizer.fit_transform(sentences).toarray()
        
        if sentence_embeddings.shape[0] < 2:
            return 0.0

        similarities = [
            cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
            for i in range(len(sentence_embeddings) - 1)
        ]
        return round(np.mean(similarities), 2) if similarities else 0.0
    except ValueError:
        return 0.0

def lda_thematic_progression(doc, num_topics=2):
    """Calculates thematic progression (LDA similarity between adjacent sentences)."""
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if len(sentences) < 2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer()
        dt_matrix = vectorizer.fit_transform(sentences)
        
        # If the matrix is empty or has fewer features than topics
        if dt_matrix.shape[1] < num_topics:
            num_topics = dt_matrix.shape[1]
        if num_topics == 0:
            return 0.0

        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        topic_distributions = lda.fit_transform(dt_matrix)

        similarities = [
            cosine_similarity([topic_distributions[i]], [topic_distributions[i + 1]])[0][0]
            for i in range(len(topic_distributions) - 1)
        ]
        return round(np.mean(similarities), 2) if similarities else 0.0
    except ValueError:
        return 0.0

def shannon_entropy(doc):
    """Calculates Shannon entropy based on word frequency."""
    words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    if not words:
        return 0.0
        
    word_count = Counter(words)
    total_words = len(words)
    
    entropy = -sum((freq / total_words) * np.log2(freq / total_words) for freq in word_count.values())
    return round(entropy, 2)


def analyze_dataframe(df, nlp, data_dir, text_column):
    """
    Applies complexity analysis to a DataFrame.
    """
    # Define the list of functions
    word_freq_partial = partial(word_freq, data_dir=data_dir)
    word_freq_partial.__name__ = "word_freq"

    lista_funct = [
        word_length,
        type_token_ratio,
        avg_sentence_length,
        calculate_avg_width,
        calculate_avg_depth,
        clause_density,
        gulpease_index,
        conceptual_density,
        calculate_coherence,
        lda_thematic_progression,
        shannon_entropy,
        word_freq_partial
    ]
    
    # Disable NER if 'conceptual_density' is NOT in the list of functions to run
    disabled_pipes = ["ner"] if "conceptual_density" not in [f.__name__ for f in lista_funct] else []
    
    try:
        df['testo_tok'] = list(tqdm(
            nlp.pipe(df[text_column], disable=disabled_pipes), 
            total=len(df), 
            desc="spaCy Processing"
        ))
    except KeyError:
        logging.error(f"Column '{text_column}' not found in DataFrame. Check your --text_column argument.")
        raise

    logging.info("Calculating complexity metrics...")
    for funct in tqdm(lista_funct, desc="Calculating Metrics"):
        df[funct.__name__] = df['testo_tok'].apply(funct)

    # Clean up the large token object column to save memory
    df = df.drop(columns=['testo_tok'])

    return df

def main():
    """
    Main entry point for the command-line script.
    """
    parser = argparse.ArgumentParser(description="Textual Complexity Analyzer")
    parser.add_argument(
        "-i", "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input file (.csv or .parquet)."
    )
    parser.add_argument(
        "-o", "--output_file", 
        type=str, 
        required=True, 
        help="Path to the output file (.csv or .parquet)."
    )
    parser.add_argument(
        "-d", "--data_dir", 
        type=str, 
        required=True, 
        help="Directory containing the vocabulary .txt files (e.g., 'parole_fondamentali.txt')."
    )
    parser.add_argument(
        "-t", "--text_column", 
        type=str, 
        default="testo", 
        help="Name of the column containing the texts (default: 'testo')."
    )
    
    args = parser.parse_args()

    # Transform path strings into Path objects
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    data_dir_path = Path(args.data_dir)

    # Load the spaCy model
    logging.info("Loading spaCy model 'it_core_news_lg'...")
    try:
        nlp = spacy.load("it_core_news_lg")
    except OSError:
        logging.error("Model 'it_core_news_lg' not found.")
        logging.error("Run 'python -m spacy download it_core_news_lg' from your terminal.")
        return

    # Load data
    logging.info(f"Loading data from {input_path}")
    if str(input_path).endswith(".csv"):
        df = pd.read_csv(input_path)
    elif str(input_path).endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        logging.error("Unsupported file format. Use .csv or .parquet")
        return

    if args.text_column not in df.columns:
        logging.error(f"Column '{args.text_column}' not found in the file.")
        return

    # Run the analysis
    df_results = analyze_dataframe(df, nlp, data_dir_path, args.text_column)

    # Save the results
    logging.info(f"Saving results to {output_path}")
    if str(output_path).endswith(".csv"):
        df_results.to_csv(output_path, index=False)
    elif str(output_path).endswith(".parquet"):
        df_results.to_parquet(output_path, index=False)
    
    logging.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()
