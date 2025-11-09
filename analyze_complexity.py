# -*- coding: utf-8 -*-
"""
Script per l'analisi della complessità testuale.

Questo script calcola una serie di metriche di complessità linguistica
per un file di testo (CSV o Parquet) e salva i risultati in un nuovo file.

Esempio di esecuzione:
    
    # Per prima cosa, assicurati di aver scaricato il modello spaCy
    # python -m spacy download it_core_news_lg

    python analyze_complexity.py \
        --input_file "percorso/del/tuo/file_input.csv" \
        --output_file "percorso/per/salvare/risultati.csv" \
        --data_dir "percorso/cartella/con/parole_fondamentali.txt"
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

# Import di Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.cluster import DBSCAN  # <-- IMPORT MANCANTE NEL TUO SCRIPT

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Funzioni di Metrica (Il tuo codice, pulito) ---

def word_length(doc: spacy.tokens.Doc) -> float:
    """Calcola la lunghezza media delle parole."""
    if not doc or len(doc) == 0:
        return 0.0
    total_length = sum(len(token.text) for token in doc)
    return round(total_length / len(doc), 2)

def type_token_ratio(doc: spacy.tokens.Doc) -> float:
    """Rapporto tra numero totale di token e numero totale di lemmi unici."""
    if not doc or len(doc) == 0:
        return 0.0
    tokens_count = len(doc)
    types_count = len(set(token.lemma_ for token in doc))
    if types_count == 0:
        return 0.0
    return round(tokens_count / types_count, 2)

def word_freq(doc: spacy.tokens.Doc, data_dir: Path) -> float:
    """
    Calcola la frequenza del vocabolario di base (fondamentale + alto uso + alta disponibilità).
    Richiede una directory contenente i file .txt del vocabolario.
    """
    if not doc or len(doc) == 0:
        return 0.0
        
    # Carica le liste di parole dai file (in modo sicuro e portatile)
    try:
        with open(data_dir / "parole_fondamentali.txt", 'r', encoding='utf-8') as f:
            p_fondamentali = set(f.read().split()) # Usa 'set' per ricerche veloci
        with open(data_dir / "parole_alto_uso.txt", 'r', encoding='utf-8') as g:
            p_alto_uso = set(g.read().split())
        with open(data_dir / "parole_alta_disponibilita.txt", 'r', encoding='utf-8') as e:
            p_alta_disponibilita = set(e.read().split())
    except FileNotFoundError as e:
        logging.warning(f"File di vocabolario non trovato: {e}. La funzione 'word_freq' restituirà 0.")
        return 0.0

    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    if not lemmas:
        return 0.0

    vocab_base_count = 0
    for token in lemmas:
        if token in p_fondamentali or token in p_alto_uso or token in p_alta_disponibilita:
            vocab_base_count += 1

    return round(vocab_base_count / len(lemmas), 2)

def avg_sentence_length(doc: spacy.tokens.Doc) -> float:
    """Calcola la lunghezza media delle frasi."""
    if not doc or len(doc) == 0:
        return 0.0
    sentences = list(doc.sents)
    sentences_count = len(sentences)
    if sentences_count == 0:
        return 0.0
    return round(len(doc) / sentences_count, 2)

def compute_depth(node) -> int:
    """Calcola ricorsivamente la profondità dell'albero sintattico."""
    if not list(node.children):
        return 0
    else:
        return 1 + max(compute_depth(child) for child in node.children)

def count_nodes_per_level(node, level: int = 0, levels: dict = None) -> dict:
    """Conta i nodi per livello nell'albero sintattico."""
    if levels is None:
        levels = defaultdict(int)
    levels[level] += 1
    for child in node.children:
        count_nodes_per_level(child, level + 1, levels)
    return levels

def calculate_avg_depth(doc: spacy.tokens.Doc) -> float:
    """Calcola la profondità media delle frasi nel documento."""
    all_depths = []
    for sent in doc.sents:
        if sent.root:
            tree_depth = compute_depth(sent.root)
            all_depths.append(tree_depth)

    return round(np.mean(all_depths), 2) if all_depths else 0.0

def calculate_avg_width(doc: spacy.tokens.Doc) -> float:
    """Calcola la larghezza media (max nodi a un livello) delle frasi."""
    all_widths = []
    for sent in doc.sents:
        if sent.root:
            levels = count_nodes_per_level(sent.root)
            if levels:
                tree_width = max(levels.values())
                all_widths.append(tree_width)

    return round(np.mean(all_widths), 2) if all_widths else 0.0

def count_clauses(doc_sent: spacy.tokens.Span) -> int:
    """Conta le clausole (verbi finiti) in una frase."""
    return sum(1 for token in doc_sent if token.pos_ == "VERB" and token.morph.get("VerbForm") == ['Fin'])

def clause_density(doc: spacy.tokens.Doc) -> float:
    """Calcola la densità media di clausole per frase."""
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    if num_sentences == 0:
        return 0.0

    total_clauses = sum(count_clauses(sentence) for sentence in sentences)
    return round(total_clauses / num_sentences, 2)

def gulpease_index(doc: spacy.tokens.Doc) -> float:
    """Calcola l'indice Gulpease."""
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

def extract_concepts(doc: spacy.tokens.Doc) -> list:
    """Estrae concetti (sostantivi + entità nominate)."""
    concepts = set(token.text.lower() for token in doc if token.pos_ == "NOUN")
    concepts.update(ent.text.lower() for ent in doc.ents)
    return list(concepts)

def deduplicate_concepts(concepts: list) -> list:
    """Deduplica i concetti usando TF-IDF e DBSCAN."""
    if not concepts:
        return []

    vectorizer = TfidfVectorizer()
    try:
        embeddings = vectorizer.fit_transform(concepts).toarray()
    except ValueError:
        # Potrebbe fallire se i concetti sono solo stop-word, ecc.
        return concepts

    if embeddings.shape[0] == 0:
        return []

    # Clustering per ridurre duplicati
    try:
        clustering = DBSCAN(eps=0.3, min_samples=1, metric="cosine").fit(embeddings)
        unique_indices = np.unique(clustering.labels_, return_index=True)[1]
        deduplicated = [concepts[i] for i in sorted(unique_indices)]
        return deduplicated
    except Exception as e:
        logging.warning(f"Clustering DBSCAN fallito: {e}. Restituisco concetti non deduplicati.")
        return concepts

def conceptual_density(doc: spacy.tokens.Doc) -> float:
    """Calcola la densità concettuale (concetti unici / frasi)."""
    concepts = extract_concepts(doc)
    deduplicated_concepts = deduplicate_concepts(concepts)
    num_sentences = len(list(doc.sents))
    
    return round(len(deduplicated_concepts) / num_sentences, 2) if num_sentences > 0 else 0.0

def calculate_coherence(doc: spacy.tokens.Doc) -> float:
    """Calcola la coerenza (similarità cosina media tra frasi adiacenti)."""
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
        # Fallisce se il vocabolario è vuoto (es. solo punteggiatura)
        return 0.0

def lda_thematic_progression(doc: spacy.tokens.Doc, num_topics: int = 2) -> float:
    """Calcola la progressione tematica (similarità LDA tra frasi adiacenti)."""
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if len(sentences) < 2:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english") # Meglio specificare una lingua
        dt_matrix = vectorizer.fit_transform(sentences)
        
        # Se la matrice è vuota o ha meno feature dei topic
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
        # Fallisce se il vocabolario è vuoto
        return 0.0

def shannon_entropy(doc: spacy.tokens.Doc) -> float:
    """Calcola l'entropia di Shannon basata sulla frequenza delle parole."""
    words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    if not words:
        return 0.0
        
    word_count = Counter(words)
    total_words = len(words)
    
    entropy = -sum((freq / total_words) * np.log2(freq / total_words) for freq in word_count.values())
    return round(entropy, 2)


# --- Funzione Principale di Analisi ---

def analyze_dataframe(df: pd.DataFrame, nlp: spacy.Language, data_dir: Path) -> pd.DataFrame:
    """
    Applica l'analisi di complessità a un DataFrame.
    """
    logging.info("Tokenizzazione dei testi con spaCy... (potrebbe richiedere tempo)")
    # Disabilita i componenti non necessari per velocizzare
    disabled_pipes = ["ner"] if "conceptual_density" not in [f.__name__ for f in lista_funct] else []
    
    df['testo_tok'] = list(tqdm(
        nlp.pipe(df['testo'], disable=disabled_pipes), 
        total=len(df), 
        desc="Elaborazione spaCy"
    ))

    # Pre-associa l'argomento data_dir a word_freq
    word_freq_partial = partial(word_freq, data_dir=data_dir)
    word_freq_partial.__name__ = "word_freq" # Mantiene il nome corretto

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
        word_freq_partial # Usa la funzione parziale
    ]

    logging.info("Calcolo delle metriche di complessità...")
    for funct in tqdm(lista_funct, desc="Calcolo Metriche"):
        df[funct.__name__] = df['testo_tok'].apply(funct)

    # Rimuove la colonna spaCy (pesante) prima di salvare
    df = df.drop(columns=['testo_tok'])

    return df

# --- Blocco di Esecuzione Principale ---

def main():
    """
    Punto di ingresso principale per lo script da riga di comando.
    """
    parser = argparse.ArgumentParser(description="Analizzatore di Complessità Testuale")
    parser.add_argument(
        "-i", "--input_file", 
        type=str, 
        required=True, 
        help="Percorso del file di input (.csv o .parquet)."
    )
    parser.add_argument(
        "-o", "--output_file", 
        type=str, 
        required=True, 
        help="Percorso del file di output (.csv o .parquet)."
    )
    parser.add_argument(
        "-d", "--data_dir", 
        type=str, 
        required=True, 
        help="Directory contenente i file .txt del vocabolario (es. 'parole_fondamentali.txt')."
    )
    parser.add_argument(
        "-t", "--text_column", 
        type=str, 
        default="testo", 
        help="Nome della colonna contenente i testi (default: 'testo')."
    )
    
    args = parser.parse_args()

    # Trasforma le stringhe dei percorsi in oggetti Path
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    data_dir_path = Path(args.data_dir)

    # 1. Carica il modello spaCy
    logging.info("Caricamento del modello spaCy 'it_core_news_lg'...")
    try:
        nlp = spacy.load("it_core_news_lg")
    except OSError:
        logging.error("Modello 'it_core_news_lg' non trovato.")
        logging.error("Esegui 'python -m spacy download it_core_news_lg' dal tuo terminale.")
        return

    # 2. Carica i dati
    logging.info(f"Caricamento dati da {input_path}")
    if str(input_path).endswith(".csv"):
        df = pd.read_csv(input_path)
    elif str(input_path).endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        logging.error("Formato file non supportato. Usare .csv o .parquet")
        return

    if args.text_column not in df.columns:
        logging.error(f"Colonna '{args.text_column}' non trovata nel file.")
        return

    # 3. Esegui l'analisi
    df_results = analyze_dataframe(df, nlp, data_dir_path)

    # 4. Salva i risultati
    logging.info(f"Salvataggio dei risultati in {output_path}")
    if str(output_path).endswith(".csv"):
        df_results.to_csv(output_path, index=False)
    elif str(output_path).endswith(".parquet"):
        df_results.to_parquet(output_path, index=False)
    
    logging.info("Analisi completata con successo.")

if __name__ == "__main__":
    main()
