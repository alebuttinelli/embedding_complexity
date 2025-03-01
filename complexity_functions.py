!python -m spacy download it_core_news_lg

import re
import spacy
import numpy as np
import time
from collections import defaultdict
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from collections import Counter
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

def word_length(doc):
  total_tokens = len(doc)
  total_length = 0
  for token in doc:
    total_length += len(token.text)
  return round(total_length/total_tokens, 2)

#rapporto tra numero totale di token e numero totale di lemmi
def type_token_ratio(doc):
  tokens = [token.text for token in doc]
  types = list(set([token.lemma_ for token in doc]))
  return round(len(tokens)/len(types), 2)

def word_freq(doc):
  with open("/content/drive/MyDrive/complessita/parole_fondamentali.txt", 'r', encoding='utf-8') as f:
    p_fondamentali = f.read().split(' ')
  with open("/content/drive/MyDrive/complessita/parole_alto_uso.txt", 'r', encoding='utf-8') as g:
    p_alto_uso = g.read().split(' ')
  with open("/content/drive/MyDrive/complessita/parole_alta_disponibilita.txt", 'r', encoding='utf-8') as e:
    p_alta_disponibilita = e.read().split()
  lemmas = [token.lemma_ for token in doc]
  p_fondamentali_counter = 0
  p_alto_uso_counter = 0
  p_alta_disponibilita_counter = 0
  p_basso_uso = 0
  for token in lemmas:
    if token in p_fondamentali:
      p_fondamentali_counter += 1
    elif token in p_alto_uso:
      p_alto_uso_counter += 1
    elif token in p_alta_disponibilita:
      p_alta_disponibilita_counter += 1
    else:
      p_basso_uso += 1
  vocab_fondamentale = p_fondamentali_counter+p_alto_uso_counter+p_alta_disponibilita_counter

  return round(vocab_fondamentale/len(doc), 2)

def avg_sentence_length(doc):
  sentences_count = len([sent.text.strip() for sent in doc.sents])
  words_count = len(doc)
  return round(words_count/sentences_count, 2)

def compute_depth(node):
    """Calcola ricorsivamente la profondità dell'albero sintattico."""
    if not list(node.children):  # Se il nodo non ha figli
        return 0
    else:
        return 1 + max(compute_depth(child) for child in node.children)

def count_nodes_per_level(node, level=0, levels=None):
    """Conta i nodi per livello nell'albero sintattico."""
    if levels is None:
        levels = defaultdict(int)
    levels[level] += 1
    for child in node.children:
        count_nodes_per_level(child, level + 1, levels)
    return levels

def calculate_avg_depth(doc):
    """Calcola la profondità media delle frasi nel documento."""
    all_depths = []

    for sent in doc.sents:
        root = [token for token in sent if token.head == token][0]  # Trova la radice
        tree_depth = compute_depth(root)
        all_depths.append(tree_depth)

    avg_depth = sum(all_depths) / len(all_depths) if all_depths else 0
    return round(avg_depth, 2)

def calculate_avg_width(doc):
    """Calcola la larghezza media delle frasi nel documento."""
    all_widths = []

    for sent in doc.sents:
        root = [token for token in sent if token.head == token][0]  # Trova la radice
        levels = count_nodes_per_level(root)
        tree_width = max(levels.values())  # Larghezza = massimo numero di nodi a un livello
        all_widths.append(tree_width)

    avg_width = sum(all_widths) / len(all_widths) if all_widths else 0
    return round(avg_width, 2)

#Clause Density: Average number of clauses per sentence.
def count_clauses(doc):
    """
    Count the number of clauses in a sentence.
    A clause is approximated by counting finite verbs (excluding participles and infinitives).
    """
    clause_count = 0
    for token in doc:
        # Check if token is a verb and finite (non-auxiliary, non-infinitive)
        if token.pos_ == "VERB" and token.morph.get("VerbForm") == ['Fin']:
            clause_count += 1
    return clause_count

def clause_density(doc):
    """
    Calculate the clause density for a text.
    Args:
    - text (str): The input text, which may contain multiple sentences.

    Returns:
    - clause_density (float): The average number of clauses per sentence.
    """
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    if num_sentences == 0:
        return 0

    total_clauses = sum(count_clauses(sentence) for sentence in sentences)
    clause_density = total_clauses / num_sentences
    return round(clause_density, 2)

def gulpease_index(doc):
  #numero di lettere
  letters_count = 0
  for token in doc:
    letters_count += len(token.text)
  #numero di parole
  words_count = len(doc)
  #numero di frasi
  sentences = [sent.text.strip() for sent in doc.sents]
  sentences_count = len(sentences)
  return round(89 - 1/10*(100*letters_count/words_count) + 3*(100*sentences_count/words_count), 2)

def extract_concepts(doc):
    """
    Estrae concetti principali da un testo utilizzando sostantivi ed entità nominate.
    """
    concepts = set()

    # Estrazione di sostantivi
    for token in doc:
        if token.pos_ == "NOUN":
            concepts.add(token.text.lower())

    # Estrazione di entità nominate
    for ent in doc.ents:
        concepts.add(ent.text.lower())

    return list(concepts)

def deduplicate_concepts(concepts):
    """
    Deduplica i concetti usando la similarità semantica basata su TF-IDF.
    Utilizza DBSCAN per clusterizzare concetti simili.
    """
    if not concepts:
        return []

    # Trasforma i concetti in vettori TF-IDF
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(concepts).toarray()

    # Clustering per ridurre duplicati
    clustering = DBSCAN(eps=0.3, min_samples=1, metric="cosine").fit(embeddings)
    unique_indices = np.unique(clustering.labels_, return_index=True)[1]

    deduplicated = [concepts[i] for i in sorted(unique_indices)]
    return deduplicated

def conceptual_density(doc):
    """
    Calcola la densità concettuale come concetti distinti per frase.
    """
    start_time = time.time()  # Inizia il timer

    concepts = extract_concepts(doc)
    deduplicated_concepts = deduplicate_concepts(concepts)

    num_sentences = len(list(doc.sents))
    conceptual_density = len(deduplicated_concepts) / num_sentences if num_sentences > 0 else 0

    end_time = time.time()  # Fine del timer
    print(f"Tempo di esecuzione: {round(end_time - start_time, 2)} secondi")

    return round(conceptual_density, 2)

def calculate_coherence(doc):
    """
    Calcola la coerenza testuale come media della similarità cosina tra frasi adiacenti.

    Args:
        doc (spacy.Doc): Documento tokenizzato da spaCy.

    Returns:
        float: Punteggio di coerenza (media della similarità cosina tra frasi adiacenti).
    """
    # Tokenizza il testo in frasi
    sentences = [sent.text.strip() for sent in doc.sents]

    # Controllo che ci siano almeno due frasi per calcolare la coerenza
    if len(sentences) < 2:
        return 0.0

    # Genera embedding con TF-IDF
    vectorizer = TfidfVectorizer()
    sentence_embeddings = vectorizer.fit_transform(sentences).toarray()

    # Calcola la similarità cosina tra frasi adiacenti, con tqdm per il tracking
    similarities = [
        cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i + 1]])[0][0]
        for i in tqdm(range(len(sentence_embeddings) - 1), desc="Calcolando coerenza", leave=False)
    ]

    # Restituisce la media della similarità cosina
    return round(np.mean(similarities), 2) if similarities else 0.0

#Thematic Progression: Measures how topics evolve through the text.
#LDA for Topic Evolution:
#Use Latent Dirichlet Allocation (LDA) to analyze topic distributions and transitions between paragraphs.

#Thematic Progression: Measures how topics evolve through the text.
#LDA for Topic Evolution:
#Use Latent Dirichlet Allocation (LDA) to analyze topic distributions and transitions between paragraphs.

from sklearn.feature_extraction.text import TfidfVectorizer

def lda_thematic_progression(doc, num_topics=2):
    """
    Calculate thematic progression by analyzing topic distributions across sentences
    using Latent Dirichlet Allocation (LDA).

    Args:
        stringa (str): Input text.
        num_topics (int): Number of topics to identify with LDA.

    Returns:
        float: Mean cosine similarity between topic distributions of consecutive sentences.
    """
    sentences = [sent.text.strip() for sent in doc.sents]  # Extract clean sentences

    # Handle the case of a single sentence
    if len(sentences) < 2:
        return 0.0  # If there is only one sentence, progression cannot be calculated

    vectorizer = TfidfVectorizer()
    dt_matrix = vectorizer.fit_transform(sentences)  # Create a document-term matrix

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    topic_distributions = lda.fit_transform(dt_matrix)  # Get topic distributions for each sentence

    similarities = []
    for i in range(len(topic_distributions) - 1):  # Iterate over pairs of consecutive sentences
        sim = cosine_similarity([topic_distributions[i]], [topic_distributions[i + 1]])[0][0]
        similarities.append(sim)

    mean_similarity = np.mean(similarities) if similarities else 0  # Handle edge case for no comparisons
    return round(mean_similarity, 2)  # Return the rounded thematic progression score

def shannon_entropy(doc):
    """
    Calculate the Shannon entropy based on word frequencies using SpaCy for tokenization.
    """
    # Extract words (excluding punctuation and spaces)
    words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    word_count = Counter(words)
    total_words = len(words)
    # Calculate entropy
    entropy = -sum((freq / total_words) * np.log2(freq / total_words) for freq in word_count.values())
    return round(entropy, 2)

def embedding_complexity(df, normalizzazione=False):

    nlp = spacy.load("it_core_news_lg")
    df['testo_tok'] = df['testo'].apply(nlp)

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
        shannon_entropy
    ]

    for funct in lista_funct:
        df[funct.__name__] = df['testo_tok'].apply(funct)

    df['embedding'] = text_to_pca_embedding(df['testo_tok'].to_list())

    # Normalizzazione dei valori numerici
    if normalizzazione:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = normalize(df[numerical_cols])

    return df
