"""
mvp1_vector_analytics_tests.py

MVP-1 Vector Analytics Testing Framework

Purpose:
    Compare tabular (ground truth) vs. vector-based query results
    for analytics questions on synthetic pet data.

    This tests whether counting documents via vector similarity
    can approximate accurate structured-query counts.

Tests A-E (per specification):
    A: How many cats? (single concept count)
    B: How many cats that can fly? (concept + 1 boolean filter)
    C: How many cats born after time T? (concept + time filter)
    D: How many cats born after T that can fly? (concept + time + 1 filter)
    E: How many cats with time + 10 field constraints? (stress test)

Methods compared:
    1. Tabular: pandas boolean filtering (ground truth)
    2. Vector: cosine similarity count above threshold

Key design decisions:
    - Uses sentence-transformers directly (no ChromaDB black box)
    - Exhaustive cosine similarity (not approximate nearest neighbor)
    - All similarity scores retained for analysis
    - Model loaded once, reused for all queries

Dependencies:
    - pandas
    - numpy
    - sentence-transformers

Author: [MVP-1 Test Framework]
Date: 2024
"""

import time
import traceback
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Path to CSV file containing pet data with unstructured_description field
this = input("enter path\n")
CSV_FILE_PATH: str = this

# Sentence-transformer model (same default as ChromaDB)
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Similarity threshold: documents with similarity > this are counted as matches
# Note: This is a critical parameter that significantly affects results
VECTOR_SIMILARITY_THRESHOLD: float = 0.5

# Random seed for any stochastic operations (reproducibility)
RANDOM_SEED: int = 42

# Numeric thresholds for Query E (10-field stress test)
# These define the filter criteria for the complex query
QUERY_E_WEIGHT_KG_MIN: float = 5.0           # weight_kg > this
QUERY_E_HEIGHT_CM_MAX: float = 80.0          # height_cm < this
QUERY_E_AGE_YEARS_MIN: int = 3               # age_years > this
QUERY_E_FRIENDS_MIN: int = 3                 # number_of_friends > this
QUERY_E_COLOR_VALUE: str = "red"             # color == this
QUERY_E_DAILY_FOOD_GRAMS_MIN: float = 100.0  # daily_food_grams > this
# Boolean fields for Query E: can_fly, can_swim, can_run, watches_youtube
# All set to True as per specification


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_csv_data(csv_file_path: str) -> pd.DataFrame:
    """
    Load synthetic pet data CSV into pandas DataFrame.

    Validates that all required columns exist for both tabular
    queries and vector embedding generation.

    Parameters
    ----------
    csv_file_path : str
        Path to CSV file. Can be relative or absolute.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all columns from CSV.
        Expected row count depends on synthetic data generation settings.

    Raises
    ------
    FileNotFoundError
        If CSV file does not exist at specified path.
    ValueError
        If required columns are missing from CSV.

    Notes
    -----
    Required columns for MVP-1 tests:
        - animal_type: categorical (cat, dog, bird, fish, turtle)
        - weight_kg: float
        - height_cm: float
        - age_years: int
        - number_of_friends: int
        - birth_date: string (YYYY-MM-DD format)
        - birth_unix: int (Unix timestamp)
        - color: categorical
        - can_fly: boolean
        - can_swim: boolean
        - can_run: boolean
        - watches_youtube: boolean
        - daily_food_grams: float
        - popularity_score: float
        - unstructured_description: string (text blob for embedding)
    """
    try:
        file_path = Path(csv_file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"CSV file not found at path: {csv_file_path}\n"
                f"Current working directory: {Path.cwd()}"
            )

        dataframe = pd.read_csv(file_path)

        # Define required columns for MVP-1 test queries
        required_columns = [
            "animal_type",
            "weight_kg",
            "height_cm",
            "age_years",
            "number_of_friends",
            "birth_date",
            "birth_unix",
            "color",
            "can_fly",
            "can_swim",
            "can_run",
            "watches_youtube",
            "daily_food_grams",
            "popularity_score",
            "unstructured_description",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"CSV missing required columns: {missing_columns}\n"
                f"Found columns: {list(dataframe.columns)}"
            )

        # Report data load summary
        print(f"[DATA LOAD] Successfully loaded: {csv_file_path}")
        print(f"[DATA LOAD] Row count: {len(dataframe)}")
        print(f"[DATA LOAD] Column count: {len(dataframe.columns)}")

        # Report animal type distribution for context
        animal_counts = dataframe["animal_type"].value_counts()
        print("[DATA LOAD] Animal type distribution:")
        for animal_type, count in animal_counts.items():
            print(f"            {animal_type}: {count}")

        return dataframe

    except FileNotFoundError:
        # Re-raise file not found with original message
        raise

    except ValueError:
        # Re-raise validation errors with original message
        raise

    except Exception as unexpected_error:
        print(f"[ERROR] Unexpected error loading CSV: {unexpected_error}")
        traceback.print_exc()
        raise


# =============================================================================
# EMBEDDING GENERATION FUNCTIONS
# =============================================================================

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load and return a sentence-transformer model.

    This function exists to centralize model loading,
    ensuring the model is loaded once and reused.

    Parameters
    ----------
    model_name : str
        Name of sentence-transformer model.
        Example: 'all-MiniLM-L6-v2' (384 dimensions)

    Returns
    -------
    SentenceTransformer
        Loaded model ready for encoding.

    Notes
    -----
    Model is downloaded on first use if not cached locally.
    Default model 'all-MiniLM-L6-v2' produces 384-dimensional vectors.
    """
    try:
        print(f"[MODEL] Loading embedding model: {model_name}")
        start_time = time.time()

        model = SentenceTransformer(model_name)

        elapsed_seconds = time.time() - start_time
        print(f"[MODEL] Model loaded in {elapsed_seconds:.2f} seconds")

        return model

    except Exception as model_load_error:
        print(f"[ERROR] Failed to load embedding model: {model_load_error}")
        traceback.print_exc()
        raise


def generate_corpus_embeddings(
    text_documents: list[str],
    embedding_model: SentenceTransformer
) -> np.ndarray:
    """
    Generate embeddings for all documents in corpus.

    This is run once before queries, storing all document
    embeddings for repeated similarity calculations.

    Parameters
    ----------
    text_documents : list[str]
        List of text strings (unstructured descriptions).
        Each string becomes one embedding vector.

    embedding_model : SentenceTransformer
        Pre-loaded sentence-transformer model.

    Returns
    -------
    np.ndarray
        Matrix of shape (n_documents, embedding_dimension).
        For 'all-MiniLM-L6-v2', embedding_dimension is 384.

    Notes
    -----
    This may take significant time for large corpora.
    Progress bar is displayed during encoding.
    """
    try:
        document_count = len(text_documents)
        print(f"[EMBED] Generating embeddings for {document_count} documents...")

        start_time = time.time()

        # Generate embeddings with progress bar
        embeddings_matrix = embedding_model.encode(
            text_documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        elapsed_seconds = time.time() - start_time

        print(f"[EMBED] Embeddings complete in {elapsed_seconds:.2f} seconds")
        print(f"[EMBED] Embedding matrix shape: {embeddings_matrix.shape}")
        print(f"[EMBED] Embedding dimension: {embeddings_matrix.shape[1]}")

        return embeddings_matrix

    except Exception as embedding_error:
        print(f"[ERROR] Failed to generate embeddings: {embedding_error}")
        traceback.print_exc()
        raise


def embed_single_query(
    query_text: str,
    embedding_model: SentenceTransformer
) -> np.ndarray:
    """
    Generate embedding vector for a single query string.

    Parameters
    ----------
    query_text : str
        Natural language query to embed.

    embedding_model : SentenceTransformer
        Pre-loaded sentence-transformer model.

    Returns
    -------
    np.ndarray
        1D embedding vector of shape (embedding_dimension,).
    """
    # encode() returns 2D array, we extract the single vector
    embedding_2d = embedding_model.encode(
        [query_text],
        convert_to_numpy=True
    )
    embedding_1d = embedding_2d[0]

    return embedding_1d


# =============================================================================
# VECTOR SIMILARITY FUNCTIONS
# =============================================================================

def calculate_cosine_similarities(
    query_vector: np.ndarray,
    corpus_embeddings_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine similarity between query and all corpus documents.

    This is EXHAUSTIVE comparison (not approximate nearest neighbor).
    Every document gets a similarity score, enabling full analytics.

    Parameters
    ----------
    query_vector : np.ndarray
        Query embedding, shape (embedding_dim,) or (1, embedding_dim).

    corpus_embeddings_matrix : np.ndarray
        All document embeddings, shape (n_documents, embedding_dim).

    Returns
    -------
    np.ndarray
        Similarity scores for all documents, shape (n_documents,).
        Values range from -1.0 to 1.0.
        1.0 = identical direction (most similar)
        0.0 = orthogonal (unrelated)
        -1.0 = opposite direction (least similar)

    Notes
    -----
    Formula: cosine_similarity = (A · B) / (||A|| * ||B||)

    This function handles edge case of zero-norm vectors
    by substituting a small epsilon to avoid division by zero.
    """
    # Ensure query is 1D for dot product
    query_vector_flat = query_vector.flatten()

    # Dot product: each document dotted with query
    # Shape: (n_documents,)
    dot_products = corpus_embeddings_matrix @ query_vector_flat

    # Calculate norms for normalization
    corpus_norms = norm(corpus_embeddings_matrix, axis=1)  # shape: (n_documents,)
    query_norm = norm(query_vector_flat)                    # scalar

    # Compute denominator, avoiding division by zero
    denominator = corpus_norms * query_norm

    # Replace any zero denominators with small epsilon
    epsilon = 1e-10
    denominator_safe = np.where(
        denominator == 0,
        epsilon,
        denominator
    )

    # Final cosine similarities
    similarities = dot_products / denominator_safe

    return similarities


def count_documents_above_threshold(
    similarity_scores: np.ndarray,
    threshold: float
) -> int:
    """
    Count how many documents have similarity above threshold.

    This is the primary metric for vector-based counting queries.

    Parameters
    ----------
    similarity_scores : np.ndarray
        Array of similarity scores for all documents.

    threshold : float
        Minimum similarity to count as a "match".
        Typical values: 0.3 (loose), 0.5 (moderate), 0.7 (strict).

    Returns
    -------
    int
        Count of documents exceeding threshold.
    """
    count = int(np.sum(similarity_scores > threshold))
    return count


def compute_similarity_statistics(
    similarity_scores: np.ndarray
) -> dict:
    """
    Compute descriptive statistics on similarity score distribution.

    Useful for understanding query behavior beyond simple counts.

    Parameters
    ----------
    similarity_scores : np.ndarray
        Array of similarity scores for all documents.

    Returns
    -------
    dict
        Dictionary containing:
        - mean: average similarity
        - std: standard deviation
        - min: minimum similarity
        - max: maximum similarity
        - median: median similarity
        - q25: 25th percentile
        - q75: 75th percentile
    """
    stats = {
        "mean": float(np.mean(similarity_scores)),
        "std": float(np.std(similarity_scores)),
        "min": float(np.min(similarity_scores)),
        "max": float(np.max(similarity_scores)),
        "median": float(np.median(similarity_scores)),
        "q25": float(np.percentile(similarity_scores, 25)),
        "q75": float(np.percentile(similarity_scores, 75)),
    }
    return stats


# =============================================================================
# TABULAR QUERY FUNCTIONS (GROUND TRUTH)
# =============================================================================

def tabular_query_a_count_cats(
    dataframe: pd.DataFrame
) -> int:
    """
    Query A (Tabular Ground Truth): How many cats?

    Simple single-field filter counting all cat records.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pet data with animal_type column.

    Returns
    -------
    int
        Exact count of rows where animal_type == 'cat'.
    """
    boolean_mask = dataframe["animal_type"] == "cat"
    count = int(boolean_mask.sum())
    return count


def tabular_query_b_count_cats_that_can_fly(
    dataframe: pd.DataFrame
) -> int:
    """
    Query B (Tabular Ground Truth): How many cats that can fly?

    Two-field filter: animal type AND boolean ability.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pet data with animal_type and can_fly columns.

    Returns
    -------
    int
        Exact count of rows where animal_type == 'cat' AND can_fly == True.
    """
    boolean_mask = (
        (dataframe["animal_type"] == "cat") &
        (dataframe["can_fly"] == True)
    )
    count = int(boolean_mask.sum())
    return count


def tabular_query_c_count_cats_born_after_time(
    dataframe: pd.DataFrame,
    time_threshold_unix: int
) -> int:
    """
    Query C (Tabular Ground Truth): How many cats born after time T?

    Two-field filter: animal type AND time comparison.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pet data with animal_type and birth_unix columns.

    time_threshold_unix : int
        Unix timestamp. Records with birth_unix > this are counted.

    Returns
    -------
    int
        Exact count of cats born after the threshold time.
    """
    boolean_mask = (
        (dataframe["animal_type"] == "cat") &
        (dataframe["birth_unix"] > time_threshold_unix)
    )
    count = int(boolean_mask.sum())
    return count


def tabular_query_d_count_cats_born_after_time_can_fly(
    dataframe: pd.DataFrame,
    time_threshold_unix: int
) -> int:
    """
    Query D (Tabular Ground Truth): Cats born after T that can fly?

    Three-field filter: animal type AND time AND boolean ability.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pet data with animal_type, birth_unix, and can_fly columns.

    time_threshold_unix : int
        Unix timestamp threshold.

    Returns
    -------
    int
        Exact count of flying cats born after threshold.
    """
    boolean_mask = (
        (dataframe["animal_type"] == "cat") &
        (dataframe["birth_unix"] > time_threshold_unix) &
        (dataframe["can_fly"] == True)
    )
    count = int(boolean_mask.sum())
    return count


def tabular_query_e_count_cats_10_field_stress_test(
    dataframe: pd.DataFrame,
    time_threshold_unix: int,
    weight_kg_min: float,
    height_cm_max: float,
    age_years_min: int,
    number_of_friends_min: int,
    color_value: str,
    daily_food_grams_min: float
) -> int:
    """
    Query E (Tabular Ground Truth): Stress test with 10+ field filters.

    Full filter set per specification:
        - animal_type == 'cat'
        - birth_unix > time_threshold
        - weight_kg > weight_kg_min
        - height_cm < height_cm_max
        - age_years > age_years_min
        - number_of_friends > number_of_friends_min
        - color == color_value
        - can_fly == True
        - can_swim == True
        - can_run == True
        - watches_youtube == True
        - daily_food_grams > daily_food_grams_min

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pet data with all required columns.

    time_threshold_unix : int
        Unix timestamp threshold for birth_unix.

    weight_kg_min : float
        Minimum weight in kilograms (exclusive: >).

    height_cm_max : float
        Maximum height in centimeters (exclusive: <).

    age_years_min : int
        Minimum age in years (exclusive: >).

    number_of_friends_min : int
        Minimum friend count (exclusive: >).

    color_value : str
        Required color value (exact match).

    daily_food_grams_min : float
        Minimum daily food in grams (exclusive: >).

    Returns
    -------
    int
        Exact count of records matching ALL criteria.

    Notes
    -----
    This is a stress test for vector embeddings.
    Tabular query handles 12 constraints trivially.
    Vector embedding may struggle to capture all constraints.
    """
    boolean_mask = (
        (dataframe["animal_type"] == "cat") &
        (dataframe["birth_unix"] > time_threshold_unix) &
        (dataframe["weight_kg"] > weight_kg_min) &
        (dataframe["height_cm"] < height_cm_max) &
        (dataframe["age_years"] > age_years_min) &
        (dataframe["number_of_friends"] > number_of_friends_min) &
        (dataframe["color"] == color_value) &
        (dataframe["can_fly"] == True) &
        (dataframe["can_swim"] == True) &
        (dataframe["can_run"] == True) &
        (dataframe["watches_youtube"] == True) &
        (dataframe["daily_food_grams"] > daily_food_grams_min)
    )
    count = int(boolean_mask.sum())
    return count


# =============================================================================
# VECTOR QUERY FUNCTIONS
# =============================================================================

def vector_query_a_count_cats(
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float
) -> dict:
    """
    Query A (Vector): How many documents about cats?

    Embeds natural language query and counts documents above threshold.

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    Returns
    -------
    dict
        Contains:
        - count: number of documents above threshold
        - similarities: full similarity array
        - query_text: the query that was embedded
        - elapsed_seconds: query execution time
    """
    query_text = "cat, feline"

    start_time = time.time()

    query_embedding = embed_single_query(query_text, embedding_model)
    similarities = calculate_cosine_similarities(query_embedding, corpus_embeddings)
    count = count_documents_above_threshold(similarities, similarity_threshold)

    elapsed_seconds = time.time() - start_time

    return {
        "count": count,
        "similarities": similarities,
        "query_text": query_text,
        "elapsed_seconds": elapsed_seconds,
    }


def vector_query_b_count_cats_that_can_fly(
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float
) -> dict:
    """
    Query B (Vector): How many documents about cats that can fly?

    Natural language query combining animal type and ability.

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    Returns
    -------
    dict
        Contains count, similarities, query_text, elapsed_seconds.
    """
    query_text = "A cat, feline, that can fly. A flying cat."

    start_time = time.time()

    query_embedding = embed_single_query(query_text, embedding_model)
    similarities = calculate_cosine_similarities(query_embedding, corpus_embeddings)
    count = count_documents_above_threshold(similarities, similarity_threshold)

    elapsed_seconds = time.time() - start_time

    return {
        "count": count,
        "similarities": similarities,
        "query_text": query_text,
        "elapsed_seconds": elapsed_seconds,
    }


def vector_query_c_count_cats_born_after_time(
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    time_threshold_year: int
) -> dict:
    """
    Query C (Vector): How many documents about cats born after year T?

    Note: Embedding models may struggle with precise temporal reasoning.
    Year is expressed in natural language form.

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    time_threshold_year : int
        Year threshold (e.g., 2020). Used in query text.

    Returns
    -------
    dict
        Contains count, similarities, query_text, elapsed_seconds.
    """
    query_text = (
        f"A cat, feline, born after the year {time_threshold_year}. "
        f"A recently born cat."
    )

    start_time = time.time()

    query_embedding = embed_single_query(query_text, embedding_model)
    similarities = calculate_cosine_similarities(query_embedding, corpus_embeddings)
    count = count_documents_above_threshold(similarities, similarity_threshold)

    elapsed_seconds = time.time() - start_time

    return {
        "count": count,
        "similarities": similarities,
        "query_text": query_text,
        "elapsed_seconds": elapsed_seconds,
    }


def vector_query_d_count_cats_born_after_time_can_fly(
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    time_threshold_year: int
) -> dict:
    """
    Query D (Vector): Cats born after year T that can fly?

    Combines animal type, time constraint, and ability in query.

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    time_threshold_year : int
        Year threshold for query text.

    Returns
    -------
    dict
        Contains count, similarities, query_text, elapsed_seconds.
    """
    query_text = (
        f"A cat, feline, born after the year {time_threshold_year}, "
        f"that can fly. A flying cat born recently."
    )

    start_time = time.time()

    query_embedding = embed_single_query(query_text, embedding_model)
    similarities = calculate_cosine_similarities(query_embedding, corpus_embeddings)
    count = count_documents_above_threshold(similarities, similarity_threshold)

    elapsed_seconds = time.time() - start_time

    return {
        "count": count,
        "similarities": similarities,
        "query_text": query_text,
        "elapsed_seconds": elapsed_seconds,
    }


def vector_query_e_count_cats_10_field_stress_test(
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    time_threshold_year: int,
    weight_kg_min: float,
    height_cm_max: float,
    age_years_min: int,
    number_of_friends_min: int,
    color_value: str,
    daily_food_grams_min: float
) -> dict:
    """
    Query E (Vector): Stress test with 10+ field descriptions.

    Natural language query attempting to capture ALL constraints
    from the tabular stress test. This tests the limits of how
    many concepts an embedding can meaningfully represent.

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    time_threshold_year : int
        Year threshold for birth time.

    weight_kg_min : float
        Minimum weight to express in query.

    height_cm_max : float
        Maximum height to express in query.

    age_years_min : int
        Minimum age to express in query.

    number_of_friends_min : int
        Minimum friends to express in query.

    color_value : str
        Color to match.

    daily_food_grams_min : float
        Minimum food amount to express in query.

    Returns
    -------
    dict
        Contains count, similarities, query_text, elapsed_seconds.

    Notes
    -----
    This query text is intentionally verbose to include all constraints.
    Expected behavior: embedding may not capture all numeric constraints
    precisely, leading to higher error vs. tabular ground truth.
    """
    query_text = (
        f"A {color_value} cat, a {color_value} feline, "
        f"born after the year {time_threshold_year}. "
        f"This cat weighs more than {weight_kg_min} kg. "
        f"This cat is shorter than {height_cm_max} cm tall. "
        f"This cat is older than {age_years_min} years old. "
        f"This cat has more than {number_of_friends_min} friends. "
        f"This cat can fly, can swim, and can run. "
        f"This cat watches youtube. "
        f"This cat eats more than {daily_food_grams_min} grams of food per day."
    )

    start_time = time.time()

    query_embedding = embed_single_query(query_text, embedding_model)
    similarities = calculate_cosine_similarities(query_embedding, corpus_embeddings)
    count = count_documents_above_threshold(similarities, similarity_threshold)

    elapsed_seconds = time.time() - start_time

    return {
        "count": count,
        "similarities": similarities,
        "query_text": query_text,
        "elapsed_seconds": elapsed_seconds,
    }


# =============================================================================
# RESULTS ANALYSIS AND REPORTING
# =============================================================================

def calculate_error_metrics(
    ground_truth_count: int,
    vector_count: int
) -> dict:
    """
    Calculate error metrics comparing vector result to ground truth.

    Parameters
    ----------
    ground_truth_count : int
        Correct count from tabular query.

    vector_count : int
        Count from vector query.

    Returns
    -------
    dict
        Contains:
        - absolute_error: |vector_count - ground_truth_count|
        - percent_error: 100 * absolute_error / ground_truth (or None if ground_truth is 0)
        - direction: 'over' | 'under' | 'exact'
        - raw_difference: vector_count - ground_truth_count (signed)
    """
    absolute_error = abs(vector_count - ground_truth_count)
    raw_difference = vector_count - ground_truth_count

    if ground_truth_count > 0:
        percent_error = 100.0 * absolute_error / ground_truth_count
    else:
        # Cannot compute meaningful percent error when ground truth is zero
        percent_error = None

    if vector_count > ground_truth_count:
        direction = "over"
    elif vector_count < ground_truth_count:
        direction = "under"
    else:
        direction = "exact"

    return {
        "absolute_error": absolute_error,
        "percent_error": percent_error,
        "direction": direction,
        "raw_difference": raw_difference,
    }


def print_single_query_report(
    query_label: str,
    query_description: str,
    tabular_count: int,
    vector_result: dict,
    similarity_threshold: float
) -> dict:
    """
    Print detailed report for a single query comparison.

    Also returns a summary dictionary for aggregation.

    Parameters
    ----------
    query_label : str
        Short label (e.g., "A", "B", "C").

    query_description : str
        Human-readable description of query.

    tabular_count : int
        Ground truth count from tabular query.

    vector_result : dict
        Result dict from vector query function.

    similarity_threshold : float
        Threshold used for vector counting.

    Returns
    -------
    dict
        Summary containing query_label, counts, errors for aggregation.
    """
    vector_count = vector_result["count"]
    similarities = vector_result["similarities"]
    query_text = vector_result["query_text"]
    elapsed_seconds = vector_result["elapsed_seconds"]

    error_metrics = calculate_error_metrics(tabular_count, vector_count)
    similarity_stats = compute_similarity_statistics(similarities)

    # Print formatted report
    print(f"\n{'='*70}")
    print(f"QUERY {query_label}: {query_description}")
    print(f"{'='*70}")

    print("\n  Vector Query Text:")
    print(f"    \"{query_text}\"")

    print("\n  COUNTS:")
    print(f"    Tabular (ground truth):  {tabular_count}")
    print(f"    Vector (threshold={similarity_threshold}): {vector_count}")

    print("\n  ERROR METRICS:")
    print(f"    Absolute error:  {error_metrics['absolute_error']}")
    if error_metrics['percent_error'] is not None:
        print(f"    Percent error:   {error_metrics['percent_error']:.2f}%")
    else:
        print("    Percent error:   N/A (ground truth is 0)")
    print(f"    Direction:       {error_metrics['direction']}")
    print(f"    Raw difference:  {error_metrics['raw_difference']:+d}")

    print("\n  SIMILARITY DISTRIBUTION:")
    print(f"    Mean:    {similarity_stats['mean']:.4f}")
    print(f"    Std:     {similarity_stats['std']:.4f}")
    print(f"    Min:     {similarity_stats['min']:.4f}")
    print(f"    Q25:     {similarity_stats['q25']:.4f}")
    print(f"    Median:  {similarity_stats['median']:.4f}")
    print(f"    Q75:     {similarity_stats['q75']:.4f}")
    print(f"    Max:     {similarity_stats['max']:.4f}")

    print("\n  TIMING:")
    print(f"    Vector query time: {elapsed_seconds:.4f} seconds")

    # Return summary for aggregation
    summary = {
        "query_label": query_label,
        "query_description": query_description,
        "tabular_count": tabular_count,
        "vector_count": vector_count,
        "absolute_error": error_metrics["absolute_error"],
        "percent_error": error_metrics["percent_error"],
        "direction": error_metrics["direction"],
        "elapsed_seconds": elapsed_seconds,
    }

    return summary


def print_summary_table(query_summaries: list[dict]) -> None:
    """
    Print aggregated summary table of all query results.

    Parameters
    ----------
    query_summaries : list[dict]
        List of summary dicts from print_single_query_report.
    """
    print(f"\n{'='*90}")
    print("SUMMARY TABLE: TABULAR vs VECTOR QUERY RESULTS")
    print(f"{'='*90}")

    # Header
    print(
        f"{'Query':<6} "
        f"{'Tabular':<10} "
        f"{'Vector':<10} "
        f"{'Error':<10} "
        f"{'% Error':<12} "
        f"{'Direction':<10} "
        f"{'Time (s)':<10}"
    )
    print("-" * 90)

    # Rows
    for summary in query_summaries:
        percent_str = (
            f"{summary['percent_error']:.1f}%"
            if summary['percent_error'] is not None
            else "N/A"
        )

        print(
            f"{summary['query_label']:<6} "
            f"{summary['tabular_count']:<10} "
            f"{summary['vector_count']:<10} "
            f"{summary['absolute_error']:<10} "
            f"{percent_str:<12} "
            f"{summary['direction']:<10} "
            f"{summary['elapsed_seconds']:<10.4f}"
        )

    print("-" * 90)

    # Aggregate statistics
    total_tabular = sum(s["tabular_count"] for s in query_summaries)
    total_vector = sum(s["vector_count"] for s in query_summaries)
    total_absolute_error = sum(s["absolute_error"] for s in query_summaries)

    valid_percent_errors = [
        s["percent_error"] for s in query_summaries
        if s["percent_error"] is not None
    ]
    mean_percent_error = (
        sum(valid_percent_errors) / len(valid_percent_errors)
        if valid_percent_errors else None
    )

    print("\nAGGREGATE METRICS:")
    print(f"  Total tabular count:     {total_tabular}")
    print(f"  Total vector count:      {total_vector}")
    print(f"  Total absolute error:    {total_absolute_error}")
    if mean_percent_error is not None:
        print(f"  Mean percent error:      {mean_percent_error:.2f}%")


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_mvp1_test_suite(csv_file_path: str) -> list[dict]:
    """
    Execute complete MVP-1 test suite: Queries A through E.

    Workflow:
        1. Load CSV data
        2. Determine time threshold (median of birth_unix)
        3. Load embedding model
        4. Generate embeddings for all documents
        5. Run each query (A-E) in both tabular and vector forms
        6. Compare results and report errors
        7. Print summary table

    Parameters
    ----------
    csv_file_path : str
        Path to CSV file with pet data and unstructured_description.

    Returns
    -------
    list[dict]
        List of summary dictionaries, one per query.
        Useful for further analysis or export.

    Raises
    ------
    Exception
        Propagates any errors from data loading, embedding, or queries.
    """
    print("=" * 90)
    print("MVP-1 VECTOR ANALYTICS TEST SUITE")
    print("Comparing Tabular (Ground Truth) vs. Vector Query Results")
    print("=" * 90)

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    print(f"\n[CONFIG] Random seed: {RANDOM_SEED}")
    print(f"[CONFIG] Similarity threshold: {VECTOR_SIMILARITY_THRESHOLD}")
    print(f"[CONFIG] Embedding model: {EMBEDDING_MODEL_NAME}")

# -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 1: LOAD DATA")
    print(f"{'='*70}")

    dataframe = load_csv_data(csv_file_path)

    # -------------------------------------------------------------------------
    # Step 2: Determine time threshold dynamically
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 2: CONFIGURE TIME THRESHOLD")
    print(f"{'='*70}")

    # Use median of birth_unix as threshold
    # This ensures roughly half the data is on each side
    time_threshold_unix = int(dataframe["birth_unix"].median())

    # Convert Unix timestamp to year for natural language query
    time_threshold_datetime = datetime.datetime.fromtimestamp(time_threshold_unix)
    time_threshold_year = time_threshold_datetime.year

    print(f"[TIME] Median birth_unix: {time_threshold_unix}")
    print(f"[TIME] Corresponds to year: {time_threshold_year}")
    print(f"[TIME] Datetime: {time_threshold_datetime.strftime('%Y-%m-%d')}")

    # Count how many records are on each side of threshold for context
    records_before_threshold = int((dataframe["birth_unix"] <= time_threshold_unix).sum())
    records_after_threshold = int((dataframe["birth_unix"] > time_threshold_unix).sum())
    print(f"[TIME] Records before/on threshold: {records_before_threshold}")
    print(f"[TIME] Records after threshold: {records_after_threshold}")

    # -------------------------------------------------------------------------
    # Step 3: Load embedding model (once, reuse for all queries)
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 3: LOAD EMBEDDING MODEL")
    print(f"{'='*70}")

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

    # -------------------------------------------------------------------------
    # Step 4: Generate embeddings for all documents
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 4: GENERATE CORPUS EMBEDDINGS")
    print(f"{'='*70}")

    # Extract unstructured descriptions as list
    description_texts = dataframe["unstructured_description"].tolist()

    # Validate no null/empty descriptions
    null_count = dataframe["unstructured_description"].isna().sum()
    if null_count > 0:
        print(f"[WARNING] Found {null_count} null descriptions, these may cause issues")

    corpus_embeddings = generate_corpus_embeddings(description_texts, embedding_model)

    # -------------------------------------------------------------------------
    # Step 5: Run all queries and collect results
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 5: EXECUTE QUERIES A THROUGH E")
    print(f"{'='*70}")

    all_query_summaries = []

    # ----- Query A: How many cats? -----
    print("\n[RUNNING] Query A: How many cats?")

    tabular_count_a = tabular_query_a_count_cats(dataframe)

    vector_result_a = vector_query_a_count_cats(
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD
    )

    summary_a = print_single_query_report(
        query_label="A",
        query_description="How many cats?",
        tabular_count=tabular_count_a,
        vector_result=vector_result_a,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD
    )
    all_query_summaries.append(summary_a)

    # ----- Query B: How many cats that can fly? -----
    print("\n[RUNNING] Query B: How many cats that can fly?")

    tabular_count_b = tabular_query_b_count_cats_that_can_fly(dataframe)

    vector_result_b = vector_query_b_count_cats_that_can_fly(
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD
    )

    summary_b = print_single_query_report(
        query_label="B",
        query_description="How many cats that can fly?",
        tabular_count=tabular_count_b,
        vector_result=vector_result_b,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD
    )
    all_query_summaries.append(summary_b)

    # ----- Query C: How many cats born after time T? -----
    print(f"\n[RUNNING] Query C: How many cats born after {time_threshold_year}?")

    tabular_count_c = tabular_query_c_count_cats_born_after_time(
        dataframe=dataframe,
        time_threshold_unix=time_threshold_unix
    )

    vector_result_c = vector_query_c_count_cats_born_after_time(
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        time_threshold_year=time_threshold_year
    )

    summary_c = print_single_query_report(
        query_label="C",
        query_description=f"How many cats born after {time_threshold_year}?",
        tabular_count=tabular_count_c,
        vector_result=vector_result_c,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD
    )
    all_query_summaries.append(summary_c)

    # ----- Query D: Cats born after T that can fly? -----
    print(f"\n[RUNNING] Query D: How many cats born after {time_threshold_year} that can fly?")

    tabular_count_d = tabular_query_d_count_cats_born_after_time_can_fly(
        dataframe=dataframe,
        time_threshold_unix=time_threshold_unix
    )

    vector_result_d = vector_query_d_count_cats_born_after_time_can_fly(
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        time_threshold_year=time_threshold_year
    )

    summary_d = print_single_query_report(
        query_label="D",
        query_description=f"Cats born after {time_threshold_year} that can fly?",
        tabular_count=tabular_count_d,
        vector_result=vector_result_d,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD
    )
    all_query_summaries.append(summary_d)

    # ----- Query E: Stress test with 10+ fields -----
    print("\n[RUNNING] Query E: Stress test (cat + time + 10 fields)")

    # Print the Query E configuration for transparency
    print("  Query E Configuration:")
    print(f"    time_threshold_year: {time_threshold_year}")
    print(f"    weight_kg_min: {QUERY_E_WEIGHT_KG_MIN}")
    print(f"    height_cm_max: {QUERY_E_HEIGHT_CM_MAX}")
    print(f"    age_years_min: {QUERY_E_AGE_YEARS_MIN}")
    print(f"    number_of_friends_min: {QUERY_E_FRIENDS_MIN}")
    print(f"    color_value: {QUERY_E_COLOR_VALUE}")
    print(f"    daily_food_grams_min: {QUERY_E_DAILY_FOOD_GRAMS_MIN}")
    print("    can_fly: True")
    print("    can_swim: True")
    print("    can_run: True")
    print("    watches_youtube: True")

    tabular_count_e = tabular_query_e_count_cats_10_field_stress_test(
        dataframe=dataframe,
        time_threshold_unix=time_threshold_unix,
        weight_kg_min=QUERY_E_WEIGHT_KG_MIN,
        height_cm_max=QUERY_E_HEIGHT_CM_MAX,
        age_years_min=QUERY_E_AGE_YEARS_MIN,
        number_of_friends_min=QUERY_E_FRIENDS_MIN,
        color_value=QUERY_E_COLOR_VALUE,
        daily_food_grams_min=QUERY_E_DAILY_FOOD_GRAMS_MIN
    )

    vector_result_e = vector_query_e_count_cats_10_field_stress_test(
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        time_threshold_year=time_threshold_year,
        weight_kg_min=QUERY_E_WEIGHT_KG_MIN,
        height_cm_max=QUERY_E_HEIGHT_CM_MAX,
        age_years_min=QUERY_E_AGE_YEARS_MIN,
        number_of_friends_min=QUERY_E_FRIENDS_MIN,
        color_value=QUERY_E_COLOR_VALUE,
        daily_food_grams_min=QUERY_E_DAILY_FOOD_GRAMS_MIN
    )

    summary_e = print_single_query_report(
        query_label="E",
        query_description="Stress test: cat + time + 10 fields",
        tabular_count=tabular_count_e,
        vector_result=vector_result_e,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD
    )
    all_query_summaries.append(summary_e)

    # -------------------------------------------------------------------------
    # Step 6: Print summary table
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 6: SUMMARY")
    print(f"{'='*70}")

    print_summary_table(all_query_summaries)

    # -------------------------------------------------------------------------
    # Step 7: Print configuration used (for reproducibility)
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CONFIGURATION USED (for reproducibility)")
    print(f"{'='*70}")
    print(f"  CSV file: {csv_file_path}")
    print(f"  Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  Similarity threshold: {VECTOR_SIMILARITY_THRESHOLD}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Time threshold (Unix): {time_threshold_unix}")
    print(f"  Time threshold (Year): {time_threshold_year}")
    print(f"  Total documents: {len(dataframe)}")
    print(f"  Embedding dimensions: {corpus_embeddings.shape[1]}")

    return all_query_summaries


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Main entry point for MVP-1 test suite.

    Executes full test suite and handles top-level errors.
    """
    print("\n" + "=" * 90)
    print("STARTING MVP-1 VECTOR ANALYTICS TEST SUITE")
    print("=" * 90 + "\n")

    try:
        # Run the test suite
        query_summaries = run_mvp1_test_suite(CSV_FILE_PATH)

        print(f"query_summaries: {query_summaries}")

        print("\n" + "=" * 90)
        print("TEST SUITE COMPLETED SUCCESSFULLY")
        print("=" * 90)

        # Return exit code 0 for success
        return

    except FileNotFoundError as file_error:
        print(f"\n[FATAL ERROR] File not found: {file_error}")
        print("\nPlease ensure the CSV file exists at the specified path.")
        print(f"Expected path: {CSV_FILE_PATH}")
        traceback.print_exc()

    except ValueError as validation_error:
        print(f"\n[FATAL ERROR] Data validation failed: {validation_error}")
        traceback.print_exc()

    except Exception as unexpected_error:
        print(f"\n[FATAL ERROR] Unexpected error: {unexpected_error}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
