# comparative_vector_tests.py
"""
comparative_vector_tests.py

Comparative Vector Analytics Testing Framework

## Purpose:
    Compare TWO vector queries (e.g., cats vs dogs) and evaluate:
    1. Direction correctness: Did vector correctly identify which is larger?
    2. Ratio accuracy: How close is vector ratio to ground truth ratio?
    3. Per-query count accuracy

## Steps to Run:
    1. Run in python venv environment:
    ```bash
    python comparative_vector_tests.py
    ```

    2. You will be prompted for:
        - Path to CSV data file
        - Similarity threshold (default 0.5)

    3. Results saved to "tests/comparative_tests/comparative_{timestamp}/" directory

## Test Matrix:
    For each test A-D, run both "cat" and "dog" query variants,
    then compare the counts.

    Test A: How many cats vs how many dogs?
    Test B: How many flying cats vs flying dogs?
    Test C: How many cats vs dogs born after time T?
    Test D: How many flying cats vs flying dogs born after T?

## Output Files:
    - comparative_summary_{timestamp}.csv (one row per test)
    - comparative_detailed_{timestamp}.csv (full metrics)
    - A_comparative_results_{timestamp}.csv (per-row for test A)
    - B_comparative_results_{timestamp}.csv (per-row for test B)
    - C_comparative_results_{timestamp}.csv (per-row for test C)
    - D_comparative_results_{timestamp}.csv (per-row for test D)

## Key Metrics:
    - direction_correct: bool (did vector agree on which animal has more?)
    - tabular_ratio: float (tabular_count_cat / tabular_count_dog)
    - vector_ratio: float (vector_count_cat / vector_count_dog)
    - ratio_error: float (absolute difference in ratios)
    - ratio_percent_error: float (percent error in ratio)

## Design:
    This is a STANDALONE script. It does not import from other project files.
    All necessary utility functions are included in this file.

## Dependencies:
    - pandas
    - numpy
    - sentence-transformers

Author: Comparative Vector Test Framework
Date: 2024-2026
"""

import time
import traceback
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
import plotly.graph_objects as go
# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Path to CSV file - prompted at runtime
CSV_FILE_PATH: str = ""

# Sentence-transformer model (same default as ChromaDB)
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Similarity threshold - prompted at runtime
VECTOR_SIMILARITY_THRESHOLD: float = 0.5

# Random seed for reproducibility
RANDOM_SEED: int = 42


# =============================================================================
# TIMESTAMP AND OUTPUT DIRECTORY SETUP
# =============================================================================


def create_timestamped_output_directory() -> tuple[str, Path]:
    """
    Create a timestamped output directory for comparative test artifacts.

    Directory structure:
        tests/comparative_tests/comparative_{YYYYMMDD_HHMMSS}/

    Returns
    -------
    tuple[str, Path]
        - timestamp_string: e.g. "20260307_143022"
        - output_directory_path: Path object for the created directory

    Raises
    ------
    Exception
        If directory creation fails.

    Notes
    -----
    Directory is created immediately upon calling this function.
    The parent directory "tests/comparative_tests" is also created if needed.
    """
    try:
        now = datetime.datetime.now()
        timestamp_string = now.strftime("%Y%m%d_%H%M%S")

        output_directory_name = f"tests/comparative_tests/comparative_{timestamp_string}"
        output_directory_path = Path.cwd() / output_directory_name

        output_directory_path.mkdir(parents=True, exist_ok=True)

        print(f"[OUTPUT] Created output directory: {output_directory_path}")

        return timestamp_string, output_directory_path

    except Exception as directory_creation_error:
        print(f"[ERROR] Failed to create output directory: {directory_creation_error}")
        traceback.print_exc()
        raise


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

    Raises
    ------
    FileNotFoundError
        If CSV file does not exist at specified path.
    ValueError
        If required columns are missing from CSV.

    Notes
    -----
    Required columns for comparative tests:
        - animal_type: categorical (cat, dog, bird, fish, turtle)
        - birth_unix: int (Unix timestamp)
        - can_fly: boolean
        - can_swim: boolean
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

        # Define required columns for comparative tests
        required_columns = [
            "animal_type",
            "birth_unix",
            "can_fly",
            "can_swim",
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
        raise

    except ValueError:
        raise

    except Exception as unexpected_error:
        print(f"[ERROR] Unexpected error loading CSV: {unexpected_error}")
        traceback.print_exc()
        raise


# =============================================================================
# EMBEDDING MODEL FUNCTIONS
# =============================================================================


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load and return a sentence-transformer model.

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

    Parameters
    ----------
    text_documents : list[str]
        List of text strings (unstructured descriptions).

    embedding_model : SentenceTransformer
        Pre-loaded sentence-transformer model.

    Returns
    -------
    np.ndarray
        Matrix of shape (n_documents, embedding_dimension).
        For 'all-MiniLM-L6-v2', embedding_dimension is 384.

    Notes
    -----
    Progress bar is displayed during encoding.
    This may take significant time for large corpora.
    """
    try:
        document_count = len(text_documents)
        print(f"[EMBED] Generating embeddings for {document_count} documents...")

        start_time = time.time()

        embeddings_matrix = embedding_model.encode(
            text_documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        elapsed_seconds = time.time() - start_time

        print(f"[EMBED] Embeddings complete in {elapsed_seconds:.2f} seconds")
        print(f"[EMBED] Embedding matrix shape: {embeddings_matrix.shape}")

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
    Every document gets a similarity score.

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

    Notes
    -----
    Formula: cosine_similarity = (A · B) / (||A|| * ||B||)
    Handles zero-norm edge case with small epsilon.
    """
    # Ensure query is 1D for dot product
    query_vector_flat = query_vector.flatten()

    # Dot product: each document dotted with query
    dot_products = corpus_embeddings_matrix @ query_vector_flat

    # Calculate norms for normalization
    corpus_norms = norm(corpus_embeddings_matrix, axis=1)
    query_norm = norm(query_vector_flat)

    # Compute denominator, avoiding division by zero
    denominator = corpus_norms * query_norm

    epsilon = 1e-10
    denominator_safe = np.where(
        denominator == 0,
        epsilon,
        denominator
    )

    similarities = dot_products / denominator_safe

    return similarities


def count_documents_above_threshold(
    similarity_scores: np.ndarray,
    threshold: float
) -> int:
    """
    Count how many documents have similarity above threshold.

    Parameters
    ----------
    similarity_scores : np.ndarray
        Array of similarity scores for all documents.

    threshold : float
        Minimum similarity to count as a "match".

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

    Parameters
    ----------
    similarity_scores : np.ndarray
        Array of similarity scores for all documents.

    Returns
    -------
    dict
        Contains mean, std, min, max, median, q25, q75.
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
# TABULAR GROUND TRUTH FUNCTIONS
# =============================================================================


def tabular_count_animal(
    dataframe: pd.DataFrame,
    animal_type: str
) -> int:
    """
    Count rows where animal_type matches specified type.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pet data with animal_type column.

    animal_type : str
        Animal type to count (e.g., "cat", "dog").

    Returns
    -------
    int
        Exact count of matching rows.
    """
    boolean_mask = dataframe["animal_type"] == animal_type
    count = int(boolean_mask.sum())
    return count


def tabular_count_animal_can_fly(
    dataframe: pd.DataFrame,
    animal_type: str
) -> int:
    """
    Count rows where animal_type matches AND can_fly is True.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pet data with animal_type and can_fly columns.

    animal_type : str
        Animal type to count.

    Returns
    -------
    int
        Exact count of matching rows.
    """
    boolean_mask = (
        (dataframe["animal_type"] == animal_type) &
        (dataframe["can_fly"] == True)
    )
    count = int(boolean_mask.sum())
    return count


def tabular_count_animal_born_after(
    dataframe: pd.DataFrame,
    animal_type: str,
    time_threshold_unix: int
) -> int:
    """
    Count rows where animal_type matches AND birth_unix > threshold.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pet data with animal_type and birth_unix columns.

    animal_type : str
        Animal type to count.

    time_threshold_unix : int
        Unix timestamp threshold.

    Returns
    -------
    int
        Exact count of matching rows.
    """
    boolean_mask = (
        (dataframe["animal_type"] == animal_type) &
        (dataframe["birth_unix"] > time_threshold_unix)
    )
    count = int(boolean_mask.sum())
    return count


def tabular_count_animal_born_after_can_fly(
    dataframe: pd.DataFrame,
    animal_type: str,
    time_threshold_unix: int
) -> int:
    """
    Count rows where animal_type matches AND birth_unix > threshold AND can_fly.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pet data with animal_type, birth_unix, and can_fly columns.

    animal_type : str
        Animal type to count.

    time_threshold_unix : int
        Unix timestamp threshold.

    Returns
    -------
    int
        Exact count of matching rows.
    """
    boolean_mask = (
        (dataframe["animal_type"] == animal_type) &
        (dataframe["birth_unix"] > time_threshold_unix) &
        (dataframe["can_fly"] == True)
    )
    count = int(boolean_mask.sum())
    return count


# =============================================================================
# GROUND TRUTH BOOLEAN ARRAY GENERATORS
#
# These produce per-row boolean arrays for confusion matrix evaluation.
# =============================================================================


def generate_ground_truth_animal(
    dataframe: pd.DataFrame,
    animal_type: str
) -> np.ndarray:
    """
    Generate per-row ground truth: Is this row the specified animal?

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    animal_type : str
        Animal type to match.

    Returns
    -------
    np.ndarray
        Boolean array, shape (n_rows,).
    """
    ground_truth_mask = (dataframe["animal_type"] == animal_type).values
    return ground_truth_mask


def generate_ground_truth_animal_can_fly(
    dataframe: pd.DataFrame,
    animal_type: str
) -> np.ndarray:
    """
    Generate per-row ground truth: Is this the animal AND can fly?

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    animal_type : str
        Animal type to match.

    Returns
    -------
    np.ndarray
        Boolean array, shape (n_rows,).
    """
    ground_truth_mask = (
        (dataframe["animal_type"] == animal_type) &
        (dataframe["can_fly"] == True)
    ).values
    return ground_truth_mask


def generate_ground_truth_animal_born_after(
    dataframe: pd.DataFrame,
    animal_type: str,
    time_threshold_unix: int
) -> np.ndarray:
    """
    Generate per-row ground truth: Is this the animal born after T?

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    animal_type : str
        Animal type to match.

    time_threshold_unix : int
        Unix timestamp threshold.

    Returns
    -------
    np.ndarray
        Boolean array, shape (n_rows,).
    """
    ground_truth_mask = (
        (dataframe["animal_type"] == animal_type) &
        (dataframe["birth_unix"] > time_threshold_unix)
    ).values
    return ground_truth_mask


def generate_ground_truth_animal_born_after_can_fly(
    dataframe: pd.DataFrame,
    animal_type: str,
    time_threshold_unix: int
) -> np.ndarray:
    """
    Generate per-row ground truth: Animal born after T that can fly?

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    animal_type : str
        Animal type to match.

    time_threshold_unix : int
        Unix timestamp threshold.

    Returns
    -------
    np.ndarray
        Boolean array, shape (n_rows,).
    """
    ground_truth_mask = (
        (dataframe["animal_type"] == animal_type) &
        (dataframe["birth_unix"] > time_threshold_unix) &
        (dataframe["can_fly"] == True)
    ).values
    return ground_truth_mask


# =============================================================================
# VECTOR QUERY FUNCTIONS
# =============================================================================


def vector_query_animal(
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    animal_type: str
) -> dict:
    """
    Vector query: How many documents about specified animal?

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    animal_type : str
        Animal type to query for ("cat" or "dog").

    Returns
    -------
    dict
        Contains count, similarities, query_text, elapsed_seconds.
    """
    query_text = f"This is about a {animal_type}."

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


def vector_query_animal_can_fly(
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    animal_type: str
) -> dict:
    """
    Vector query: How many documents about specified animal that can fly?

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    animal_type : str
        Animal type to query for ("cat" or "dog").

    Returns
    -------
    dict
        Contains count, similarities, query_text, elapsed_seconds.
    """
    if animal_type == "cat":
        species_word = "feline"
    elif animal_type == "dog":
        species_word = "canine"
    else:
        species_word = animal_type

    query_text = f"A {animal_type}, {species_word}, that can fly. A flying {animal_type}."

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


def vector_query_animal_born_after(
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    animal_type: str,
    time_threshold_year: int
) -> dict:
    """
    Vector query: How many documents about animal born after year T?

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    animal_type : str
        Animal type to query for.

    time_threshold_year : int
        Year threshold for birth.

    Returns
    -------
    dict
        Contains count, similarities, query_text, elapsed_seconds.
    """
    if animal_type == "cat":
        species_word = "feline"
    elif animal_type == "dog":
        species_word = "canine"
    else:
        species_word = animal_type

    query_text = (
        f"A {animal_type}, {species_word}, born after the year {time_threshold_year}. "
        f"A recently born {animal_type}."
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


def vector_query_animal_born_after_can_fly(
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    animal_type: str,
    time_threshold_year: int
) -> dict:
    """
    Vector query: Animal born after year T that can fly?

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    animal_type : str
        Animal type to query for.

    time_threshold_year : int
        Year threshold for birth.

    Returns
    -------
    dict
        Contains count, similarities, query_text, elapsed_seconds.
    """
    if animal_type == "cat":
        species_word = "feline"
    elif animal_type == "dog":
        species_word = "canine"
    else:
        species_word = animal_type

    query_text = (
        f"A {animal_type}, {species_word}, born after the year {time_threshold_year}, "
        f"that can fly. A flying {animal_type} born recently."
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
# CONFUSION MATRIX FUNCTIONS
# =============================================================================


def classify_rows_into_confusion_categories(
    ground_truth_boolean_array: np.ndarray,
    vector_predicted_boolean_array: np.ndarray
) -> np.ndarray:
    """
    Classify each row as TP, FP, FN, or TN.

    Definitions:
        TP (True Positive):  ground_truth=True  AND predicted=True
        FP (False Positive): ground_truth=False AND predicted=True
        FN (False Negative): ground_truth=True  AND predicted=False
        TN (True Negative):  ground_truth=False AND predicted=False

    Parameters
    ----------
    ground_truth_boolean_array : np.ndarray
        Boolean array from tabular ground truth. Shape (n_rows,).

    vector_predicted_boolean_array : np.ndarray
        Boolean array from vector similarity > threshold. Shape (n_rows,).

    Returns
    -------
    np.ndarray
        String array of shape (n_rows,) with values "TP", "FP", "FN", "TN".
    """
    if len(ground_truth_boolean_array) != len(vector_predicted_boolean_array):
        raise ValueError(
            f"Array length mismatch: ground_truth has {len(ground_truth_boolean_array)} "
            f"rows, predicted has {len(vector_predicted_boolean_array)} rows."
        )

    row_count = len(ground_truth_boolean_array)
    confusion_categories = np.empty(row_count, dtype="U2")

    tp_mask = ground_truth_boolean_array & vector_predicted_boolean_array
    confusion_categories[tp_mask] = "TP"

    fp_mask = (~ground_truth_boolean_array) & vector_predicted_boolean_array
    confusion_categories[fp_mask] = "FP"

    fn_mask = ground_truth_boolean_array & (~vector_predicted_boolean_array)
    confusion_categories[fn_mask] = "FN"

    tn_mask = (~ground_truth_boolean_array) & (~vector_predicted_boolean_array)
    confusion_categories[tn_mask] = "TN"

    return confusion_categories


def compute_confusion_matrix_counts(
    confusion_categories: np.ndarray
) -> dict:
    """
    Count TP, FP, FN, TN from per-row classification array.

    Parameters
    ----------
    confusion_categories : np.ndarray
        String array with values "TP", "FP", "FN", "TN".

    Returns
    -------
    dict
        Contains tp_count, fp_count, fn_count, tn_count, total_count.
    """
    tp_count = int(np.sum(confusion_categories == "TP"))
    fp_count = int(np.sum(confusion_categories == "FP"))
    fn_count = int(np.sum(confusion_categories == "FN"))
    tn_count = int(np.sum(confusion_categories == "TN"))
    total_count = tp_count + fp_count + fn_count + tn_count

    return {
        "tp_count": tp_count,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "tn_count": tn_count,
        "total_count": total_count,
    }


def compute_confusion_matrix_metrics(
    confusion_counts: dict
) -> dict:
    """
    Compute precision, recall, F1, accuracy from confusion counts.

    Parameters
    ----------
    confusion_counts : dict
        Contains tp_count, fp_count, fn_count, tn_count.

    Returns
    -------
    dict
        Contains precision, recall, f1_score, accuracy.
        Values are None if denominator is zero.
    """
    tp = confusion_counts["tp_count"]
    fp = confusion_counts["fp_count"]
    fn = confusion_counts["fn_count"]
    tn = confusion_counts["tn_count"]
    total = confusion_counts["total_count"]

    precision_denominator = tp + fp
    if precision_denominator > 0:
        precision = tp / precision_denominator
    else:
        precision = None

    recall_denominator = tp + fn
    if recall_denominator > 0:
        recall = tp / recall_denominator
    else:
        recall = None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1_score = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score = None

    if total > 0:
        accuracy = (tp + tn) / total
    else:
        accuracy = None

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


# =============================================================================
# COMPARATIVE EVALUATION FUNCTIONS
# =============================================================================


def evaluate_comparative_counts(
    tabular_count_cat: int,
    tabular_count_dog: int,
    vector_count_cat: int,
    vector_count_dog: int
) -> dict:
    """
    Evaluate comparative query: Did vector correctly identify which is larger?

    This is the PRIMARY metric for comparative tests.

    Parameters
    ----------
    tabular_count_cat : int
        Ground truth count for cats.

    tabular_count_dog : int
        Ground truth count for dogs.

    vector_count_cat : int
        Vector query count for cats.

    vector_count_dog : int
        Vector query count for dogs.

    Returns
    -------
    dict
        Contains:
        - tabular_direction: str ("cat_higher", "dog_higher", "equal")
        - vector_direction: str ("cat_higher", "dog_higher", "equal")
        - direction_correct: bool
        - tabular_ratio: float (cat/dog ratio, inf if dog=0)
        - vector_ratio: float (cat/dog ratio, inf if dog=0)
        - ratio_error: float or None
        - ratio_percent_error: float or None
        - tabular_difference: int (cat - dog)
        - vector_difference: int (cat - dog)

    Notes
    -----
    Direction correctness is the key metric. A vector system that
    correctly identifies "more cats than dogs" (or vice versa) is
    useful for comparative analytics even if absolute counts are off.
    """
    # Determine ground truth direction
    if tabular_count_cat > tabular_count_dog:
        tabular_direction = "cat_higher"
    elif tabular_count_dog > tabular_count_cat:
        tabular_direction = "dog_higher"
    else:
        tabular_direction = "equal"

    # Determine vector direction
    if vector_count_cat > vector_count_dog:
        vector_direction = "cat_higher"
    elif vector_count_dog > vector_count_cat:
        vector_direction = "dog_higher"
    else:
        vector_direction = "equal"

    # Direction correctness
    direction_correct = (tabular_direction == vector_direction)

    # Compute ratios (cat / dog)
    if tabular_count_dog > 0:
        tabular_ratio = tabular_count_cat / tabular_count_dog
    else:
        tabular_ratio = float('inf') if tabular_count_cat > 0 else 1.0

    if vector_count_dog > 0:
        vector_ratio = vector_count_cat / vector_count_dog
    else:
        vector_ratio = float('inf') if vector_count_cat > 0 else 1.0

    # Ratio error (only if both finite)
    if tabular_ratio != float('inf') and vector_ratio != float('inf'):
        ratio_error = abs(vector_ratio - tabular_ratio)
        if tabular_ratio > 0:
            ratio_percent_error = 100.0 * ratio_error / tabular_ratio
        else:
            ratio_percent_error = None
    else:
        ratio_error = None
        ratio_percent_error = None

    # Raw differences
    tabular_difference = tabular_count_cat - tabular_count_dog
    vector_difference = vector_count_cat - vector_count_dog

    return {
        "tabular_direction": tabular_direction,
        "vector_direction": vector_direction,
        "direction_correct": direction_correct,
        "tabular_ratio": tabular_ratio,
        "vector_ratio": vector_ratio,
        "ratio_error": ratio_error,
        "ratio_percent_error": ratio_percent_error,
        "tabular_difference": tabular_difference,
        "vector_difference": vector_difference,
        "tabular_count_cat": tabular_count_cat,
        "tabular_count_dog": tabular_count_dog,
        "vector_count_cat": vector_count_cat,
        "vector_count_dog": vector_count_dog,
    }


# =============================================================================
# PER-ROW RESULTS DATAFRAME CONSTRUCTION
# =============================================================================


def build_per_row_results_dataframe(
    dataframe: pd.DataFrame,
    similarities_cat: np.ndarray,
    similarities_dog: np.ndarray,
    similarity_threshold: float,
    ground_truth_cat: np.ndarray,
    ground_truth_dog: np.ndarray
) -> pd.DataFrame:
    """
    Build per-row DataFrame with both cat and dog similarity scores
    and confusion classifications.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Original dataset (for unique_id if present).

    similarities_cat : np.ndarray
        Cat query similarity scores for all rows.

    similarities_dog : np.ndarray
        Dog query similarity scores for all rows.

    similarity_threshold : float
        Threshold used for match prediction.

    ground_truth_cat : np.ndarray
        Boolean ground truth for cat query.

    ground_truth_dog : np.ndarray
        Boolean ground truth for dog query.

    Returns
    -------
    pd.DataFrame
        Per-row results with columns for both cat and dog queries.
    """
    # Vector predictions
    vector_predicted_cat = similarities_cat > similarity_threshold
    vector_predicted_dog = similarities_dog > similarity_threshold

    # Confusion classifications
    confusion_cat = classify_rows_into_confusion_categories(
        ground_truth_cat, vector_predicted_cat
    )
    confusion_dog = classify_rows_into_confusion_categories(
        ground_truth_dog, vector_predicted_dog
    )

    # Build DataFrame
    result_df = pd.DataFrame({
        "row_index": np.arange(len(dataframe)),
        "similarity_cat": similarities_cat,
        "similarity_dog": similarities_dog,
        "similarity_threshold": similarity_threshold,
        "vector_predicted_cat": vector_predicted_cat,
        "vector_predicted_dog": vector_predicted_dog,
        "ground_truth_cat": ground_truth_cat,
        "ground_truth_dog": ground_truth_dog,
        "confusion_class_cat": confusion_cat,
        "confusion_class_dog": confusion_dog,
    })

    # Add unique_id if present in original dataframe
    if "unique_id" in dataframe.columns:
        result_df.insert(0, "unique_id", dataframe["unique_id"].values)

    return result_df


# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================


def format_metric_for_display(
    metric_value: float | None,
    format_string: str = ".4f"
) -> str:
    """
    Format a metric value for printing, handling None and inf.

    Parameters
    ----------
    metric_value : float or None
        The metric to format.

    format_string : str
        Python format specifier.

    Returns
    -------
    str
        Formatted string, "N/A" if None, "inf" if infinity.
    """
    if metric_value is None:
        return "N/A"
    if metric_value == float('inf'):
        return "inf"
    return f"{metric_value:{format_string}}"


def print_comparative_test_report(
    test_label: str,
    test_description: str,
    comparative_result: dict,
    cat_confusion_counts: dict,
    dog_confusion_counts: dict,
    cat_confusion_metrics: dict,
    dog_confusion_metrics: dict,
    cat_similarity_stats: dict,
    dog_similarity_stats: dict,
    cat_query_text: str,
    dog_query_text: str,
    elapsed_cat: float,
    elapsed_dog: float
) -> None:
    """
    Print detailed report for a single comparative test.

    Parameters
    ----------
    test_label : str
        Short label (e.g., "A", "B").

    test_description : str
        Human-readable description of test.

    comparative_result : dict
        Output of evaluate_comparative_counts.

    cat_confusion_counts : dict
        TP/FP/FN/TN counts for cat query.

    dog_confusion_counts : dict
        TP/FP/FN/TN counts for dog query.

    cat_confusion_metrics : dict
        Precision/recall/F1/accuracy for cat query.

    dog_confusion_metrics : dict
        Precision/recall/F1/accuracy for dog query.

    cat_similarity_stats : dict
        Similarity distribution stats for cat query.

    dog_similarity_stats : dict
        Similarity distribution stats for dog query.

    cat_query_text : str
        Query text used for cat.

    dog_query_text : str
        Query text used for dog.

    elapsed_cat : float
        Query time for cat.

    elapsed_dog : float
        Query time for dog.
    """
    print(f"\n{'='*80}")
    print(f"COMPARATIVE TEST {test_label}: {test_description}")
    print(f"{'='*80}")

    # Direction result (key metric)
    direction_symbol = "✓" if comparative_result["direction_correct"] else "✗"
    print(f"\n  DIRECTION RESULT: {direction_symbol} {'CORRECT' if comparative_result['direction_correct'] else 'INCORRECT'}")
    print(f"    Ground truth direction: {comparative_result['tabular_direction']}")
    print(f"    Vector direction:       {comparative_result['vector_direction']}")

    # Counts comparison
    print("\n  COUNTS:")
    print(f"    {'':20} {'Tabular':>10} {'Vector':>10} {'Error':>10}")
    print(f"    {'-'*50}")
    cat_error = comparative_result["vector_count_cat"] - comparative_result["tabular_count_cat"]
    dog_error = comparative_result["vector_count_dog"] - comparative_result["tabular_count_dog"]
    print(f"    {'Cats:':<20} {comparative_result['tabular_count_cat']:>10} {comparative_result['vector_count_cat']:>10} {cat_error:>+10}")
    print(f"    {'Dogs:':<20} {comparative_result['tabular_count_dog']:>10} {comparative_result['vector_count_dog']:>10} {dog_error:>+10}")
    print(f"    {'-'*50}")
    print(f"    {'Difference (C-D):':<20} {comparative_result['tabular_difference']:>10} {comparative_result['vector_difference']:>10}")

    # Ratio comparison
    print("\n  RATIO (Cat/Dog):")
    print(f"    Tabular ratio:       {format_metric_for_display(comparative_result['tabular_ratio'], '.4f')}")
    print(f"    Vector ratio:        {format_metric_for_display(comparative_result['vector_ratio'], '.4f')}")
    if comparative_result['ratio_error'] is not None:
        print(f"    Ratio error:         {comparative_result['ratio_error']:.4f}")
    if comparative_result['ratio_percent_error'] is not None:
        print(f"    Ratio percent error: {comparative_result['ratio_percent_error']:.2f}%")

    # Query texts
    print("\n  QUERY TEXTS:")
    print(f"    Cat: \"{cat_query_text}\"")
    print(f"    Dog: \"{dog_query_text}\"")

    # Per-animal confusion matrix
    print("\n  CONFUSION MATRIX - CATS:")
    print(f"    TP: {cat_confusion_counts['tp_count']:>6}  FP: {cat_confusion_counts['fp_count']:>6}  "
          f"FN: {cat_confusion_counts['fn_count']:>6}  TN: {cat_confusion_counts['tn_count']:>6}")
    print(f"    Precision: {format_metric_for_display(cat_confusion_metrics['precision'])}  "
          f"Recall: {format_metric_for_display(cat_confusion_metrics['recall'])}  "
          f"F1: {format_metric_for_display(cat_confusion_metrics['f1_score'])}")

    print("\n  CONFUSION MATRIX - DOGS:")
    print(f"    TP: {dog_confusion_counts['tp_count']:>6}  FP: {dog_confusion_counts['fp_count']:>6}  "
          f"FN: {dog_confusion_counts['fn_count']:>6}  TN: {dog_confusion_counts['tn_count']:>6}")
    print(f"    Precision: {format_metric_for_display(dog_confusion_metrics['precision'])}  "
          f"Recall: {format_metric_for_display(dog_confusion_metrics['recall'])}  "
          f"F1: {format_metric_for_display(dog_confusion_metrics['f1_score'])}")

    # Similarity distributions
    print("\n  SIMILARITY DISTRIBUTIONS:")
    print(f"    {'':8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
    print(f"    {'-'*48}")
    print(f"    {'Cats:':<8} {cat_similarity_stats['mean']:>8.4f} {cat_similarity_stats['std']:>8.4f} "
          f"{cat_similarity_stats['min']:>8.4f} {cat_similarity_stats['max']:>8.4f} {cat_similarity_stats['median']:>8.4f}")
    print(f"    {'Dogs:':<8} {dog_similarity_stats['mean']:>8.4f} {dog_similarity_stats['std']:>8.4f} "
          f"{dog_similarity_stats['min']:>8.4f} {dog_similarity_stats['max']:>8.4f} {dog_similarity_stats['median']:>8.4f}")

    # Timing
    print(f"\n  TIMING: Cat {elapsed_cat:.4f}s, Dog {elapsed_dog:.4f}s")


def print_comparative_summary_table(all_results: list[dict]) -> None:
    """
    Print summary table of all comparative test results.

    Parameters
    ----------
    all_results : list[dict]
        List of result dictionaries from all tests.
    """
    print(f"\n{'='*100}")
    print("COMPARATIVE TESTS SUMMARY: CAT vs DOG")
    print(f"{'='*100}")

    # Header
    print(
        f"{'Test':<6} "
        f"{'Description':<35} "
        f"{'Tab_C':>7} "
        f"{'Tab_D':>7} "
        f"{'Vec_C':>7} "
        f"{'Vec_D':>7} "
        f"{'Dir':>6} "
        f"{'T_Ratio':>8} "
        f"{'V_Ratio':>8} "
        f"{'R_Err%':>8}"
    )
    print("-" * 100)

    # Rows
    for r in all_results:
        dir_symbol = "✓" if r["direction_correct"] else "✗"

        tabular_ratio_str = format_metric_for_display(r["tabular_ratio"], ".3f")
        vector_ratio_str = format_metric_for_display(r["vector_ratio"], ".3f")

        if r["ratio_percent_error"] is not None:
            ratio_err_str = f"{r['ratio_percent_error']:.1f}%"
        else:
            ratio_err_str = "N/A"

        print(
            f"{r['test_label']:<6} "
            f"{r['test_description']:<35} "
            f"{r['tabular_count_cat']:>7} "
            f"{r['tabular_count_dog']:>7} "
            f"{r['vector_count_cat']:>7} "
            f"{r['vector_count_dog']:>7} "
            f"{dir_symbol:>6} "
            f"{tabular_ratio_str:>8} "
            f"{vector_ratio_str:>8} "
            f"{ratio_err_str:>8}"
        )

    print("-" * 100)

    # Aggregate direction accuracy
    direction_correct_count = sum(1 for r in all_results if r["direction_correct"])
    total_tests = len(all_results)
    direction_accuracy = 100.0 * direction_correct_count / total_tests if total_tests > 0 else 0

    print(f"\nDIRECTION ACCURACY: {direction_correct_count}/{total_tests} ({direction_accuracy:.1f}%)")


# =============================================================================
# CSV SAVING FUNCTIONS
# =============================================================================


def save_comparative_summary_csv(
    all_results: list[dict],
    output_directory: Path,
    timestamp_string: str
) -> Path:
    """
    Save comparative summary to CSV.

    Parameters
    ----------
    all_results : list[dict]
        List of result dictionaries.

    output_directory : Path
        Directory to save into.

    timestamp_string : str
        Timestamp for filename.

    Returns
    -------
    Path
        Path to saved CSV file.
    """
    summary_df = pd.DataFrame(all_results)

    filename = f"comparative_summary_{timestamp_string}.csv"
    filepath = output_directory / filename

    summary_df.to_csv(filepath, index=False)
    print(f"[SAVE] Comparative summary: {filepath}")

    return filepath


def save_per_row_results_csv(
    per_row_df: pd.DataFrame,
    test_label: str,
    output_directory: Path,
    timestamp_string: str
) -> Path:
    """
    Save per-row comparative results to CSV.

    Parameters
    ----------
    per_row_df : pd.DataFrame
        Per-row results DataFrame.

    test_label : str
        Test identifier (e.g., "A").

    output_directory : Path
        Directory to save into.

    timestamp_string : str
        Timestamp for filename.

    Returns
    -------
    Path
        Path to saved CSV file.
    """
    filename = f"{test_label}_comparative_results_{timestamp_string}.csv"
    filepath = output_directory / filename

    per_row_df.to_csv(filepath, index=False)
    print(f"[SAVE] Test {test_label} per-row results: {filepath}")

    return filepath


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================


def run_comparative_test(
    test_label: str,
    test_description: str,
    dataframe: pd.DataFrame,
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    tabular_count_func_cat,
    tabular_count_func_dog,
    ground_truth_func_cat,
    ground_truth_func_dog,
    vector_query_func_cat,
    vector_query_func_dog,
    output_directory: Path,
    timestamp_string: str,
    pca_model: PCA | None = None,
) -> tuple[dict, PCA | None]:
    """
    Run a single comparative test (cat vs dog) and return results.

    Parameters
    ----------
    test_label : str
        Short label (e.g., "A").

    test_description : str
        Human-readable description.

    dataframe : pd.DataFrame
        Full dataset.

    corpus_embeddings : np.ndarray
        Pre-computed embeddings.

    embedding_model : SentenceTransformer
        Model for query embedding.

    similarity_threshold : float
        Match threshold.

    tabular_count_func_cat : callable
        Function returning tabular count for cats.

    tabular_count_func_dog : callable
        Function returning tabular count for dogs.

    ground_truth_func_cat : callable
        Function returning ground truth boolean array for cats.

    ground_truth_func_dog : callable
        Function returning ground truth boolean array for dogs.

    vector_query_func_cat : callable
        Function returning vector query result dict for cats.

    vector_query_func_dog : callable
        Function returning vector query result dict for dogs.

    output_directory : Path
        Directory for output files.

    timestamp_string : str
        Timestamp for filenames.

    Returns
    -------
    dict
        Complete result dictionary for this test.
    """
    # --- Tabular ground truth counts ---
    tabular_count_cat = tabular_count_func_cat()
    tabular_count_dog = tabular_count_func_dog()

    # --- Ground truth boolean arrays ---
    ground_truth_cat = ground_truth_func_cat()
    ground_truth_dog = ground_truth_func_dog()

    # --- Vector queries ---
    vector_result_cat = vector_query_func_cat()
    vector_result_dog = vector_query_func_dog()

    vector_count_cat = vector_result_cat["count"]
    vector_count_dog = vector_result_dog["count"]
    similarities_cat = vector_result_cat["similarities"]
    similarities_dog = vector_result_dog["similarities"]

# --- 3D PCA Visualization ---
    pca_model = generate_comparative_pca_plot(
        corpus_embeddings=corpus_embeddings,
        similarities_cat=similarities_cat,
        similarities_dog=similarities_dog,
        similarity_threshold=similarity_threshold,
        cat_query_text=vector_result_cat["query_text"],
        dog_query_text=vector_result_dog["query_text"],
        embedding_model=embedding_model,
        test_label=test_label,
        output_directory=output_directory,
        timestamp_string=timestamp_string,
        pca_model=pca_model,
    )

    # --- Comparative evaluation ---
    comparative_result = evaluate_comparative_counts(
        tabular_count_cat=tabular_count_cat,
        tabular_count_dog=tabular_count_dog,
        vector_count_cat=vector_count_cat,
        vector_count_dog=vector_count_dog
    )

    # --- Per-animal confusion matrices ---
    vector_predicted_cat = similarities_cat > similarity_threshold
    vector_predicted_dog = similarities_dog > similarity_threshold

    confusion_cat = classify_rows_into_confusion_categories(ground_truth_cat, vector_predicted_cat)
    confusion_dog = classify_rows_into_confusion_categories(ground_truth_dog, vector_predicted_dog)

    cat_confusion_counts = compute_confusion_matrix_counts(confusion_cat)
    dog_confusion_counts = compute_confusion_matrix_counts(confusion_dog)

    cat_confusion_metrics = compute_confusion_matrix_metrics(cat_confusion_counts)
    dog_confusion_metrics = compute_confusion_matrix_metrics(dog_confusion_counts)

    # --- Similarity statistics ---
    cat_similarity_stats = compute_similarity_statistics(similarities_cat)
    dog_similarity_stats = compute_similarity_statistics(similarities_dog)

    # --- Print report ---
    print_comparative_test_report(
        test_label=test_label,
        test_description=test_description,
        comparative_result=comparative_result,
        cat_confusion_counts=cat_confusion_counts,
        dog_confusion_counts=dog_confusion_counts,
        cat_confusion_metrics=cat_confusion_metrics,
        dog_confusion_metrics=dog_confusion_metrics,
        cat_similarity_stats=cat_similarity_stats,
        dog_similarity_stats=dog_similarity_stats,
        cat_query_text=vector_result_cat["query_text"],
        dog_query_text=vector_result_dog["query_text"],
        elapsed_cat=vector_result_cat["elapsed_seconds"],
        elapsed_dog=vector_result_dog["elapsed_seconds"]
    )

    # --- Build and save per-row results ---
    per_row_df = build_per_row_results_dataframe(
        dataframe=dataframe,
        similarities_cat=similarities_cat,
        similarities_dog=similarities_dog,
        similarity_threshold=similarity_threshold,
        ground_truth_cat=ground_truth_cat,
        ground_truth_dog=ground_truth_dog
    )

    save_per_row_results_csv(
        per_row_df=per_row_df,
        test_label=test_label,
        output_directory=output_directory,
        timestamp_string=timestamp_string
    )

    # --- Build result dictionary ---
    result = {
        "test_label": test_label,
        "test_description": test_description,
        "tabular_count_cat": tabular_count_cat,
        "tabular_count_dog": tabular_count_dog,
        "vector_count_cat": vector_count_cat,
        "vector_count_dog": vector_count_dog,
        "tabular_direction": comparative_result["tabular_direction"],
        "vector_direction": comparative_result["vector_direction"],
        "direction_correct": comparative_result["direction_correct"],
        "tabular_ratio": comparative_result["tabular_ratio"],
        "vector_ratio": comparative_result["vector_ratio"],
        "ratio_error": comparative_result["ratio_error"],
        "ratio_percent_error": comparative_result["ratio_percent_error"],
        "tabular_difference": comparative_result["tabular_difference"],
        "vector_difference": comparative_result["vector_difference"],
        "cat_tp": cat_confusion_counts["tp_count"],
        "cat_fp": cat_confusion_counts["fp_count"],
        "cat_fn": cat_confusion_counts["fn_count"],
        "cat_tn": cat_confusion_counts["tn_count"],
        "cat_precision": cat_confusion_metrics["precision"],
        "cat_recall": cat_confusion_metrics["recall"],
        "cat_f1": cat_confusion_metrics["f1_score"],
        "dog_tp": dog_confusion_counts["tp_count"],
        "dog_fp": dog_confusion_counts["fp_count"],
        "dog_fn": dog_confusion_counts["fn_count"],
        "dog_tn": dog_confusion_counts["tn_count"],
        "dog_precision": dog_confusion_metrics["precision"],
        "dog_recall": dog_confusion_metrics["recall"],
        "dog_f1": dog_confusion_metrics["f1_score"],
        "cat_sim_mean": cat_similarity_stats["mean"],
        "cat_sim_max": cat_similarity_stats["max"],
        "dog_sim_mean": dog_similarity_stats["mean"],
        "dog_sim_max": dog_similarity_stats["max"],
        "cat_query_text": vector_result_cat["query_text"],
        "dog_query_text": vector_result_dog["query_text"],
        "cat_elapsed_seconds": vector_result_cat["elapsed_seconds"],
        "dog_elapsed_seconds": vector_result_dog["elapsed_seconds"],
    }

    return result, pca_model


def run_comparative_test_suite(
    csv_file_path: str,
    similarity_threshold: float
) -> list[dict]:
    """
    Execute complete comparative test suite: Tests A through D.

    Workflow:
        1. Create timestamped output directory
        2. Load CSV data
        3. Load embedding model
        4. Generate embeddings for all documents
        5. Determine time threshold (median of birth_unix)
        6. Run comparative tests A-D
        7. Print summary table
        8. Save summary CSV

    Parameters
    ----------
    csv_file_path : str
        Path to CSV file with pet data.

    similarity_threshold : float
        Threshold for vector matching.

    Returns
    -------
    list[dict]
        List of result dictionaries, one per test.
    """
    print("=" * 90)
    print("COMPARATIVE VECTOR ANALYTICS TEST SUITE")
    print("Comparing Cat vs Dog Query Results")
    print("=" * 90)

    # Set random seed
    np.random.seed(RANDOM_SEED)
    print(f"\n[CONFIG] Random seed: {RANDOM_SEED}")
    print(f"[CONFIG] Similarity threshold: {similarity_threshold}")
    print(f"[CONFIG] Embedding model: {EMBEDDING_MODEL_NAME}")

    # -------------------------------------------------------------------------
    # Step 1: Create output directory
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 1: CREATE OUTPUT DIRECTORY")
    print(f"{'='*70}")

    timestamp_string, output_directory = create_timestamped_output_directory()

    # -------------------------------------------------------------------------
    # Step 2: Load data
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 2: LOAD DATA")
    print(f"{'='*70}")

    dataframe = load_csv_data(csv_file_path)

    # -------------------------------------------------------------------------
    # Step 3: Load embedding model
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 3: LOAD EMBEDDING MODEL")
    print(f"{'='*70}")

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

    # -------------------------------------------------------------------------
    # Step 4: Generate embeddings
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 4: GENERATE CORPUS EMBEDDINGS")
    print(f"{'='*70}")

    description_texts = dataframe["unstructured_description"].tolist()
    corpus_embeddings = generate_corpus_embeddings(description_texts, embedding_model)

    # PCA model fitted once, reused for all test visualizations
    pca_model = None

    # -------------------------------------------------------------------------
    # Step 5: Configure time threshold
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 5: CONFIGURE TIME THRESHOLD")
    print(f"{'='*70}")

    time_threshold_unix = int(dataframe["birth_unix"].median())  # type: ignore[arg-type]
    time_threshold_datetime = datetime.datetime.fromtimestamp(time_threshold_unix)
    time_threshold_year = time_threshold_datetime.year

    print(f"[TIME] Median birth_unix: {time_threshold_unix}")
    print(f"[TIME] Corresponds to year: {time_threshold_year}")

    # -------------------------------------------------------------------------
    # Step 6: Run comparative tests A-D
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 6: RUN COMPARATIVE TESTS A-D")
    print(f"{'='*70}")

    all_results = []

    # --- Test A: How many cats vs how many dogs? ---
    print("\n[RUNNING] Test A: How many cats vs how many dogs?")

    result_a, pca_model = run_comparative_test(
        test_label="A",
        test_description="Count: cats vs dogs",
        dataframe=dataframe,
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        tabular_count_func_cat=lambda: tabular_count_animal(dataframe, "cat"),
        tabular_count_func_dog=lambda: tabular_count_animal(dataframe, "dog"),
        ground_truth_func_cat=lambda: generate_ground_truth_animal(dataframe, "cat"),
        ground_truth_func_dog=lambda: generate_ground_truth_animal(dataframe, "dog"),
        vector_query_func_cat=lambda: vector_query_animal(
            corpus_embeddings, embedding_model, similarity_threshold, "cat"
        ),
        vector_query_func_dog=lambda: vector_query_animal(
            corpus_embeddings, embedding_model, similarity_threshold, "dog"
        ),
        output_directory=output_directory,
        timestamp_string=timestamp_string,
        pca_model=pca_model,
    )
    all_results.append(result_a)

    # --- Test B: How many flying cats vs flying dogs? ---
    print("\n[RUNNING] Test B: How many flying cats vs flying dogs?")

    result_b, pca_model = run_comparative_test(
        test_label="B",
        test_description="Count: flying cats vs flying dogs",
        dataframe=dataframe,
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        tabular_count_func_cat=lambda: tabular_count_animal_can_fly(dataframe, "cat"),
        tabular_count_func_dog=lambda: tabular_count_animal_can_fly(dataframe, "dog"),
        ground_truth_func_cat=lambda: generate_ground_truth_animal_can_fly(dataframe, "cat"),
        ground_truth_func_dog=lambda: generate_ground_truth_animal_can_fly(dataframe, "dog"),
        vector_query_func_cat=lambda: vector_query_animal_can_fly(
            corpus_embeddings, embedding_model, similarity_threshold, "cat"
        ),
        vector_query_func_dog=lambda: vector_query_animal_can_fly(
            corpus_embeddings, embedding_model, similarity_threshold, "dog"
        ),
        output_directory=output_directory,
        timestamp_string=timestamp_string,
        pca_model=pca_model,
    )
    all_results.append(result_b)

    # --- Test C: How many cats vs dogs born after T? ---
    print(f"\n[RUNNING] Test C: How many cats vs dogs born after {time_threshold_year}?")

    result_c, pca_model = run_comparative_test(
        test_label="C",
        test_description=f"Count: cats vs dogs born after {time_threshold_year}",
        dataframe=dataframe,
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        tabular_count_func_cat=lambda: tabular_count_animal_born_after(
            dataframe, "cat", time_threshold_unix
        ),
        tabular_count_func_dog=lambda: tabular_count_animal_born_after(
            dataframe, "dog", time_threshold_unix
        ),
        ground_truth_func_cat=lambda: generate_ground_truth_animal_born_after(
            dataframe, "cat", time_threshold_unix
        ),
        ground_truth_func_dog=lambda: generate_ground_truth_animal_born_after(
            dataframe, "dog", time_threshold_unix
        ),
        vector_query_func_cat=lambda: vector_query_animal_born_after(
            corpus_embeddings, embedding_model, similarity_threshold,
            "cat", time_threshold_year
        ),
        vector_query_func_dog=lambda: vector_query_animal_born_after(
            corpus_embeddings, embedding_model, similarity_threshold,
            "dog", time_threshold_year
        ),
        output_directory=output_directory,
        timestamp_string=timestamp_string,
        pca_model=pca_model,
    )
    all_results.append(result_c)

    # --- Test D: How many flying cats vs flying dogs born after T? ---
    print(f"\n[RUNNING] Test D: Flying cats vs flying dogs born after {time_threshold_year}?")

    result_d, pca_model = run_comparative_test(
        test_label="D",
        test_description=f"Count: flying cats vs flying dogs after {time_threshold_year}",
        dataframe=dataframe,
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        tabular_count_func_cat=lambda: tabular_count_animal_born_after_can_fly(
            dataframe, "cat", time_threshold_unix
        ),
        tabular_count_func_dog=lambda: tabular_count_animal_born_after_can_fly(
            dataframe, "dog", time_threshold_unix
        ),
        ground_truth_func_cat=lambda: generate_ground_truth_animal_born_after_can_fly(
            dataframe, "cat", time_threshold_unix
        ),
        ground_truth_func_dog=lambda: generate_ground_truth_animal_born_after_can_fly(
            dataframe, "dog", time_threshold_unix
        ),
        vector_query_func_cat=lambda: vector_query_animal_born_after_can_fly(
            corpus_embeddings, embedding_model, similarity_threshold,
            "cat", time_threshold_year
        ),
        vector_query_func_dog=lambda: vector_query_animal_born_after_can_fly(
            corpus_embeddings, embedding_model, similarity_threshold,
            "dog", time_threshold_year
        ),
        output_directory=output_directory,
        timestamp_string=timestamp_string,
        pca_model=pca_model,
    )
    all_results.append(result_d)

    # -------------------------------------------------------------------------
    # Step 7: Print summary table
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 7: SUMMARY")
    print(f"{'='*70}")

    print_comparative_summary_table(all_results)

    # -------------------------------------------------------------------------
    # Step 8: Save summary CSV
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 8: SAVE SUMMARY")
    print(f"{'='*70}")

    save_comparative_summary_csv(
        all_results=all_results,
        output_directory=output_directory,
        timestamp_string=timestamp_string
    )

    # -------------------------------------------------------------------------
    # Step 9: Configuration summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CONFIGURATION USED (for reproducibility)")
    print(f"{'='*70}")
    print(f"  CSV file: {csv_file_path}")
    print(f"  Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  Similarity threshold: {similarity_threshold}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Time threshold (Unix): {time_threshold_unix}")
    print(f"  Time threshold (Year): {time_threshold_year}")
    print(f"  Total documents: {len(dataframe)}")
    print(f"  Output directory: {output_directory}")
    print(f"  Timestamp: {timestamp_string}")

    # List saved files
    print("\n  Saved files:")
    for saved_file in sorted(output_directory.iterdir()):
        print(f"    {saved_file.name}")

    return all_results


# =============================================================================
# 3D PCA VISUALIZATION FUNCTIONS
# =============================================================================


def generate_comparative_pca_plot(
    corpus_embeddings: np.ndarray,
    similarities_cat: np.ndarray,
    similarities_dog: np.ndarray,
    similarity_threshold: float,
    cat_query_text: str,
    dog_query_text: str,
    embedding_model: SentenceTransformer,
    test_label: str,
    output_directory: Path,
    timestamp_string: str,
    pca_model: PCA | None = None,
) -> PCA:
    """
    Generate a 3D PCA scatter plot comparing cat vs dog query results.

    Corpus points are colored by match category:
        - cat_only:  cat similarity > threshold, dog <= threshold
        - dog_only:  dog similarity > threshold, cat <= threshold
        - both:      both similarities > threshold
        - neither:   both similarities <= threshold

    Both query vectors are embedded, projected into the same PCA space,
    and displayed as star markers.

    PCA is fitted on the corpus embeddings if no pre-fitted model is
    provided. The fitted model is returned for reuse across tests.

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed document embeddings, shape (n_docs, 384).

    similarities_cat : np.ndarray
        Cosine similarity scores for cat query, shape (n_docs,).

    similarities_dog : np.ndarray
        Cosine similarity scores for dog query, shape (n_docs,).

    similarity_threshold : float
        Threshold used for match determination.

    cat_query_text : str
        Query text used for cat vector query.

    dog_query_text : str
        Query text used for dog vector query.

    embedding_model : SentenceTransformer
        Model used to embed query texts into same vector space.

    test_label : str
        Short identifier for the test (e.g., "A", "B").

    output_directory : Path
        Directory to save HTML output file.

    timestamp_string : str
        Timestamp string for filename uniqueness.

    pca_model : PCA or None
        Pre-fitted PCA model. If None, a new PCA(n_components=3) is
        fitted on corpus_embeddings. Pass the returned model to
        subsequent calls to ensure all plots share the same PCA space.

    Returns
    -------
    PCA
        The fitted PCA model (newly fitted or the one passed in).

    Raises
    ------
    Exception
        Logged with traceback if plot generation fails.
        Does NOT re-raise — plot failure should not halt the test suite.

    Notes
    -----
    Output file: {test_label}_comparative_pca_3d_{timestamp_string}.html
    The HTML file is fully self-contained (no external JS dependencies).
    """
    try:
        print(f"[VIZ] Generating comparative 3D PCA plot for Test {test_label}...")

        # ---- Step 1: Fit or reuse PCA model ----
        if pca_model is None:
            print("[VIZ] Fitting PCA(n_components=3) on corpus embeddings...")
            pca_model = PCA(n_components=3, random_state=RANDOM_SEED)
            pca_model.fit(corpus_embeddings)
            explained_variance_total = float(
                np.sum(pca_model.explained_variance_ratio_)
            )
            print(
                f"[VIZ] PCA explained variance (3 components): "
                f"{explained_variance_total:.4f}"
            )
        else:
            print("[VIZ] Reusing pre-fitted PCA model.")

        # ---- Step 2: Transform corpus to 3D ----
        corpus_3d = pca_model.transform(corpus_embeddings)

        # ---- Step 3: Embed and project both query vectors ----
        cat_query_embedding = embed_single_query(cat_query_text, embedding_model)
        dog_query_embedding = embed_single_query(dog_query_text, embedding_model)

        # PCA.transform expects 2D input
        cat_query_3d = pca_model.transform(cat_query_embedding.reshape(1, -1))[0]
        dog_query_3d = pca_model.transform(dog_query_embedding.reshape(1, -1))[0]

        # ---- Step 4: Classify corpus points into match categories ----
        cat_match_mask = similarities_cat > similarity_threshold
        dog_match_mask = similarities_dog > similarity_threshold

        category_labels = np.empty(len(corpus_embeddings), dtype="U10")
        category_labels[cat_match_mask & ~dog_match_mask] = "cat_only"
        category_labels[~cat_match_mask & dog_match_mask] = "dog_only"
        category_labels[cat_match_mask & dog_match_mask] = "both"
        category_labels[~cat_match_mask & ~dog_match_mask] = "neither"

        # ---- Step 5: Define colors and build traces ----
        category_config = {
            "cat_only": {"color": "orange", "name": "Cat Only"},
            "dog_only": {"color": "dodgerblue", "name": "Dog Only"},
            "both":     {"color": "red", "name": "Both"},
            "neither":  {"color": "lightgray", "name": "Neither"},
        }

        fig = go.Figure()

        for category_key, config in category_config.items():
            mask = category_labels == category_key
            point_count = int(np.sum(mask))

            if point_count == 0:
                continue

            # Build hover text: row index + both similarity scores
            hover_texts = [
                (
                    f"Row {idx}<br>"
                    f"Cat sim: {similarities_cat[idx]:.4f}<br>"
                    f"Dog sim: {similarities_dog[idx]:.4f}"
                )
                for idx in np.where(mask)[0]
            ]

            fig.add_trace(go.Scatter3d(
                x=corpus_3d[mask, 0],
                y=corpus_3d[mask, 1],
                z=corpus_3d[mask, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=config["color"],
                    opacity=0.6,
                ),
                name=f"{config['name']} ({point_count})",
                text=hover_texts,
                hoverinfo="text",
            ))

        # ---- Step 6: Add query point markers ----
        fig.add_trace(go.Scatter3d(
            x=[cat_query_3d[0]],
            y=[cat_query_3d[1]],
            z=[cat_query_3d[2]],
            mode="markers+text",
            marker=dict(
                size=12,
                color="orange",
                symbol="diamond",
                line=dict(width=2, color="black"),
            ),
            name="Cat Query",
            text=["Cat Query"],
            textposition="top center",
            hovertext=f"Cat Query<br>{cat_query_text[:80]}",
            hoverinfo="text",
        ))

        fig.add_trace(go.Scatter3d(
            x=[dog_query_3d[0]],
            y=[dog_query_3d[1]],
            z=[dog_query_3d[2]],
            mode="markers+text",
            marker=dict(
                size=12,
                color="dodgerblue",
                symbol="diamond",
                line=dict(width=2, color="black"),
            ),
            name="Dog Query",
            text=["Dog Query"],
            textposition="top center",
            hovertext=f"Dog Query<br>{dog_query_text[:80]}",
            hoverinfo="text",
        ))

        # ---- Step 7: Layout and title ----
        plot_title = (
            f"Test {test_label} — Comparative PCA 3D<br>"
            f"<sub>Cat: \"{cat_query_text[:60]}\" | "
            f"Dog: \"{dog_query_text[:60]}\" | "
            f"Threshold: {similarity_threshold}</sub>"
        )

        fig.update_layout(
            title=plot_title,
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
            margin=dict(l=0, r=0, b=0, t=80),
        )

        # ---- Step 8: Save HTML ----
        filename = (
            f"{test_label}_comparative_pca_3d_{timestamp_string}.html"
        )
        filepath = output_directory / filename

        fig.write_html(
            str(filepath),
            include_plotlyjs=True,
            full_html=True,
        )

        print(f"[VIZ] Saved: {filepath}")

        return pca_model

    except Exception as visualization_error:
        print(
            f"[ERROR] Failed to generate comparative PCA plot "
            f"for Test {test_label}: {visualization_error}"
        )
        traceback.print_exc()
        # Return pca_model even on failure so subsequent tests can still try
        # If pca_model was None and fitting failed, return None
        return pca_model  # type: ignore[return-value]

# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Main entry point for comparative vector analytics test suite.

    Prompts user for CSV path and similarity threshold, then runs tests.
    """
    print("\n" + "=" * 90)
    print("STARTING COMPARATIVE VECTOR ANALYTICS TEST SUITE")
    print("=" * 90 + "\n")

    try:
        # Prompt for CSV path
        csv_path_input = input(
            "Step 1: Enter path to synthetic_biology_dataset_augmented.csv:\n"
        )
        csv_file_path = csv_path_input.strip()

        if not csv_file_path:
            print("[ERROR] No CSV path provided. Exiting.")
            return

        # Prompt for threshold
        threshold_input = input(
            "\nStep 2: Enter similarity threshold (default 0.5):\n"
        )
        threshold_input = threshold_input.strip()

        if threshold_input:
            try:
                similarity_threshold = float(threshold_input)
            except ValueError:
                print(f"[ERROR] Invalid threshold '{threshold_input}'. Using default 0.5")
                similarity_threshold = 0.5
        else:
            similarity_threshold = 0.5

        # Run test suite
        all_results = run_comparative_test_suite(
            csv_file_path=csv_file_path,
            similarity_threshold=similarity_threshold
        )
        print(f"\nall_results: \n{all_results}")

        print("\n" + "=" * 90)
        print("COMPARATIVE TEST SUITE COMPLETED SUCCESSFULLY")
        print("=" * 90)

    except FileNotFoundError as file_error:
        print(f"\n[FATAL ERROR] File not found: {file_error}")
        traceback.print_exc()

    except ValueError as validation_error:
        print(f"\n[FATAL ERROR] Data validation failed: {validation_error}")
        traceback.print_exc()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test suite cancelled by user.")

    except Exception as unexpected_error:
        print(f"\n[FATAL ERROR] Unexpected error: {unexpected_error}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
