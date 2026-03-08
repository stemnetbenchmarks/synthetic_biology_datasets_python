# mvp1_hybrid_analytics_eval_vN.py
"""
mvp1_hybrid_analytics_eval_vN.py

## Steps to Run:
    1. run in python venv env:
    ```bash
    python mvp1_hybrid_analytics_eval_vN.py
    ```

    2. You will be prompted for threshold and data file path.

    3. See results in terminal and saved to "tests/tests_{timestamp}" dir

MVP-1 Vector Analytics Testing Framework
WITH Per-Row Confusion Matrix Evaluation

Purpose:
    Compare tabular (ground truth) vs. vector-based query results
    for analytics questions on synthetic pet data.

    This tests whether counting documents via vector similarity
    can approximate accurate structured-query counts.

    v4 adds:
        - Per-row ground truth boolean tables
        - Per-row confusion matrix classification (TP, FP, FN, TN)
        - Precision, Recall, F1, Accuracy metrics per query
        - Timestamped output directory with CSV artifacts
        - Combined summary report CSV

Tests A-E (per specification):
    A: How many cats? (single concept count)
    B: How many cats that can fly? (concept + 1 boolean filter)
    C: How many cats born after time T? (concept + time filter)
    D: How many cats born after T that can fly? (concept + time + 1 filter)
    E: How many cats with time + 5 field constraints? (stress test)

Methods compared:
    1. Tabular: pandas boolean filtering (ground truth)
    2. Vector: cosine similarity count above threshold

Output Files (in tests/tests_{timestamp}/ directory):
    - test_ground_truth_table_{timestamp}.csv
    - A_test_results_vector_confusion_matrix_{timestamp}.csv
    - B_test_results_vector_confusion_matrix_{timestamp}.csv
    - C_test_results_vector_confusion_matrix_{timestamp}.csv
    - D_test_results_vector_confusion_matrix_{timestamp}.csv
    - E_test_results_vector_confusion_matrix_{timestamp}.csv
    - summary_report_{timestamp}.csv

Key design decisions:
    - Uses sentence-transformers directly (no ChromaDB black box)
    - Exhaustive cosine similarity (not approximate nearest neighbor)
    - All similarity scores retained for analysis
    - Model loaded once, reused for all queries
    - Every row gets a TP/FP/FN/TN classification per query

Dependencies:
    - pandas
    - numpy
    - sentence-transformers

Author: [MVP-1 Test Framework]
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


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Path to CSV file containing pet data with unstructured_description field
this = input("\nStep 1: Enter path to .../synthetic_biology_dataset_augmented.csv \n")
CSV_FILE_PATH: str = this

# Sentence-transformer model (same default as ChromaDB)
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Similarity threshold: documents with similarity > this are counted as matches
# Note: This is a critical parameter that significantly affects results
# This is to be set per-test, by user, 0.5 is a baseline default
user_threshold = input("\nStep 2: Pick vector-match threshold (default: use 0.5)\n")
float_threshold = float(user_threshold)
VECTOR_SIMILARITY_THRESHOLD: float = float_threshold

# Random seed for any stochastic operations (reproducibility)
RANDOM_SEED: int = 42


# =============================================================================
# TIMESTAMP AND OUTPUT DIRECTORY SETUP
# =============================================================================


def create_timestamped_output_directory() -> tuple[str, Path]:
    """
    Create a timestamped output directory for test artifacts.

    Directory is created in the current working directory with format:
        tests/tests_{YYYYMMDD_HHMMSS}/

    Returns
    -------
    tuple[str, Path]
        - timestamp_string: e.g. "20260307_143022"
        - output_directory_path: Path object for the created directory

    Notes
    -----
    Directory is created immediately. If it already exists
    (extremely unlikely given second-resolution timestamps),
    no error is raised (exist_ok=True).
    """
    try:
        now = datetime.datetime.now()
        timestamp_string = now.strftime("%Y%m%d_%H%M%S")

        output_directory_name = f"tests/tests_{timestamp_string}"
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
        raise

    except ValueError:
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

    This is the primary metric for vector-based counting queries.

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
        Dictionary containing mean, std, min, max, median, q25, q75.
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
# GROUND TRUTH GENERATION FUNCTIONS
#
# These produce per-row boolean arrays indicating whether each row
# SHOULD match a given query, according to structured tabular logic.
#
# These are the authoritative "correct answers" against which vector
# results are evaluated.
#
# Each function returns a numpy boolean array of shape (n_rows,).
# =============================================================================

def generate_ground_truth_query_a(
    dataframe: pd.DataFrame
) -> np.ndarray:
    """
    Generate per-row ground truth for Query A: Is this row a cat?

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    Returns
    -------
    np.ndarray
        Boolean array, shape (n_rows,). True where animal_type == 'cat'.
    """
    ground_truth_mask = (dataframe["animal_type"] == "cat").values
    return ground_truth_mask


def generate_ground_truth_query_b(
    dataframe: pd.DataFrame
) -> np.ndarray:
    """
    Generate per-row ground truth for Query B: Is this a cat that can fly?

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    Returns
    -------
    np.ndarray
        Boolean array. True where animal_type == 'cat' AND can_fly == True.
    """
    ground_truth_mask = (
        (dataframe["animal_type"] == "cat") &
        (dataframe["can_fly"] == True)
    ).values
    return ground_truth_mask


def generate_ground_truth_query_c(
    dataframe: pd.DataFrame,
    time_threshold_unix: int
) -> np.ndarray:
    """
    Generate per-row ground truth for Query C: Is this a cat born after T?

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    time_threshold_unix : int
        Unix timestamp. Rows with birth_unix > this are True.

    Returns
    -------
    np.ndarray
        Boolean array. True where cat AND born after threshold.
    """
    ground_truth_mask = (
        (dataframe["animal_type"] == "cat") &
        (dataframe["birth_unix"] > time_threshold_unix)
    ).values
    return ground_truth_mask


def generate_ground_truth_query_d(
    dataframe: pd.DataFrame,
    time_threshold_unix: int
) -> np.ndarray:
    """
    Generate per-row ground truth for Query D: Cat born after T that can fly?

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    time_threshold_unix : int
        Unix timestamp threshold.

    Returns
    -------
    np.ndarray
        Boolean array. True where cat AND born after T AND can fly.
    """
    ground_truth_mask = (
        (dataframe["animal_type"] == "cat") &
        (dataframe["birth_unix"] > time_threshold_unix) &
        (dataframe["can_fly"] == True)
    ).values
    return ground_truth_mask


def generate_ground_truth_query_e(
    dataframe: pd.DataFrame,
    time_threshold_unix: int,
    cat_median_weight_kg: float,
    cat_median_height_cm: float,
    cat_median_daily_food_grams: float,
    cat_majority_can_run: bool
) -> np.ndarray:
    """
    Generate per-row ground truth for Query E: 5-field stress test.

    Filters:
        1. animal_type == 'cat'
        2. birth_unix > time_threshold_unix
        3. weight_kg > cat_median_weight_kg
        4. height_cm < cat_median_height_cm
        5. daily_food_grams > cat_median_daily_food_grams
        6. can_run == cat_majority_can_run

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    time_threshold_unix : int
        Unix timestamp threshold.

    cat_median_weight_kg : float
        Cat-specific median weight. Filter: weight > this.

    cat_median_height_cm : float
        Cat-specific median height. Filter: height < this.

    cat_median_daily_food_grams : float
        Cat-specific median food. Filter: food > this.

    cat_majority_can_run : bool
        Cat-specific majority class for can_run.

    Returns
    -------
    np.ndarray
        Boolean array. True where ALL six conditions are met.
    """
    ground_truth_mask = (
        (dataframe["animal_type"] == "cat") &
        (dataframe["birth_unix"] > time_threshold_unix) &
        (dataframe["weight_kg"] > cat_median_weight_kg) &
        (dataframe["height_cm"] < cat_median_height_cm) &
        (dataframe["daily_food_grams"] > cat_median_daily_food_grams) &
        (dataframe["can_run"] == cat_majority_can_run)
    ).values
    return ground_truth_mask


# =============================================================================
# CONFUSION MATRIX COMPUTATION
#
# Given a per-row ground truth boolean array and a per-row vector
# prediction boolean array, classify every row as TP, FP, FN, or TN.
#
# This is the core evaluation step that goes beyond count-comparison.
# A count of 4000 vs 4000 could still be the WRONG 4000 documents.
# The confusion matrix reveals whether the right documents were matched.
# =============================================================================


def classify_rows_into_confusion_categories(
    ground_truth_boolean_array: np.ndarray,
    vector_predicted_boolean_array: np.ndarray
) -> np.ndarray:
    """
    Classify each row as TP, FP, FN, or TN.

    Definitions:
        TP (True Positive):  ground_truth=True  AND predicted=True
            -> correctly matched
        FP (False Positive): ground_truth=False AND predicted=True
            -> incorrectly matched (should not have been returned)
        FN (False Negative): ground_truth=True  AND predicted=False
            -> missed (should have been returned but was not)
        TN (True Negative):  ground_truth=False AND predicted=False
            -> correctly excluded

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

    Raises
    ------
    ValueError
        If arrays have different lengths.
    """
    if len(ground_truth_boolean_array) != len(vector_predicted_boolean_array):
        raise ValueError(
            f"Array length mismatch: ground_truth has {len(ground_truth_boolean_array)} "
            f"rows, predicted has {len(vector_predicted_boolean_array)} rows. "
            f"These must be the same length (one entry per document)."
        )

    row_count = len(ground_truth_boolean_array)

    # Pre-allocate string array
    confusion_categories = np.empty(row_count, dtype="U2")

    # Vectorized classification using boolean logic
    # True Positive: both true
    tp_mask = ground_truth_boolean_array & vector_predicted_boolean_array
    confusion_categories[tp_mask] = "TP"

    # False Positive: predicted true but ground truth false
    fp_mask = (~ground_truth_boolean_array) & vector_predicted_boolean_array
    confusion_categories[fp_mask] = "FP"

    # False Negative: ground truth true but predicted false
    fn_mask = ground_truth_boolean_array & (~vector_predicted_boolean_array)
    confusion_categories[fn_mask] = "FN"

    # True Negative: both false
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
        Contains:
        - tp_count: int
        - fp_count: int
        - fn_count: int
        - tn_count: int
        - total_count: int (should equal len of input)
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
    Compute precision, recall, F1, and accuracy from confusion counts.

    Handles edge cases where denominators are zero (returns None).

    Parameters
    ----------
    confusion_counts : dict
        Contains tp_count, fp_count, fn_count, tn_count.

    Returns
    -------
    dict
        Contains:
        - precision: float or None (TP / (TP + FP))
        - recall: float or None (TP / (TP + FN))
        - f1_score: float or None (2 * precision * recall / (precision + recall))
        - accuracy: float or None ((TP + TN) / total)
        - specificity: float or None (TN / (TN + FP))

    Notes
    -----
    Precision answers: "Of all documents the vector matched, how many were correct?"
    Recall answers:    "Of all documents that should have matched, how many did?"
    F1 answers:        "Harmonic mean balancing precision and recall."
    Accuracy answers:  "Of all documents, how many were correctly classified?"
    Specificity:       "Of all non-matching documents, how many were correctly excluded?"
    """
    tp = confusion_counts["tp_count"]
    fp = confusion_counts["fp_count"]
    fn = confusion_counts["fn_count"]
    tn = confusion_counts["tn_count"]
    total = confusion_counts["total_count"]

    # Precision: TP / (TP + FP)
    # "Of what we matched, how many were correct?"
    precision_denominator = tp + fp
    if precision_denominator > 0:
        precision = tp / precision_denominator
    else:
        # No documents were predicted as matches at all
        precision = None

    # Recall: TP / (TP + FN)
    # "Of what should have matched, how many did we find?"
    recall_denominator = tp + fn
    if recall_denominator > 0:
        recall = tp / recall_denominator
    else:
        # No documents should have matched (ground truth is all-negative)
        recall = None

    # F1 Score: harmonic mean of precision and recall
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1_score = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score = None

    # Accuracy: (TP + TN) / total
    if total > 0:
        accuracy = (tp + tn) / total
    else:
        accuracy = None

    # Specificity: TN / (TN + FP)
    # "Of non-matching documents, how many were correctly excluded?"
    specificity_denominator = tn + fp
    if specificity_denominator > 0:
        specificity = tn / specificity_denominator
    else:
        specificity = None

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "specificity": specificity,
    }


# =============================================================================
# PER-ROW RESULTS DATAFRAME CONSTRUCTION
#
# Build a DataFrame containing, for each row in the corpus:
#   - row_index (integer position in original data)
#   - similarity_score (cosine similarity for this query)
#   - vector_predicted_match (bool: similarity > threshold)
#   - ground_truth_match (bool: from tabular filter)
#   - confusion_class (str: TP, FP, FN, or TN)
#
# This is the artifact that allows auditing individual document
# classifications after the test run.
# =============================================================================

# =============================================================================
# FALSE POSITIVE DIAGNOSTIC FUNCTIONS
#
# For each FP row, determine WHICH filter conditions that row fails.
# This enables analysis of what the vector model is "confused by"
# vs. what it correctly distinguishes but over-matches on.
#
# A fp_wrong_X = True means: this row does NOT satisfy condition X,
# meaning X is a contributing reason this row is a false positive.
#
# For non-FP rows, all diagnostic columns are False and count is 0.
# =============================================================================


def compute_fp_diagnostic_columns(
    dataframe: pd.DataFrame,
    confusion_categories: np.ndarray,
    query_filter_conditions: dict[str, tuple]
) -> pd.DataFrame:
    """
    For each FP row, check which query filter conditions that row fails.

    Each filter condition becomes a boolean column: True means the row
    FAILS that condition (i.e., that field is a reason this row should
    not have been matched).

    For non-FP rows, all diagnostic columns are False.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Original dataset with all structured fields.
        Must contain all columns referenced in query_filter_conditions.

    confusion_categories : np.ndarray
        String array of shape (n_rows,) with values "TP", "FP", "FN", "TN".
        Only rows classified as "FP" get diagnostic analysis.

    query_filter_conditions : dict[str, tuple]
        Maps field names to (operator_string, threshold_value) tuples.
        Supported operators: "==", ">", "<", ">=", "<=", "!="

        Example for Query D:
            {
                "animal_type": ("==", "cat"),
                "birth_unix": (">", 1515042000),
                "can_fly": ("==", True),
            }

    Returns
    -------
    pd.DataFrame
        Contains columns:
        - fp_wrong_{field_name}: bool, one per filter condition
        - fp_wrong_fields_count: int, total failed conditions per row

        All False / 0 for non-FP rows.
        Shape: (n_rows, len(query_filter_conditions) + 1)

    Raises
    ------
    ValueError
        If an unsupported operator string is encountered.
        If a referenced column is missing from dataframe.
    """
    try:
        row_count = len(dataframe)
        fp_mask = confusion_categories == "FP"

        # Validate that all referenced columns exist in dataframe
        missing_columns = [
            field_name for field_name in query_filter_conditions
            if field_name not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                f"FP diagnostic references columns not in dataframe: {missing_columns}. "
                f"Available columns: {list(dataframe.columns)}"
            )

        # Supported comparison operators mapped to callables
        operator_dispatch = {
            "==": lambda series, val: series == val,
            "!=": lambda series, val: series != val,
            ">":  lambda series, val: series > val,
            "<":  lambda series, val: series < val,
            ">=": lambda series, val: series >= val,
            "<=": lambda series, val: series <= val,
        }

        # Build one boolean column per filter condition
        diagnostic_columns: dict[str, np.ndarray] = {}

        for field_name, (operator_string, threshold_value) in query_filter_conditions.items():

            if operator_string not in operator_dispatch:
                raise ValueError(
                    f"Unsupported operator '{operator_string}' for field '{field_name}'. "
                    f"Supported operators: {list(operator_dispatch.keys())}"
                )

            # Evaluate whether each row PASSES this condition
            comparison_function = operator_dispatch[operator_string]
            row_passes_condition = comparison_function(
                dataframe[field_name], threshold_value
            ).values

            # "Wrong" means the row FAILS the condition
            # But only mark as wrong for FP rows; non-FP rows get False
            fp_wrong_for_field = np.zeros(row_count, dtype=bool)
            fp_wrong_for_field[fp_mask] = ~row_passes_condition[fp_mask]

            column_name = f"fp_wrong_{field_name}"
            diagnostic_columns[column_name] = fp_wrong_for_field

        # Count total wrong fields per row
        if len(diagnostic_columns) > 0:
            stacked_wrong_fields = np.column_stack(
                list(diagnostic_columns.values())
            )
            fp_wrong_fields_count = stacked_wrong_fields.sum(axis=1).astype(int)
        else:
            fp_wrong_fields_count = np.zeros(row_count, dtype=int)

        diagnostic_columns["fp_wrong_fields_count"] = fp_wrong_fields_count

        fp_diagnostic_dataframe = pd.DataFrame(diagnostic_columns)

        return fp_diagnostic_dataframe

    except Exception as fp_diagnostic_error:
        print(f"[ERROR] Failed to compute FP diagnostics: {fp_diagnostic_error}")
        traceback.print_exc()
        raise


def summarize_fp_diagnostic_columns(
    fp_diagnostic_dataframe: pd.DataFrame,
    confusion_categories: np.ndarray,
    query_label: str,
    output_directory: Path,
    timestamp_string: str,
) -> dict[str, dict]:
    """
    Summarize which fields contribute most to false positives for one query.
    Prints summary to stdout AND saves to CSV file.

    Reports two metrics per field:
        1. fp_rows_affected_proportion: Of all FP rows, what fraction fail
           on this field? (These overlap; a row can fail multiple fields.)
        2. error_contribution_proportion: Of all field-failure instances
           across all FP rows, what fraction come from this field?
           (These are mutually exclusive and sum to 1.0.)

    Parameters
    ----------
    fp_diagnostic_dataframe : pd.DataFrame
        Output of compute_fp_diagnostic_columns.
        Contains fp_wrong_{field} boolean columns and fp_wrong_fields_count.

    confusion_categories : np.ndarray
        String array with "TP", "FP", "FN", "TN" per row.
        Used to count total FP rows.

    query_label : str
        Query identifier for printing and filename (e.g., "A", "B").

    output_directory : Path
        Directory to save the summary CSV.

    timestamp_string : str
        Timestamp for filename.

    Returns
    -------
    dict[str, dict]
        Keyed by field name (e.g., "animal_type").
        Each value contains:
        - wrong_count: int
        - fp_rows_affected_proportion: float or None
        - error_contribution_proportion: float or None

    Notes
    -----
    Saves CSV to: {output_directory}/{query_label}_fp_diagnostic_summary_{timestamp}.csv
    """
    fp_total_count = int(np.sum(confusion_categories == "FP"))

    # Identify the fp_wrong_ boolean columns (exclude fp_wrong_fields_count)
    fp_wrong_columns = [
        col for col in fp_diagnostic_dataframe.columns
        if col.startswith("fp_wrong_") and col != "fp_wrong_fields_count"
    ]

    # Calculate total field-failure instances for error contribution denominator
    # This is the sum of all fp_wrong_X values across all rows
    total_field_failure_instances = 0
    field_wrong_counts: dict[str, int] = {}

    for column_name in fp_wrong_columns:
        wrong_count = int(fp_diagnostic_dataframe[column_name].sum()) # type: ignore[arg-type]
        field_wrong_counts[column_name] = wrong_count
        total_field_failure_instances += wrong_count

    # Build summary dict and prepare rows for CSV
    field_summary: dict[str, dict] = {}
    summary_rows_for_csv: list[dict] = []

    print(f"\n  FP FIELD DIAGNOSTIC (Query {query_label}):")

    if fp_total_count == 0:
        print("    No false positives to diagnose.")
        # Save empty CSV with headers only
        empty_summary_df = pd.DataFrame(columns=[
            "query_label",
            "field_name",
            "wrong_count",
            "fp_rows_affected_proportion",
            "error_contribution_proportion",
        ])
        filename = f"{query_label}_fp_diagnostic_summary_{timestamp_string}.csv"
        filepath = output_directory / filename
        empty_summary_df.to_csv(filepath, index=False)
        print(f"    [SAVE] FP diagnostic summary saved: {filepath}")
        return field_summary

    print(f"    Total FP rows: {fp_total_count}")
    print(f"    Total field-failure instances: {total_field_failure_instances}")
    print()
    print(f"    {'Field':<30} {'Wrong':>8} {'FP Rows':>12} {'Error':>12}")
    print(f"    {'':30} {'Count':>8} {'Affected %':>12} {'Contrib %':>12}")
    print(f"    {'-'*62}")

    for column_name in fp_wrong_columns:
        wrong_count = field_wrong_counts[column_name]

        # Extract clean field name (remove "fp_wrong_" prefix)
        field_name = column_name.replace("fp_wrong_", "")

        # Metric 1: Proportion of FP rows affected by this field
        # (overlapping - one row can fail multiple fields)
        if fp_total_count > 0:
            fp_rows_affected_proportion = wrong_count / fp_total_count
        else:
            fp_rows_affected_proportion = None

        # Metric 2: Error contribution (mutually exclusive, sums to 1.0)
        # What fraction of total field-failures come from this field?
        if total_field_failure_instances > 0:
            error_contribution_proportion = wrong_count / total_field_failure_instances
        else:
            error_contribution_proportion = None

        field_summary[field_name] = {
            "wrong_count": wrong_count,
            "fp_rows_affected_proportion": fp_rows_affected_proportion,
            "error_contribution_proportion": error_contribution_proportion,
        }

        summary_rows_for_csv.append({
            "query_label": query_label,
            "field_name": field_name,
            "wrong_count": wrong_count,
            "fp_rows_affected_proportion": fp_rows_affected_proportion,
            "error_contribution_proportion": error_contribution_proportion,
        })

        # Format for display
        affected_display = (
            f"{fp_rows_affected_proportion * 100:.2f}%"
            if fp_rows_affected_proportion is not None else "N/A"
        )
        contrib_display = (
            f"{error_contribution_proportion * 100:.2f}%"
            if error_contribution_proportion is not None else "N/A"
        )

        print(
            f"    {field_name:<30} {wrong_count:>8} {affected_display:>12} {contrib_display:>12}"
        )

    # Verify error contribution sums to ~100%
    total_contribution = sum(
        s["error_contribution_proportion"]
        for s in field_summary.values()
        if s["error_contribution_proportion"] is not None
    )
    print(f"    {'-'*62}")
    print(f"    {'TOTAL':<30} {total_field_failure_instances:>8} {'(overlap)':>12} {total_contribution * 100:.2f}%")

    # Report distribution of fp_wrong_fields_count among FP rows
    fp_mask = confusion_categories == "FP"
    if fp_total_count > 0:
        fp_wrong_counts_among_fps = fp_diagnostic_dataframe.loc[
            fp_mask, "fp_wrong_fields_count"
        ]
        print("\n    FP wrong-fields-count distribution (among FP rows):")
        print(f"      Mean:   {fp_wrong_counts_among_fps.mean():.2f}")
        print(f"      Median: {fp_wrong_counts_among_fps.median():.1f}")
        print(f"      Min:    {fp_wrong_counts_among_fps.min()}")
        print(f"      Max:    {fp_wrong_counts_among_fps.max()}")

    # Save to CSV
    summary_df = pd.DataFrame(summary_rows_for_csv)
    filename = f"{query_label}_fp_diagnostic_summary_{timestamp_string}.csv"
    filepath = output_directory / filename
    summary_df.to_csv(filepath, index=False)
    print(f"\n    [SAVE] FP diagnostic summary saved: {filepath}")

    return field_summary


def build_per_row_confusion_matrix_dataframe(
    dataframe: pd.DataFrame,
    similarity_scores: np.ndarray,
    similarity_threshold: float,
    ground_truth_boolean_array: np.ndarray,
    query_filter_conditions: dict[str, tuple]
) -> pd.DataFrame:
    """
    Build a per-row DataFrame with similarity, prediction, truth,
    confusion class, AND FP diagnostic columns.

    This DataFrame is the primary audit artifact. Any row can be inspected
    to see why it was classified as TP, FP, FN, or TN, and for FP rows,
    which specific filter fields the row fails on.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Original full dataset. Must contain unique_id and all columns
        referenced in query_filter_conditions.

    similarity_scores : np.ndarray
        Cosine similarity scores for all documents. Shape (n_rows,).

    similarity_threshold : float
        Threshold used for vector match prediction.

    ground_truth_boolean_array : np.ndarray
        Boolean ground truth for this query. Shape (n_rows,).

    query_filter_conditions : dict[str, tuple]
        Maps field names to (operator_string, threshold_value) tuples.
        Used for FP diagnostic analysis.
        Example: {"animal_type": ("==", "cat"), "can_fly": ("==", True)}

    Returns
    -------
    pd.DataFrame
        Columns:
        - unique_id: int (from original dataset)
        - row_index: int (0-based position in original dataset)
        - similarity_score: float
        - similarity_threshold_used: float
        - vector_predicted_match: bool
        - ground_truth_match: bool
        - confusion_class: str ("TP", "FP", "FN", or "TN")
        - fp_wrong_{field_name}: bool (one per filter condition)
        - fp_wrong_fields_count: int
    """
    # Compute vector predictions
    vector_predicted_boolean_array = similarity_scores > similarity_threshold

    # Classify each row
    confusion_categories = classify_rows_into_confusion_categories(
        ground_truth_boolean_array=ground_truth_boolean_array,
        vector_predicted_boolean_array=vector_predicted_boolean_array
    )

    # Build base DataFrame
    per_row_results_dataframe = pd.DataFrame({
        "unique_id": dataframe["unique_id"].values,
        "row_index": np.arange(len(similarity_scores)),
        "similarity_score": similarity_scores,
        "similarity_threshold_used": similarity_threshold,
        "vector_predicted_match": vector_predicted_boolean_array,
        "ground_truth_match": ground_truth_boolean_array,
        "confusion_class": confusion_categories,
    })

    # Compute FP diagnostic columns
    fp_diagnostic_dataframe = compute_fp_diagnostic_columns(
        dataframe=dataframe,
        confusion_categories=confusion_categories,
        query_filter_conditions=query_filter_conditions,
    )

    # Concatenate base results with FP diagnostics
    per_row_results_dataframe = pd.concat(
        [per_row_results_dataframe, fp_diagnostic_dataframe],
        axis=1,
    )

    return per_row_results_dataframe

# =============================================================================
# GROUND TRUTH TABLE CONSTRUCTION AND SAVING
# =============================================================================

def build_combined_ground_truth_dataframe(
    dataframe: pd.DataFrame,
    time_threshold_unix: int,
    query_e_thresholds: dict
) -> pd.DataFrame:
    """
    Build a single DataFrame with ground truth boolean columns for all queries A-E.

    This combined table allows inspection of which rows should match
    each query, without re-running the tabular filters.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    time_threshold_unix : int
        Unix timestamp threshold used for queries C, D, E.

    query_e_thresholds : dict
        Data-derived thresholds for Query E. Must contain:
        - cat_median_weight_kg
        - cat_median_height_cm
        - cat_median_daily_food_grams
        - cat_majority_can_run

    Returns
    -------
    pd.DataFrame
        Columns:
        - row_index: int
        - query_A_ground_truth: bool (is cat)
        - query_B_ground_truth: bool (is cat that can fly)
        - query_C_ground_truth: bool (is cat born after T)
        - query_D_ground_truth: bool (is cat born after T, can fly)
        - query_E_ground_truth: bool (5-field stress test)
    """
    ground_truth_a = generate_ground_truth_query_a(dataframe)
    ground_truth_b = generate_ground_truth_query_b(dataframe)
    ground_truth_c = generate_ground_truth_query_c(dataframe, time_threshold_unix)
    ground_truth_d = generate_ground_truth_query_d(dataframe, time_threshold_unix)
    ground_truth_e = generate_ground_truth_query_e(
        dataframe=dataframe,
        time_threshold_unix=time_threshold_unix,
        cat_median_weight_kg=query_e_thresholds["cat_median_weight_kg"],
        cat_median_height_cm=query_e_thresholds["cat_median_height_cm"],
        cat_median_daily_food_grams=query_e_thresholds["cat_median_daily_food_grams"],
        cat_majority_can_run=query_e_thresholds["cat_majority_can_run"],
    )

    combined_ground_truth_dataframe = pd.DataFrame({
        "row_index": np.arange(len(dataframe)),
        "query_A_ground_truth": ground_truth_a,
        "query_B_ground_truth": ground_truth_b,
        "query_C_ground_truth": ground_truth_c,
        "query_D_ground_truth": ground_truth_d,
        "query_E_ground_truth": ground_truth_e,
    })

    # Print summary counts for verification
    print("[GROUND TRUTH] Per-query positive counts:")
    for col in ["query_A_ground_truth", "query_B_ground_truth",
                "query_C_ground_truth", "query_D_ground_truth",
                "query_E_ground_truth"]:
        positive_count = int(combined_ground_truth_dataframe[col].sum()) # type: ignore[arg-type]
        print(f"  {col}: {positive_count} positives out of {len(dataframe)}")

    return combined_ground_truth_dataframe


def save_ground_truth_csv(
    ground_truth_dataframe: pd.DataFrame,
    output_directory: Path,
    timestamp_string: str
) -> Path:
    """
    Save combined ground truth table to CSV.

    Parameters
    ----------
    ground_truth_dataframe : pd.DataFrame
        Combined ground truth with columns for queries A-E.

    output_directory : Path
        Directory to save into.

    timestamp_string : str
        Timestamp for filename.

    Returns
    -------
    Path
        Path to saved CSV file.
    """
    filename = f"test_ground_truth_table_{timestamp_string}.csv"
    filepath = output_directory / filename

    ground_truth_dataframe.to_csv(filepath, index=False)
    print(f"[SAVE] Ground truth table saved: {filepath}")

    return filepath


# =============================================================================
# PER-QUERY CONFUSION MATRIX CSV SAVING
# =============================================================================

def save_per_query_confusion_matrix_csv(
    per_row_results_dataframe: pd.DataFrame,
    query_label: str,
    output_directory: Path,
    timestamp_string: str
) -> Path:
    """
    Save per-row confusion matrix results for one query to CSV.

    Parameters
    ----------
    per_row_results_dataframe : pd.DataFrame
        Per-row results with similarity, prediction, truth, confusion class.

    query_label : str
        Query identifier (e.g., "A", "B", "C", "D", "E").

    output_directory : Path
        Directory to save into.

    timestamp_string : str
        Timestamp for filename.

    Returns
    -------
    Path
        Path to saved CSV file.
    """
    filename = f"{query_label}_test_results_vector_confusion_matrix_{timestamp_string}.csv"
    filepath = output_directory / filename

    per_row_results_dataframe.to_csv(filepath, index=False)
    print(f"[SAVE] Query {query_label} confusion matrix saved: {filepath}")

    return filepath


# =============================================================================
# SUMMARY REPORT CSV SAVING
# =============================================================================


def save_summary_report_csv(
    all_query_summaries: list[dict],
    output_directory: Path,
    timestamp_string: str
) -> Path:
    """
    Save aggregated summary report for all queries to CSV.

    One row per query (A-E) with counts, error metrics,
    confusion matrix counts, and derived metrics.

    Parameters
    ----------
    all_query_summaries : list[dict]
        List of summary dicts, one per query.

    output_directory : Path
        Directory to save into.

    timestamp_string : str
        Timestamp for filename.

    Returns
    -------
    Path
        Path to saved CSV file.
    """
    summary_dataframe = pd.DataFrame(all_query_summaries)

    filename = f"summary_report_{timestamp_string}.csv"
    filepath = output_directory / filename

    summary_dataframe.to_csv(filepath, index=False)
    print(f"[SAVE] Summary report saved: {filepath}")

    return filepath


# =============================================================================
# TABULAR QUERY FUNCTIONS (GROUND TRUTH)
#
# These return integer counts only, for backward compatibility.
# Ground truth boolean arrays are generated by the separate
# generate_ground_truth_query_* functions above.
# =============================================================================

def tabular_query_a_count_cats(
    dataframe: pd.DataFrame
) -> int:
    """
    Query A (Tabular Ground Truth): How many cats?

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


# =============================================================================
# QUERY E THRESHOLD COMPUTATION
# =============================================================================

def compute_query_e_thresholds_from_data(
    dataframe: pd.DataFrame
) -> dict:
    """
    Compute Query E filter thresholds from cat-specific data distributions.

    Uses medians for numeric fields and majority class for boolean fields.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset. Cat rows are filtered internally.

    Returns
    -------
    dict
        Contains:
        - cat_median_weight_kg: float
        - cat_median_height_cm: float
        - cat_median_daily_food_grams: float
        - cat_majority_can_run: bool
        - cat_count_total: int
    """
    try:
        cat_rows = dataframe[dataframe["animal_type"] == "cat"]
        cat_count_total = len(cat_rows)

        if cat_count_total == 0:
            raise ValueError(
                "No cat records found in dataset. "
                "Cannot compute Query E thresholds."
            )

        cat_median_weight_kg = float(cat_rows["weight_kg"].median()) # type: ignore[arg-type]
        cat_median_height_cm = float(cat_rows["height_cm"].median()) # type: ignore[arg-type]
        cat_median_daily_food_grams = float(cat_rows["daily_food_grams"].median()) # type: ignore[arg-type]

        can_run_value_counts = cat_rows["can_run"].value_counts() # type: ignore[arg-type]
        cat_majority_can_run = bool(can_run_value_counts.index[0])

        thresholds = {
            "cat_median_weight_kg": cat_median_weight_kg,
            "cat_median_height_cm": cat_median_height_cm,
            "cat_median_daily_food_grams": cat_median_daily_food_grams,
            "cat_majority_can_run": cat_majority_can_run,
            "cat_count_total": cat_count_total,
        }

        print(f"  [QUERY E THRESHOLDS] Computed from {cat_count_total} cat records:")
        print(f"    cat_median_weight_kg:        {cat_median_weight_kg:.2f}")
        print(f"    cat_median_height_cm:        {cat_median_height_cm:.2f}")
        print(f"    cat_median_daily_food_grams: {cat_median_daily_food_grams:.2f}")
        print(f"    cat_majority_can_run:        {cat_majority_can_run}")

        return thresholds

    except Exception as threshold_error:
        print(f"[ERROR] Failed to compute Query E thresholds: {threshold_error}")
        traceback.print_exc()
        raise


def tabular_query_e_count_cats_5_field_stress_test(
    dataframe: pd.DataFrame,
    time_threshold_unix: int,
    cat_median_weight_kg: float,
    cat_median_height_cm: float,
    cat_median_daily_food_grams: float,
    cat_majority_can_run: bool
) -> int:
    """
    Query E Revised (Tabular Ground Truth): 5-field stress test.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Full pet dataset.

    time_threshold_unix : int
        Unix timestamp threshold for birth_unix.

    cat_median_weight_kg : float
        Cat-specific median weight. Filter: weight > this.

    cat_median_height_cm : float
        Cat-specific median height. Filter: height < this.

    cat_median_daily_food_grams : float
        Cat-specific median food. Filter: food > this.

    cat_majority_can_run : bool
        Cat-specific majority class for can_run.

    Returns
    -------
    int
        Exact count of records matching ALL criteria.
    """
    boolean_mask = (
        (dataframe["animal_type"] == "cat") &
        (dataframe["birth_unix"] > time_threshold_unix) &
        (dataframe["weight_kg"] > cat_median_weight_kg) &
        (dataframe["height_cm"] < cat_median_height_cm) &
        (dataframe["daily_food_grams"] > cat_median_daily_food_grams) &
        (dataframe["can_run"] == cat_majority_can_run)
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
    query_text = "This is about a cat."

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

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    time_threshold_year : int
        Year threshold.

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

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed embeddings for all documents.

    embedding_model : SentenceTransformer
        Model for embedding the query.

    similarity_threshold : float
        Minimum similarity to count as match.

    time_threshold_year : int
        Year threshold.

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


def vector_query_e_count_cats_5_field_stress_test(
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    time_threshold_year: int,
    cat_median_weight_kg: float,
    cat_median_height_cm: float,
    cat_median_daily_food_grams: float,
    cat_majority_can_run: bool
) -> dict:
    """
    Query E Revised (Vector): 5-field stress test via embedding.

    Parameters
    ----------
    corpus_embeddings : np.ndarray
        Pre-computed document embeddings.

    embedding_model : SentenceTransformer
        Model for query embedding.

    similarity_threshold : float
        Minimum similarity to count as match.

    time_threshold_year : int
        Year for time filter in query text.

    cat_median_weight_kg : float
        Median weight for query text.

    cat_median_height_cm : float
        Median height for query text.

    cat_median_daily_food_grams : float
        Median food for query text.

    cat_majority_can_run : bool
        Whether majority of cats can run.

    Returns
    -------
    dict
        Contains count, similarities, query_text, elapsed_seconds.
    """
    if cat_majority_can_run:
        can_run_phrase = "This cat can run."
    else:
        can_run_phrase = "This cat cannot run."

    query_text = (
        f"This is about a cat, born after the year {time_threshold_year}. "
        f"This cat weighs more than {cat_median_weight_kg:.1f} kg. "
        f"This cat is shorter than {cat_median_height_cm:.1f} cm tall. "
        f"This cat eats more than {cat_median_daily_food_grams:.1f} grams of food per day. "
        f"{can_run_phrase}"
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
        Contains absolute_error, percent_error, direction, raw_difference.
    """
    absolute_error = abs(vector_count - ground_truth_count)
    raw_difference = vector_count - ground_truth_count

    if ground_truth_count > 0:
        percent_error = 100.0 * absolute_error / ground_truth_count
    else:
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


def format_metric_for_display(
    metric_value: float | None,
    format_string: str = ".4f"
) -> str:
    """
    Format a metric value for printing, handling None gracefully.

    Parameters
    ----------
    metric_value : float or None
        The metric to format.

    format_string : str
        Python format specifier (e.g. ".4f", ".2%").

    Returns
    -------
    str
        Formatted string, or "N/A" if None.
    """
    if metric_value is None:
        return "N/A"
    return f"{metric_value:{format_string}}"


def print_single_query_report(
    query_label: str,
    query_description: str,
    tabular_count: int,
    vector_result: dict,
    similarity_threshold: float,
    ground_truth_boolean_array: np.ndarray
) -> dict:
    """
    Print detailed report for a single query comparison,
    INCLUDING confusion matrix metrics.

    Also returns a summary dictionary for aggregation and CSV export.

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

    ground_truth_boolean_array : np.ndarray
        Per-row ground truth for this query. Shape (n_rows,).

    Returns
    -------
    dict
        Summary containing query_label, counts, errors, confusion matrix
        counts, and derived metrics. Used for summary table and CSV export.
    """
    vector_count = vector_result["count"]
    similarities = vector_result["similarities"]
    query_text = vector_result["query_text"]
    elapsed_seconds = vector_result["elapsed_seconds"]

    # --- Count-based error metrics (same as v3) ---
    error_metrics = calculate_error_metrics(tabular_count, vector_count)
    similarity_stats = compute_similarity_statistics(similarities)

    # --- Per-row confusion matrix ---
    # Compute vector prediction boolean array
    vector_predicted_boolean_array = similarities > similarity_threshold

    # Classify each row
    confusion_categories = classify_rows_into_confusion_categories(
        ground_truth_boolean_array=ground_truth_boolean_array,
        vector_predicted_boolean_array=vector_predicted_boolean_array
    )

    # Count TP, FP, FN, TN
    confusion_counts = compute_confusion_matrix_counts(confusion_categories)

    # Compute precision, recall, F1, accuracy, specificity
    confusion_metrics = compute_confusion_matrix_metrics(confusion_counts)

    # --- Print formatted report ---
    print(f"\n{'='*70}")
    print(f"QUERY {query_label}: {query_description}")
    print(f"{'='*70}")

    print("\n  Vector Query Text:")
    print(f"    \"{query_text}\"")

    print("\n  COUNTS:")
    print(f"    Tabular (ground truth):  {tabular_count}")
    print(f"    Vector (threshold={similarity_threshold}): {vector_count}")

    print("\n  COUNT ERROR METRICS:")
    print(f"    Absolute error:  {error_metrics['absolute_error']}")
    if error_metrics['percent_error'] is not None:
        print(f"    Percent error:   {error_metrics['percent_error']:.2f}%")
    else:
        print("    Percent error:   N/A (ground truth is 0)")
    print(f"    Direction:       {error_metrics['direction']}")
    print(f"    Raw difference:  {error_metrics['raw_difference']:+d}")

    # --- Confusion matrix section ---
    print("\n  CONFUSION MATRIX:")
    print(f"    True Positives  (TP): {confusion_counts['tp_count']:>6}  "
          f"(correctly matched)")
    print(f"    False Positives (FP): {confusion_counts['fp_count']:>6}  "
          f"(incorrectly matched, should not have been returned)")
    print(f"    False Negatives (FN): {confusion_counts['fn_count']:>6}  "
          f"(missed, should have been returned)")
    print(f"    True Negatives  (TN): {confusion_counts['tn_count']:>6}  "
          f"(correctly excluded)")
    print(f"    Total:                {confusion_counts['total_count']:>6}")

    print("\n  CONFUSION MATRIX DERIVED METRICS:")
    print(f"    Precision:   {format_metric_for_display(confusion_metrics['precision'])}  "
          f"(of matched docs, fraction that were correct)")
    print(f"    Recall:      {format_metric_for_display(confusion_metrics['recall'])}  "
          f"(of correct docs, fraction that were found)")
    print(f"    F1 Score:    {format_metric_for_display(confusion_metrics['f1_score'])}  "
          f"(harmonic mean of precision and recall)")
    print(f"    Accuracy:    {format_metric_for_display(confusion_metrics['accuracy'])}  "
          f"(fraction of all docs correctly classified)")
    print(f"    Specificity: {format_metric_for_display(confusion_metrics['specificity'])}  "
          f"(of non-matching docs, fraction correctly excluded)")

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

    # --- Build summary dict for aggregation and CSV ---
    summary = {
        "query_label": query_label,
        "query_description": query_description,
        "query_text": query_text,
        "similarity_threshold": similarity_threshold,
        "tabular_count": tabular_count,
        "vector_count": vector_count,
        "absolute_error": error_metrics["absolute_error"],
        "percent_error": error_metrics["percent_error"],
        "direction": error_metrics["direction"],
        "tp_count": confusion_counts["tp_count"],
        "fp_count": confusion_counts["fp_count"],
        "fn_count": confusion_counts["fn_count"],
        "tn_count": confusion_counts["tn_count"],
        "precision": confusion_metrics["precision"],
        "recall": confusion_metrics["recall"],
        "f1_score": confusion_metrics["f1_score"],
        "accuracy": confusion_metrics["accuracy"],
        "specificity": confusion_metrics["specificity"],
        "similarity_mean": similarity_stats["mean"],
        "similarity_std": similarity_stats["std"],
        "similarity_min": similarity_stats["min"],
        "similarity_max": similarity_stats["max"],
        "similarity_median": similarity_stats["median"],
        "elapsed_seconds": elapsed_seconds,
    }

    return summary


def print_summary_table(query_summaries: list[dict]) -> None:
    """
    Print aggregated summary table of all query results,
    including confusion matrix metrics.

    Parameters
    ----------
    query_summaries : list[dict]
        List of summary dicts from print_single_query_report.
    """
    print(f"\n{'='*120}")
    print("SUMMARY TABLE: TABULAR vs VECTOR QUERY RESULTS (WITH CONFUSION MATRIX)")
    print(f"{'='*120}")

    # Header
    print(
        f"{'Query':<6} "
        f"{'Tabular':<8} "
        f"{'Vector':<8} "
        f"{'Error':<7} "
        f"{'%Err':<8} "
        f"{'Dir':<6} "
        f"{'TP':<7} "
        f"{'FP':<7} "
        f"{'FN':<7} "
        f"{'TN':<7} "
        f"{'Prec':<7} "
        f"{'Recall':<7} "
        f"{'F1':<7} "
        f"{'Acc':<7} "
        f"{'Time(s)':<8}"
    )
    print("-" * 120)

    # Rows
    for s in query_summaries:
        percent_str = (
            f"{s['percent_error']:.1f}%"
            if s['percent_error'] is not None
            else "N/A"
        )

        print(
            f"{s['query_label']:<6} "
            f"{s['tabular_count']:<8} "
            f"{s['vector_count']:<8} "
            f"{s['absolute_error']:<7} "
            f"{percent_str:<8} "
            f"{s['direction']:<6} "
            f"{s['tp_count']:<7} "
            f"{s['fp_count']:<7} "
            f"{s['fn_count']:<7} "
            f"{s['tn_count']:<7} "
            f"{format_metric_for_display(s['precision'], '.3f'):<7} "
            f"{format_metric_for_display(s['recall'], '.3f'):<7} "
            f"{format_metric_for_display(s['f1_score'], '.3f'):<7} "
            f"{format_metric_for_display(s['accuracy'], '.3f'):<7} "
            f"{s['elapsed_seconds']:<8.4f}"
        )

    print("-" * 120)

    # Aggregate statistics
    total_tabular = sum(s["tabular_count"] for s in query_summaries)
    total_vector = sum(s["vector_count"] for s in query_summaries)
    total_absolute_error = sum(s["absolute_error"] for s in query_summaries)

    total_tp = sum(s["tp_count"] for s in query_summaries)
    total_fp = sum(s["fp_count"] for s in query_summaries)
    total_fn = sum(s["fn_count"] for s in query_summaries)
    total_tn = sum(s["tn_count"] for s in query_summaries)

    print("\nAGGREGATE METRICS:")
    print(f"  Total tabular count:     {total_tabular}")
    print(f"  Total vector count:      {total_vector}")
    print(f"  Total absolute error:    {total_absolute_error}")
    print(f"  Total TP: {total_tp}  FP: {total_fp}  FN: {total_fn}  TN: {total_tn}")


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================


def run_mvp1_test_suite(csv_file_path: str) -> list[dict]:
    """
    Execute complete MVP-1 test suite: Queries A through E.

    Workflow:
        1. Load CSV data
        2. Create timestamped output directory
        3. Determine time threshold (median of birth_unix)
        4. Load embedding model
        5. Generate embeddings for all documents
        6. Compute Query E thresholds from data
        7. Generate and save combined ground truth table
        8. Run each query (A-E) in both tabular and vector forms
        9. For each query, build and save per-row confusion matrix CSV
        10. Compare results and report errors with confusion matrix metrics
        11. Print summary table
        12. Save summary report CSV

    Parameters
    ----------
    csv_file_path : str
        Path to CSV file with pet data and unstructured_description.

    Returns
    -------
    list[dict]
        List of summary dictionaries, one per query.

    Raises
    ------
    Exception
        Propagates any errors from data loading, embedding, or queries.
    """
    print("=" * 90)
    print("MVP-1 VECTOR ANALYTICS TEST SUITE (v4: WITH CONFUSION MATRIX)")
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
    # Step 2: Create timestamped output directory
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 2: CREATE OUTPUT DIRECTORY")
    print(f"{'='*70}")

    timestamp_string, output_directory = create_timestamped_output_directory()

    # -------------------------------------------------------------------------
    # Step 3: Determine time threshold dynamically
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 3: CONFIGURE TIME THRESHOLD")
    print(f"{'='*70}")

    time_threshold_unix = int(dataframe["birth_unix"].median()) # type: ignore[arg-type]
    time_threshold_datetime = datetime.datetime.fromtimestamp(time_threshold_unix)
    time_threshold_year = time_threshold_datetime.year

    print(f"[TIME] Median birth_unix: {time_threshold_unix}")
    print(f"[TIME] Corresponds to year: {time_threshold_year}")
    print(f"[TIME] Datetime: {time_threshold_datetime.strftime('%Y-%m-%d')}")

    records_before_threshold = int((dataframe["birth_unix"] <= time_threshold_unix).sum())
    records_after_threshold = int((dataframe["birth_unix"] > time_threshold_unix).sum())
    print(f"[TIME] Records before/on threshold: {records_before_threshold}")
    print(f"[TIME] Records after threshold: {records_after_threshold}")

    # -------------------------------------------------------------------------
    # Step 4: Load embedding model
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 4: LOAD EMBEDDING MODEL")
    print(f"{'='*70}")

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

    # -------------------------------------------------------------------------
    # Step 5: Generate embeddings for all documents
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 5: GENERATE CORPUS EMBEDDINGS")
    print(f"{'='*70}")

    description_texts = dataframe["unstructured_description"].tolist()

    null_count = dataframe["unstructured_description"].isna().sum()
    if null_count > 0:
        print(f"[WARNING] Found {null_count} null descriptions, these may cause issues")

    corpus_embeddings = generate_corpus_embeddings(description_texts, embedding_model)

    # -------------------------------------------------------------------------
    # Step 6: Compute Query E thresholds from data
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 6: COMPUTE QUERY E THRESHOLDS")
    print(f"{'='*70}")

    query_e_thresholds = compute_query_e_thresholds_from_data(dataframe)

    # -------------------------------------------------------------------------
    # Step 7: Generate and save combined ground truth table
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 7: GENERATE AND SAVE GROUND TRUTH TABLE")
    print(f"{'='*70}")

    combined_ground_truth_dataframe = build_combined_ground_truth_dataframe(
        dataframe=dataframe,
        time_threshold_unix=time_threshold_unix,
        query_e_thresholds=query_e_thresholds
    )

    save_ground_truth_csv(
        ground_truth_dataframe=combined_ground_truth_dataframe,
        output_directory=output_directory,
        timestamp_string=timestamp_string
    )

    # Extract individual ground truth arrays for per-query use
    # (these are the same values that went into the combined table)
    ground_truth_a = combined_ground_truth_dataframe["query_A_ground_truth"].values
    ground_truth_b = combined_ground_truth_dataframe["query_B_ground_truth"].values
    ground_truth_c = combined_ground_truth_dataframe["query_C_ground_truth"].values
    ground_truth_d = combined_ground_truth_dataframe["query_D_ground_truth"].values
    ground_truth_e = combined_ground_truth_dataframe["query_E_ground_truth"].values

    # -------------------------------------------------------------------------
    # Step 8: Execute queries A through E
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 8: EXECUTE QUERIES A THROUGH E")
    print(f"{'='*70}")

    all_query_summaries = []

    # --- list of query configurations to iterate through ---
    # Each entry: (label, description, tabular_func, vector_func, ground_truth_array, extra_kwargs)
    # This structure avoids copy-paste repetition while keeping each query distinct.

    # ----- Query A -----
    print("\n[RUNNING] Query A: How many cats?")

    tabular_count_a = tabular_query_a_count_cats(dataframe)
    vector_result_a = vector_query_a_count_cats(
        corpus_embeddings, embedding_model, VECTOR_SIMILARITY_THRESHOLD
    )

    # Define filter conditions for Query A: single field
    query_a_filter_conditions = {
        "animal_type": ("==", "cat"),
    }

    # Build and save per-row confusion matrix with FP diagnostics
    per_row_results_a = build_per_row_confusion_matrix_dataframe(
        dataframe=dataframe,
        similarity_scores=vector_result_a["similarities"],
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        ground_truth_boolean_array=ground_truth_a,  # type: ignore[arg-type]
        query_filter_conditions=query_a_filter_conditions,
    )
    save_per_query_confusion_matrix_csv(
        per_row_results_a, "A", output_directory, timestamp_string
    )

    summary_a = print_single_query_report(
        query_label="A",
        query_description="How many cats?",
        tabular_count=tabular_count_a,
        vector_result=vector_result_a,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        ground_truth_boolean_array=ground_truth_a,  # type: ignore[arg-type]
    )
    all_query_summaries.append(summary_a)

    # FP diagnostic summary for Query A
    summarize_fp_diagnostic_columns(
        fp_diagnostic_dataframe=per_row_results_a,
        confusion_categories=per_row_results_a["confusion_class"].values, # type: ignore[arg-type]
        query_label="A",
        output_directory=output_directory,
        timestamp_string=timestamp_string,
    )

    # ----- Query B -----
    print("\n[RUNNING] Query B: How many cats that can fly?")

    tabular_count_b = tabular_query_b_count_cats_that_can_fly(dataframe)
    vector_result_b = vector_query_b_count_cats_that_can_fly(
        corpus_embeddings, embedding_model, VECTOR_SIMILARITY_THRESHOLD
    )

    # Define filter conditions for Query B: animal type + one boolean
    query_b_filter_conditions = {
        "animal_type": ("==", "cat"),
        "can_fly": ("==", True),
    }

    per_row_results_b = build_per_row_confusion_matrix_dataframe(
        dataframe=dataframe,
        similarity_scores=vector_result_b["similarities"],
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        ground_truth_boolean_array=ground_truth_b,  # type: ignore[arg-type]
        query_filter_conditions=query_b_filter_conditions,
    )
    save_per_query_confusion_matrix_csv(
        per_row_results_b, "B", output_directory, timestamp_string
    )

    summary_b = print_single_query_report(
        query_label="B",
        query_description="How many cats that can fly?",
        tabular_count=tabular_count_b,
        vector_result=vector_result_b,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        ground_truth_boolean_array=ground_truth_b,  # type: ignore[arg-type]
    )
    all_query_summaries.append(summary_b)

    summarize_fp_diagnostic_columns(
        fp_diagnostic_dataframe=per_row_results_b,
        confusion_categories=per_row_results_b["confusion_class"].values, # type: ignore[arg-type]
        query_label="B",
        output_directory=output_directory,
        timestamp_string=timestamp_string,
    )

    # ----- Query C -----
    print(f"\n[RUNNING] Query C: How many cats born after {time_threshold_year}?")

    tabular_count_c = tabular_query_c_count_cats_born_after_time(
        dataframe, time_threshold_unix
    )
    vector_result_c = vector_query_c_count_cats_born_after_time(
        corpus_embeddings, embedding_model, VECTOR_SIMILARITY_THRESHOLD,
        time_threshold_year
    )

    # Define filter conditions for Query C: animal type + time
    query_c_filter_conditions = {
        "animal_type": ("==", "cat"),
        "birth_unix": (">", time_threshold_unix),
    }

    per_row_results_c = build_per_row_confusion_matrix_dataframe(
        dataframe=dataframe,
        similarity_scores=vector_result_c["similarities"],
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        ground_truth_boolean_array=ground_truth_c,  # type: ignore[arg-type]
        query_filter_conditions=query_c_filter_conditions,
    )
    save_per_query_confusion_matrix_csv(
        per_row_results_c, "C", output_directory, timestamp_string
    )

    summary_c = print_single_query_report(
        query_label="C",
        query_description=f"How many cats born after {time_threshold_year}?",
        tabular_count=tabular_count_c,
        vector_result=vector_result_c,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        ground_truth_boolean_array=ground_truth_c,  # type: ignore[arg-type]
    )
    all_query_summaries.append(summary_c)

    summarize_fp_diagnostic_columns(
        fp_diagnostic_dataframe=per_row_results_c,
        confusion_categories=per_row_results_c["confusion_class"].values, # type: ignore[arg-type]
        query_label="C",
        output_directory=output_directory,
        timestamp_string=timestamp_string,
    )

    # ----- Query D -----
    print(f"\n[RUNNING] Query D: Cats born after {time_threshold_year} that can fly?")

    tabular_count_d = tabular_query_d_count_cats_born_after_time_can_fly(
        dataframe, time_threshold_unix
    )
    vector_result_d = vector_query_d_count_cats_born_after_time_can_fly(
        corpus_embeddings, embedding_model, VECTOR_SIMILARITY_THRESHOLD,
        time_threshold_year
    )

    # Define filter conditions for Query D: animal type + time + one boolean
    query_d_filter_conditions = {
        "animal_type": ("==", "cat"),
        "birth_unix": (">", time_threshold_unix),
        "can_fly": ("==", True),
    }

    per_row_results_d = build_per_row_confusion_matrix_dataframe(
        dataframe=dataframe,
        similarity_scores=vector_result_d["similarities"],
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        ground_truth_boolean_array=ground_truth_d,  # type: ignore[arg-type]
        query_filter_conditions=query_d_filter_conditions,
    )
    save_per_query_confusion_matrix_csv(
        per_row_results_d, "D", output_directory, timestamp_string
    )

    summary_d = print_single_query_report(
        query_label="D",
        query_description=f"Cats born after {time_threshold_year} that can fly?",
        tabular_count=tabular_count_d,
        vector_result=vector_result_d,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        ground_truth_boolean_array=ground_truth_d, # type: ignore[arg-type]
    )
    all_query_summaries.append(summary_d)

    summarize_fp_diagnostic_columns(
        fp_diagnostic_dataframe=per_row_results_d,
        confusion_categories=per_row_results_d["confusion_class"].values, # type: ignore[arg-type]
        query_label="D",
        output_directory=output_directory,
        timestamp_string=timestamp_string,
    )

    # ----- Query E -----
    print("\n[RUNNING] Query E: Stress test (cat + time + 5 data-derived fields)")

    print("  Query E Configuration (data-derived):")
    print(f"    time_threshold_unix: {time_threshold_unix}")
    print(f"    time_threshold_year: {time_threshold_year}")
    print(f"    weight_kg > {query_e_thresholds['cat_median_weight_kg']:.2f} (cat median)")
    print(f"    height_cm < {query_e_thresholds['cat_median_height_cm']:.2f} (cat median)")
    print(f"    daily_food_grams > {query_e_thresholds['cat_median_daily_food_grams']:.2f} (cat median)")
    print(f"    can_run == {query_e_thresholds['cat_majority_can_run']} (cat majority class)")

    tabular_count_e = tabular_query_e_count_cats_5_field_stress_test(
        dataframe=dataframe,
        time_threshold_unix=time_threshold_unix,
        cat_median_weight_kg=query_e_thresholds["cat_median_weight_kg"],
        cat_median_height_cm=query_e_thresholds["cat_median_height_cm"],
        cat_median_daily_food_grams=query_e_thresholds["cat_median_daily_food_grams"],
        cat_majority_can_run=query_e_thresholds["cat_majority_can_run"],
    )

    print(f"  [GROUND TRUTH CHECK] Tabular count for Query E: {tabular_count_e}")
    if tabular_count_e == 0:
        print("  [WARNING] Ground truth is 0. Vector comparison will not be meaningful.")

    vector_result_e = vector_query_e_count_cats_5_field_stress_test(
        corpus_embeddings=corpus_embeddings,
        embedding_model=embedding_model,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        time_threshold_year=time_threshold_year,
        cat_median_weight_kg=query_e_thresholds["cat_median_weight_kg"],
        cat_median_height_cm=query_e_thresholds["cat_median_height_cm"],
        cat_median_daily_food_grams=query_e_thresholds["cat_median_daily_food_grams"],
        cat_majority_can_run=query_e_thresholds["cat_majority_can_run"],
    )

    # Define filter conditions for Query E: 6 fields (animal + time + 4 numeric/boolean)
    query_e_filter_conditions = {
        "animal_type": ("==", "cat"),
        "birth_unix": (">", time_threshold_unix),
        "weight_kg": (">", query_e_thresholds["cat_median_weight_kg"]),
        "height_cm": ("<", query_e_thresholds["cat_median_height_cm"]),
        "daily_food_grams": (">", query_e_thresholds["cat_median_daily_food_grams"]),
        "can_run": ("==", query_e_thresholds["cat_majority_can_run"]),
    }

    per_row_results_e = build_per_row_confusion_matrix_dataframe(
        dataframe=dataframe,
        similarity_scores=vector_result_e["similarities"],
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        ground_truth_boolean_array=ground_truth_e,  # type: ignore[arg-type]
        query_filter_conditions=query_e_filter_conditions,
    )
    save_per_query_confusion_matrix_csv(
        per_row_results_e, "E", output_directory, timestamp_string
    )

    summary_e = print_single_query_report(
        query_label="E",
        query_description="Stress test: cat + time + 5 data-derived fields",
        tabular_count=tabular_count_e,
        vector_result=vector_result_e,
        similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
        ground_truth_boolean_array=ground_truth_e,  # type: ignore[arg-type]
    )
    all_query_summaries.append(summary_e)

    summarize_fp_diagnostic_columns(
        fp_diagnostic_dataframe=per_row_results_e,
        confusion_categories=per_row_results_e["confusion_class"].values, # type: ignore[arg-type]
        query_label="E",
        output_directory=output_directory,
        timestamp_string=timestamp_string,
    )

    # -------------------------------------------------------------------------
    # Step 9: Print summary table
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 9: SUMMARY")
    print(f"{'='*70}")

    print_summary_table(all_query_summaries)

    # -------------------------------------------------------------------------
    # Step 10: Save summary report CSV
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 10: SAVE SUMMARY REPORT")
    print(f"{'='*70}")

    save_summary_report_csv(
        all_query_summaries=all_query_summaries,
        output_directory=output_directory,
        timestamp_string=timestamp_string
    )

    # -------------------------------------------------------------------------
    # Step 11: Print configuration used (for reproducibility)
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
    print(f"  Output directory: {output_directory}")
    print(f"  Timestamp: {timestamp_string}")

    # List all saved files for audit trail
    print("\n  Saved files:")
    for saved_file in sorted(output_directory.iterdir()):
        print(f"    {saved_file.name}")

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
    print("STARTING MVP-1 VECTOR ANALYTICS TEST SUITE (v4: WITH CONFUSION MATRIX)")
    print("=" * 90 + "\n")

    try:
        # Run the test suite
        query_summaries = run_mvp1_test_suite(CSV_FILE_PATH)

        print(f"\nquery_summaries: {query_summaries}")

        print("\n" + "=" * 90)
        print("TEST SUITE COMPLETED SUCCESSFULLY")
        print("=" * 90)

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
