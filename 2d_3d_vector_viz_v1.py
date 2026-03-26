# -*- coding: utf-8 -*-
"""2d_3d_vector_viz_v1.ipynb
Created 
Mar 8, 2026 with Google Colaboratory
"""

# vector_visualization.py
"""
vector_visualization.py

Plotly HTML Visualization of Vector Space for Query Analysis

Purpose:
    Visualize corpus embeddings projected to 2D/3D via PCA.
    Color documents by confusion class (TP/FP/FN/TN) for single-query plots.
    Color documents by match category for comparative (cat vs dog) plots.
    Mark query vector position in the projected space.

## Steps to Run:
    1. Run in python venv environment:
    ```bash
    python vector_visualization.py
    ```

    2. You will be prompted for:
        - Path to CSV data file
        - Similarity threshold

    3. HTML plots saved to "visualizations/viz_{timestamp}/" directory

## Output Files (per query A-D, plus comparative):
    - A_pca_2d_{timestamp}.html
    - A_pca_3d_{timestamp}.html
    - B_pca_2d_{timestamp}.html
    - B_pca_3d_{timestamp}.html
    - C_pca_2d_{timestamp}.html
    - C_pca_3d_{timestamp}.html
    - D_pca_2d_{timestamp}.html
    - D_pca_3d_{timestamp}.html
    - comparative_cat_vs_dog_pca_2d_{timestamp}.html
    - comparative_cat_vs_dog_pca_3d_{timestamp}.html
    - pca_explained_variance_{timestamp}.html

## Key Design Decisions:
    - PCA implemented with numpy only (no sklearn dependency)
    - PCA fitted once on full corpus, query vectors projected into same space
    - TN points subsampled to avoid dominating the plot (configurable)
    - Cosine similarity limitation is reported in plot title/subtitle
    - Hover text includes: row_index, animal_type, similarity_score, confusion_class

## Interpretation Warning (displayed in plots):
    PCA projects 384D -> 2D/3D. Euclidean distance in this plot does NOT equal
    cosine similarity. Treat as exploratory visualization only.

## Dependencies:
    - pandas
    - numpy
    - sentence-transformers
    - plotly  (pip install plotly)

Author: Vector Visualization Module
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

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as plotly_import_error:
    print(
        f"[FATAL] plotly is required for visualization.\n"
        f"Install with: pip install plotly\n"
        f"Error: {plotly_import_error}"
    )
    raise


# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
RANDOM_SEED: int = 42

# Maximum number of TN (True Negative) points to display.
# TN often dominates (~16000 of 20000). Subsampling keeps plots readable.
# All TP, FP, FN are always shown (these are the interesting cases).
MAX_TN_DISPLAY_COUNT: int = 2000

# PCA components to compute (need 3 for 3D plots)
PCA_N_COMPONENTS: int = 3

# Color maps (fixed, not configurable, for consistency across runs)
CONFUSION_CLASS_COLOR_MAP: dict[str, str] = {
    "TP": "#2ecc71",   # green   - correctly matched
    "FP": "#e74c3c",   # red     - incorrectly matched
    "FN": "#f39c12",   # orange  - missed
    "TN": "#bdc3c7",   # gray    - correctly excluded (subsampled)
}

CONFUSION_CLASS_OPACITY_MAP: dict[str, float] = {
    "TP": 0.85,
    "FP": 0.85,
    "FN": 0.85,
    "TN": 0.25,   # TN gets low opacity: large group, less interesting
}

COMPARATIVE_MATCH_COLOR_MAP: dict[str, str] = {
    "neither":  "#95a5a6",  # gray   - matched by neither query
    "cat_only": "#3498db",  # blue   - matched only by cat query
    "dog_only": "#e67e22",  # orange - matched only by dog query
    "both":     "#9b59b6",  # purple - matched by both queries
}

COMPARATIVE_MATCH_OPACITY_MAP: dict[str, float] = {
    "neither":  0.20,
    "cat_only": 0.85,
    "dog_only": 0.85,
    "both":     0.90,
}

ANIMAL_TYPE_COLOR_MAP: dict[str, str] = {
    "cat":    "#3498db",   # blue
    "dog":    "#e67e22",   # orange
    "bird":   "#2ecc71",   # green
    "fish":   "#e74c3c",   # red
    "turtle": "#9b59b6",   # purple
}


# =============================================================================
# TIMESTAMP AND OUTPUT DIRECTORY
# =============================================================================


def create_visualization_output_directory() -> tuple[str, Path]:
    """
    Create timestamped output directory for HTML visualization files.

    Directory structure:
        visualizations/viz_{YYYYMMDD_HHMMSS}/

    Returns
    -------
    tuple[str, Path]
        - timestamp_string
        - output_directory_path
    """
    try:
        now = datetime.datetime.now()
        timestamp_string = now.strftime("%Y%m%d_%H%M%S")

        output_directory_path = Path.cwd() / f"visualizations/viz_{timestamp_string}"
        output_directory_path.mkdir(parents=True, exist_ok=True)

        print(f"[VIZ OUTPUT] Created directory: {output_directory_path}")

        return timestamp_string, output_directory_path

    except Exception as directory_error:
        print(f"[ERROR] Failed to create visualization output directory: {directory_error}")
        traceback.print_exc()
        raise


# =============================================================================
# DATA LOADING (same as other scripts, standalone)
# =============================================================================


def load_csv_data(csv_file_path: str) -> pd.DataFrame:
    """
    Load synthetic pet data CSV into pandas DataFrame.

    Parameters
    ----------
    csv_file_path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Dataset with all fields.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If required columns are missing.
    """
    try:
        file_path = Path(csv_file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_file_path}\n"
                f"CWD: {Path.cwd()}"
            )

        dataframe = pd.read_csv(file_path)

        required_columns = [
            "animal_type",
            "birth_unix",
            "can_fly",
            "unstructured_description",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"CSV missing required columns: {missing_columns}"
            )

        print(f"[DATA] Loaded {len(dataframe)} rows from {csv_file_path}")

        animal_counts = dataframe["animal_type"].value_counts()
        print("[DATA] Animal type distribution:")
        for animal_type, count in animal_counts.items():
            print(f"         {animal_type}: {count}")

        return dataframe

    except (FileNotFoundError, ValueError):
        raise
    except Exception as unexpected_error:
        print(f"[ERROR] Unexpected data load error: {unexpected_error}")
        traceback.print_exc()
        raise


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load sentence-transformer model.

    Parameters
    ----------
    model_name : str
        Model name, e.g. 'all-MiniLM-L6-v2'.

    Returns
    -------
    SentenceTransformer
        Loaded model.
    """
    try:
        print(f"[MODEL] Loading: {model_name}")
        start_time = time.time()
        model = SentenceTransformer(model_name)
        print(f"[MODEL] Loaded in {time.time() - start_time:.2f}s")
        return model

    except Exception as model_error:
        print(f"[ERROR] Failed to load model: {model_error}")
        traceback.print_exc()
        raise


def generate_corpus_embeddings(
    text_documents: list[str],
    embedding_model: SentenceTransformer
) -> np.ndarray:
    """
    Generate embeddings for all corpus documents.

    Parameters
    ----------
    text_documents : list[str]
        List of text strings.

    embedding_model : SentenceTransformer
        Pre-loaded model.

    Returns
    -------
    np.ndarray
        Shape (n_documents, embedding_dim).
    """
    try:
        print(f"[EMBED] Generating embeddings for {len(text_documents)} documents...")
        start_time = time.time()

        embeddings_matrix = embedding_model.encode(
            text_documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"[EMBED] Done in {time.time() - start_time:.2f}s, shape: {embeddings_matrix.shape}")

        return embeddings_matrix

    except Exception as embedding_error:
        print(f"[ERROR] Embedding generation failed: {embedding_error}")
        traceback.print_exc()
        raise


def embed_single_query_text(
    query_text: str,
    embedding_model: SentenceTransformer
) -> np.ndarray:
    """
    Embed a single query string to a 1D vector.

    Parameters
    ----------
    query_text : str
        Query string to embed.

    embedding_model : SentenceTransformer
        Pre-loaded model.

    Returns
    -------
    np.ndarray
        Shape (embedding_dim,).
    """
    embedding_2d = embedding_model.encode(
        [query_text],
        convert_to_numpy=True
    )
    return embedding_2d[0]


# =============================================================================
# COSINE SIMILARITY
# =============================================================================


def calculate_cosine_similarities(
    query_vector: np.ndarray,
    corpus_embeddings_matrix: np.ndarray
) -> np.ndarray:
    """
    Exhaustive cosine similarity: query vs all corpus documents.

    Parameters
    ----------
    query_vector : np.ndarray
        Shape (embedding_dim,).

    corpus_embeddings_matrix : np.ndarray
        Shape (n_documents, embedding_dim).

    Returns
    -------
    np.ndarray
        Similarity scores, shape (n_documents,). Range [-1, 1].
    """
    query_flat = query_vector.flatten()
    dot_products = corpus_embeddings_matrix @ query_flat
    corpus_norms = norm(corpus_embeddings_matrix, axis=1)
    query_norm = norm(query_flat)

    denominator = corpus_norms * query_norm
    epsilon = 1e-10
    denominator_safe = np.where(denominator == 0, epsilon, denominator)

    return dot_products / denominator_safe


# =============================================================================
# PCA: NUMPY IMPLEMENTATION (NO SKLEARN)
# =============================================================================


def compute_pca_on_corpus_embeddings(
    corpus_embeddings_matrix: np.ndarray,
    n_components: int = 3
) -> dict:
    """
    Fit PCA on corpus embeddings using numpy only (no sklearn required).

    Uses eigendecomposition of the covariance matrix.
    For a (20000, 384) corpus, the covariance matrix is (384, 384),
    which is small enough to compute exactly.

    Parameters
    ----------
    corpus_embeddings_matrix : np.ndarray
        Shape (n_documents, embedding_dim). The full corpus.

    n_components : int
        Number of principal components to compute.
        Must be <= embedding_dim. Default 3 (for 2D and 3D plots).

    Returns
    -------
    dict
        Contains:
        - projected_corpus: np.ndarray, shape (n_documents, n_components)
            Each document projected into PCA space.
        - pca_components: np.ndarray, shape (embedding_dim, n_components)
            Principal component vectors (columns).
        - corpus_mean_vector: np.ndarray, shape (embedding_dim,)
            Mean of corpus, used to center new vectors.
        - explained_variance_ratio: np.ndarray, shape (n_components,)
            Fraction of total variance explained by each component.
        - cumulative_variance_explained: float
            Total variance explained by all n_components together.

    Notes
    -----
    np.linalg.eigh returns eigenvalues in ASCENDING order.
    We sort descending to get principal components in order of importance.

    The query vector is NOT included in the PCA fit.
    PCA is fitted on corpus only, then query vectors are projected
    into the same space using project_query_vector_into_pca_space().
    This is the correct approach: PCA defines the space, queries are visitors.
    """
    try:
        n_documents, embedding_dim = corpus_embeddings_matrix.shape
        print(f"[PCA] Fitting PCA on corpus shape {corpus_embeddings_matrix.shape}")
        print(f"[PCA] Requesting {n_components} components")

        if n_components > embedding_dim:
            raise ValueError(
                f"n_components ({n_components}) cannot exceed "
                f"embedding_dim ({embedding_dim})."
            )

        start_time = time.time()

        # --- Step 1: Center the data ---
        corpus_mean_vector = np.mean(corpus_embeddings_matrix, axis=0)
        centered_corpus = corpus_embeddings_matrix - corpus_mean_vector

        # --- Step 2: Compute covariance matrix ---
        # centered_corpus.T has shape (384, 20000)
        # np.cov computes (384, 384) covariance matrix
        # This is manageable even for large n_documents
        covariance_matrix = np.cov(centered_corpus.T)

        # --- Step 3: Eigendecomposition ---
        # eigh is for symmetric matrices (covariance matrices are symmetric)
        # Returns eigenvalues in ASCENDING order
        eigenvalues_ascending, eigenvectors_ascending = np.linalg.eigh(covariance_matrix)

        # --- Step 4: Sort DESCENDING (largest variance first) ---
        descending_sort_indices = np.argsort(eigenvalues_ascending)[::-1]
        eigenvalues_descending = eigenvalues_ascending[descending_sort_indices]
        # eigenvectors columns correspond to eigenvalues
        eigenvectors_descending = eigenvectors_ascending[:, descending_sort_indices]

        # --- Step 5: Select top n_components ---
        top_pca_components = eigenvectors_descending[:, :n_components]

        # --- Step 6: Project corpus into PCA space ---
        # (n_documents, embedding_dim) @ (embedding_dim, n_components)
        #   = (n_documents, n_components)
        projected_corpus = centered_corpus @ top_pca_components

        # --- Step 7: Compute explained variance ratios ---
        # Only sum positive eigenvalues for total variance
        positive_eigenvalues = eigenvalues_descending[eigenvalues_descending > 0]
        total_variance = float(np.sum(positive_eigenvalues))

        top_eigenvalues = eigenvalues_descending[:n_components]
        explained_variance_ratio = np.clip(
            top_eigenvalues / total_variance,
            0.0,
            1.0
        )
        cumulative_variance_explained = float(np.sum(explained_variance_ratio))

        elapsed_seconds = time.time() - start_time

        # Report
        print(f"[PCA] Completed in {elapsed_seconds:.2f}s")
        for component_index in range(n_components):
            pct = explained_variance_ratio[component_index] * 100
            print(f"[PCA]   PC{component_index + 1}: {pct:.2f}% variance explained")
        print(f"[PCA]   Cumulative ({n_components} components): "
              f"{cumulative_variance_explained * 100:.2f}%")
        print(f"[PCA]   NOTE: {(1.0 - cumulative_variance_explained) * 100:.1f}% "
              f"of variance is NOT shown in 2D/3D projection")

        return {
            "projected_corpus": projected_corpus,
            "pca_components": top_pca_components,
            "corpus_mean_vector": corpus_mean_vector,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance_explained": cumulative_variance_explained,
        }

    except Exception as pca_error:
        print(f"[ERROR] PCA computation failed: {pca_error}")
        traceback.print_exc()
        raise


def project_query_vector_into_pca_space(
    query_vector: np.ndarray,
    pca_result: dict
) -> np.ndarray:
    """
    Project a single query vector into the fitted PCA space.

    The query vector is centered using the CORPUS mean (not its own mean),
    then projected using the corpus PCA components. This correctly places
    the query relative to the corpus in PCA space.

    Parameters
    ----------
    query_vector : np.ndarray
        Shape (embedding_dim,). The embedded query.

    pca_result : dict
        Output of compute_pca_on_corpus_embeddings().
        Must contain 'corpus_mean_vector' and 'pca_components'.

    Returns
    -------
    np.ndarray
        Shape (n_components,). Query coordinates in PCA space.
    """
    # Center using corpus mean (not query's own mean)
    corpus_mean = pca_result["corpus_mean_vector"]
    pca_components = pca_result["pca_components"]

    centered_query = query_vector - corpus_mean
    projected_query = centered_query @ pca_components

    return projected_query


# =============================================================================
# SUBSAMPLING UTILITY
# =============================================================================


def subsample_tn_indices(
    confusion_class_array: np.ndarray,
    max_tn_display_count: int,
    random_seed: int
) -> np.ndarray:
    """
    Subsample TN indices to keep plot readable.

    TP, FP, FN are always shown (small counts, interesting cases).
    TN can number in the tens of thousands and would dominate the plot.

    Parameters
    ----------
    confusion_class_array : np.ndarray
        String array with values "TP", "FP", "FN", "TN".

    max_tn_display_count : int
        Maximum TN rows to display.

    random_seed : int
        Random seed for reproducible subsampling.

    Returns
    -------
    np.ndarray
        Boolean mask, shape (n_rows,). True = include in plot.
        All TP/FP/FN rows are True. TN rows are True up to max_tn_display_count.
    """
    rng = np.random.default_rng(seed=random_seed)

    # Include all non-TN rows
    non_tn_mask = confusion_class_array != "TN"

    # TN indices
    tn_indices = np.where(confusion_class_array == "TN")[0]

    tn_count = len(tn_indices)

    if tn_count <= max_tn_display_count:
        # All TN fit within limit
        include_mask = np.ones(len(confusion_class_array), dtype=bool)
        print(f"[SUBSAMPLE] All {tn_count} TN rows included (within limit {max_tn_display_count})")
    else:
        # Randomly subsample TN
        sampled_tn_indices = rng.choice(
            tn_indices,
            size=max_tn_display_count,
            replace=False
        )
        include_mask = non_tn_mask.copy()
        include_mask[sampled_tn_indices] = True
        print(f"[SUBSAMPLE] TN subsampled: {max_tn_display_count} of {tn_count} shown")

    return include_mask


# =============================================================================
# VISUALIZATION DATAFRAME BUILDERS
# =============================================================================


def build_single_query_visualization_dataframe(
    dataframe: pd.DataFrame,
    pca_result: dict,
    similarities: np.ndarray,
    ground_truth_boolean_array: np.ndarray,
    similarity_threshold: float,
    max_tn_display_count: int,
    random_seed: int
) -> pd.DataFrame:
    """
    Build a plotting DataFrame for a single query visualization.

    Each row is one document. Includes PCA coordinates,
    confusion class, similarity score, animal type, hover text.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Original dataset. Must contain 'animal_type'.
        Optionally contains 'unique_id'.

    pca_result : dict
        Output of compute_pca_on_corpus_embeddings().
        Contains 'projected_corpus' with shape (n_docs, n_components).

    similarities : np.ndarray
        Cosine similarity scores for all documents. Shape (n_docs,).

    ground_truth_boolean_array : np.ndarray
        Boolean ground truth for this query. Shape (n_docs,).

    similarity_threshold : float
        Match threshold.

    max_tn_display_count : int
        Maximum TN rows to include.

    random_seed : int
        Seed for TN subsampling reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:
        - pc1, pc2, pc3: float (PCA coordinates)
        - similarity_score: float
        - ground_truth: bool
        - vector_predicted: bool
        - confusion_class: str (TP/FP/FN/TN)
        - animal_type: str
        - hover_text: str (for plotly hover)
        - plot_color: str (hex color)
        - plot_opacity: float
    """
    projected_corpus = pca_result["projected_corpus"]
    n_components = projected_corpus.shape[1]

    # Compute predictions and confusion classes
    vector_predicted = similarities > similarity_threshold

    # Classify each row
    confusion_classes = np.empty(len(similarities), dtype="U2")
    confusion_classes[ground_truth_boolean_array & vector_predicted] = "TP"
    confusion_classes[(~ground_truth_boolean_array) & vector_predicted] = "FP"
    confusion_classes[ground_truth_boolean_array & (~vector_predicted)] = "FN"
    confusion_classes[(~ground_truth_boolean_array) & (~vector_predicted)] = "TN"

    # Subsample TN
    include_mask = subsample_tn_indices(
        confusion_class_array=confusion_classes,
        max_tn_display_count=max_tn_display_count,
        random_seed=random_seed
    )

    # Build plotting DataFrame
    viz_df = pd.DataFrame({
        "pc1": projected_corpus[:, 0],
        "pc2": projected_corpus[:, 1],
        "pc3": projected_corpus[:, 2] if n_components >= 3 else 0.0,
        "similarity_score": similarities,
        "ground_truth": ground_truth_boolean_array,
        "vector_predicted": vector_predicted,
        "confusion_class": confusion_classes,
        "animal_type": dataframe["animal_type"].values,
    })

    # Add unique_id if available
    if "unique_id" in dataframe.columns:
        viz_df["unique_id"] = dataframe["unique_id"].values
    else:
        viz_df["unique_id"] = np.arange(len(dataframe))

    # Add hover text
    viz_df["hover_text"] = (
        "ID: " + viz_df["unique_id"].astype(str) + "<br>"
        + "Animal: " + viz_df["animal_type"] + "<br>"
        + "Similarity: " + viz_df["similarity_score"].round(4).astype(str) + "<br>"
        + "Confusion: " + viz_df["confusion_class"] + "<br>"
        + "Ground truth: " + viz_df["ground_truth"].astype(str)
    )

    # Map colors and opacity
    viz_df["plot_color"] = viz_df["confusion_class"].map(CONFUSION_CLASS_COLOR_MAP)
    viz_df["plot_opacity"] = viz_df["confusion_class"].map(CONFUSION_CLASS_OPACITY_MAP)

    # Apply subsample mask
    viz_df = viz_df[include_mask].copy()

    print(f"[VIZ DF] Single query: {len(viz_df)} rows "
          f"(TP={int((viz_df['confusion_class'] == 'TP').sum())}, "
          f"FP={int((viz_df['confusion_class'] == 'FP').sum())}, "
          f"FN={int((viz_df['confusion_class'] == 'FN').sum())}, "
          f"TN={int((viz_df['confusion_class'] == 'TN').sum())} displayed)")

    return viz_df


def build_comparative_visualization_dataframe(
    dataframe: pd.DataFrame,
    pca_result: dict,
    similarities_cat: np.ndarray,
    similarities_dog: np.ndarray,
    ground_truth_cat: np.ndarray,
    ground_truth_dog: np.ndarray,
    similarity_threshold: float,
    max_neither_display_count: int,
    random_seed: int
) -> pd.DataFrame:
    """
    Build a plotting DataFrame for comparative (cat vs dog) visualization.

    Each row is labeled by match category:
    - "cat_only": matched by cat query, not dog query
    - "dog_only": matched by dog query, not cat query
    - "both": matched by both queries
    - "neither": matched by neither query (subsampled)

    Parameters
    ----------
    dataframe : pd.DataFrame
        Original dataset.

    pca_result : dict
        Output of compute_pca_on_corpus_embeddings().

    similarities_cat : np.ndarray
        Cat query similarities. Shape (n_docs,).

    similarities_dog : np.ndarray
        Dog query similarities. Shape (n_docs,).

    ground_truth_cat : np.ndarray
        Boolean ground truth for cat query.

    ground_truth_dog : np.ndarray
        Boolean ground truth for dog query.

    similarity_threshold : float
        Match threshold.

    max_neither_display_count : int
        Maximum "neither" rows to display.

    random_seed : int
        Seed for subsampling.

    Returns
    -------
    pd.DataFrame
        Plotting DataFrame with PCA coords, match category, hover text.
    """
    projected_corpus = pca_result["projected_corpus"]
    n_components = projected_corpus.shape[1]

    # Compute predictions
    predicted_cat = similarities_cat > similarity_threshold
    predicted_dog = similarities_dog > similarity_threshold

    # Assign match category
    match_category = np.empty(len(similarities_cat), dtype="U10")
    match_category[predicted_cat & predicted_dog] = "both"
    match_category[predicted_cat & (~predicted_dog)] = "cat_only"
    match_category[(~predicted_cat) & predicted_dog] = "dog_only"
    match_category[(~predicted_cat) & (~predicted_dog)] = "neither"

    # Subsample "neither" (equivalent to TN in single query)
    rng = np.random.default_rng(seed=random_seed)
    neither_indices = np.where(match_category == "neither")[0]
    non_neither_mask = match_category != "neither"

    if len(neither_indices) <= max_neither_display_count:
        include_mask = np.ones(len(match_category), dtype=bool)
    else:
        sampled_neither = rng.choice(
            neither_indices,
            size=max_neither_display_count,
            replace=False
        )
        include_mask = non_neither_mask.copy()
        include_mask[sampled_neither] = True

    # Build DataFrame
    viz_df = pd.DataFrame({
        "pc1": projected_corpus[:, 0],
        "pc2": projected_corpus[:, 1],
        "pc3": projected_corpus[:, 2] if n_components >= 3 else 0.0,
        "similarity_cat": similarities_cat,
        "similarity_dog": similarities_dog,
        "ground_truth_cat": ground_truth_cat,
        "ground_truth_dog": ground_truth_dog,
        "predicted_cat": predicted_cat,
        "predicted_dog": predicted_dog,
        "match_category": match_category,
        "animal_type": dataframe["animal_type"].values,
    })

    if "unique_id" in dataframe.columns:
        viz_df["unique_id"] = dataframe["unique_id"].values
    else:
        viz_df["unique_id"] = np.arange(len(dataframe))

    # Hover text
    viz_df["hover_text"] = (
        "ID: " + viz_df["unique_id"].astype(str) + "<br>"
        + "Animal: " + viz_df["animal_type"] + "<br>"
        + "Cat sim: " + viz_df["similarity_cat"].round(4).astype(str) + "<br>"
        + "Dog sim: " + viz_df["similarity_dog"].round(4).astype(str) + "<br>"
        + "Match: " + viz_df["match_category"]
    )

    # Colors and opacity
    viz_df["plot_color"] = viz_df["match_category"].map(COMPARATIVE_MATCH_COLOR_MAP)
    viz_df["plot_opacity"] = viz_df["match_category"].map(COMPARATIVE_MATCH_OPACITY_MAP)

    # Apply subsample mask
    viz_df = viz_df[include_mask].copy()

    print(f"[VIZ DF] Comparative: {len(viz_df)} rows displayed "
          f"(cat_only={int((viz_df['match_category']=='cat_only').sum())}, "
          f"dog_only={int((viz_df['match_category']=='dog_only').sum())}, "
          f"both={int((viz_df['match_category']=='both').sum())}, "
          f"neither={int((viz_df['match_category']=='neither').sum())} displayed)")

    return viz_df


# =============================================================================
# PLOTLY FIGURE CREATION
# =============================================================================


def create_single_query_pca_2d_figure(
    viz_df: pd.DataFrame,
    projected_query_vector: np.ndarray,
    pca_result: dict,
    query_text: str,
    test_label: str,
    test_description: str,
    similarity_threshold: float,
    tabular_count: int,
    vector_count: int
) -> go.Figure:
    """
    Create 2D PCA scatter plot for a single query.

    Documents are colored by confusion class (TP/FP/FN/TN).
    The query vector is plotted as a large star marker.

    Parameters
    ----------
    viz_df : pd.DataFrame
        Output of build_single_query_visualization_dataframe().

    projected_query_vector : np.ndarray
        Shape (n_components,). Query vector in PCA space.
        Output of project_query_vector_into_pca_space().

    pca_result : dict
        Contains explained_variance_ratio.

    query_text : str
        Original query string (shown in plot).

    test_label : str
        Short label e.g. "A".

    test_description : str
        Human-readable description.

    similarity_threshold : float
        Threshold used for matching.

    tabular_count : int
        Ground truth count.

    vector_count : int
        Vector query count.

    Returns
    -------
    go.Figure
        Plotly figure object (not yet saved).
    """
    pc1_variance = pca_result["explained_variance_ratio"][0] * 100
    pc2_variance = pca_result["explained_variance_ratio"][1] * 100
    cumulative_variance = pca_result["cumulative_variance_explained"] * 100

    # Build figure using go.Figure with scatter traces
    # (Using go directly rather than px to have full control over each
    #  confusion class's color and opacity independently)
    fig = go.Figure()

    # Add one trace per confusion class (for clean legend)
    for class_label in ["TN", "FN", "FP", "TP"]:
        # TN drawn first (bottom), TP drawn last (top)
        class_mask = viz_df["confusion_class"] == class_label
        class_df = viz_df[class_mask]

        if len(class_df) == 0:
            continue

        color_hex = CONFUSION_CLASS_COLOR_MAP[class_label]
        opacity = CONFUSION_CLASS_OPACITY_MAP[class_label]

        fig.add_trace(go.Scatter(
            x=class_df["pc1"],
            y=class_df["pc2"],
            mode="markers",
            name=class_label,
            marker=dict(
                color=color_hex,
                opacity=opacity,
                size=5,
                line=dict(width=0)
            ),
            text=class_df["hover_text"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ))

    # Add query vector as a large star marker (drawn last = on top)
    fig.add_trace(go.Scatter(
        x=[projected_query_vector[0]],
        y=[projected_query_vector[1]],
        mode="markers+text",
        name="Query Vector",
        marker=dict(
            symbol="star",
            color="#f1c40f",    # bright yellow
            size=20,
            line=dict(color="#2c3e50", width=1.5)
        ),
        text=["Query"],
        textposition="top center",
        hovertemplate=(
            f"<b>Query Vector</b><br>"
            f"Query: {query_text[:60]}...<br>"
            f"PC1: {projected_query_vector[0]:.4f}<br>"
            f"PC2: {projected_query_vector[1]:.4f}"
            "<extra></extra>"
        ),
        showlegend=True,
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text=(
                f"Query {test_label}: {test_description}<br>"
                f"<sup>PCA 2D | threshold={similarity_threshold} | "
                f"Tabular={tabular_count} | Vector={vector_count} | "
                f"Variance shown: {cumulative_variance:.1f}%</sup>"
            ),
            font=dict(size=14)
        ),
        xaxis_title=f"PC1 ({pc1_variance:.1f}% variance)",
        yaxis_title=f"PC2 ({pc2_variance:.1f}% variance)",
        legend_title="Confusion Class",
        legend=dict(
            itemsizing="constant",
            font=dict(size=12)
        ),
        hovermode="closest",
        width=1000,
        height=700,
        annotations=[
            dict(
                text=(
                    f"⚠ PCA projects 384D → 2D. Only {cumulative_variance:.1f}% variance shown. "
                    f"Euclidean distance here ≠ cosine similarity. Qualitative only."
                ),
                xref="paper", yref="paper",
                x=0.0, y=-0.08,
                showarrow=False,
                font=dict(size=10, color="#7f8c8d"),
                align="left"
            )
        ]
    )

    return fig


def create_single_query_pca_3d_figure(
    viz_df: pd.DataFrame,
    projected_query_vector: np.ndarray,
    pca_result: dict,
    query_text: str,
    test_label: str,
    test_description: str,
    similarity_threshold: float,
    tabular_count: int,
    vector_count: int
) -> go.Figure:
    """
    Create 3D PCA scatter plot for a single query.

    Parameters are identical to create_single_query_pca_2d_figure()
    except this produces a 3D scatter (go.Scatter3d).

    Parameters
    ----------
    viz_df : pd.DataFrame
        Output of build_single_query_visualization_dataframe().
        Must contain pc1, pc2, pc3 columns.

    projected_query_vector : np.ndarray
        Shape (n_components,) with at least 3 values.

    pca_result : dict
        Contains explained_variance_ratio (at least 3 values).

    query_text : str
        Original query string.

    test_label : str
        Short label e.g. "A".

    test_description : str
        Human-readable description.

    similarity_threshold : float
        Threshold used for matching.

    tabular_count : int
        Ground truth count.

    vector_count : int
        Vector query count.

    Returns
    -------
    go.Figure
        Plotly 3D figure object.
    """
    pc1_variance = pca_result["explained_variance_ratio"][0] * 100
    pc2_variance = pca_result["explained_variance_ratio"][1] * 100
    pc3_variance = pca_result["explained_variance_ratio"][2] * 100
    cumulative_variance = pca_result["cumulative_variance_explained"] * 100

    fig = go.Figure()

    # One trace per confusion class
    for class_label in ["TN", "FN", "FP", "TP"]:
        class_mask = viz_df["confusion_class"] == class_label
        class_df = viz_df[class_mask]

        if len(class_df) == 0:
            continue

        color_hex = CONFUSION_CLASS_COLOR_MAP[class_label]
        opacity = CONFUSION_CLASS_OPACITY_MAP[class_label]

        fig.add_trace(go.Scatter3d(
            x=class_df["pc1"],
            y=class_df["pc2"],
            z=class_df["pc3"],
            mode="markers",
            name=class_label,
            marker=dict(
                color=color_hex,
                opacity=opacity,
                size=3,
                line=dict(width=0)
            ),
            text=class_df["hover_text"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ))

    # Query vector star
    fig.add_trace(go.Scatter3d(
        x=[projected_query_vector[0]],
        y=[projected_query_vector[1]],
        z=[projected_query_vector[2]],
        mode="markers+text",
        name="Query Vector",
        marker=dict(
            symbol="diamond",
            color="#f1c40f",
            size=12,
            line=dict(color="#2c3e50", width=1)
        ),
        text=["Query"],
        hovertemplate=(
            f"<b>Query Vector</b><br>"
            f"Query: {query_text[:60]}...<br>"
            f"PC1: {projected_query_vector[0]:.4f}<br>"
            f"PC2: {projected_query_vector[1]:.4f}<br>"
            f"PC3: {projected_query_vector[2]:.4f}"
            "<extra></extra>"
        ),
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Query {test_label}: {test_description} [3D PCA]<br>"
                f"<sup>threshold={similarity_threshold} | "
                f"Tabular={tabular_count} | Vector={vector_count} | "
                f"Variance shown: {cumulative_variance:.1f}%</sup>"
            ),
            font=dict(size=13)
        ),
        scene=dict(
            xaxis_title=f"PC1 ({pc1_variance:.1f}%)",
            yaxis_title=f"PC2 ({pc2_variance:.1f}%)",
            zaxis_title=f"PC3 ({pc3_variance:.1f}%)",
        ),
        legend_title="Confusion Class",
        width=1000,
        height=750,
    )

    return fig


def create_comparative_pca_2d_figure(
    viz_df: pd.DataFrame,
    projected_cat_query_vector: np.ndarray,
    projected_dog_query_vector: np.ndarray,
    pca_result: dict,
    cat_query_text: str,
    dog_query_text: str,
    test_label: str,
    test_description: str,
    similarity_threshold: float,
    tabular_count_cat: int,
    tabular_count_dog: int,
    vector_count_cat: int,
    vector_count_dog: int,
    direction_correct: bool
) -> go.Figure:
    """
    Create 2D PCA scatter plot for comparative cat vs dog query.

    Documents colored by match category (cat_only/dog_only/both/neither).
    Both cat and dog query vectors are marked with distinct star markers.

    Parameters
    ----------
    viz_df : pd.DataFrame
        Output of build_comparative_visualization_dataframe().

    projected_cat_query_vector : np.ndarray
        Cat query vector in PCA space. Shape (n_components,).

    projected_dog_query_vector : np.ndarray
        Dog query vector in PCA space. Shape (n_components,).

    pca_result : dict
        Contains explained_variance_ratio.

    cat_query_text : str
        Original cat query string.

    dog_query_text : str
        Original dog query string.

    test_label : str
        Short label (e.g., "A").

    test_description : str
        Human-readable description.

    similarity_threshold : float
        Match threshold.

    tabular_count_cat : int
        Ground truth cat count.

    tabular_count_dog : int
        Ground truth dog count.

    vector_count_cat : int
        Vector cat count.

    vector_count_dog : int
        Vector dog count.

    direction_correct : bool
        Whether vector correctly identified which animal has more matches.

    Returns
    -------
    go.Figure
        Plotly 2D comparative figure.
    """
    pc1_variance = pca_result["explained_variance_ratio"][0] * 100
    pc2_variance = pca_result["explained_variance_ratio"][1] * 100
    cumulative_variance = pca_result["cumulative_variance_explained"] * 100

    direction_symbol = "✓" if direction_correct else "✗"

    fig = go.Figure()

    # One trace per match category (neither drawn first = bottom)
    for category_label in ["neither", "cat_only", "dog_only", "both"]:
        category_mask = viz_df["match_category"] == category_label
        category_df = viz_df[category_mask]

        if len(category_df) == 0:
            continue

        color_hex = COMPARATIVE_MATCH_COLOR_MAP[category_label]
        opacity = COMPARATIVE_MATCH_OPACITY_MAP[category_label]

        fig.add_trace(go.Scatter(
            x=category_df["pc1"],
            y=category_df["pc2"],
            mode="markers",
            name=category_label.replace("_", " ").title(),
            marker=dict(
                color=color_hex,
                opacity=opacity,
                size=5,
                line=dict(width=0)
            ),
            text=category_df["hover_text"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ))

    # Cat query vector marker
    fig.add_trace(go.Scatter(
        x=[projected_cat_query_vector[0]],
        y=[projected_cat_query_vector[1]],
        mode="markers+text",
        name="Cat Query",
        marker=dict(
            symbol="star",
            color="#3498db",    # blue
            size=22,
            line=dict(color="#1a252f", width=1.5)
        ),
        text=["Cat Query"],
        textposition="top center",
        hovertemplate=(
            f"<b>Cat Query Vector</b><br>"
            f"Text: {cat_query_text[:60]}...<br>"
            f"Tabular: {tabular_count_cat} | Vector: {vector_count_cat}"
            "<extra></extra>"
        ),
        showlegend=True,
    ))

    # Dog query vector marker
    fig.add_trace(go.Scatter(
        x=[projected_dog_query_vector[0]],
        y=[projected_dog_query_vector[1]],
        mode="markers+text",
        name="Dog Query",
        marker=dict(
            symbol="star",
            color="#e67e22",    # orange
            size=22,
            line=dict(color="#1a252f", width=1.5)
        ),
        text=["Dog Query"],
        textposition="top center",
        hovertemplate=(
            f"<b>Dog Query Vector</b><br>"
            f"Text: {dog_query_text[:60]}...<br>"
            f"Tabular: {tabular_count_dog} | Vector: {vector_count_dog}"
            "<extra></extra>"
        ),
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Comparative {test_label}: {test_description}<br>"
                f"<sup>Cat: Tabular={tabular_count_cat} / Vector={vector_count_cat} | "
                f"Dog: Tabular={tabular_count_dog} / Vector={vector_count_dog} | "
                f"Direction correct: {direction_symbol} | "
                f"Variance shown: {cumulative_variance:.1f}%</sup>"
),
            font=dict(size=13)
        ),
        xaxis_title=f"PC1 ({pc1_variance:.1f}% variance)",
        yaxis_title=f"PC2 ({pc2_variance:.1f}% variance)",
        legend_title="Match Category",
        legend=dict(
            itemsizing="constant",
            font=dict(size=12)
        ),
        hovermode="closest",
        width=1000,
        height=700,
        annotations=[
            dict(
                text=(
                    f"⚠ PCA projects 384D → 2D. Only {cumulative_variance:.1f}% variance shown. "
                    f"Euclidean distance here ≠ cosine similarity. Qualitative only."
                ),
                xref="paper", yref="paper",
                x=0.0, y=-0.08,
                showarrow=False,
                font=dict(size=10, color="#7f8c8d"),
                align="left"
            )
        ]
    )

    return fig


def create_comparative_pca_3d_figure(
    viz_df: pd.DataFrame,
    projected_cat_query_vector: np.ndarray,
    projected_dog_query_vector: np.ndarray,
    pca_result: dict,
    cat_query_text: str,
    dog_query_text: str,
    test_label: str,
    test_description: str,
    similarity_threshold: float,
    tabular_count_cat: int,
    tabular_count_dog: int,
    vector_count_cat: int,
    vector_count_dog: int,
    direction_correct: bool
) -> go.Figure:
    """
    Create 3D PCA scatter plot for comparative cat vs dog query.

    Parameters are identical to create_comparative_pca_2d_figure()
    except this produces a 3D scatter (go.Scatter3d).

    Parameters
    ----------
    viz_df : pd.DataFrame
        Output of build_comparative_visualization_dataframe().
        Must contain pc1, pc2, pc3 columns.

    projected_cat_query_vector : np.ndarray
        Cat query vector in PCA space. Shape (n_components,).

    projected_dog_query_vector : np.ndarray
        Dog query vector in PCA space. Shape (n_components,).

    pca_result : dict
        Contains explained_variance_ratio (at least 3 values).

    cat_query_text : str
        Original cat query string.

    dog_query_text : str
        Original dog query string.

    test_label : str
        Short label (e.g., "A").

    test_description : str
        Human-readable description.

    similarity_threshold : float
        Match threshold.

    tabular_count_cat : int
        Ground truth cat count.

    tabular_count_dog : int
        Ground truth dog count.

    vector_count_cat : int
        Vector cat count.

    vector_count_dog : int
        Vector dog count.

    direction_correct : bool
        Whether vector correctly identified which animal has more matches.

    Returns
    -------
    go.Figure
        Plotly 3D comparative figure.
    """
    pc1_variance = pca_result["explained_variance_ratio"][0] * 100
    pc2_variance = pca_result["explained_variance_ratio"][1] * 100
    pc3_variance = pca_result["explained_variance_ratio"][2] * 100
    cumulative_variance = pca_result["cumulative_variance_explained"] * 100

    direction_symbol = "✓" if direction_correct else "✗"

    fig = go.Figure()

    # One trace per match category
    for category_label in ["neither", "cat_only", "dog_only", "both"]:
        category_mask = viz_df["match_category"] == category_label
        category_df = viz_df[category_mask]

        if len(category_df) == 0:
            continue

        color_hex = COMPARATIVE_MATCH_COLOR_MAP[category_label]
        opacity = COMPARATIVE_MATCH_OPACITY_MAP[category_label]

        fig.add_trace(go.Scatter3d(
            x=category_df["pc1"],
            y=category_df["pc2"],
            z=category_df["pc3"],
            mode="markers",
            name=category_label.replace("_", " ").title(),
            marker=dict(
                color=color_hex,
                opacity=opacity,
                size=3,
                line=dict(width=0)
            ),
            text=category_df["hover_text"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ))

    # Cat query vector
    fig.add_trace(go.Scatter3d(
        x=[projected_cat_query_vector[0]],
        y=[projected_cat_query_vector[1]],
        z=[projected_cat_query_vector[2]],
        mode="markers+text",
        name="Cat Query",
        marker=dict(
            symbol="diamond",
            color="#3498db",
            size=12,
            line=dict(color="#1a252f", width=1)
        ),
        text=["Cat Query"],
        hovertemplate=(
            f"<b>Cat Query Vector</b><br>"
            f"Text: {cat_query_text[:60]}...<br>"
            f"Tabular: {tabular_count_cat} | Vector: {vector_count_cat}"
            "<extra></extra>"
        ),
        showlegend=True,
    ))

    # Dog query vector
    fig.add_trace(go.Scatter3d(
        x=[projected_dog_query_vector[0]],
        y=[projected_dog_query_vector[1]],
        z=[projected_dog_query_vector[2]],
        mode="markers+text",
        name="Dog Query",
        marker=dict(
            symbol="diamond",
            color="#e67e22",
            size=12,
            line=dict(color="#1a252f", width=1)
        ),
        text=["Dog Query"],
        hovertemplate=(
            f"<b>Dog Query Vector</b><br>"
            f"Text: {dog_query_text[:60]}...<br>"
            f"Tabular: {tabular_count_dog} | Vector: {vector_count_dog}"
            "<extra></extra>"
        ),
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Comparative {test_label}: {test_description} [3D PCA]<br>"
                f"<sup>Cat: {tabular_count_cat}/{vector_count_cat} | "
                f"Dog: {tabular_count_dog}/{vector_count_dog} | "
                f"Direction: {direction_symbol} | "
                f"Variance shown: {cumulative_variance:.1f}%</sup>"
            ),
            font=dict(size=13)
        ),
        scene=dict(
            xaxis_title=f"PC1 ({pc1_variance:.1f}%)",
            yaxis_title=f"PC2 ({pc2_variance:.1f}%)",
            zaxis_title=f"PC3 ({pc3_variance:.1f}%)",
        ),
        legend_title="Match Category",
        width=1000,
        height=750,
    )

    return fig


# =============================================================================
# EXPLAINED VARIANCE SUMMARY PLOT
# =============================================================================


def create_explained_variance_bar_figure(
    pca_result: dict,
    n_bars_to_show: int = 20
) -> go.Figure:
    """
    Create a bar chart of explained variance per principal component.

    This is a diagnostic plot showing how much information each
    PC captures. Useful for understanding how much is lost in 2D/3D.

    Parameters
    ----------
    pca_result : dict
        Output of compute_pca_on_corpus_embeddings().
        Uses explained_variance_ratio.

    n_bars_to_show : int
        Number of PCs to display in the bar chart.
        Default 20 (first 20 components).

    Returns
    -------
    go.Figure
        Bar chart of explained variance per component.
    """
    explained_ratios = pca_result["explained_variance_ratio"]
    n_to_show = min(n_bars_to_show, len(explained_ratios))

    component_labels = [f"PC{i + 1}" for i in range(n_to_show)]
    variance_percents = [float(r * 100) for r in explained_ratios[:n_to_show]]

    # Cumulative variance line
    cumulative_percents = list(np.cumsum(variance_percents))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart: individual variance per component
    fig.add_trace(
        go.Bar(
            x=component_labels,
            y=variance_percents,
            name="Individual Variance %",
            marker_color="#3498db",
            opacity=0.7,
        ),
        secondary_y=False,
    )

    # Line chart: cumulative variance
    fig.add_trace(
        go.Scatter(
            x=component_labels,
            y=cumulative_percents,
            name="Cumulative Variance %",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )

    # Highlight first 2 and first 3 PC cumulative totals
    pc2_cumulative = float(np.sum(explained_ratios[:2]) * 100)
    pc3_cumulative = float(np.sum(explained_ratios[:3]) * 100)

    fig.add_hline(
        y=pc2_cumulative,
        line_dash="dash",
        line_color="#2ecc71",
        annotation_text=f"PC1+PC2: {pc2_cumulative:.1f}%",
        annotation_position="right",
        secondary_y=True,
    )

    fig.add_hline(
        y=pc3_cumulative,
        line_dash="dot",
        line_color="#9b59b6",
        annotation_text=f"PC1+PC2+PC3: {pc3_cumulative:.1f}%",
        annotation_position="right",
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text=(
                f"PCA Explained Variance (first {n_to_show} components)<br>"
                f"<sup>2D plot shows {pc2_cumulative:.1f}% of variance. "
                f"3D plot shows {pc3_cumulative:.1f}% of variance.</sup>"
            ),
            font=dict(size=14)
        ),
        xaxis_title="Principal Component",
        legend=dict(font=dict(size=12)),
        width=900,
        height=500,
        bargap=0.2,
    )

    fig.update_yaxes(
        title_text="Individual Variance (%)",
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Cumulative Variance (%)",
        range=[0, 100],
        secondary_y=True
    )

    return fig


# =============================================================================
# HTML SAVE FUNCTION
# =============================================================================


def save_plotly_figure_as_html(
    figure: go.Figure,
    output_filepath: Path
) -> None:
    """
    Save a Plotly figure as a self-contained HTML file.

    The HTML file includes all plotly.js inline so it is viewable
    without an internet connection.

    Parameters
    ----------
    figure : go.Figure
        Plotly figure to save.

    output_filepath : Path
        Full path including filename and .html extension.

    Raises
    ------
    Exception
        If file write fails.

    Notes
    -----
    include_plotlyjs="cdn" would produce a smaller file but requires
    internet connection to view. We use include_plotlyjs=True for
    fully self-contained offline files.

    File size note: self-contained HTML with plotly.js is ~3-4 MB per file.
    For 20k data points, the data portion adds ~1-2 MB.
    Total ~4-6 MB per file is normal and acceptable.
    """
    try:
        figure.write_html(
            str(output_filepath),
            include_plotlyjs=True,      # self-contained, no internet needed
            full_html=True,
        )
        print(f"[SAVE] HTML plot saved: {output_filepath}")

    except Exception as save_error:
        print(f"[ERROR] Failed to save HTML plot to {output_filepath}: {save_error}")
        traceback.print_exc()
        raise


# =============================================================================
# ORCHESTRATION FUNCTIONS
# =============================================================================


def generate_single_query_html_plots(
    test_label: str,
    test_description: str,
    query_text: str,
    corpus_embeddings: np.ndarray,
    pca_result: dict,
    similarities: np.ndarray,
    ground_truth_boolean_array: np.ndarray,
    dataframe: pd.DataFrame,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    tabular_count: int,
    vector_count: int,
    output_directory: Path,
    timestamp_string: str,
    max_tn_display_count: int = MAX_TN_DISPLAY_COUNT,
    random_seed: int = RANDOM_SEED
) -> None:
    """
    Generate and save 2D and 3D PCA HTML plots for one query.

    Orchestrates:
        1. Build visualization DataFrame
        2. Embed and project query vector into PCA space
        3. Create 2D figure and save HTML
        4. Create 3D figure and save HTML

    Parameters
    ----------
    test_label : str
        Short label e.g. "A".

    test_description : str
        Human-readable description of the query.

    query_text : str
        Original query text string used for this query.
        Will be embedded to get the query vector position in PCA space.

    corpus_embeddings : np.ndarray
        Full corpus embeddings. Shape (n_docs, embedding_dim).

    pca_result : dict
        Output of compute_pca_on_corpus_embeddings().
        Shared across all queries (PCA fitted once on corpus).

    similarities : np.ndarray
        Cosine similarity scores from this query. Shape (n_docs,).

    ground_truth_boolean_array : np.ndarray
        Boolean ground truth for this query. Shape (n_docs,).

    dataframe : pd.DataFrame
        Original dataset for animal_type and unique_id.

    embedding_model : SentenceTransformer
        Model for embedding the query text.

    similarity_threshold : float
        Match threshold.

    tabular_count : int
        Ground truth count.

    vector_count : int
        Vector query count.

    output_directory : Path
        Directory to save HTML files.

    timestamp_string : str
        Timestamp for filenames.

    max_tn_display_count : int
        Maximum TN rows to show (avoids plot saturation).

    random_seed : int
        Seed for reproducible TN subsampling.
    """
    try:
        print(f"\n[VIZ] Generating plots for Query {test_label}: {test_description}")

        # --- Embed and project the query vector ---
        query_vector = embed_single_query_text(query_text, embedding_model)
        projected_query = project_query_vector_into_pca_space(query_vector, pca_result)

        # --- Build visualization DataFrame ---
        viz_df = build_single_query_visualization_dataframe(
            dataframe=dataframe,
            pca_result=pca_result,
            similarities=similarities,
            ground_truth_boolean_array=ground_truth_boolean_array,
            similarity_threshold=similarity_threshold,
            max_tn_display_count=max_tn_display_count,
            random_seed=random_seed
        )

        # --- Create and save 2D figure ---
        figure_2d = create_single_query_pca_2d_figure(
            viz_df=viz_df,
            projected_query_vector=projected_query,
            pca_result=pca_result,
            query_text=query_text,
            test_label=test_label,
            test_description=test_description,
            similarity_threshold=similarity_threshold,
            tabular_count=tabular_count,
            vector_count=vector_count
        )

        filepath_2d = output_directory / f"{test_label}_pca_2d_{timestamp_string}.html"
        save_plotly_figure_as_html(figure_2d, filepath_2d)

        # --- Create and save 3D figure ---
        figure_3d = create_single_query_pca_3d_figure(
            viz_df=viz_df,
            projected_query_vector=projected_query,
            pca_result=pca_result,
            query_text=query_text,
            test_label=test_label,
            test_description=test_description,
            similarity_threshold=similarity_threshold,
            tabular_count=tabular_count,
            vector_count=vector_count
        )

        filepath_3d = output_directory / f"{test_label}_pca_3d_{timestamp_string}.html"
        save_plotly_figure_as_html(figure_3d, filepath_3d)

    except Exception as single_query_viz_error:
        print(f"[ERROR] Failed to generate plots for Query {test_label}: "
              f"{single_query_viz_error}")
        traceback.print_exc()
        # Non-fatal: continue with remaining queries


def generate_comparative_html_plots(
    test_label: str,
    test_description: str,
    cat_query_text: str,
    dog_query_text: str,
    corpus_embeddings: np.ndarray,
    pca_result: dict,
    similarities_cat: np.ndarray,
    similarities_dog: np.ndarray,
    ground_truth_cat: np.ndarray,
    ground_truth_dog: np.ndarray,
    dataframe: pd.DataFrame,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    tabular_count_cat: int,
    tabular_count_dog: int,
    vector_count_cat: int,
    vector_count_dog: int,
    direction_correct: bool,
    output_directory: Path,
    timestamp_string: str,
    max_neither_display_count: int = MAX_TN_DISPLAY_COUNT,
    random_seed: int = RANDOM_SEED
) -> None:
    """
    Generate and save 2D and 3D PCA HTML plots for a comparative query.

    Orchestrates:
        1. Build comparative visualization DataFrame
        2. Embed and project both cat and dog query vectors
        3. Create 2D comparative figure and save HTML
        4. Create 3D comparative figure and save HTML

    Parameters
    ----------
    test_label : str
        Short label e.g. "A".

    test_description : str
        Human-readable description.

    cat_query_text : str
        Query text used for the cat query.

    dog_query_text : str
        Query text used for the dog query.

    corpus_embeddings : np.ndarray
        Full corpus embeddings.

    pca_result : dict
        Fitted PCA result (shared across all plots).

    similarities_cat : np.ndarray
        Cat query similarity scores. Shape (n_docs,).

    similarities_dog : np.ndarray
        Dog query similarity scores. Shape (n_docs,).

    ground_truth_cat : np.ndarray
        Boolean ground truth for cat query.

    ground_truth_dog : np.ndarray
        Boolean ground truth for dog query.

    dataframe : pd.DataFrame
        Original dataset.

    embedding_model : SentenceTransformer
        Model for embedding query texts.

    similarity_threshold : float
        Match threshold.

    tabular_count_cat : int
        Ground truth cat count.

    tabular_count_dog : int
        Ground truth dog count.

    vector_count_cat : int
        Vector cat count.

    vector_count_dog : int
        Vector dog count.

    direction_correct : bool
        Whether vector correctly identified direction.

    output_directory : Path
        Directory to save HTML files.

    timestamp_string : str
        Timestamp for filenames.

    max_neither_display_count : int
        Maximum "neither" rows to display.

    random_seed : int
        Seed for subsampling.
    """
    try:
        print(f"\n[VIZ] Generating comparative plots for Test {test_label}: {test_description}")

        # --- Embed and project both query vectors ---
        cat_query_vector = embed_single_query_text(cat_query_text, embedding_model)
        dog_query_vector = embed_single_query_text(dog_query_text, embedding_model)

        projected_cat_query = project_query_vector_into_pca_space(cat_query_vector, pca_result)
        projected_dog_query = project_query_vector_into_pca_space(dog_query_vector, pca_result)

        # Report query vector positions for transparency
        print(f"[VIZ]   Cat query PCA position: "
              f"PC1={projected_cat_query[0]:.4f}, PC2={projected_cat_query[1]:.4f}")
        print(f"[VIZ]   Dog query PCA position: "
              f"PC1={projected_dog_query[0]:.4f}, PC2={projected_dog_query[1]:.4f}")

        # Distance between the two query vectors in PCA space
        # (informative but not equal to cosine distance in original space)
        pca_distance_between_queries = float(
            np.linalg.norm(projected_cat_query[:2] - projected_dog_query[:2])
        )
        print(f"[VIZ]   Distance between cat/dog query vectors in PCA 2D space: "
              f"{pca_distance_between_queries:.4f}")

        # --- Build comparative visualization DataFrame ---
        viz_df = build_comparative_visualization_dataframe(
            dataframe=dataframe,
            pca_result=pca_result,
            similarities_cat=similarities_cat,
            similarities_dog=similarities_dog,
            ground_truth_cat=ground_truth_cat,
            ground_truth_dog=ground_truth_dog,
            similarity_threshold=similarity_threshold,
            max_neither_display_count=max_neither_display_count,
            random_seed=random_seed
        )

        # --- Create and save 2D comparative figure ---
        figure_2d = create_comparative_pca_2d_figure(
            viz_df=viz_df,
            projected_cat_query_vector=projected_cat_query,
            projected_dog_query_vector=projected_dog_query,
            pca_result=pca_result,
            cat_query_text=cat_query_text,
            dog_query_text=dog_query_text,
            test_label=test_label,
            test_description=test_description,
            similarity_threshold=similarity_threshold,
            tabular_count_cat=tabular_count_cat,
            tabular_count_dog=tabular_count_dog,
            vector_count_cat=vector_count_cat,
            vector_count_dog=vector_count_dog,
            direction_correct=direction_correct
        )

        filepath_2d = output_directory / f"comparative_{test_label}_pca_2d_{timestamp_string}.html"
        save_plotly_figure_as_html(figure_2d, filepath_2d)

        # --- Create and save 3D comparative figure ---
        figure_3d = create_comparative_pca_3d_figure(
            viz_df=viz_df,
            projected_cat_query_vector=projected_cat_query,
            projected_dog_query_vector=projected_dog_query,
            pca_result=pca_result,
            cat_query_text=cat_query_text,
            dog_query_text=dog_query_text,
            test_label=test_label,
            test_description=test_description,
            similarity_threshold=similarity_threshold,
            tabular_count_cat=tabular_count_cat,
            tabular_count_dog=tabular_count_dog,
            vector_count_cat=vector_count_cat,
            vector_count_dog=vector_count_dog,
            direction_correct=direction_correct
        )

        filepath_3d = output_directory / f"comparative_{test_label}_pca_3d_{timestamp_string}.html"
        save_plotly_figure_as_html(figure_3d, filepath_3d)

    except Exception as comparative_viz_error:
        print(f"[ERROR] Failed to generate comparative plots for Test {test_label}: "
              f"{comparative_viz_error}")
        traceback.print_exc()
        # Non-fatal: continue with remaining tests


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================


def run_visualization_suite(
    csv_file_path: str,
    similarity_threshold: float
) -> None:
    """
    Execute complete visualization suite for all queries A-D.

    Workflow:
        1. Create output directory
        2. Load data
        3. Load embedding model
        4. Generate corpus embeddings
        5. Fit PCA ONCE on corpus (reused for all plots)
        6. Save explained variance plot
        7. For each query A-D:
            - Run tabular ground truth
            - Run vector queries (cat and dog)
            - Generate single-query plots for cat query
            - Generate single-query plots for dog query
            - Generate comparative cat-vs-dog plots
        8. Print list of all saved files

    Parameters
    ----------
    csv_file_path : str
        Path to synthetic biology CSV file.

    similarity_threshold : float
        Cosine similarity threshold for match classification.

    Notes
    -----
    PCA is fitted ONCE on the full corpus and reused for every plot.
    This ensures all plots share the same coordinate space,
    making them directly comparable.

    The embedding model is also loaded ONCE and reused.
    Corpus embeddings are generated ONCE and reused.
    """
    print("=" * 90)
    print("VECTOR SPACE VISUALIZATION SUITE")
    print("PCA 2D/3D Plots for Queries A-D (Single and Comparative)")
    print("=" * 90)

    np.random.seed(RANDOM_SEED)
    print(f"\n[CONFIG] Random seed: {RANDOM_SEED}")
    print(f"[CONFIG] Similarity threshold: {similarity_threshold}")
    print(f"[CONFIG] Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"[CONFIG] PCA components: {PCA_N_COMPONENTS}")
    print(f"[CONFIG] Max TN display: {MAX_TN_DISPLAY_COUNT}")

    # -------------------------------------------------------------------------
    # Step 1: Output directory
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 1: CREATE OUTPUT DIRECTORY")
    print(f"{'='*70}")

    timestamp_string, output_directory = create_visualization_output_directory()

    # -------------------------------------------------------------------------
    # Step 2: Load data
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 2: LOAD DATA")
    print(f"{'='*70}")

    dataframe = load_csv_data(csv_file_path)

    # -------------------------------------------------------------------------
    # Step 3: Load model
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 3: LOAD EMBEDDING MODEL")
    print(f"{'='*70}")

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

    # -------------------------------------------------------------------------
    # Step 4: Generate corpus embeddings
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 4: GENERATE CORPUS EMBEDDINGS")
    print(f"{'='*70}")

    description_texts = dataframe["unstructured_description"].tolist()
    corpus_embeddings = generate_corpus_embeddings(description_texts, embedding_model)

    # -------------------------------------------------------------------------
    # Step 5: Fit PCA ONCE on corpus
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 5: FIT PCA ON CORPUS (once, reused for all plots)")
    print(f"{'='*70}")

    pca_result = compute_pca_on_corpus_embeddings(
        corpus_embeddings_matrix=corpus_embeddings,
        n_components=PCA_N_COMPONENTS
    )

    # -------------------------------------------------------------------------
    # Step 6: Save explained variance diagnostic plot
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 6: SAVE EXPLAINED VARIANCE PLOT")
    print(f"{'='*70}")

    variance_figure = create_explained_variance_bar_figure(
        pca_result=pca_result,
        n_bars_to_show=20
    )
    variance_filepath = output_directory / f"pca_explained_variance_{timestamp_string}.html"
    save_plotly_figure_as_html(variance_figure, variance_filepath)

    # -------------------------------------------------------------------------
    # Step 7: Configure time threshold (same logic as other scripts)
    # -------------------------------------------------------------------------
    time_threshold_unix = int(dataframe["birth_unix"].median())  # type: ignore[arg-type]
    time_threshold_year = datetime.datetime.fromtimestamp(time_threshold_unix).year
    print(f"\n[TIME] Threshold: Unix={time_threshold_unix}, Year={time_threshold_year}")

    # -------------------------------------------------------------------------
    # Step 8: Run queries and generate plots A-D
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 8: RUN QUERIES AND GENERATE PLOTS A-D")
    print(f"{'='*70}")

    # Define query configurations for A-D
    # Each entry: (label, description, cat_query_text, dog_query_text,
    #              cat_tabular_func, dog_tabular_func,
    #              cat_gt_func, dog_gt_func)
    #
    # All tabular and ground truth functions are lambdas that close over
    # dataframe and time_threshold_unix as needed.

    query_configurations = [
        {
            "label": "A",
            "description": "Count: cats vs dogs",
            "cat_query_text": "This is about a cat.",
            "dog_query_text": "This is about a dog.",
            "cat_tabular_count": lambda: int((dataframe["animal_type"] == "cat").sum()),
            "dog_tabular_count": lambda: int((dataframe["animal_type"] == "dog").sum()),
            "cat_ground_truth": lambda: (dataframe["animal_type"] == "cat").values,
            "dog_ground_truth": lambda: (dataframe["animal_type"] == "dog").values,
        },
        {
            "label": "B",
            "description": "Count: flying cats vs flying dogs",
            "cat_query_text": "A cat, feline, that can fly. A flying cat.",
            "dog_query_text": "A dog, canine, that can fly. A flying dog.",
            "cat_tabular_count": lambda: int(
                ((dataframe["animal_type"] == "cat") & (dataframe["can_fly"] == True)).sum()
            ),
            "dog_tabular_count": lambda: int(
                ((dataframe["animal_type"] == "dog") & (dataframe["can_fly"] == True)).sum()
            ),
            "cat_ground_truth": lambda: (
                (dataframe["animal_type"] == "cat") & (dataframe["can_fly"] == True)
            ).values,
            "dog_ground_truth": lambda: (
                (dataframe["animal_type"] == "dog") & (dataframe["can_fly"] == True)
            ).values,
        },
        {
            "label": "C",
            "description": f"Count: cats vs dogs born after {time_threshold_year}",
            "cat_query_text": (
                f"A cat, feline, born after the year {time_threshold_year}. "
                f"A recently born cat."
            ),
            "dog_query_text": (
                f"A dog, canine, born after the year {time_threshold_year}. "
                f"A recently born dog."
            ),
            "cat_tabular_count": lambda: int(
                (
                    (dataframe["animal_type"] == "cat") &
                    (dataframe["birth_unix"] > time_threshold_unix)
                ).sum()
            ),
            "dog_tabular_count": lambda: int(
                (
                    (dataframe["animal_type"] == "dog") &
                    (dataframe["birth_unix"] > time_threshold_unix)
                ).sum()
            ),
            "cat_ground_truth": lambda: (
                (dataframe["animal_type"] == "cat") &
                (dataframe["birth_unix"] > time_threshold_unix)
            ).values,
            "dog_ground_truth": lambda: (
                (dataframe["animal_type"] == "dog") &
                (dataframe["birth_unix"] > time_threshold_unix)
            ).values,
        },
        {
            "label": "D",
            "description": f"Flying cats vs flying dogs born after {time_threshold_year}",
            "cat_query_text": (
                f"A cat, feline, born after the year {time_threshold_year}, "
                f"that can fly. A flying cat born recently."
            ),
            "dog_query_text": (
                f"A dog, canine, born after the year {time_threshold_year}, "
                f"that can fly. A flying dog born recently."
            ),
            "cat_tabular_count": lambda: int(
                (
                    (dataframe["animal_type"] == "cat") &
                    (dataframe["birth_unix"] > time_threshold_unix) &
                    (dataframe["can_fly"] == True)
                ).sum()
            ),
            "dog_tabular_count": lambda: int(
                (
                    (dataframe["animal_type"] == "dog") &
                    (dataframe["birth_unix"] > time_threshold_unix) &
                    (dataframe["can_fly"] == True)
                ).sum()
            ),
            "cat_ground_truth": lambda: (
                (dataframe["animal_type"] == "cat") &
                (dataframe["birth_unix"] > time_threshold_unix) &
                (dataframe["can_fly"] == True)
            ).values,
            "dog_ground_truth": lambda: (
                (dataframe["animal_type"] == "dog") &
                (dataframe["birth_unix"] > time_threshold_unix) &
                (dataframe["can_fly"] == True)
            ).values,
        },
    ]

    # Run each configuration
    for config in query_configurations:

        test_label = config["label"]
        test_description = config["description"]
        cat_query_text = config["cat_query_text"]
        dog_query_text = config["dog_query_text"]

        print(f"\n[RUNNING] Test {test_label}: {test_description}")

        # Tabular counts
        tabular_count_cat = config["cat_tabular_count"]()
        tabular_count_dog = config["dog_tabular_count"]()

        # Ground truth arrays
        ground_truth_cat = config["cat_ground_truth"]()
        ground_truth_dog = config["dog_ground_truth"]()

        # Vector queries (cat)
        cat_query_vector = embed_single_query_text(cat_query_text, embedding_model)
        similarities_cat = calculate_cosine_similarities(cat_query_vector, corpus_embeddings)
        vector_count_cat = int(np.sum(similarities_cat > similarity_threshold))

        # Vector queries (dog)
        dog_query_vector = embed_single_query_text(dog_query_text, embedding_model)
        similarities_dog = calculate_cosine_similarities(dog_query_vector, corpus_embeddings)
        vector_count_dog = int(np.sum(similarities_dog > similarity_threshold))

        # Direction correctness
        if tabular_count_cat > tabular_count_dog:
            tabular_direction = "cat_higher"
        elif tabular_count_dog > tabular_count_cat:
            tabular_direction = "dog_higher"
        else:
            tabular_direction = "equal"

        if vector_count_cat > vector_count_dog:
            vector_direction = "cat_higher"
        elif vector_count_dog > vector_count_cat:
            vector_direction = "dog_higher"
        else:
            vector_direction = "equal"

        direction_correct = (tabular_direction == vector_direction)

        print(f"  Cat:  tabular={tabular_count_cat}, vector={vector_count_cat}")
        print(f"  Dog:  tabular={tabular_count_dog}, vector={vector_count_dog}")
        print(f"  Direction correct: {direction_correct}")

        # --- Single query plots: cat ---
        generate_single_query_html_plots(
            test_label=f"{test_label}_cat",
            test_description=f"{test_description} [cat only]",
            query_text=cat_query_text,
            corpus_embeddings=corpus_embeddings,
            pca_result=pca_result,
            similarities=similarities_cat,
            ground_truth_boolean_array=ground_truth_cat,
            dataframe=dataframe,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            tabular_count=tabular_count_cat,
            vector_count=vector_count_cat,
            output_directory=output_directory,
            timestamp_string=timestamp_string
        )

        # --- Single query plots: dog ---
        generate_single_query_html_plots(
            test_label=f"{test_label}_dog",
            test_description=f"{test_description} [dog only]",
            query_text=dog_query_text,
            corpus_embeddings=corpus_embeddings,
            pca_result=pca_result,
            similarities=similarities_dog,
            ground_truth_boolean_array=ground_truth_dog,
            dataframe=dataframe,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            tabular_count=tabular_count_dog,
            vector_count=vector_count_dog,
            output_directory=output_directory,
            timestamp_string=timestamp_string
        )

        # --- Comparative plots: cat vs dog ---
        generate_comparative_html_plots(
            test_label=test_label,
            test_description=test_description,
            cat_query_text=cat_query_text,
            dog_query_text=dog_query_text,
            corpus_embeddings=corpus_embeddings,
            pca_result=pca_result,
            similarities_cat=similarities_cat,
            similarities_dog=similarities_dog,
            ground_truth_cat=ground_truth_cat,
            ground_truth_dog=ground_truth_dog,
            dataframe=dataframe,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            tabular_count_cat=tabular_count_cat,
            tabular_count_dog=tabular_count_dog,
            vector_count_cat=vector_count_cat,
            vector_count_dog=vector_count_dog,
            direction_correct=direction_correct,
            output_directory=output_directory,
            timestamp_string=timestamp_string
        )

    # -------------------------------------------------------------------------
    # Step 9: Final report
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 9: ALL PLOTS SAVED")
    print(f"{'='*70}")

    print(f"\n  Output directory: {output_directory}")
    print(f"\n  Saved files:")
    for saved_file in sorted(output_directory.iterdir()):
        file_size_mb = saved_file.stat().st_size / (1024 * 1024)
        print(f"    {saved_file.name:<65} {file_size_mb:.1f} MB")

    print(f"\n[CONFIG RECORD]")
    print(f"  CSV: {csv_file_path}")
    print(f"  Model: {EMBEDDING_MODEL_NAME}")
    print(f"  Threshold: {similarity_threshold}")
    print(f"  Timestamp: {timestamp_string}")


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Main entry point for visualization suite.

    Prompts for CSV path and threshold, then runs all plots.
    """
    print("\n" + "=" * 90)
    print("STARTING VECTOR SPACE VISUALIZATION SUITE")
    print("=" * 90 + "\n")

    try:
        csv_path_input = input(
            "Step 1: Enter path to synthetic_biology_dataset_augmented.csv:\n"
        )
        csv_file_path = csv_path_input.strip()

        if not csv_file_path:
            print("[ERROR] No CSV path provided. Exiting.")
            return

        threshold_input = input(
            "\nStep 2: Enter similarity threshold (press Enter for default 0.5):\n"
        ).strip()

        if threshold_input:
            try:
                similarity_threshold = float(threshold_input)
            except ValueError:
                print(f"[WARNING] Invalid threshold '{threshold_input}', using 0.5")
                similarity_threshold = 0.5
        else:
            similarity_threshold = 0.5

        print(f"\n[CONFIG] Using threshold: {similarity_threshold}")

        run_visualization_suite(
            csv_file_path=csv_file_path,
            similarity_threshold=similarity_threshold
        )

        print("\n" + "=" * 90)
        print("VISUALIZATION SUITE COMPLETED SUCCESSFULLY")
        print("=" * 90)

    except FileNotFoundError as file_error:
        print(f"\n[FATAL ERROR] File not found: {file_error}")
        traceback.print_exc()

    except ValueError as validation_error:
        print(f"\n[FATAL ERROR] Validation error: {validation_error}")
        traceback.print_exc()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Cancelled by user.")

    except Exception as unexpected_error:
        print(f"\n[FATAL ERROR] Unexpected error: {unexpected_error}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

