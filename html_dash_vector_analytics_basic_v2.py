# -*- coding: utf-8 -*-
"""html_dash_vector_analytics_basic_vN.py

1. place in a dir with these files:
    /A_fp_diagnostic_summary_{}.csv
    /B_fp_diagnostic_summary_{}.csv
    /C_fp_diagnostic_summary_{}.csv
    /D_fp_diagnostic_summary_{}.csv
    /E_fp_diagnostic_summary_{}.csv
    /summary_report_{}.csv

2. Run it:
```bash
python html_dash_vector_analytics_basic_v{N}.py
```

"""

import os
import sys
import glob
import logging
import traceback
import pandas as pd
import plotly.graph_objects as go
import numpy as np

"""
================================================================================
DARK-MODE DIAGNOSTIC DASHBOARD - DOCUMENTATION
================================================================================

PROJECT OVERVIEW
----------------
This module generates a self-contained, interactive HTML dashboard for visualizing
query performance diagnostics from a vector search evaluation system. The dashboard
is designed as a "dark-mode only" interface to reduce eye strain during extended
analysis sessions.

The dashboard consumes two types of CSV data files:
1. Summary Report: Contains per-query performance metrics (precision, recall, etc.)
2. Diagnostic Reports: Contains per-field error contribution analysis for false positives

The output is a single HTML file with embedded JavaScript and CSS that can be
opened in any modern web browser without requiring a server or additional dependencies.


DESIGN PHILOSOPHY
-----------------
1. CLARITY OVER BREVITY
   - The original dashboard was "too brief" - this version prioritizes showing MORE
     data through MORE visualizations rather than condensing information.
   - Each metric gets its own dedicated chart by default.
   - Metrics are only combined when logically related (e.g., similarity statistics).

2. DARK-MODE ONLY
   - Single color scheme optimized for dark backgrounds.
   - Muted pastel colors chosen for visibility without harsh contrast.
   - No light-mode toggle - this is intentionally a focused design decision.

3. COMMUNICATION IS THE GOAL
   - Extensive logging at every step for operational transparency.
   - Charts include descriptive titles explaining what they measure.
   - Hover tooltips provide precise values.
   - Query Reference tab provides full context for interpreting results.

4. GRACEFUL DEGRADATION
   - Missing data results in "no data available" messages, not errors.
   - Missing numeric values are filled with zeros.
   - Empty diagnostic files (indicating no errors) are handled gracefully.
   - Each chart function returns None if it cannot render, and the HTML
     assembly handles this by showing a fallback message.

5. MODULAR FUNCTIONAL DESIGN (NOT OOP)
   - Each chart has its own dedicated function for easy modification.
   - Functions are pure where possible (data in, figure out).
   - No class hierarchies to navigate.
   - Easy to add new charts by following the existing pattern.


ARCHITECTURE OVERVIEW
---------------------

┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENTRY POINT                                       │
│                         generate_dashboard()                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FILE DISCOVERY & LOADING                              │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────┐     │
│  │ find_matching_files()   │    │ load_summary_report()               │     │
│  │ (glob pattern matching) │    │ load_diagnostic_reports()           │     │
│  └─────────────────────────┘    └─────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHART GENERATION                                     │
│                       generate_all_charts()                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Classification Metrics:                                               │   │
│  │   create_chart_tp_fp_fn_tn_counts()                                  │   │
│  │   create_chart_precision_per_query()                                 │   │
│  │   create_chart_recall_per_query()                                    │   │
│  │   create_chart_f1_score_per_query()                                  │   │
│  │   create_chart_accuracy_per_query()                                  │   │
│  │   create_chart_specificity_per_query()                               │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │ Count Analysis:                                                       │   │
│  │   create_chart_tabular_vs_vector_counts()                            │   │
│  │   create_chart_absolute_and_percent_error()                          │   │
│  │   create_chart_direction_indicator()                                 │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │ Similarity Statistics:                                                │   │
│  │   create_chart_similarity_statistics_combined()                      │   │
│  │   create_chart_similarity_range_per_query()                          │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │ Timing:                                                               │   │
│  │   create_chart_elapsed_seconds_per_query()                           │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │ Diagnostic Analysis:                                                  │   │
│  │   create_chart_error_contribution_proportion()                       │   │
│  │   create_chart_fp_rows_affected_proportion()                         │   │
│  │   create_chart_wrong_count_by_field()                                │   │
│  │   create_chart_diagnostic_stacked_proportions()                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HTML ASSEMBLY                                        │
│                  assemble_complete_html_dashboard()                          │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────┐     │
│  │ generate_css_styles()   │    │ generate_javascript_for_tabs()      │     │
│  └─────────────────────────┘    └─────────────────────────────────────┘     │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────┐     │
│  │ convert_figure_to_      │    │ generate_dark_mode_html_table()     │     │
│  │ html_div()              │    │ generate_query_reference_html()     │     │
│  └─────────────────────────┘    └─────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FILE OUTPUT                                          │
│                   write_html_dashboard_to_file()                             │
└─────────────────────────────────────────────────────────────────────────────┘


TAB STRUCTURE
-------------
The dashboard uses a tab-based layout with the following organization:

Tab 1: "Classification Metrics"
    - TP/FP/FN/TN counts (grouped bar chart)
    - Precision per query (bar chart)
    - Recall per query (bar chart)
    - F1 Score per query (bar chart)
    - Accuracy per query (bar chart)
    - Specificity per query (bar chart)

Tab 2: "Count Analysis"
    - Tabular vs Vector counts comparison (grouped bar)
    - Absolute & Percent Error (dual-axis: bars + line)
    - Retrieval Direction indicator (diverging bar: over/under)

Tab 3: "Similarity Statistics"
    - Combined statistics: mean, median, min, max, std (multi-trace)
    - Min-Max range with mean indicator (filled area + markers)

Tab 4: "Timing"
    - Query execution elapsed seconds (bar chart)

Tab 5: "Diagnostic Analysis"
    - Error contribution proportion by field (grouped bar)
    - FP rows affected proportion by field (grouped bar)
    - Wrong count by field (grouped bar)
    - Stacked error contribution view (stacked bar)

Tab 6: "Raw Data Tables"
    - Summary report as HTML table
    - Diagnostic report as HTML table

Tab 7: "Query Reference"
    - Query label, description, threshold, and full query text for each query


COLOR PALETTE
-------------
The dark mode color scheme is defined in the DARK_MODE_COLORS dictionary:

    Background colors:
        background: "#121212"        (near-black, main background)
        paper_background: "#1a1a1a"  (slightly lighter, for chart areas)
        table_header_bg: "#2a2a2a"   (table headers)
        table_row_alt: "#1e1e1e"     (alternating table rows)

    Text colors:
        text_primary: "#e0e0e0"      (main text, high contrast)
        text_secondary: "#a0a0a0"    (secondary/dimmer text)

    Structural colors:
        grid_lines: "#333333"        (subtle grid lines)
        table_border: "#444444"      (table borders)

    Chart colors (muted pastels):
        pastel_cyan: "#7ec8e3"       (primary accent)
        pastel_lavender: "#b39ddb"   (secondary accent)
        pastel_green: "#81c784"      (positive indicators)
        pastel_orange: "#ffb74d"     (warning/attention)
        pastel_coral: "#ef9a9a"      (negative/error indicators)
        pastel_blue_gray: "#90a4ae"  (neutral)
        pastel_pink: "#f48fb1"
        pastel_teal: "#80cbc4"
        pastel_yellow: "#fff59d"
        pastel_lime: "#c5e1a5"


CONFIGURATION
-------------
The module uses environment variables for configuration (no hardcoding):

    SUMMARY_FILE_PATTERN
        Default: "summary_report_*.csv"
        Glob pattern to find summary report CSV files.

    DIAGNOSTIC_FILE_PATTERN
        Default: "*_fp_diagnostic_summary_*.csv"
        Glob pattern to find diagnostic report CSV files.

    OUTPUT_HTML_PATH
        Default: "diagnostic_dashboard_v2.html"
        Output file path for the generated HTML dashboard.

Example usage:
    export SUMMARY_FILE_PATTERN="results/summary_*.csv"
    export DIAGNOSTIC_FILE_PATTERN="results/*_diagnostic_*.csv"
    export OUTPUT_HTML_PATH="output/my_dashboard.html"
    python dashboard_generator.py


EXPECTED INPUT DATA SCHEMAS
---------------------------

Summary Report CSV (required columns):
    query_label              : str   - Unique identifier for the query (e.g., "A", "B", "C")
    query_description        : str   - Human-readable description of the query
    query_text               : str   - The actual query text used for vector search
    similarity_threshold     : float - Threshold used for similarity matching
    tabular_count            : int   - Expected count from tabular/ground truth data
    vector_count             : int   - Actual count retrieved by vector search
    absolute_error           : int   - |tabular_count - vector_count|
    percent_error            : float - Percentage error relative to tabular_count
    direction                : str   - "over" or "under" indicating retrieval direction
    tp_count                 : int   - True positive count
    fp_count                 : int   - False positive count
    fn_count                 : int   - False negative count
    tn_count                 : int   - True negative count
    precision                : float - TP / (TP + FP), may be NaN if undefined
    recall                   : float - TP / (TP + FN), may be NaN if undefined
    f1_score                 : float - Harmonic mean of precision and recall
    accuracy                 : float - (TP + TN) / Total
    specificity              : float - TN / (TN + FP)
    similarity_mean          : float - Mean similarity score of results
    similarity_std           : float - Standard deviation of similarity scores
    similarity_min           : float - Minimum similarity score
    similarity_max           : float - Maximum similarity score
    similarity_median        : float - Median similarity score
    elapsed_seconds          : float - Query execution time

Diagnostic Report CSV (required columns):
    query_label                   : str   - Query identifier (matches summary report)
    field_name                    : str   - Name of the data field analyzed
    wrong_count                   : int   - Count of incorrect values in this field
    fp_rows_affected_proportion   : float - Proportion of FP rows with errors in this field
    error_contribution_proportion : float - This field's contribution to total errors


DEPENDENCIES
------------
    pandas      : DataFrame operations and CSV parsing
    plotly      : Interactive chart generation (graph_objects)
    numpy       : Numeric operations and NaN handling

The output HTML uses Plotly.js from CDN (https://cdn.plot.ly/plotly-2.27.0.min.js)
so no local JavaScript files are required.


EXTENSION GUIDE
---------------
To add a new chart:

1. Create a new function following the pattern:

   def create_chart_my_new_metric(summary_dataframe: pd.DataFrame) -> go.Figure | None:
       '''Docstring explaining the chart.'''
       try:
           required_columns = ['query_label', 'my_metric']
           missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
           if missing_columns:
               logger.warning(f"Cannot create My Metric chart. Missing columns: {missing_columns}")
               return None

           figure = go.Figure()
           figure.add_trace(go.Bar(...))

           layout_config = create_dark_mode_layout(
               title_text="My New Metric",
               xaxis_title="Query Label",
               yaxis_title="Value",
               height=400
           )
           figure.update_layout(**layout_config)

           logger.info("Successfully created My Metric chart.")
           return figure

       except Exception as general_exception:
           logger.error("Failed to create My Metric chart.")
           logger.error(traceback.format_exc())
           return None

2. Add the chart to the appropriate tab in generate_all_charts():

   charts_by_tab["My Tab Name"] = [
       ("My New Metric", create_chart_my_new_metric(summary_dataframe)),
   ]

To add a new tab:

1. Add a new list of (title, figure) tuples to charts_by_tab in generate_all_charts()
2. The tab will automatically appear in the navigation

To modify the color scheme:

1. Edit the DARK_MODE_COLORS dictionary at the top of the module
2. All charts and HTML elements reference this dictionary


KNOWN LIMITATIONS
-----------------
1. Single summary file: If multiple summary files match the pattern, only the
   first one is used. A warning is logged.

2. Large datasets: The HTML file embeds all chart data, which can result in
   large file sizes for datasets with many rows. Consider aggregating data
   before dashboard generation if this becomes an issue.

3. Browser compatibility: Tested on modern browsers (Chrome, Firefox, Safari,
   Edge). Older browsers may have issues with CSS Grid or Plotly.js features.

4. No real-time updates: This is a static HTML file generator, not a live
   dashboard. Re-run the script to update with new data.


ERROR HANDLING PHILOSOPHY
-------------------------
- All functions use try-except blocks with full traceback logging
- Chart functions return None on failure (graceful degradation)
- File operations raise RuntimeError with context
- The main function uses sys.exit(1) for critical failures
- Missing data columns result in warnings, not errors


TESTING RECOMMENDATIONS
-----------------------
1. Test with empty diagnostic files (simulating no errors)
2. Test with missing columns in CSV files
3. Test with NaN/null values in numeric columns
4. Test with single query vs multiple queries
5. Test with very long query_text values
6. Verify HTML renders correctly in target browsers


MAINTENANCE NOTES
-----------------
- Plotly.js version is pinned to 2.27.0 in the CDN link; update if needed
- CSS is embedded in the HTML (no external stylesheet to maintain)
- JavaScript is minimal and vanilla (no framework dependencies)
- All logging uses the standard library logging module


ORIGINAL REQUIREMENTS SUMMARY
-----------------------------
From the project specification:
- Dark-mode only (black background, muted pastels)
- Show MORE visualizations, not fewer (alpha demo was "too brief")
- Vertical scrolling through plots is acceptable
- Tabs for organization are acceptable
- Skip empty plots or show "no data" message
- Use zero for missing numeric values
- Show each diagnostic separately by default
- Combine logically related metrics where appropriate
- Query descriptions in separate reference section

================================================================================
END OF DOCUMENTATION
================================================================================
"""

# Configure logging to ensure we announce problems and communicate clearly.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DARK MODE COLOR PALETTE CONFIGURATION
# =============================================================================
# Muted pastels visible on dark backgrounds, designed for reduced eye strain.

DARK_MODE_COLORS = {
    "background": "#121212",           # Near-black background
    "paper_background": "#1a1a1a",     # Slightly lighter for plot areas
    "text_primary": "#e0e0e0",         # Light gray primary text
    "text_secondary": "#a0a0a0",       # Dimmer secondary text
    "grid_lines": "#333333",           # Subtle grid lines
    "table_border": "#444444",         # Table border color
    "table_header_bg": "#2a2a2a",      # Table header background
    "table_row_alt": "#1e1e1e",        # Alternating row background

    # Chart colors - muted pastels
    "pastel_cyan": "#7ec8e3",
    "pastel_lavender": "#b39ddb",
    "pastel_green": "#81c784",
    "pastel_orange": "#ffb74d",
    "pastel_coral": "#ef9a9a",
    "pastel_blue_gray": "#90a4ae",
    "pastel_pink": "#f48fb1",
    "pastel_teal": "#80cbc4",
    "pastel_yellow": "#fff59d",
    "pastel_lime": "#c5e1a5",
}

# Ordered list for cycling through chart colors
CHART_COLOR_SEQUENCE = [
    DARK_MODE_COLORS["pastel_cyan"],
    DARK_MODE_COLORS["pastel_lavender"],
    DARK_MODE_COLORS["pastel_green"],
    DARK_MODE_COLORS["pastel_orange"],
    DARK_MODE_COLORS["pastel_coral"],
    DARK_MODE_COLORS["pastel_blue_gray"],
    DARK_MODE_COLORS["pastel_pink"],
    DARK_MODE_COLORS["pastel_teal"],
    DARK_MODE_COLORS["pastel_yellow"],
    DARK_MODE_COLORS["pastel_lime"],
]


# =============================================================================
# FILE DISCOVERY AND DATA LOADING FUNCTIONS
# =============================================================================

def find_matching_files(file_pattern: str) -> list[str]:
    """
    Finds all file paths matching a given glob pattern.

    This function scans the file system using a wildcard pattern and returns
    all matching file paths. It is used to discover summary and diagnostic
    CSV files dynamically.

    Args:
        file_pattern (str): The wildcard string used to locate matching files.
                           Examples: "summary_report_*.csv", "*_diagnostic_*.csv"

    Returns:
        list[str]: A list of file paths that match the provided pattern.
                  Returns an empty list if no matches are found.

    Raises:
        RuntimeError: If an unexpected error occurs during file system scanning,
                     such as permission errors or invalid pattern syntax.
    """
    try:
        logger.info(f"Scanning for files matching pattern: {file_pattern}")
        matched_file_paths = glob.glob(file_pattern)
        logger.info(f"Discovered {len(matched_file_paths)} files matching '{file_pattern}'.")
        return matched_file_paths
    except Exception as general_exception:
        logger.error(f"Failed to find files for pattern '{file_pattern}'.")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"File discovery failed: {general_exception}") from general_exception


def load_summary_report(file_path: str) -> pd.DataFrame:
    """
    Loads the main summary report CSV into a Pandas DataFrame.

    This function reads the primary summary data file which contains
    per-query performance metrics including counts, precision, recall,
    similarity statistics, and timing information.

    Missing numeric values are filled with 0 to ensure chart stability.

    Args:
        file_path (str): The exact file path to the summary report CSV.

    Returns:
        pd.DataFrame: The loaded summary data with missing values handled.

    Raises:
        FileNotFoundError: If the summary file does not exist at the specified path.
        RuntimeError: If parsing or reading the CSV fails due to format issues.
    """
    try:
        logger.info(f"Loading summary report from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The required summary file does not exist: {file_path}")

        summary_dataframe = pd.read_csv(file_path)
        logger.info(f"Successfully loaded summary report with {len(summary_dataframe)} rows.")

        # Identify numeric columns and fill missing values with 0
        numeric_columns = summary_dataframe.select_dtypes(include=[np.number]).columns
        summary_dataframe[numeric_columns] = summary_dataframe[numeric_columns].fillna(0)
        logger.info(f"Filled missing values in {len(numeric_columns)} numeric columns with 0.")

        return summary_dataframe

    except FileNotFoundError:
        # Re-raise FileNotFoundError without wrapping
        raise
    except Exception as general_exception:
        logger.error(f"Failed to load summary report from '{file_path}'.")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Summary report loading failed: {general_exception}") from general_exception


def load_diagnostic_reports(file_paths: list[str]) -> pd.DataFrame:
    """
    Reads multiple diagnostic CSV files, concatenates them, and prepares for visualization.

    Diagnostic files contain per-field error analysis showing which data fields
    contributed to false positive errors. This function combines multiple files
    (one per query with errors) into a unified DataFrame.

    Missing numeric values are filled with 0 to ensure chart stability.
    Rows with empty query labels are removed as they provide no useful grouping.

    Args:
        file_paths (list[str]): A list of file paths pointing to diagnostic CSV files.
                               Can be empty if no diagnostic files exist.

    Returns:
        pd.DataFrame: A combined DataFrame containing all diagnostic records.
                     Returns an empty DataFrame if no valid files or rows exist.

    Raises:
        RuntimeError: If reading or concatenating the CSVs fails due to format issues.
    """
    try:
        logger.info(f"Attempting to load and combine {len(file_paths)} diagnostic files.")

        # Handle case where no diagnostic files exist (no errors to report)
        if not file_paths:
            logger.info("No diagnostic files provided. Returning empty DataFrame.")
            return pd.DataFrame()

        dataframes_list = []

        for path in file_paths:
            logger.info(f"Reading diagnostic file: {path}")
            # Some files might only contain headers; Pandas handles this gracefully.
            diagnostic_df = pd.read_csv(path)

            # Only add non-empty dataframes
            if not diagnostic_df.empty:
                dataframes_list.append(diagnostic_df)
            else:
                logger.info(f"Skipping empty diagnostic file: {path}")

        if not dataframes_list:
            logger.warning("No non-empty diagnostic dataframes were loaded. Returning empty DataFrame.")
            return pd.DataFrame()

        combined_diagnostic_dataframe = pd.concat(dataframes_list, ignore_index=True)

        # Remove rows where the query label is NaN (cannot group meaningfully)
        if 'query_label' in combined_diagnostic_dataframe.columns:
            original_row_count = len(combined_diagnostic_dataframe)
            combined_diagnostic_dataframe = combined_diagnostic_dataframe.dropna(subset=['query_label'])
            dropped_count = original_row_count - len(combined_diagnostic_dataframe)
            if dropped_count > 0:
                logger.info(f"Dropped {dropped_count} rows with missing query_label values.")

        # Fill missing numeric values with 0
        numeric_columns = combined_diagnostic_dataframe.select_dtypes(include=[np.number]).columns
        combined_diagnostic_dataframe[numeric_columns] = combined_diagnostic_dataframe[numeric_columns].fillna(0)

        logger.info(f"Successfully combined diagnostic reports into {len(combined_diagnostic_dataframe)} rows.")
        return combined_diagnostic_dataframe

    except Exception as general_exception:
        logger.error("Failed to load or combine diagnostic reports.")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Diagnostic report compilation failed: {general_exception}") from general_exception


# =============================================================================
# PLOTLY DARK MODE LAYOUT CONFIGURATION
# =============================================================================

def create_dark_mode_layout(
    title_text: str,
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 400,
    show_legend: bool = True,
    barmode: str = "group"
) -> dict:
    """
    Creates a standardized dark mode layout configuration dictionary for Plotly figures.

    This function generates consistent styling across all charts, ensuring
    visual coherence and reduced eye strain with the dark color scheme.

    Args:
        title_text (str): The main title to display above the chart.
        xaxis_title (str): Label for the X-axis. Empty string if not needed.
        yaxis_title (str): Label for the Y-axis. Empty string if not needed.
        height (int): The height of the chart in pixels.
        show_legend (bool): Whether to display the legend.
        barmode (str): Bar chart mode - "group", "stack", or "overlay".

    Returns:
        dict: A dictionary containing all layout configuration parameters
             ready to be passed to fig.update_layout().
    """
    layout_configuration = {
        "title": {
            "text": title_text,
            "font": {"color": DARK_MODE_COLORS["text_primary"], "size": 16},
            "x": 0.5,  # Center the title
            "xanchor": "center"
        },
        "paper_bgcolor": DARK_MODE_COLORS["paper_background"],
        "plot_bgcolor": DARK_MODE_COLORS["background"],
        "font": {"color": DARK_MODE_COLORS["text_primary"]},
        "height": height,
        "showlegend": show_legend,
        "barmode": barmode,
        "legend": {
            "bgcolor": DARK_MODE_COLORS["paper_background"],
            "bordercolor": DARK_MODE_COLORS["grid_lines"],
            "borderwidth": 1,
            "font": {"color": DARK_MODE_COLORS["text_primary"]}
        },
        "xaxis": {
            "title": {"text": xaxis_title, "font": {"color": DARK_MODE_COLORS["text_secondary"]}},
            "tickfont": {"color": DARK_MODE_COLORS["text_secondary"]},
            "gridcolor": DARK_MODE_COLORS["grid_lines"],
            "linecolor": DARK_MODE_COLORS["grid_lines"],
            "zerolinecolor": DARK_MODE_COLORS["grid_lines"]
        },
        "yaxis": {
            "title": {"text": yaxis_title, "font": {"color": DARK_MODE_COLORS["text_secondary"]}},
            "tickfont": {"color": DARK_MODE_COLORS["text_secondary"]},
            "gridcolor": DARK_MODE_COLORS["grid_lines"],
            "linecolor": DARK_MODE_COLORS["grid_lines"],
            "zerolinecolor": DARK_MODE_COLORS["grid_lines"]
        },
        "margin": {"l": 60, "r": 40, "t": 60, "b": 60}
    }

    return layout_configuration


# =============================================================================
# INDIVIDUAL CHART CREATION FUNCTIONS - SUMMARY METRICS
# =============================================================================

def create_chart_tp_fp_fn_tn_counts(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a grouped bar chart showing True Positive, False Positive,
    False Negative, and True Negative counts per query.

    This chart provides a quick overview of classification performance
    across all queries, showing the raw count distribution.

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label, tp_count, fp_count, fn_count, tn_count columns.

    Returns:
        go.Figure | None: A Plotly Figure object with the bar chart,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'tp_count', 'fp_count', 'fn_count', 'tn_count']

        # Verify all required columns exist
        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create TP/FP/FN/TN chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        # Add each count type as a separate trace
        figure.add_trace(go.Bar(
            name='True Positives',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['tp_count'],
            marker_color=DARK_MODE_COLORS["pastel_green"],
            hovertemplate="Query: %{x}<br>True Positives: %{y}<extra></extra>"
        ))

        figure.add_trace(go.Bar(
            name='False Positives',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['fp_count'],
            marker_color=DARK_MODE_COLORS["pastel_coral"],
            hovertemplate="Query: %{x}<br>False Positives: %{y}<extra></extra>"
        ))

        figure.add_trace(go.Bar(
            name='False Negatives',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['fn_count'],
            marker_color=DARK_MODE_COLORS["pastel_orange"],
            hovertemplate="Query: %{x}<br>False Negatives: %{y}<extra></extra>"
        ))

        figure.add_trace(go.Bar(
            name='True Negatives',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['tn_count'],
            marker_color=DARK_MODE_COLORS["pastel_blue_gray"],
            hovertemplate="Query: %{x}<br>True Negatives: %{y}<extra></extra>"
        ))

        # Apply dark mode layout
        layout_config = create_dark_mode_layout(
            title_text="Classification Counts: TP / FP / FN / TN per Query",
            xaxis_title="Query Label",
            yaxis_title="Count",
            height=450,
            show_legend=True,
            barmode="group"
        )
        figure.update_layout(**layout_config)

        logger.info("Successfully created TP/FP/FN/TN counts chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create TP/FP/FN/TN counts chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_precision_per_query(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a bar chart showing Precision metric per query.

    Precision measures the proportion of positive identifications that
    were actually correct (TP / (TP + FP)).

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label and precision columns.

    Returns:
        go.Figure | None: A Plotly Figure object with the bar chart,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'precision']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Precision chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        figure.add_trace(go.Bar(
            name='Precision',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['precision'],
            marker_color=DARK_MODE_COLORS["pastel_cyan"],
            hovertemplate="Query: %{x}<br>Precision: %{y:.4f}<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="Precision per Query (TP / (TP + FP))",
            xaxis_title="Query Label",
            yaxis_title="Precision (0-1)",
            height=350,
            show_legend=False
        )
        figure.update_layout(**layout_config)

        # Set y-axis range for proportion metrics
        figure.update_yaxes(range=[0, 1.05])

        logger.info("Successfully created Precision chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Precision chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_recall_per_query(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a bar chart showing Recall metric per query.

    Recall measures the proportion of actual positives that were
    correctly identified (TP / (TP + FN)).

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label and recall columns.

    Returns:
        go.Figure | None: A Plotly Figure object with the bar chart,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'recall']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Recall chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        figure.add_trace(go.Bar(
            name='Recall',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['recall'],
            marker_color=DARK_MODE_COLORS["pastel_lavender"],
            hovertemplate="Query: %{x}<br>Recall: %{y:.4f}<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="Recall per Query (TP / (TP + FN))",
            xaxis_title="Query Label",
            yaxis_title="Recall (0-1)",
            height=350,
            show_legend=False
        )
        figure.update_layout(**layout_config)

        figure.update_yaxes(range=[0, 1.05])

        logger.info("Successfully created Recall chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Recall chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_f1_score_per_query(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a bar chart showing F1 Score metric per query.

    F1 Score is the harmonic mean of Precision and Recall, providing
    a balanced measure of classification performance.

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label and f1_score columns.

    Returns:
        go.Figure | None: A Plotly Figure object with the bar chart,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'f1_score']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create F1 Score chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        figure.add_trace(go.Bar(
            name='F1 Score',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['f1_score'],
            marker_color=DARK_MODE_COLORS["pastel_green"],
            hovertemplate="Query: %{x}<br>F1 Score: %{y:.4f}<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="F1 Score per Query (Harmonic Mean of Precision & Recall)",
            xaxis_title="Query Label",
            yaxis_title="F1 Score (0-1)",
            height=350,
            show_legend=False
        )
        figure.update_layout(**layout_config)

        figure.update_yaxes(range=[0, 1.05])

        logger.info("Successfully created F1 Score chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create F1 Score chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_accuracy_per_query(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a bar chart showing Accuracy metric per query.

    Accuracy measures the proportion of all predictions that were
    correct ((TP + TN) / Total).

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label and accuracy columns.

    Returns:
        go.Figure | None: A Plotly Figure object with the bar chart,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'accuracy']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Accuracy chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        figure.add_trace(go.Bar(
            name='Accuracy',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['accuracy'],
            marker_color=DARK_MODE_COLORS["pastel_teal"],
            hovertemplate="Query: %{x}<br>Accuracy: %{y:.4f}<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="Accuracy per Query ((TP + TN) / Total)",
            xaxis_title="Query Label",
            yaxis_title="Accuracy (0-1)",
            height=350,
            show_legend=False
        )
        figure.update_layout(**layout_config)

        figure.update_yaxes(range=[0, 1.05])

        logger.info("Successfully created Accuracy chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Accuracy chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_specificity_per_query(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a bar chart showing Specificity metric per query.

    Specificity measures the proportion of actual negatives that were
    correctly identified (TN / (TN + FP)).

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label and specificity columns.

    Returns:
        go.Figure | None: A Plotly Figure object with the bar chart,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'specificity']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Specificity chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        figure.add_trace(go.Bar(
            name='Specificity',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['specificity'],
            marker_color=DARK_MODE_COLORS["pastel_pink"],
            hovertemplate="Query: %{x}<br>Specificity: %{y:.4f}<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="Specificity per Query (TN / (TN + FP))",
            xaxis_title="Query Label",
            yaxis_title="Specificity (0-1)",
            height=350,
            show_legend=False
        )
        figure.update_layout(**layout_config)

        figure.update_yaxes(range=[0, 1.05])

        logger.info("Successfully created Specificity chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Specificity chart.")
        logger.error(traceback.format_exc())
        return None


# =============================================================================
# INDIVIDUAL CHART CREATION FUNCTIONS - COUNT ANALYSIS
# =============================================================================

def create_chart_tabular_vs_vector_counts(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a grouped bar chart comparing tabular (expected) counts
    versus vector (retrieved) counts per query.

    This visualization helps identify queries where the vector search
    significantly over-retrieved or under-retrieved compared to the
    expected tabular baseline.

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label, tabular_count, and vector_count columns.

    Returns:
        go.Figure | None: A Plotly Figure object with the comparison chart,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'tabular_count', 'vector_count']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Tabular vs Vector chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        figure.add_trace(go.Bar(
            name='Tabular Count (Expected)',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['tabular_count'],
            marker_color=DARK_MODE_COLORS["pastel_cyan"],
            hovertemplate="Query: %{x}<br>Expected (Tabular): %{y}<extra></extra>"
        ))

        figure.add_trace(go.Bar(
            name='Vector Count (Retrieved)',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['vector_count'],
            marker_color=DARK_MODE_COLORS["pastel_lavender"],
            hovertemplate="Query: %{x}<br>Retrieved (Vector): %{y}<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="Expected vs Retrieved Counts: Tabular (Baseline) vs Vector (Search Results)",
            xaxis_title="Query Label",
            yaxis_title="Count",
            height=400,
            show_legend=True,
            barmode="group"
        )
        figure.update_layout(**layout_config)

        logger.info("Successfully created Tabular vs Vector counts chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Tabular vs Vector counts chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_absolute_and_percent_error(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a dual-axis chart showing both absolute error and percent error per query.

    Absolute error shows the raw count difference, while percent error
    normalizes this against the expected count. Showing both together
    provides context (a 100 count difference means different things
    for queries expecting 200 vs 10000).

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label, absolute_error, and percent_error columns.

    Returns:
        go.Figure | None: A Plotly Figure object with dual y-axes,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'absolute_error', 'percent_error']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Error chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        # Absolute error on primary y-axis
        figure.add_trace(go.Bar(
            name='Absolute Error (Count)',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['absolute_error'],
            marker_color=DARK_MODE_COLORS["pastel_coral"],
            yaxis='y',
            hovertemplate="Query: %{x}<br>Absolute Error: %{y}<extra></extra>"
        ))

        # Percent error on secondary y-axis (line chart for visual distinction)
        figure.add_trace(go.Scatter(
            name='Percent Error (%)',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['percent_error'],
            mode='lines+markers',
            marker=dict(color=DARK_MODE_COLORS["pastel_yellow"], size=10),
            line=dict(color=DARK_MODE_COLORS["pastel_yellow"], width=2),
            yaxis='y2',
            hovertemplate="Query: %{x}<br>Percent Error: %{y:.2f}%<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="Error Analysis: Absolute Error (bars) and Percent Error (line)",
            xaxis_title="Query Label",
            yaxis_title="Absolute Error (Count)",
            height=450,
            show_legend=True
        )

        # Add secondary y-axis configuration
        layout_config['yaxis2'] = {
            'title': {'text': 'Percent Error (%)', 'font': {'color': DARK_MODE_COLORS["pastel_yellow"]}},
            'tickfont': {'color': DARK_MODE_COLORS["pastel_yellow"]},
            'overlaying': 'y',
            'side': 'right',
            'gridcolor': DARK_MODE_COLORS["grid_lines"],
            'zerolinecolor': DARK_MODE_COLORS["grid_lines"]
        }

        figure.update_layout(**layout_config)

        logger.info("Successfully created Absolute and Percent Error chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Error chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_direction_indicator(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a visual indicator showing whether each query over-retrieved
    or under-retrieved compared to expected counts.

    Uses a diverging bar chart with positive values for "over" and
    negative values for "under" to clearly show the direction of error.

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label, direction, and absolute_error columns.

    Returns:
        go.Figure | None: A Plotly Figure object with direction indicators,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'direction', 'absolute_error']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Direction chart. Missing columns: {missing_columns}")
            return None

        # Create directional values: positive for "over", negative for "under"
        directional_values = []
        colors = []

        for _, row in summary_dataframe.iterrows():
            if row['direction'] == 'over':
                directional_values.append(row['absolute_error'])
                colors.append(DARK_MODE_COLORS["pastel_coral"])  # Over = coral/red tint
            elif row['direction'] == 'under':
                directional_values.append(-row['absolute_error'])
                colors.append(DARK_MODE_COLORS["pastel_cyan"])  # Under = cyan/blue tint
            else:
                directional_values.append(0)
                colors.append(DARK_MODE_COLORS["pastel_blue_gray"])

        figure = go.Figure()

        figure.add_trace(go.Bar(
            name='Retrieval Direction',
            x=summary_dataframe['query_label'],
            y=directional_values,
            marker_color=colors,
            hovertemplate="Query: %{x}<br>Direction: " + summary_dataframe['direction'] + "<br>Magnitude: %{y}<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="Retrieval Direction: Over-retrieved (positive) vs Under-retrieved (negative)",
            xaxis_title="Query Label",
            yaxis_title="Signed Error Count",
            height=400,
            show_legend=False
        )
        figure.update_layout(**layout_config)

        # Add a zero line for reference
        figure.add_hline(y=0, line_dash="dash", line_color=DARK_MODE_COLORS["text_secondary"])

        logger.info("Successfully created Direction indicator chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Direction chart.")
        logger.error(traceback.format_exc())
        return None


# =============================================================================
# INDIVIDUAL CHART CREATION FUNCTIONS - SIMILARITY STATISTICS
# =============================================================================

def create_chart_similarity_statistics_combined(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a combined chart showing all similarity statistics per query.

    Displays mean, standard deviation, min, max, and median similarity
    scores together for comprehensive comparison. Uses multiple traces
    to show the distribution characteristics.

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label and similarity statistic columns.

    Returns:
        go.Figure | None: A Plotly Figure object with combined similarity stats,
                         or None if required columns are missing.
    """
    try:
        required_columns = [
            'query_label', 'similarity_mean', 'similarity_std',
            'similarity_min', 'similarity_max', 'similarity_median'
        ]

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Similarity Statistics chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        # Mean as primary bar
        figure.add_trace(go.Bar(
            name='Mean',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['similarity_mean'],
            marker_color=DARK_MODE_COLORS["pastel_cyan"],
            hovertemplate="Query: %{x}<br>Mean: %{y:.4f}<extra></extra>"
        ))

        # Median as secondary bar
        figure.add_trace(go.Bar(
            name='Median',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['similarity_median'],
            marker_color=DARK_MODE_COLORS["pastel_lavender"],
            hovertemplate="Query: %{x}<br>Median: %{y:.4f}<extra></extra>"
        ))

        # Min as line
        figure.add_trace(go.Scatter(
            name='Min',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['similarity_min'],
            mode='lines+markers',
            marker=dict(color=DARK_MODE_COLORS["pastel_coral"], size=8),
            line=dict(color=DARK_MODE_COLORS["pastel_coral"], width=2),
            hovertemplate="Query: %{x}<br>Min: %{y:.4f}<extra></extra>"
        ))

        # Max as line
        figure.add_trace(go.Scatter(
            name='Max',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['similarity_max'],
            mode='lines+markers',
            marker=dict(color=DARK_MODE_COLORS["pastel_green"], size=8),
            line=dict(color=DARK_MODE_COLORS["pastel_green"], width=2),
            hovertemplate="Query: %{x}<br>Max: %{y:.4f}<extra></extra>"
        ))

        # Standard deviation as error bars on mean would be ideal,
        # but showing as separate trace for clarity
        figure.add_trace(go.Scatter(
            name='Std Dev',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['similarity_std'],
            mode='lines+markers',
            marker=dict(color=DARK_MODE_COLORS["pastel_orange"], size=8, symbol='diamond'),
            line=dict(color=DARK_MODE_COLORS["pastel_orange"], width=2, dash='dot'),
            hovertemplate="Query: %{x}<br>Std Dev: %{y:.4f}<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="Similarity Score Statistics: Mean, Median, Min, Max, and Standard Deviation",
            xaxis_title="Query Label",
            yaxis_title="Similarity Score",
            height=500,
            show_legend=True
        )
        figure.update_layout(**layout_config)

        logger.info("Successfully created Similarity Statistics combined chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Similarity Statistics chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_similarity_range_per_query(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a range chart (like a candlestick without open/close) showing
    the min-max range of similarity scores per query with mean marked.

    This provides an alternative view focusing on the spread of similarity
    scores for each query.

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label, similarity_min, similarity_max, similarity_mean columns.

    Returns:
        go.Figure | None: A Plotly Figure object showing similarity ranges,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'similarity_min', 'similarity_max', 'similarity_mean']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Similarity Range chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        # Add range as filled area between min and max
        # Using scatter with fill for the range visualization
        query_labels = summary_dataframe['query_label'].tolist()

        # Upper bound (max)
        figure.add_trace(go.Scatter(
            name='Max',
            x=query_labels,
            y=summary_dataframe['similarity_max'],
            mode='lines+markers',
            marker=dict(color=DARK_MODE_COLORS["pastel_green"], size=8),
            line=dict(color=DARK_MODE_COLORS["pastel_green"], width=1),
            hovertemplate="Query: %{x}<br>Max: %{y:.4f}<extra></extra>"
        ))

        # Lower bound (min) with fill to upper
        figure.add_trace(go.Scatter(
            name='Min',
            x=query_labels,
            y=summary_dataframe['similarity_min'],
            mode='lines+markers',
            marker=dict(color=DARK_MODE_COLORS["pastel_coral"], size=8),
            line=dict(color=DARK_MODE_COLORS["pastel_coral"], width=1),
            fill='tonexty',
            fillcolor='rgba(126, 200, 227, 0.2)',  # Transparent cyan fill
            hovertemplate="Query: %{x}<br>Min: %{y:.4f}<extra></extra>"
        ))

        # Mean as distinct markers
        figure.add_trace(go.Scatter(
            name='Mean',
            x=query_labels,
            y=summary_dataframe['similarity_mean'],
            mode='markers',
            marker=dict(
                color=DARK_MODE_COLORS["pastel_yellow"],
                size=12,
                symbol='diamond',
                line=dict(color=DARK_MODE_COLORS["text_primary"], width=1)
            ),
            hovertemplate="Query: %{x}<br>Mean: %{y:.4f}<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="Similarity Score Range: Min to Max with Mean Indicator",
            xaxis_title="Query Label",
            yaxis_title="Similarity Score",
            height=400,
            show_legend=True
        )
        figure.update_layout(**layout_config)

        logger.info("Successfully created Similarity Range chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Similarity Range chart.")
        logger.error(traceback.format_exc())
        return None


# =============================================================================
# INDIVIDUAL CHART CREATION FUNCTIONS - TIMING
# =============================================================================

def create_chart_elapsed_seconds_per_query(summary_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a bar chart showing query execution time (elapsed seconds) per query.

    This helps identify performance bottlenecks and queries that may need
    optimization.

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label and elapsed_seconds columns.

    Returns:
        go.Figure | None: A Plotly Figure object with timing data,
                         or None if required columns are missing.
    """
    try:
        required_columns = ['query_label', 'elapsed_seconds']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Elapsed Seconds chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        figure.add_trace(go.Bar(
            name='Elapsed Time',
            x=summary_dataframe['query_label'],
            y=summary_dataframe['elapsed_seconds'],
            marker_color=DARK_MODE_COLORS["pastel_teal"],
            hovertemplate="Query: %{x}<br>Elapsed: %{y:.4f} seconds<extra></extra>"
        ))

        layout_config = create_dark_mode_layout(
            title_text="Query Execution Time (Elapsed Seconds)",
            xaxis_title="Query Label",
            yaxis_title="Seconds",
            height=350,
            show_legend=False
        )
        figure.update_layout(**layout_config)

        logger.info("Successfully created Elapsed Seconds chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Elapsed Seconds chart.")
        logger.error(traceback.format_exc())
        return None


# =============================================================================
# INDIVIDUAL CHART CREATION FUNCTIONS - DIAGNOSTIC ANALYSIS
# =============================================================================

def create_chart_error_contribution_proportion(diagnostic_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a grouped bar chart showing error contribution proportion per field,
    grouped by query label.

    Error contribution proportion indicates what fraction of the total
    false positive errors each field is responsible for.

    Args:
        diagnostic_dataframe (pd.DataFrame): The diagnostic data containing
            query_label, field_name, and error_contribution_proportion columns.

    Returns:
        go.Figure | None: A Plotly Figure object with error contributions,
                         or None if dataframe is empty or missing required columns.
    """
    try:
        if diagnostic_dataframe.empty:
            logger.info("Diagnostic dataframe is empty. Skipping error contribution chart.")
            return None

        required_columns = ['query_label', 'field_name', 'error_contribution_proportion']

        missing_columns = [col for col in required_columns if col not in diagnostic_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Error Contribution chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        # Get unique queries and fields
        unique_queries = diagnostic_dataframe['query_label'].unique()
        unique_fields = diagnostic_dataframe['field_name'].unique()

        # Create a trace for each field, showing values across queries
        for field_index, field_name in enumerate(unique_fields):
            field_data = diagnostic_dataframe[diagnostic_dataframe['field_name'] == field_name]

            # Build values array aligned with unique_queries
            y_values = []
            for query in unique_queries:
                query_field_data = field_data[field_data['query_label'] == query]
                if not query_field_data.empty:
                    y_values.append(query_field_data['error_contribution_proportion'].values[0])
                else:
                    y_values.append(0)  # No data for this query-field combination

            color_index = field_index % len(CHART_COLOR_SEQUENCE)

            figure.add_trace(go.Bar(
                name=field_name,
                x=list(unique_queries),
                y=y_values,
                marker_color=CHART_COLOR_SEQUENCE[color_index],
                hovertemplate=f"Query: %{{x}}<br>Field: {field_name}<br>Contribution: %{{y:.4f}}<extra></extra>"
            ))

        layout_config = create_dark_mode_layout(
            title_text="Error Contribution Proportion by Field (per Query)",
            xaxis_title="Query Label",
            yaxis_title="Error Contribution Proportion",
            height=450,
            show_legend=True,
            barmode="group"
        )
        figure.update_layout(**layout_config)

        logger.info("Successfully created Error Contribution Proportion chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Error Contribution chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_fp_rows_affected_proportion(diagnostic_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a grouped bar chart showing the proportion of false positive rows
    affected by each field, grouped by query label.

    This shows what fraction of all FP rows had an error in each specific field.

    Args:
        diagnostic_dataframe (pd.DataFrame): The diagnostic data containing
            query_label, field_name, and fp_rows_affected_proportion columns.

    Returns:
        go.Figure | None: A Plotly Figure object with FP rows affected data,
                         or None if dataframe is empty or missing required columns.
    """
    try:
        if diagnostic_dataframe.empty:
            logger.info("Diagnostic dataframe is empty. Skipping FP rows affected chart.")
            return None

        required_columns = ['query_label', 'field_name', 'fp_rows_affected_proportion']

        missing_columns = [col for col in required_columns if col not in diagnostic_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create FP Rows Affected chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        unique_queries = diagnostic_dataframe['query_label'].unique()
        unique_fields = diagnostic_dataframe['field_name'].unique()

        for field_index, field_name in enumerate(unique_fields):
            field_data = diagnostic_dataframe[diagnostic_dataframe['field_name'] == field_name]

            y_values = []
            for query in unique_queries:
                query_field_data = field_data[field_data['query_label'] == query]
                if not query_field_data.empty:
                    y_values.append(query_field_data['fp_rows_affected_proportion'].values[0])
                else:
                    y_values.append(0)

            color_index = field_index % len(CHART_COLOR_SEQUENCE)

            figure.add_trace(go.Bar(
                name=field_name,
                x=list(unique_queries),
                y=y_values,
                marker_color=CHART_COLOR_SEQUENCE[color_index],
                hovertemplate=f"Query: %{{x}}<br>Field: {field_name}<br>FP Rows Affected: %{{y:.4f}}<extra></extra>"
            ))

        layout_config = create_dark_mode_layout(
            title_text="FP Rows Affected Proportion by Field (per Query)",
            xaxis_title="Query Label",
            yaxis_title="Proportion of FP Rows Affected",
            height=450,
            show_legend=True,
            barmode="group"
        )
        figure.update_layout(**layout_config)

        logger.info("Successfully created FP Rows Affected Proportion chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create FP Rows Affected chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_wrong_count_by_field(diagnostic_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a grouped bar chart showing the raw wrong count per field,
    grouped by query label.

    This shows the absolute number of incorrect values for each field.

    Args:
        diagnostic_dataframe (pd.DataFrame): The diagnostic data containing
            query_label, field_name, and wrong_count columns.

    Returns:
        go.Figure | None: A Plotly Figure object with wrong count data,
                         or None if dataframe is empty or missing required columns.
    """
    try:
        if diagnostic_dataframe.empty:
            logger.info("Diagnostic dataframe is empty. Skipping wrong count chart.")
            return None

        required_columns = ['query_label', 'field_name', 'wrong_count']

        missing_columns = [col for col in required_columns if col not in diagnostic_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Wrong Count chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        unique_queries = diagnostic_dataframe['query_label'].unique()
        unique_fields = diagnostic_dataframe['field_name'].unique()

        for field_index, field_name in enumerate(unique_fields):
            field_data = diagnostic_dataframe[diagnostic_dataframe['field_name'] == field_name]

            y_values = []
            for query in unique_queries:
                query_field_data = field_data[field_data['query_label'] == query]
                if not query_field_data.empty:
                    y_values.append(query_field_data['wrong_count'].values[0])
                else:
                    y_values.append(0)

            color_index = field_index % len(CHART_COLOR_SEQUENCE)

            figure.add_trace(go.Bar(
                name=field_name,
                x=list(unique_queries),
                y=y_values,
                marker_color=CHART_COLOR_SEQUENCE[color_index],
                hovertemplate=f"Query: %{{x}}<br>Field: {field_name}<br>Wrong Count: %{{y}}<extra></extra>"
            ))

        layout_config = create_dark_mode_layout(
            title_text="Wrong Count by Field (per Query)",
            xaxis_title="Query Label",
            yaxis_title="Wrong Count",
            height=450,
            show_legend=True,
            barmode="group"
        )
        figure.update_layout(**layout_config)

        logger.info("Successfully created Wrong Count by Field chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Wrong Count chart.")
        logger.error(traceback.format_exc())
        return None


def create_chart_diagnostic_stacked_proportions(diagnostic_dataframe: pd.DataFrame) -> go.Figure | None:
    """
    Creates a stacked bar chart showing the relative contribution of each field
    to false positive errors, making it easy to see which fields dominate.

    This is an alternative view to the grouped bar chart, emphasizing
    the composition of errors.

    Args:
        diagnostic_dataframe (pd.DataFrame): The diagnostic data containing
            query_label, field_name, and error_contribution_proportion columns.

    Returns:
        go.Figure | None: A Plotly Figure object with stacked proportions,
                         or None if dataframe is empty or missing required columns.
    """
    try:
        if diagnostic_dataframe.empty:
            logger.info("Diagnostic dataframe is empty. Skipping stacked proportions chart.")
            return None

        required_columns = ['query_label', 'field_name', 'error_contribution_proportion']

        missing_columns = [col for col in required_columns if col not in diagnostic_dataframe.columns]
        if missing_columns:
            logger.warning(f"Cannot create Stacked Proportions chart. Missing columns: {missing_columns}")
            return None

        figure = go.Figure()

        unique_queries = diagnostic_dataframe['query_label'].unique()
        unique_fields = diagnostic_dataframe['field_name'].unique()

        for field_index, field_name in enumerate(unique_fields):
            field_data = diagnostic_dataframe[diagnostic_dataframe['field_name'] == field_name]

            y_values = []
            for query in unique_queries:
                query_field_data = field_data[field_data['query_label'] == query]
                if not query_field_data.empty:
                    y_values.append(query_field_data['error_contribution_proportion'].values[0])
                else:
                    y_values.append(0)

            color_index = field_index % len(CHART_COLOR_SEQUENCE)

            figure.add_trace(go.Bar(
                name=field_name,
                x=list(unique_queries),
                y=y_values,
                marker_color=CHART_COLOR_SEQUENCE[color_index],
                hovertemplate=f"Query: %{{x}}<br>Field: {field_name}<br>Contribution: %{{y:.4f}}<extra></extra>"
            ))

        layout_config = create_dark_mode_layout(
            title_text="Error Contribution Composition (Stacked by Field)",
            xaxis_title="Query Label",
            yaxis_title="Cumulative Error Contribution",
            height=450,
            show_legend=True,
            barmode="stack"
        )
        figure.update_layout(**layout_config)

        logger.info("Successfully created Stacked Proportions chart.")
        return figure

    except Exception as general_exception:
        logger.error("Failed to create Stacked Proportions chart.")
        logger.error(traceback.format_exc())
        return None


# =============================================================================
# HTML TABLE GENERATION FUNCTIONS
# =============================================================================

def generate_dark_mode_html_table(
    dataframe: pd.DataFrame,
    table_id: str,
    table_title: str
) -> str:
    """
    Converts a Pandas DataFrame into a dark-mode styled HTML table.

    The table uses the dark mode color scheme with proper contrast
    for readability on dark backgrounds.

    Args:
        dataframe (pd.DataFrame): The data to convert to an HTML table.
        table_id (str): A unique HTML id attribute for the table element.
        table_title (str): A title to display above the table.

    Returns:
        str: An HTML string containing the styled table with title.
             Returns a message indicating no data if dataframe is empty.
    """
    try:
        if dataframe.empty:
            return f"""
            <div class="table-section">
                <h2 style="color: {DARK_MODE_COLORS['text_primary']};">{table_title}</h2>
                <p style="color: {DARK_MODE_COLORS['text_secondary']};">No data available for this table.</p>
            </div>
            """

        # Generate HTML table with dark mode styling classes
        html_table = dataframe.to_html(
            index=False,
            classes='dark-table',
            table_id=table_id,
            border=0,
            na_rep='—'  # Em dash for missing values
        )

        wrapped_html = f"""
        <div class="table-section">
            <h2 style="color: {DARK_MODE_COLORS['text_primary']};">{table_title}</h2>
            <div class="table-scroll-container">
                {html_table}
            </div>
        </div>
        """

        return wrapped_html

    except Exception as general_exception:
        logger.error(f"Failed to generate HTML table '{table_title}'.")
        logger.error(traceback.format_exc())
        return f"""
        <div class="table-section">
            <h2 style="color: {DARK_MODE_COLORS['text_primary']};">{table_title}</h2>
            <p style="color: {DARK_MODE_COLORS['pastel_coral']};">Error generating table: {general_exception}</p>
        </div>
        """


def generate_query_reference_html(summary_dataframe: pd.DataFrame) -> str:
    """
    Generates an HTML section displaying query descriptions and full query text
    as a reference guide.

    This provides context for interpreting the charts by showing what each
    query label actually represents.

    Args:
        summary_dataframe (pd.DataFrame): The summary data containing
            query_label, query_description, and query_text columns.

    Returns:
        str: An HTML string containing the formatted query reference section.
    """
    try:
        required_columns = ['query_label', 'query_description', 'query_text']

        missing_columns = [col for col in required_columns if col not in summary_dataframe.columns]
        if missing_columns:
            return f"""
            <div class="query-reference-section">
                <h2 style="color: {DARK_MODE_COLORS['text_primary']};">Query Reference</h2>
                <p style="color: {DARK_MODE_COLORS['text_secondary']};">
                    Query reference information not available. Missing columns: {', '.join(missing_columns)}
                </p>
            </div>
            """

        reference_items_html = ""

        for _, row in summary_dataframe.iterrows():
            query_label = row.get('query_label', 'Unknown')
            query_description = row.get('query_description', 'No description available')
            query_text = row.get('query_text', 'No query text available')
            similarity_threshold = row.get('similarity_threshold', 'N/A')

            reference_items_html += f"""
            <div class="query-reference-item">
                <h3 style="color: {DARK_MODE_COLORS['pastel_cyan']};">
                    Query {query_label}
                </h3>
                <p style="color: {DARK_MODE_COLORS['text_primary']};">
                    <strong>Description:</strong> {query_description}
                </p>
                <p style="color: {DARK_MODE_COLORS['text_secondary']};">
                    <strong>Similarity Threshold:</strong> {similarity_threshold}
                </p>
                <div class="query-text-box">
                    <strong style="color: {DARK_MODE_COLORS['text_primary']};">Query Text:</strong>
                    <pre style="color: {DARK_MODE_COLORS['pastel_lavender']};">{query_text}</pre>
                </div>
            </div>
            """

        complete_reference_html = f"""
        <div class="query-reference-section">
            <h2 style="color: {DARK_MODE_COLORS['text_primary']};">Query Reference Guide</h2>
            <p style="color: {DARK_MODE_COLORS['text_secondary']};">
                Detailed information about each query, including description and the actual query text used.
            </p>
            {reference_items_html}
        </div>
        """

        return complete_reference_html

    except Exception as general_exception:
        logger.error("Failed to generate query reference HTML.")
        logger.error(traceback.format_exc())
        return f"""
        <div class="query-reference-section">
            <h2 style="color: {DARK_MODE_COLORS['text_primary']};">Query Reference</h2>
            <p style="color: {DARK_MODE_COLORS['pastel_coral']};">Error generating reference: {general_exception}</p>
        </div>
        """


# =============================================================================
# MAIN HTML DASHBOARD ASSEMBLY FUNCTIONS
# =============================================================================

def generate_css_styles() -> str:
    """
    Generates the complete CSS stylesheet for the dark mode dashboard.

    This includes styles for the overall layout, tabs, tables, charts,
    and all interactive elements.

    Returns:
        str: A complete CSS stylesheet as a string.
    """
    css_stylesheet = f"""
    <style>
        /* ===== BASE STYLES ===== */
        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: {DARK_MODE_COLORS['background']};
            color: {DARK_MODE_COLORS['text_primary']};
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}

        h1 {{
            color: {DARK_MODE_COLORS['text_primary']};
            border-bottom: 2px solid {DARK_MODE_COLORS['pastel_cyan']};
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        h2 {{
            color: {DARK_MODE_COLORS['text_primary']};
            margin-top: 30px;
            margin-bottom: 15px;
        }}

        h3 {{
            color: {DARK_MODE_COLORS['pastel_cyan']};
            margin-top: 20px;
            margin-bottom: 10px;
        }}

        /* ===== TAB NAVIGATION ===== */
        .tab-navigation {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            background-color: {DARK_MODE_COLORS['paper_background']};
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid {DARK_MODE_COLORS['grid_lines']};
        }}

        .tab-button {{
            background-color: {DARK_MODE_COLORS['background']};
            color: {DARK_MODE_COLORS['text_secondary']};
            border: 1px solid {DARK_MODE_COLORS['grid_lines']};
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 5px;
            font-size: 14px;
            transition: all 0.2s ease;
        }}

        .tab-button:hover {{
            background-color: {DARK_MODE_COLORS['grid_lines']};
            color: {DARK_MODE_COLORS['text_primary']};
        }}

        .tab-button.active {{
            background-color: {DARK_MODE_COLORS['pastel_cyan']};
            color: {DARK_MODE_COLORS['background']};
            border-color: {DARK_MODE_COLORS['pastel_cyan']};
            font-weight: bold;
        }}

        /* ===== TAB CONTENT ===== */
        .tab-content {{
            display: none;
            padding: 20px;
            background-color: {DARK_MODE_COLORS['paper_background']};
            border-radius: 8px;
            border: 1px solid {DARK_MODE_COLORS['grid_lines']};
            margin-bottom: 20px;
        }}

        .tab-content.active {{
            display: block;
        }}

        /* ===== CHART CONTAINERS ===== */
        .chart-container {{
            margin-bottom: 30px;
            padding: 15px;
            background-color: {DARK_MODE_COLORS['background']};
            border-radius: 8px;
            border: 1px solid {DARK_MODE_COLORS['grid_lines']};
        }}

        .chart-container h3 {{
            margin-top: 0;
            margin-bottom: 15px;
        }}

        /* ===== TABLES ===== */
        .table-section {{
            margin-bottom: 30px;
        }}

        .table-scroll-container {{
            overflow-x: auto;
            max-width: 100%;
        }}

        .dark-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background-color: {DARK_MODE_COLORS['paper_background']};
        }}

        .dark-table th {{
            background-color: {DARK_MODE_COLORS['table_header_bg']};
            color: {DARK_MODE_COLORS['pastel_cyan']};
            padding: 12px 10px;
            text-align: left;
            border-bottom: 2px solid {DARK_MODE_COLORS['pastel_cyan']};
            white-space: nowrap;
        }}

        .dark-table td {{
            padding: 10px;
            border-bottom: 1px solid {DARK_MODE_COLORS['grid_lines']};
            color: {DARK_MODE_COLORS['text_primary']};
        }}

        .dark-table tr:nth-child(even) {{
            background-color: {DARK_MODE_COLORS['table_row_alt']};
        }}

        .dark-table tr:hover {{
            background-color: {DARK_MODE_COLORS['grid_lines']};
        }}

        /* ===== QUERY REFERENCE ===== */
        .query-reference-section {{
            margin-top: 20px;
        }}

        .query-reference-item {{
            background-color: {DARK_MODE_COLORS['background']};
            border: 1px solid {DARK_MODE_COLORS['grid_lines']};
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}

        .query-reference-item h3 {{
            margin-top: 0;
        }}

        .query-text-box {{
            background-color: {DARK_MODE_COLORS['paper_background']};
            border: 1px solid {DARK_MODE_COLORS['grid_lines']};
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }}

        .query-text-box pre {{
            margin: 5px 0 0 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
        }}

        /* ===== UTILITY CLASSES ===== */
        .no-data-message {{
            color: {DARK_MODE_COLORS['text_secondary']};
            font-style: italic;
            padding: 20px;
            text-align: center;
            background-color: {DARK_MODE_COLORS['background']};
            border-radius: 8px;
            border: 1px dashed {DARK_MODE_COLORS['grid_lines']};
        }}

        .dashboard-header {{
            margin-bottom: 30px;
        }}

        .dashboard-header p {{
            color: {DARK_MODE_COLORS['text_secondary']};
            margin-top: 10px;
        }}
    </style>
    """

    return css_stylesheet


def generate_javascript_for_tabs() -> str:
    """
    Generates the JavaScript code for tab switching functionality.

    This creates a simple, lightweight tab system without external dependencies.

    Returns:
        str: JavaScript code as a string to be embedded in the HTML.
    """
    javascript_code = """
    <script>
        /**
         * Switches the visible tab content based on the clicked tab button.
         *
         * @param {Event} event - The click event from the tab button.
         * @param {string} tabId - The ID of the tab content to display.
         */
        function switchTab(event, tabId) {
            // Hide all tab contents
            const allTabContents = document.querySelectorAll('.tab-content');
            allTabContents.forEach(function(content) {
                content.classList.remove('active');
            });

            // Remove active class from all tab buttons
            const allTabButtons = document.querySelectorAll('.tab-button');
            allTabButtons.forEach(function(button) {
                button.classList.remove('active');
            });

            // Show the selected tab content
            const selectedTabContent = document.getElementById(tabId);
            if (selectedTabContent) {
                selectedTabContent.classList.add('active');
            }

            // Add active class to the clicked button
            event.currentTarget.classList.add('active');

            // Trigger resize event to ensure Plotly charts render correctly
            // (Charts may not render properly if hidden when page loads)
            window.dispatchEvent(new Event('resize'));
        }

        // Initialize first tab as active on page load
        document.addEventListener('DOMContentLoaded', function() {
            const firstTabButton = document.querySelector('.tab-button');
            const firstTabContent = document.querySelector('.tab-content');

            if (firstTabButton) {
                firstTabButton.classList.add('active');
            }
            if (firstTabContent) {
                firstTabContent.classList.add('active');
            }
        });
    </script>
    """

    return javascript_code


def convert_figure_to_html_div(figure: go.Figure | None, fallback_message: str = "No data available for this visualization.") -> str:
    """
    Converts a Plotly Figure to an HTML div string for embedding.

    If the figure is None (indicating no data or an error), returns
    a styled fallback message.

    Args:
        figure (go.Figure | None): The Plotly figure to convert, or None.
        fallback_message (str): Message to display if figure is None.

    Returns:
        str: HTML div string containing the chart or fallback message.
    """
    try:
        if figure is None:
            return f'<div class="no-data-message">{fallback_message}</div>'

        # Convert figure to HTML without the full HTML document structure
        # include_plotlyjs='cdn' uses CDN for smaller file size
        html_div = figure.to_html(
            full_html=False,
            include_plotlyjs=False,  # We'll include it once in the head
            div_id=None  # Let Plotly generate unique IDs
        )

        return f'<div class="chart-container">{html_div}</div>'

    except Exception as general_exception:
        logger.error("Failed to convert figure to HTML div.")
        logger.error(traceback.format_exc())
        return f'<div class="no-data-message">Error rendering chart: {general_exception}</div>'


def assemble_complete_html_dashboard(
    summary_dataframe: pd.DataFrame,
    diagnostic_dataframe: pd.DataFrame,
    all_charts: dict[str, list[tuple[str, go.Figure | None]]]
) -> str:
    """
    Assembles all components into a complete HTML dashboard document.

    This function combines CSS styles, JavaScript, tab navigation,
    all charts organized by tab, and data tables into a single
    cohesive HTML document.

    Args:
        summary_dataframe (pd.DataFrame): The summary data for tables.
        diagnostic_dataframe (pd.DataFrame): The diagnostic data for tables.
        all_charts (dict[str, list[tuple[str, go.Figure | None]]]):
            A dictionary where keys are tab names and values are lists
            of (chart_title, figure) tuples.

    Returns:
        str: A complete HTML document as a string.
    """
    try:
        logger.info("Assembling complete HTML dashboard document.")

        # Generate CSS
        css_styles = generate_css_styles()

        # Generate JavaScript
        javascript_code = generate_javascript_for_tabs()

        # Generate tab navigation buttons
        tab_buttons_html = ""
        tab_contents_html = ""

        tab_names = list(all_charts.keys())

        for tab_index, tab_name in enumerate(tab_names):
            tab_id = f"tab-{tab_index}"
            active_class = "active" if tab_index == 0 else ""

            # Create tab button
            tab_buttons_html += f"""
            <button class="tab-button {active_class}" onclick="switchTab(event, '{tab_id}')">
                {tab_name}
            </button>
            """

            # Create tab content
            charts_in_tab = all_charts[tab_name]
            charts_html = ""

            for chart_title, chart_figure in charts_in_tab:
                chart_html_div = convert_figure_to_html_div(
                    chart_figure,
                    fallback_message=f"No data available for: {chart_title}"
                )
                charts_html += chart_html_div

            tab_contents_html += f"""
            <div id="{tab_id}" class="tab-content {active_class}">
                <h2>{tab_name}</h2>
                {charts_html}
            </div>
            """

        # Generate data tables tab content
        summary_table_html = generate_dark_mode_html_table(
            summary_dataframe,
            table_id="summary-data-table",
            table_title="Summary Report Data"
        )

        diagnostic_table_html = generate_dark_mode_html_table(
            diagnostic_dataframe,
            table_id="diagnostic-data-table",
            table_title="Diagnostic Report Data"
        )

        query_reference_html = generate_query_reference_html(summary_dataframe)

        # Add Raw Data tab
        data_tab_id = f"tab-{len(tab_names)}"
        tab_buttons_html += f"""
        <button class="tab-button" onclick="switchTab(event, '{data_tab_id}')">
            Raw Data Tables
        </button>
        """

        tab_contents_html += f"""
        <div id="{data_tab_id}" class="tab-content">
            <h2>Raw Data Tables</h2>
            {summary_table_html}
            {diagnostic_table_html}
        </div>
        """

        # Add Query Reference tab
        reference_tab_id = f"tab-{len(tab_names) + 1}"
        tab_buttons_html += f"""
        <button class="tab-button" onclick="switchTab(event, '{reference_tab_id}')">
            Query Reference
        </button>
        """

        tab_contents_html += f"""
        <div id="{reference_tab_id}" class="tab-content">
            {query_reference_html}
        </div>
        """

        # Assemble the complete HTML document
        complete_html_document = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Diagnostic Results Dashboard - Dark Mode</title>

            <!-- Plotly.js from CDN -->
            <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

            {css_styles}
        </head>
        <body>
            <div class="dashboard-header">
                <h1>Diagnostic Results Dashboard</h1>
                <p>
                    Interactive visualization of query performance metrics, similarity statistics,
                    and diagnostic error analysis. Use the tabs below to navigate between sections.
                </p>
            </div>

            <nav class="tab-navigation">
                {tab_buttons_html}
            </nav>

            <main>
                {tab_contents_html}
            </main>

            {javascript_code}
        </body>
        </html>
        """

        logger.info("Successfully assembled complete HTML dashboard document.")
        return complete_html_document

    except Exception as general_exception:
        logger.error("Failed to assemble complete HTML dashboard.")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Dashboard assembly failed: {general_exception}") from general_exception


def write_html_dashboard_to_file(html_content: str, output_file_path: str) -> None:
    """
    Writes the complete HTML dashboard content to a file.

    Args:
        html_content (str): The complete HTML document as a string.
        output_file_path (str): The desired file path for the output HTML file.

    Raises:
        RuntimeError: If file writing permissions or I/O errors occur.
    """
    try:
        logger.info(f"Writing HTML dashboard to file: {output_file_path}")

        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(html_content)

        # Get file size for logging
        file_size_bytes = os.path.getsize(output_file_path)
        file_size_kb = file_size_bytes / 1024

        logger.info(f"Successfully wrote HTML dashboard to '{output_file_path}' ({file_size_kb:.2f} KB).")

    except PermissionError as permission_error:
        logger.error(f"Permission denied writing to '{output_file_path}'.")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Permission denied: {permission_error}") from permission_error
    except Exception as general_exception:
        logger.error(f"Failed to write HTML dashboard to '{output_file_path}'.")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Dashboard file writing failed: {general_exception}") from general_exception


# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def generate_all_charts(
    summary_dataframe: pd.DataFrame,
    diagnostic_dataframe: pd.DataFrame
) -> dict[str, list[tuple[str, go.Figure | None]]]:
    """
    Generates all charts organized by tab category.

    This function creates every visualization and organizes them into
    a dictionary structure suitable for the tab-based dashboard layout.

    Args:
        summary_dataframe (pd.DataFrame): The summary data for chart generation.
        diagnostic_dataframe (pd.DataFrame): The diagnostic data for chart generation.

    Returns:
        dict[str, list[tuple[str, go.Figure | None]]]: A dictionary where:
            - Keys are tab names (strings)
            - Values are lists of (chart_title, figure) tuples
    """
    try:
        logger.info("Generating all dashboard charts organized by tab category.")

        charts_by_tab = {}

        # ----- TAB: Classification Metrics -----
        classification_metrics_charts = [
            ("TP/FP/FN/TN Counts", create_chart_tp_fp_fn_tn_counts(summary_dataframe)),
            ("Precision", create_chart_precision_per_query(summary_dataframe)),
            ("Recall", create_chart_recall_per_query(summary_dataframe)),
            ("F1 Score", create_chart_f1_score_per_query(summary_dataframe)),
            ("Accuracy", create_chart_accuracy_per_query(summary_dataframe)),
            ("Specificity", create_chart_specificity_per_query(summary_dataframe)),
        ]
        charts_by_tab["Classification Metrics"] = classification_metrics_charts

        # ----- TAB: Count Analysis -----
        count_analysis_charts = [
            ("Tabular vs Vector Counts", create_chart_tabular_vs_vector_counts(summary_dataframe)),
            ("Absolute & Percent Error", create_chart_absolute_and_percent_error(summary_dataframe)),
            ("Retrieval Direction", create_chart_direction_indicator(summary_dataframe)),
        ]
        charts_by_tab["Count Analysis"] = count_analysis_charts

        # ----- TAB: Similarity Statistics -----
        similarity_stats_charts = [
            ("Similarity Statistics (Combined)", create_chart_similarity_statistics_combined(summary_dataframe)),
            ("Similarity Range (Min-Max)", create_chart_similarity_range_per_query(summary_dataframe)),
        ]
        charts_by_tab["Similarity Statistics"] = similarity_stats_charts

        # ----- TAB: Timing -----
        timing_charts = [
            ("Query Execution Time", create_chart_elapsed_seconds_per_query(summary_dataframe)),
        ]
        charts_by_tab["Timing"] = timing_charts

        # ----- TAB: Diagnostic Analysis -----
        # This tab may be empty if no diagnostic data exists (no errors)
        diagnostic_analysis_charts = [
            ("Error Contribution Proportion", create_chart_error_contribution_proportion(diagnostic_dataframe)),
            ("FP Rows Affected Proportion", create_chart_fp_rows_affected_proportion(diagnostic_dataframe)),
            ("Wrong Count by Field", create_chart_wrong_count_by_field(diagnostic_dataframe)),
            ("Error Contribution (Stacked)", create_chart_diagnostic_stacked_proportions(diagnostic_dataframe)),
        ]
        charts_by_tab["Diagnostic Analysis"] = diagnostic_analysis_charts

        # Count total charts generated
        total_chart_count = sum(len(charts) for charts in charts_by_tab.values())
        non_null_chart_count = sum(
            1 for charts in charts_by_tab.values()
            for _, fig in charts if fig is not None
        )

        logger.info(f"Generated {non_null_chart_count} charts with data out of {total_chart_count} total chart slots.")

        return charts_by_tab

    except Exception as general_exception:
        logger.error("Failed to generate all charts.")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Chart generation failed: {general_exception}") from general_exception


def generate_dashboard() -> None:
    """
    Main orchestration function for dashboard generation.

    This function:
    1. Reads environment variables for file path configurations
    2. Discovers and loads summary and diagnostic data files
    3. Generates all visualization charts
    4. Assembles the complete HTML dashboard
    5. Writes the output file

    Environment Variables:
        SUMMARY_FILE_PATTERN: Glob pattern for summary CSV files (default: "summary_report_*.csv")
        DIAGNOSTIC_FILE_PATTERN: Glob pattern for diagnostic CSV files (default: "*_fp_diagnostic_summary_*.csv")
        OUTPUT_HTML_PATH: Output file path (default: "diagnostic_dashboard_v2.html")

    Raises:
        SystemExit: If critical errors occur that prevent dashboard generation.
    """
    # Fetch configurations from environment variables
    # Using environment variables prevents hardcoding and allows flexible deployment
    summary_file_pattern = os.environ.get("SUMMARY_FILE_PATTERN", "summary_report_*.csv")
    diagnostic_file_pattern = os.environ.get("DIAGNOSTIC_FILE_PATTERN", "*_fp_diagnostic_summary_*.csv")
    output_html_path = os.environ.get("OUTPUT_HTML_PATH", "diagnostic_dashboard_v2.html")

    try:
        logger.info("=" * 60)
        logger.info("Starting dark-mode dashboard generation sequence.")
        logger.info("=" * 60)

        # Log configuration for transparency
        logger.info(f"Configuration - Summary pattern: {summary_file_pattern}")
        logger.info(f"Configuration - Diagnostic pattern: {diagnostic_file_pattern}")
        logger.info(f"Configuration - Output path: {output_html_path}")

        # ----- STEP 1: Discover and Load Summary File -----
        logger.info("-" * 40)
        logger.info("STEP 1: Discovering summary report files.")

        summary_files = find_matching_files(summary_file_pattern)

        if not summary_files:
            logger.error("No summary report files found. Cannot proceed without core data.")
            logger.error(f"Searched pattern: {summary_file_pattern}")
            sys.exit(1)

        # Use the first matched summary file if multiple exist
        # Log warning if multiple found for transparency
        if len(summary_files) > 1:
            logger.warning(f"Found {len(summary_files)} summary files. Using first: {summary_files[0]}")

        primary_summary_file = summary_files[0]
        summary_df = load_summary_report(primary_summary_file)

        # ----- STEP 2: Discover and Load Diagnostic Files -----
        logger.info("-" * 40)
        logger.info("STEP 2: Discovering diagnostic report files.")

        diagnostic_files = find_matching_files(diagnostic_file_pattern)

        if not diagnostic_files:
            logger.info("No diagnostic files found. This may indicate no errors occurred (which is good).")
            logger.info("Diagnostic analysis charts will show 'no data' messages.")

        diagnostic_df = load_diagnostic_reports(diagnostic_files)

        # ----- STEP 3: Generate All Charts -----
        logger.info("-" * 40)
        logger.info("STEP 3: Generating visualization charts.")

        all_charts_by_tab = generate_all_charts(summary_df, diagnostic_df)

        # ----- STEP 4: Assemble HTML Dashboard -----
        logger.info("-" * 40)
        logger.info("STEP 4: Assembling HTML dashboard document.")

        complete_html_content = assemble_complete_html_dashboard(
            summary_dataframe=summary_df,
            diagnostic_dataframe=diagnostic_df,
            all_charts=all_charts_by_tab
        )

        # ----- STEP 5: Write Output File -----
        logger.info("-" * 40)
        logger.info("STEP 5: Writing HTML dashboard to file.")

        write_html_dashboard_to_file(complete_html_content, output_html_path)

        # ----- COMPLETION -----
        logger.info("=" * 60)
        logger.info("Dashboard generation sequence completed successfully.")
        logger.info(f"Output file: {output_html_path}")
        logger.info("=" * 60)

    except FileNotFoundError as file_error:
        logger.critical(f"Required file not found: {file_error}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
    except RuntimeError as runtime_error:
        logger.critical(f"Runtime error during dashboard generation: {runtime_error}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
    except Exception as general_exception:
        logger.critical("An unexpected critical failure occurred during dashboard generation.")
        logger.critical(f"Exception type: {type(general_exception).__name__}")
        logger.critical(f"Exception message: {general_exception}")
        logger.critical(traceback.format_exc())
        sys.exit(1)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    generate_dashboard()
