"""
csv_summary_blurb_augmentor.py

Purpose:
--------
This script reads a structured CSV dataset containing animal records,
and produces an augmented copy with an additional column containing
unstructured natural language descriptions ("blurbs") of each record.

Intent:
-------
The goal is to create a dataset that contains BOTH structured tabular data
AND unstructured text descriptions of the same information. This augmented
dataset can then be used to test and compare vector-database analytics
capabilities against known ground-truth structured data.

By deterministically generating text blurbs from known structured fields,
we create a controlled environment where:
- We KNOW what information is in each blurb
- We can verify if vector search finds the correct documents
- We can test corpus analytics questions with known correct answers

Input:
------
A CSV file with the following fields:
    animal_type, weight_kg, height_cm, age_years, number_of_friends,
    birth_date, birth_unix, color, can_fly, can_swim, can_run,
    watches_youtube, daily_food_grams, popularity_score, social_media

Output:
-------
A new CSV file with all original columns plus one new column:
    unstructured_description

The unstructured_description contains a natural language blurb that
describes the structured data using one of three rotating templates.

Design Decisions:
-----------------
- Template selection is deterministic: row_index % 3
- Float values are rounded to 3 decimal places in blurbs
- Boolean fields are converted to natural language phrases
- Original data is NOT modified; a new file is created
- No randomness; output is fully reproducible

Author: Synthetic Biology Datasets Project
"""

import csv
import traceback
import os
import sys


def round_float_to_three_decimals(value_to_round: float) -> float:
    """
    Round a floating point number to exactly 3 decimal places.

    Intent:
    -------
    Float values in the source CSV often have many decimal places
    (e.g., 28.602669690571616). For human-readable blurbs, we want
    cleaner numbers (e.g., 28.603). Three decimal places provides
    sufficient precision while maintaining readability.

    Parameters:
    -----------
    value_to_round : float
        The floating point number to be rounded.

    Returns:
    --------
    float
        The input value rounded to 3 decimal places.

    Example:
    --------
    >>> round_float_to_three_decimals(28.602669690571616)
    28.603
    """
    rounded_value = round(value_to_round, 3)
    return rounded_value


def convert_boolean_string_to_python_bool(boolean_string: str) -> bool:
    """
    Convert a string representation of a boolean to a Python bool.

    Intent:
    -------
    CSV files store all values as strings. When we read "True" or "False"
    from a CSV, we get the string "True", not the Python boolean True.
    This function handles that conversion safely.

    Parameters:
    -----------
    boolean_string : str
        A string that should be either "True" or "False" (case-sensitive
        as produced by Python's str(bool) conversion).

    Returns:
    --------
    bool
        True if the string is "True", False otherwise.

    Notes:
    ------
    This function treats any value that is not exactly "True" as False.
    This is a deliberate safety choice: unknown values default to False
    rather than raising an exception, making the pipeline more robust
    to minor data variations.
    """
    if boolean_string == "True":
        return True
    else:
        return False


def convert_can_ability_bool_to_phrase(can_do_ability: bool) -> str:
    """
    Convert a boolean ability flag to a natural language phrase.

    Intent:
    -------
    For abilities like "can_fly", "can_swim", "can_run", we want to
    produce readable phrases like "can" or "cannot" for use in
    natural language blurbs.

    Parameters:
    -----------
    can_do_ability : bool
        True if the animal has this ability, False otherwise.

    Returns:
    --------
    str
        "can" if True, "cannot" if False.

    Example:
    --------
    >>> convert_can_ability_bool_to_phrase(True)
    "can"
    >>> convert_can_ability_bool_to_phrase(False)
    "cannot"
    """
    if can_do_ability:
        return "can"
    else:
        return "cannot"


def convert_watches_youtube_bool_to_phrase(watches_youtube: bool) -> str:
    """
    Convert the watches_youtube boolean to a natural language phrase.

    Intent:
    -------
    The "watches_youtube" field needs a slightly different phrasing
    than abilities. We use "likes to" / "does not like to" to make
    grammatically correct sentences like "It likes to watch youtube."

    Parameters:
    -----------
    watches_youtube : bool
        True if the animal watches youtube, False otherwise.

    Returns:
    --------
    str
        "likes to" if True, "does not like to" if False.

    Example:
    --------
    >>> convert_watches_youtube_bool_to_phrase(True)
    "likes to"
    >>> convert_watches_youtube_bool_to_phrase(False)
    "does not like to"
    """
    if watches_youtube:
        return "likes to"
    else:
        return "does not like to"


def select_template_index_for_row(row_index: int) -> int:
    """
    Deterministically select which template (0, 1, or 2) to use for a row.

    Intent:
    -------
    We want variety in the blurb formats (to make vector search more
    realistic/challenging), but we also want reproducibility (so tests
    are repeatable). Using modulo arithmetic on the row index achieves
    both goals: the template varies by row, but the same row always
    gets the same template.

    Parameters:
    -----------
    row_index : int
        The zero-based index of the current row in the dataset.

    Returns:
    --------
    int
        An integer 0, 1, or 2 indicating which template to use.

    Example:
    --------
    >>> select_template_index_for_row(0)
    0
    >>> select_template_index_for_row(1)
    1
    >>> select_template_index_for_row(2)
    2
    >>> select_template_index_for_row(3)
    0
    """
    template_index = row_index % 3
    return template_index


def generate_blurb_using_template_zero(
    animal_type: str,
    weight_kg_rounded: float,
    height_cm_rounded: float,
    age_years: int,
    number_of_friends: int,
    birth_date: str,
    color: str,
    can_fly_phrase: str,
    can_swim_phrase: str,
    watches_youtube_phrase: str,
    daily_food_grams_rounded: float,
    popularity_score_rounded: float
) -> str:
    """
    Generate an unstructured text blurb using template format 0.

    Intent:
    -------
    This is the first of three template formats. It presents information
    in a specific order starting with animal type, then food, age, friends,
    physical characteristics, and abilities.

    Parameters:
    -----------
    animal_type : str
        The type of animal (e.g., "cat", "dog", "bird").
    weight_kg_rounded : float
        The animal's weight in kg, rounded to 3 decimal places.
    height_cm_rounded : float
        The animal's height in cm, rounded to 3 decimal places.
    age_years : int
        The animal's age in years.
    number_of_friends : int
        How many friends the animal has.
    birth_date : str
        The animal's birth date as a string (e.g., "2021-01-25").
    color : str
        The animal's color.
    can_fly_phrase : str
        "can" or "cannot" for flying ability.
    can_swim_phrase : str
        "can" or "cannot" for swimming ability.
    watches_youtube_phrase : str
        "likes to" or "does not like to" for youtube watching.
    daily_food_grams_rounded : float
        Daily food allowance in grams, rounded to 3 decimal places.
    popularity_score_rounded : float
        Popularity score, rounded to 3 decimal places.

    Returns:
    --------
    str
        A natural language blurb describing the animal.
    """
    blurb_text = (
        f"This is about a {animal_type}. "
        f"It gets a daily allowance of {daily_food_grams_rounded} grams of food. "
        f"It is {age_years} years old, and has {number_of_friends} friends. "
        f"This {animal_type} weighs {weight_kg_rounded} kg, and is {height_cm_rounded} cm tall. "
        f"It has a popularity_score of {popularity_score_rounded}. "
        f"It was born on {birth_date}, and is {color}. "
        f"It {can_fly_phrase} fly, and {can_swim_phrase} swim. "
        f"It {watches_youtube_phrase} watch youtube."
    )
    return blurb_text


def generate_blurb_using_template_one(
    animal_type: str,
    weight_kg_rounded: float,
    height_cm_rounded: float,
    age_years: int,
    number_of_friends: int,
    birth_date: str,
    color: str,
    can_fly_phrase: str,
    can_swim_phrase: str,
    watches_youtube_phrase: str,
    daily_food_grams_rounded: float,
    popularity_score_rounded: float
) -> str:
    """
    Generate an unstructured text blurb using template format 1.

    Intent:
    -------
    This is the second of three template formats. It presents information
    in a different order than template 0, and uses exclamation marks for
    some ability statements to create textual variety.

    Parameters:
    -----------
    (Same as generate_blurb_using_template_zero)

    Returns:
    --------
    str
        A natural language blurb describing the animal.
    """
    blurb_text = (
        f"This {animal_type} weighs {weight_kg_rounded} kg, and is {height_cm_rounded} cm tall. "
        f"It is {color} and was born on {birth_date}. "
        f"It gets {daily_food_grams_rounded} grams of food each day. "
        f"It is {age_years} years old. "
        f"It {can_fly_phrase} fly! "
        f"It {can_swim_phrase} swim! "
        f"This {animal_type} has {number_of_friends} friends and has a popularity_score of {popularity_score_rounded}. "
        f"It {watches_youtube_phrase} watch youtube."
    )
    return blurb_text


def generate_blurb_using_template_two(
    animal_type: str,
    weight_kg_rounded: float,
    height_cm_rounded: float,
    age_years: int,
    number_of_friends: int,
    birth_date: str,
    color: str,
    can_fly_phrase: str,
    can_swim_phrase: str,
    watches_youtube_phrase: str,
    daily_food_grams_rounded: float,
    popularity_score_rounded: float
) -> str:
    """
    Generate an unstructured text blurb using template format 2.

    Intent:
    -------
    This is the third of three template formats. It presents information
    in yet another order, with different punctuation patterns (exclamation
    at the start, periods for abilities in the middle).

    Parameters:
    -----------
    (Same as generate_blurb_using_template_zero)

    Returns:
    --------
    str
        A natural language blurb describing the animal.
    """
    blurb_text = (
        f"This is a {animal_type}! "
        f"It is {height_cm_rounded} cm tall. "
        f"It gets {daily_food_grams_rounded} grams of food each day and weighs {weight_kg_rounded} kg! "
        f"It is {color}, and it was born on {birth_date}. "
        f"It is {age_years} years old! "
        f"It {can_swim_phrase} swim. "
        f"It {can_fly_phrase} fly. "
        f"It {watches_youtube_phrase} watch youtube. "
        f"It has {number_of_friends} friends and has a popularity_score of {popularity_score_rounded}."
    )
    return blurb_text


def generate_blurb_for_row(row_data: dict, row_index: int) -> str:
    """
    Generate an unstructured text blurb for a single row of data.

    Intent:
    -------
    This function is the main orchestrator for blurb generation. It:
    1. Extracts and converts all necessary fields from the row
    2. Rounds float values to 3 decimal places
    3. Converts boolean strings to natural language phrases
    4. Selects the appropriate template based on row index
    5. Calls the appropriate template function

    This function encapsulates all the data transformation logic,
    keeping the main processing loop clean and simple.

    Parameters:
    -----------
    row_data : dict
        A dictionary containing one row of CSV data, where keys are
        column names and values are string representations of the data.
    row_index : int
        The zero-based index of this row in the dataset.

    Returns:
    --------
    str
        A natural language blurb describing the animal in this row.

    Raises:
    -------
    ValueError
        If required fields are missing or cannot be converted.
    KeyError
        If expected column names are not present in row_data.
    """
    # Extract and convert string values to appropriate types
    # Note: CSV reader returns all values as strings

    animal_type = row_data["animal_type"]
    color = row_data["color"]
    birth_date = row_data["birth_date"]

    # Convert numeric strings to numbers, then round floats
    weight_kg_raw = float(row_data["weight_kg"])
    weight_kg_rounded = round_float_to_three_decimals(weight_kg_raw)

    height_cm_raw = float(row_data["height_cm"])
    height_cm_rounded = round_float_to_three_decimals(height_cm_raw)

    daily_food_grams_raw = float(row_data["daily_food_grams"])
    daily_food_grams_rounded = round_float_to_three_decimals(daily_food_grams_raw)

    popularity_score_raw = float(row_data["popularity_score"])
    popularity_score_rounded = round_float_to_three_decimals(popularity_score_raw)

    # Integer fields
    age_years = int(row_data["age_years"])
    number_of_friends = int(row_data["number_of_friends"])

    # Convert boolean strings to Python bools, then to phrases
    can_fly_bool = convert_boolean_string_to_python_bool(row_data["can_fly"])
    can_fly_phrase = convert_can_ability_bool_to_phrase(can_fly_bool)

    can_swim_bool = convert_boolean_string_to_python_bool(row_data["can_swim"])
    can_swim_phrase = convert_can_ability_bool_to_phrase(can_swim_bool)

    watches_youtube_bool = convert_boolean_string_to_python_bool(row_data["watches_youtube"])
    watches_youtube_phrase = convert_watches_youtube_bool_to_phrase(watches_youtube_bool)

    # Select template based on row index
    template_index = select_template_index_for_row(row_index)

    # Generate blurb using the selected template
    if template_index == 0:
        generated_blurb = generate_blurb_using_template_zero(
            animal_type=animal_type,
            weight_kg_rounded=weight_kg_rounded,
            height_cm_rounded=height_cm_rounded,
            age_years=age_years,
            number_of_friends=number_of_friends,
            birth_date=birth_date,
            color=color,
            can_fly_phrase=can_fly_phrase,
            can_swim_phrase=can_swim_phrase,
            watches_youtube_phrase=watches_youtube_phrase,
            daily_food_grams_rounded=daily_food_grams_rounded,
            popularity_score_rounded=popularity_score_rounded
        )
    elif template_index == 1:
        generated_blurb = generate_blurb_using_template_one(
            animal_type=animal_type,
            weight_kg_rounded=weight_kg_rounded,
            height_cm_rounded=height_cm_rounded,
            age_years=age_years,
            number_of_friends=number_of_friends,
            birth_date=birth_date,
            color=color,
            can_fly_phrase=can_fly_phrase,
            can_swim_phrase=can_swim_phrase,
            watches_youtube_phrase=watches_youtube_phrase,
            daily_food_grams_rounded=daily_food_grams_rounded,
            popularity_score_rounded=popularity_score_rounded
        )
    else:
        generated_blurb = generate_blurb_using_template_two(
            animal_type=animal_type,
            weight_kg_rounded=weight_kg_rounded,
            height_cm_rounded=height_cm_rounded,
            age_years=age_years,
            number_of_friends=number_of_friends,
            birth_date=birth_date,
            color=color,
            can_fly_phrase=can_fly_phrase,
            can_swim_phrase=can_swim_phrase,
            watches_youtube_phrase=watches_youtube_phrase,
            daily_food_grams_rounded=daily_food_grams_rounded,
            popularity_score_rounded=popularity_score_rounded
        )

    return generated_blurb


def read_csv_file_to_list_of_dicts(input_file_path: str) -> tuple[list[dict], list[str]]:
    """
    Read a CSV file and return its contents as a list of dictionaries.

    Intent:
    -------
    This function handles the file I/O for reading the input CSV.
    It uses csv.DictReader which automatically uses the first row
    as column headers and returns each subsequent row as a dictionary.

    We also return the fieldnames separately so we can preserve
    column order when writing the output file.

    Parameters:
    -----------
    input_file_path : str
        The path to the input CSV file.

    Returns:
    --------
    tuple[list[dict], list[str]]
        A tuple containing:
        - A list of dictionaries, one per row
        - A list of field names (column headers) in original order

    Raises:
    -------
    FileNotFoundError
        If the input file does not exist.
    PermissionError
        If the file cannot be read due to permissions.
    ValueError
        If the CSV file is empty or has no header row.
    """
    list_of_row_dicts = []
    fieldnames_list = []

    with open(input_file_path, mode='r', encoding='utf-8', newline='') as csv_file_handle:
        csv_dict_reader = csv.DictReader(csv_file_handle)

        # Read all rows into memory as dictionaries
        for row_dict in csv_dict_reader:
            list_of_row_dicts.append(row_dict)

        # Capture the fieldnames from the reader AFTER reading rows
        # This ensures fieldnames is populated (it's None before first read)
        reader_fieldnames = csv_dict_reader.fieldnames

        # Explicit check for None to satisfy type checker and handle edge case
        if reader_fieldnames is None:
            raise ValueError(
                f"CSV file appears to be empty or has no header row: {input_file_path}"
            )

        fieldnames_list = list(reader_fieldnames)

    return list_of_row_dicts, fieldnames_list


def write_augmented_data_to_csv(
    output_file_path: str,
    augmented_rows: list[dict],
    output_fieldnames: list[str]
) -> None:
    """
    Write the augmented data to a new CSV file.

    Intent:
    -------
    This function handles the file I/O for writing the output CSV.
    It takes the list of augmented row dictionaries (which now include
    the unstructured_description column) and writes them to a new file.

    The original file is never modified; we always write to a new file.

    Parameters:
    -----------
    output_file_path : str
        The path where the output CSV file should be written.
    augmented_rows : list[dict]
        A list of dictionaries, one per row, including the new
        unstructured_description field.
    output_fieldnames : list[str]
        The list of column names in the order they should appear
        in the output file.

    Returns:
    --------
    None

    Raises:
    -------
    PermissionError
        If the output file cannot be written due to permissions.
    OSError
        If there are other I/O errors (disk full, etc.).
    """
    with open(output_file_path, mode='w', encoding='utf-8', newline='') as csv_file_handle:
        csv_dict_writer = csv.DictWriter(
            csv_file_handle,
            fieldnames=output_fieldnames
        )

        # Write the header row
        csv_dict_writer.writeheader()

        # Write all data rows
        for row_dict in augmented_rows:
            csv_dict_writer.writerow(row_dict)


def generate_output_file_path(input_file_path: str) -> str:
    """
    Generate the output file path based on the input file path.

    Intent:
    -------
    We want a predictable, non-destructive naming convention for output
    files. This function takes the input path and produces an output
    path with "_augmented" inserted before the file extension.

    Example: "dataset.csv" -> "dataset_augmented.csv"
    Example: "data/my_file.csv" -> "data/my_file_augmented.csv"

    Parameters:
    -----------
    input_file_path : str
        The path to the input CSV file.

    Returns:
    --------
    str
        The generated output file path.
    """
    # Split the path into directory, filename, and extension
    directory_path = os.path.dirname(input_file_path)
    filename_with_extension = os.path.basename(input_file_path)

    # Split filename and extension
    filename_without_extension, file_extension = os.path.splitext(filename_with_extension)

    # Construct the new filename
    augmented_filename = f"{filename_without_extension}_augmented{file_extension}"

    # Reconstruct the full path
    if directory_path:
        output_file_path = os.path.join(directory_path, augmented_filename)
    else:
        output_file_path = augmented_filename

    return output_file_path


def process_csv_and_add_blurbs(input_file_path: str, output_file_path: str) -> int:
    """
    Main processing function: read CSV, add blurbs, write augmented CSV.

    Intent:
    -------
    This is the main orchestration function that coordinates the entire
    augmentation process:
    1. Read the input CSV
    2. For each row, generate a blurb
    3. Add the blurb to the row data
    4. Write all augmented rows to the output CSV

    This function is separate from main() to allow for easier testing
    and reuse. It contains the core business logic without command-line
    argument handling.

    Parameters:
    -----------
    input_file_path : str
        The path to the input CSV file.
    output_file_path : str
        The path where the augmented CSV should be written.

    Returns:
    --------
    int
        The number of rows processed.

    Raises:
    -------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the CSV data cannot be processed (missing fields, bad values).
    """
    # Read the input CSV
    print(f"Reading input file: {input_file_path}")
    rows_list, original_fieldnames = read_csv_file_to_list_of_dicts(input_file_path)
    print(f"Found {len(rows_list)} rows to process")

    # Prepare the output fieldnames (original plus new column)
    new_column_name = "unstructured_description"
    output_fieldnames = original_fieldnames + [new_column_name]

    # Process each row: generate blurb and add to row dict
    augmented_rows_list = []

    """
    To prevent status prints from being too abundant
    as data files are typically not of piceune quantity,
    getting list-length//10, and using:
        if (row_index + 1) % short_status_base == 0:
    should give usually 10 status prints, also fitting
    within 80x24 terminal size.
    """
    short_status_base = len(rows_list)//10

    for row_index, row_dict in enumerate(rows_list):
        # Generate the blurb for this row
        blurb_text = generate_blurb_for_row(row_dict, row_index)

        # Create a new dict with the blurb added (don't modify original)
        augmented_row_dict = dict(row_dict)
        augmented_row_dict[new_column_name] = blurb_text

        augmented_rows_list.append(augmented_row_dict)

        # Progress indicator for large files
        if (row_index + 1) % short_status_base == 0:
            print(f"  Processed {row_index + 1} rows...")

    # Write the augmented data to the output file
    print(f"Writing output file: {output_file_path}")
    write_augmented_data_to_csv(output_file_path, augmented_rows_list, output_fieldnames)

    rows_processed_count = len(augmented_rows_list)
    print(f"Successfully processed {rows_processed_count} rows")

    return rows_processed_count


def main() -> None:
    """
    Entry point for the CSV summary blurb augmentor script.

    Intent:
    -------
    This function handles:
    1. Command-line argument parsing
    2. Input validation
    3. Calling the main processing function
    4. Error handling and user feedback

    Usage:
    ------
    python csv_summary_blurb_augmentor.py <input_csv_path> [output_csv_path]

    If output_csv_path is not provided, it will be auto-generated by
    adding "_augmented" to the input filename.

    Examples:
    ---------
    python csv_summary_blurb_augmentor.py dataset.csv
    python csv_summary_blurb_augmentor.py data/input.csv data/output.csv
    """
    try:
        # Parse command-line arguments
        if len(sys.argv) < 2:
            print("Usage: python csv_summary_blurb_augmentor.py <input_csv_path> [output_csv_path]")
            print("")
            print("Arguments:")
            print("  input_csv_path   - Path to the input CSV file (required)")
            print("  output_csv_path  - Path for the output CSV file (optional)")
            print("")
            print("If output_csv_path is not provided, the output will be named")
            print("<input_filename>_augmented.csv in the same directory.")
            sys.exit(1)

        input_csv_path = sys.argv[1]

        # Determine output path (from argument or auto-generated)
        if len(sys.argv) >= 3:
            output_csv_path = sys.argv[2]
        else:
            output_csv_path = generate_output_file_path(input_csv_path)

        # Validate input file exists
        if not os.path.exists(input_csv_path):
            print(f"ERROR: Input file not found: {input_csv_path}")
            sys.exit(1)

        # Check that we're not about to overwrite the input file
        input_absolute_path = os.path.abspath(input_csv_path)
        output_absolute_path = os.path.abspath(output_csv_path)

        if input_absolute_path == output_absolute_path:
            print("ERROR: Output path is the same as input path.")
            print("This script does not modify files in place.")
            print("Please specify a different output path.")
            sys.exit(1)

        # Run the main processing
        print("=" * 60)
        print("CSV Summary Blurb Augmentor")
        print("=" * 60)

        rows_processed = process_csv_and_add_blurbs(input_csv_path, output_csv_path)

        print("=" * 60)
        print(f"Complete! Processed {rows_processed} rows.")
        print(f"Output written to: {output_csv_path}")
        print("=" * 60)

    except FileNotFoundError as file_not_found_error:
        print(f"ERROR: File not found - {file_not_found_error}")
        traceback.print_exc()
        sys.exit(1)

    except PermissionError as permission_error:
        print(f"ERROR: Permission denied - {permission_error}")
        traceback.print_exc()
        sys.exit(1)

    except KeyError as key_error:
        print(f"ERROR: Missing expected column in CSV - {key_error}")
        print("Please ensure the input CSV has all required columns.")
        traceback.print_exc()
        sys.exit(1)

    except ValueError as value_error:
        print(f"ERROR: Invalid data value - {value_error}")
        print("Please ensure all numeric fields contain valid numbers.")
        traceback.print_exc()
        sys.exit(1)

    except Exception as unexpected_error:
        print(f"ERROR: An unexpected error occurred - {unexpected_error}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
