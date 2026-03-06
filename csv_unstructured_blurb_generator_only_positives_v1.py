"""
csv_unstructured_blurb_generator.py

Purpose:
--------
This script reads a structured CSV dataset containing animal records,
and produces an augmented copy with an additional column containing
unstructured natural language descriptions ("blurbs") of each record.

Due to negative-concept recognition problems with some models,
this verson of the script only uses positive boolean values,
and present other numerical values (no null value blurbs).

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
- Boolean fields are ONLY included when True (no negative statements)
- Null/empty fields are omitted from blurbs
- Numeric zero IS included (e.g., "has 0 friends")
- Age is calculated dynamically from birth_unix timestamp
- Original data is NOT modified; a new file is created
- No randomness; output is fully reproducible

Author: Synthetic Biology Datasets Project
"""

import csv
import traceback
import os
import sys
from datetime import datetime, timezone

def generate_blurb_for_row(row_data: dict, row_index: int) -> str:
    """
    Generate an unstructured text blurb for a single row of data.

    Intent:
    -------
    This function is the main orchestrator for blurb generation. It:
    1. Extracts and validates all necessary fields from the row
    2. Checks each field for presence using is_field_value_present_and_positive
    3. Rounds float values to 3 decimal places (when present)
    4. Calculates age dynamically from birth_unix timestamp
    5. Converts boolean strings to Python bools
    6. Selects the appropriate template based on row index
    7. Calls the appropriate template function

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
    KeyError
        If expected column names are not present in row_data.
    """
    # Extract string values that are used directly
    animal_type = row_data["animal_type"]
    color = row_data["color"]
    birth_date = row_data["birth_date"]

    # Check and convert numeric fields - None if not present/positive
    weight_kg_raw = row_data["weight_kg"]
    if is_field_value_present_and_positive(weight_kg_raw):
        weight_kg_rounded = round_float_to_three_decimals(float(weight_kg_raw))
    else:
        weight_kg_rounded = None

    height_cm_raw = row_data["height_cm"]
    if is_field_value_present_and_positive(height_cm_raw):
        height_cm_rounded = round_float_to_three_decimals(float(height_cm_raw))
    else:
        height_cm_rounded = None

    daily_food_grams_raw = row_data["daily_food_grams"]
    if is_field_value_present_and_positive(daily_food_grams_raw):
        daily_food_grams_rounded = round_float_to_three_decimals(float(daily_food_grams_raw))
    else:
        daily_food_grams_rounded = None

    popularity_score_raw = row_data["popularity_score"]
    if is_field_value_present_and_positive(popularity_score_raw):
        popularity_score_rounded = round_float_to_three_decimals(float(popularity_score_raw))
    else:
        popularity_score_rounded = None

    number_of_friends_raw = row_data["number_of_friends"]
    if is_field_value_present_and_positive(number_of_friends_raw):
        number_of_friends = int(number_of_friends_raw)
    else:
        number_of_friends = None

    # Calculate age from birth_unix timestamp
    birth_unix_raw = row_data["birth_unix"]
    if is_field_value_present_and_positive(birth_unix_raw):
        birth_unix_timestamp = float(birth_unix_raw)
        age_years = calculate_age_in_whole_years_from_unix_timestamp(birth_unix_timestamp)
    else:
        age_years = None

    # Convert boolean strings to Python bools
    can_fly = convert_boolean_string_to_python_bool(row_data["can_fly"])
    can_swim = convert_boolean_string_to_python_bool(row_data["can_swim"])
    can_run = convert_boolean_string_to_python_bool(row_data["can_run"])
    watches_youtube = convert_boolean_string_to_python_bool(row_data["watches_youtube"])

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
            can_fly=can_fly,
            can_swim=can_swim,
            can_run=can_run,
            watches_youtube=watches_youtube,
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
            can_fly=can_fly,
            can_swim=can_swim,
            can_run=can_run,
            watches_youtube=watches_youtube,
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
            can_fly=can_fly,
            can_swim=can_swim,
            can_run=can_run,
            watches_youtube=watches_youtube,
            daily_food_grams_rounded=daily_food_grams_rounded,
            popularity_score_rounded=popularity_score_rounded
        )

    return generated_blurb


def generate_blurb_using_template_zero(
    animal_type: str,
    weight_kg_rounded: float | None,
    height_cm_rounded: float | None,
    age_years: int | None,
    number_of_friends: int | None,
    birth_date: str,
    color: str,
    can_fly: bool,
    can_swim: bool,
    can_run: bool,
    watches_youtube: bool,
    daily_food_grams_rounded: float | None,
    popularity_score_rounded: float | None
) -> str:
    """
    Generate an unstructured text blurb using template format 0.

    Intent:
    -------
    This is the first of three template formats. It presents information
    in a specific order starting with animal type, then food, age, friends,
    physical characteristics, and abilities.

    Boolean ability fields are only included when True.
    Numeric fields are only included when not None.

    Template 0 ability order: fly, swim at end; run in middle (after birth/color).

    Parameters:
    -----------
    animal_type : str
        The type of animal (e.g., "cat", "dog", "bird").
    weight_kg_rounded : float | None
        The animal's weight in kg, or None if not present.
    height_cm_rounded : float | None
        The animal's height in cm, or None if not present.
    age_years : int | None
        The animal's age in years, or None if not present.
    number_of_friends : int | None
        How many friends the animal has, or None if not present.
    birth_date : str
        The animal's birth date as a string.
    color : str
        The animal's color.
    can_fly : bool
        True if the animal can fly.
    can_swim : bool
        True if the animal can swim.
    can_run : bool
        True if the animal can run.
    watches_youtube : bool
        True if the animal watches youtube.
    daily_food_grams_rounded : float | None
        Daily food in grams, or None if not present.
    popularity_score_rounded : float | None
        Popularity score, or None if not present.

    Returns:
    --------
    str
        A natural language blurb describing the animal.
    """
    sentence_parts = []

    # Core identifying information
    sentence_parts.append(f"This is about a {animal_type}.")

    # Food information (if present)
    if daily_food_grams_rounded is not None:
        sentence_parts.append(f"It gets a daily allowance of {daily_food_grams_rounded} grams of food.")

    # Age (if present)
    if age_years is not None:
        sentence_parts.append(f"It is {age_years} years old.")

    # Friends (if present) - note: 0 is valid
    if number_of_friends is not None:
        sentence_parts.append(f"It has {number_of_friends} friends.")

    # Physical characteristics (if present)
    if weight_kg_rounded is not None and height_cm_rounded is not None:
        sentence_parts.append(f"This {animal_type} weighs {weight_kg_rounded} kg, and is {height_cm_rounded} cm tall.")
    elif weight_kg_rounded is not None:
        sentence_parts.append(f"This {animal_type} weighs {weight_kg_rounded} kg.")
    elif height_cm_rounded is not None:
        sentence_parts.append(f"This {animal_type} is {height_cm_rounded} cm tall.")

    # Popularity (if present)
    if popularity_score_rounded is not None:
        sentence_parts.append(f"It has a popularity_score of {popularity_score_rounded}.")

    # Birth and color
    sentence_parts.append(f"It was born on {birth_date}, and is {color}.")

    # Run in middle position for template 0
    run_sentence = build_ability_sentence_if_positive("run", can_run)
    if run_sentence:
        sentence_parts.append(run_sentence)

    # Fly and swim at end for template 0
    fly_sentence = build_ability_sentence_if_positive("fly", can_fly)
    if fly_sentence:
        sentence_parts.append(fly_sentence)

    swim_sentence = build_ability_sentence_if_positive("swim", can_swim)
    if swim_sentence:
        sentence_parts.append(swim_sentence)

    # Youtube at end
    youtube_sentence = build_youtube_sentence_if_positive(watches_youtube)
    if youtube_sentence:
        sentence_parts.append(youtube_sentence)

    blurb_text = " ".join(sentence_parts)
    return blurb_text


def generate_blurb_using_template_one(
    animal_type: str,
    weight_kg_rounded: float | None,
    height_cm_rounded: float | None,
    age_years: int | None,
    number_of_friends: int | None,
    birth_date: str,
    color: str,
    can_fly: bool,
    can_swim: bool,
    can_run: bool,
    watches_youtube: bool,
    daily_food_grams_rounded: float | None,
    popularity_score_rounded: float | None
) -> str:
    """
    Generate an unstructured text blurb using template format 1.

    Intent:
    -------
    This is the second of three template formats. It presents information
    in a different order, starting with physical characteristics.
    Uses exclamation marks for ability statements.

    Boolean ability fields are only included when True.
    Numeric fields are only included when not None.

    Template 1 ability order: fly, swim in middle with exclamation; run at end.

    Parameters:
    -----------
    (Same as generate_blurb_using_template_zero)

    Returns:
    --------
    str
        A natural language blurb describing the animal.
    """
    sentence_parts = []

    # Physical characteristics first
    if weight_kg_rounded is not None and height_cm_rounded is not None:
        sentence_parts.append(f"This {animal_type} weighs {weight_kg_rounded} kg, and is {height_cm_rounded} cm tall.")
    elif weight_kg_rounded is not None:
        sentence_parts.append(f"This {animal_type} weighs {weight_kg_rounded} kg.")
    elif height_cm_rounded is not None:
        sentence_parts.append(f"This {animal_type} is {height_cm_rounded} cm tall.")
    else:
        sentence_parts.append(f"This is a {animal_type}.")

    # Color and birth
    sentence_parts.append(f"It is {color} and was born on {birth_date}.")

    # Food (if present)
    if daily_food_grams_rounded is not None:
        sentence_parts.append(f"It gets {daily_food_grams_rounded} grams of food each day.")

    # Age (if present)
    if age_years is not None:
        sentence_parts.append(f"It is {age_years} years old.")

    # Fly and swim in middle with exclamation for template 1
    fly_sentence = build_ability_sentence_if_positive("fly", can_fly, "!")
    if fly_sentence:
        sentence_parts.append(fly_sentence)

    swim_sentence = build_ability_sentence_if_positive("swim", can_swim, "!")
    if swim_sentence:
        sentence_parts.append(swim_sentence)

    # Friends and popularity (if present)
    if number_of_friends is not None and popularity_score_rounded is not None:
        sentence_parts.append(f"This {animal_type} has {number_of_friends} friends and has a popularity_score of {popularity_score_rounded}.")
    elif number_of_friends is not None:
        sentence_parts.append(f"This {animal_type} has {number_of_friends} friends.")
    elif popularity_score_rounded is not None:
        sentence_parts.append(f"This {animal_type} has a popularity_score of {popularity_score_rounded}.")

    # Youtube
    youtube_sentence = build_youtube_sentence_if_positive(watches_youtube)
    if youtube_sentence:
        sentence_parts.append(youtube_sentence)

    # Run at end for template 1
    run_sentence = build_ability_sentence_if_positive("run", can_run, "!")
    if run_sentence:
        sentence_parts.append(run_sentence)

    blurb_text = " ".join(sentence_parts)
    return blurb_text


def generate_blurb_using_template_two(
    animal_type: str,
    weight_kg_rounded: float | None,
    height_cm_rounded: float | None,
    age_years: int | None,
    number_of_friends: int | None,
    birth_date: str,
    color: str,
    can_fly: bool,
    can_swim: bool,
    can_run: bool,
    watches_youtube: bool,
    daily_food_grams_rounded: float | None,
    popularity_score_rounded: float | None
) -> str:
    """
    Generate an unstructured text blurb using template format 2.

    Intent:
    -------
    This is the third of three template formats. It presents information
    in yet another order, with exclamation at the start.

    Boolean ability fields are only included when True.
    Numeric fields are only included when not None.

    Template 2 ability order: run early (after intro); swim, fly later.

    Parameters:
    -----------
    (Same as generate_blurb_using_template_zero)

    Returns:
    --------
    str
        A natural language blurb describing the animal.
    """
    sentence_parts = []

    # Intro with exclamation
    sentence_parts.append(f"This is a {animal_type}!")

    # Run early for template 2
    run_sentence = build_ability_sentence_if_positive("run", can_run)
    if run_sentence:
        sentence_parts.append(run_sentence)

    # Height (if present)
    if height_cm_rounded is not None:
        sentence_parts.append(f"It is {height_cm_rounded} cm tall.")

    # Food and weight (if present)
    if daily_food_grams_rounded is not None and weight_kg_rounded is not None:
        sentence_parts.append(f"It gets {daily_food_grams_rounded} grams of food each day and weighs {weight_kg_rounded} kg!")
    elif daily_food_grams_rounded is not None:
        sentence_parts.append(f"It gets {daily_food_grams_rounded} grams of food each day.")
    elif weight_kg_rounded is not None:
        sentence_parts.append(f"It weighs {weight_kg_rounded} kg!")

    # Color and birth
    sentence_parts.append(f"It is {color}, and it was born on {birth_date}.")

    # Age (if present)
    if age_years is not None:
        sentence_parts.append(f"It is {age_years} years old!")

    # Swim and fly later for template 2
    swim_sentence = build_ability_sentence_if_positive("swim", can_swim)
    if swim_sentence:
        sentence_parts.append(swim_sentence)

    fly_sentence = build_ability_sentence_if_positive("fly", can_fly)
    if fly_sentence:
        sentence_parts.append(fly_sentence)

    # Youtube
    youtube_sentence = build_youtube_sentence_if_positive(watches_youtube)
    if youtube_sentence:
        sentence_parts.append(youtube_sentence)

    # Friends and popularity at end (if present)
    if number_of_friends is not None and popularity_score_rounded is not None:
        sentence_parts.append(f"It has {number_of_friends} friends and has a popularity_score of {popularity_score_rounded}.")
    elif number_of_friends is not None:
        sentence_parts.append(f"It has {number_of_friends} friends.")
    elif popularity_score_rounded is not None:
        sentence_parts.append(f"It has a popularity_score of {popularity_score_rounded}.")

    blurb_text = " ".join(sentence_parts)
    return blurb_text


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


def calculate_age_in_whole_years_from_unix_timestamp(birth_unix_timestamp: float) -> int:
    """
    Calculate age in whole years from a Unix timestamp to current UTC time.

    Intent:
    -------
    Rather than relying on a static age_years field in the CSV (which
    becomes stale over time), we calculate age dynamically from the
    birth timestamp. This ensures blurbs always reflect current age.

    The calculation is birthday-aware: if the current date is before
    this year's birthday, we subtract one year from the simple
    year difference.

    Parameters:
    -----------
    birth_unix_timestamp : float
        Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)
        representing the birth date/time.

    Returns:
    --------
    int
        The age in whole (completed) years.

    Example:
    --------
    If birth_unix_timestamp corresponds to 2020-06-15, and today is
    2024-03-01, the function returns 3 (not 4, because the 2024
    birthday hasn't occurred yet).
    """
    try:
        # Convert Unix timestamp to datetime object in UTC
        birth_datetime = datetime.fromtimestamp(birth_unix_timestamp, tz=timezone.utc)

        # Get current UTC datetime
        current_datetime = datetime.now(tz=timezone.utc)

        # Calculate the difference in years (simple subtraction)
        age_in_years = current_datetime.year - birth_datetime.year

        # Adjust if birthday hasn't occurred yet this year
        # Compare (month, day) tuples to determine if birthday has passed
        birth_month_day = (birth_datetime.month, birth_datetime.day)
        current_month_day = (current_datetime.month, current_datetime.day)

        if current_month_day < birth_month_day:
            age_in_years -= 1

        return age_in_years

    except (ValueError, OSError) as timestamp_error:
        # Handle invalid timestamps (e.g., negative values, overflow)
        print(f"WARNING: Invalid birth timestamp {birth_unix_timestamp}: {timestamp_error}")
        traceback.print_exc()
        # Return 0 as a safe fallback rather than crashing
        return 0


def is_field_value_present_and_positive(field_value) -> bool:
    """
    Check if a field value is present and positive (should be included in blurb).

    Intent:
    -------
    We only want to include information in blurbs when the field has
    meaningful content. This function implements the rule:
    - None, empty string, boolean False → NOT positive (exclude)
    - Numeric zero → IS positive (include, e.g., "has 0 friends")
    - Non-empty strings, boolean True, any number → positive (include)

    Parameters:
    -----------
    field_value : any
        The value to check. Can be None, str, bool, int, float, etc.

    Returns:
    --------
    bool
        True if the value should be included in the blurb, False otherwise.

    Examples:
    ---------
    >>> is_field_value_present_and_positive(None)
    False
    >>> is_field_value_present_and_positive("")
    False
    >>> is_field_value_present_and_positive(False)
    False
    >>> is_field_value_present_and_positive(0)
    True
    >>> is_field_value_present_and_positive("blue")
    True
    >>> is_field_value_present_and_positive(True)
    True
    """
    # None is not positive
    if field_value is None:
        return False

    # Boolean check must come before numeric check
    # (because bool is a subclass of int in Python)
    if isinstance(field_value, bool):
        return field_value

    # Empty string is not positive
    if isinstance(field_value, str) and field_value == "":
        return False

    # Everything else (including numeric zero) is positive
    return True


def build_ability_sentence_if_positive(ability_name: str, can_do_ability: bool, punctuation: str = ".") -> str:
    """
    Build an ability sentence if the ability is True, otherwise return empty string.

    Intent:
    -------
    For boolean ability fields (can_fly, can_swim, can_run), we only want
    to generate a sentence when the ability is True. When False, we omit
    the sentence entirely rather than saying "cannot fly" (per requirements).

    Parameters:
    -----------
    ability_name : str
        The name of the ability (e.g., "fly", "swim", "run").
    can_do_ability : bool
        True if the animal has this ability, False otherwise.
    punctuation : str, optional
        The punctuation to end the sentence with. Defaults to "."
        Can be "!" for emphasis in certain templates.

    Returns:
    --------
    str
        A sentence like "It can fly." if True, empty string "" if False.

    Examples:
    ---------
    >>> build_ability_sentence_if_positive("fly", True)
    "It can fly."
    >>> build_ability_sentence_if_positive("swim", True, "!")
    "It can swim!"
    >>> build_ability_sentence_if_positive("run", False)
    ""
    """
    if can_do_ability:
        return f"It can {ability_name}{punctuation}"
    else:
        return ""


def build_youtube_sentence_if_positive(watches_youtube: bool) -> str:
    """
    Build a youtube watching sentence if True, otherwise return empty string.

    Intent:
    -------
    The watches_youtube field gets special phrasing ("likes to watch youtube")
    rather than the simpler "can" phrasing used for abilities. When False,
    we omit the sentence entirely rather than saying "does not like to watch".

    Parameters:
    -----------
    watches_youtube : bool
        True if the animal watches youtube, False otherwise.

    Returns:
    --------
    str
        "It likes to watch youtube." if True, empty string "" if False.

    Examples:
    ---------
    >>> build_youtube_sentence_if_positive(True)
    "It likes to watch youtube."
    >>> build_youtube_sentence_if_positive(False)
    ""
    """
    if watches_youtube:
        return "It likes to watch youtube."
    else:
        return ""


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

    # To prevent status prints from being too abundant
    # as data files are typically not of picayune quantity,
    # getting list-length//10, and using:
    #     if (row_index + 1) % short_status_base == 0:
    # should give usually 10 status prints, also fitting
    # within 80x24 terminal size.
    short_status_base = len(rows_list) // 10

    # Avoid division by zero for very small files
    if short_status_base == 0:
        short_status_base = 1

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
    python csv_unstructured_blurb_generator.py <input_csv_path> [output_csv_path]

    If output_csv_path is not provided, it will be auto-generated by
    adding "_augmented" to the input filename.

    Examples:
    ---------
    python csv_unstructured_blurb_generator.py dataset.csv
    python csv_unstructured_blurb_generator.py data/input.csv data/output.csv
    """
    try:
        # Parse command-line arguments
        if len(sys.argv) < 2:
            print("Usage: python csv_unstructured_blurb_generator_etc... .py <input_csv_path> [output_csv_path]")
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
