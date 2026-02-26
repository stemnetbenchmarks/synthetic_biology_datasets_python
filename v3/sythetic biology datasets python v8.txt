#### sythetic_biology_datasets_python
# data generation tools
## Make and use: synthetic_biology_dataset.csv
##### GGAshbrook 2025.04

# Contents:
- Introduction/Abstract
- Test Questions Organized by Category
- Dataset Design



# Introduction/Abstract

### Core parts:

A. A dataset with a parameter for dataset size: With a transparent validation of each data field. 'synthetic_biology_dataset.csv'
- the original raw data fields are validated with scores
- the answers are validated with scores
- the validated answers (not separately calculated) are pulled into the QA document.

B. Q&A: The questions and correct-answers as a document, based on the particular dataset.

Plus:
1. make a blank score_sheet.csv:
```
date, question, model, data_shape, time, score, notes
```
2. Because having a fixed-answer exam has problems, the patterns and trends are to be set in a 'configuration' section to allow novel tests.
3. Documentation included in zip
4. Additional Analysis, Validation, Checking Notebook in zip

### Testing:

There are several types of tests that can be done in terms of AI performance, the focus here is on the interplay or conversion between structured-tabular data (such as a database) on the one hand and unstructured natural-language (such as a user's inputs and questions) on the other hand.

It should be possible to make user-friendly, low maintenance, verifiable tools to:
A. generate a dataset that is testable, verifiable, and verified, and that is non-private 'testing data' safe to use on various products, and that can be made to a user-specified size of dataset.
B. generate and verify answers to pre-set questions, based on the verified data
C. Evaluate and compare the performance of models in a standardized and well-defined way, comparing apples to apples.  

Here the dataset will be specifically designed for the specific questions to be asked about it (see details below).


### Example Types of Questions (Summary):
1. What ~simple tabular-operations can the AI reliably do when specifically named?

2. What ~simple tabular-operations can the AI correctly interpret from a non-technical question?  

3. fuzzy questions about the material in given slice of data

4. Non-simple field-engineered and grouped questions given only raw data?
examples maybe standard time engineered field questions and relative time (temp-field) questions: in the last X months...y?

5. To the extent that we know, is the 'AI' (a model or a composite of parts) using a single stateless context window, or is it 'function-calling' and 'tool-using'?
Note: Context windows are large, but 2.5

6. Statistical Questions that go beyond ~descriptive statistics

7. Asking for (results of) basic pandas functions:
- shape()
- unique()
- value_counts()
- describe()
- .sum(), .mean/median/mode()

Etc.


### Context:

Many applications of AI/ML can be tested for performance. Sometimes the term 'benchmark' is used, though the meaning of that term varies. Though generative and vector producing deep learning models have become more performative since late 2022 (when the public release of chat-GPT by open-AI popularized 'foundation-models' sometimes called 'LLM's), there has been a significant absence of organized infrastructure investment in the area of meaningful and usable AI model and service testing and training-data tools; namely, there has been a silence (for whatever reasons) from government standards bodies such as NIST, international standards bodies such as ISO, or trade-group standards organization (which may themselves be missing for AI), academic tools have been missing, as have been private industry based tools such as in the past would have been produced by Bell Labs, Xero Park, IBM, or ambiguous publishers such as RAND. (This topic could be gone into in much more detail, but that is not meant to be the focus here. See: let's test models' for some more discussion.) https://medium.com/@GeoffreyGordonAshbrook/lets-test-models-and-let-s-do-tasks-84777f80eb99 







# Test Questions Organized by Category
## Basic Statistics Questions
1. "What is the most common animal type?"
ANSWER: All animal types (cat, dog, bird, fish, turtle) appear with equal frequency.

2. "What is the average weight of all animals?"
ANSWER: Approximately 23 kg (average of all types' means: turtles 50kg, birds 30kg, fish 20kg, dogs 10kg, cats 5kg)

3. "What is the average weight of cats?"
ANSWER: Approximately 5 kg

4. "Which animal type is heaviest on average?"
ANSWER: Turtles (average ~50kg)

5. "How many birds are in the dataset?"
ANSWER: Exactly 20% of the dataset (balanced distribution)

6. "How many friends do cats have on average?"
ANSWER: Approximately 5 friends (base value for cats)

7. "Which animal type has the most friends?"
ANSWER: Dogs (approximately 10 friends on average.


## Categorical Analysis Questions
8. "What is the most common color?"
ANSWER: Blue (50% of all animals)

9. "Is there a relationship between color and animal type?"
ANSWER: Yes, strong correlation: fish are predominantly blue (80%), cats red (80%), dogs green (80%), birds gray (80%), turtles mixed (80%)

10. "What percentage of red animals can swim?"
ANSWER: Approximately 90% (since red animals are primarily cats, which swim at 90% rate)

11. "What is the average weight of blue animals compared to green ones?"
ANSWER: Blue animals heavier (mostly fish at 20kg and some turtles at 50kg) compared to green animals (mostly dogs at 10kg)


## Time Series Analysis Questions
12. "When do most animals tend to be born?"
ANSWER: In winter months (December, January, February)

13. "Is there a seasonal pattern to births?"
ANSWER: Yes, animals are only born in winter months

14. "How many animals were born in the first quarter of the year?"
ANSWER: Approximately 2/3 of all animals (those born in January and February)


## Correlation and Relationship Questions
15. "Is there a correlation between weight and height?"
ANSWER: Yes, strong negative correlation - smaller animals tend to be taller

16. "What factors predict food consumption?"
ANSWER: Weight is inversely related to food consumption - smaller animals eat more

17. "Is there a relationship between age and number of friends?"
ANSWER: Yes, cyclical relationship - number of friends follows a cosine wave pattern with age, cycling every 7-8 years

18. "Do animals with more friends tend to be more popular?"
ANSWER: No direct relationship programmed between these variables


## Boolean Pattern Questions
19. "What percentage of animals can fly?"
ANSWER: Approximately 50% (randomly distributed)

20. "Which animal types are most likely to swim?"
ANSWER: Cats and birds (approximately 90% can swim, compared to 10% for other types)

21. "How does an animal's age affect its ability to run?"
ANSWER: For dogs and birds, very young (< 3 years) and older (> 8 years) animals can run, while middle-aged ones cannot. For other animals, no age-related pattern.

22. "What percentage of animals watch YouTube?"
ANSWER: About 20% overall, with fish and turtles much more likely (90%) than other animals (10%)

23. "Which animals can both swim and fly?"
ANSWER: About 45% of cats and birds can both swim and fly (90% swimming rate × 50% flying rate)


## Complex Pattern Detection Questions
24. "How does height affect popularity score?"
ANSWER: U-shaped relationship - very tall and very short animals are more popular than medium-height animals

25. "What makes an animal popular?"
ANSWER: Extreme heights (either very tall or very short) contribute to popularity

## Text Analysis Questions (Social Media)
26. "Which animal type mentions films most often in their social media?"
ANSWER: [Will depend on our implementation]

27. "Which animal type has the most positive sentiment in their social media posts?"
ANSWER: [Will depend on our implementation]

28. "Which animal type has the most negative sentiment in their social media posts?"
ANSWER: [Will depend on our implementation]

29. "What topics do fish typically discuss in their social media?"
ANSWER: [Will depend on our implementation]



# Dataset Design: ~14 Core Fields 

1. animal_type (string): "cat", "dog", "bird", "fish", "turtle" (balanced value counts)
### Testing purpose: Categorical field for grouping and filtering
### Questions it enables: 
"What is the most common animal type?", 
"How many birds are in the dataset?"
"Which is more common, birds or turtles?"
"What is the most common animal type?"
ANSWER: All animal types (cat, dog, bird, fish, turtle) appear with equal frequency, as we've implemented balanced distribution.

2. weight_kg (float): Weight with realistic distributions (what pattern/question is this for?)
### Testing purpose: Basic descriptive statistics and group comparisons of said statistics: see pandas describe()
### Questions it enables: 
"What is the average weight of each animal type?",
"Which animal type is heaviest?"

#### Sample Answers:
QUESTION: "What is the average weight of all animals?"
ANSWER: Approximately 23 kg (average of all animal types' means, with turtles heaviest at 50kg, birds at 30kg, fish at 20kg, dogs at 10kg, cats at 5kg)

QUESTION: "What is the average weight of cats?"
ANSWER: Approximately 5 kg

QUESTION: "Which animal type is heaviest on average?"
ANSWER: Turtles (average ~50kg)

2. height_cm (float): Strong negative correlation with weight (must not be able to guess/cheat) (what pattern/question is this for?)
### Testing purpose: Correlation detection
### Questions it enables: 
"Is there a relationship between weight and height?", 
"How does height change as weight increases?"
"Is there a correlation between weight and height?"
ANSWER: Yes, there is a strong negative correlation - smaller animals tend to be taller
"What affects the popularity score?"
ANSWER: There is a U-shaped relationship - very tall and very short animals are more popular than medium-height animals

4. number_of_friends (int):
### Testing purpose: Integer field with cyclical/periodic pattern detection
Default pattern: Cosine relationship with age that creates a wave pattern:
Copynumber_of_friends = base_value + amplitude * cos(frequency * age)
Where:

Base values vary by animal type (dogs: 10, cats: 5, birds: 8, fish: 3, turtles: 2)
Animals gain and lose friends in cycles approximately every 7-8 years
Overall range between 1-15 friends

### Questions it enables:

"How many friends do cats have on average?"
"Is there a relationship between age and number of friends?"
"Which animal type has the most friends?"
"Do animals with more friends tend to be more popular?"
"How does the number of friends change as animals get older?"

This creates a non-intuitive pattern that:
Can't be guessed without analyzing the data
Tests detection of cyclical patterns
Provides a genuine integer field for operations
Creates interesting possibilities for multivariate analysis

#### Sample Answers:
QUESTION: "How many animals were born in the first quarter of the year?"
ANSWER: Approximately 2/3 of all animals (those born in January and February)

QUESTION: "Is there a seasonal pattern to births?"
ANSWER: Yes, animals are only born in winter months (December, January, February)

QUESTION: "What's the median birth timestamp and what date does it correspond to?"
ANSWER: [Value will depend on data generation] which corresponds to a date in January or February

5. birth_date (date): Birth date with strong seasonal patterns (only in winter months) (must not be able to guess/cheat)
### Testing purpose: Date-time data handling and pattern detection in time-features not explicitly in the date-time object
### Questions it enables: 
"When do most animals tend to be born?", 
"Is there a seasonal pattern to births?"


6. birth_unix (int): Same date in Unix epoch format
### Testing purpose: Time representation conversion
Better questions needed
### Questions it enables: 
"Can the model convert between date formats?", 
"Can analysis be performed on numeric timestamp data?"

7. color (string): "red", "green", "blue", "gray", "mixed" (imbalanced value counts) (must not be able to guess/cheat)
"What is the most common color?"
ANSWER: Blue (50% of all animals, due to our specified dominant distribution)

"Is there a relationship between color and animal type?"
ANSWER: Yes, strong correlation: fish are predominantly blue (80%), cats are predominantly red (80%), dogs are predominantly green (80%), birds are predominantly gray (80%), and turtles are predominantly mixed (80%)

### Testing purpose:
Categorical distribution analysis
Testing conditional statistics across groups
Testing pattern recognition between categorical variables
### Default pattern:
Strong but incomplete correlation with animal_type:
- Fish are predominantly blue (80%)
- Cats are predominantly red (80%)
- Dogs are predominantly green (80%)
- Birds are predominantly gray (80%)
- Turtles are predominantly mixed (80%)
- Remaining 20% randomly distributed among other colors
### Questions it enables:
- "What is the most common color?"
- "Is there a relationship between color and animal type?"
- "What percentage of red animals can swim?"
- "What is the average weight of blue animals compared to green ones?"
- "Which color has the highest average popularity score?"
- "Are there any animal types that never appear in certain colors?"
#### Why imbalanced data is useful:
- Tests frequency-based analysis: Model must recognize dominant colors in dataset
- Tests conditional probability: "If an animal is green, what's the probability it's a dog?"
- Enables complex filtering: Combinations like "blue fish that can fly" test multi-condition queries
- Tests model's data exploration: Can it identify the dominant color for each animal type?

### Sample Answer:
QUESTION: "What is the average weight of blue animals compared to green ones?"
ANSWER: Blue animals are heavier on average (predominantly fish at 20kg and some turtles at 50kg) compared to green animals (predominantly dogs at 10kg)

# Boolean Capability Fields (4 total)
8. can_fly (boolean): random, no correlation (must not be able to guess/cheat)
### Testing purpose: Random boolean with no correlations
### Questions it enables: 
"What percentage of individual animals can fly?", 
"What animals are best at flying?"
"What percentage of animals can fly?"
ANSWER: Approximately 50% (randomly distributed)

9. can_swim (boolean): Strong but "imperfect correlation"?? (only cats and birds)  (must not be able to guess/cheat)
### Testing purpose: Counter-intuitive correlation (only cats and birds)
### Questions it enables: "Which animal types are most likely to swim?", "Is there a pattern to swimming ability?"
"Which animal types are most likely to swim?"
ANSWER: Cats and birds (approximately 90% can swim, compared to 10% for other types)

"Which animals can both swim and fly?"
ANSWER: About 45% of cats and birds can both swim and fly (90% swimming rate × 50% flying rate)

10. can_run (boolean): non-linear pattern for dogs and birds, random for others
### Testing purpose: Non-linear age-dependent pattern
### Questions it enables: 
"How does age affect running ability?", 
"Which animals can both fly and run?"
"How does an animal's age affect its ability to run?"
ANSWER: For dogs and birds, very young (< 3 years) and older (> 8 years) animals can run, while middle-aged ones cannot. For other animals, there is no age-related pattern.

11. watches_youtube (boolean): strong linear (must not be able to guess/cheat)
### Testing purpose: strong linear correlation detection (fish and turtles)
### Questions it enables: 
"Which animals watch YouTube?", 
"Is there a relationship between YouTube watching and other traits?"
"What percentage of animals watch YouTube?"
ANSWER: About 20% overall, with fish and turtles much more likely (90%) than other animals (10%)

## Relationship Testing Fields (3 total)
12. daily_food_grams (float): Weak linear correlation with weight
### Testing purpose: Inverse linear relationship detection involving float
### Questions it enables: 
"What factors predict food consumption?", 
"Do heavier animals eat more?"

13. popularity_score (float): Non-linear relationship with height (U-shaped)
### Testing purpose: Non-linear relationship detection
### Questions it enables: 
"What makes an animal popular?", 
"Is there a U-shaped relationship between height and popularity?"

14. social_media (string): varied content with sentiment, deterministically generated.
### Testing purpose: sentiment analysis, asking about text-block content.
"Where animal's (type) social media talks about films?"
"Which animal (type) has the highest or lowest sentiment?"

Note: use assorted phrases that different types of analysis might pick up: film, movies, on stage, on camera, acting, performance, etc.



