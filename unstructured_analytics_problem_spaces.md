Testing Tradeoffs in Hybrid Data Analytics

Mapping Unstructured-Analytics Problem-Spaces: Comparing Methods, Scopes, and Evaluations for Quantitative Vector Comparison of "Documents" or Unstructured Data-Objects in a context of structured, unstructured, vector and generative tools.
2026.02.26-... G.G.Ashbrook

Performance and Tradeoffs in Hybrid Data Analytics
& Evaluation-Tests with a Focus on Concept ("Embedding") vectors.

~Hybrid Structured Unstructured Analytics with Vectors & Generation

## MVP-1 Code-Utilities Outline:

1. Report for hopefully both non-technical and technical audiences.
- Including broader scope beyond MVP-1
- Outlining Concepts
- Explaining The Evaluations
- Explaining real world applications

[DONE] 2. set up a minimal open-source local vector Database for testing
(possibly self-made in python with numpy/pandas)
- home-made using numpy, sentence-transformers library, common vanilla bert? embedder

[DONE] 3. Single query Tests: Notebook to compare the results of the four types of approaches, using +1 +10 standard questions.
5 tests with granular reporting

[DONE] 4. Comparative Query Tests: 
four tests

5. Set up a document-structuring-extraction test-set.
1. take table
2. make blurb
3. extract from blurb
- 1 or N blurb batch
4. compare to table
- 1 or N blurb batch
5. score: confusion matrix, which fields are what % of errors

A. iterative crawl
- n fields
- n rows

B. mulit-batch
- N rows per batch
- N batches

+
HTLM dashboard reports...

# Intro:
One reason why it is good to have an abstract 'base-came' or 'square one' or 'home base' framework, is that it is easy to become unmoored and lost after a few design decisions, no longer knowing where you are going, where you are, or how to evaluate what you are planning.

Having an abstract problem-space map can help to stay oriented in real-life confusion situations. 

As a warning, there will likely be future cases where a problem-space is significantly undefined, where the question and the quality of the answer are both highly subjective and highly noisy (e.g. See Khaneman/Tversy for an extensive systematic problem-space of known perception/articulation/decision/communication problems.). In some of these cases a combination of back-of-the-knapkin and vague-feedback will be fine, but in other (perhaps most) situations the noise will overwhelm the signal, and to propose an answer to Steve Gibson's question: 
Q: What happens when you remove the science from computer science?
A: The result is a fantasy that is divorced from reality, moreover a fantasy that doubles-down on more-fantasy when empirical breakdown and collapse results, preventing the identification and solution of problems, which prevents corrections, maintainability, sustainability, and survival.

Glossary Note: The term 'hybrid' here is meant to refer to any combination of any number of methods, not meant to refer to a specific combination: hybrid vs. single-method.


## Overall Plan and Scope:
### Overt Research Goals:
1. map out problem space
2. identify mechanical and high level taxonomy/phylogeny of query types
3. run basic tests on basic query types
4. super-MVP full-stack (cli but api-compatible) system for user to load and query data on basic high level query types

Between these four areas, how much can we quantify, and provide modular testing platforms for per-project evaluation, for the tradeoffs and use-case-appropriateness or these four types/categories of query-tech-stack:

#### A. Tabular-Structured Queries/Analytics, 
#### B. Vector Queries/Analytics, 
#### C. Structuring/Extraction Queries/Analytics, 
#### D. Hybrid Queries/Analytics (Any structured data + any vector data, often including analyzing and comparing the results of vector searches, such as chunk quantities and chunk distances, but in some cases chunk-metadata extracted from matched chunks may be the target result rather than the quantity or distance of the chunks in question; or the target may be pattern analysis on N (e.g. 1000) vector-queries). 

Roughly:
- Structured tabular has worked for many years, but the overhead of setting up the data and databases and maintaining those is not always practical and feasible. The quagmire of database, data-wearhouse, and data-lake, and data-silo management is a vast area with ever more per-case tools and possible solutions (each with tradeoffs, and often significant costs over time (the cost of simply running one (even un-used) large PostgreSQL database in the cloud is larger than the budget of most startups). 
- Vector search can be incredibly effective for single-document retrieval, but this does not help with many 'stateful' or 'analytic' type needs. A persistent benefit of simple vector search is the incredible open-ended-ness of uses: it is not rigidly made for one known narrow task only, but can be utilized for novel and ad-hoc queries.
- Data-structuring/extraction can be a solution IF you know ahead of time what you want to extract, or if you have the resources to run a new extraction (as may be worthwhile if the question is not a one-off question). 

### Application Goals:
An application-goal and scope of this study/paper is to emphasize the planning steps of: 
1. articulating what specific questions and ranges of questions are in scope
2. evaluate what tools (or combinations of tools) do and do not match the questions in scope
3. clearly define an MVP-1 that starts with the lowest hanging fruit inside that scope: queries on unstructured data that can be systematically evaluated for correctness (that have a known correct answer to which the system output can be reliably compared)
4. perform rigorous testing of that MVP-1
5. Review the results before iterating
6. use all those data to carefully formulate an MVP-2


As I expect vector-analytics to be an ever-growing area with ever more diverse tools and methods, this study is meant to help form a home-base foundation starting with the most conservatively reliable analytics, with a back-burner idea to eventually supporting research into ever more of that tree.

How-many-chunks, and how-near-distance (either a normalized metric or using a threshold for 'close enough' for a known case, or some other configuration) are likely the main initial tools for quantitative vector analytics.

Multi-result combined analysis also likely has conservative starting places, such as comparing two counts (which is bigger). This may be comparing two points (or ranges) in time for a general time-trend, or two different items (more cats than dogs vs. more dogs than cat). 

Fancier and more theoretical multi-vector searches have a less simple problem space: how many searches would be needed to generate data-points vs. how resource-costly would it be to do N searches, vs. how reliable are those searches? 

Yet other forms of vector analytics such as dynamic-manifold modeling are probably very immature sciences as of 2026. But while outside of the MVP1 or MVP2 for detailed plans here, those are the future frontiers to pioneer. (And this may merge back into 'generative' modeling, which is generally considered separate for now and not the focus of this set of tests (though running a parallel set of tests on a 'custom trained' generative model is of course a required next-step overall). The future of generative and vector-analytics is most likely not simply-separate-areas (as even crude 'RAG' may illustrate today).



# 1. The Problem-Space of Document-Corpus Quantitative Vector Analytics
A primary use of a vector-database is for single document or few document search, whereby a deep-learning concept-meaning-context vector (unhelpfully called an 'embedding-vector' in jargon) is used to identify a closest match to a query. This is a significant tool that allows for sub-symbolic 'unstructured' search among 'unstructured' data: a classic structured search in a tabular or structured database is a ~"SQL" type search where data are tabular in fixed rows and columns with fixed datatype values. E.g. You might have a database of id_numbers and person_age, you can do direct analytics on these data, or to search for a specific record, say: what is the age of a given id_number?

Technologies of vector-search primarily allow for search, for example if you have unstructured text-blob documents that mention an id_number and an age, you can, with reasonable reliability find the document or documents that contain that information (though structuring the output of the search is another problem).

Analytics, or meta-data-analytics about the 'corpus' or 'documents' (as Natural Language Processing and Data Science discipline-fields often refer to the overall dataset and the non-tabular pieces of that dataset) tends to be a weakness of vector-search systems. For example, a vector-database of a book (or library of books) can be exceptionally good at answering individual-fact search questions such as: What did character X say about character Y's pet? 
(For example: What did Ron say to Harry and Hermiony about Hagrid's pet dragon after it bit Ron? (Note: There is also an unknown factor in terms of domain-specific training being training into the vector-model (which is very broadly the same as more famous 'generative' models, made popularly-known by Chat-bot-LLM web platforms).) Asking this question of a structured-tabular database of rows and columns (or graph-type, etc.) with a structured query (unless deliberately created just for this question) would be somewhere between challenging and utterly impossible to even define. But analytical or meta-data questions about the corpus cannot be done using a single-search mechanism. To a human user, there is often a blurry spectrum ranging from 'narrow single fact search' questions (like above) to about-the-corpus questions such as: How many times is X mentioned? or How many chapters are there? or How does X event compare with a distant Y event? Or What is the oldest document-record? Some of these are multi-point comparison conceptual questions, some of these are tabular-data analytical questions.

Just because vector-analytics is rarely done does not mean that it cannot be. But what kinds of vector-analytics are low-hanging-fruit, which are more difficult, which are likely infeasible, and what might need to be known before-hand to make a set of vector-analysis tools that will fit enough of the expected questions in a domain? 


## 'The Quorum Rule' 
The Problem of Undefined Questions/Tasks: If you gave X task to 10 competent people, could you reasonably-often get one correct solution? 

One of the issues here, that you may have seen coming in the above spectrum of query-types, is overly broad or vague questions. Some questions can helpfully if vaguely be answered such as: Who is Ron? (Answer: Ron is Harry's school-friend.), where the answer does not need to be exactly the same or uniform in order to be useful. Other questions, such as "What does Harry do?", or "What are books about?", or "What happens in history?", or "What results from physics?", or "What is?", are undefined or under-defined, where there is no practical or technical 'task' that is a satisfactory solution.
Another way to look at this task-definition problem may be more simply rooted in the history of AI-Data-Science, and may provide us with a helpful rule of thumb benchmark. The original 1956 definition of "AI" was that if you have a defined task that a human-person can do, can a STEM-Tech way to do that task without a person? Let's use this 'Must be able to be done by a person' clause and make a test-rule: If you gave X task to 10 competent people, could you reasonably often get one correct solution? E.g. 10 people can identify a cat photo. 10 people can identify a yellow cat. 10 people can count how many yellow cats there are. 10 people can read a sum-total from a clear PDF. 10 people could compare N clear PDF sum-totals. But 10 people cannot turn a vague PDF (say, post-modern poetry) into a single structured table, or identify exact objects in a blurry mess of pixels that show nothing in particular (or any Rorschach test type question). As an important project planning note, it is often very difficult for people to perceive whether the tasks they are considering and proposing are clearly defined or not (e.g. unexpected variability in poorly-formed Winograd schema questions, see: https://medium.com/@GeoffreyGordonAshbrook/lets-test-models-and-let-s-do-tasks-84777f80eb99). 

Asking if a vector database can answer an answer-able question (e.g. using 'the quorum rule') may be a helpful marker to refer to for end-users. And if end-users are not able to tell if their questions pass the quorum test or not, then they will be mixing apples and oranges in their plans and not be able to pick the right tool for a given task.

Any set of queries that pass the quorum rule test (or are otherwise clearly defined) should be able to be analyzed using a variation on the evaluation-test suite presented here.

Any query that cannot pass the quorum rule test must be acknowledged to not be clearly defined, placing it outside the reach of some methods and tools.

Defining needs and goals, defining project areas, and understanding and defining categories of types of systems are important requirements for having maintainable and maintained definitions, processes, and systems.
See: https://github.com/lineality/definition_behavior_studies et al

### Cross-Document Analytics (vs. Document-Search):

The topic here is looking at at least some cross-document analytics may be 
approached using a vector-database.

Question about uses of vector databases to compare (across) documents.
Does it make sense that you could compare the results of two queries to see which of those two returned more results/closer-results, and even quantify how many results and how close?



## Testing Corpus Analytics with Vectors

We will construct a dataset that inherently contains high-quality structured-tabular-data and corresponding unstructured text-data.

To create an empirical sandbox to pose and test clear use-cases, we can use a standard generic non-proprietary dummy-data context of "cats on the internet" (or pets more broadly). More specifically, we can leverage an existing system designed to create, test, and validate tabular-data-analysis. The tabular form of the data and the validated structured-analysis-results will form a foundation on which to build (a ground truth against which to compare) analysis of unstructured data (an unstructured, or de-structured, format of the same values in the structured data). This can be used to generate N-rows data systematically.

https://github.com/stemnetbenchmarks/synthetic_biology_datasets_python  

The existing synthetic data generator (synthetic_biology_datasets_python) has been supplemented with a description-blurb generator (csv_summary_blurb_augmentor.py) that deterministically converts known high quality (deterministically generated) tabular data into randomly varying formats of unstructured text-blobs (starting with 3 formats for mvp). While we may not expect (unstructured)vector-analytics to be able to perform every task that (structured)tabular-analytics can, this unified data-generation, testing, and validation framework allows unstructured ~queries to be tested with the same rigor as, and along side, corresponding structured ~queries. (Where 'query' here refers to whatever analysis being applied.)



## Fields and Data

(See appendix 2 for more details and examples on structured and unstructured data)

Here is one sample of structured fields put into an unstructured text-blob.
This is a {animal_type}! It is {height_cm} cm tall. It gets {daily_food_grams} grams of food each day and weighs {weight_kg} kg! It is {color}, and it was born on {birth_date}. It is {age_years} years old! It {can_swim} swim. It {can_fly} fly. It {watches_youtube} [likes to, does not like to] watch youtube. It has {number_of_friends} friends and has a popularity_score of {popularity_score}.



## Planning Vector-Analytics Queries

The primary performance goal is approximately accurate results, as in trend/pattern analysis. There probably will be unavoidable noise, collisions, and granularity issues in any vector-database system to some degree. 

The primary research task is to articulate and test as many 'types' of vector searches, possibly in a phylogeny of types: a phylogeny of analytic questions mapped over spaces of feasibility, and generalization (implementation flexibility, and range of query flexibility).  
A principle division in types is how 'generalizable' a search is.

One of the great strengths of the generative 'chat-gpt' style system is that it has a minimal interface: anything noisy in, something noisy out; no configuration needed for a specific question.


For a search-retrieval vector query, likewise, it is general. One basic configuration can be used for most basic vector-searches (this is probably rarely tweaked ala-carte per query in most deployed systems).

But of the range of possible structured-data-analytics questions, while some sets of these may be sufficiently emulated with vector analytics, how many different query-mechanics are needed? 

single reference counting and two-reference-count-comparisons (e.g. which is bigger) can probably each be used (if with varied quality results) for those classes of questions.


A. chunk vector
B. chunk metadata
C. the chunk itself
D. a structured version of that chunk


## Framing analytics questions:

Single: 
- How many documents mention cats?
- How many documents mention flying cats?

Comparative:
- Are there more dog-mentions than cat-mentions?
- What animals are best at flying?

What are known specific questions?
What are known ranges of similar questions?




Here are some example structured analytics questions (and a few unstructured questions that are likely not possible to answer with vectors)


## Test Questions Organized by Category

Note: These questions and answers are in human-form, not in equation form. This may or may not be an interface/UIUX topic. Some people will assume a query-answer will be mathematical/statistical, other people will assume the final answer will be a human expression for a human context (for example, the stated conclusion in context based on the math). As with the first question ("What is the most common animal type?") There isn't one single format of answer in all cases, (e.g. where all or more than one are equally (most) common).


## Basic Structured-Data Categories

An important theme here, which may circle back to the theme of known-common-repeating questions vs. newer and often fuzzier questions, many questions that sound 'singular' can be answered in many different valid ways. This is one of the potential strength areas of generative and vector models, where contextual meaning and unwieldy variations can be navigated (if not wonderfully or uniformly) both with input and output. 

### Basic Statistics, EDA, & Descriptive-Statistics Questions
1. "What is the most common animal type?"
ANSWER: "All animal types (cat, dog, bird, fish, turtle) appear with equal frequency."

2. "What is the average weight of all animals?"
ANSWER: "Approximately 23 kg (average of all types' means: turtles 50kg, birds 30kg, fish 20kg, dogs 10kg, cats 5kg)."

3. "What is the average weight of cats?"
ANSWER: "Approximately 5 kg."
Note: This answer could take many possible forms: common range, or a more verbose answer of more descriptive statistics, outlier analysis, etc.

4. "Which animal type is heaviest on average?"
ANSWER: "Turtles (average ~50kg)."

5. "How many birds are in the dataset?"
ANSWER: "Exactly 20% of the dataset (balanced distribution)."

6. "How many friends do cats have on average?"
ANSWER: "Approximately 5 friends (base value for cats)."

7. "Which animal type has the most friends?"
ANSWER: "Dogs (approximately 10 friends on average."


### Categorical Analysis Questions
8. "What is the most common color?"
ANSWER: "Blue (50% of all animals)."

9. "Is there a relationship between color and animal type?"
ANSWER: "Yes, strong correlation: fish are predominantly blue (80%), cats red (80%), dogs green (80%), birds gray (80%), turtles mixed (80%)."

10. "What percentage of red animals can swim?"
ANSWER: "Approximately 90% (since red animals are primarily cats, which swim at 90% rate)."

11. "What is the average weight of blue animals compared to green ones?"
ANSWER: "Blue animals are heavier (mostly fish at 20kg and some turtles at 50kg) compared to green animals (mostly dogs at 10kg)."


### Time Series Analysis Questions
12. "When do most animals tend to be born?"
"ANSWER: In winter months (December, January, February)."

13. "Is there a seasonal pattern to births?"
"ANSWER: Yes, animals are only born in winter months."

14. "How many animals were born in the first quarter of the year?"
ANSWER: Approximately 2/3 of all animals (those born in January and February)


## Correlation and Relationship Questions
15. "Is there a correlation between weight and height?"
ANSWER: "Yes, strong negative correlation - smaller animals tend to be taller,"

16. "What factors predict food consumption?"
ANSWER: "Weight is inversely related to food consumption - smaller animals eat more."

17. "Is there a relationship between age and number of friends?"
ANSWER: "Yes, cyclical relationship - number of friends follows a cosine wave pattern with age, cycling every 7-8 years"

18. "Do animals with more friends tend to be more popular?"
ANSWER: "[Yes / No] animals with more friends [do / don't] tend to be more popular."


### Boolean Pattern Questions
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


### Complex Pattern Detection Questions
24. "How does height affect popularity score?"
ANSWER: U-shaped relationship - very tall and very short animals are more popular than medium-height animals

25. "What makes an animal popular?"
ANSWER: Extreme heights (either very tall or very short) contribute to popularity

## Text Analysis Questions (Social Media)
26. "Which animal type mentions films most often in their social media?"
ANSWER: "[blank animal] mentions films most often in their social media."


27. "Which animal type has the most positive sentiment in their social media posts?"
ANSWER: "[blank animal] has the most positive sentiment in their social media posts."

28. "Which animal type has the most negative sentiment in their social media posts?"
ANSWER: "[blank animal] has the most negative sentiment in their social media posts"


29. "What topics do fish typically discuss in their social media?"
ANSWER: "Fish typically discuss [blank, blank, blank] in their social media."



## Other Research Questions (For Structured/Unstructured Queries):

1. At what point, or for what sets of questions, is a processing of 'structuring the data' into tabular (or graph, etc.) form, better than trying to use vectors or generative analysis?
(e.g. instead of processing the data (making vectors of chunks) into a vector database and using vector analytics or RAG, scanning and 'structuring' the data into a known desired tabular form and then performing normal tabular analytics on the tabular version (a converted-to structured) form of those unstructured data)?
In what ways is vector-analytics a subset of data-structuring and data-extraction problem-space, where (probably importantly) vector analysis does or does not not 'speak' or 'generate' (if intermediate) structured data?

E.g. What sets of questions could not be answered by a preliminary document-structuring step? (either not answered by tabular data, or not feasible to perform on the fly)


- What are questions we can ask (with a given interface)?
- How can we make an interface of tools manageable?

A more specific example of the features and trade-offs of a data-structuring or data-extraction system can be shown by the structured-query validation notebook itself. 

1. Completely Fixed: ~Dashboard
To the degree that the exact questions are known, a fixed 'dashboard' UI (not a notebook) could be created (such as using Plotly), or generated as a static or interactive .html doc, such that the report can be refreshed or re-generated at any time in a few seconds.

2. Flexible Specifics: Notebook
If the overall framework is the same but the person wanted to put in a novel query element (such as a Llama as a completely new pet type, or wanting to do a new comparison but of the came classic form) then a Notebook is ideal for flexibility, and a non-static ~Dashboard is likely possible.

But if the questions are not either 100% fixed in every way or not standard in type (not standard structured query questions on known tables and fields) then this data-structuring or data-extraction approach would be too rigid.

As a note: it might be the case that over-time more conserved cases would aggregate, making a dynamic or static structured-query report useful, though at the very start of a project these queries may be still undiscovered.


2. For what conceptually-based and meta-data-stateful type questions would a generative model trained on those domain specific data (and general world data) be the best option? (E.g. Asking out of the blue (not assuming that the answer is literally already in one chunk of the vector data) what recent or historical court cases may relate to a specific new court case is not a structured-data question, or a basic RAG question, or a vector-analytics question).
https://www.economist.com/science-and-technology/2026/02/18/the-human-exposome-project-will-map-how-environmental-factors-shape-health
https://www.linkedin.com/in/thomas-hartung-27a36516/

This (thankfully open-available) paper is terrifyingly full of vagaries and word-salad, but there are a number of notable issues and methods mentioned.
https://link.springer.com/article/10.1007/s00204-025-04286-8

While this paper to some degree focuses on predictive-analysis and causal-factor analysis rather than search, sort, descriptive analysis, it raises interesting questions about overlap or non-overlap between sets of questions.


3. For the sake of problem-space mapping, could it be demonstrated that there is no single vector analytics for all quorum-rule passing structured analytics questions (or contrary-wise, general configurations are found)?

4. The focus here is unstructured text and image data that could be tabular, but another large case is domains such as financial or medical tabular data to which people may want to apply vector-based tools. Are those cases separate, or somewhat spoken to in the above question types? E.g. For a given set of numerical small-business financial data, there may be a nearly infinite number of different structured analytics queries and transformations that could be done across myriad contexts to ask and answer many questions.


5. Another level of implementation question might be the feasibility of a fixed-static and tested system reliable within known tolerance for known questions, vs. an intermediary 'function-call' system that translates a human query into a coding-problem solved by generative-AI, that could potentially answer a wider range of analytic questions with a simple user interface with the likely catastrophic caveat and cost of a complete loss of reliability, with an unknown and ever fluctuating portion of answers collapsed into either incoherent slop or invisible-slop that is utter nonsense but is perceived as hard calculation. For a given use-case, 'total-mess 50% of the time' could be entirely reasonable (e.g. for a semi-smart vacuum cleaner), but in other cases this should be very soberly kept at a safe distance from consideration. 

6. Dynamic Vector Processes and The Manifold Hypothesis
While raw-vector models and generative models may be identical except for the 'head' of the output, it is likely that in that the early 2020s we did not understand more deeply how the two work and differ. 

It may be that in a generative model operates through a soft-of-recursive-ish process of starting with the same initial vector, then adding a granularity token using that input token set as context, and repeating N times, tracing out a narrow contextual 'manifold' within the meaning-concept-vector-space defined, prescribed, or proscribed, but the original input (and directed by how well (or poorly) the model has learned to trace an meaningful "manifold").

In the early 2020s people were, so far as I know (there is more research than is possible to find), exclusively interested in looking at the high level converted token output, not in, for example, having meta-data about the manifold-pattern be itself 'output' of the 'query' to the model. 

While generative manifold-tracing is one way to explore and use the N-dimensional-hypervolume (which is also a definition of an 'ecology' in biology, as a side note) this same space, and these same type of manifold patterns, can be explored either manually or with more manual or comparative intervention. 

Both for analysis and 'generation,' the scope of early 2020s work is probably not using many tools and approaches that can be brought to questions.

A persistent theme is mapping concept-data to detailed structured data, and mapping concept-"analysis" to ~deterministic (or 'same input, same output') structured-data analysis.

It may be helpful to use the Kahneman and Tversky terms for "System 1" for meaning-concept fuzzy processing (fast for biological-humans, slow for computer-machines) and "System 2" for deliberative algorithmic well defined analysis (that is slow for biological-humans but fast for computer-machines).
https://en.wikipedia.org/wiki/Daniel_Kahneman 
https://www.amazon.com/Thinking-Fast-Slow-Daniel-Kahneman/dp/0374533555 

This topic may both touch on the limitations of single-vector content capacity, and on the topic of ~searching manifolds rather than single vectors: dynamic vector analytics vs. static.


7. Raw Data Depth
One of the recurring questions here is where and how and whether what is here being approached as a vector-query question should entirely be a (likely generative or in some narrow cases GOFAI-rules) document extraction structuring step: instead of trying to ("System 2") analyze data from a vector, extract the data from the document and ("System 2") analyze those data the normal way. In such as case there is no need to make a vector database at all (if that is the entire scope). 
But...
There may be (or not) cases where there is additional relevant 'information' in the raw original unstructured data than only the assumed scope of defined fields. If you really only want to know about (sticking with out demo-context) pet color and pet activities for one fixed data-table, then possibly the vector-step is not useful. But if there are an open-ended number of questions for an open-ended set of inputs, including photos, and you do find yourself having new future questions such as 'family-ness' questions or even 'location-ness' questions, those can all be explored with the vector-database approach. 
(Note: This may also connect back to the other persistent theme which is vector-database search vs. trained model. As in the case of Zephyre, a spin-off from Mistral 7b, Zephyre was trained for about $500 (USD) in a few hours on a small-set of partially synthetic data (if I recall correctly), and that was in 2023. It is possible that a topic-specific-expert 3b or 2b or 200m model can be trained with less cost and have more cross-contextual utility than a vector database - though, again, the vector database let's you see the original documents (and yet yet again again, a properly trained model may (may) be able to reliably-enough tell you that same document source information...)

8. Result Comparison vs. Result Chaining
Is there a difference between comparing vector results and 'chaining' vector search results, where the final results will be an 'inner-join' on the two result sets?


9. Query-Chaining and Granularity
It may be useful to have a systematic way to chain vector queries to narrow down results.

Both vector models and generative models (again, the difference potentially being only in the modular 'head' and not in the vector-space itself) have a range of good-ability and limit of 'bandwidth' for combining multiple tasks at once. 

E.g. 
Let's say you want to make a query that has N parts: cats, dogs, kids, families, red, blue, swimming, beach-balls, sun, trees, birthday-ranges, specified activities, etc. etc.

Each of these can be a clear query with a well focused result. But as N grows, the focus of the model gets diluted. 

10. Stress Tests
While likely outside the range and scope of a specific narrow tool, there are various edge-case behaviors that point to unknowns in the boundaries and workings of deep-learning systems. 

11. The importance of guard-rails, case by case
Generative-guestimation technology is groundbreaking in being able to generate mostly-on-topic-ish content. For some vague-scale tasks this is sufficient, but for highly-detailed tasks this is somewhere between insufficient and obviously completely the wrong category of tool.

https://www.bbc.co.uk/mediacentre/2025/new-ebu-research-ai-assistants-news-content 

12. Continual Study & Oversight

13. Search-Retrieveal without 'generation,' a document-retrieval system.


14. 
- search and sort
- extract specific image data
- summarize
- compare/chain queries
- 


15. Mapping out cases and example of queries:
- "Pull up a chunk that matches X." (solid RAG or vector-search question)

- "Give me descriptive statistics of per chapter and whole book instance-frequencies of the letter 'e' in ranked list for by book, chapter, page (A. this is not a 'RAG' question, B. there is a serious granularity problem: most generative and embedding models cannot 'see' individual characters such as 'e')

- "How many (multi-page) PDF docs talk about 'cats' and which talk about them more-often? (This is a state-full question asking about structured-data analytics very much across-chunks)

- Are there more mentions of yellow-cats vs. pink-cats? (This is a 'comparative' multi-query operation, but it should be very possible both do two vector queries, to compare the results, and to answer the question meaningfully based on what a vector-search sees: compare quantitatively how many, how distant, chunks returned from each of two queries.)


16. Types of Questions and Number of Tools
E.g. If there are 10 types of questions, we may need to expect ~10 tools, with a user-interface designed to use a set of tools.


17. Sudden shifts between trivial and non-trivial problem spaces
Maybe relating to John McCarthy's early AI observation: "Easy things can be hard.":
'Problems' (or queries) that often sound identical or trivially-different to people (often based on empirical experiences) can be significant invisible changes in problem difficulty.

Two examples:
1. The question of a simple count is so simple and trivial that may not have been included in the original structured data query tests. Counting is not usually considered advanced analytics when your data are already structured.

But let's take the example of someone who has 10 'guide to pet owner books' and says: (Query) I want to know how many times cats are mentioned in these books?

It is a trivial python process to load each book and count string-matches for 'cat'. (e.g. ' cat ' ' cats ', forced to lowercase with no symbols).

But then, thinking about an empirical/physical book, the person might say: Great, now break that down by:
- Book
- Section
- Chapter
- Paragraph
- Sentence

To a book-group sitting in a circle holding physical books and counting manually, this question will not appear to be (or be) very different. But a vector model sees 'chunks' and it is non-trivial how either a vector model or a tabular structuring process would be able to view an .epub book or a flat .txt file in such a standardized hierarchy.

If two books are chunked in the same way, a relative comparison of 


Even books vs. animals. It would be most likely effective to use vector-comparison analytics to say if 'cat-ness' or 'dog-ness' were more prevalent. But even the question of picking out two specific books may or may not be something that a vector-chunk has any information on, and may or may not be handled by in-chunk meta-data (added to each chunk) or a previous step BEFORE vector search that did a structured-search for only chunks relating to that source-book (so, not a vector-search). 


2. 'cat' vs. 'cat-ness'
In NLP, some things can be handled as single strings, but once something cannot it rapidly escalates into an unsolve-able problem. This is in part how and where concept-vector models including generative models have made significant advances, moving from an unsolved problem to a messy-partial solution (that people sadly rush to assume is a complete unified single naive STEM solution). 



18. Time series Trends and Forcasting
In some specific cases, time series trend information may be derived from vectors, but it is likely that in most cases a data-structuring will be desired. For example, even if the 'signal' in the data is a single vector-distance strength for a concept, for the purposes of forecasting analytics it will likely be practical to have that 'engineered' field as a structured field in a structured table.


19. The context-window-problem:
In case it is not clear yet: You cannot put gigabytes of tabular data into the search-context-history 'prompt' of a generative model. 

It might sound like it makes sense to say: "Let's just ask AI instead of doing structured-queries of our tabular data!" but it is extremely unclear what the design, scope, goals, mechanics, tech-stack, etc. are for such a project. 

If someone said: "Instead of using fossil fuels to power our cars, let's use AI!" or "Instead of plants using sunlight or people eating animals, let's use AI!" Despite enthusiasm, some quasi-ideas do not make any sense.


# 19 Transient and Persistent `Edge Cases` with deep learning and embedding-vector models:
- "The Negation Problem in Embedding Models": where a concept in the model does not distinguish between positive and negative. 

# 20 Example: Metadata and Cross-Document State with Concept-Vectors

As a hypothetical ~case-study in vector search granularity and statefulness, let's say our pet-data comes from reports from different organizations: ASPCA, EU, World-Bank, Pets-Org, etc. If each organization, each report, each section of each report, and each page of each report, each have a unique-ID that can be filtered on in a structured query, then it would be possible for that to be a context of a 'hybrid' vector search, where one or more such comparative hybrid searches were pre-set. The user enters their query, and the result is a comparison dashboard showing, e.g. per organization, per report, the chunk-distance, chunk-quantity, and chunk-clustering, for each (with a possible 'headline' section showing the top N organizations and top N reports.

(and such could also be done for sections and pages for a single specified report) 

But without this 'metadata' (or data in another table linked to the chunk somehow) about where the chunk came from, the vector-model itself only sees chunks made of tokens in 'concept-matrix-space,' the vector model itself cannot analytically count how many times the string 'cat' appears in a chunk. A sufficiently sophisticated model may be able to guesstimate such a quantity, but the primary domain of the vector is tokens in a chunk in a concept-space.

Also note: For tabular data searching ten million records is simple, that is what tabular data is for. But doing and comparing ten million separate vector queries (or 10 septillion vector queries) might become too costly in money, time, server-traffic, energy, etc. And if the people doing the queries do not understand the problem-space and think all queries are equally cheap, that could be an expense issue. 


# 21. Primaries measures in vector analytics:
1. Distance of Chunks
2. Quantity of Chunks
3. Clustering of Chunks

Probably for the most part some combination of quantity of chunks at least N distance will be the primary analytic for 'signal strength' when doing queries that are cross-document (rather than single-document retrieval).

But what could varying patterns in clustering-behavior say that might be useful? How many different kinds of chunk-clustering-behavior can be identified? What qualitative or quantitative questions might this help to answer?

## 22. Tradeoffs in Vector Search
There are also likely options and tradeoffs in terms of exhaustive vs. speed-optimized searches, in case there are clear 'fast and fuzzy is good' vs. 'slow and accurate is good' use-cases. 

This may also be an area where low-code/no-code platforms create the standard (using the car acceleration analogy) 0-59 MPH very quickly, but being unable to ever reach the actual production needs, with time and costs ballooning exponentially after the initial cheap-rapid-acceleration.


## 23. Performance Degradation Patterns
Based on the results of a given test of a given size on a given model using a given set of questions, what patterns (or results) emerge in terms of how difficulties and scales and various factors create a 'performance-space' within the project's 'problems-space.'




# 2. Phylogenies / Taxonomies of Analytics:



Vector-Query-Data Analysis:
Use the results of a vector-database queries as the data to analyze.


## ~ Dependability/Consistency Tests
One area of tests that may need to emerge for foundation-models is tests around variations in performance, perhaps especially since many people seem to be firmed rooted in a Star-Trek-Next-Generation-Starship-Enterprise-AI mindset where 'AI' and 'Computer' are fantasy-mechanical and operate through uniform calculations that always happen the the same deterministic ways. Foundation-model output is very not uniform across time or across tasks. Embedding vector models are likely less temporally temperamental than generative models, but seemingly-the-same-to-humans tasks may result in not-at-all-the-same performance.
- Scale of combined single-query elements (scale)
- Concept-Qualifiers including negative/negation variants
- Database-size (scale)


## Model performance analytics:
- For the purposes of training a classification head on a foundation model, or reducing the size/cost of larger model for a specific task, or fine-tuning a model for vectors, generation, classification, or some other head, it may not be trivial to track and benchmark your model, especially if the topics and scale and details of use shift-over time (e.g. where versions of a model may diverge in details of target use conditions).


## Mechanics Types:
A. Ways to examine the results of one vector query/search:
- how many chunks
- cross-document-strength (e.g. how many chunks are with N distance from target)
- how distance-close (to the target vector: variations: closest, aggregate-closeness 'average'; e.g. a small cluster that is closer vs. a larger cluster that is further)
- how clustered
- how disperse

maybe:
--Average similarity score across top-N results
--Distribution of scores
--Number of results above a threshold
--Score of the top result specifically

B. 
We can compare the results of two queries to see which of those two returned more results/closer-results, and even quantify how many results and how close. How many variations are there on this vector-result-comparison approach?


## 'counts vs. presence/existence vs. relative prominence'
1. clear raw count (trivial, sub-string search-match)
2. fuzzy raw count (probably a structuring-step item)
3. hierarchical count (infeasible, possibly a structuring-step item)
4. specific source count comparison (infeasible without metadata/structuring)
5. relative prominence comparison (trivial, vector-analytics)
6. presence/existence (trivial, vector-analytics)


## EDA & Descriptive Statistics Type

Some terms and norms from common python data science are likely helpful to draw upon. For example, the python pandas-library dataframe is a bread-and-butter data-science tool (if not (intended) for production applications) and provides a useful .describe() method for standard descriptive statistics.

While the goal may be to create a vector-dataframe or hybrid (vector) dataframe, the terms 'database,' 'data table,' 'dataframe,' (and even a file types that can be converted into (and so may be equivalent with (either alone or including a meta-data file) one of the above items, such as a .csv file)) (note: 'spreadsheet' or 'sheet' are sometimes lumped in with the above, but I would prefer not to go down that quagmire road here.) are overlapping and not clearly distinguished from each-other (often based on software-specific conventions and branding (there are more terms as well such as 'data warehouse' and 'data-lake') rather than STEM definitions).

From:
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html




Examples

#### Describing a numeric Series.
```
s = pd.Series([1, 2, 3])
s.describe()
count    3.0
mean     2.0
std      1.0
min      1.0
25%      1.5
50%      2.0
75%      2.5
max      3.0
dtype: float64
```
#### Describing a categorical Series.
```
s = pd.Series(["a", "a", "b", "c"])
s.describe()
count     4
unique    3
top       a
freq      2
dtype: object
```
(end quote from pandas .describe() documentation)

And 'Value Counts'

To give the count or normalized-proportion of each unique field-value or 'class' (as in known-class classification).

See:  https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html 



Where 'pre-filtered' means having additions query-filters before doing the ~descriptive statistics:

Standard sets of descriptive questions may be:

1. Single-field numeric descriptive statistics
- count
- mean
- standard deviation
- min (smallest value)
- 25% percentile (first quartile)
- (median) 50% percentile (2nd quartile)
- 75% percentile (third quartile)
- max (largest value)

And two additional very frequently useful standards:
(e.g. for estimating and filtering out potential (baseline) outliers)
- 1.5 IQR lower bound
- 1.5 IQR upper bound

2. Single-field categorical descriptive statistics
- count
- unique (how many unique categories)
- top
- freq
&

3. Pre-filtered numeric descriptive statistics 
4. Pre-filtered categorical descriptive statistics 


5. Single-field categorical value-count per field-value/class.
6. Pre-filtered categorical value-count

&

7. Single-field categorical normalized "value-count" per field-value/class.
8. Pre-filtered categorical normalized "value-count" 

Another area which sometimes can be important is mapping out where (sometimes to exclude) empty values (which can also apply to unstructured text blobs)
e.g. https://github.com/lineality/language_detect 



## Analytics-Types:

1. Non-Quorum Questions:
- Queries with no known stable interpretation
- undefined
- too broad
- under-defined
- uninterpretable

2. narrow search category: 
- Find one item
- is one item available or not 
- How many items are found?

3. Comparing N searches:
Doing N narrow searches, then an additional step of comparing the ~meta-data of those results.
- are there more A or B?
- are A or B closer-matched to their target? (vs. disperse, ambiguous)
- how close is the closest A vs. B?

4. Absolute quantitative comparison:
- probably very challenging or not possible
- "how many degrees lower is the lowest temperature from the highest"

5. Descriptive Statistics:
- probably very challenging or not possible
- likely better suited to a separate previous structuring step.
- What is the mean/media/mode height... tricky.
(vectors can't 'say' a number, but they may 'see' numbers. 
E.g. "Median age is under 5" might actually be a legitimate vector question, e.g. comparing vector-result-profiles for "age is under 5" and "age is over ten" (not idea examples there)

6. Information-flow and correlation questions:
- e.g. "Which animal types are most likely to swim?"
- e.g. "How does an animal's age affect its ability to run?"
- e.g. "Is there a relationship between age and number of friends?"
- In theory vector-analysis might actually be a shortcut to this analysis, where vector-distance between groups/culters can be a proxy for information-flow/correlation. E.g. If you can compare the optional animal vectors for the cluster of records that are 'run' or 'swim'.
- Also: this type of question might be well suited to pretrained or fine-tuned domain data (or depending on how that is done). 

7. Sorting Questions: 
(what is the oldest record), in cases where this is a 'vector-distance-sort' the difficulty is easiest, but if this is a sort that cannot be handled through vector-comparison, that may be tricky. It may be that some (more) sorting questions can be translated into vector-distance questions, though the model may need to be trained for that set of concepts. 'oldness' in theory should be able to be a vector, even if that means having a comparative set of timestamps in each chunk (that is entirely feasible, usually standard chunk metadata best practice). 

8. Time Series Analysis:
- Hmmm...
- 

9. Relative Categorial Prominence:
- what X (species, colour) is most prominent?

10. A/B testing, hypothesis testing

11. 


Other:
- Possibly compare the differences between two different distance-calculations for the same query, to get a different kind of data about the patterns.


# MVPs & MVP-1

Before running the tests, existing python code is used to generate the size of data needed and the specified unstructured blurbs. 

Tests are carries out using the .csv data that includes the unstructured_description field. Some tests only use the tabular data, some tests only use the vectorized unstructured blurb, some tests use both (hybrid).

Also before running the test, various configurations are set by the user:
1. which embedding model to use (default exists)
2. what vector-distance function to use (default exists, cosine_similarities() )
3. what the vector search threshold is (default is 0.5)
4. set a random-seed if needed for python for reproducibility
etc.

The overall workflow of MVP tests are:
1. Load .csv, probably into pandas.
2. create embedding (concept) vectors for chunks (rows) and store in 'database' (probably pandas)
3. run mvp-queries A-E on each of Tabular, Vector, Hybrid

(Note: Time-cost of query should be recorded, as that might be the main difference between compared methods.)

4. basic report comparing results, where tabular results are considered ground truth, and the other are evaluated

(MVP-1.1 will not include separate process tests such as data-structuring or negative-concept-detection)


Rational for using using 'sentence transformers' 'numpy' 'pandas' libraries and not a solution such as ChromaDB: The Black Box & low-code/no-code Problems
ChromaDB is largely a black box; it can provide top-N results but no analytics or transparency about how those results were selected. Any pseudo-analytics about those results do not describe how those results were actually obtained, and so do not describe the data-set as a whole, they only describe the black-block results and even then not compared with anything else.

ChromaDB may internally use approximate nearest neighbor search (ANN), not K-nn, with fuzzy optimizations for speed such as HNSW. In some real-world use-cases the performance may be adequate, but for testing and analytics purposes we would risk introducing abstractions, unknowns, and black box results into our framework of comparisons.

For example, there may be no way to tell if chunks are being missed due to the model, or due to the black-box-optimizations of a given low-code-no-code abstraction-layer, so trying to evaluate and compare model performance and metric performance (et al) would be under a cloud of obscurity: possibly good enough, possibly not, and no transparency into the details or meaning of the numbers used. 

Third party dependencies should be something planned for. 


#### Results: Count & Confusion Matrix
The tabular results of tests A-E form a per-row ground-truth for evaluating the results of later vector, hybrid, and other tests (e.g. re-structured data extracted from unstructured data, custom-trained generative models, classification-models, generative-agent-run-tabular-analytics, etc.)

Each row (one chunk here is one row) that is matched by a non-tabular test can/will be checked, either with a table or manually. But false negatives (omissions) will require a results table (as there is no retrieved item to inspect for a not-retrieved-item).

The tabular results may be stored in a dataframe and should also be saved as a .csv file. Likewise, the test results compared to the ground truth can be classified as TP, FP, FN, TN in a results table for later inspection, auditing, and reproduction of results.

Tests A-E will probably have a boolean ground truth table
e.g. {A}_test_ground_truth_table_{timestamp}.csv
and a file for the vector-distance/similarity score, the threshold, and the confusion matrix values (one-hot)

{A}_test_results_vector_confusion_matrix_{timestamp}.csv

Files saved in (path) {cwd}/tests/tests_{timestamp}/


1. Raw Vector Analytics

Glossary note: 'tabular' refers to a row and columns table with known datatypes and one field per column.

As an example of an empirical exploration of tradeoffs, we can look at vector-data, structured-tabular-data, structured-extracted data, and hybrid vector/structured-tabular-data by experimenting with the efficiency/correctness using fields including known time-fields in unstructured data (with a parallel structured dataset). 

e.g.
How many cats were born after T posix-time?
Adding scope: How many yellow cats were born after T posix time? (etc, with N more fields added).

For structured-tabular data this kind of search (assuming you have an available or engineered standardized time field (which IRL you must not assume), is a classic use-case for tabular data.

Tradeoffs may look like this:
1. tabular search is fast and dependable (assuming rare database problems are not an issue, and usually they are not)
2. A hybrid search where the time data are in meta-data can add reliability speed to a vector search, where only the records with X value-range for time are searched (narrowing the search space, often significantly).
3. Entirely vector (assuming it works well enough) allows for ad-hoc one-off searches.

Sets of tests: where time plus N additional fields are added to a query
e.g. time + cat (cat's birthdays after T time)
time + cat + 4 other fields
time + cat + 14 other fields


### Query Tests:
(in this case, how many cats and how many cat-docs are the same)
This might be interpreted as 'search docs about cats', abbreviated. 
A. How many cats? (one field raw doc-count)
B. How many cats + 1 filter (e.g. cats who can fly)
C. How many cats born after T (time series doc-count)
D. Stress test: (Time + 1 field) How many cats born after T who can fly
E. How many cats born after T who... (Stress test: (Time + 5 fields) )
1. weight_kg > median_cat_weight, (relative range)
2. height_cm < median_cat_weight, (relative range)
3. daily_food_grams > median_cat_weight, relative range)
4. can_run == majority_class_cat_run class,
5. animal_type == cats (fixed),

The system will use: cat specific median weight, height, food-grams and the majority class for can_run. 

The process is fixed to avoid repeatability audit issues and for extra clarity, the printed report will show the computed cat specific thresholds so results are auditable.

What exactly was missed when a match was incorrect? How is it wrong? For a false-negative (omission) we cannot dissect, but for a false-positive we CAN: we can list which fields were (or were not) correctly matched (e.g. incorrect fields made

There should explanatory fields in the confusion matrix table, one hot FP false positive fields to indicate and be able to analyze detailed false-positive match component results.
e.g. one hot boolean fields
fp_animal_type boolean
fp_weight_kg boolean
fp_daily_food_grams boolean
fp_height_cm boolean
fp_time boolean
fp_can_run boolean
+
fp_fields_count, int (total number of fp fields for that row)


the report can also summarize which fields contributed more to errors, e.g. proportion and count


#### Sample False-Positive Field(column) Overall Report
```text
 FP FIELD DIAGNOSTIC (Query E):
    Total FP rows: 4350
    Field                                FP Wrong Count   Proportion
    --------------------------------------------------------------
    fp_wrong_animal_type                            627       0.1441
    fp_wrong_birth_unix                             242       0.0556
    fp_wrong_weight_kg                             2031       0.4669
    fp_wrong_height_cm                             2105       0.4839
    fp_wrong_daily_food_grams                      2404       0.5526
    fp_wrong_can_run                               2283       0.5248
```

Note: age_years > N, may be a stretch-goal test about time interpretation (not for first MVP)


1. Tabular-Query Only: standard tabular as a baseline
This does not use the unstructured field at all,
this only uses the structured tabular fields.
A.
B.
C.
D.
E.

2. Vector-Query Only
v1: 
- "This is about a cat." as the vector search input
- "This is about a cat born after T year" (perhaps)
- "This is about a cat, a feline, born after T year and that can fly: a flying cat."
- This is about a flying cat: a feline cat who can fly, the cat weights over N kg. The cat is shorter than N cm, and has more than N friends. This can can [fly,swim,run], watches youtube and eats more than N grams of food per day.
A. 
B. 
C. 
D. 
E. 

3. Hybrid (e.g. use time-field from data)
For MVP, only tabular vs. vector. (no string search, etc.)
A.
B.
C.
D.
E.

##### Vector Variability Note
One of the uses of this tool should be to illustrate, test, experiment, and measure how unstructured vector queries (which maybe all vector queries) are subject to a persnickety contextual and many-factor dependent 'prompt-engineering' challenge.
"This is about a cat." vs. "cat, feline"

In early tests, "cat, feline" was ~0.47 but below the 0.5 match threshold (failing to ever match correctly), and for the same documents the more human (but conceptually identical, to some eye) phrase "This is about a cat." matched exactly to the data with no errors.
Note: This likely depends on the embedding model, as the classic example of "king, queen" "man, woman" vector distance (admittedly a very, very different embedding-vector-model) did deal with single words.

As in tests that can scale:
https://medium.com/@GeoffreyGordonAshbrook/modularizing-problem-space-for-ai-following-a-wedge-by-sight-ab88796c4b57 

Note: more scale can be tested as well (but you will need enough fields and data and the more you filter down the smaller that final selection is, e.g. if you want 50 fields, you will likely need to engineer the fields to allow enough rows to match positively through all those filter layers).
+ 1, +5, +15, +25, +50 etc.


### Comparison Test:
1. accuracy of only-one-item count vs. item + time (e.g. cat-count vs. cats after time T count)

## This is a single-query-test, a comparison could be done with two of these:
e.g. cats at time T1 vs. cats after T2, or Cats vs. dogs after time-T

A. A single Query Tests
B. Comparative Multi-Query Tests


## Goals:
1. How many resources are saved by using a hybrid approach, and how much correctness is gained?

2. Where is the 'scope depth limit' (Kasparov Event Horizon) for a given model

3. As quantitatively as possible, what are the tradeoffs of our four bins:
#### A. Tabular-Structured Queries/Analytics, 
#### B. Vector Queries/Analytics, 
#### C. Structuring/Extraction Queries/Analytics, 
#### D. Hybrid Queries/Analytics, 

4. Test the reliability of structuring/extracting data.
Structuring Unstructured Data Note:
Because the unstructured text was derived from a table of structured data (that can be verified and validated) it should be possible (relatively trivial) to evaluate the re-structuring of those data.

What question-groups go better (or at all) for what tools?
What cost/performance for what tools?

Note: a tabular search does not care how many modules are added to a search, for the most part.

5.  Specific Vector tests, such as negative descriptions and 'The Negation Problem in Embedding Models', which may eventually involve comparing different models, different distance metrics, and fine-tuned versions of models. 


#### Extraction Testing

Mode 1:
1. iterate N rows through table
2. make N blurbs
3. extract from blurb
4. compare to table
5. score: confusion matrix, which fields are what % of errors

mode one simply starts at the top of the table,
optional offset-range user input is trivial too.

Mode 2: (possibly a completely separate python to keep code clean)
Mode 2 is batch-extraction mode. user specifies size and quantity of batches.
e.g. for batch_size=10, batch_quantity=2
this is like N=2 above, except that instead of one row, it is one batch of rows.
the first 10 rows will all be put into one long text blob and checked, then the next 10 rows 
will be put into the next extraction. 

1. iterate B(int) batches of R(int) rows through table
2. make blurb-batches
3. extract batch of dicts from from blurbs
4. compare to table
5. score: confusion matrix, which fields are what % of errors 

For batch mode, the validation will be slightly different, as it will primarily check to see if any of the N rows match.
If match fails, we should also check to see which row might match most closely. If there is a best-fit row,
(e.g. only one row that is off by 1 or 2 fields) then we can estimate what the problem was (which fields caused it to fail). 



Other Tests:
- depending on the format of the output, a classification-head put on a fine-tuned foundation-model may be a long term strategy for a more stable and improve-able pathway for data-structuring, with the big assumption that the output is known and of a type that can be output by a known-class classification model. (e.g. binary classifications,vs. fixed mutually exclusive categories, vs. non-mutually exclusive classes)





# Appendix 1: Vector Distance Metrics

1. cosine_similarity_distance
2. correlation_distance_dissimilarity_measure
3. pearson_correlation
4. canberra_distance
5. euclidean_distance
6. manhattan_distance
7. minkowski_distance
8. squared_euclidean_distance_dissimilarity_measure
9. chebyshev_distance
10. kendalls_rank_correlation
11. bray_curtis_distance_dissimilarity
12. normalized_dot_product
13. spearmans_rank_correlation
14. total_variation_distance_dissimilarity_measure
15. mahalanobis_distance
16. wasserstein_distance

See python functions for most (not mahalanobis) here:
https://github.com/lineality/arxiv_explorer_tools 

Note: If a vector database has been designed and trained based on a given distance calculation type, performance and results may be degraded if you do not use that calculation (or cannot find out what that calculation is).



# Datasets:
- Traditional Hand-Labeled Data
- User-Feedback Data
- Deterministic-Synthetic Data
- Unit-Test type Diagnostic data
- User-Level Testing Data: clear examples of user-use-case queries)



# Appendix 2: Fields & Sample Data


## Fields:
### Original Fields
```
unique_id,animal_type,weight_kg,height_cm,age_years,number_of_friends,birth_date,birth_unix,color,can_fly,can_swim,can_run,watches_youtube,daily_food_grams,popularity_score,social_media
```
### Plus the new field
```
unstructured_description
```

### Sample data to show datatypes:
```
animal_type,weight_kg,height_cm,age_years,number_of_friends,birth_date,birth_unix,color,can_fly,can_swim,can_run,watches_youtube,daily_food_grams,popularity_score,social_media
bird,28.602669690571616,60.91621816546925,9,7,2016-12-04,1480827600,gray,False,True,True,False,85.27717719215109,10.0,Watched that new avian documentary. The cinematography was stunning!
cat,2.769048995177867,95.21641089107854,5,4,2021-01-25,1611550800,red,True,True,False,False,119.21688319456632,10.0,"Watched them clean up my fur from the couch. Entertaining show, 4/10."
```


### Proposed MVP variations in unstructured blob formats:

Note: There are two versions of this process. The main version to be used here by default does not generate negative boolean concepts (e.g. There is only "can fly." There is no 'cannot fly.'). In order to deliberately test a model's ability to see (or inability to see) positive vs. negative concepts, there is also a version of this that makes both statements.

#### format_1:

This is about a {animal_type}.It gets a daily allowance of {daily_food_grams} grams of food. It is {age_years} years old, and has {number_of_friends} friends. This {animal_type} weighs {weight_kg} kg, and is {height_cm} cm tall. It has a popularity_score of {popularity_score}. It was born on {birth_date}, and is {color}. It {can_fly} fly, and {can_swim} swim. It {watches_youtube} [likes to, does not like to] watch youtube.
#### format_2:
This {animal_type} weighs {weight_kg} kg, and is {height_cm} cm tall. It is {color} and was born on {birth_date}. It gets {daily_food_grams} grams of food each day. It is {age_years} years old. It {can_fly} fly! It {can_swim} swim! This {animal_type} has {number_of_friends} friends and has a popularity_score of {popularity_score}. It {watches_youtube} [likes to, does not like to] watch youtube.
### format_3:
This is a {animal_type}! It is {height_cm} cm tall. It gets {daily_food_grams} grams of food each day and weighs {weight_kg} kg! It is {color}, and it was born on {birth_date}. It is {age_years} years old! It {can_swim} swim. It {can_fly} fly. It {watches_youtube} [likes to, does not like to] watch youtube. It has {number_of_friends} friends and has a popularity_score of {popularity_score}.

# Appendix 3: Basic Python Setup
For .py file or notebook (.pynb file)
```
from sentence_transformers import SentenceTransformer
import numpy as np

# Same embedding model as ChromaDB uses by default
model = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus of unstructured descriptions
documents = [
   "This is a cat! It is 95.2 cm tall...",
    "This bird weighs 28.6 kg...",
    "This horse can swim"
    # etc.
]

# Generate embeddings 
embeddings = model.encode(documents)  # returns numpy array, shape (n_docs, 384)

# Query:
query = "animals that can swim"
query_embedding = model.encode([query])  # shape (1, 384)
```

#### Analytics
```python
# Cosine similarity to ALL documents (not top-K)
from numpy.linalg import norm

def cosine_similarities(query_vec, corpus_matrix):
    # query_vec: (384,) or (1, 384)
    # corpus_matrix: (n_docs, 384)
    query_vec = query_vec.flatten()
    dots = corpus_matrix @ query_vec
    norms = norm(corpus_matrix, axis=1) * norm(query_vec)
    return dots / norms

similarities = cosine_similarities(query_embedding, embeddings)
# Similarity score for every document

# Analytics:
count_above_threshold = np.sum(similarities > 0.5)
distance_distribution = 1 - similarities  # convert to distance

mean_distance = np.mean(distance_distribution)
std_distance = np.std(distance_distribution)

# Compare two queries exhaustively:
query_a_sims = cosine_similarities(model.encode(["cats"]), embeddings)
query_b_sims = cosine_similarities(model.encode(["dogs"]), embeddings)

cats_above_05 = np.sum(query_a_sims > 0.5)
dogs_above_05 = np.sum(query_b_sims > 0.5)

print(f"""

count_above_threshold -> {count_above_threshold}
distance_distribution -> {distance_distribution}
      
mean_distance -> {mean_distance}
std_distance -> {std_distance}


query_a_sims -> {query_a_sims}
query_b_sims -> {query_b_sims}

cats_above_05 -> {cats_above_05}
dogs_above_05 -> {dogs_above_05}
""")
```


# Future Questions:
- What analytic-ish questions can fully-trained (or fine-tuned) generative/vector models answer?


# Links, References:

https://www.bbc.co.uk/mediacentre/2025/new-ebu-research-ai-assistants-news-content 

https://medium.com/@GeoffreyGordonAshbrook/lets-test-models-and-let-s-do-tasks-84777f80eb99

https://en.wikipedia.org/wiki/Rorschach_test

https://en.wikipedia.org/wiki/Terry_Winograd

https://en.wikipedia.org/wiki/Winograd_schema_challenge 

https://www.economist.com/science-and-technology/2026/02/18/the-human-exposome-project-will-map-how-environmental-factors-shape-health 

https://en.wikipedia.org/wiki/Daniel_Kahneman 
https://www.amazon.com/Thinking-Fast-Slow-Daniel-Kahneman/dp/0374533555 

https://medium.com/@GeoffreyGordonAshbrook/modularizing-problem-space-for-ai-following-a-wedge-by-sight-ab88796c4b57 

https://github.com/lineality/definition_behavior_studies 
.....

TODO:

(listed where?)

Design Question
What vector-analytics-queries can be tested other than single-quantitative and multi-comparative queries such as:
1. count cats (count cats +Time; +N other fields)
2. compare-count of two queries (e.g. two times, or two animals, etc.)



watch out for undefined words

define project areas: link






