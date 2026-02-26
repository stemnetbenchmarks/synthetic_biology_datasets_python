# https://github.com/stemnetbenchmarks/synthetic_biology_datasets_python

# 1. Clone Repo and go to project directory/folder
- decide where to put the project e.g. .../code/ or .../projects/
- in terminal clone repo:
```bash
https://github.com/stemnetbenchmarks/synthetic_biology_datasets_python.git
```
or w/ ssh:
```bash
git@github.com:stemnetbenchmarks/synthetic_biology_datasets_python.git
```
- go into that repo:
```bash
cd synthetic_biology_datasets_python
```

# 2. Build python environment: venv env
```bash
python -m venv env; source env/bin/activate
python -m pip install --upgrade pip; python -m pip install -r requirements.txt
```

# 3. Generate Data-Set Bundle
### Method 1: python script
- in env, run data generation .py file
```bash
python data_generation_tools_python_v9.py
```
### Method 2: run Analyzer in jupyter notebook
- Start Notebook:
```bash
jupyter lab
```
- open synthetic_bio_validation_check_notebook_v5.ipynb
- run all cells

# 4. Compare other test results to this benchmark
- double check that these data are being generated and varified/validated correctly
- use a given other tool to generate analysis results
- compare those results with what the benchmark says they should be

# 5. Optional: customize
- change size of dataset easily
- change ranges of deterministic output (patterns in data)
- deeper modification as you like

