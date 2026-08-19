# Po Valley Methane Forecasting

This project investigates the short-term forecasting of weekly atmospheric methane concentrations over the Po Valley using Copernicus Sentinel-5P data.

It is being developed as part of a personal machine learning portfolio and covers the end-to-end workflow of a regression problem, from satellite data acquisition and exploratory data analysis to feature engineering, model training, and evaluation. A lightweight RAG assistant based on technical documentation and scientific literature is added as a complementary extension.


## Project overview

The project studies the performance of different classical machine learning models for short-term forecasting of Sentinel-5P XCH4 observations over the Po Valley, with particular attention to their performance relative to a persistence baseline.

The forecasting task consists of predicting the weekly mean XCH4 value for a given week and geographic cell, using information from the previous weeks.

The target quantity, XCH4, is the total column-averaged dry-air mole fraction of methane, and is not a direct measure of methane emissions from the ground.

As a complementary section, a lightweight RAG assistant is implemented, based on a short technical report of this project and a few scientific articles. The system covers the end-to-end pipeline from corpus chunking, query-directed retrieval and reranking, and answer generation. The default generative model is a small local open-source language model served through Ollama.

## Dataset

The dataset is a multiyear dataset extracted from the Copernicus Data Space Ecosystem, filtered using a geographic mask and a temporal cell-coverage mask, and subsequently subjected to cleaning and feature engineering. 

The raw version can be represented as an XCH4 dataset defined over a three dimensional time x latitude × longitude cube.

The selected temporal interval covers the years 2019 to 2024, with 2024 reserved for final model testing and evaluation.


### Data preprocessing

Missing timestamps are filled.

A geographic mask covering a manually defined region corresponding to the Po Valley is applied.

A temporal cell-coverage mask that excludes geographic cells with an insufficient number of observations is applied.

### Feature engineering

Engineered features include XCH4 observations over the previous two weeks, XCH4 variation over the last week, mean value and standard deviation over the previous three weeks, period of the year encoded as a cyclic sinusoidal variable.

### Model-ready dataset

The model-ready version of the dataset is formed by the following predictor and metadata columns:

| Column | Data type |
| ------------- | -------------- |
| Week | datetime64[ns] |
| Target_week | datetime64[ns] |
| cell_id | int64 |
| x | float64 |
| y | float64 |
| CH4 | float32 |
| ch4_prev_1w | float32 |
| ch4_prev_2w | float32 |
| ch4_change_1w | float32 |
| ch4_mean_last_3w | float64 |
| ch4_std_last_3w | float64 |
| season_sin | float64 |
| season_cos | float64 |

and the following target:

| Column | Data type |
| ------------- | -------------- |
| target_next_week | float32 |

## Temporal validation

Three candidate machine learning models are tested and compared with each other and with the persistence baseline over two disjoint annual temporal validation folds, 2022 and 2023. For each fold, the preceding years are used as the training period.

The best-performing model is then trained again over the 2019-2023 period and evaluated on the final 2024 test period.

## Models

### Persistence

The persistence baseline is the naive model that predicts the average XCH4 value for a given week to be the same as the previous week.

### Ridge

The Ridge model is a standard ridge linear regression model with regularization parameter alpha=1. A normalization scaling was applied to the features before training. 

### Default-like Random Forest

In order to observe the performance of a non-linear predictive model, a Random Forest regressor was trained with default-like parameters:

```python
n_estimators=300,
random_state=42,
n_jobs=-1
```

### Regularized Random Forest

A customized Random Forest configuration with large leaf sizes, selected through preliminary parameter tuning, was also trained:

```python
n_estimators=300,
max_depth=None,
min_samples_leaf=25,
max_features="sqrt",
max_samples=0.8,
random_state=42,
n_jobs=-1,
```

This version of the Random Forest model was selected among several Random Forest parameter settings after a preliminary experiment on a development subset.

## Final results

The temporal validation results show that Ridge is the only candidate model that consistently improves over the persistence baseline across both validation folds. Ridge is therefore selected for the final multiyear train-test experiment.

The final 2024 test shows the following results for Persistence and Ridge:

| Model | MAE | RMSE | R2 | prediction_std | correlation |
|------|---------:|----------:|-------:|-----:|--------:|
| Persistence | 16.220643 | 21.245858 | 0.018069 | 20.769970 | 0.494532 |
| Ridge | 14.749186 | 19.673392 | 0.158041 | 14.876011 | 0.485777 |

The table shows that Ridge is the best-performing model on the final 2024 test, outperforming the persistence baseline on both MAE and RMSE. In particular, on the final 2024 test period, Ridge reduces MAE by approximately 9.1% and RMSE by approximately 7.4% relative to the persistence baseline.

In the final test, Ridge errors are higher than in the validation folds, but the model retains a modest out-of-sample advantage over the persistence baseline despite substantial week-to-week variability in the XCH4 observations

## Scientific interpretation

The results suggest that a regularized linear combination of recent XCH4 history, spatial information, and seasonal features contains useful predictive information beyond the current-week observation.

This predictive relationship should not be interpreted as evidence of a causal or intrinsically linear physical dependence.

## RAG extension

A RAG system was implemented to answer simple questions about the project and the scientific interpretation of Sentinel-5P methane observations.

The retrieval corpus of the system is formed by six scientific articles and an internal technical report of the results of this project. 

The complete end-to-end pipeline from the query request to the answer generation requires the chunking of the global corpus, the embedding of the chunks, the retrieval of the most relevant chunks, a reranking of the most relevant chunks operated by an additional model based on concept similarity with the query, a final selection on the reranked chunks, a final prompt generation, and an answer generation operated by an Ollama local language model.

All the documents of the corpus are listed below and included in the repository.

### Scientific articles

- Apituley et al., Sentinel-5 precursor/TROPOMI Level 2 Product User Manual Methane. Royal Netherlands Meteorological Institute (2022), doc.number SRON-S5P-LEV2-MA-001.

- Ehret et al., Global Tracking and Quantification of Oil and Gas Methane Emissions from Recurrent Sentinel-2 Imagery. Environ. Sci. Technol 56 (14), pp. 10517–10529 (2022) (preprint version used), DOI 10.1021/acs.est.1c08575, arxiv v. 2110.11832.

- Hu et al., The operational methane retrieval algorithm for TROPOMI. Atmos. Meas. Tech. 9, pp. 5423-5440 (2016), DOI 10.5194/amt-9-5423-2016.

- Lauvaux et al., Global assessment of oil and gas methane ultra-emitters. Science 375, pp. 557–561 (2022), DOI 10.1126/science.abj4351.

- Peng et al., High-resolution assessment of coal mining methane emissions by satellite in Shanxi, China. iScience 26 (12), pp. 1-13 (2023), DOI 10.1016/j.isci.2023.108375.

- Sha et al., Validation of methane and carbon monoxide from Sentinel-5 Precursor using TCCON and NDACC-IRWG stations. Atmos. Meas. Tech. 14, pp. 6249–6304 (2021), DOI 10.5194/amt-14-6249-2021.

### Models

- Default embedding model: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`.

- Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`.

- Ollama default generative model: `qwen3:1.7b`.

### Complete pipeline function

An additional module `pipeline.py` is included to define a function `answer_query` that runs the complete RAG pipeline in a single function call. It can be used to run a single experiment or query without executing the individual pipeline steps manually.

## Repository structure

```text
Po-Valley-Methane-Forecasting/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── regions/
│
├── notebooks/
│   ├── 01_data_access_and_inspections.ipynb
│   ├── 02_preprocessing_and_candidate_dataset.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training_and_evaluation.ipynb
│   └── 05_rag_experiments.ipynb
│
├── rag/
│    └── documents/
│
├── src/po_valley_methane_forecasting/
│       ├── __init__.py
│       ├── data_access.py
│       ├── evaluation.py
│       ├── features.py
│       ├── models.py
│       ├── paths.py
│       ├── preprocessing.py
│       ├── validation.py
│       └── rag/
│           ├── __init__.py
│           ├── embedding.py
│           ├── generation.py
│           ├── ingestion.py
│           ├── pipeline.py
│           ├── reranking.py
│           └── retrieval.py
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Installation and usage

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/P-Gigli/Po-Valley-Methane-Forecasting
cd Po-Valley-Methane-Forecasting

python -m venv venv
```
Activate the environment and install the required dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The main machine learning workflow can be reproduced through the notebooks
in the `notebooks/` directory, which should be run in numerical order.

### Running the RAG assistant

The RAG extension uses Ollama for local answer generation.
Ollama must therefore be installed separately from the Python dependencies.

The default generative model is:

`qwen3:1.7b`

After installing Ollama, make sure the model is available locally:

```bash
ollama run qwen3:1.7b
```

The complete RAG pipeline can then be tested through `answer_query()`:


```python
from po_valley_methane_forecasting.rag.pipeline import answer_query

answer = answer_query(
    "What does Sentinel-5P XCH4 represent?"
)

print(answer)
```

For step-by-step RAG experiments, see `05_rag_experiments.ipynb`.

## Future work

Possible future improvements include:

- Extension or correction of the geographic region.

- Changing of the generative model in the RAG system, possibly the Ollama model Qwen 3:4b or superior, or a remote API, for example OpenAI API. This would provide an informative comparison between different generative language models on a limited corpus of documents.  