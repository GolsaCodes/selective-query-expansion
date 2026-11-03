# Selective Query Expansion

### Learning to Select Between Original and Expanded Queries for Sparse Retrieval

## Overview

This project investigates whether Machine Learning models can predict when an LLM-expanded query will perform better than its original version in sparse information retrieval.

Building on the previous [LLM-Based Query Expansion for Dense and Sparse Retrieval](https://github.com/GolsaCodes/llm-query-expansion-retrieval) project, this work focuses on the BM25 retrieval pipeline and explores an adaptive query selection strategy.

Instead of always using the original or expanded query, the objective is to train a classifier that selects the query version expected to achieve better retrieval performance.

The approach consists of three main stages:

1. Extracting query-level features from original and expanded queries and their corresponding retrieval runs.
2. Create binary labels based on the NDCG@10 performance of each query version.
3. Training and evaluating machine learning classifiers to predict the better-performing query strategy.

## Repository Structure

```text
adaptive-query-expansion/
│
├── external/
│   └── llm-query-expansion-retrieval/  # Previous query expansion and retrieval project (Git submodule)
│
├── selective_model.ipynb
│
├── .gitmodules
├── .gitignore
│
└── README.md
```

## Repository Setup

Clone the repository together with its submodules:

```bash
git clone --recurse-submodules https://github.com/GolsaCodes/adaptive-query-expansion.git
```

If you have already cloned the repository without its submodules, initialize them using:

```bash
git submodule update --init --recursive
```

After cloning, open `selective_model.ipynb` to explore the feature extraction, dataset preparation, and classification experiments.

## Experimental Pipeline

The main experimental workflow is implemented in `selective_model.ipynb`:

1. Load the SciFact dataset and query expansion outputs from the previous project.
2. Obtain the original and expanded BM25 retrieval runs.
3. Extract query-level features.
4. Generate binary labels by comparing the NDCG@10 scores of original and expanded queries.
5. Exclude queries with equal NDCG@10 scores.
6. Train and evaluate Random Forest, XGBoost, and Logistic Regression classifiers.

The objective is to investigate whether query-level features can help predict which query representation is likely to achieve better retrieval effectiveness.

## Project Workflow

### 1. Data and Retrieval Runs

This project uses the same SciFact dataset and LLM-generated expanded queries as the previous project.

Sparse retrieval is performed using BM25. Retrieval runs are generated for both:

- Original queries
- Expanded queries

The resulting runs are used to calculate query-level features and determine which query version achieves better retrieval performance.

The previous project is included as a Git submodule.

### 2. Feature Extraction

Several features are extracted to characterize the relationship between the original and expanded queries and their retrieval behavior.

The following six features are used as model inputs:

```python
X = df[
    [
        'intersection',
        'cosine_similarity',
        'mean_rank_displacement',
        'expansion_ratio',
        'query_coverage',
        'mean_TF_density'
    ]
]
```

**Feature descriptions:**

| Feature                  | Description                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------- |
| `intersection`           | Measures the lexical overlap between original and expanded queries.                   |
| `cosine_similarity`      | Measures semantic similarity between the original and expanded query representations. |
| `mean_rank_displacement` | Captures changes in document rankings between retrieval runs.                         |
| `expansion_ratio`        | Measures the relative increase in query length after expansion.                       |
| `query_coverage`         | Measures the extent to which the expanded query covers the original query.            |
| `mean_TF_density`        | Represents the average term-frequency density associated with the query terms.        |

### 3. Target Label Generation

The target variable is `original_or_expanded`.

Labels are assigned by comparing the NDCG@10 scores of the original and expanded queries:

- **Original (0):** The original query achieves a higher NDCG@10.
- **Expanded (1):** The expanded query achieves a higher NDCG@10.

Queries for which both versions achieve exactly the same NDCG@10 are excluded because neither strategy is better.

Out of the initial 300 queries, 117 queries remain after removing ties.

This creates a binary classification task in which the model learns to predict which query strategy performs better.

### 4. Feature Correlation Analysis

The correlation between each feature and the difference in retrieval performance (`delta_ndcg`) is examined.

| Feature                | Correlation with delta_ndcg |
| ---------------------- | --------------------------: |
| cosine_similarity      |                      -0.219 |
| query_coverage         |                      -0.210 |
| intersection           |                      -0.099 |
| mean_rank_displacement |                       0.207 |
| expansion_ratio        |                       0.246 |
| mean_TF_density        |                       0.288 |

**Observations:**

- `mean_TF_density` has the strongest positive correlation with `delta_ndcg` among the selected features.
- `expansion_ratio` and `mean_rank_displacement` also show positive correlations.
- `cosine_similarity` and `query_coverage` show negative correlations.
- Overall, the correlations are relatively weak, suggesting that these individual features have limited linear relationships with the performance difference.

These results motivate further investigation into feature quality and the potential need for more informative query-level representations.

## Machine Learning Models

Three classification algorithms are evaluated:

1. **Random Forest Classifier**
2. **XGBoost Classifier**
3. **Logistic Regression**

The models use the six extracted features to predict whether the original or expanded query achieves a higher NDCG@10.

## Experimental Results

### 1. Random Forest

| Metric         |  Score |
| -------------- | -----: |
| Accuracy       | 0.6250 |
| Macro F1-score | 0.5636 |

**Classification Report:**

| Class         | Precision | Recall | F1-score | Support |
| ------------- | --------: | -----: | -------: | ------: |
| Original (0)  |      0.67 |   0.80 |     0.73 |      15 |
| Expanded (1)  |      0.50 |   0.33 |     0.40 |       9 |
| Macro Average |      0.58 |   0.57 |     0.56 |      24 |

**Confusion Matrix:**

```text
[[12  3]
 [ 6  3]]
```

The Random Forest model achieves 62.5% accuracy. However, its recall for the Expanded class is relatively low, indicating difficulty identifying queries for which expansion performs better.

### 2. XGBoost Classifier

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 0.6250 |
| Balanced Accuracy | 0.5889 |
| Macro F1-score    | 0.5901 |

**Classification Report:**

| Class         | Precision | Recall | F1-score | Support |
| ------------- | --------: | -----: | -------: | ------: |
| Original (0)  |      0.69 |   0.73 |     0.71 |      15 |
| Expanded (1)  |      0.50 |   0.44 |     0.47 |       9 |
| Macro Average |      0.59 |   0.59 |     0.59 |      24 |

**Confusion Matrix:**

```text
[[11  4]
 [ 5  4]]
```

XGBoost achieves the same accuracy as Random Forest but obtains a slightly higher macro F1-score and balanced accuracy. It also performs somewhat better at identifying the Expanded class.

### 3. Logistic Regression

Logistic Regression is evaluated using 5-fold cross-validation.

| Metric   |   Mean | Standard Deviation |
| -------- | -----: | -----------------: |
| Accuracy | 0.6493 |             0.0568 |
| F1-score | 0.6417 |             0.0589 |

**Fold-level Accuracy:**

```text
[0.5833, 0.7500, 0.6087, 0.6522, 0.6522]
```

**Fold-level F1-score:**

```text
[0.5714, 0.7429, 0.5965, 0.6462, 0.6515]
```

Logistic Regression achieves a mean accuracy of approximately 64.93% and a mean F1-score of approximately 64.17% across the five folds.

Among the reported experiments, it shows the highest mean accuracy and F1-score.

## Discussion

The experimental results indicate that predicting the better-performing query strategy is a challenging classification task.

Although Logistic Regression achieves a mean accuracy of approximately 64.9%, the overall predictive performance remains limited.

Several factors may contribute to this result:

- **Limited dataset size:** Only 117 non-tied queries remain from the original 300 queries.
- **Feature limitations:** The six extracted features may not capture enough information about when query expansion improves retrieval.

## Limitations and Future Work

This study provides an initial investigation of adaptive query expansion, but several improvements remain possible:

- Extract additional query-level and retrieval-based features.
- Investigate feature interactions and nonlinear relationships.
- Evaluate feature importance and perform feature selection.
- Explore alternative classification algorithms.
- Increase the number of evaluated queries and datasets.
- Apply more rigorous cross-validation and hyperparameter optimization.
- Investigate whether regression-based approaches can predict the expected NDCG@10 difference directly.

An important direction for future work is developing a more reliable decision model that can determine when query expansion is beneficial without requiring retrieval evaluation for every query.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- BM25
- Gemini Flash 3.5
- BEIR / SciFact
- Hugging Face Transformers

## Conclusion

This project explores an adaptive approach to query expansion by training Machine Learning models to select between original and LLM-expanded queries for sparse retrieval.

Using six query-level features and NDCG@10-based labels, three classifiers are evaluated: Random Forest, XGBoost, and Logistic Regression.

The results show that the selection task is challenging with the current feature set and dataset size. Although Logistic Regression achieves the best reported mean cross-validation performance, the predictive results indicate that more informative features and further experimentation are needed.

The project provides a foundation for developing adaptive retrieval systems that apply query expansion selectively rather than uniformly.
