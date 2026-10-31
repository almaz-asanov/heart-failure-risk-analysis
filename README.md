# Heart Failure Clinical Data Analysis



## Project Overview



This project explores a clinical dataset and compares machine learning classification models for predicting the `DEATH\_EVENT` target.



The goal is to practice the end-to-end machine learning workflow, including exploratory data analysis, preprocessing, model evaluation, cross-validation, and comparison of classification algorithms.



**Disclaimer:** This is an educational project. The models are not intended for clinical use or medical decision-making.



## Dataset



The project uses the Heart Failure Clinical Records dataset, containing 299 patient records and 12 input features.



The target variable is `DEATH_EVENT`:

- `0` — death event not recorded during the observation period.

- `1` — death event recorded during the observation period.



The dataset includes demographic information, clinical measurements, and medical history indicators.



## Workflow



1. Loaded and validated the dataset.

2. Checked dataset dimensions, missing values, duplicate records, and target distribution.

3. Performed exploratory data analysis and visualized relationships between selected features and the target.

4. Split the data into training and test sets using stratified sampling.

5. Established a majority-class baseline.

6. Evaluated K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest.

7. Used stratified 5-fold cross-validation to compare model performance.

8. Evaluated the models on a held-out test set.



Feature scaling was included in pipelines for KNN and Logistic Regression to prevent information leakage during cross-validation.



## Models and Evaluation



The models were evaluated using:

- **Recall** — the proportion of positive cases correctly identified.

- **Precision** — the proportion of predicted positive cases that were actually positive.

- **F1-score** — the harmonic mean of precision and recall.

- **Balanced accuracy** — average recall across both classes.

- **Confusion matrix** — a breakdown of correct and incorrect predictions.



## Test Set Results



| Model | Accuracy | Recall (Class 1) | Precision (Class 1) | F1-score (Class 1) |
|---|---:|---:|---:|---:|
| KNN | 0.72 | 0.42 | 0.57 | 0.48 |
| Logistic Regression | 0.77 | 0.79 | 0.60 | 0.68 |
| Random Forest | 0.80 | 0.89 | 0.63 | 0.74 |



Random Forest achieved the highest recall and F1-score for class `1` on this test split. It correctly identified 17 of the 19 positive cases, while producing 10 false-positive predictions.



These results apply only to this particular test split and should not be interpreted as evidence of clinical performance.



## Limitations

- The dataset is small, containing only 299 records.

- Evaluation results can vary depending on the train-test split.

- The test set contains only 60 records, so a small number of predictions can substantially affect the metrics.

- The project does not establish clinical validity or suitability for real-world medical decisions.



## How to Run



Clone the repository and navigate to the project directory.



Install the required Python packages:



```bash

pip install pandas numpy matplotlib seaborn scikit-learn

```



Run the analysis:



```bash

python main.py

```



## Technologies



- Python

- pandas

- NumPy

- Matplotlib

- Seaborn

- scikit-learn



## Future Improvements



- Add automated tests for data validation and preprocessing.

- Improve project structure by separating data loading, analysis, and model evaluation.

- Investigate feature importance and model interpretability.

- Add reproducible visualizations and a clearer experiment summary.

- Evaluate model stability with repeated cross-validation.



