import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate


#heart_failure_data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
data_path = (Path(__file__).resolve().parent / 'heart_failure_clinical_records_dataset.csv')
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')
    data = pd.read_csv(path)
    required_columns = {
        "age",
        "ejection_fraction",
        "serum_creatinine",
        "DEATH_EVENT",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    if data.empty:
        raise ValueError("The dataset is empty.")
    if data["DEATH_EVENT"].isna().any():
        raise ValueError("The target column contains missing values.")
    if not data["DEATH_EVENT"].isin([0, 1]).all():
        raise ValueError("The target column contains not 0 or 1.")
    return data


heart_failure_data = load_data(data_path)
print("Dataset shape:", heart_failure_data.shape)
print("Missing values:", heart_failure_data.isna().sum().sum())
print("Duplicate rows:", heart_failure_data.duplicated().sum())
print("Target distribution:",heart_failure_data["DEATH_EVENT"].value_counts())
print(heart_failure_data.head())

age_bins = [40, 50, 60, 70, 80, 90, 100]
age_labels = ["40–49", "50–59", "60–69", "70–79", "80–89", "90–99"]
age_data = heart_failure_data.assign(age_group = pd.cut(heart_failure_data["age"], bins = age_bins, labels = age_labels, right = False))

age_summary = (
    age_data.groupby("age_group", observed = False)
    .agg(
        participant_count=("DEATH_EVENT", "size"),
        death_count=("DEATH_EVENT", "sum"),
        death_rate=("DEATH_EVENT", "mean"),
    )
    .reset_index()
)

print("\nDeath-event summary by age group:")
print(age_summary)

sns.barplot(
    data = age_summary,
    x = "age_group",
    y = "death_rate",
)

plt.xlabel("Age group")
plt.ylabel("Proportion with DEATH_EVENT = 1")
plt.title("Observed death-event proportion by age group")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

sns.relplot(x = "serum_creatinine", y = "ejection_fraction", hue = "DEATH_EVENT", kind = "scatter", style = "smoking", data = heart_failure_data, aspect = 16/9);
agg_data2 = heart_failure_data.groupby(['age', 'ejection_fraction'], as_index = False).DEATH_EVENT.sum()
agg_data2.columns = ['Age', 'ejection_fraction', 'Deaths']
agg_data2_p = agg_data2.pivot(index = 'Age', columns = 'ejection_fraction', values = 'Deaths')
agg_data2_p.fillna(0, inplace = True)
agg_data2_p.head(20)
plt.subplots(figsize=(20,15))
sns.heatmap(agg_data2_p, annot = True)
plt.show()

X = heart_failure_data.drop('DEATH_EVENT', axis = 1)
y = heart_failure_data['DEATH_EVENT']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

neighbors_values = [3, 5, 7, 9, 11, 15, 21, 25] # best number of neighbours is 5
cv_results = []

for k in neighbors_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "recall": "recall",
            "f1": "f1",
            "balanced_accuracy": "balanced_accuracy",
        },
    )

    cv_results.append({
        "n_neighbors": k,
        "mean_recall": scores["test_recall"].mean(),
        "std_recall": scores["test_recall"].std(),
        "mean_f1": scores["test_f1"].mean(),
        "mean_balanced_accuracy": (
            scores["test_balanced_accuracy"].mean()
        ),
    })

cv_results_df = pd.DataFrame(cv_results)

print("\nKNN cross-validation results:")
print(
    cv_results_df
    .sort_values("mean_recall", ascending=False)
    .round(3)
    .to_string(index=False)
)

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors = 15))
])

knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

print("Baseline model:")
print(classification_report(y_test, baseline_pred, zero_division = 0))

print("KNN model:")
print(classification_report(y_test, y_pred, zero_division = 0))

print("KNN confusion matrix:")
print(confusion_matrix(y_test, y_pred))