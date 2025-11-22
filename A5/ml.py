from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# Configuration
DATA = Path("data_proc/grouped_chunks.csv")
OUT_IMG_DIR = Path("plots_kfold")
OUT_IMG_DIR.mkdir(exist_ok=True, parents=True)
N_SPLITS = 3 # each fold will include all 3 APTs for stratification 
SEED = 42

df = pd.read_csv(DATA)
print("Total samples:", len(df))
print("Class distribution:\n", df["label"].value_counts())

labels_all = sorted(df["label"].unique())

# Confusion matrix plot
def plot_confusion_matrix(cm, classes, title, out_path):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 verticalalignment="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# Running K-fold
def run_kfold_experiment(name, clf, ngram_range):
    print("\n" + "="*80)
    print(f"{name} with ngram_range={ngram_range}")
    print("="*80)

    pipe = Pipeline([
        ("vec", CountVectorizer(
            token_pattern=r"[^ ]+",
            ngram_range=ngram_range,
            min_df=1
        )),
        ("clf", clf)
    ])

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_y_true, all_y_pred = [], []
    fold = 1
    fold_metrics = []

    for train_idx, test_idx in skf.split(df["text"], df["label"], groups=df["group"]):
        X_train, X_test = df.iloc[train_idx]["text"], df.iloc[test_idx]["text"]
        y_train, y_test = df.iloc[train_idx]["label"], df.iloc[test_idx]["label"]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        print(f"\nFold {fold}: accuracy={acc:.3f}, macro-F1={f1_macro:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0, digits=3))

        cm = confusion_matrix(y_test, y_pred, labels=labels_all)
        plot_confusion_matrix(
            cm,
            classes=labels_all,
            title=f"{name} {ngram_range} Fold {fold}",
            out_path=OUT_IMG_DIR / f"cm_{name}_{ngram_range[0]}-{ngram_range[1]}_fold{fold}.png"
        )

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        fold_metrics.append((acc, f1_macro))
        fold += 1

    acc_mean = np.mean([m[0] for m in fold_metrics])
    f1_mean = np.mean([m[1] for m in fold_metrics])
    print(f"\n{name} ({ngram_range}) Mean Accuracy: {acc_mean:.3f}, Mean Macro-F1: {f1_mean:.3f}\n")

# Main
if __name__ == "__main__":
    setups = [
        ("SVM",  LinearSVC(),                         (1, 1)),
        ("SVM",  LinearSVC(),                         (1, 2)),
        ("KNN3", KNeighborsClassifier(n_neighbors=3), (1, 1)),
        ("KNN3", KNeighborsClassifier(n_neighbors=3), (1, 2)),
        ("DT",   DecisionTreeClassifier(random_state=SEED), (1, 1)),
        ("DT",   DecisionTreeClassifier(random_state=SEED), (1, 2))
    ]

    for args in setups:
        run_kfold_experiment(*args)
