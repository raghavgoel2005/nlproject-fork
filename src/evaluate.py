import sys
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.config import LABELS, SPLITS_DIR, METRICS_FILE

def evaluate_predictions(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1,
        "confusion_matrix": cm
    }

def print_metrics(metrics_data):
    for lang, metrics in metrics_data.items():
        print(f"=== {lang.upper()} ===")
        print(f"Macro F1 : {metrics['macro_f1']:.4f}")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print("Confusion Matrix:")
        cm = metrics["confusion_matrix"]
        headers = [f"{l:>8}" for l in LABELS]
        print(f"{'':>10}" + "".join(headers))
        for idx, row in enumerate(cm):
            row_str = "".join(f"{val:>8}" for val in row)
            print(f"{LABELS[idx]:>10}" + row_str)
        print()

def run_evaluation(save=True):
    from inference import predict
    test_path = SPLITS_DIR / "test.csv"
    if not test_path.exists():
        sys.stderr.write("Error: test.csv not found in splits/.\n")
        sys.exit(1)

    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    languages = ["english", "hindi", "bangla"]
    metrics_data = {}

    for lang in languages:
        sub = df_test[df_test["language"] == lang]
        if sub.empty:
            continue
        y_true = sub["label"].tolist()
        texts = sub["text"].tolist()
        res = predict(texts)
        y_pred = [r["label"] for r in res]
        metrics_data[lang] = evaluate_predictions(y_true, y_pred)

    if save:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

    print_metrics(metrics_data)
    return metrics_data

if __name__ == "__main__":
    run_evaluation()
