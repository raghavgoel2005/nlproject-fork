import sys
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
from src.config import MODEL_DIR, MAX_LENGTH, PRESENT_DIR, ROOT_DIR
from src.data import clean_text
from src.registry import get_function

hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_MODEL = None
_TOKENIZER = None

def load_inference_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER
    has_weights = (MODEL_DIR / "config.json").exists() and any((MODEL_DIR / f).exists() for f in ["model.safetensors", "pytorch_model.bin"])
    if not has_weights:
        print("Model weights not found. Running automated splitting and training pipeline...")
        from src.config import SPLITS_DIR
        if not (SPLITS_DIR / "train.csv").exists():
            split_fn = get_function("split")
            if split_fn is not None:
                split_fn()
        train_fn = get_function("train")
        if train_fn is not None:
            train_fn()
    if not (MODEL_DIR / "config.json").exists():
        sys.stderr.write(f"Error: Model not found at {MODEL_DIR}.\n")
        sys.exit(1)
    device = torch.device("cpu")
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    _MODEL.eval()
    return _MODEL, _TOKENIZER

def predict(text):
    model, tokenizer = load_inference_model()
    device = model.device
    
    is_single = isinstance(text, str)
    texts = [text] if is_single else list(text)
    
    cleaned = [clean_text(t) if isinstance(t, str) else "" for t in texts]
    results = []

    batch_size = 32
    for i in range(0, len(cleaned), batch_size):
        chunk = cleaned[i:i+batch_size]
        inputs = tokenizer(
            chunk,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
            
        for prob in probs:
            pred_id = int(prob.argmax())
            pred_label = model.config.id2label.get(pred_id, str(pred_id))
            scores = {
                model.config.id2label.get(idx, str(idx)): float(round(p, 4))
                for idx, p in enumerate(prob)
            }
            results.append({
                "label": pred_label,
                "scores": scores
            })

    return results[0] if is_single else results

def run_csv(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        cand = PRESENT_DIR / csv_path_str
        if cand.exists():
            csv_path = cand
        elif csv_path.name in ("sample", "sample.csv", "samples", "samples.csv") and (PRESENT_DIR / "sample.csv").exists():
            csv_path = PRESENT_DIR / "sample.csv"
        else:
            sys.stderr.write(f"Error: CSV file not found: {csv_path}\n")
            sys.exit(1)
        
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            sys.stderr.write(f"Error reading CSV: {e}\n")
            return

    text_col = None
    for c in ["text", "sentence", "comment", "tweet", "comments"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]

    texts = df[text_col].fillna("").astype(str).tolist()
    results = predict(texts)
    preds = [r["label"] for r in results]

    if "predicted_label" in df.columns:
        df["predicted_label"] = preds
    else:
        col_idx = df.columns.get_loc(text_col)
        df.insert(col_idx + 1, "predicted_label", preds)

    stem = csv_path.stem.replace("_labelled", "")
    out_name = f"{stem}_labelled.csv"
    out_path = csv_path.parent / out_name
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(str(out_path.resolve()))
    for idx, row in df.iterrows():
        txt = str(row[text_col])
        if len(txt) > 60:
            txt = txt[:57] + "..."
        item_id = row["id"] if "id" in row else idx + 1
        print(f"  {item_id}. [{row['predicted_label']}] {txt}")

def run_eval():
    from src.config import METRICS_FILE, LABELS
    if METRICS_FILE.exists():
        import json
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
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
    else:
        eval_fn = get_function("eval_model")
        if eval_fn is not None:
            eval_fn()
        else:
            sys.stderr.write(f"Error: Metrics not found at {METRICS_FILE}. Run src/evaluate.py to generate.\n")

def run_menu():
    while True:
        try:
            print("1. Test an individual comment")
            print("2. Show evaluation metrics")
            print("3. Process CSV from present/")
            print("4. Exit")
            choice = input("Select an option (1-4): ").strip()
            if not choice or choice.lower() in ("4", "exit", "quit", "q"):
                break
            if choice == "1":
                line = input("Enter comment: ").strip()
                if line and line.lower() not in ("exit", "quit"):
                    res = predict(line)
                    print(f"Stance: {res['label']}")
                print()
            elif choice == "2":
                run_eval()
            elif choice == "3":
                files = [f for f in PRESENT_DIR.glob("*.csv") if not f.name.endswith("_labelled.csv")]
                if not files:
                    print(f"No CSV files found in {PRESENT_DIR}")
                    print()
                    continue
                if len(files) == 1:
                    target = files[0]
                else:
                    for idx, f in enumerate(files, 1):
                        print(f"  {idx}. {f.name}")
                    sel = input("Select file number or enter name (or press enter for default): ").strip()
                    if not sel:
                        target = files[0]
                    elif sel.isdigit() and 1 <= int(sel) <= len(files):
                        target = files[int(sel) - 1]
                    else:
                        target = PRESENT_DIR / sel
                if target.exists():
                    run_csv(str(target))
                else:
                    sys.stderr.write(f"Error: File not found: {target}\n")
                print()
            else:
                if choice.lower() not in ("exit", "quit"):
                    res = predict(choice)
                    print(f"Stance: {res['label']}")
                    print()
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_menu()
    elif sys.argv[1] == "--eval":
        run_eval()
    elif sys.argv[1] == "--csv":
        if len(sys.argv) < 3:
            sys.stderr.write("Error: Missing CSV path after --csv.\n")
            sys.exit(1)
        run_csv(sys.argv[2])
    else:
        text_arg = " ".join(sys.argv[1:])
        res = predict(text_arg)
        print(f"Stance: {res['label']}")
