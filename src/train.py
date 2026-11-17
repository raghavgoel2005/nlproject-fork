import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
from sklearn.metrics import f1_score
from src.config import (
    ROOT_DIR, SPLITS_DIR, MODEL_DIR, LOGS_DIR, CHECKPOINTS_DIR,
    MODEL_NAME, LABELS, LABEL2ID, ID2LABEL, MAX_LENGTH,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, RANDOM_SEED,
    get_device
)

hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class StanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LENGTH):
        self.texts = list(texts)
        self.labels = [LABEL2ID[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def setup_logger():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOGS_DIR / "train.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def compute_weights(labels):
    counts = np.bincount([LABEL2ID[l] for l in labels], minlength=len(LABELS))
    total = len(labels)
    weights = total / (len(LABELS) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)

def evaluate_val(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return float(f1)

def train_single_run(train_df, val_df, lr, epochs, batch_size, run_id, logger, quick_test=False):
    set_seed(RANDOM_SEED + run_id)
    device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    if quick_test:
        train_df = train_df.head(16)
        val_df = val_df.head(16)
        epochs = 1

    train_dataset = StanceDataset(train_df["text"], train_df["label"], tokenizer)
    val_dataset = StanceDataset(val_df["text"], val_df["label"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    weights = compute_weights(train_df["label"]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_f1 = -1.0
    run_ckpt_dir = CHECKPOINTS_DIR / f"run_{run_id}"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if quick_test:
                break

        val_f1 = evaluate_val(model, val_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))
        logger.info(f"Run {run_id} | Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(run_ckpt_dir)
            tokenizer.save_pretrained(run_ckpt_dir)

    return best_val_f1, run_ckpt_dir

def train_model(quick_test=False):
    logger = setup_logger()
    logger.info("Starting training process")
    
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    if not train_path.exists() or not val_path.exists():
        sys.stderr.write("Error: Splits not found. Run python -m src.run split first.\n")
        return None

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    val_df = pd.read_csv(val_path, encoding="utf-8-sig")

    if quick_test:
        _, ckpt = train_single_run(train_df, val_df, 2e-5, 1, 16, 0, logger, quick_test=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(ckpt)
        mod = AutoModelForSequenceClassification.from_pretrained(ckpt)
        mod.save_pretrained(MODEL_DIR)
        tok.save_pretrained(MODEL_DIR)
        return MODEL_DIR

    opt_lr = 2.5e-5
    opt_epochs = 4
    opt_batch_size = 16

    logger.info(f"Training optimal model with lr={opt_lr}, epochs={opt_epochs}, batch_size={opt_batch_size}")
    best_val_f1, ckpt_dir = train_single_run(
        train_df, val_df, opt_lr, opt_epochs, opt_batch_size, 0, logger
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_tok = AutoTokenizer.from_pretrained(ckpt_dir)
    best_mod = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    best_mod.save_pretrained(MODEL_DIR)
    best_tok.save_pretrained(MODEL_DIR)
    logger.info(f"Final model saved to {MODEL_DIR} with Validation Macro F1: {best_val_f1:.4f}")

    return MODEL_DIR
