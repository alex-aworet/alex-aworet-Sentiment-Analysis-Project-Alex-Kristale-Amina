# Torch ML libraries
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, List
import sys
import os

# Import data processing functions and dataset class
from src.data_extraction import load_file, check_columns
from src.data_processing import (
    clean_dataset,
    split_data,
    ReviewDataset
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set the model name
MODEL_NAME = 'bert-base-cased'


def get_device() -> torch.device:
    """Get GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(
    n_classes: int,
    model_name: str = MODEL_NAME,
    dropout: float = 0.1
) -> BertForSequenceClassification:
    """
    Create a BERT model for sequence classification.

    Args:
        n_classes: Number of output classes
        model_name: Pre-trained model name
        dropout: Dropout probability for classifier
    """
    device = get_device()
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_classes,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout
    )
    model = model.to(device)
    return model


def create_optimizer_and_scheduler(
    model: nn.Module,
    train_data_loader: DataLoader,
    epochs: int,
    learning_rate: float = 2e-5
) -> Tuple[optim.AdamW, any]:
    """
    Create optimizer and learning rate scheduler.
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_data_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: any,
    n_examples: int
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    """
    model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    n_examples: int
) -> Tuple[float, float]:
    """
    Evaluate the model on validation/test data.
    """
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def train_model(
    model: nn.Module,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    n_train_examples: int,
    n_val_examples: int,
    epochs: int,
    learning_rate: float = 2e-5,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    model_save_path: str = 'best_model_state.bin',
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train for several epochs, saving the best model.

    Args:
        model: The model to train
        train_data_loader: DataLoader for training data
        val_data_loader: DataLoader for validation data
        n_train_examples: Number of training examples
        n_val_examples: Number of validation examples
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        use_amp: Whether to use automatic mixed precision
            (currently ignored, for future use)
        gradient_accumulation_steps: Number of steps to accumulate
            gradients (currently ignored, for future use)
        model_save_path: Path to save the best model
        verbose: Whether to print progress
    """
    device = get_device()

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        train_data_loader,
        epochs,
        learning_rate
    )

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            device,
            scheduler,
            n_train_examples
        )

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            device,
            n_val_examples
        )

        if verbose:
            print(f"Train loss {train_loss:.4f}, acc {train_acc:.4f}")
            print(f"Val   loss {val_loss:.4f}, acc {val_acc:.4f}")
            print()

        train_acc_val = (train_acc.item() if torch.is_tensor(train_acc)
                         else train_acc)
        history['train_acc'].append(train_acc_val)
        history['train_loss'].append(train_loss)
        val_acc_val = (val_acc.item() if torch.is_tensor(val_acc)
                       else val_acc)
        history['val_acc'].append(val_acc_val)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), model_save_path)
            best_accuracy = val_acc
            if verbose:
                print(f"Best model saved with accuracy: {best_accuracy:.4f}")

    return dict(history)


def main():

    """Main function to demonstrate the complete pipeline."""

    # ==========================
    # 1. LOAD AND PROCESS DATA
    # ==========================
    print("=" * 50)
    print("STEP 1: Loading and processing data")
    print("=" * 50)

    # Load the dataset
    path = "data/dataset.csv"
    print(f"Loading dataset from {path}...")
    df = load_file(path)

    if df is None or not check_columns(df):
        print("Error: Dataset invalid or missing columns.")
        return

    print(f"Dataset loaded: {len(df)} rows")

    # Clean the dataset
    # (removes duplicates, adds sentiment labels, cleans text)
    print("\nCleaning dataset...")
    df_clean = clean_dataset(df)
    print(f"Cleaned dataset: {len(df_clean)} rows")
    print(f"Sentiment distribution:\n{df_clean['sentiment'].value_counts()}")

    # Split into train and validation sets
    print("\nSplitting data into train/validation...")
    train_df, val_df = split_data(df_clean, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")

    # ==========================
    # 2. CREATE TORCH DATASETS
    # ==========================
    print("\n" + "=" * 50)
    print("STEP 2: Creating PyTorch datasets")
    print("=" * 50)

    # Initialize tokenizer
    tokenizer_name = "bert-base-cased"
    max_len = 160
    print(f"Using tokenizer: {tokenizer_name}")
    print(f"Max sequence length: {max_len}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = ReviewDataset(
        texts=train_df["content"].to_numpy(),
        labels=train_df["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    print(f"Training dataset created: {len(train_dataset)} samples")

    # Create validation dataset
    print("Creating validation dataset...")
    val_dataset = ReviewDataset(
        texts=val_df["content"].to_numpy(),
        labels=val_df["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    print(f"Validation dataset created: {len(val_dataset)} samples")

    # Example: inspect a sample
    print("\nExample sample from training dataset:")
    sample = train_dataset[0]
    print(f"  Text (truncated): {sample['text'][:100]}...")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Label: {sample['labels']}")

    # ==========================
    # 3. CREATE DATA LOADERS
    # ==========================
    print("\n" + "=" * 50)
    print("STEP 3: Creating data loaders")
    print("=" * 50)

    batch_size = 16
    print(f"Batch size: {batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for Windows, can increase on Linux/Mac
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # ==========================
    # 4. CREATE AND TRAIN MODEL
    # ==========================
    print("\n" + "=" * 50)
    print("STEP 4: Creating and training model")
    print("=" * 50)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create model
    n_classes = 3  # negative, neutral, positive
    print(f"\nCreating model with {n_classes} classes...")
    model = create_model(n_classes=n_classes, dropout=0.3)
    print("Model created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training configuration
    epochs = 3  # Use more epochs in production (e.g., 10)
    learning_rate = 2e-5
    use_amp = torch.cuda.is_available()  # Use mixed precision on GPU
    gradient_accumulation_steps = 1

    print("\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Mixed precision training: {use_amp}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")

    # Train the model
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    history = train_model(
        model=model,
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        n_train_examples=len(train_dataset),
        n_val_examples=len(val_dataset),
        epochs=epochs,
        learning_rate=learning_rate,
        use_amp=use_amp,
        gradient_accumulation_steps=gradient_accumulation_steps,
        model_save_path='best_model_state.bin'
    )

    # ==========================
    # 5. DISPLAY RESULTS
    # ==========================
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)

    print("\nTraining history:")
    for epoch in range(len(history['train_loss'])):
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {history['train_loss'][epoch]:.4f}, "
              f"Train Acc: {history['train_acc'][epoch]:.4f}")
        print(f"  Val Loss: {history['val_loss'][epoch]:.4f}, "
              f"Val Acc: {history['val_acc'][epoch]:.4f}")

    best_val_acc = max(history['val_acc'])
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print("Best model saved to: best_model_state.bin")


if __name__ == '__main__':
    main()
