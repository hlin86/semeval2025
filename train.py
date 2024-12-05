import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np

# Load the GND dataset and TIBKAT CSV files using the data_handler and label_metadata modules
from data_handler import load_dev_data_one_hot, load_training_data_one_hot_labelsets
from label_metadata import generateLabelMetadata

# Load data and generate unique labels
def load_gnd_data(gnd_path):
    with open(gnd_path, 'r', encoding="utf-8") as f:
        return json.load(f)

def load_tibkat_data():
    tibkat_train_data, unique_labels = load_training_data_one_hot_labelsets()
    tibkat_dev_data = load_dev_data_one_hot()
    return tibkat_train_data, tibkat_dev_data, unique_labels

# Prepare training and validation data
def prepare_data_for_training():
    tibkat_train_data, tibkat_dev_data, unique_labels = load_tibkat_data()
    subject_metadata = generateLabelMetadata(unique_labels)

    # Tokenizer using AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Convert the data to a suitable format for the Trainer API
    def tokenize_function(examples):
        return tokenizer(examples["input"], padding="max_length", truncation=True, max_length=512)

    train_dataset = Dataset.from_pandas(tibkat_train_data.to_pandas())
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    dev_dataset = Dataset.from_pandas(tibkat_dev_data)
    dev_dataset = dev_dataset.map(tokenize_function, batched=True)

    return train_dataset, dev_dataset, unique_labels, subject_metadata

# Calculate top-k accuracy
def calculate_top_k_accuracy(predictions, labels, k=5):
    top_k_preds = torch.topk(predictions, k, dim=1).indices  # Get the top-k predictions
    correct = 0
    for i in range(len(labels)):
        if labels[i] in top_k_preds[i]:
            correct += 1
    return correct / len(labels)

# Calculate precision, recall, and F1 score at k
def calculate_precision_recall_f1_at_k(predictions, labels, k=5):
    top_k_preds = torch.topk(predictions, k, dim=1).indices
    true_positives = 0
    relevant = 0
    predicted = 0
    for i in range(len(labels)):
        if labels[i] in top_k_preds[i]:
            true_positives += 1
        if labels[i] != -1:  # Count relevant if the label is not 'none'
            relevant += 1
        predicted += 1
    precision = true_positives / predicted if predicted != 0 else 0
    recall = true_positives / relevant if relevant != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return precision, recall, f1

# Calculate mean average precision (mAP)
def calculate_map(predictions, labels, k=5):
    avg_precision = 0
    for i in range(len(labels)):
        top_k_preds = torch.topk(predictions[i], k).indices
        relevant_labels = [label for label in labels[i] if label != -1]
        if not relevant_labels:
            continue
        precision_at_k = len(set(top_k_preds) & set(relevant_labels)) / k
        avg_precision += precision_at_k
    return avg_precision / len(labels)

# Train the model using the Trainer API
def train_model():
    # Ensure we are using GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, dev_dataset, unique_labels, subject_metadata = prepare_data_for_training()

    # Load model using AutoModelForSequenceClassification (for multi-label classification)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=len(unique_labels), problem_type="multi_label_classification"
    )

    # Move model to the selected device (GPU/CPU)
    model.to(device)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
    )

    # Train the model
    trainer.train()

# Evaluate the model on the test set
def evaluate_model(model, dev_dataset, k=5):
    # Ensure the model is on the correct device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    trainer = Trainer(
        model=model,
        tokenizer=AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
    )
    
    # Make predictions on the dev dataset
    predictions, labels, metrics = trainer.predict(dev_dataset)
    
    # Move predictions and labels to device if necessary
    predictions = torch.tensor(predictions).to(device)
    labels = torch.tensor(labels).to(device)
    
    # Calculate top-k accuracy
    top_k_accuracy = calculate_top_k_accuracy(predictions, labels, k)
    
    # Calculate precision, recall, f1 score at k
    precision, recall, f1 = calculate_precision_recall_f1_at_k(predictions, labels, k)
    
    # Calculate mean average precision (mAP)
    map_value = calculate_map(predictions, labels, k)
    
    # Print out the results
    print(f"Mean Top-{k} Accuracy: {top_k_accuracy:.4f}")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"F1@{k}: {f1:.4f}")
    print(f"Mean Average Precision (mAP): {map_value:.4f}")
    
    return top_k_accuracy, precision, recall, f1, map_value

# Running the training and evaluation
if __name__ == "__main__":
    train_model()
    
    # Assuming that the model and dev dataset are saved after training
    model = AutoModelForSequenceClassification.from_pretrained("./results")
    dev_dataset = load_tibkat_data()[1]  # Reload dev dataset
    evaluate_model(model, dev_dataset, k=5)
