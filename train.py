import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from data_handler import load_training_data_one_hot_labelsets
import time
import numpy as np
import pandas as pd

# Determine device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
max_sequence_length = 256
pretrained_model = "distilbert-base-multilingual-cased"
epochs = 15  # 3
learning_rate = 1e-5  # Experiment with different learning rates
batch_size = 32  # 64  # Experiment with different batch sizes

# Define a dataset class for multi-label tasks
class TextDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_length):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.inputs[index]
        label = torch.tensor(self.labels[index].squeeze(), dtype=torch.float32)  # Remove .toarray()
        tokenized_data = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized_data["input_ids"].squeeze(0),
            "attention_mask": tokenized_data["attention_mask"].squeeze(0),
            "labels": label
        }

# Define the classification model
class MultiLabelClassifier(nn.Module):
    def __init__(self, pretrained_model, label_count):
        super(MultiLabelClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(pretrained_model)
        self.fc_layer = nn.Linear(self.base_model.config.hidden_size, label_count)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Take the [CLS] token's representation
        predictions = self.fc_layer(pooled_output)
        
        loss = None
        if labels is not None:
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(predictions, labels)
        
        return {"loss": loss, "logits": predictions}

# Function to compute metrics
def compute_metrics(y_true, y_pred, k=5):
    y_pred_top_k = np.argsort(y_pred, axis=1)[:, -k:]  # Top-k predictions
    y_true_binary = y_true.astype(bool)
    
    precision_list, recall_list, f1_list = [], [], []

    for i in range(y_true.shape[0]):
        y_pred_k_binary = np.zeros_like(y_true[i])
        y_pred_k_binary[y_pred_top_k[i]] = 1  # Set top-k predictions to 1

        precision_list.append(precision_score(y_true[i], y_pred_k_binary, zero_division=0))
        recall_list.append(recall_score(y_true[i], y_pred_k_binary, zero_division=0))
        f1_list.append(f1_score(y_true[i], y_pred_k_binary, zero_division=0))

    return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

# Function to evaluate the model
def evaluate_model(model, val_loader, device, k=5):
    model.eval()
    all_logits, all_labels = [], []
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs, masks, labels)  # Pass labels for loss computation
            logits = outputs["logits"].cpu().numpy()
            loss = outputs["loss"].item()

            all_logits.append(logits)
            all_labels.append(labels.cpu().numpy())
            epoch_loss += loss

    y_pred = np.vstack(all_logits)
    y_true = np.vstack(all_labels)
    precision, recall, f1 = compute_metrics(y_true, y_pred, k)

    return precision, recall, f1, epoch_loss / len(val_loader)

# Function to train the model
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks, labels)
            loss = outputs["loss"]
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Training Loss = {epoch_loss / len(train_loader):.4f}, Time = {time.time() - start_time:.2f}s")

        # Evaluate after each epoch
        precision, recall, f1, eval_loss = evaluate_model(model, val_loader, device, k=5)
        print(f"Epoch {epoch + 1}: Precision@5 = {precision:.4f}, Recall@5 = {recall:.4f}, F1@5 = {f1:.4f}, Eval Loss = {eval_loss:.4f}")

# Load data and preprocess
data, label_names = load_training_data_one_hot_labelsets()
texts = data["input"]
labels = data.drop("input")

x_train, x_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
train_dataset = TextDataset(x_train.to_list(), y_train.to_numpy(), tokenizer, max_sequence_length)
val_dataset = TextDataset(x_val.to_list(), y_val.to_numpy(), tokenizer, max_sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model and optimizer setup
model = MultiLabelClassifier(pretrained_model, len(label_names)).to(device)

# Experiment with different optimizers and learning rates
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Uncomment to try SGD

# Train the model
train_model(model, train_loader, val_loader, optimizer, device, epochs)

'''
Epoch 1: Training Loss = 0.1836, Time = 241.88s
Epoch 1: Precision@5 = 0.0002, Recall@5 = 0.0004, F1@5 = 0.0003, Eval Loss = 0.0411
Epoch 2: Training Loss = 0.0229, Time = 247.32s
Epoch 2: Precision@5 = 0.0002, Recall@5 = 0.0004, F1@5 = 0.0003, Eval Loss = 0.0126
Epoch 3: Training Loss = 0.0086, Time = 244.53s
Epoch 3: Precision@5 = 0.0124, Recall@5 = 0.0256, F1@5 = 0.0151, Eval Loss = 0.0059
Epoch 4: Training Loss = 0.0044, Time = 245.65s
Epoch 4: Precision@5 = 0.0141, Recall@5 = 0.0267, F1@5 = 0.0169, Eval Loss = 0.0033
Epoch 5: Training Loss = 0.0027, Time = 245.53s
Epoch 5: Precision@5 = 0.0139, Recall@5 = 0.0265, F1@5 = 0.0167, Eval Loss = 0.0022
Epoch 6: Training Loss = 0.0019, Time = 245.00s
Epoch 6: Precision@5 = 0.0139, Recall@5 = 0.0265, F1@5 = 0.0167, Eval Loss = 0.0017
Epoch 7: Training Loss = 0.0015, Time = 232.02s
Epoch 7: Precision@5 = 0.0143, Recall@5 = 0.0269, F1@5 = 0.0171, Eval Loss = 0.0014
Epoch 8: Training Loss = 0.0013, Time = 165.31s
Epoch 8: Precision@5 = 0.0137, Recall@5 = 0.0260, F1@5 = 0.0163, Eval Loss = 0.0012
Epoch 9: Training Loss = 0.0012, Time = 165.30s
Epoch 9: Precision@5 = 0.0139, Recall@5 = 0.0265, F1@5 = 0.0167, Eval Loss = 0.0012
Epoch 10: Training Loss = 0.0011, Time = 165.23s
Epoch 10: Precision@5 = 0.0144, Recall@5 = 0.0270, F1@5 = 0.0171, Eval Loss = 0.0011
Epoch 11: Training Loss = 0.0011, Time = 165.39s
Epoch 11: Precision@5 = 0.0152, Recall@5 = 0.0291, F1@5 = 0.0182, Eval Loss = 0.0011
Epoch 12: Training Loss = 0.0011, Time = 165.67s
Epoch 12: Precision@5 = 0.0187, Recall@5 = 0.0348, F1@5 = 0.0223, Eval Loss = 0.0011
Epoch 13: Training Loss = 0.0011, Time = 165.67s
Epoch 13: Precision@5 = 0.0242, Recall@5 = 0.0484, F1@5 = 0.0294, Eval Loss = 0.0011
Epoch 14: Training Loss = 0.0011, Time = 165.72s
Epoch 14: Precision@5 = 0.0307, Recall@5 = 0.0594, F1@5 = 0.0369, Eval Loss = 0.0011
Epoch 15: Training Loss = 0.0011, Time = 165.84s
Epoch 15: Precision@5 = 0.0374, Recall@5 = 0.0713, F1@5 = 0.0449, Eval Loss = 0.0011
'''