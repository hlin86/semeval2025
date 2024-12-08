LLM4Subjects

SemEval 2025, Library Tagging Problem, XMLC

Multi-Label Text Classification with DistilBERT

This project implements a multi-label text classification system using DistilBERT. It supports training, evaluating, and fine-tuning on custom datasets. The project uses PyTorch and Hugging Face's Transformers library for efficient computation and state-of-the-art language representation.

Features

Multi-label classification with a pretrained distilbert-base-multilingual-cased model.
Metrics computation (Precision@k, Recall@k, F1@k).
Configurable hyperparameters for sequence length, batch size, learning rate, and number of epochs.
Support for GPUs for faster computation.

Command
Packages Installation

pip install -r requirments.txt

Run Train File

python train.py
