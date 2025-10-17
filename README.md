
# PyTorch LSTM Next Word Predictor

This project implements a **Long Short-Term Memory (LSTM)** neural network using PyTorch to perform a **next word prediction** task based on a custom text corpus.

The model is trained on a small, domain-specific text document (a Q&A/FAQ about a Data Science Mentorship Program) to learn word sequences and predict the word most likely to follow a given input sequence.

***

## ✨ Key Features

* **Data Preprocessing:** Uses `nltk` for tokenization and builds a custom vocabulary including an `<unk>` token.
* **Sequence Generation:** Creates all possible next-word prediction pairs from the corpus.
* **Sequence Padding:** Applies **pre-padding** (using `0`) to ensure all input sequences have a uniform length of **61**.
* **PyTorch Implementation:** Defines a standard LSTM model, uses a `CustomDataset` and `DataLoader` for the training pipeline.
* **Evaluation:** Calculates prediction accuracy on the training set.

***

## ⚙️ Model Architecture

The `LSTMModel` is structured as follows:

| Layer | Input Size | Output Size | Notes |
| :--- | :--- | :--- | :--- |
| **Embedding** | `vocab_size` (289) | 100 | Maps word indices to dense vectors. |
| **LSTM** | 100 | 150 (hidden size) | Processes the sequence data. Uses `batch_first=True`. |
| **Linear (FC)** | 150 | `vocab_size` (289) | Predicts the probability distribution over the entire vocabulary. |

The model uses the **final hidden state** output from the LSTM layer for the classification task.

### Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Vocabulary Size** | 289 (includes `<unk>` and padding index `0`) |
| **Max Sequence Length** | 62 (Input **X** is length 61, Target **y** is the 62nd word) |
| **Epochs** | 50 |
| **Learning Rate** | 0.001 |
| **Optimizer** | Adam |
| **Loss Function** | CrossEntropyLoss |

***

## 🚀 Getting Started

### Prerequisites

You need Python and the following libraries:

```bash
pip install torch numpy nltk
