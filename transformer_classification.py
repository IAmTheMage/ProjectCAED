# -*- coding: utf-8 -*-
"""transformer_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cdFFXinVpSfioKfmPzdt4PHixZePhHH_

# Load Libraries, Read Data, and Label Mapping
"""

import math
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator

from model_defs import TextIter, Embeddings, PositionalEncoding, SingleHeadAttention,  MultiHeadAttention, LayerNorm, ResidualConnection, FeedForward
from model_defs import SingleEncoder, EncoderBlocks, TransformerEncoderModel

MODE='TRAIN'

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

file_name = 'data.csv'


df = pd.read_csv(file_name, encoding='utf-8')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train['Category'].astype(str)
df_train['Message'].astype(str)
df_test['Category'].astype(str)
df_test['Message'].astype(str)
#print(df_train.head())

labels = df_train["Category"].unique()
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label


#print(id2label)
#print(label2id)

"""# Build Vocabulary"""

# Load tokenizer
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
# Initialize training data iterator


# Build vocabulary
def yield_tokens(data_iter):
    for text in data_iter:
        yield word_tokenize(str(text), language='portuguese')

data_iter = TextIter(df_train)
vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])
#print(vocab.get_stoi())

#text_unk = 'this is jkjkj' # jkjkj is an unknown vocab
#seq_unk = [vocab[word] for word in word_tokenize(str(text_unk), language='portuguese')]

#print(tokenizer(text_unk))
#print(seq_unk)

# We will use this example throughout the article
text = "Neste wikilivro, será escrito simplesmente formula_5 para denotar formula_6"
seq = [vocab[word] for word in word_tokenize(str(text), language='portuguese')]

#print(seq)

"""# Word Embedding"""



hidden_size = 4

input_data = torch.LongTensor(seq).unsqueeze(0)
emb_model = Embeddings(hidden_size, len(vocab))
token_emb = emb_model(input_data)
print(f'Size of token embedding: {token_emb.size()}')

"""# Positional Encoding"""



pe_model = PositionalEncoding(d_model=4, vocab_size=len(vocab))
output_pe = pe_model(token_emb)

print(f'Size of output embedding: {output_pe.size()}')

"""# Self-Attention"""




mult_att = MultiHeadAttention(h=4, d_model=4)
output_mult_att = mult_att(output_pe)

#print(f'Size of output embedding after multi-head attention: {output_mult_att.size()}')

"""# Residual Connection"""



res_conn_1 = ResidualConnection(d_model=4)
output_res_conn_1 = res_conn_1(output_pe, output_mult_att)

#print(f'Size of output embedding after residual connection: {output_res_conn_1.size()}')

"""# Feed-Forward"""



ff = FeedForward(d_model=4, d_ff=12)
output_ff = ff(output_res_conn_1)

#print(f'Size of output embedding after feed-forward network: {output_ff.size()}')

res_conn_2 = ResidualConnection(d_model=4)
output_res_conn_2 = res_conn_2(output_res_conn_1, output_ff)

#print(f'Size of output embedding after second residual: {output_res_conn_2.size()}')

"""# Encoder Stack"""



"""# Transformer Encoder Model"""


device = torch.device("cuda" if use_cuda else "cpu")
model = TransformerEncoderModel(len(vocab), d_model=300, nhead=4, d_ff=50,
                                    N=6, dropout=0.1).to(device)

"""# Dataloader"""

class TextDataset(torch.utils.data.Dataset):

  def __init__(self, input_data):
      self.text = input_data['Message'].values.tolist()
      self.label = [int(label2id[i]) for i in input_data['Category'].values.tolist()]

  def __len__(self):
      return len(self.label)

  def get_sequence_token(self, idx):
      sequence = [vocab[word] for word in word_tokenize(str(self.text[idx]), language='portuguese')]
      len_seq = len(sequence)
      return sequence, len_seq

  def get_labels(self, idx):
      return self.label[idx]

  def __getitem__(self, idx):
      sequence, len_seq = self.get_sequence_token(idx)
      label = self.get_labels(idx)
      return sequence, label, len_seq

def collate_fn(batch):

    sequences, labels, lengths = zip(*batch)
    max_len = max(lengths)

    for i in range(len(batch)):
        if len(sequences[i]) != max_len:
          for j in range(len(sequences[i]),max_len):
            sequences[i].append(0)

    return torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

"""# Model Training"""

def train(model, dataset, epochs, lr, bs):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters()
    if p.requires_grad), lr=lr)
    train_dataset = TextDataset(dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=bs, collate_fn=collate_fn, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        total_loss_train = 0
        total_acc_train = 0
        for train_sequence, train_label in tqdm(train_dataloader):

            # Model prediction
            predictions = model(train_sequence.to(device))
            labels = train_label.to(device)
            loss = criterion(predictions, labels)

            # Calculate accuracy and loss per batch
            correct = (predictions.argmax(axis=1) == labels)
            acc = correct.sum().item() / correct.size(0)
            total_acc_train += correct.sum().item()
            total_loss_train += loss.item()

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        print(f'Epochs: {epoch + 1} | Loss: {total_loss_train / len(train_dataset): .3f} | Accuracy: {total_acc_train / len(train_dataset): .3f}')


def predict(text, model):
    sequence = torch.tensor([vocab[word] for word in word_tokenize(text, language='portuguese')], dtype=torch.long).unsqueeze(0)
    output = model(sequence.to(device))
    prediction = id2label[output.argmax(axis=1).item()]

    return prediction

if __name__ == '__main__': 
    epochs = 30
    lr = 1e-4
    batch_size = 4
    train(model, df_train, epochs, lr, batch_size)
    path = 'model.pth'

    # Salvar o estado do modelo
    torch.save(model.state_dict(), path)

    """# Model Prediction"""



    idx = 24
    text = df_test['Message'].values.tolist()[idx]
    gt = df_test['Category'].values.tolist()[idx]
    prediction = predict(text)

    print(f'Text: {text}')
    print(f'Ground Truth: {gt}')
    print(f'Prediction: {prediction}')

    idx = 35
    text = df_test['Message'].values.tolist()[idx]
    gt = df_test['Category'].values.tolist()[idx]
    prediction = predict(text)

    print(f'Text: {text}')
    print(f'Ground Truth: {gt}')
    print(f'Prediction: {prediction}')

