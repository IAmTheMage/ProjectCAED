import heuristic
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from model_defs import TextIter, Embeddings, PositionalEncoding, SingleHeadAttention,  MultiHeadAttention, LayerNorm, ResidualConnection, FeedForward
from model_defs import SingleEncoder, EncoderBlocks, TransformerEncoderModel


from torch.optim import Adam
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
from bert_defs import Dataset, BertClassifier

from nltk.tokenize import word_tokenize

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
# Initialize training data iterator

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


# Build vocabulary
def yield_tokens(data_iter):
    for text in data_iter:
        yield word_tokenize(str(text), language='portuguese')

data_iter = TextIter(df_train)
vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

def predict(text, model):
    sequence = torch.tensor([vocab[word] for word in word_tokenize(text, language='portuguese')], dtype=torch.long).unsqueeze(0)
    output = model(sequence.to(device))
    prediction = id2label[output.argmax(axis=1).item()]

    return prediction

device = torch.device("cuda" if use_cuda else "cpu")
model = TransformerEncoderModel(len(vocab), d_model=300, nhead=4, d_ff=50,
                                    N=6, dropout=0.1).to(device)

if __name__ == "__main__":
    while True:
        mode = int(input("Selecione a ação: 1(Heuristica de inversão), 2(Modelo de Transformadores, inferência), 3(BERT, inferência), Qualquer outro valor(SAIR)\n"))
        if mode == 1:
            sentence = str(input("Digite a frase a ser invertida: "))
            dependency = heuristic.Dependency(sentence)
            sentence_changed = dependency.mount()
            print(f"Frase invertida: {sentence_changed}") 
        elif mode == 2:
            print("Start load model")
            model.load_state_dict(torch.load("./model.pth"))
            text = str(input("Digite a frase a ser classificada: "))
            gt = str(input("Digite como você classificaria a frase de acordo com os 4 níveis, 1 para ensino fundamental 1, 2 para ensino fundamental 2...: "))
            prediction = predict(text, model)

            print(f'Text: {text}')
            print(f'Ground Truth: {gt}')
            print(f'Prediction: {prediction}')

        elif mode == 3:
            labels = ['Ensino Fundamental 1',
                'Ensino Fundamental 2',
                'Ensino Médio',
                'Ensino Superior',
            ]
            model = BertClassifier()
            model.load_state_dict(torch.load("./model_bert.pth"))
            text = str(input("Digite o texto para realizar a previsão: "))
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            model.eval()
            inputs = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            with torch.no_grad():
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                # Passar os inputs pelo modelo
                output = model(input_ids, attention_mask)
                predictions = torch.argmax(output, dim=1)
                print(f"Previsão: {labels[predictions.item()]}")