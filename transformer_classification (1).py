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
import torchtext
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from model_defs import TextIter, Embeddings, PositionalEncoding, SingleHeadAttention,  MultiHeadAttention, LayerNorm, ResidualConnection, FeedForward
from model_defs import SingleEncoder, EncoderBlocks

MODE='TRAIN'

device = torch.device("cuda")

file_name = 'data.csv'


if MODE == 'TRAIN':
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

class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, N,
                dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.emb = Embeddings(d_model, vocab_size)
        self.pos_encoder = PositionalEncoding(d_model=d_model, vocab_size=vocab_size)

        attn = MultiHeadAttention(nhead, d_model)
        ff = FeedForward(d_model, d_ff, dropout)
        self.transformer_encoder = EncoderBlocks(SingleEncoder(d_model, attn, ff, dropout), N)
        self.classifier = nn.Linear(d_model, 4)
        self.d_model = d_model

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

device = torch.device("cuda")
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
def save_vector_to_file(vector, file_name):
    print("SAVING VECTOR")
    with open(file_name, 'a+') as file:
        for element in vector:
            file.write(str(element) + '\n')


loss_vec = []
accuracy_vec = []
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
        loss_vec.append(total_loss_train)
        accuracy_vec.append(total_acc_train)
        print(f'Epochs: {epoch + 1} | Loss: {total_loss_train / len(train_dataset): .3f} | Accuracy: {total_acc_train / len(train_dataset): .3f}')


def predict(text):
    sequence = torch.tensor([vocab[word] for word in word_tokenize(text, language='portuguese')], dtype=torch.long).unsqueeze(0)
    output = model(sequence.to(device))
    prediction = id2label[output.argmax(axis=1).item()]

    return prediction

if __name__ == '__main__': 
    if MODE == 'TRAIN':
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
    else:
        print("Start load model")
        model.load_state_dict(torch.load("./model.pth"))
        text = "Política. “Impeachment” é um termo de origem inglesa que significa impedimento. Trazido para o âmbito político, estabelece um instrumento pelo qual os regimes liberais traçam a limitação de poderes dos membros do Poder Executivo. Dessa forma, não poderíamos forjar esse tipo de recurso consolidado em governos onde haja um cenário político centralizador, como nos regimes monárquicos, totalitaristas ou ditatoriais. O impeachment foi criado no contexto político britânico medieval, onde observamos um diferenciado processo de consolidação da monarquia. Historicamente, a formação do Estado Nacional Britânico nunca veio de fato a instituir a figura de um rei que exercesse amplos poderes sob a população. Contudo, no caso inglês, o impeachment só era utilizado quando um funcionário ou ministro fazia mal uso de suas prerrogativas políticas. Além disso, o impeachment poderia acarretar em outras punições criminais. Ao longo do tempo, o impeachment britânico veio a cair em desuso e foi posteriormente substituído por outros instrumentos jurídicos. Tal transformação se deu principalmente porque as agitações políticas causadas por esse tipo de processo geravam um enorme desgaste. Dessa maneira, os britânicos resolveram substituí-lo pelo voto de censura. Nesse novo modelo, o parlamento realizava uma votação que decidia se determinado membro do Executivo era digno ou não de sua confiança. Quando algum integrante do Poder Executivo chegava a ser punido, isso indicava o desejo do parlamento em promover a substituição do acusado. Consequentemente eram realizadas novas eleições, nas quais a população viria a escolher um substituto capaz para assumir a função desocupada. Com esse novo artifício, o governo parlamentar britânico criou uma alternativa que transformava o seu impeachment em um processo bem menos impactante. Nos Estados Unidos da América, esse tipo de uso do impeachment viria a ganhar novos contornos. Geralmente, o representante político que fosse vítima de um não deveria sofrer nenhum tipo de sanção criminal. O acusado somente perdia o direito de continuar a exercer as funções atribuídas pelo seu cargo político. Apesar de nunca ter sido efetivamente utilizado contra um presidente norte-americano, alguns casos chegaram bem perto disso. No mandato de Richard Nixon (1969 – 1974), investigações comprovaram as ações de espionagem de integrantes de seu governo contra membros do partido democrata. Com isso, o Congresso Norte-Americano organizou um processo de impeachment contra Nixon. Contudo, antes disso, o próprio presidente decidiu renunciar ao cargo. Décadas mais tarde, o presidente Bill Clinton sofreu um processo de impeachment devido a um escândalo sexual. No entanto, o Senado não reconheceu a validade do processo. Em terras brasileiras esse artifício político foi utilizado contra o presidente Fernando Collor de Melo e, em certa medida, marcou a consolidação da democracia no país. Após a formação de uma Comissão Parlamentar de Inquérito (CPI), membros do Poder Legislativo comprovaram várias denúncias de corrupção contra a presidente. Com isso, mediante a votação da Câmara dos Deputados e a aprovação do Senado, Fernando Collor foi destituído do cargo e perdeu seus direitos políticos durante oito anos."
        gt = "Ensino Fundamental 2"
        prediction = predict(text)

        print(f'Text: {text}')
        print(f'Ground Truth: {gt}')
        print(f'Prediction: {prediction}')

