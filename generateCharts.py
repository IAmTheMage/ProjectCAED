import matplotlib.pyplot as plt

# Função para ler o arquivo e extrair os dados
def ler_arquivo(nome_arquivo):
    epochs = []
    loss = []
    accuracy = []

    with open(nome_arquivo, 'r') as file:
        linhas = file.readlines()
        for linha in linhas:
            valores = linha.split('|')
            if len(valores) >= 3:
                epoch = int(valores[0].split(':')[1])
                epochs.append(epoch)

                loss_valor = float(valores[1].split(':')[1])
                loss.append(loss_valor)

                acc_valor = float(valores[2].split(':')[1])
                accuracy.append(acc_valor)

    return epochs, loss, accuracy

# Função para plotar o gráfico de Loss
def plot_loss(epochs, loss):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, marker='o', linestyle='-')
    plt.title('Loss por Época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# Função para plotar o gráfico de Acurácia
def plot_accuracy(epochs, accuracy):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracy, marker='o', linestyle='-')
    plt.title('Acurácia por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.grid(True)
    plt.show()

# Nome do arquivo a ser lido
nome_do_arquivo = 'loss.txt'  # Substitua pelo nome do seu arquivo

# Ler o arquivo e extrair os dados
epochs, loss, accuracy = ler_arquivo(nome_do_arquivo)

# Plotar os gráficos separados para Loss e Acurácia
plot_loss(epochs, loss)
plot_accuracy(epochs, accuracy)
