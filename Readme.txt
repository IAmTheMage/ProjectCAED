Instruções de como prosseguir para rodar o código:

Links Úteis:

Link contendo um zip que tem o arquivo do dataset em formato csv, o dicionário, o modelo básico de transformadores e o modelo bert 
ambos já pré treinados(modelo bert pré treinado utilizado o bert-base-cased): https://drive.google.com/file/d/1FVi01bSysS-E6u7o1NgQxGrPpNqDbpTC/view?usp=sharing


Link para o dicionário utilizado por uma das heurísticas: https://drive.google.com/file/d/1a1z3V4fwZFeZcNfuLRPzk4lJBXCGnETP/view, 
não é obrigatório ter ele disponível pois ele será gerado automaticamente pelo código caso não exista, porém ele estar no mesmo diretório acelera 
muito o tempo de carregamento dos utilitários necessários

Link para o arquivo contento o dataset modificado e combinado em um único arquivo CSV, utilizado principalmente em treinamento:  https://drive.google.com/file/d/1OE62C1mCvgk0IZJUqzrSDu5EUu89YWfD/view?usp=sharing, 
necessário para executar o treinamento do modelo

Arquivo pré treinado do modelo básico de transformadores: https://drive.google.com/file/d/1NjcuXOJOyMcAfQdf6m6fFK_hDkZ7BdTb/view?usp=sharing
Arquivo pré treinado do modelo BERT: https://drive.google.com/file/d/1sjIA-NHxtVAHfrHnEjoicX3CQagcb6zC/view?usp=sharing

repositório do Github: https://github.com/IAmTheMage/ProjectCAED

OBS: Os modelos.pth estão com CUDA ativado, e apenas funcionarão nele pois são resultados do treinamento com CUDA, 
caso não esteja presente o CUDA no ambiente utilizado deve-se utilizar o Google Colab

Arquivos de Treinamento: 
executar python bert_model.py para executar o treinamento do bert-base-case
executar python transformer_classification.py para executar o treinamento do modelo de transformadores simples

Arquivo de inferência e heuristica:

extrair o zip ou colocar cada arquivo individualmente na pasta referênte ao código antes da execução da inferência.

Executar python main.py para testar o código