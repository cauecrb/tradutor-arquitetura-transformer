# Tradutor, arquitetura transformer
#ATENÇÃO: este código pode demorar muito tempo para concluir o preprocessamento e treinamento da rede.

Código de rede neural seguindo a arquitetura transformer para traduzir testos de inglês para português

## Setup

são necessarias as bibliotecas do tensorflow instaladas:

```bash
pip install TensorFlow
pip install tensorflow_datasets
pip install tensorflow_text
```

## uso

Com a data base de frases em inglês e português dentro da pasta basededados/pt-en/ deve-se ter 2 arquivos europarl-v7.pt-en.pt e europarl-v7.pt-en.en, 
executa-se o código para que os dados sejam pre processados, e a rede neural treinada, apos o treino, passa-se um texto em inglês para a função translate() e a rede traduzira
conforme aprendeu.
as basses de dados podem ser baixadas em: https://www.statmt.org/europarl/

Cauê Rafael Burgardt

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Translator, transformer architecture
# ATTENTION: this code can take a long time to complete the preprocessing and training of the network.

Neural network code following the transformer architecture to translate texts from English to Portuguese

## Setup

installed tensorflow libraries are required:

`` bash
pip install TensorFlow
pip install tensorflow_datasets
pip install tensorflow_text
``

## usage

With the base date of phrases in English and Portuguese within the folder basededados/pt-en/ you must have 2 files europarl-v7.pt-en.pt and europarl-v7.pt-en.en,
the code is executed so that the data is pre-processed, and the trained neural network, after training, a text in English is passed to the translate () function and the translated network
as you learned.
databases can be downloaded at: https://www.statmt.org/europarl/

Cauê Rafael Burgardt
