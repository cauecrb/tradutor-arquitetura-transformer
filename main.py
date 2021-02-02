#tradutor usando transformer
#Cauê Rafael Burgardt

#importaçoes
import numpy as np
import math
import  re
import time
import zipfile
import random
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_text as text


'''from google.colab import drive
%tensroflow_version 2.x'''

# pré processamento de dados

'''#extraindo arquivo
path = '/basededados/pt-en.tgz'
tgz_object = zipfile.ZipFile(file = path, mode = 'r')
tgz_object.extractall('./')
tgz_object.close()'''

#lendo os arquivos
with open('basededados/pt-en/europarl-v7.pt-en.en', mode ='r', encoding='utf-8') as f:
    europarl_en = f.read()
with open('basededados/pt-en/europarl-v7.pt-en.pt', mode ='r', encoding='utf-8') as f:
    europarl_pt = f.read()

print(europarl_en[0:100])
en = europarl_en.split('\n')
print(len(en))

pt = europarl_pt.split('\n')
print(len(pt))

for _ in range(5):
    print('-----------')
    i = random.randint(0, len(en) - 1)
    print(en[i])
    print(pt[i])

#limpeza dos dados

corpo_en = europarl_en
corpo_en = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpo_en)
corpo_en = re.sub(r".\$\$\$", '', corpo_en)
corpo_en = re.sub(r" +", " ", corpo_en)
corpo_en = corpo_en.split('\n')

corpo_pt = europarl_pt
corpo_pt = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpo_pt)
corpo_pt = re.sub(r".\$\$\$", '', corpo_pt)
corpo_pt = re.sub(r" +", " ", corpo_pt)
corpo_pt = corpo_pt.split('\n')

print(len(corpo_pt))
print(len(corpo_en))

#tokenização
'''
é um dicionario que transforma cada palavra
em um numero
'''
tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpo_en, target_vocab_size=2**13)
print(tokenizer_en.vocab_size)
tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpo_pt, target_vocab_size=2**13)
print(tokenizer_pt.vocab_size)
'''tok = text.Tokenizer.tokenize(corpo_pt)
print(tok)
tokenizer = text.WhitespaceTokenizer()
tokenizer_en = tokenizer.tokenize(corpo_en.encode('UTF-8'))
print(tokenizer_en.vocab_size)'''

vocab_size_en = tokenizer_en.vocab_size + 2
vocab_size_pt = tokenizer_pt.vocab_size + 2

inputs = [[vocab_size_en - 2] + tokenizer_en.encode(sentence) + [vocab_size_en -1 ] for sentence in corpo_en]

for _ in range(5):
    print(inputs[random.randint(0, len(inputs) -1 )])

outputs = [[vocab_size_pt - 2] + tokenizer_pt.encode(sentence) + [vocab_size_pt -1 ] for sentence in corpo_pt]

for _ in range(5):
    print(outputs[random.randint(0, len(outputs) -1 )])

max_length = 60

'''#remoção de sentençãs muito longas, para diminuir o tempo de treinamento

idx_to_remove = [count for count, sent in enumerate(inputs) if len(sent) > max_length]

print(len(idx_to_remove))

for idx in reversed(idx_to_remove):
    del inputs[idx]
    del outputs[idx]

idx_to_remove = [count for count, sent in enumerate(outputs) if len(sent) > max_length]

for idx in reversed(idx_to_remove):
    del inputs[idx]
    del outputs[idx]

print(len(inputs))
print(len(outputs))
'''

#preenxendo as frases com 0 para deixar todas do mesmo tamanho
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, value=0, padding='post', maxlen=max_length)
outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs, value=0, padding='post', maxlen=max_length)

for _ in range(5):
    print(outputs[random.randint(0, len(outputs) -1 )])

#quebrando a base de dados em partes menoers e
#transformando em formato dataset do tf

batch_size= 64
buffer_size = 20000

dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
dataset = dataset.cache()
dataset = dataset.shuffle(buffer_size).batch(batch_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(dataset)

#----- Modelo -----

#embedding
class PositionalEncoding(layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__
        
    def get_angles(selfself, pos, i, d_model):
        angles = 1 / np.power(10000.,(2*(i // 2)) / np.float32(d_model))
        return pos * angles # (seq_length, d_model)

    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model[np.newaxis, :], d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return inputs + tf.cast(pos_encoding, tf.float32)

# mecanismo de atenção
def scaled_dot_product_attention(queries, keys, values, mask):
    product = tf.matmul(queries, keys, transpose_b=True)
    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(keys_dim)

    if mask is not None:
        scaled_product += (mask * -1e9) # 0.0000000001

    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)
    return attention
print('criado mecanismo de atenção')

#mult-head attention
class MultiHeadAttention(layers.Layer):
    def __init__(self, nb_proj):
        super(MultiHeadAttention, self).__init__()
        self.nb_proj = nb_proj

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.nb_proj == 0

        self.d_proj = self.d_model // self.nb_proj

        self.query_lin = layers.Dense(units= self.d_model)
        self.key_lin = layers.Dense(units=self.d_model)
        self.value_lin = layers.Dense(units=self.d_model)

        self.final_lin = layers.Dense(units=self.d_model)

    def split_proj(self, inputs, batch_size): #inputs: (batch_size, seq_length, d_model
        shape = (batch_size, -1, self.nb_proj, self.d_proj)
        splitted_inputs = tf.reshape(inputs, shape = shape) # (batch_size, seq_length, nb_proj, d_proj

        return tf.transpose(splitted_inputs, perm=[0, 2, 1, 3])#(batch_size, nb_proj, seq_length, d_proj)

    def call(self, queries, keys, values, mask):
        batch_size = tf.shape(queries)[0]

        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)

        queries = self.split_proj(queries, batch_size)
        keys = self.split_proj(keys, batch_size)
        values = self.split_proj(values, batch_size)

        attention = scaled_dot_product_attention(queries, keys, values, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention, shape=(batch_size, -1, self.d_model))

        outputs = self.final_lin(concat_attention)
        return outputs

    print('criado camada de mult-head attention')

#encoder
class EncoderLayer(layers.Layer):
    def __init__(self, FFN_units, nb_proj, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        self.multi_head_attention = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.dense_1 = layers.Dense(units=self.FFN_units, activation='relu')
        self.dense_2 = layers.Dense(units=self.FFN_units, activation='relu')
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)

        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask, training):
        attention = self.multi_head_attention(inputs, inputs, inputs, mask)
        attention = self.dropout_1(attention, training = training)
        attention = self.norm_1(attention + inputs)

        outputs = self.dense_1(attention)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_2(outputs, training = training)
        outputs = self.norm_2(outputs + attention)

        return outputs

    print("criada o encoder")

class Encoder(layers.Layer):
    def __init__(self,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.nb_layers = nb_layers
        self.d_model = d_model
        
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.enc_layer = [EncoderLayer(FFN_units,nb_proj, dropout_rate) for _ in range(nb_layers) ]

    def call(self, inputs, mask, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.nb_layers):
            outputs = self.enc_layer[i](outputs, mask, training)

        return outputs
print("criado encoder 2")

#decoder 1:  segundo o diagrama da arquietura transformer
#nececitamos duas camadas de multihead attention

class DecoderLayer(layers.Layer):
    def __init__(self, FFN_units, nb_proj, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        self.multi_head_attention_1 = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.multi_head_attention_2 = MultiHeadAttention(self.nb_proj)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

        self.dense_1 = layers.Dense(units=self.FFN_units, activation='relu')
        self.dense_2 = layers.Dense(units=self.FFN_units, activation='relu')
        self.dropout_3 = layers.Dropout(rate=self.dropout_rate)
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)


    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        attention = self.multi_head_attention_1(inputs, inputs, inputs, mask_1)
        attention = self.dropout_1(attention, training)
        attention = self.norm_1(attention + inputs)

        attention_2 = self.multi_head_attention_2(attention, enc_outputs, enc_outputs, mask_2)
        attention_2 = self.dropout_2(attention_2, training)
        attention_2 = self.norm_2(attention_2 + attention)

        outputs = self.dense_1(attention_2)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_3(outputs, training)
        outputs = self.norm_3(outputs + attention_2)

        return outputs

print("criação decoder 1")

class Decoder(layers.Layer):
    def __init__(self,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.nb_layers = nb_layers

        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)

        self.dec_layers = [DecoderLayer(FFN_units, nb_proj, dropout_rate) for i in range(nb_layers)]

    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.nb_layers):
            outputs = self.dec_layers[i](outputs, enc_outputs, mask_1, mask_2, training)

        return outputs

print("decoder 2")

#juntando o mdelo transformer

class Transformer(tf.keras.Model):
    def __init__(self,
                 vocab_size_enc,
                 vocab_size_dec,
                 d_model,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 name="transformer"):
        super(Transformer, self).__init__(name=name)

        self.encoder = Encoder(nb_layers, FFN_units, nb_proj,
                               dropout_rate,vocab_size_enc, d_model)
        self.decoder= Decoder(nb_layers, FFN_units, nb_proj,
                              dropout_rate,vocab_size_dec, d_model)
        self.last_linear = layers.Dense(units=vocab_size_dec, name='lin_output')

    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len,seq_len)), -1, 0)
        return look_ahead_mask

    def call(self, enc_inputs, dec_inputs, training):