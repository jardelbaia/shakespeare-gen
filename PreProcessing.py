import tensorflow as tf
import numpy as np
from GetFun import get_vocab,split_input_target

tf.enable_eager_execution()
#download and open file
file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(file, 'rb').read().decode(encoding='utf-8')

#get vocab (unique characters)
#creates a dictionary of unique characters (character : index)
#creates an array of unique charatcters
vocab,vocab_size = get_vocab(text)
char_to_index = {u:i for i,u in enumerate(vocab)}
index_to_char = np.array(vocab)

#text to int and slice to get sequences
seq_length = 100
int_text = [char_to_index[char] for char in text]
sliced = tf.data.Dataset.from_tensor_slices(int_text)
sequences = sliced.batch(seq_length+1, drop_remainder=True)

#creates input and targets 
BUFFER_SIZE = 10000
BATCH_SIZE = 64
dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
