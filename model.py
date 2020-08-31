import tensorflow as tf

# ---------------------------------------------------------------------------------------------------------- #
# Gets input and targets of a sentence.																		 #
# Knowing n words (input), predict the next word (last target word)	             							 #
# ---------------------------------------------------------------------------------------------------------- #
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# ---------------------------------------------------------------------------------------------------------- #
# Creates Recurrent Neural (LSTM) Net model																	 #
# ---------------------------------------------------------------------------------------------------------- #
def create_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

# ---------------------------------------------------------------------------------------------------------- #
# Loss function                       																		 #
# ---------------------------------------------------------------------------------------------------------- #
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

