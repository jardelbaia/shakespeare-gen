import tensorflow as tf

# ---------------------------------------------------------------------------------------------------------- #
# Gets unique characters of text (vocab) and vocab length                                                    #
# ---------------------------------------------------------------------------------------------------------- #
def get_vocab(text):
	vocab = sorted(set(text))
	return vocab,len(vocab)

# ---------------------------------------------------------------------------------------------------------- #
# Gets input and targets of a sentence.																		 #
# Knowing n words (input), predict the next word (last target word)	             							 #
# ---------------------------------------------------------------------------------------------------------- #
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# ---------------------------------------------------------------------------------------------------------- #
# Creates Recurrent Neural Net (LSTM) model																	 #
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

# ---------------------------------------------------------------------------------------------------------- #
# Generate new text                   																		 #
# ---------------------------------------------------------------------------------------------------------- #
def generate(model, initial_string,num_characters,index_to_char,char_to_index):
	#start string to ints
	str_num = [char_to_index[s] for s in initial_string]
	str_num = tf.expand_dims(str_num, 0)

	#empty list to store the generated text
	generated_text = []

	#predict
	temp = 1.0
	model.reset_states()
	for i in range(num_characters):
		pred = model(str_num)
		pred = tf.squeeze(pred,0)
		pred = pred/temp
		pred_id = tf.random.categorical(pred, num_samples=1)[-1,0].numpy()
		str_num = tf.expand_dims([pred_id], 0)
		generated_text.append(index_to_char[pred_id])

	return (initial_string + ''.join(generated_text))
