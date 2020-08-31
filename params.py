seq_length = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000
embedding_dim = 256
rnn_units = 1024
EPOCHS=30

# ---------------------------------------------------------------------------------------------------------- #
# Gets unique characters of text (vocab) and vocab length                                                    #
# ---------------------------------------------------------------------------------------------------------- #
def get_vocab(text):
	vocab = sorted(set(text))
	return vocab,len(vocab)
	
