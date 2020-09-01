import tensorflow as tf
import numpy as np
from PreProcessing import *
from GetFun import create_model,loss

embedding_dim = 256
rnn_units = 1024
EPOCHS=30

#creates model
model = create_model(vocab_size,embedding_dim,rnn_units,BATCH_SIZE)
model.compile(optimizer='adam',loss=loss)

#start training
history = model.fit(dataset, epochs=EPOCHS)

#save model
model.save('models/shakespeare_model.h5')
