from tensorflow import keras
import tensorflow as tf
from PreProcessing import index_to_char,char_to_index,vocab_size
from GetFun import generate,create_model

model = create_model(vocab_size,256,1024, batch_size=1)
model.load_weights("models/shakespeare_model.h5")
model.build(tf.TensorShape([1, None]))

with open("output/output1.txt", "w") as f:
	f.write(generate(model,u"ROMEO: ",2000,index_to_char,char_to_index))


