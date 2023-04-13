#importing the libraries
import tensorflow  as tf
import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import joblib
import random
import gradio as gr

res = json.load(open('covid_responses.json'))

file = pd.read_csv('data.csv')
data = file[['inputs', 'tags']]
data = data.sample(frac=1)


file = pd.read_csv('data.csv')
data = file[['inputs', 'tags']]
data = data.sample(frac=1)


import string
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

#tokenize the data
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
#apply padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)

#encoding the outputs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])
input_shape = x_train.shape[1]


vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

def create_model():
    i = Input(shape=(18,))
    x = Embedding(788+1,10)(i)
    x = LSTM(10,return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(output_length,activation="softmax")(x)
    model  = Model(i,x)
    model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
    
    return model


model = tf.keras.models.load_model('saved_model/my_model')




def model_output(prediction_input):
  import random
  #removing punctuation and converting to lowercase
  prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
  prediction_input = ''.join(prediction_input)
  texts_p = []
  texts_p.append(prediction_input)

  #tokenizing and padding
  prediction_input = tokenizer.texts_to_sequences(texts_p)
  prediction_input = np.array(prediction_input).reshape(-1)
  prediction_input = pad_sequences([prediction_input],18)

  #getting output from model
  output = model.predict(prediction_input)
  output = output.argmax()

  #finding the right tag and predicting
  response_tag = le.inverse_transform([output])[0]
  final_response = str(random.choice(res[response_tag]))

  return final_response


dialog_app = gr.Interface(model_output, 
                        gr.Textbox(placeholder="Enter your question"), 
                        "text",
                        examples=[["What is COVID-19?"],[ "What are the symptoms?"], ["What are the precautions?"], ["What is the treatment?"]],
                        title="COVID-19 Chatbot",
                        description="Ask your questions about COVID-19",
)
dialog_app.launch()