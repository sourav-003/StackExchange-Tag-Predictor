
import gradio as gr
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import pickle
import os # Import os for reading params.txt

# Define the model architecture (if you only saved weights)
# You need to know your vocabulary size and max sequence length here
x_voc_size = 12575 # Replace with your actual vocabulary size from your notebook
max_len = 100 # Replace with your actual max_len from your notebook

# Rebuild the model architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
model = Sequential()
model.add(Embedding(x_voc_size, 50, trainable = True, input_shape=(max_len,),mask_zero=True))
model.add(GRU(128))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='sigmoid'))

# Load the best weights
try:
    model.load_weights("weights.best.keras")
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Please ensure 'weights.best.keras' is in the same directory as app.py")


# Load the tokenizer
try:
    with open('x_tokenizer.pickle', 'rb') as handle:
        x_tokenizer = pickle.load(handle)
except FileNotFoundError:
    print("Error: x_tokenizer.pickle not found. Please ensure it is in the same directory as app.py")
    exit() # Exit if a critical file is missing

# Load the MultiLabelBinarizer
try:
    with open('mlb.pickle', 'rb') as handle:
        mlb = pickle.load(handle)
except FileNotFoundError:
    print("Error: mlb.pickle not found. Please ensure it is in the same directory as app.py")
    exit() # Exit if a critical file is missing

# Load opt and max_len from params.txt
opt = 0.34 # Default value, will try to load from file
max_len = 100 # Default value, will try to load from file
try:
    with open('params.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("opt:"):
                opt = float(line.split(":")[1])
            elif line.startswith("max_len:"):
                max_len = int(line.split(":")[1])
except FileNotFoundError:
    print("Warning: params.txt not found. Using default values for opt and max_len.")
except Exception as e:
    print(f"Warning: Error reading params.txt: {e}. Using default values for opt and max_len.")


# Preprocessing function (same as in the notebook)
def cleaner(text):
  text = BeautifulSoup(text, "html.parser").get_text()
  text = re.sub("[^a-zA-Z]", " ", text)
  text = text.lower()
  tokens = text.split()
  return " ".join(tokens)

# Classification function (same as in the notebook)
def classify(pred_prob,thresh):
  y_pred_seq = []
  for i in pred_prob:
    temp=[]
    for j in i:
      if j>=thresh:
        temp.append(1)
      else:
        temp.append(0)
    y_pred_seq.append(temp)
  return y_pred_seq

# Prediction function (adapted for a single comment input)
def predict_tag(comment):
  #preprocess
  cleaned_text = [cleaner(comment)]

  #convert to integer sequences
  seq = x_tokenizer.texts_to_sequences(cleaned_text)

  #pad the sequence
  pad_seq = pad_sequences(seq,  padding='post', maxlen=max_len)

  #make predictions
  pred_prob = model.predict(pad_seq)
  classes = classify(pred_prob,opt)[0]

  classes = np.array([classes])
  predicted_tags = mlb.inverse_transform(classes)
  return ", ".join(predicted_tags[0]) # Return as a comma-separated string

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_tag,
    inputs=gr.Textbox(lines=5, label="Enter StackExchange Question Text"),
    outputs=gr.Textbox(label="Predicted Tags"),
    title="StackExchange Tag Predictor",
    description="Predict tags for a given StackExchange question using a trained deep learning model."
)

# Launch the interface
# In a Hugging Face Space, this will be handled automatically
# iface.launch() # Uncomment this line to run locally
