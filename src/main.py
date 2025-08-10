
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load handwritten digit nn model
model = tf.saved_model.load('./h.model')

# Load sentiment analysis nn model
sa_model = load_model('./static/my_model.keras')

# Load fashion mnist nn model
mn_model = load_model('./static/mnist_model.keras')

# Load the dataset
path = "./static/IMDB Dataset.csv"
df = pd.read_csv(path)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route('/')
def index_view():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/text')
def text():
    return render_template('text.html')

@app.route('/clothes')
def clothes():
    return render_template('clothes.html')
        
@app.route('/classify',methods=['GET','POST'])
def predict_view():
    if request.method == 'POST':
        #grid = request.data.decode('UTF-8')
        if request.is_json:
            grid = request.get_json()
            prediction = model(grid['theGrid'])
            print(prediction)
            class_prediction = str(np.argmax(prediction))
            return {'prob': class_prediction}
        else:
            return "Unable to read the file. Please check file extension"
        
@app.route('/image',methods=['GET','POST'])
def clothes_view():
    if request.method == 'POST':
        if request.is_json:
            grid = request.get_json()
            
            image_path = f'./static/images/{grid}.png'
            img = load_img(image_path)

            img_array = img_to_array(img)
            img_array = 255 - img_array # Changes background 255 to 0, to match the mnist training set
            img_array = np.array(img_array)
            
            img_array = img_array / 255.0 # normalize
            img_array = tf.image.rgb_to_grayscale(img_array) # will return shape (28, 28, 1)
            img_array = tf.squeeze(img_array, axis = -1) # shape is (28, 28)
            img_array = tf.expand_dims(img_array, axis = 0) 
            
            predictions = mn_model.predict(img_array)
            return {'prob_im': str(class_names[np.argmax(predictions[0])])}
        else:
            return "Unable to read the file. Please check file extension"


# Util
# Function to clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    return text.lower().strip()

@app.route('/sentiment', methods=['GET','POST'])
def sentiment_view():
    if request.method == 'POST':
        grid = request.get_json()
        if request.is_json:
            # Clean the reviews
            df['review'] = df['review'].apply(clean_text)

            # Tokenization and padding
            tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
            tokenizer.fit_on_texts(df['review'])
            sequences = tokenizer.texts_to_sequences(df['review']) 
            padded_sequences = pad_sequences(sequences, maxlen=200)
            # Convert sentiment labels to binary
            df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

            # Splitting the data into features (X) and labels (y)
            X = padded_sequences
            y = df['sentiment'].values

            sample_sequences = tokenizer.texts_to_sequences([grid['text']])
            sample_padded = pad_sequences(sample_sequences, maxlen=200)

            predictions = sa_model.predict(sample_padded)
            return {'prob_sa': "Positive" if predictions[0][0] > 0.5 else "Negative"}

if __name__ == '__main__':
    app.run(debug=True)