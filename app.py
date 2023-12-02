import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Convert all the words to lowercase
    tokens = [word.lower() for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stem the words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    lst = []
    lst.append(preprocessed_text)
    return lst

app = Flask(__name__) # Initialize the flask App

model = pickle.load(open('myModel.pkl', 'rb')) # Load the trained model
vactorizer = pickle.load(open('vactorizer.pkl', 'rb'))


@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the request
    text = request.form['text']

    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Vectorize the preprocessed text
    vectorized_text = vactorizer.transform(preprocessed_text).toarray()

    # Make a prediction
    prediction = model.predict_proba(vectorized_text)
    msg = None
    if(prediction[0][1] >= 0.5): 
        msg = "this is a HATE SPEECH text"
    else :
        msg = "this is a NORMAL text"


    # Return the prediction
    return render_template('index.html', prediction_text='Predicted text: {} '.format(msg))


if __name__ == "__main__":
    app.run(debug=True)





'''
@app.route('/predict', methods=['POST'])
def predict():
    # UI rendering the results
    # Retrieve values from a form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features) # Make a prediction

    return render_template('index.html', prediction_text='Predicted Species: {}'.format(prediction)) # Render the predicted result
'''