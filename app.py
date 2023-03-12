from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
#from preprocess import *

#text preprocessing########################################################################################
import nltk
# from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

list_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


def preprocess(line):
    ps = PorterStemmer()

    review = re.sub('[^a-zA-Z]', ' ', line) #leave only characters from a to z
    review = review.lower() #lower the text
    review = review.split() #turn string into list of words
    #apply Stemming
    review = [ps.stem(word) for word in review if not word in list_words] #delete stop words like I, and ,OR   review = ' '.join(review)
    #trun list into sentences
    return " ".join(review)

###############################################################################################################################################################

encoder = pickle.load(open('encoder.pkl', 'rb'))
cv = pickle.load(open('CountVectorizer.pkl', 'rb'))

model = tf.keras.models.load_model('my_model.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return "THE BASIC API IS WORKING FINE"

@app.route('/predict', methods=['POST'] )
def predict():

    text = request.form.get('text')
    input = preprocess(text)

    array = cv.transform([input]).toarray()

    pred = model.predict(array)
    a = np.argmax(pred, axis=1)
    prediction = encoder.inverse_transform(a)[0]

    # print(prediction)
    # print(text)
    # print(type(text))


    result = prediction

    print(result)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
