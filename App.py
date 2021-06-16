from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pnd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

#loaded_model = pickle.load(open('final_model1.pkl', 'rb'))

dataf = pnd.read_csv("fake_or_real_news.csv")


def detecting_fake_news(var):    
#retrieving the best model for prediction call
    load_model = pickle.load(open('final_model1.pkl', 'rb'))
    prediction = load_model.predict([var])
    #prob = load_model.predict_proba([var])

    return prediction

#def fake_news_det(news):

   
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred =detecting_fake_news(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)