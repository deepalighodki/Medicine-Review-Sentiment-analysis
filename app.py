# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:43:56 2020

@author: Dipu
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)
df=pd.DataFrame()
#decorator (@app.route('/')) to specify the URL that should trigger the execution of the home function

@app.route('/')
def home():
        return render_template('home.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
	
     df = pd.read_csv('train.csv', encoding="latin-1")
     df.drop(['Id','drugName','condition','date','usefulCount'], axis=1, inplace=True)
     names=['message','label']   
     df.columns=names
 # Features and Labels
     X = df['message']
     y = df['label']

 # Extract Feature With CountVectorizer
     cv = CountVectorizer()
     X = cv.fit_transform(X) # Fit the Data
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
     #Naive Bayes Classifier
     from sklearn.naive_bayes import MultinomialNB

     clf = MultinomialNB()
     clf.fit(X_train,y_train)
     clf.score(X_test,y_test)
 #Alternative Usage of Saved Model
 # joblib.dump(clf, 'dr_model.pkl')
 # dr_model = open('dr_model.pkl','rb')
 # clf = joblib.load(dr_model)

     if request.method == 'POST':
                message = request.form['message']
                data = [message]
                vect = cv.transform(data)
                my_prediction = clf.predict(vect)
     return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
     app.run(debug=True,use_reloader=False)