from flask import Flask, flash, request, render_template, redirect
from flask import url_for
from markupsafe import escape
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'nitika'

vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", 'rb'))

df = pd.read_csv('news.csv')
df['content'] = df['text']

x = df['content']
y = df['label']

vectorizer = TfidfVectorizer()
vectorizer.fit(x.values.astype('U'))
x = vectorizer.transform(x.values.astype('U'))

from scipy.sparse import csr_matrix

sm = csr_matrix(x)
X = sm.toarray()
Y = y.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template("index.html")
database= {"nitika":"123",
           "ishan":"1234"}
database1= {"nitika":"123",
           "ishan":"1234"}

@app.route('/PREDICTION', methods=['GET', 'POST'])
def PREDICTION():
    if request.method == "POST":
        news = str(request.form['news'])
        prediction = model.predict(vectorizer.transform([news]))[0]
        return render_template("predict.html", prediction_text="News headline is -> {}".format(prediction))
    else:
        return render_template("predict.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method =='POST':
        UserName = request.form.get('UserName')
        Password = request.form.get('Password')

        if UserName in database1:
             flash('Username already exist!')
             return render_template('signup.html')
         
        flash('Signup succesfull')
        return render_template( 'login.html')
       
    return render_template('signup.html')
    
    

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        UserName=request.form.get('UserName')
        Password=request.form.get('Password')

        if UserName not in database:
             flash('Invalid username')
             return render_template('login.html')
        flash('Login successful')
        return render_template('index.html', name=UserName)
    return render_template('index.html')
            
@app.route('/homep')
def homep():
    return render_template("index.html")

if __name__ == '__main__':
    app.run()
