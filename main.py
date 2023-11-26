import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

flask_app_=Flask(__name__, template_folder='templates', static_folder='static')
model = pickle.load(open(r'notebook\finaal.pkl', 'rb'))

#name of columns of the data
input_names = ['Pclass', 'Sex', 'Age','SibSp', 'Fare','Embarked','family']

cat_features = ['Sex', 'Embarked']


@flask_app_.route('/')
def home():
    return render_template('home.html')

@flask_app_.route('/predict', methods=['POST'])
def predict():
    features=[]

    for col in input_names:
        #get values typed in the page
        value=request.form.get(col)

        if col in cat_features:
            #apply label encoder on the categorical features
            le=pickle.load(open(r'notebook\{}_le.pkl'.format(col), 'rb'))
            v=le.transform(np.array([value]))
            scaler = pickle.load(open(r'notebook\{}_scale.pkl'.format(col), 'rb'))
            z=scaler.transform([v])
            features.append(float(z)) #convert 'v' to float instead of array

        else:
            scaler = pickle.load(open(r'notebook\{}_scale.pkl'.format(col), 'rb'))
            zz=scaler.transform(np.array([[value]]))

            features.append(float(zz))

    #shaping the feature
    print(features) #debugging
    x=np.array(features).reshape(1,7)
    y_pred=model.predict(x)

    result=[]
    if y_pred==1:
        result='passenger survived'
    else:
        result='passenger not survived'
    print(y_pred)#debugging
    return render_template('result.html',prediction_text=result)

if __name__ =='__main__':
    flask_app_.run(debug=True)


