import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('linearregression1.pkl','rb')) 

@app.route('/')
def home():
  
    return render_template("index2.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp = float(request.args.get('exp'))
    
    prediction = model.predict([[exp]])
    
        
    return render_template('index2.html', prediction_text='Regression Model  has predicted price for given SqurFeet is : {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
