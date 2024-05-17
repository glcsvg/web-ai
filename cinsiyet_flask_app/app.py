from flask import Flask,render_template,url_for,request
from flask_material import Material

import pandas as pd 
import numpy as np 

from sklearn.externals import joblib
app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/veriler.csv")
    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		height= request.form['height']
		weight = request.form['weight']
		age = request.form['age']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [height,weight,age]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'logitmodel':
		    logit_model = joblib.load('data/veriler_log_reg_model.pkl')
		    result_prediction = logit_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('data/veriler_logreg_model.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'svmmodel':
			knn_model = joblib.load('data/veriler_logreg_model.pkl')
			result_prediction = knn_model.predict(ex1)

	return render_template('index.html',height=height,
		weight=weight,
		age=sepal_length,
		petal_length=age,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)

if __name__ == '__main__':
	app.run(debug=True)