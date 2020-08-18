from flask import Flask
from flask import request, jsonify
import pickle
import numpy as np
import pandas as pd
import os


app = Flask(__name__)

@app.route('/')
def hello():
    return "Welcome to boston house price prediction!/n usage:/predict_single?CRIM=...&ZN=...&..."


@app.route('/predict_single')
def predict_single():
    with open('my_model.pkl', 'rb') as file:
        model = pickle.load(file)

    CRIM = float(request.args.get('CRIM'))
    ZN = float(request.args.get('ZN'))
    INDUS = float(request.args.get('INDUS'))
    CHAS = float(request.args.get('CHAS'))
    NOX = float(request.args.get('NOX'))
    RM = float(request.args.get('RM'))
    AGE = float(request.args.get('AGE'))
    DIS = float(request.args.get('DIS'))
    RAD = float(request.args.get('RAD'))
    TAX = float(request.args.get('TAX'))
    PTRATIO = float(request.args.get('PTRATIO'))
    B = float(request.args.get('B'))
    LSTAT = float(request.args.get('LSTAT'))

    variables = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

    return str(model.predict(variables)[0])

@app.route('/predict_json')
def predict_json():
    
    with open('my_model.pkl', 'rb') as file:
        model = pickle.load(file)

    data = request.get_json()
    df = pd.DataFrame(data=data['data'])
    print(df)

    pred = model.predict(df)

    output = {}
    for i in range(len(pred)):
        output[i]=pred[i]

    return jsonify(output)

@app.route('/predict_list')
def predict_list():

    with open('my_model.pkl', 'rb') as file:
        model = pickle.load(file)

    data = request.get_json()
    df = pd.DataFrame(data)
    pred = model.predict(df)

    output = {}
    for i in range(len(pred)):
        output[i]=pred[i]

    return jsonify(output)


if __name__ == '__main__':
    # with open('my_model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()
