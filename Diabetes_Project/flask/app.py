from flask import Flask, request, url_for, redirect, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("Diabetes.pkl", "rb"))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']

    new_data_df = pd.DataFrame([pd.Series([text2, text5, text6, text8])])
    print(new_data_df)
    prediction = model.predict_proba(new_data_df)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    output1 = str(float(output) * 100) + '%'
    if output > str(0.5):
        return render_template('result.html',
                               pred=f'You have chance of having diabetes.\nProbability of having Diabetes is {output1}')
    else:
        return render_template('result.html', pred=f'You are safe.\n Probability of having diabetes is {output1}')


if __name__ == '__main__':
    app.run(debug=True)