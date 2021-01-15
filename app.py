from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("students.pkl", "rb"))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    
 
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5])])
    print(row_df)
    prediction=model.predict_proba(row_df)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    output = str(float(output))+'%'

    if output < str(0.4):
        return render_template('result.html', pred=f'You are not in danger of Spill Over. '.format(output))
    else : 
        return render_template('result.html',pred=f'You have already Spilled Over. '.format(output))



if __name__ == '__main__':
    app.run(debug=True)
