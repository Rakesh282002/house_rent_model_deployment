from flask import Flask, render_template,url_for,request,redirect
import pickle
import numpy as np
model=pickle.load(open("E:\ML Projects\hrp_model.pkl",'rb'))
app = Flask(__name__)
@app.route('/',methods=["GET","POST"])
def home():
    return render_template("house_rent_web.html")
@app.route('/home1',methods=['POST'])
def home1():
    data1=request.form['bh']
    data2=request.form['sz']
    data3=request.form['at']
    data4=request.form['ct']
    data5=request.form['fs']
    data6=request.form['tp']
    data7=request.form['poc']
    
    data1 = np.asarray(data1, dtype='int32')
    data2 = np.asarray(data2, dtype='int32')
    data3 = np.asarray(data3, dtype='int64')
    data4 = np.asarray(data4, dtype='int64')
    data5 = np.asarray(data5, dtype='int64')
    data6 = np.asarray(data6, dtype='int64')
    data7 = np.asarray(data7, dtype='int64')
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7]])
    pred=model.predict(arr)
    return render_template("after.html",data=round(pred[0],0))
if __name__=="__main__":
    app.run(debug=True)