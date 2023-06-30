from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    list = [int(x) for x in request.form.values()]

    loaded_model = pickle.load(open("./model.pkl", "rb"))
    final = np.array(list)

    prediction = loaded_model.predict([final])
    output = "Predicted Weather :  " + str(prediction[0])

    print(prediction)
    return render_template('index.html',pred = output)


if __name__ == "__main__":
    app.run(debug = True)
