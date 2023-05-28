from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['GET', 'POST'])
def main():

    # If a form is submitted
    if request.method == "POST":

        clf = joblib.load("clf.pkl")

        # Get values through input bars
        height = request.form.get("height")
        weight = request.form.get("weight")

        # Put inputs to dataframe
        X = pd.DataFrame([[height, weight]], columns=["Height", "Weight"])

        # Get prediction
        prediction = clf.predict(X)[0]

    else:
        prediction = ""

    return render_template("website.html", output=prediction)


if __name__ == '__main__':
    app.run(debug=True)
