import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'
ALLOWED_EXTENSIONS = {'csv'}
model = joblib.load('fake_review_detection_model_svc.pkl')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('detection'))
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    email = request.form['email']

    session['user'] = {'first_name': first_name, 'last_name': last_name, 'email': email}
    return redirect(url_for('detection'))


@app.route('/detection')
def detection():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('detection.html')


@app.route('/analyze_review', methods=['POST'])
def analyze_review():
    if 'user' not in session:
        return redirect(url_for('index'))

    review_text = request.form['review_text']
    review_transformed = model.named_steps['tfidf'].transform(
        model.named_steps['bow'].transform([review_text])
    )
    prediction_probabilities = model.named_steps['classifier'].predict_proba(review_transformed)
    percentage_fake = prediction_probabilities[0][1] * 100
    percentage_real = prediction_probabilities[0][0] * 100
    review_result = f"Fake: {percentage_fake:.2f}%, Real: {percentage_real:.2f}%"
    return render_template('detection.html', review_result=review_result)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
