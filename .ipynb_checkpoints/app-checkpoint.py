from flask import Flask, render_template, request, jsonify, make_response, g
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import random
import pdfkit
import sqlite3
import os

app = Flask(__name__)

DATABASE = 'search_history.db'
UPLOAD_FOLDER = 'uploads'  # Folder to save the Excel file

# Function to create the SQLite database and table
def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        # Check if the table already exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_history';")
        table_exists = cursor.fetchone()
        if not table_exists:
            # Table does not exist, create it
            with app.open_resource('schema.sql', mode='r') as f:
                cursor.executescript(f.read())
            db.commit()
        else:
            print("Table search_history already exists.")

# Function to get a database connection
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE, check_same_thread=False)
    return db

# Function to insert a search entry into the database
def insert_search_entry(term, result):
    # Truncate the term (title) to 70 characters
    truncated_term = term[:70]
    db = get_db()
    db.execute('INSERT INTO search_history (term, result) VALUES (?, ?)', (truncated_term, result))
    db.commit()

# Load datasets
true_df = pd.read_csv('True.csv')
true_df['label'] = 0  
fake_df = pd.read_csv('Fake.csv')
fake_df['label'] = 1  
df = pd.concat([true_df, fake_df], ignore_index=True)

# Convert non-string text data to string
df['text'] = df['text'].astype(str)

# Text preprocessing and feature extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X = tfidf_vectorizer.fit_transform(df['text'])
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model pipeline with DecisionTreeClassifier
model = make_pipeline(DecisionTreeClassifier())  # Change here

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Generate classification report
classification_rep = classification_report(y_test, y_pred)

# Print classification report
print("Classification Report:")
print(classification_rep)

# Function to retrieve search history from the database
def get_search_history():
    db = get_db()
    cur = db.execute('SELECT * FROM search_history ORDER BY id DESC')
    return cur.fetchall()

@app.route('/')
def home():
    return render_template('index.html',
                           true_wordcloud="",
                           fake_wordcloud="",
                           result="",
                           plot_url="",
                           precision="",
                           recall="",
                           f1="",
                           classification_report=classification_rep,
                           search_history=get_search_history())

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    input_vector = tfidf_vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    probabilities = model.predict_proba(input_vector)[0]

    if prediction[0] == 0:
        result = "The news is likely to be true with a probability of " + str(probabilities[0])
    else:
        result = "The news is likely to be fake with a probability of " + str(probabilities[1])

    # Generate data for the scatter plot
    categories = ['True', 'Fake']
    x_values = np.arange(len(categories))
    y_values = probabilities

    # Plot the scatter plot
    plt.figure()
    plt.scatter(x_values, y_values, c=['green', 'red'])
    plt.xticks(x_values, categories)
    plt.xlabel('Category')
    plt.ylabel('Probability')
    plt.title('Probability Distribution')
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Calculate additional metrics
    precision_value = precision_score(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    f1_value = f1_score(y_test, y_pred)

    # Insert search entry into database
    insert_search_entry(user_input, result)

    return jsonify(result=result, plot_url=plot_url, precision=precision_value, recall=recall_value, f1=f1_value)

@app.route('/convert_to_pdf', methods=['POST'])
def convert_to_pdf():
    html = render_template('index.html',
                           true_wordcloud="",
                           fake_wordcloud="",
                           result=request.form['result'],
                           plot_url=request.form['plot_url'],
                           precision=request.form['precision'],
                           recall=request.form['recall'],
                           f1=request.form['f1'],
                           search_history=get_search_history())
    pdf = pdfkit.from_string(html, False)
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=output.pdf'
    return response

@app.route('/download_excel', methods=['GET'])
def download_excel():
    # Get search history from the database
    search_history = get_search_history()
    # Create a DataFrame from the search history
    df = pd.DataFrame(search_history, columns=['Title', 'Probability', 'True/Fake'])
    # Define the path to save the Excel file
    excel_file_path = os.path.join(UPLOAD_FOLDER, 'search_history.xlsx')
    # Save the DataFrame to an Excel file
    df.to_excel(excel_file_path, index=False)
    return "Excel file has been saved to the server."

if __name__ == '__main__':
    init_db()  # Initialize the SQLite database
    app.run(debug=True)
