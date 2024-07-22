from flask import Flask, render_template, request, make_response, jsonify
import pdfkit
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import random

app = Flask(__name__)

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

# Initialize the model pipeline
model = make_pipeline(MultinomialNB())

# Hyperparameter tuning
param_grid = {'multinomialnb__alpha': [0.1, 0.5, 1.0]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Evaluate the model
model = grid_search.best_estimator_
y_pred = model.predict(X_test)

# Define global variables for result and plot_url
result = ""
plot_url = ""
precision_value = ""
recall_value = ""
f1_value = ""

def scrape_news():
    # Dummy function for demonstration
    # Replace this with actual web scraping code
    # For example, you can use BeautifulSoup to scrape news articles from a news website
    # Here, we'll generate some dummy news articles
    headlines = ['Breaking: Scientists Discover New Planet', 'COVID-19 Vaccine Rollout Begins Worldwide', 'Stock Market Surges to Record High']
    articles = [' '.join([random.choice(['Lorem', 'Ipsum', 'Dolor', 'Sit', 'Amet']) for _ in range(50)]) for _ in range(3)]
    return headlines, articles

@app.route('/')
def home():
    global plot_url
    return render_template('index.html', result=result, plot_url=plot_url, precision=precision_value, recall=recall_value, f1=f1_value)

@app.route('/predict', methods=['POST'])
def predict():
    global result, plot_url, precision_value, recall_value, f1_value
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

    # Prepare the response data
    response_data = {
        'result': result,
        'plot_url': plot_url,
        'precision': precision_value,
        'recall': recall_value,
        'f1': f1_value
    }

    # Return the response as JSON
    return jsonify(response_data)

@app.route('/convert_to_pdf', methods=['POST'])
def convert_to_pdf():
    global result, plot_url, precision_value, recall_value, f1_value
    # Render the HTML template with the latest values and graph
    rendered_html = render_template('index.html', result=result, plot_url=plot_url, precision=precision_value, recall=recall_value, f1=f1_value)
    
    # Create a PDF from the rendered HTML
    options = {
        'enable-local-file-access': None
    }
    pdf = pdfkit.from_string(rendered_html, False, options=options)

    # Set response headers for PDF
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=News_Classifier_Result.pdf'

    return response

if __name__ == '__main__':
    app.run(debug=True)
