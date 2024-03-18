from flask import Flask, render_template, request, jsonify
import pickle
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)
model_file_path = "Logistic_Regression_model.pkl"
with open(model_file_path, "rb") as model_file:
    loaded_model = pickle.load(model_file)

vectorizer_file_path = "TfidfVectorizer.pkl"
with open(vectorizer_file_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

stopwords_set = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove punctuation
    text = "".join([char if char not in string.punctuation else ' ' for char in text])
    # Tokenize, lowercase, and remove stopwords
    tokens = [word.lower() for word in text.split() if word.lower() not in stopwords_set]
    return tokens

def lemmatize_text(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        if not news_text:
            return jsonify({'error': 'Empty input!'}), 400
        
        # Preprocess and vectorize the input text
        preprocessed_text = preprocess_text(news_text)
        lemmatized_text = lemmatize_text(preprocessed_text)
        text_vector = vectorizer.transform([lemmatized_text])

        # Make prediction using the loaded model
        prediction = loaded_model.predict(text_vector)

        if prediction[0] == 1:
            result = "Real News"
        else:
            result = "Fake News"

        response = {
            'prediction': result,
            'news_text': news_text
        }

        return jsonify(response)
    else:
        return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True)


