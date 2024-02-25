from flask import Flask, render_template, request
import pickle
from nltk.stem import WordNetLemmatizer
import string
import warnings
warnings.filterwarnings("ignore")








app = Flask(__name__)

# Load the pre-trained Logistic Regression model
model_file_path = "Logistic_Regression_model.pkl"
with open(model_file_path, "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Load the TfidfVectorizer used for vectorizing text
vectorizer_file_path = "TfidfVectorizer.pkl"  # Replace with the actual file path if saved separately
with open(vectorizer_file_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

text_column = "Text"
target_column = "Target_col"
stopword = set(stopwords.words("english"))

text_column = "text"
label_column = "label"

stop_words = set(stopwords.words('english'))
words = word_tokenize(text_column)




def preprocess_text(text):
    remove_punc = (char for char in text if char not in string.punctuation)
    clean_words = "".join(remove_punc)
    text = (word for word in clean_words.split() if word.lower() not in stopword)
    return list(text)






lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    lemmatized_text = " ".join([lemmatizer.lemmatize(word) for word in text])
    return lemmatized_text



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        
        # Preprocess and vectorize the input text
        preprocessed_text = preprocess_text(news_text)
        lemmatized_text = lemmatize_text(preprocessed_text)
        text_vector = vectorizer.transform([lemmatized_text])

        # Make prediction using the loaded model
        prediction = loaded_model.predict(text_vector)

        return render_template('index.html', prediction=prediction[0], news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)


