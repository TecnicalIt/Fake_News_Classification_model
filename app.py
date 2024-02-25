from flask import Flask, render_template, request
import pickle
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
import warnings
 
warnings.filterwarnings("ignore")
app = Flask(__name__)
model_file_path = "models/Logistic_Regression_model.pkl"
with open(model_file_path, "rb") as model_file:
    loaded_model = pickle.load(model_file)

local_directory = r"/workspaces/Fake_News_Classification_model/models/TfidfVectorizer.pkl"

# Define and save TfidfVectorizer to the local directory
vectorizer = None  # Initialize vectorizer
vectorizer_file_path = local_directory + r"\TfidfVectorizer.pkl"

# ... (other code)

if __name__ == '__main__':
    app.run(debug=True)
