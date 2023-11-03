import pandas as pd
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

df = pd.read_csv("jobs.csv")
print(df.head())

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    # Join filtered tokens back into a string
    text = " ".join(filtered_text)
    
    return text

df["key_skills"] = df["key_skills"].apply(preprocess_text)


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["key_skills"])

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["key_skills"])


y = df["job_title"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_test)



param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)

key_skills = ["python", "data analysis", "html", "css", "javascript"]




from joblib import dump, load

# Save the model to a file
dump(svm_model, 'svm_model.joblib')


# svm_model = load('svm_model.joblib')
# y_pred = svm_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: {:.2f}%".format(accuracy*100))
# print(classification_report(y_test, y_pred))

# Preprocess the text
key_skills = [preprocess_text(text) for text in key_skills]

# Convert the text into numerical features
X_new = vectorizer.transform(key_skills)

# Use the trained model to predict the job title
job_title = svm_model.predict(X_new)


print(job_title)

