import nltk
import os
import re
import joblib
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import warnings
from sklearn.model_selection import GridSearchCV

# Ensure nltk data path is set
nltk_data_path = '/Users/shaikalmasazam/nltk_data'
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

warnings.filterwarnings("ignore")

def preprocess_text(text):
    # Removing special characters and converting to lowercase
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text).lower()
    # Tokenizing the text by words
    words = text.split()
    # Removing stop words using nltk stopwords
    words = [word for word in words if word not in set(stopwords.words('english'))]
    # Stemming the words
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    # Joining the stemmed words
    processed_text = ' '.join(words)
    return processed_text

# Loading the dataset with error handling
try:
    df = pd.read_csv('/Users/shaikalmasazam/Downloads/Restaurant-Review-Sentiment-Analysis-master/Restaurant_Reviews.tsv', sep='\t', on_bad_lines='skip')
except pd.errors.ParserError as e:
    print(f"Error parsing file: {e}")
    exit()

# Preprocessing the text
df['Processed_Text'] = df['Review'].apply(preprocess_text)

# Creating Bag of Words model with n-grams
cv = CountVectorizer(max_features=1500, ngram_range=(1, 2))
X = cv.fit_transform(df['Processed_Text']).toarray()
y = df.iloc[:, 1].values

# Creating a pickle file for the CountVectorizer model
joblib.dump(cv, "cv.pkl")

# Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier_nb = MultinomialNB(alpha=0.2)
classifier_nb.fit(X_train, y_train)

# Make predictions on the test set
y_pred_nb = classifier_nb.predict(X_test)

# Evaluate the Naive Bayes model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")

f1_nb = f1_score(y_test, y_pred_nb)
print(f"Naive Bayes F1 Score: {f1_nb}")

# Creating a pickle file for the Multinomial Naive Bayes model
joblib.dump(classifier_nb, "model_nb.pkl")

# Fitting Random Forest to the Training set
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0)
classifier_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = classifier_rf.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")

f1_rf = f1_score(y_test, y_pred_rf)
print(f"Random Forest F1 Score: {f1_rf}")

# Fitting SVM to the Training set
classifier_svm = SVC(kernel='linear', random_state=0)
classifier_svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = classifier_svm.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")

f1_svm = f1_score(y_test, y_pred_svm)
print(f"SVM F1 Score: {f1_svm}")

# Fitting Logistic Regression to the Training set
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = classifier_lr.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")

f1_lr = f1_score(y_test, y_pred_lr)
print(f"Logistic Regression F1 Score: {f1_lr}")

# Define the parameter grid for Logistic Regression tuning
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}

# Create the GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(random_state=0), param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Use the best parameters to train the model
best_classifier_lr = LogisticRegression(random_state=0, **best_params)
best_classifier_lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred_best_lr = best_classifier_lr.predict(X_test)

# Evaluate the tuned Logistic Regression model
accuracy_best_lr = accuracy_score(y_test, y_pred_best_lr)
print(f"Tuned Logistic Regression Accuracy: {accuracy_best_lr}")
print(f"Best Parameters: {best_params}")

f1_best_lr = f1_score(y_test, y_pred_best_lr)
print(f"Tuned Logistic Regression F1 Score: {f1_best_lr}")
