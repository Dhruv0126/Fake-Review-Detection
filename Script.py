# Part 1 .........

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


nltk.download('omw-1.4')
df = pd.read_csv('fake reviews dataset.csv')
df.head()
df.isnull().sum()


df.info()
ddf['rating'].value_counts()
plt.figure(figsize=(15,8))
labels = df['rating'].value_counts().keys()
values = df['rating'].value_counts().values
explode = (0.1,0,0,0,0)
plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')
plt.title('Proportion of each rating',fontweight='bold',fontsize=25,pad=20,color='crimson')
plt.show()

def clean_text(text):
    nopunc = [w for w in text if w not in string.punctuation]
    nopunc = ''.join(nopunc)
    return  ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

df['text_'][0], clean_text(df['text_'][0])
df['text_'].head().apply(clean_text)
df.shape
#df['text_'] = df['text_'].apply(clean_text)
df['text_'] = df['text_'].astype(str)

def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])


preprocess(df['text_'][4])

df['text_'][:10000] = df['text_'][:10000].apply(preprocess)

df['text_'][10001:20000] = df['text_'][10001:20000].apply(preprocess)

df['text_'][20001:30000] = df['text_'][20001:30000].apply(preprocess)


df['text_'][30001:40000] = df['text_'][30001:40000].apply(preprocess)

df['text_'][40001:40432] = df['text_'][40001:40432].apply(preprocess)
df['text_'] = df['text_'].str.lower()
stemmer = PorterStemmer()
def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])
df['text_'] = df['text_'].apply(lambda x: stem_words(x))
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
df["text_"] = df["text_"].apply(lambda text: lemmatize_words(text))
df['text_'].head()
df.to_csv('Preprocessed Fake Reviews Detection Dataset.csv')

df = pd.read_csv('Preprocessed Fake Reviews Detection Dataset.csv')
df.head()
df.drop('Unnamed: 0',axis=1,inplace=True)

df.head()
df.dropna(inplace=True)
df['length'] = df['text_'].apply(len)

df.info()

plt.hist(df['length'],bins=50)
plt.show()
df.groupby('label').describe()
df.hist(column='length',by='label',bins=50,color='blue',figsize=(12,5))
plt.show()
df[df['label']=='OR'][['text_','length']].sort_values(by='length',ascending=False).head().iloc[0].text_

df.length.describe()
def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

bow_transformer = CountVectorizer(analyzer=text_process)
bow_transformer

bow_transformer.fit(df['text_'])
print("Total Vocabulary:",len(bow_transformer.vocabulary_))
review4 = df['text_'][3]
review4
bow_msg4 = bow_transformer.transform([review4])
print(bow_msg4)
print(bow_msg4.shape)

print(bow_transformer.get_feature_names()[15841])
print(bow_transformer.get_feature_names()[23848])
bow_reviews = bow_transformer.transform(df['text_'])
print("Sparsity:",np.round((bow_reviews.nnz/(bow_reviews.shape[0]*bow_reviews.shape[1]))*100,2))


tfidf_transformer = TfidfTransformer().fit(bow_reviews)
tfidf_rev4 = tfidf_transformer.transform(bow_msg4)
print(bow_msg4)
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['mango']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['book']])


tfidf_reviews = tfidf_transformer.transform(bow_reviews)
print("Shape:",tfidf_reviews.shape)
print("No. of Dimensions:",tfidf_reviews.ndim)

review_train, review_test, label_train, label_test = train_test_split(df['text_'],df['label'],test_size=0.35)
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])


pipeline.fit(review_train,label_train)
predictions = pipeline.predict(review_test)
predictions

print('Classification Report:',classification_report(label_test,predictions))
print('Confusion Matrix:',confusion_matrix(label_test,predictions))
print('Accuracy Score:',accuracy_score(label_test,predictions))


print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,predictions)*100,2)) + '%')

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
])

pipeline.fit(review_train,label_train)


rfc_pred = pipeline.predict(review_test)
rfc_pred

print('Classification Report:',classification_report(label_test,rfc_pred))
print('Confusion Matrix:',confusion_matrix(label_test,rfc_pred))
print('Accuracy Score:',accuracy_score(label_test,rfc_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,rfc_pred)*100,2)) + '%')


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',DecisionTreeClassifier())
])


pipeline.fit(review_train,label_train)

dtree_pred = pipeline.predict(review_test)
dtree_pred

print('Classification Report:',classification_report(label_test,dtree_pred))
print('Confusion Matrix:',confusion_matrix(label_test,dtree_pred))
print('Accuracy Score:',accuracy_score(label_test,dtree_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',KNeighborsClassifier(n_neighbors=2))
])

pipeline.fit(review_train,label_train)

knn_pred = pipeline.predict(review_test)
knn_pred
print('Classification Report:',classification_report(label_test,knn_pred))
print('Confusion Matrix:',confusion_matrix(label_test,knn_pred))
print('Accuracy Score:',accuracy_score(label_test,knn_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,knn_pred)*100,2)) + '%')
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',SVC())
])
pipeline.fit(review_train,label_train)
svc_pred = pipeline.predict(review_test)
svc_pred
print('Classification Report:',classification_report(label_test,svc_pred))
print('Confusion Matrix:',confusion_matrix(label_test,svc_pred))
print('Accuracy Score:',accuracy_score(label_test,svc_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,svc_pred)*100,2)) + '%')
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',LogisticRegression())
])
pipeline.fit(review_train,label_train)
lr_pred = pipeline.predict(review_test)
lr_pred
print('Classification Report:',classification_report(label_test,lr_pred))
print('Confusion Matrix:',confusion_matrix(label_test,lr_pred))
print('Accuracy Score:',accuracy_score(label_test,lr_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,lr_pred)*100,2)) + '%')


print('Performance of various ML models:')
print('\n')
print('Logistic Regression Prediction Accuracy:',str(np.round(accuracy_score(label_test,lr_pred)*100,2)) + '%')
print('K Nearest Neighbors Prediction Accuracy:',str(np.round(accuracy_score(label_test,knn_pred)*100,2)) + '%')
print('Decision Tree Classifier Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')
print('Random Forests Classifier Prediction Accuracy:',str(np.round(accuracy_score(label_test,rfc_pred)*100,2)) + '%')
print('Support Vector Machines Prediction Accuracy:',str(np.round(accuracy_score(label_test,svc_pred)*100,2)) + '%')
print('Multinomial Naive Bayes Prediction Accuracy:',str(np.round(accuracy_score(label_test,predictions)*100,2)) + '%')

# finally select SVM as a prediction algo 

import joblib

# Train the model (using your pipeline)
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process,max_features=5000)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC(probability=True ))
])

pipeline.fit(review_train, label_train)

# Save the trained model
joblib.dump(pipeline, 'fake_review_detection_model_svc.pkl')

# Later, to predict new data:
loaded_model = joblib.load('fake_review_detection_model_svc.pkl')

import joblib
import shap
import numpy as np

# Load the trained pipeline
pipeline = joblib.load('fake_review_detection_model_svc.pkl')

# Use a small background dataset for SHAP
background_data = pipeline.named_steps['tfidf'].transform(
    pipeline.named_steps['bow'].transform(review_train[:10])  # Use 10 samples
)

# Initialize SHAP KernelExplainer
explainer = shap.KernelExplainer(
    pipeline.named_steps['classifier'].predict_proba,
    background_data
)

# New review to explain
new_review = ["THIS PRODUCT IS NICE"]
new_review_transformed = pipeline.named_steps['tfidf'].transform(
    pipeline.named_steps['bow'].transform(new_review)
)

# Compute SHAP values
shap_values = explainer.shap_values(new_review_transformed)

# Visualize SHAP explanation
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], new_review)
