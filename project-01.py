import numpy as np
import tensorflow
import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


import pandas as pd
data=pd.read_csv(r"C:\Users\yogass\Desktop\dataset\train.txt",sep=';')
data.columns = ["Text", "Emotions"]
#print(data.head)

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

vectorizer=TfidfVectorizer()
tfidf_matrix= vectorizer.fit_transform(texts)
tfidf_array= tfidf_matrix.toarray()
feature_names = vectorizer.get_feature_names_out()

#label the output
label_encoder= LabelEncoder()
label= label_encoder.fit_transform(labels)
print(len(labels))

print(pd.unique(labels))

ohe= OneHotEncoder()
ohe_labels=ohe.fit_transform(labels).toarray()

