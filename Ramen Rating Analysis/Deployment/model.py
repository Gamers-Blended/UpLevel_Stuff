import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../Dataset/thebiglist_clean_extended_final.csv")

# qcut() automatically splits data into intervals
df["binStars"] = pd.qcut(df["Stars"], q = 2, labels = [0, 1])

# prepare independent and dependent variables
X = df.drop(["binStars", "Stars"], axis = 1)
y = df["binStars"]

# split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size = 0.2,
                                                   stratify = y)


# LogisticRegression
logr = LogisticRegression()
logr.fit(X_train, y_train)

# DecisionTreeClassifier
dtclf = DecisionTreeClassifier()
dtclf.fit(X_train, y_train)

# RandomForestClassifier
rfclf = RandomForestClassifier()
rfclf.fit(X_train, y_train)

# Save models to disk
pickle.dump(logr, open('logr_model.pkl', 'wb'))
pickle.dump(dtclf, open('dtclf_model.pkl', 'wb'))
pickle.dump(rfclf, open('rfclf_model.pkl', 'wb'))

# Load model to compare results
model = pickle.load(open('logr_model.pkl', 'rb'))
print(model.predict(X_test))