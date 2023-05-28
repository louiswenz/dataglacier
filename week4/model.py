import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

df = pd.read_csv("data.csv")

X = df[["Height", "Weight"]]
y = df["Species"]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
# predictions = knn.predict(X)

joblib.dump(knn, "clf.pkl")
