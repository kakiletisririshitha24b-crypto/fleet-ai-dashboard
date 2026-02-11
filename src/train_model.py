import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import joblib

df = pd.read_csv("data/auto-mpg.csv")

# Data Cleaning
df.replace("?", pd.NA, inplace=True)
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df.dropna(inplace=True)

X = df[['cylinders','displacement','horsepower','weight','acceleration']]
y = df['mpg']

# Regression Model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor()
model.fit(X_train,y_train)

joblib.dump(model,"models/fuel_model.pkl")

# Clustering Model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

joblib.dump(kmeans,"models/cluster_model.pkl")

print("Models trained and saved successfully!")
