import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/house_prices.csv")
X = df[["area"]]
y = df["price"]

model = LinearRegression().fit(X, y)
print("Model trained. Score1:", model.score(X, y))
print("this add-model-score branch")
print("3")