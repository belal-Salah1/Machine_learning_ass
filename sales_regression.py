import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("SuperMarketSales.csv")
df["Month"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.month

y = df["Weekly_Sales"]
results = {}
for col in ["Store", "Temperature", "Fuel_Price", "CPI", "Month"]:
    X_train, X_test, y_train, y_test = train_test_split(df[[col]], y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    results[col] = mean_squared_error(y_test, model.predict(X_test))

mse = pd.Series(results, name="MSE").sort_values()
print(mse.to_string(float_format="{:,.0f}".format))
print(f"\nBest X variable: {mse.idxmin()}")
