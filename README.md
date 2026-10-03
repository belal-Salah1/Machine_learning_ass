# Weekly Sales Prediction — Simple Linear Regression

Predicts supermarket `Weekly_Sales` with five simple linear regression models, each using a single X variable, and compares them by test-set MSE.

## Files

| File | Description |
| --- | --- |
| `SuperMarketSales.csv` | Dataset: `Store`, `Date`, `Temperature`, `Fuel_Price`, `CPI`, `Weekly_Sales` (6,435 rows) |
| `sales_regression.py` | Trains one model per X variable and prints the MSE of each |
| `report.pdf` | Report with the method, MSE table, and conclusion |

## Method

- A new feature, `Month`, is extracted from the `Date` column.
- For each of `Store`, `Temperature`, `Fuel_Price`, `CPI`, and `Month`, a `LinearRegression` model is trained on 80% of the data and tested on the remaining 20% (`random_state=42`).
- Models are compared by mean squared error (MSE) on the test set.

## Run

```bash
pip install pandas scikit-learn
python sales_regression.py
```

## Results

| X Variable | MSE |
| --- | ---: |
| Store | 284,296,629,512 |
| CPI | 320,370,061,621 |
| Temperature | 320,515,110,459 |
| Month | 320,917,790,541 |
| Fuel_Price | 322,204,752,659 |

**Conclusion:** `Store` is the best X variable, with the lowest MSE.
