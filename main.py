import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

data = fetch_california_housing(as_frame=True)
df = data.frame

X = df[['MedInc']]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# ЗАВДАННЯ 1:
# ==========================================
poly_2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2 = poly_2.fit_transform(X_train)
X_test_poly2 = poly_2.transform(X_test)

model_1 = LinearRegression()
model_1.fit(X_train_poly2, y_train)

y_pred_test_1 = model_1.predict(X_test_poly2)
r2_task1 = r2_score(y_test, y_pred_test_1)

print(f"Результат Завдання 1 (R² на тест для ступеня 2): {r2_task1:.4f}")
print("\n" + "=" * 50 + "\n")

# ==========================================
# ЗАВДАННЯ 2:
# ==========================================
degrees = [1, 2, 3]
results = []

for deg in degrees:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    r2_train = r2_score(y_train, model.predict(X_train_poly))
    r2_test = r2_score(y_test, model.predict(X_test_poly))

    results.append({
        'Ступінь полінома': deg,
        'R² поїзд': r2_train,
        'Тест R²': r2_test
    })

results_df = pd.DataFrame(results)

print("Результати Завдання 2:")
print(results_df.to_string(index=False))
