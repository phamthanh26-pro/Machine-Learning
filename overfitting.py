import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Tạo dữ liệu có quan hệ phi tuyến và nhiễu
rng = np.random.default_rng(42)

x = np.linspace(-3, 3, 120)
y = 0.5 * x**3 - 2 * x + 1 + rng.normal(0, 2, size=len(x))

data = pd.DataFrame({
    "x": x,
    "y": y
})

# Trộn dữ liệu trước khi chia
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Chia train/validation/test: 60% / 20% / 20%
train_end = int(len(data) * 0.6)
validation_end = int(len(data) * 0.8)

train = data.iloc[:train_end]
validation = data.iloc[train_end:validation_end]
test = data.iloc[validation_end:]


def polynomial_features(x, degree):
    """Tạo ma trận [1, x, x^2, ..., x^degree]."""
    x = np.asarray(x)

    return np.column_stack([
        x ** power for power in range(degree + 1)
    ])


def train_model(x, y, degree):
    X = polynomial_features(x, degree)

    # Nghiệm least squares cho linear regression
    weights = np.linalg.lstsq(X, y, rcond=None)[0]

    return weights


def predict(x, weights):
    degree = len(weights) - 1
    X = polynomial_features(x, degree)

    return X @ weights


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 3. Huấn luyện các mô hình có bậc khác nhau
results = []

for degree in range(1, 13):
    weights = train_model(train["x"], train["y"], degree)

    train_prediction = predict(train["x"], weights)
    validation_prediction = predict(validation["x"], weights)
    test_prediction = predict(test["x"], weights)

    results.append({
        "degree": degree,
        "train_mse": mse(train["y"], train_prediction),
        "validation_mse": mse(validation["y"], validation_prediction),
        "test_mse": mse(test["y"], test_prediction)
    })

results_df = pd.DataFrame(results)

print(results_df)


# 4. Chọn mô hình có validation error nhỏ nhất
best_degree = int(
    results_df.loc[
        results_df["validation_mse"].idxmin(),
        "degree"
    ]
)

print("\nBậc mô hình tốt nhất:", best_degree)


# 5. Huấn luyện lại bằng train + validation
train_validation = pd.concat([train, validation])

best_weights = train_model(
    train_validation["x"],
    train_validation["y"],
    best_degree
)

final_test_prediction = predict(test["x"], best_weights)
final_test_mse = mse(test["y"], final_test_prediction)

print("Test MSE của mô hình cuối:", final_test_mse)


# 6. Vẽ training error và validation error
plt.figure(figsize=(10, 5))

plt.plot(
    results_df["degree"],
    results_df["train_mse"],
    marker="o",
    label="Training MSE"
)

plt.plot(
    results_df["degree"],
    results_df["validation_mse"],
    marker="o",
    label="Validation MSE"
)

plt.plot(
    results_df["degree"],
    results_df["test_mse"],
    marker="o",
    label="Test MSE"
)

plt.axvline(
    best_degree,
    color="green",
    linestyle="--",
    label=f"Best degree = {best_degree}"
)

plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.title("Training, Validation và Test Error")
plt.legend()
plt.grid(True)
plt.show()


# 7. Vẽ đường fit của ba trường hợp
x_plot = np.linspace(-3, 3, 300)

degrees_to_plot = [1, 3, 12]

plt.figure(figsize=(10, 6))

plt.scatter(train["x"], train["y"], label="Training data")
plt.scatter(validation["x"], validation["y"], label="Validation data")
plt.scatter(test["x"], test["y"], label="Test data")

for degree in degrees_to_plot:
    weights = train_model(train["x"], train["y"], degree)
    y_plot = predict(x_plot, weights)

    plt.plot(
        x_plot,
        y_plot,
        label=f"Degree {degree}"
    )

plt.xlabel("x")
plt.ylabel("y")
plt.title("Underfitting, Good Fit và Overfitting")
plt.legend()
plt.grid(True)
plt.show()