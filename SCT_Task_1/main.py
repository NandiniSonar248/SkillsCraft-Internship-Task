import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("D:\SkillCraft Task\SCT_Task_1\house_price_dataset (1).csv")

print("Dataset Preview:\n", df.head())

# Features and Target
X = df[["square_footage", "bedrooms", "bathrooms"]]
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Coefficients
print("\nModel Details:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Test Prediction
sample = [[2000, 3, 2]]
predicted_price = model.predict(sample)
print("\nPredicted price for 2000 sqft, 3 bed, 2 bath:", predicted_price[0])

# Visualization (optional)
plt.scatter(df["square_footage"], df["price"])
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("House Price vs Square Footage")
plt.show()