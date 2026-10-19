
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# 1. Load data
data = pd.read_csv("data/houses.csv")

# 2. Separate input and output
X = data[["size_sqft"]]
y = data["price_lakhs"]

# 3. Train the model
model = LinearRegression()
model.fit(X, y)

# 4. Save the trained model
joblib.dump(model, "model.joblib")

# 5. Check performance on training data
print("Training R2 score:", model.score(X, y))
print("Model saved successfully!")
