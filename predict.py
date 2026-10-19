
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model.joblib")

# New house size
new_house = pd.DataFrame({"size_sqft": [900]})

# Predict price
prediction = model.predict(new_house)

print(f"Predicted house price: ₹{prediction[0]:.2f} lakhs")
