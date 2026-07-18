import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import gradio as gr
import numpy as np
import joblib

df = pd.read_csv("boston.csv")

df = df[df['MEDV'] < 50]
df['CRIM'] = np.log1p(df['CRIM'])
df['LSTAT'] = np.log1p(df['LSTAT'])
df['TAX'] = np.log1p(df['TAX'])

# split to feature and target
X = df.drop('MEDV', axis = 1)
y = df['MEDV']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Scaling; ensure features like TAX and RM are on thesame scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initializing and training
# we use a max_depth of 6 to prevent the model from memorizing (overfitting) the data
model = RandomForestRegressor(n_estimators = 100, max_depth = 6, random_state = 42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"R2 Score: {r2:.4f}")
print(f"MAE: ${mae * 1000:,.2f}")
print(f"RMSE: ${rmse * 1000:,.2f}")