import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

 
data = {
  "Amount": [25.0, 32.5, 15.0, 22.1, 8500.0],
  "Time_spent_sec": [120, 95, 150, 110, 5]
 }
df = pd.DataFrame(data)

model = IsolationForest(contamination=0.20, random_state=42)
df["Anomaly_Label"] = model.fit_predict(df)

df["score"] = model.decision_function(df[["Amount", 'Time_spent_sec']])

print(df.to_string(index=False))
 