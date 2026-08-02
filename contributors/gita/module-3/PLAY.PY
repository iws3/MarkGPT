import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
# Load everything we saved from the training step.
# ---------------------------------------------------------------------------
# joblib.load() reads back the exact fitted model object we saved earlier —
# same internal coefficients/trees, nothing re-trained, nothing re-computed.
log_reg = joblib.load('log_reg_model.joblib')
rf = joblib.load('rf_model.joblib')

# Logistic Regression was trained on SCALED features, so it must also
# predict on SCALED test features — using the same scaler's fitted
# mean/std from Part 10. Random Forest was trained on the unscaled,
# encoded features, so it predicts on the unscaled test set.
X_test_scaled = pd.read_csv('X_test_scaled.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')['target']

# ---------------------------------------------------------------------------
# .predict() — this is the moment of truth: the model has NEVER seen these
# 184 rows during training or cross-validation. This is the first and only
# time it makes predictions on genuinely held-out data.
# ---------------------------------------------------------------------------
log_pred = log_reg.predict(X_test_scaled)
rf_pred = rf.predict(X_test)

# ---------------------------------------------------------------------------
# confusion_matrix(y_true, y_pred) arguments:
#   y_test    -> the REAL answers (what actually happened to each patient)
#   log_pred  -> the model's GUESSES for those same 184 patients, in the
#                same row order
# It returns a 2x2 numpy array. By scikit-learn's convention, rows are the
# TRUE class (0 then 1) and columns are the PREDICTED class (0 then 1):
#     [[TN, FP],
#      [FN, TP]]
# ---------------------------------------------------------------------------
cm_log = confusion_matrix(y_test, log_pred)
cm_rf = confusion_matrix(y_test, rf_pred)

print("Logistic Regression confusion matrix:")
print(cm_log)
print("Random Forest confusion matrix:")
print(cm_rf)

# ---------------------------------------------------------------------------
# Visualization — ConfusionMatrixDisplay
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# display_labels=[...] -> renames the 0/1 axis ticks to readable words
# instead of bare numbers, so the plot reads naturally.
disp_log = ConfusionMatrixDisplay(cm_log, display_labels=['No Disease', 'Disease'])
# cmap='Blues' -> colors darker squares = higher counts, lighter = lower.
# ax=axes[0]   -> draws this specific plot into the left subplot rather than
#                 opening a new figure window.
disp_log.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Logistic Regression', fontweight='bold')

disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=['No Disease', 'Disease'])
disp_rf.plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title('Random Forest', fontweight='bold')

plt.suptitle('Confusion Matrices on the Held-Out Test Set (184 patients)', fontweight='bold')
plt.tight_layout()
plt.savefig('eda/confusion_matrices.png', dpi=130)
plt.close()
print("\nSaved confusion matrix plot.")