import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay

# Simulated data
y_true = [1]*11 + [0]*11                 # 11 registered, 11 unregistered
y_pred = [1]*10 + [0] + [1]*2 + [0]*9    # TP=10, FN=1, FP=2, TN=9

# Confusion Matrix
labels = ['Unregistered', 'Registered']
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Face Recognition (Simulated)")
plt.grid(False)
plt.show()

# Accuracy and report
accuracy = accuracy_score(y_true, y_pred)
print(f"\n🎯 Accuracy of model is {accuracy * 100:.2f}%\n")

# Detailed classification report
report = classification_report(y_true, y_pred, target_names=labels, digits=2)
print("📊 Classification Report:\n")
print(report)
