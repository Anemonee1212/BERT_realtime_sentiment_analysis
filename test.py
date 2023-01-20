import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_text as text

from bert import data_val, y_val
from sklearn.metrics import roc_curve, precision_recall_curve


my_model = tf.keras.models.load_model("saved_models/twitter_bert")

y_val_pred = my_model.predict(data_val)

fpr, tpr, _ = roc_curve(y_val, y_val_pred)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristics")
plt.savefig("output/roc.png")

precision, recall, _ = precision_recall_curve(y_val, y_val_pred)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("output/prc.png")
