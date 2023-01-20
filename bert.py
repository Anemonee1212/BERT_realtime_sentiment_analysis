# EECS 6893 Project
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from sklearn.model_selection import train_test_split

__author__ = ["Anne Wei", "Robert Shi"]

# ========== Hyper-parameters ==========
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
epochs = 5
lr = 3e-5
seed = 3407

# ========== Reading dataset ==========
data = pd.read_csv("dataset/data.csv", names = ["sent", "id", "time", "flag", "user", "text"])

# ========== Data cleaning ==========
data.drop(columns = ["id", "time", "flag", "user"], inplace = True)
data.text = data.text.str.replace(r"https?:\/\/\S*\s?", "", regex = True)
data.text = data.text.str.replace(r"@\w+", "", regex = True)
data.sent = data.sent > 0

x_train, x_val, y_train, y_val = train_test_split(data.text, data.sent, test_size = 0.2, random_state = seed)
data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train.astype("int")))
data_train = data_train.batch(batch_size).prefetch(buffer_size = AUTOTUNE)
data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val.astype("int")))
data_val = data_val.batch(batch_size).prefetch(buffer_size = AUTOTUNE)

# ========== Preprocessing ==========
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = "text")
preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name = "preprocessing")
prep_text = preprocessing_layer(text_input)


# ========== Model construction ==========
class BERTClassifier(tf.keras.Model):
    """
    Bidirectional Encoder Representations from Transformers
    """
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.d1 = tf.keras.layers.Dense(32, activation = "relu")
        self.d2 = tf.keras.layers.Dense(1, activation = "sigmoid", name = "classifier")
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.bert = tf.keras.Model(text_input, encode_output["pooled_output"])

    def call(self, x, training = False):
        x = self.bert(x)
        x = self.d1(x)
        if training:
            x = self.dropout(x)

        return self.d2(x)


# ========== Helper function ==========
def plot_loss_accuracy(history_dict, save_path):
    """
    Make 2 plots of 1) Training and validation loss, 2) Training and validation accuracy
    """
    fig = plt.figure(figsize = (10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(history_dict["loss"], label = "Training loss")
    plt.plot(history_dict["val_loss"], label = "Validation loss")
    plt.title("Training and validation loss")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history_dict["binary_accuracy"], label = "Training acc")
    plt.plot(history_dict["val_binary_accuracy"], label = "Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc = "lower right")

    plt.savefig(save_path)
    return 0


if __name__ == "__main__":
    """
    Main method. Execute when training BERT-base on single device.
    """
    # ========== Importing pre-trained BERT encoder ==========
    tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable = True, name = "BERT_encoder")
    encode_output = encoder(prep_text)

    # ========== Training ==========
    print("Session Initiated.")

    model = BERTClassifier()
    adam_opt = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer = adam_opt, loss = "binary_crossentropy", metrics = ["binary_accuracy"])
    print(f"Training model with {tfhub_handle_encoder}")
    hist = model.fit(data_train, validation_data = data_val, epochs = epochs)
    model.save("output/twitter_lstm", include_optimizer = False)

    print("Session Terminated.")

    # ========== Visualization ==========
    plot_loss_accuracy(hist.history, "output/loss_acc.png")
