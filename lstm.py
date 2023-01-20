import numpy as np
import tensorflow as tf

from bert import AUTOTUNE, batch_size, data_train, data_val, plot_loss_accuracy, seed

# ========== Hyper-parameters ==========
epochs = 5
lr = 1e-3
vocab_size = 5000

# ========== Preprocessing ==========
encoder = tf.keras.layers.TextVectorization(max_tokens = vocab_size)
encoder.adapt(data_train.map(lambda text, sent: text))
vocab = np.array(encoder.get_vocabulary())


# ========== Model construction ==========
class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embed = tf.keras.layers.Embedding(
            input_dim = len(encoder.get_vocabulary()), output_dim = 64,
            # Use masking to handle the variable sequence lengths
            mask_zero = True
        )
        self.blstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.blstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.d1 = tf.keras.layers.Dense(64, activation = "relu")
        self.d2 = tf.keras.layers.Dense(64, activation = "relu")
        self.out = tf.keras.layers.Dense(1, activation = "sigmoid")
        self.reshape = tf.keras.layers.Reshape((128, 1), input_shape = (128,))

    def call(self, x):
        x = encoder(x)
        x = self.embed(x)
        x = self.blstm1(x)
        x = self.reshape(x)
        x = self.blstm2(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)


if __name__ == "__main__":
    # ========== Training ==========
    print("Session Initiated.")

    model = LSTM()
    adam_opt = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer = adam_opt, loss = "binary_crossentropy", metrics = ["binary_accuracy"])
    hist = model.fit(data_train.take(1), validation_data = data_val.take(1), epochs = epochs)
    # model.save("output/twitter_lstm", include_optimizer = False)

    print("Session Terminated.")
    # ========== Visualization ==========
    plot_loss_accuracy(hist.history, "output/loss_acc_lstm.png")
