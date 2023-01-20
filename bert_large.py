import json
import os

from bert import *

# ========== Hyper-parameters and global variables ==========
workers = ["localhost:10086", "localhost:12315"]
# Change the above port id's to your local server when replicating.
# Here 10086 is Robert's PC with RTX 3080 (10GB), and 12315 is Anne's PC with RTX 2070 Super (8GB).
# The total Graph Mem meets the 16GB requirement of training BERT-large in practice.
worker_num = len(workers)
batch_size_global = batch_size * worker_num
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {"worker": workers},
    "task": {"type": "worker", "index": 0}
    # Here we set 10086 (workers[0]) as master node. Please change the environment variable on each device accordingly.
})
checkpoint_dir = "/content/bert_large_ckpts"  # Save checkpoints as unexpected errors could happen

# ========== Importing pre-trained BERT-large encoder ==========
tfhub_handle_encoder_large = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4"
encoder_large = hub.KerasLayer(tfhub_handle_encoder_large, trainable = True, name = "BERT_encoder")
encode_output_large = encoder_large(prep_text)
# Here `prep_text` is directly imported from bert.py to avoid code repetition.
# If solely running bert_large on local host without dependency on bert.py, please copy the data preprocessing part
# from bert.py. Also change the data transformation part below.

# Transform dataset with batch size 32 (on single node) to 64 (on two nodes parallel)
data_train = data_train.unbatch().batch(batch_size_global).prefetch(buffer_size = AUTOTUNE)
data_val = data_val.unbatch().batch(batch_size_global).prefetch(buffer_size = AUTOTUNE)


# ========== Model construction ==========
class BERTLarge(tf.keras.Model):
    """
    BERT-large Model
    """
    def __init__(self):
        super(BERTLarge, self).__init__()
        self.d1 = tf.keras.layers.Dense(32, activation = "relu")
        self.d2 = tf.keras.layers.Dense(1, activation = "sigmoid", name = "classifier")
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.bert = tf.keras.Model(text_input, encode_output_large["pooled_output"])

    def call(self, x, training = False):
        x = self.bert(x)
        x = self.d1(x)
        if training:
            x = self.dropout(x)

        return self.d2(x)


if __name__ == "__main__":
    """
    Execute ONLY when the cluster of workers are set up as above.
    """
    # Model distribution across workers
    print("Session Initiated.")

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    # Here we used default `AUTO` communication across device, which is Nvidia NCCL library.
    with strategy.scope():
        model = BERTLarge()
        adam_opt = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer = adam_opt, loss = "binary_crossentropy", metrics = ["binary_accuracy"])

    # Checkpoints set-up and rolling-back if error encountered
    print(f"Training model with {tfhub_handle_encoder_large}")
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_dir)]
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None:
        print("Loading weights from", latest)
        model.load_weights(latest)
    else:
        print("Checkpoint not found. Starting from scratch.")

    # ========== Training ==========
    hist = model.fit(data_train, validation_data = data_val, epochs = epochs, callbacks = callbacks)
    model.save("/output/twitter_bert_large", include_optimizer = False)

    print("Session Terminated.")

    # ========== Visualization ==========
    plot_loss_accuracy(hist.history, "/output/loss_acc_large.png")
