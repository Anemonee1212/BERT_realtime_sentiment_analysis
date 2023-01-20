# FIFA World Cup Twitter Sentiment Analysis
### EECS 6893 Group 18: Anne Wei, Robert Shi

## Supplementary files
Files > 100 MB cannot be committed to GitHub. Please access through https://drive.google.com/drive/folders/1CD2go0M8eqwxXQbx2WSAAGgZzNgvAUmZ?usp=sharing

`dataset/data.csv` is the raw dataset we used. Please integrate it into `dataset/` directory.

`saved_models/` folder contains BERT-base and BERT-large models. Please save them as-is in the root directory.

## Code walkthrough (files are ordered)
1. `exploratory_analysis.py`\
Perform exploratory, pre-model analysis mainly using Python `NLTK`.

2. `bert.py`\
Implement BERT-base model. Set up to be trained on 1 single device (with GPU Memory >= 6GB).\
__Note:__ This file can be seen as "`main.py`", on which other files depends. However, such dependency is not\
necessary, as long as required variables and functions are correctly re-assigned in other programs (e.g., `lstm.py`\
or `bert_large.py`). The dependency is established only to avoid code repetition for readability.

3. `lstm.py`\
Implement LSTM model using TensorFlow Keras.\
__Note:__ Removing the `from bert import *` dependency can potentially accelerate performance.

4. `bert_large.py`\
Implement BERT-large model. Needs to be trained on multiple workers configured on the same cluster (with total\
GPU Mem >= 16GB).\
__Note:__ It is strongly recommended that the dependency on `bert.py` is removed. Otherwise `bert.py` has to be\
initiated on both devices too.
    #### Important Note: The model is partitioned and distributed across devices. The dataset is not distributed.
    #### Please make sure that each device access the same dataset and runs the same code (except for configs).
    #### Please change the environment variables accordingly. More details see comments in `bert_large.py`.

5. `scrape.ipynb`\
A sample code for you to have an overview of how static, historical Twitter data is scraped using `tweepy`.\
Please do not simply hit "Run All".

6. `twitter_http_client.ipynb` and `spark_streaming.ipynb`\
The code is primarily based on the homework assignment. Please deploy it on GCP, and run\
`twitter_http_client` before `spark_streaming`.\
__Note:__ The code to deploy BERT model is provided but commented, because it is computationally demanding to\
to real-time prediction without proper parallelization. We suggest using GCP nodes with high memory when trying\
to replicate.

7. `app.py`\
Implement an interactive front-end visualization using `flask` and `plotly`.\
Execute with `python -m flask run` in Terminal.

8. `final_analysis.ipynb`\
Perform multiple analyses which generate insights about public sentiment to FIFA 2022.

9. `scratch.ipynb` and `test.py`\
The scratch papers for team members to run temporary code. Please ignore them.

## Environment setup (conda recommended)
`nltk==3.7`, `tweepy==4.12.1` (`conda-forge` channel), `flask==2.1.3`, `plotly==5.9.0`, and other required\
packages including `pandas`, `matplotlib`, etc.

#### Important: TensorFlow environment for this project is a little tricky, because:

- Training and deploying pre-trained BERT model from TensorFlow Hub requires `tensorflow_text>=2.8` (install\
with `pip install -q -U "tensorflow-text==2.8.*"`).
- `tensorflow_text>=2.8` requires `tensorflow>=2.8`, and potentially `keras>=2.8`.
- However, there has been issues with TensorFlow and Keras when using the newest versions, especially when\
their version does not match, e.g., `tensorflow==2.8.4` while `keras==2.8.0`.
- My solution on my local device is to first install TensorFlow with new versions, based on which we install\
`tensorflow_text==2.8.0`, and finally install the GPU version of Keras with earlier stable versions, e.g.,\
`conda install -c anaconda keras-gpu=2.6.0`.
- My solution on GCP is to first install TensorFlow with earlier versions like 2.6.0, and then directly call\
`import tensorflow as tf` in Jupyter Notebook. In this context, upon installing TensorFlow Text later, the version\
of TensorFlow will be "locked" at 2.6.0. This wierd action __will__ cause conflicts between version dependencies,\
but the `import tensorflow_text as text` package is actually never used when deploying the model -- it just\
needs to be imported. Thus, the rest of the code can run successfully. (Please set the Linux OS to Debian 1.5, so\
that Anaconda can be enabled in GCP Gateway.)
