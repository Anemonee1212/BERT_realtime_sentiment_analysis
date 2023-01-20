import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

nltk.download('vader_lexicon')
stop_words = stopwords.words("english")

data = pd.read_csv("dataset/data.csv", names = ["sent", "id", "time", "flag", "user", "text"])
print(data.head(10))

# ========== Data cleaning ==========
data.drop(columns = ["id", "time", "flag", "user"], inplace = True)  # Unused columns
data.sent = data.sent > 0  # Parse sentiment into boolean

# Check how many chars and words in each tweet
data["num_char"] = data.text.str.len()
print(data.num_char.mean())
data["num_word"] = data.text.map(lambda t: len(word_tokenize(t)))
print(data.num_word.mean())


def remove_at_hyperlink(s):
    s = re.sub(r"https?:\/\/\S*\s?", "", s)
    s = re.sub(r"@\w+", "", s)
    return s


def preprocess_sentence(s):
    words = word_tokenize(s)
    words = [w.lower() for w in words if w.isalpha()]
    words = [w for w in words if w not in stop_words]
    return words


data.text = data.text.map(remove_at_hyperlink)
data["words"] = data.text.map(preprocess_sentence)

# ========== Plot the first visualization (diagonal) ==========
data_words = data.drop(columns = ["text"]).explode("words").reset_index(drop = True)
data_word_count = data_words.groupby(["sent", "words"]).size().reset_index(name = "count")

pos_word_count = data_word_count[data_word_count.sent]
pos_word_count.drop(columns = ["sent"], inplace = True)
pos_word_count.rename(columns = {"count": "pos_count"}, inplace = True)
pos_word_count.set_index(keys = "words", inplace = True)
neg_word_count = data_word_count[np.logical_not(data_word_count.sent)]
neg_word_count.drop(columns = ["sent"], inplace = True)
neg_word_count.rename(columns = {"count": "neg_count"}, inplace = True)
neg_word_count.set_index(keys = "words", inplace = True)

pos_neg_word_count = pos_word_count.join(neg_word_count, how = "outer")
pos_neg_word_count.fillna(value = 1, inplace = True)
pos_neg_word_count["log_pos_count"] = np.log(pos_neg_word_count["pos_count"])
pos_neg_word_count["log_neg_count"] = np.log(pos_neg_word_count["neg_count"])
pos_neg_word_count = pos_neg_word_count[pos_neg_word_count["log_pos_count"] + pos_neg_word_count["log_neg_count"] >= 14]
pos_neg_word_count["log_odds"] = np.log(pos_neg_word_count["pos_count"] / pos_neg_word_count["neg_count"])

plt.style.use("ggplot")
plt.scatter(
    pos_neg_word_count.log_neg_count, pos_neg_word_count.log_pos_count,
    marker = ".", c = -pos_neg_word_count.log_odds, cmap = "coolwarm", alpha = 0.7
)

x = np.linspace(6, 12, 10)
plt.plot(x, x, "--", c = "black", linewidth = 1)
plt.xlim(6, 11)
plt.ylim(6, 11)
plt.xlabel("ln(Negative word count)")
plt.ylabel("ln(Positive word count)")
plt.show()

for word in pos_neg_word_count.index:
    if pos_neg_word_count.log_odds[word] >= 1.5:
        plt.annotate(
            word, (pos_neg_word_count.log_neg_count[word], pos_neg_word_count.log_pos_count[word]),
            c = "blue", alpha = 0.5
        )
    elif pos_neg_word_count.log_odds[word] <= -1.5:
        plt.annotate(
            word, (pos_neg_word_count.log_neg_count[word], pos_neg_word_count.log_pos_count[word]),
            c = "red", alpha = 0.5
        )


# ========== Plot the second visualization (Vader) ==========
sid = SentimentIntensityAnalyzer()

data.drop(columns = ["words"], inplace = True)
data["vader_score"] = data.text.map(lambda s: sid.polarity_scores(s)["compound"])
data.sort_values("vader_score", inplace = True)
data.reset_index(inplace = True, drop = True)
data["color"] = "red"
data["color"][data.sent] = "blue"

data0 = data[data.vader_score != 0]
print(data.sent.eq(data0.vader_score > 0).mean())
data1k = data.sample(frac = 1 / 1000).sort_index()

plt.scatter(data1k.index, data1k.vader_score, marker = ".", c = data1k.color, alpha = 0.1)
plt.xticks([])
plt.ylabel("Vader score")
plt.show()
