
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################
# 1. Text Preprocessing
##################################################

df = pd.read_csv("datasets/amazon_reviews.csv", sep=",")
print(df.head())

###############################
# Normalizing Case Folding
###############################

print( df['reviewText'].str.lower())

###############################
# Puncutations
###############################
# noktalama işaretlerini yok eder.

df['reviewText'] = df['reviewText'].str.replace(r'[^\w\s]', '', regex=True)
print(df['reviewText'])

###############################
# number
###############################
# sayıları yok eder.

df['reviewText'] = df['reviewText'].str.replace('\d', '', regex=True)
print(df['reviewText'])

###############################
# stopwords ( bağlaç, edat gibi ölçüm ifade etmeyen kelimler kaldırılıyor)
###############################
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # Bunu set olarak alırsan daha hızlı çalışır

# Stopwords'leri metinden çıkar
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(word for word in str(x).split() if word.lower() not in stop_words))

# Sonucu ekrana yazdır
print(df["reviewText"])


###############################
# rare words
###############################

temp_df = pd.Series(" ".join(df["reviewText"]).split()).value_counts()
print(temp_df)

drops = temp_df[temp_df <= 1]
print(drops)

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in x.split() if not x in drops))
print(df["reviewText"])


###############################
# Tokenization (cümleleri parçalara ayırma)
###############################

nltk.download('punkt_tab')

# Tokenization işlemi
df["tokenized_review"] = df["reviewText"].apply(lambda x: TextBlob(str(x)).words)

# Sonucu ekrana yazdır
print(df[["reviewText", "tokenized_review"]].head())

###############################
# lemmatization (kelimleri köklerine ayırma)
###############################
nltk.download('wordnet')

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(df["reviewText"])


###############################
# text viziualization
###############################

# text frekanslarına ayırma

tf = df["reviewText"].apply(lambda x: pd.value_counts(str(x).split(" ")))

# Toplam frekansı hesapla
tf = tf.sum(axis=0).reset_index()
tf.columns = ["words", "tf"]

tf = tf.sort_values(by="tf", ascending=False)
print(tf)

###############################
# bar plot
###############################

tf[tf["tf"] > 500].plot.bar(x ="words", y="tf")
plt.show()


###############################
# word cloud
###############################

text = " ".join(i for i in df.reviewText)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("worldCloud.png")

###############################
# şablonlara göre world cloud
###############################

tr_mask = np.array(Image.open("images/tr.png"))

wc = WordCloud(background_color="white", 
               max_words=1000, 
               mask=tr_mask,
               contour_width=3,
               contour_color="firebrick")

wc.generate(text)
plt.figure(figsize=[10,10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

###############################
# Sentiment Analizi (duygu)
###############################
# elde bulunna metnin duygu durumunu matematiksel olarak ortaya koymaktadır. Bir metnin pozitif mi negatif mi olduğunu gösterir.

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# Metnin duygu skorlarını al
score = sia.polarity_scores("the film was awesome")

# Sonucu yazdır
print(score)

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_score"] = df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])





###############################
# Sentiment Modeling
###############################

###############################
# feature Engineering
###############################



df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")



df["sentiment_label"].value_counts() 

df.groupby("sentiment_label")["overall"].mean()


df["sentiment_label"] =LabelEncoder().fit_transform(df["sentiment_label"])
print(df["sentiment_label"])

y = df["sentiment_label"]
x = df["reviewText"]

###############################
# count vectors
###############################

a = "n gramlar anlaşılabilmesi için daha uzun bir metin üzerinde test edilecek. N gramlar birlikte kullanılan kelimlerin kombinasyonlarını gösterir ve feature üretmek için kullanılır."

TextBlob(a).ngrams(3)

from sklearn.feature_extraction.text import CountVectorizer

corpus = ["this is the first document", "this is the second document", "and this is the third one", "is this is the fourth one"]

Vectorizer = CountVectorizer
X_c = Vectorizer.fit_transform(corpus)
Vectorizer.get_feature_names()
X_c.toarray()

Vectorizer2 = CountVectorizer(analyzer="word", ngram_range=(2,2))
X_n = Vectorizer2.fit_transform(corpus)
Vectorizer2.get_feature_names()
X_n.toarray()

Vectorizer = CountVectorizer()
X_count = Vectorizer.fit_transform(x)
Vectorizer.get_feature_names()[0:15]
X_count.toarray()[0:15]

# term frequency 
# t teriminin ilgili dökümandaki freakansı / dökümandaki toplam terim sayısı


from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer()
X_tfidf = tf_idf_word_vectorizer.fit_transform(x)

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tfidf_ngram = tf_idf_ngram_vectorizer.fit_transform(x)


############################################
#logistic regression
############################################

log_model = LogisticRegression().fit(X_tfidf,y)
cross_val_score(log_model,
                X_tfidf,
                y,
                scoring="accuracy",
                cv=5).mean()
