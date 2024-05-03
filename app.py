import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from flask import Flask, request, render_template

nltk.download('stopwords')
nltk.download('wordnet')

# cleaning the text

def cleantext(text):

    # removing the "\"
    text = re.sub("'\''", "", text)
    # removing special symbols
    text = re.sub("[^a-zA-Z]", " ", text)
    # removing the whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()
    return text


# removing the stopwords

def removestopwords(text):
    stop_words = set(stopwords.words('english'))
    removedstopword = [word for word in text.split() if word not in stop_words]
    return ' '.join(removedstopword)


# lemmatizing the text

def lemmatizing(text):
    lemma = WordNetLemmatizer()
    lem_sentence = ""
    for word in text.split():
        lem = lemma.lemmatize(word)
        lem_sentence += lem
        lem_sentence += " "
    lem_sentence = lem_sentence.strip()
    return lem_sentence


# stemming the text

def stemming(text):

    stemmer = PorterStemmer()
    stemmed_sentence = ""
    for word in text.split():
        stem = stemmer.stem(word)
        stemmed_sentence += stem
        stemmed_sentence += " "

    stemmed_sentence = stemmed_sentence.strip()
    return stemmed_sentence


def prediction(text, model, tfidf_vectorizer):

    text = cleantext(text)
    text = removestopwords(text)
    text = lemmatizing(text)
    text = stemming(text)
    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)

    newmapper = {0: 'Fantasy', 1: 'Science Fiction', 2: 'Crime Fiction',
                 3: 'Historical novel', 4: 'Horror', 5: 'Thriller'}

    return newmapper[predicted[0]]


# Loading the neural net(mlp classifier)
with open('bookgenre_net.pkl', 'rb') as file:
    SNet = pickle.load(file)

with open('tfidfvector.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict_genre():

    if request.method == 'POST':

        mydict = request.form
        text = mydict["summary"]
        pred = prediction(text, SNet, tfidf_vectorizer)

        return render_template('index.html', genre=pred, text=str(text)[:100], showresult=True)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
