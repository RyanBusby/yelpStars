import sys
sys.path.append('../myLib')

import tensorflow as tf
import pyspark as ps
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import IDF, PCA
from pyspark.ml import PipelineModel
from scipy.sparse import csr_matrix
from flask import Blueprint, Flask, render_template, request

from engine import Engine

app = Flask(__name__)
sc = ps.SparkContext('local[8]')
sc.addFile('../myLib/engine.py')
sc.addFile('../myLib/funcLib.py')

m = 100000
k = 5
eng = Engine(sc, m, k)

@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')

@app.route('/predictStars', methods=["POST"])
def predictStars():
	r = str(request.form['review'])
	pred = eng.transformInput(r)
	stars = 'static/%s.png' % pred
	return render_template('results.html', png=stars)

if __name__== '__main__':
	app.run(threaded=False)
