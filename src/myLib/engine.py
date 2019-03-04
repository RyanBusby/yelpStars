
import pyspark as ps
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import IDF, PCA
from pyspark.ml import PipelineModel
from scipy.sparse import csr_matrix
import tensorflow as tf

from funcLib import *

class Engine():
	def __init__(self, sc, m, k):
		self.spark = ps.sql.SparkSession(sc)
		self.k = k
		self.lda = PipelineModel.load(
			'../../data/models/%sk_ldamodel' % str(m/1000)
		)
		self.pca = PipelineModel.load(
			'../../data/models/%sk_pcamodel' % str(m/1000)
		)
		self.idf = PipelineModel.load(
			'../../data/models/%sk_tfidfmodel'%str(m/1000)
		)
		checkpoint_path = "../../data/models/tensorflow/cp.ckpt"
		self.n = len(self.lda.stages[0].vocabulary) + (2*self.k)
		self.mlp = tf.keras.models.Sequential(
			[
				tf.keras.layers.Dense(
					int(self.n*1.1), input_shape=(self.n,)
				),
				tf.keras.layers.Dropout(0.2),
				tf.keras.layers.Dense(self.n, activation=tf.nn.relu),
				tf.keras.layers.Dropout(0.2),
				tf.keras.layers.Dense(
					int(self.n/2), activation=tf.nn.relu
				),
				tf.keras.layers.Dropout(0.2),
				tf.keras.layers.Dense(10, activation=tf.nn.softmax)
			]
		)
		self.mlp.compile(
			optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
		)
		self.mlp.load_weights(checkpoint_path)

	def process(self, df):
		stop_words = set(stopwords.words('english'))
		wn = WordNetLemmatizer()
		ss = SnowballStemmer('english')
		intab = r'''~!@#$%^&*()_+=-[]\{}|;':",./<>?'''
		outtab = '`' * len(intab)
		trantab = str.maketrans(intab, outtab)
		cols = ['id', 'text']
		stemdf = self.spark.createDataFrame(
			df.rdd.map(
				lambda x: (
					x[0],
					[
						wn.lemmatize(ss.stem(token))
						for token in x[1]\
							.translate(trantab)\
							.replace('`', '')\
							.split()\
						if token.lower() not in stop_words
					]
				)
			),
			cols
		)

		return stemdf

	def transformInput(self, r):
		df = self.process(self.spark.createDataFrame([(0, r)]))
		lda_rdd= self.lda.transform(df).select('id','topicDistribution')\
		.rdd.flatMap(ldaExplode)
		pca_rdd = self.pca.transform(df).select('id','pca_features')\
		.rdd.flatMap(pcaExplode)
		idf_rdd = self.idf.transform(df).select('id', 'tfidf')\
		.rdd.flatMap(svectorExplode)
		X = lda_rdd.union(pca_rdd.union(idf_rdd))

		shape = (1, self.n)
		row_indices, col_indices, data = zip(*X.collect())
		mat = csr_matrix((data, (row_indices, col_indices)), shape=shape)
		return self.mlp.predict_classes(mat)[0]

if __name__ == '__main__':
	sc = ps.SparkContext('local[5]')
	k = 5
	m = 1000000 # change hard coded funclib accordingly
	eng = Engine(sc, m, k)
	r = """I have come here pretty frequently since moving to SF. I will admit my first time in I did freak out over the high prices- but I have to remind myself these store owners are just as subject to high rent as the rest of us. They gotta make a profit too! For a corner market, I'm actually pretty impressed with their selection...down to the gluten free cookies! They really have everything one would need to make a pretty decent home cooked meal- produce, grains, bread, some packaged meat like sausage. Good quality stuff too! Not just ritz crackers, Campbell soup, and dusty condoms like a typical corner market lol. I read a lot of reviews about the bad attitude but I've personally never experienced it. In fact, everyone there has been really nice!"""
	print(eng.transformInput(r))
