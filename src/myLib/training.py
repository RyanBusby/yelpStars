
import pyspark as ps
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import IDF, PCA
from pyspark.ml import Pipeline
from scipy.sparse import csr_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split

from modelPerformanceHeatmap import makeHeatMap

def svectorExplode(x):
    rowIndex, svector = x
    for featureIndex, avalue in zip(svector.indices, svector.values):
        yield rowIndex, 1+featureIndex, float(avalue)

def pcaExplode(x):
    rowIndex, dvector = x
    for n, feat in enumerate(dvector):
        yield rowIndex, 1+num_tokens+n, float(feat)

def ldaExplode(x):
    rowIndex, dvector = x
    for n, feat in enumerate(dvector):
        yield rowIndex, 1+num_tokens+k+n, float(feat)

def getMAT(rdd):
    shape = (m, 1+num_tokens+(2*k))
    # https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/hadoop.md
    row_indices, col_indices, data = zip(*rdd.collect())
    mat = csr_matrix((data, (row_indices,col_indices)), shape=shape)
    return mat

def process(df):
    stop_words = set(stopwords.words('english'))
    wn = WordNetLemmatizer()
    ss = SnowballStemmer('english')
    intab = r'''~!@#$%^&*()_+=-[]\{}|;':",./<>?'''
    # intab = r'''~!@#$%^&*()_+=-[]\{}|;':",./<>?¡™£¢∞§¶•ªº–≠«»‘’“”πøˆ¨¥†®´∑œ©∆˚¬…æΩ≈ç√˜µ≤≥÷'''
    outtab = '`' * len(intab)
    trantab = str.maketrans(intab, outtab)
    cols = ['id', 'text']
    stemdf = spark.createDataFrame(
        df.rdd.zipWithIndex().map(
            lambda x: (
                x[1],
                [
                    wn.lemmatize(ss.stem(token))
                    for token in x[0]['text']\
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

def fitPipeline(df):
    cv = CountVectorizer(
        inputCol='text', outputCol='features', maxDF=.9, minDF=.2
    )
    cv2 = CountVectorizer(
        inputCol='text', outputCol='features', maxDF=.95, minDF=.05
    )
    lda = LDA(k=k, seed=42, optimizer='em')
    idf = IDF(inputCol='features', outputCol='tfidf')
    pca = PCA(k=k, inputCol='tfidf', outputCol='pca_features')

    lda_pipeline = Pipeline(stages=[cv2, lda])
    pca_pipeline = Pipeline(stages=[cv2, idf, pca])
    tfidf_pipeline = Pipeline(stages=[cv, idf])
    print('fitting lda ... '.upper())
    lda_model = lda_pipeline.fit(df)
    print('fitting pca ... '.upper())
    pca_model = pca_pipeline.fit(df)
    print('fitting tfidf ... '.upper())
    tfidf_model = tfidf_pipeline.fit(df)

    num_tokens = len(lda_model.stages[0].vocabulary)
    print('number of tfidf features: {}'.upper().format(num_tokens))

    return lda_model, pca_model, tfidf_model, num_tokens

def flattenRDDs(df, lda, pca, tfidf):
    print('lda transforming ... '.upper())
    lda_rdd = lda.transform(df).select('id','topicDistribution')\
    .rdd.flatMap(ldaExplode)

    print('pca transforming ... '.upper())
    pca_rdd = pca.transform(df).select('id','pca_features')\
    .rdd.flatMap(pcaExplode)

    print('tfidf transforming ... '.upper())
    tfidf_rdd = tfidf.transform(df).select('id', 'tfidf')\
    .rdd.flatMap(svectorExplode)

    return lda_rdd, pca_rdd, tfidf_rdd

def makePipelines(df):
    labels_rdd = df.select('stars').rdd.zipWithIndex().map(
        lambda x: (x[1], 0, x[0]['stars'])
    )
    print('processing reviews ... '.upper())
    df = process(df)
    global num_tokens
    lda, pca, idf, num_tokens = fitPipeline(df)
    lda.save('../../data/models/%sk_ldamodel' % str(int(m/1000)))
    pca.save('../../data/models/%sk_pcamodel' % str(int(m/1000)))
    idf.save('../../data/models/%sk_tfidfmodel'%str(int(m/1000)))

    lda_rdd, pca_rdd, tfidf_rdd = flattenRDDs(df, lda, pca, idf)
    X = labels_rdd.union(lda_rdd.union(pca_rdd.union(tfidf_rdd)))
    return X

def getMLP(X):
    y = X[:,0]
    X = X[:,1:]
    n = X.shape[1]
    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state=42)
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(int(n*1.1), input_shape=(n,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(int(n/2), activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ]
    )
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    checkpointpath='../../data/models/tensorflow/cp.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpointpath, save_weights_only=True, verbose=1
    )
    model.fit(X_train, y_train, epochs=10, callbacks=[cp_callback])
    fname = '../../data/%s_performance.png' % str(int(m/1000))
    y_preds = model.predict_classes(X_test)
    y_test = y_test.toarray().reshape(y_preds.shape)
    makeHeatMap(y_test, y_preds, k, fname)

if __name__ == '__main__':
    spark = ps.sql.SparkSession(ps.SparkContext('local[8]'))
    k = 5
    m = 110011
    print('reading reviews.json into spark ... '.upper())
    X = spark.read.json('../../data/reviews.json').select(
        'review_id', 'text', 'stars'
    ).limit(m)
    X = makePipelines(X)
    # print('collecting rows ... '.upper())
    print('collecting {} rows ... '.upper().format(X.count()))
    X = getMAT(X) # save npz for mlp tuning
    print('start training mlp ... '.upper())
    getMLP(X)
