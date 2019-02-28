# 185 for the 100k model
# 184 for the 110011 model
def svectorExplode(x):
	rowIndex, svector = x
	for featureIndex, avalue in zip(svector.indices, svector.values):
		yield rowIndex, featureIndex, float(avalue)

def pcaExplode(x):
	rowIndex, dvector = x
	for n, feat in enumerate(dvector):
		yield rowIndex, 279+n, float(feat)

def ldaExplode(x):
	rowIndex, dvector = x
	for n, feat in enumerate(dvector):
		yield rowIndex, 279+5+n, float(feat)
