# pass n_tokens and k in from engine instead of hardcoding
def svectorExplode(x):
	rowIndex, svector = x
	for featureIndex, avalue in zip(svector.indices, svector.values):
		yield rowIndex, featureIndex, float(avalue)

def pcaExplode(x):
	rowIndex, dvector = x
	for n, feat in enumerate(dvector):
		# yield rowIndex, n_tokens+n, float(feat)
		yield rowIndex, 279+n, float(feat)

def ldaExplode(x):
	rowIndex, dvector = x
	for n, feat in enumerate(dvector):
		# yield rowIndex, n_tokens+k+n, float(feat)
		yield rowIndex, 279+5+n, float(feat)
