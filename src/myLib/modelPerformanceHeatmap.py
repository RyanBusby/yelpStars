import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')

def makeHeatMap(y_true, y_pred, k, fname):
    labels = [str(n) for n in range(1, k+1)]
    a = confusion_matrix(y_true, y_pred, labels=range(1, k+1))
    fig, ax = plt.subplots()
    im = ax.imshow(a, cmap='Blues')
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for i in range(k):
        for j in range(k):
            text = ax.text(
                j, i, a[i, j], ha="center", va="center", color="w"
            )
    ax.grid(False)
    ax.set_title('True Stars v. Predicted Stars')
    fig.tight_layout()
    plt.savefig(fname, dpi=300)
