import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np

# Load Data
train_df = pd.read_csv("../common/data/train.csv")
test_df = pd.read_csv("../common/data/test.csv")
validation_df = pd.read_csv("../common/data/valid.csv")
y_test = np.zeros(len(test_df))

def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples



class TFIdfPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values, data.Utterance.values))

    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]


if __name__ =="__main__":
    # Evaluate TFIDF predictor
    pred = TFIdfPredictor()
    pred.train(train_df)
    y = [pred.predict(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
    for n in [1, 2, 5, 10]:
        print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y, y_test, n)))
