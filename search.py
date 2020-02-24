import tensorflow as tf 
import tensorflow_hub as hub 
import numpy as np 
import pandas as pd 
from collections import Counter
from tqdm import tqdm
from pymagnitude import Magnitude
from nltk import word_tokenize
glove = Magnitude("../nlp-framework/vectors/glove.twitter.27B.100d.magnitude")


def pipeline_avg_glove(text):
    return np.average(glove.query(word_tokenize(text)), axis = 0)

if __name__ == '__main__':

    embed = hub.load('4')
    data = pd.read_csv('dataset/training-set.csv')
    counts = Counter(data.Intent.values)

    filtered_intents = [text for text, count in counts.items() if count > 4]

    representations = {}
    test_set = {}
    data = data[data['Intent'].isin(filtered_intents)]
    for intent in tqdm(filtered_intents):
        temp = data[data['Intent'] == intent]
        vecs = []
        for utterances in temp.Utterance.values[:-1]: 
            #vecs.append(pipeline_avg_glove(utterances))
            vecs.append(embed([utterances]))
        vecs = np.average(np.array(vecs), axis = 0)
        representations[intent] = vecs
        test_set[intent] = utterances

    rows = []
    for test_label, test_utterance in tqdm(test_set.items()):
        results = {}
        #test_embedding = pipeline_avg_glove(test_utterance)
        test_embedding = embed([test_utterance])
        for rep_intent, rep_vector in representations.items():
            results[rep_intent] = np.inner(rep_vector, test_embedding)
        
        rows.append([test_utterance, test_label, max(results, key=results.get)])

    result_df = pd.DataFrame(rows)
    result_df.columns = ['utterance', 'true', 'predict']

    print(sum(result_df['true'].values == result_df['predict'].values))