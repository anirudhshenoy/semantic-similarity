import tensorflow as tf 
import tensorflow_hub as hub 
import numpy as np 
import pandas as pd 
from collections import Counter
from tqdm import tqdm
from pymagnitude import Magnitude
from nltk import word_tokenize
from fuzzywuzzy import fuzz
glove = Magnitude("../nlp-framework/vectors/glove.twitter.27B.100d.magnitude")


def pipeline_avg_glove(text):
    return np.average(glove.query(word_tokenize(text)), axis = 0)

if __name__ == '__main__':
    train_split = 1

    #embed = hub.load('4')
    data = pd.read_csv('dataset/training-set.csv')
    counts = Counter(data.Intent.values)

    filtered_intents = [text for text, count in counts.items() if count > 4]

    representations = {}
    test_set = {}
    data = data[data['Intent'].isin(filtered_intents)]
    for intent in tqdm(filtered_intents):
        temp = data[data['Intent'] == intent]
        vecs = []
        for utterances in temp.Utterance.values[:-train_split]: 
            vecs.append(pipeline_avg_glove(utterances))
            #vecs.append(embed([utterances]))
        vecs = np.average(np.array(vecs), axis = 0)
        representations[intent] = {
            'vector' : vecs,
            'utterances' : temp.Utterance.values[:-train_split]
        }
        test_set[intent] = utterances

    rows = []
    for test_label, test_utterance in tqdm(test_set.items()):
        results = {}
        results_fuzz = {}
        test_embedding = pipeline_avg_glove(test_utterance)
        #test_embedding = embed([test_utterance])
        for rep_intent, rep_data in representations.items():
            results[rep_intent] = np.inner(rep_data['vector'], test_embedding)
            results_fuzz[rep_intent] = max([fuzz.token_set_ratio(test_utterance, x) for x in rep_data['utterances']])
        
        #rows.append([test_utterance, test_label, max(results, key=results.get),  max(results_fuzz, key=results_fuzz.get)])

    results_weights = {}
    for weight in tqdm(np.arange(0,1.1, 0.1)):
        rows = []
        for test_label, test_utterance in test_set.items():
            for intent, cosine in results.items():
                results_weights[intent] = (weight * results[intent]) + ((1-weight) * results_fuzz[intent])

            rows.append([test_utterance, test_label, max(results_weights, key=results_weights.get)])
        result_df = pd.DataFrame(rows)
    #result_df.columns = ['utterance', 'true', 'predict', 'predict_fuzz']
        result_df.columns = ['utterance', 'true', 'predict']
        print(sum(result_df['true'].values == result_df['predict'].values)/result_df.shape[0])
    #print(sum(result_df['true'].values == result_df['predict_fuzz'].values)/result_df.shape[0])