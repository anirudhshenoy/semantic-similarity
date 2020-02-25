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
    train_split = 0
    PRIMARY_THRESHOLD = 0.9
    SECONDARY_THRESHOLD = 0.7
    SIMILARITY_THRESHOLD = 0.7




    embed = hub.load('4')
    data = pd.read_csv('dataset/flipkart.csv')
    counts = Counter(data.Intent.values)

    filtered_intents = [text for text, count in counts.items() if count > 0]

    representations = {}
    test_set = {}
    data = data[data['Intent'].isin(filtered_intents)]
    for intent in tqdm(filtered_intents):
        temp = data[data['Intent'] == intent]
        vecs = []
        #for utterances in temp.Utterance.values[:-train_split]: 
        for utterances in temp.Utterance.values: 
            #vecs.append(pipeline_avg_glove(utterances))
            vecs.append(embed([utterances]))
        vecs = np.average(np.array(vecs), axis = 0)
        representations[intent] = {
            'vector' : vecs,
            'utterances' : temp.Utterance.values#[:-train_split]
        }
        #test_set[intent] = utterances

    weight = 0.9
    """
    rows = []
    for test_label, test_utterance in tqdm(test_set.items()):
        results = {}
        results_fuzz = {}
        results_weights = {}

        #test_embedding = pipeline_avg_glove(test_utterance)
        test_embedding = embed([test_utterance])
        for rep_intent, rep_data in representations.items():
            results[rep_intent] = np.inner(rep_data['vector'], test_embedding)
            results_fuzz[rep_intent] = max([fuzz.token_set_ratio(test_utterance, x)/100 for x in rep_data['utterances']])
            results_weights[rep_intent] = (weight * results[rep_intent]) + ((1-weight) * results_fuzz[rep_intent])

        
        rows.append([test_utterance, test_label, max(results, key=results.get),  max(results_fuzz, key=results_fuzz.get), max(results_weights, key=results_weights.get)])
    result_df = pd.DataFrame(rows)
    result_df.columns = ['utterance', 'true', 'predict', 'predict_fuzz', 'predict_weight']
    print(sum(result_df['true'].values == result_df['predict_weight'].values)/result_df.shape[0])
    """

    rows = []
    for _, outer_data in tqdm(representations.items()):
        row = []
        for _, inner_data in representations.items():
            cos_dist = np.inner(inner_data['vector'], outer_data['vector'])
            fuzz_score = max([fuzz.token_sort_ratio(outer_data['utterances'][0], x)/100 for x in inner_data['utterances']])
            row.append({
                'utterance' : inner_data['utterances'][0],
                'confidence': (weight * cos_dist) + ((1-weight) * fuzz_score)
                })
        rows.append(row)


    row_counter = 0 
    file = open("similarity.txt","w") 

    for _, data in tqdm(representations.items()):
        file.write(data['utterances'][0] + ': \n')
        for conf_uttr in rows[row_counter]:
            if conf_uttr['utterance'] == data['utterances'][0]:
                continue
            elif conf_uttr['confidence'] > SIMILARITY_THRESHOLD:
                file.write(conf_uttr['utterance'] + '  ')
                file.write(str(conf_uttr['confidence']))
                file.write('\n')
        row_counter += 1 
        file.write('\n\n')
    file.close()


"""
    while True: 
        print('\nEnter Input: ')
        user_input = input()
        #input_embedding = pipeline_avg_glove(user_input)
        input_embedding =  embed([user_input])
        results = []
        for rep_intent, rep_data in representations.items():
            cos_dist = np.inner(rep_data['vector'], input_embedding)
            fuzz_score = max([fuzz.token_sort_ratio(user_input, x)/100 for x in rep_data['utterances']])
            results.append({
                'Utterance' : rep_data['utterances'][0],
                'Confidence': (weight * cos_dist) + ((1-weight) * fuzz_score)
            })

        results = pd.DataFrame(results)
        results = results.sort_values(by = ['Confidence'], ascending = False)
        #max_values = sorted(results_weights.values())
        #print('Journey: ' + str(max(results_weights, key=results_weights.get)))
        #print('Confidence: ' +str(max(results_weights.values())))
        #print('\n')
        if results.Confidence.values[0] > PRIMARY_THRESHOLD:
            print(results.Utterance.values[0])
        else:
            print('Did you mean :')
            for utterance, confidence in zip(results.Utterance.values, results.Confidence.values):
                if confidence > SECONDARY_THRESHOLD:
                    print(utterance)
                else:
                    break
            print('\n')
"""