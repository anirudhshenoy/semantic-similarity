import tensorflow as tf 
import tensorflow_hub as hub 
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from nltk import word_tokenize
from fuzzywuzzy import fuzz
import math
#from collections import Counter
#from pymagnitude import Magnitude
#glove = Magnitude("../nlp-framework/vectors/glove.twitter.27B.100d.magnitude")
#embed = hub.load('4')

PRIMARY_THRESHOLD = 0.9
SECONDARY_THRESHOLD = 0.7
SIMILARITY_THRESHOLD = 0.7
WEIGHT = 0.9


def pipeline_avg_glove(text):
    return np.average(glove.query(word_tokenize(text)), axis = 0)

def similarity(sent_enc_rep):
    rows = []
    for _, outer_data in tqdm(sent_enc_rep.items()):
        rows.append(predict(outer_data['utterances'][0], sent_enc_rep, return_res = True))
    row_counter = 0 
    file = open("similarity.txt","w") 
    for _, data in tqdm(sent_enc_rep.items()):
        file.write(data['utterances'][0] + ': \n')
        for conf_uttr in rows[row_counter]:
            if conf_uttr['Utterance'] == data['utterances'][0]:
                continue
            elif conf_uttr['Confidence'] > SIMILARITY_THRESHOLD:
                file.write(conf_uttr['Utterance'] + '  ')
                file.write(str(conf_uttr['Confidence']))
                file.write('\n')
        row_counter += 1 
        file.write('\n\n')
    file.close()


def train(data, embed, uttr_col = 'Utterance', intent_col = 'Intent'):
    print('Training....')
    sent_enc_rep = {}
    for intent in tqdm(data[intent_col].unique()):
        vecs = []
        for utterance in data[data.Intent == intent][uttr_col].values:
            vecs.append(embed([utterance]))
        sent_enc_rep[intent] = {
            'sent_vector' : np.average(np.array(vecs), axis = 0)[0],
            'utterances' : data[data.Intent == intent][uttr_col].values
        }
    return sent_enc_rep

def predict(user_input, sent_enc_rep, embed, return_res = False, PRIMARY_THRESHOLD = 0.9, SECONDARY_THRESHOLD = 0.7, WEIGHT = 0.9):
    #input_embedding = pipeline_avg_glove(user_input)
    input_embedding =  embed([user_input])
    results = []
    for rep_intent, rep_data in sent_enc_rep.items():
        cos_dist = np.inner(rep_data['sent_vector'], input_embedding)
        fuzz_score = max([fuzz.token_sort_ratio(user_input, x)/100 for x in rep_data['utterances']])
        results.append({
            'Utterance' : rep_data['utterances'][0],
            'Confidence': (WEIGHT * cos_dist) + ((1-WEIGHT) * fuzz_score)
        })
    if return_res:
        return results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by = ['Confidence'], ascending = False)
    if results_df.Confidence.values[0] > PRIMARY_THRESHOLD:
        print('Journey:')
        print(results_df.Utterance.values[0] + ' : ' + str(results_df.Confidence.values[0]))
    else:
        print('Did you mean :')
        for utterance, confidence in zip(results_df.Utterance.values, results_df.Confidence.values):
            if confidence > SECONDARY_THRESHOLD:
                print(utterance)
            else:
                break
        print('\n')

def preprocess(data):
    data.drop(['Tagging'], inplace = True, axis = 1)
    data = data.fillna('NAN')
    rows = []
    for title, intent, utterances in tqdm(data.values):
        #title = title.astype('str')
        if title != 'NAN':
            title = title.replace('\u200b', '')
            title = title.replace('\n', '')
            rows.append([title, intent])
        for utterance in utterances.split('|'):
            utterance = utterance.replace('\u200b', '')
            utterance = utterance.replace('\n', '')

            rows.append([utterance, intent])

    data_mod = pd.DataFrame(rows, columns = ['Utterance', 'Intent'])
    data_mod.to_csv('itd_preprocessed.csv')
    return data_mod 


if __name__ == '__main__':
    embed = hub.load('4')

    data = pd.read_csv('itd_preprocessed.csv')
    data = data.dropna()
    #print(preprocess(data))
        #test_set[intent] = utterances
    
    sent_enc_rep = train(data, embed)
    #similarity(sent_enc_rep)
    while True:
        user_input = input()
        predict(user_input, sent_enc_rep, embed)
   


def test_split ():
    train_split = 0
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