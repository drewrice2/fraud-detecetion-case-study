import pandas as pd
import numpy as np
import cPickle as pickle
from model import feature_engineer as FE
from pymongo import MongoClient
# !
# establish MONGO connection by typing 'mongod' in the terminal
# !

def build_X(dataframe):
    '''
    FUNC: build X and y

    input: dataframe
    output: X dataframe
    '''

    feature_engineered = FE(dataframe)
    X = feature_engineered[['venue_latitude','venue_longitude','no_name','uprcse_percent','gts','channels','user_type','user_age','has_logo','body_length','name_length','org_twitter','org_facebook','event_created','event_published','event_end','event_start','event_duration','loc','missing_count']]
    # feature engineer and build X dataframe for prediction
    return X

def model_and_db(X):
    '''
    FUNC: unpickle the model, predict the label / generate probability score, and save the entry to MONGO db / collection

    input: X dataframe, after feature engineering and feature selection
    output: None
    '''

    # unpickle!
    with open('path/to/pickle.pkl') as f:
        model = pickle.load(f)

    # predict label and probability
    predicted_label = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    X['predict_label'] = predicted_label
    X['predicted_proba_not_fraud'] = probs[0]
    X['predicted_proba_fraud'] = probs[1]

    # conversion to dict / string data so MONGO can read in the dataframe
    X = X.to_dict()
    X = {str(k):str(v) for k,v in X.items()}

    # establish MONGO connection
    connection =  MongoClient()
    db = connection['fraud_preds']
    coll = db['preds']
    coll.insert(X)

if __name__ == '__main__':
    # read in local test data
    df = pd.read_csv('path/to/test/examples.csv')

    # run script
    X = build_X(df.sample(1))
    model_and_db(X)
