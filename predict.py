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
    output: X dataframe, feature selection
    '''

    feature_engineered = FE(dataframe)
    X = feature_engineered[['venue_latitude','venue_longitude','no_name','uprcse_percent','gts','channels','user_type','user_age','has_logo','body_length','name_length','org_twitter','org_facebook','event_created','event_published','event_end','event_start','event_duration','loc','missing_count']]
    return X

def model_and_db(X, connection, model):
    '''
    FUNC: unpickle the model, predict the label / generate probability score, and save the entry to MONGO db / collection

    input:  X dataframe, after feature engineering and feature selection
            PyMongo connection object (MongoClient)
            model, unpickled model
    output: None
    '''

    # predict label and probability
    predicted_label = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    X['predict_label'] = predicted_label
    X['predicted_proba_not_fraud'] = probs[0]
    X['predicted_proba_fraud'] = probs[1]

    # conversion to dict / string data so MONGO can read in the dataframe
    X = X.to_dict()
    X = {str(k):str(v) for k,v in X.items()}

    # establish MONGO connection and write to the database
    db = connection['fraud_preds']
    coll = db['preds']
    coll.insert(X)
    cursor_query = coll.find({"predict_label": "{0: 1}"}, {"_id": True, "predicted_proba_fraud": True, "predicted_proba_not_fraud": True})
    return predicted_label, probs[0], probs[1], list(cursor_query)


def grab_fraud():
    '''
    FUNC: grab all instances of Fraud classification in the Mongo DB

    input: None
    output: query results
    '''
    cursor_query = coll.find({"predict_label": "{0: 1}"}, {"_id": True, "predicted_proba_fraud": True, "predicted_proba_not_fraud": True})
    return list(cursor_query)
