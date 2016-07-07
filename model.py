import pandas as pd
import numpy as np
import cPickle as pickle
from  sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

# PARENT / MAIN FUNCTION
#
# reads in train data
# runs through other functions and builds model
def build_model():
    df = pd.read_json('path/to/data.json')
    df = feature_engineer(df)
    X, y = build_X_and_y(df)
    # build Random Forest
    model = RandomForestClassifier(n_estimators=50,oob_score=True)
    model.fit(X, y)
    pickle_that_model(model)

# FUNC: pickle the model!
#
# input: model
# output: none
def pickle_that_model(model):
    with open("path/to/pickle.pkl", 'w') as f:
        pickle.dump(model, f)

# FUNC: feature engineer multiple features in an attempt to improve the model
#
# input: raw dataframe
# output: dataframe with new features included
def feature_engineer(dataframe):
    # count NaN's, add column
    temporary_df = dataframe.replace(r'\s+( +\.)|#',np.nan,regex=True).replace('',np.nan)
    missin_list=[]
    for i, j in enumerate(temporary_df.count(1)):
        missin_list.append((temporary_df.shape[1])-j)
    dataframe['missing_count'] = missin_list

    # fill NaN's
    dataframe.fillna(0,inplace=True)

    # create labels
    dataframe['label'] = np.where((dataframe['acct_type']=='fraudster_event') | (dataframe['acct_type']=='fraudster') , 1, 0)

    # create uppercase percentage metric
    dataframe['uprcse_ltrs'] = map(lambda message: sum(1 for c in message if c.isupper()), dataframe['name'])
    dataframe['name_len'] = map(lambda message: len(message), dataframe['name'])
    dataframe['uprcse_percent'] = dataframe['uprcse_ltrs'] / dataframe['name_len']
    dataframe.uprcse_percent.fillna(np.mean(dataframe.uprcse_percent),inplace=True)

    # create name_length columns
    dataframe['no_name'] = np.where(dataframe['name_length']==0, 1, 0)

    # location feature engineering
    dataframe['event_duration'] = abs(dataframe['event_start'] - dataframe['event_end'])
    dummies = pd.get_dummies(dataframe['venue_state'])
    dataframe['loc'] = dummies.values.argmax(axis=1)

    return dataframe

# FUNC: build X and y
#
# input: dataframe
# output: X dataframe and y dataframe
def build_X_and_y(dataframe):
    X = dataframe[['venue_latitude','venue_longitude','no_name','uprcse_percent','gts','channels','user_type','user_age','has_logo','body_length','name_length','org_twitter','org_facebook','event_created','event_published','event_end','event_start','event_duration','loc','missing_count']]
    y = dataframe[['label']]

    return X, y

if __name__ == '__main__':
    build_model()
