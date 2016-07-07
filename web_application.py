from flask import Flask, request, render_template
import json
import requests
import socket
import time
from datetime import datetime
import cPickle as pickle
from pandas.io.json import json_normalize
from predict import build_X, model_and_db, grab_fraud
from pymongo import MongoClient
import pandas as pd
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# unpickle the model
with open('path/to/pickle.pkl') as f:
    model = pickle.load(f)

# establish MongoDB connection
connection =  MongoClient()

# initialize Flask
# receive data from server
app = Flask(__name__)
PORT = 5003
REGISTER_URL = "http://IP.Address/register"
DATA = []
TIMESTAMP = []

# score page
@app.route('/score', methods=['POST'])
def score():
    DATA.append(json.dumps(request.json, sort_keys=True, indent=4, separators=(',', ': ')))
    TIMESTAMP.append(time.time())
    return ""

# check page
@app.route('/check')
def check():
    # counts session data
    line1 = "Session data received: {0}".format(len(DATA))
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime('%Y-%m-%d %H:%M:%S')

        # information on the latest data point received
        line2 = "Latest datapoint received at: {0}".format(data_time)
        line3 = DATA[-1]

        # if data is received, return all possible fraud cases to the site
        if line3:
            df_1 = json.loads(DATA[-1])
            df_raw = json_normalize(df_1)
            X = build_X(df_raw)

            # string representation of the labels
            prediction = model_and_db(X, connection, model)
            if prediction[0] == 0:
                pred = 'Not fraudulent'
            elif prediction[0] == 1:
                pred = 'Fraudulent'
            else:
                pred = 'Error'

            # reading info from prediction
            prob_not_fraud = prediction[1]
            prob_fraud = prediction[2]

            # more information on the latest datapoint received
            line4 = "\t- Classification = {0}\n\t- Probability of fraud = {1}\n\t- Probability of not fraud = {2}".format(pred, prob_fraud, prob_not_fraud)

            # return dataframe of Mongo instances where prediction == Fraud
            df = pd.DataFrame(prediction[3])
            df.columns = ['ID', 'Probability of Fraud', 'Probability of Not Fraud']
            line5 = df
            output = "{0}\n{1}\n{2}\n\nFraudulent Entries:\n\n{3}".format(line2, line1, line4, line5)
    else:
        line2 = "Waiting on new datapoint..."
        output = "{0}\n{1}".format(line1, line2)

    return output, 200, {'Content-Type': 'text/css; charset=utf-8'}

def register_for_ping(ip, port):
    registration_data = {'ip': ip, 'port': port}
    requests.post(REGISTER_URL, data=registration_data)

if __name__ == '__main__':
    # Register for pinging service
    ip_address = socket.gethostbyname(socket.gethostname())
    print "Attempting to register %s:%d" % (ip_address, PORT)
    register_for_ping(ip_address, str(PORT))

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
