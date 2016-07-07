# Fraud Detection Case Study

A two-day case study on fraud detection. The goal of this sprint was to create an end-to-end prediction platform.

We began with feature selection and engineering. Some of the features we engineered were:
- Count NaNs, or missing data, per column as a percentage
- A percentage of uppercase characters for each title
- Event duration field

Based on the assumption that misclassifying true fraudulent cases costed us significantly higher than misclassifying true non-fraud cases, we modeled to minimize false negatives. After a train / test split, we iteratively tested the random forest model and selected the features that gave us the best result.

The model was designed to take one instance, classify it as fraud or not with associated probability scores, then save the results to a Mongo database. We then initialized a site on our local designed to receive one request and go through the previously described steps.

A server sent out live requests, or unseen data in JSON format, to the site we set up. We then classified and stored those new requests an the Mongo database. We coded up a dashboard on the splash page of the site for a quick-view of essential info. Essentially, we wanted to make potentially fraudulent cases accessible at a glance.

![Dashboard Example](/Users/drewrice/Desktop/Galvanize/Github/fraud-detection-case-study-DSI/Dashboard_example.png)

Technologies used:
- Python 2.7
- SKLearn's RandomForestClassifier and train_test_split
- Mongo DB, via PyMongo
- Flask
- Pandas, numpy

*Scott Contri, Clay Porter, Drew Rice, 2016.*
