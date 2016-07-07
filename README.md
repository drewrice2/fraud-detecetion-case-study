# Fraud Detection Case Study

A two-day case study on fraud detection. The goal of this sprint was to create an end-to-end prediction platform.

We began with feature selection and engineering. Based on the assumption that misclassifying true fraudulent cases costed us significantly higher than misclassifying true non-fraud, we modeled to minimize false negatives. After a train / test split, we iteratively tested the random forest model and selected the features that gave us the best result.

The model was designed to take one instance, classify it as fraud or not with associated probability scores, then save the results to a Mongo database. We then initialized a site on our local designed to receive one request and go through the previously described steps.

A server sent out live requests, or unseen data, to the site we set up.

Technologies used:
    - Python 2.7
    - SKLearn's RandomForestClassifier and train_test_split
    - Mongo DB, via PyMongo
    - Flask
    - Pandas, numpy



*Scott Contri, Clay Porter, Drew Rice, 2016.*
