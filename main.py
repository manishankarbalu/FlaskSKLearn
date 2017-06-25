import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)

# inputs
training_data = 'data/concrete.csv'
include = ['cement' ,  'slag',  'ash' , 'water' , 'superplastic' , 'coarseagg'  ,'fineagg'  ,'age'  ,'strength']
dependent_variable = include[-1]

model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# These will be populated at training time
model_columns = None
clf = None


@app.route('/predict', methods=['GET'])
def predict():
    if clf:
        try:
            json_ = [
	{'cement': 540,  'slag':0.0 ,  'ash':0.0 , 'water':162.0 , 'superplastic':2.5 , 'coarseagg':1040.0  ,'fineagg':676.0  ,'age':28 }
      ]#request.json
            query = pd.get_dummies(pd.DataFrame(json_))

            for col in model_columns:
                if col not in query.columns:
                    query[col] = 0

            prediction = list(clf.predict(query))

            return jsonify({'prediction': prediction})

        except Exception, e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print 'train first'
        return 'no model here'


@app.route('/train', methods=['GET'])
def train():
    #Change this training part with data cleaning and assigning scikit-learn modellig
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model

    df = pd.read_csv(training_data)
    df_ = df[include]

    categoricals = []  # going to one-hot encode categorical variables

    for col, col_type in df_.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    print x
    y = df_ohe[dependent_variable]
    
    #poly = PolynomialFeatures(degree=3)
    #X_ = poly.fit_transform(x)
    
    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    
    global clf
    clf = linear_model.LinearRegression()
    start = time.time()
    clf.fit(x, y)
    print 'Trained in %.1f seconds' % (time.time() - start)
    print 'Model training score: %s' % clf.score(x, y)

    joblib.dump(clf, model_file_name)

    return 'Success'


@app.route('/clearModel', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model cleared'

    except Exception, e:
        print str(e)
        return 'Could not remove model'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception, e:
        port = 5000

    try:
        clf = joblib.load(model_file_name)
        print 'model loaded'
        model_columns = joblib.load(model_columns_file_name)
        print 'model columns loaded'

    except Exception, e:
        print 'No model here'
        print 'Train first'
        print str(e)
        clf = None

    app.run(host='0.0.0.0', port=port, debug=True)
