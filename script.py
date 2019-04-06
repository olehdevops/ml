import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():

    df = pd.read_csv('dataset_kyiv.csv').set_index('date')

    df = df.drop(['mintempm', 'maxtempm'], axis=1)

    X = df[[col for col in df.columns if col != 'meantempm']]


    feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

    regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[50, 50],
                                          model_dir='tf_wx_model')

    def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
        return tf.estimator.inputs.pandas_input_fn(x=X, y=y, num_epochs=num_epochs, shuffle=shuffle, batch_size=batch_size)


    pred = regressor.predict(input_fn=wx_input_fn(X,
                                                  num_epochs=1,
                                                  shuffle=False))
    predictions = np.array([p['predictions'][0] for p in pred])
    # print(predictions)


    return 'Hello World! Prediction temperature'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
