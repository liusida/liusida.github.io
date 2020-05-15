---
layout: post
title: from Tensorflow to Keras
---

I know 3 high level api for deep learning. They are Tensorflow.contrib.learn(SKFlow), TFLearn and Keras. All of them are great tools, but maybe I like Keras because of the easy style of code.

When I came through the [Tensorflow Tutorial of Deep and Wide Learning][1], I felt I can translate the code from tf.contrib.learn into Keras. 

![Tensorflow Wide and Deep Learning](https://www.tensorflow.org/versions/r0.11/images/wide_n_deep.svg)

In Tensorflow's tutorial, they said that, using Wide&Deep Model together, they could improve the accuracy from 83.6% to about 84.4%. After my Keras code was finished, I found that the accuracy of my new model is more than 85%. Although I used more units in the model, that's still nice I think!~

The dataset can be download here: [train_dataset][adult.data] and [test_dataset][adult.test]

Here is my code for Keras:

```python
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


def load_df(filename):
	with open(filename, 'r') as f:
		df = pd.read_csv(f, names = COLUMNS, skipinitialspace=True, engine='python')
	return df

def preprocess(df):
	df[LABEL_COLUMN] = (df['income_bracket'].apply(lambda x: ">50K" in x)).astype(int)
	df.pop("income_bracket")
	y = df[LABEL_COLUMN].values
	df.pop(LABEL_COLUMN)

	# This makes One-Hot Encoding:
	df = pd.get_dummies(df, columns=[x for x in CATEGORICAL_COLUMNS])
	# This makes scaled:
	df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

	X = df.values
	return X,y


def main():
	df_train = load_df('data/traindata')
	df_test = load_df('data/testdata')
	train_len = len(df_train)
	df = pd.concat([df_train, df_test])
	X, y = preprocess(df)
	X_train = X[:train_len]
	y_train = y[:train_len]
	X_test = X[train_len:]
	y_test = y[train_len:]

	model = Sequential()
	model.add(Dense(1024, input_dim=X_train.shape[1], activation='relu'))
	model.add(Dense(1024))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='rmsprop',
					loss='binary_crossentropy',
					metrics=['accuracy'])

	model.fit(X_train, y_train, nb_epoch=10, batch_size=32 )
	loss, accuracy = model.evaluate(X_test, y_test)
	print("")
	print(accuracy)

if (__name__=='__main__'):
	main()
```

During coding, I found preprocessing is very important:
* By using [One-Hot Encoding][3], I even don't need to treat Wide and Deep model separately. I just one-hot encode the categorical columns, and they can be feed together with continuous columns.
* By using [MaxMinScaler][2], I can avoid the result goes to one side at all. For example, if I don't use scaler, my prediction should be all 0s or 1s, because some columns have very big numbers, the model will ignore other columns' effect.

And thanks to sklearn and pandas, Keras don't need to do those preprocessing at all, which Tensorflow is doing these staff inside the model. I think it's better to do that before we feed the data, so it will be much clear what we are feeding to the neural network.


[1]:https://www.tensorflow.org/versions/r0.11/tutorials/wide_and_deep/index.html
[2]:http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
[3]:http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
[adult.data]:https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
[adult.test]:https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
