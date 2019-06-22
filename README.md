# auto_ml
> Automated machine learning for production and analytics

[![Build Status](https://travis-ci.org/ClimbsRocks/auto_ml.svg?branch=master)](https://travis-ci.org/ClimbsRocks/auto_ml)
[![Documentation Status](http://readthedocs.org/projects/auto-ml/badge/?version=latest)](http://auto-ml.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/auto_ml.svg)](https://badge.fury.io/py/auto_ml)
[![Coverage Status](https://coveralls.io/repos/github/ClimbsRocks/auto_ml/badge.svg?branch=master&cacheBuster=1)](https://coveralls.io/github/ClimbsRocks/auto_ml?branch=master&cacheBuster=1)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]((https://img.shields.io/github/license/mashape/apistatus.svg))
<!-- Stars badge?! -->

## Installation

- `pip install auto_ml`

## Getting started

```python
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset

df_train, df_test = get_boston_dataset()

column_descriptions = {
    'MEDV': 'output',
    'CHAS': 'categorical'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df_train)

ml_predictor.score(df_test, df_test.MEDV)
```

## Show off some more features!

auto_ml is designed for production. Here's an example that includes serializing and loading the trained model, then getting predictions on single dictionaries, roughly the process you'd likely follow to deploy the trained model.

```python
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model

# Load data
df_train, df_test = get_boston_dataset()

# Tell auto_ml which column is 'output'
# Also note columns that aren't purely numerical
# Examples include ['nlp', 'date', 'categorical', 'ignore']
column_descriptions = {
  'MEDV': 'output'
  , 'CHAS': 'categorical'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df_train)

# Score the model on test data
test_score = ml_predictor.score(df_test, df_test.MEDV)

# auto_ml is specifically tuned for running in production
# It can get predictions on an individual row (passed in as a dictionary)
# A single prediction like this takes ~1 millisecond
# Here we will demonstrate saving the trained model, and loading it again
file_name = ml_predictor.save()

trained_model = load_ml_model(file_name)

# .predict and .predict_proba take in either:
# A pandas DataFrame
# A list of dictionaries
# A single dictionary (optimized for speed in production evironments)
predictions = trained_model.predict(df_test)
print(predictions)
```

## 3rd Party Packages- Deep Learning with TensorFlow & Keras, XGBoost, LightGBM, CatBoost

auto_ml has all of these awesome libraries integrated!
Generally, just pass one of them in for model_names.
`ml_predictor.train(data, model_names=['DeepLearningClassifier'])`

Available options are
- `DeepLearningClassifier` and `DeepLearningRegressor`
- `XGBClassifier` and `XGBRegressor`
- `LGBMClassifier` and `LGBMRegressor`
- `CatBoostClassifier` and `CatBoostRegressor`

All of these projects are ready for production. These projects all have prediction time in the 1 millisecond range for a single prediction, and are able to be serialized to disk and loaded into a new environment after training.

Depending on your machine, they can occasionally be difficult to install, so they are not included in auto_ml's default installation. You are responsible for installing them yourself. auto_ml will run fine without them installed (we check what's installed before choosing which algorithm to use).


## Feature Responses
Get linear-model-esque interpretations from non-linear models. See the [docs](http://auto-ml.readthedocs.io/en/latest/feature_responses.html) for more information and caveats.


## Classification

Binary and multiclass classification are both supported. Note that for now, labels must be integers (0 and 1 for binary classification). auto_ml will automatically detect if it is a binary or multiclass classification problem - you just have to pass in `ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)`


## Feature Learning

Also known as "finally found a way to make this deep learning stuff useful for my business". Deep Learning is great at learning important features from your data. But the way it turns these learned features into a final prediction is relatively basic. Gradient boosting is great at turning features into accurate predictions, but it doesn't do any feature learning.

In auto_ml, you can now automatically use both types of models for what they're great at. If you pass `feature_learning=True, fl_data=some_dataframe` to `.train()`, we will do exactly that: train a deep learning model on your `fl_data`. We won't ask it for predictions (standard stacking approach), instead, we'll use it's penultimate layer to get it's 10 most useful features. Then we'll train a gradient boosted model (or any other model of your choice) on those features plus all the original features.

Across some problems, we've witnessed this lead to a 5% gain in accuracy, while still making predictions in 1-4 milliseconds, depending on model complexity.

`ml_predictor.train(df_train, feature_learning=True, fl_data=df_fl_data)`

This feature only supports regression and binary classification currently. The rest of auto_ml supports multiclass classification.

## Categorical Ensembling

Ever wanted to train one market for every store/customer, but didn't want to maintain hundreds of thousands of independent models? With `ml_predictor.train_categorical_ensemble()`, we will handle that for you. You'll still have just one consistent API, `ml_predictor.predict(data)`, but behind this single API will be one model for each category you included in your training data.

Just tell us which column holds the category you want to split on, and we'll handle the rest. As always, saving the model, loading it in a different environment, and getting speedy predictions live in production is baked right in.

`ml_predictor.train_categorical_ensemble(df_train, categorical_column='store_name')`


### More details available in the docs

http://auto-ml.readthedocs.io/en/latest/


### Advice

Before you go any further, try running the code. Load up some data (either a DataFrame, or a list of dictionaries, where each dictionary is a row of data). Make a `column_descriptions` dictionary that tells us which attribute name in each row represents the value we're trying to predict. Pass all that into `auto_ml`, and see what happens!

Everything else in these docs assumes you have done at least the above. Start there and everything else will build on top. But this part gets you the output you're probably interested in, without unnecessary complexity.


## Docs

The full docs are available at https://auto_ml.readthedocs.io
Again though, I'd strongly recommend running this on an actual dataset before referencing the docs any futher.


## What this project does

Automates the whole machine learning process, making it super easy to use for both analytics, and getting real-time predictions in production.

A quick overview of buzzwords, this project automates:

- Analytics (pass in data, and auto_ml will tell you the relationship of each variable to what it is you're trying to predict).
- Feature Engineering (particularly around dates, and NLP).
- Robust Scaling (turning all values into their scaled versions between the range of 0 and 1, in a way that is robust to outliers, and works with sparse data).
- Feature Selection (picking only the features that actually prove useful).
- Data formatting (turning a DataFrame or a list of dictionaries into a sparse matrix, one-hot encoding categorical variables, taking the natural log of y for regression problems, etc).
- Model Selection (which model works best for your problem- we try roughly a dozen apiece for classification and regression problems, including favorites like XGBoost if it's installed on your machine).
- Hyperparameter Optimization (what hyperparameters work best for that model).
- Big Data (feed it lots of data- it's fairly efficient with resources).
- Unicorns (you could conceivably train it to predict what is a unicorn and what is not).
- Ice Cream (mmm, tasty...).
- Hugs (this makes it much easier to do your job, hopefully leaving you more time to hug those those you care about).


### Running the tests

If you've cloned the source code and are making any changes (highly encouraged!), or just want to make sure everything works in your environment, run
`nosetests -v tests`.

CI is also set up, so if you're developing on this, you can just open a PR, and the tests will run automatically on Travis-CI.

The tests are relatively comprehensive, though as with everything with auto_ml, I happily welcome your contributions here!

[![Analytics](https://ga-beacon.appspot.com/UA-58170643-5/auto_ml/readme)](https://github.com/igrigorik/ga-beacon)
[?1049h[?1h=[1;50r[34l[34h[?25h[23m[24m[0m[H[J[?25l[50;1H"echo" [新文件][1;1H[33m  1 [0m
[1m[34m~                                                                                                                                                                                             [3;1H~                                                                                                                                                                                             [4;1H~                                                                                                                                                                                             [5;1H~                                                                                                                                                                                             [6;1H~                                                                                                                                                                                             [7;1H~                                                                                                                                                                                             [8;1H~                                                                                                                                                                                             [9;1H~                                                                                                                                                                                             [10;1H~                                                                                                                                                                                             [11;1H~                                                                                                                                                                                             [12;1H~                                                                                                                                                                                             [13;1H~                                                                                                                                                                                             [14;1H~                                                                                                                                                                                             [15;1H~                                                                                                                                                                                             [16;1H~                                                                                                                                                                                             [17;1H~                                                                                                                                                                                             [18;1H~                                                                                                                                                                                             [19;1H~                                                                                                                                                                                             [20;1H~                                                                                                                                                                                             [21;1H~                                                                                                                                                                                             [22;1H~                                                                                                                                                                                             [23;1H~                                                                                                                                                                                             [24;1H~                                                                                                                                                                                             [25;1H~                                                                                                                                                                                             [26;1H~                                                                                                                                                                                             [27;1H~                                                                                                                                                                                             [28;1H~                                                                                                                                                                                             [29;1H~                                                                                                                                                                                             [30;1H~                                                                                                                                                                                             [31;1H~                                                                                                                                                                                             [32;1H~                                                                                                                                                                                             [33;1H~                                                                                                                                                                                             [34;1H~                                                                                                                                                                                             [35;1H~                                                                                                                                                                                             [36;1H~                                                                                                                                                                                             [37;1H~                                                                                                                                                                                             [38;1H~                                                                                                                                                                                             [39;1H~                                                                                                                                                                                             [40;1H~                                                                                                                                                                                             [41;1H~                                                                                                                                                                                             [42;1H~                                                                                                                                                                                             [43;1H~                                                                                                                                                                                             [44;1H~                                                                                                                                                                                             [45;1H~                                                                                                                                                                                             [46;1H~                                                                                                                                                                                             [47;1H~                                                                                                                                                                                             [48;1H~                                                                                                                                                                                             [49;1H~                                                                                                                                                                                             [0m[50;173H0,0-1[8C全部[1;5H[34h[?25h[?25l[50;1H[1m-- 插入 --[0m[50;11H[K[50;173H0,1[10C全部[1;5Ht init
[33m  2 [0mgit add README.md[2;22H[K[3;1H[33m  3 [0mgit commit -m "first commit"[3;33H[K[4;1H[33m  4 [0mgit remote add origin https://github.com/xiaojingyi/caffe-test.git[4;71H[K[5;1H[33m  5 [0mgit push -u origin master[5;30H[K[50;173H5,26[5;30H[34h[?25h[?25l[50;1H[K[50;173H5,25[9C全部[5;29H[34h[?25h[?25l[50;1H输入  :quit<Enter>  退出 Vim[50;173H[K[50;173H5,25[9C全部[5;29H[34h[?25h[?25l[50;173H[K[50;173H5,25[9C全部[5;29H[34h[?25h[?25l[50;173H[K[50;173H5,25[9C全部[5;29H[34h[?25h[?25l[50;173H[K[50;173H5,25[9C全部[5;29H[34h[?25h[?25l[50;163H^M[5;29H[50;163H  [5;29H[34h[?25h[?25l[50;163H^M[5;29H[50;163H  [5;29H[34h[?25h[?25l[50;163H^M[5;29H[50;163H  [5;29H[34h[?25h[?25l[50;163H^M[5;29H[50;163H  [5;29H[34h[?25h[?25l[50;163H^M[5;29H[50;163H  [5;29H[34h[?25h[?25l[50;163H^[[5;29H[34h[?25h[?25l[50;163H  [5;29H[34h[?25h[?25l[50;163H^[[5;29H[34h[?25h[?25l[50;163H  [5;29H[34h[?25h[?25l[50;163Hd[5;29H[34h[?25h[?25l[50;164Hs[5;29H[50;163H  [5;29H[34h[?25h[?25l[50;163Ha[5;29H[50;163H [5;30H[50;1H[1m-- 插入 --[0m[50;11H[K[50;173H5,26[9C全部[5;30H[34h[?25h[?25ld[50;176H7[5;31H[34h[?25h[?25l
[33m  6 [0m[6;5H[K[50;173H6,1 [6;5H[34h[?25h[?25l
[33m  7 [0m[7;5H[K[50;173H7[7;5H[34h[?25h[?25lf[50;175H2[7;6H[34h[?25h[?25ld[50;175H3[7;7H[34h[?25h[?25ls[50;175H4[7;8H[34h[?25h[?25l
[33m  8 [0m[8;5H[K[50;173H8,1[8;5H[34h[?25h[?25lf[50;175H2[8;6H[34h[?25h[?25ld[50;175H3[8;7H[34h[?25h[?25ls[50;175H4[8;8H[34h[?25h[?25l
[33m  9 [0m[9;5H[K[50;173H9,1[9;5H[34h[?25h[?25lf[50;175H2[9;6H[34h[?25h[?25ls[50;175H3[9;7H[34h[?25h[?25l
[33m 10 [0m[10;5H[K[50;173H10,1[10;5H[34h[?25h[?25lf[50;176H2[10;6H[34h[?25h[?25ls[50;176H3[10;7H[34h[?25h[?25ld[50;176H4[10;8H[34h[?25h[?25l
[33m 11 [0m[11;5H[K[50;174H1,1[11;5H[34h[?25h[?25lf[50;176H2[11;6H[34h[?25h[?25ls[50;176H3[11;7H[34h[?25h[?25ld[50;176H4[11;8H[34h[?25h[?25l
[33m 12 [0m[12;5H[K[50;174H2,1[12;5H[34h[?25h[?25lf[50;176H2[12;6H[34h[?25h[?25l
[33m 13 [0m[13;5H[K[50;174H3,1[13;5H[34h[?25h[?25ls[50;176H2[13;6H[34h[?25h[?25ld[50;176H3[13;7H[34h[?25h[?25lf[50;176H4[13;8H[34h[?25h[?25l
[33m 14 [0m[14;5H[K[50;174H4,1[14;5H[34h[?25h[?25ls[50;176H2[14;6H[34h[?25h[?25ld[50;176H3[14;7H[34h[?25h[?25lf[50;176H4[14;8H[34h[?25h[?25l
[33m 15 [0mdsf[15;8H[K[50;174H5[15;8H[34h[?25h[?25l
[33m 16 [0m[16;5H[K[50;174H6,1[16;5H[34h[?25h[?25ls[50;176H2[16;6H[34h[?25h[?25ld[50;176H3[16;7H[34h[?25h[?25l
[33m 17 [0m[17;5H[K[50;174H7,1[17;5H[34h[?25h[?25lf[50;176H2[17;6H[34h[?25h[?25ls[50;176H3[17;7H[34h[?25h[?25ld[50;176H4[17;8H[34h[?25h[?25l
[33m 18 [0m[18;5H[K[50;174H8,1[18;5H[34h[?25h[?25lf[50;176H2[18;6H[34h[?25h[?25ls[50;176H3[18;7H[34h[?25h[?25ld[50;176H4[18;8H[34h[?25h[?25l
[33m 19 [0m[19;5H[K[50;174H9,1[19;5H[34h[?25h[?25lf[50;176H2[19;6H[34h[?25h[?25ls[50;176H3[19;7H[34h[?25h[?25ld[50;176H4[19;8H[34h[?25h[?25l
[33m 20 [0m[20;5H[K[50;173H20,1[20;5H[34h[?25h[?25lf[50;176H2[20;6H[34h[?25h[?25l
[33m 21 [0m[21;5H[K[50;174H1,1[21;5H[34h[?25h[?25ls[50;176H2[21;6H[34h[?25h[?25ld[50;176H3[21;7H[34h[?25h[?25lf[50;176H4[21;8H[34h[?25h[?25l
[33m 22 [0m[22;5H[K[50;174H2,1[22;5H[34h[?25h[?25ls[50;176H2[22;6H[34h[?25h[?25ld[50;176H3[22;7H[34h[?25h[?25l
[33m 23 [0m[23;5H[K[50;174H3,1[23;5H[34h[?25h[?25lf[50;176H2[23;6H[34h[?25h[?25l
[33m 24 [0m[24;5H[K[50;174H4,1[24;5H[34h[?25h[?25l
[33m 25 [0m[25;5H[K[50;174H5[25;5H[34h[?25h[?25l
[33m 26 [0m[26;5H[K[50;174H6[26;5H[34h[?25h[?25l
[33m 27 [0m[27;5H[K[50;174H7[27;5H[34h[?25h[?25l
[33m 28 [0m[28;5H[K[50;174H8[28;5H[34h[?25h[?25l
[33m 29 [0m[29;5H[K[50;174H9[29;5H[34h[?25h[?25l
[33m 30 [0m[30;5H[K[50;173H30[30;5H[34h[?25h[?25l
[33m 31 [0m[31;5H[K[50;174H1[31;5H[34h[?25h[?25l
[33m 32 [0m[32;5H[K[50;174H2[32;5H[34h[?25h[?25l
[33m 33 [0m[33;5H[K[50;174H3[33;5H[34h[?25h[?25l
[33m 34 [0m[34;5H[K[50;174H4[34;5H[34h[?25h[?25l[50;1H[K[50;173H34,0-1[7C全部[34;5H[34h[?25h[?25l[50;1H输入  :quit<Enter>  退出 Vim[50;173H[K[50;173H34,0-1[7C全部[34;5H[34h[?25h[?25l[50;173H[K[50;173H34,0-1[7C全部[34;5H[34h[?25h[?25l[50;163H^M[34;5H[50;163H  [34;5H[34h[?25h[?25l[50;163H^M[34;5H[50;163H  [34;5H[34h[?25h[?25l[50;163H^M[34;5H[50;163H  [34;5H[34h[?25h[?25l[50;163H^M[34;5H[50;163H  [34;5H[34h[?25h