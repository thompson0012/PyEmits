![Project Icon](./assets/icon.png)

# What is Pyemits

PyEmits, a python package for easy manipulation in time-series data.

The ultimate goal:

> Keep it simple and stupid

> Make everything configurable

> Uniform API for machine learning and deep learning

# Why need Pyemits?

Time-series data is very common in real life.

- Engineering
- FSI industry (Financial Services Industry)
- FMCG (Fast Moving Consumer Good)

Data scientist's work consists of:

- forecasting
- prediction/simulation
- data preparation
- cleansing
- anomaly detection
- descriptive data analysis/exploratory data analysis/data profile
- data processing and ETL pipeline scripts

each new business unit shall build the following wheels again and again

## if you are facing these problems, then Pyemits is fit for you

1. data processing and ETL pipeline
    1. extraction
    2. transformation
        1. cleansing
        2. feature engineering
        3. remove outliers
        4. AI landing for prediction, forecasting
    3. write it back to database
2. ml framework
    1. multiple model training
    2. multiple model prediction
    3. kfold validation
    4. anomaly detection
    5. forecasting
    6. develop deep learning model (regression)
    7. ensemble modelling
3. exploratory data analysis
    1. descriptive data analysis
    2. data profile
    3. data set comparison

data scientist need to write different code to develop their model is there a package integrate all ml lib with single
simple api? That's why I create this project.

This project is under active development, free to use (Apache 2.0)
I am happy to see anyone can contribute for more advancement on features

# New feature:

[data processing pipeline](#data-processing-pipeline)

[db connection and manipulation](#io)

# Development Progress

<table>
    <tr>
        <td>Features</td>
        <td>Progress</td>
        <td>Available at version</td>
        <td>Notes</td>
    </tr>
    <tr>
        <td>PyOD integration</td>
        <td>80%</td>
        <td>0.1.2</td>
        <td>model parameters config are not yet finished</td>
    </tr>
    <tr>
        <td>XGBoost integration</td>
        <td>80%</td>
        <td>0.1.2</td>
        <td>model parameters config are not yet finished</td>
    </tr>
    <tr>
        <td>LightGBM integration</td>
        <td>80%</td>
        <td>0.1.2</td>
        <td>model parameters config are not yet finished</td>
    </tr>
    <tr>
        <td>Sklearn model integration</td>
        <td>80%</td>
        <td>0.1.2</td>
        <td>model parameters config are not yet finished</td>
    </tr>
    <tr>
        <td>Keras integration</td>
        <td>100%</td>
        <td>0.1.2</td>
        <td></td>
    </tr>
    <tr>
        <td>Pytorch_lightning integration</td>
        <td>100%</td>
        <td>0.1.2</td>
        <td></td>
    </tr>
    <tr>
        <td>MXnet integration</td>
        <td>0%</td>
        <td>tbc</td>
        <td></td>
    </tr>
    <tr>
        <td>DB connection</td>
        <td>0%</td>
        <td>tbc</td>
        <td></td>
    </tr>
    <tr>
        <td>aggregation</td>
        <td>0%</td>
        <td>0.1.3</td>
        <td></td>
    </tr>
    <tr>
        <td>cleansing</td>
        <td>0%</td>
        <td>0.1.3</td>
        <td></td>
    </tr>
    <tr>
        <td>dimensional reduction</td>
        <td>0%</td>
        <td>0.1.3</td>
        <td></td>
    </tr>
    <tr>
        <td>Kalman filtering</td>
        <td>0%</td>
        <td>0.1.3 or later</td>
        <td></td>
    </tr>
    <tr>
        <td>model evaluation and visualization</td>
        <td>0%</td>
        <td>0.1.3 or later</td>
        <td></td>
    </tr>
    <tr>
        <td>data profile for exploration</td>
        <td>20%</td>
        <td>0.1.3 or later</td>
        <td>finished data statistics only</td>
    </tr>
    <tr>
        <td>forecast at scale</td>
        <td>100%</td>
        <td>0.1.2</td>
        <td>see preprocessing.scaling.py</td>
    </tr>

</table>

# Release Update

<table>
    <tr>
        <td>Version</td>
        <td>Features</td>
        <td>Notes</td>
    </tr>
    <tr>
        <td>0.1</td>
        <td>initialization of project</td>
        <td></td>
    </tr>
    <tr>
        <td>0.1.1</td>
        <td>RegTrainer/ParallelTrainer/KFoldCV</td>
        <td></td>
    </tr>
    <tr>
        <td>0.1.2</td>
        <td>PyOD/Keras/Pytorch_lightning/scaling/splitting</td>
        <td></td>
    </tr>

</table>

# Install

```shell
pip install pyemits
```

# Features highlight

> scikit-learn API style
>
> inherit the design concept of pyecharts, "everything is configurable"
>
> highly flexible configuration items, can easily integrate with existing model
>
> easily integrate to SaaS product for product proof of concept

# Easy training

```python
import numpy as np

from pyemits.core.ml.regression.trainer import RegTrainer, RegressionDataModel

X = np.random.randint(1, 100, size=(1000, 10))
y = np.random.randint(1, 100, size=(1000, 1))

raw_data_model = RegressionDataModel(X, y)
trainer = RegTrainer(['XGBoost'], [None], raw_data_model)
trainer.fit()

```

# Accept neural network as model

```python
import numpy as np

from pyemits.core.ml.regression.trainer import RegTrainer, RegressionDataModel
from pyemits.core.ml.regression.nn import KerasWrapper

X = np.random.randint(1, 100, size=(1000, 10, 10))
y = np.random.randint(1, 100, size=(1000, 4))

keras_lstm_model = KerasWrapper.from_simple_lstm_model((10, 10), 4)
raw_data_model = RegressionDataModel(X, y)
trainer = RegTrainer([keras_lstm_model], [None], raw_data_model)
trainer.fit()
```

also keep flexibility on customized model

```python
import numpy as np

from pyemits.core.ml.regression.trainer import RegTrainer, RegressionDataModel
from pyemits.core.ml.regression.nn import KerasWrapper

X = np.random.randint(1, 100, size=(1000, 10, 10))
y = np.random.randint(1, 100, size=(1000, 4))

from keras.layers import Dense, Dropout, LSTM
from keras import Sequential

model = Sequential()
model.add(LSTM(128,
               activation='softmax',
               input_shape=(10, 10),
               ))
model.add(Dropout(0.1))
model.add(Dense(4))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

keras_lstm_model = KerasWrapper(model, nickname='LSTM')
raw_data_model = RegressionDataModel(X, y)
trainer = RegTrainer([keras_lstm_model], [None], raw_data_model)
trainer.fit()
```

or attach it in algo config

```python
import numpy as np

from pyemits.core.ml.regression.trainer import RegTrainer, RegressionDataModel
from pyemits.core.ml.regression.nn import KerasWrapper
from pyemits.common.config_model import KerasSequentialConfig

X = np.random.randint(1, 100, size=(1000, 10, 10))
y = np.random.randint(1, 100, size=(1000, 4))

from keras.layers import Dense, Dropout, LSTM
from keras import Sequential

keras_lstm_model = KerasWrapper(nickname='LSTM')
config = KerasSequentialConfig(layer=[LSTM(128,
                                           activation='softmax',
                                           input_shape=(10, 10),
                                           ),
                                      Dropout(0.1),
                                      Dense(4)],
                               compile=dict(loss='mse', optimizer='adam', metrics=['mse']))

raw_data_model = RegressionDataModel(X, y)
trainer = RegTrainer([keras_lstm_model],
                     [config],
                     raw_data_model,
                     {'fit_config': [dict(epochs=10, batch_size=32)]})
trainer.fit()
```

PyTorch, MXNet under development you can leave me a message if you want to contribute

# MultiOutput training

```python
import numpy as np

from pyemits.core.ml.regression.trainer import RegressionDataModel, MultiOutputRegTrainer
from pyemits.core.preprocessing.splitting import SlidingWindowSplitter

X = np.random.randint(1, 100, size=(10000, 1))
y = np.random.randint(1, 100, size=(10000, 1))

# when use auto-regressive like MultiOutput, pls set ravel = True
# ravel = False, when you are using LSTM which support multiple dimension
splitter = SlidingWindowSplitter(24, 24, ravel=True)
X, y = splitter.split(X, y)

raw_data_model = RegressionDataModel(X, y)
trainer = MultiOutputRegTrainer(['XGBoost'], [None], raw_data_model)
trainer.fit()
```

# Parallel training

- provide fast training using parallel job
- use RegTrainer as base, but add Parallel running

```python
import numpy as np

from pyemits.core.ml.regression.trainer import RegressionDataModel, ParallelRegTrainer

X = np.random.randint(1, 100, size=(10000, 1))
y = np.random.randint(1, 100, size=(10000, 1))

raw_data_model = RegressionDataModel(X, y)
trainer = ParallelRegTrainer(['XGBoost', 'LightGBM'], [None, None], raw_data_model)
trainer.fit()
```

or you can use RegTrainer for multiple model, but it is not in Parallel job

```python
import numpy as np

from pyemits.core.ml.regression.trainer import RegressionDataModel, RegTrainer

X = np.random.randint(1, 100, size=(10000, 1))
y = np.random.randint(1, 100, size=(10000, 1))

raw_data_model = RegressionDataModel(X, y)
trainer = RegTrainer(['XGBoost', 'LightGBM'], [None, None], raw_data_model)
trainer.fit()
```

# KFold training

- KFoldConfig is global config, will apply to all

```python
import numpy as np

from pyemits.core.ml.regression.trainer import RegressionDataModel, KFoldCVTrainer
from pyemits.common.config_model import KFoldConfig

X = np.random.randint(1, 100, size=(10000, 1))
y = np.random.randint(1, 100, size=(10000, 1))

raw_data_model = RegressionDataModel(X, y)
trainer = KFoldCVTrainer(['XGBoost', 'LightGBM'], [None, None], raw_data_model,
                         {'kfold_config': KFoldConfig(n_splits=10)})
trainer.fit()
```

# Easy prediction

```python
import numpy as np
from pyemits.core.ml.regression.trainer import RegressionDataModel, RegTrainer
from pyemits.core.ml.regression.predictor import RegPredictor

X = np.random.randint(1, 100, size=(10000, 1))
y = np.random.randint(1, 100, size=(10000, 1))

raw_data_model = RegressionDataModel(X, y)
trainer = RegTrainer(['XGBoost', 'LightGBM'], [None, None], raw_data_model)
trainer.fit()

predictor = RegPredictor(trainer.clf_models, 'RegTrainer')
predictor.predict(RegressionDataModel(X))

```

# Forecast at scale

- see examples: [forecast at scale.ipynb](./examples/forecast%20at%20scale.ipynb)

# Data Model

```python
from pyemits.common.data_model import RegressionDataModel
import numpy as np

X = np.random.randint(1, 100, size=(1000, 10, 10))
y = np.random.randint(1, 100, size=(1000, 1))

data_model = RegressionDataModel(X, y)

```

directly write an attribute to the data model

```python
data_model._update_attributes('X_shape', (1000, 10, 10))
data_model.X_shape
>> > (1000, 10, 10)
```

write something to the meta data

```python
data_model.add_meta_data('dimension', (1000, 10, 10))
data_model.meta_data
>> > {'dimension': (1000, 10, 10)}
```

# Anomaly detection (partial finished)

- see: [anomaly detection](./examples/anomaly%20detector.ipynb)
- root cause analyzer (under development)
- Kalman filter (under development)

```python
from pyemits.core.ml.anomaly_detection.predictor import AnomalyPredictor
from pyemits.core.ml.anomaly_detection.trainer import AnomalyTrainer, PyodWrapper
from pyemits.common.data_model import AnomalyDataModel
from pyemits.common.config_model import PyodIforestConfig
from pyod.models.iforest import IForest
from pyod.utils import generate_data

contamination = 0.1  # percentage of outliers
n_train = 1000  # number of training points
n_test = 200  # number of testing points

X_train, y_train, X_test, y_test = generate_data(
    n_train=n_train, n_test=n_test, contamination=contamination)

# highly flexible model config, accept str, PyodWrapper with/without initialized model
# either one
trainer = AnomalyTrainer(['IForest', PyodWrapper(IForest()), PyodWrapper(IForest), 'IForest', 'IForest', 'IForest'],
                         None, AnomalyDataModel(X_train))
trainer = AnomalyTrainer([PyodWrapper(IForest(contamination=0.05)), PyodWrapper(IForest)],
                         [None, PyodIforestConfig(contamination=0.05)], AnomalyDataModel(X_train))
trainer.fit()

# option 1
predictor = AnomalyPredictor(trainer.clf_models)

# option 2
predictor = AnomalyPredictor(trainer.clf_models,
                             other_config={'standard_scaler': predictor.misc_container['standard_scaler']})

# option 3
predictor = AnomalyPredictor(trainer.clf_models,
                             other_config={'standard_scaler': predictor.misc_container['standard_scaler'],
                                           'combination_config': {'n_buckets': 5}}, combination_method='moa')

predictor.predict(AnomalyDataModel(X_test))

```

# Data processing pipeline

it features in the following:

- easy configuration
    - register steps, tasks in data processing pipeline
- log data result in each tasks, each steps
- record the flow of pipeline, from steps to work (from marco to micro)

you can embed other function features in the task, but parameter: "data" is required to be passed in

e.g. add email notification, add log, upload to database etc...

```python

from pyemits.core.preprocessing.pipeline import DataNode, NumpyDataNode, PandasDataFrameDataNode, PandasSeriesDataNode,

Pipeline, Step, Task
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random(size=(20, 20)))

dn = PandasDataFrameDataNode.from_pandas(df)


def sum_each_col(data, a=1, b=2):
    return data.sum()


def sum_series(data):
    return np.array([data.sum()])
```

task registration and arguments registration

```python
task_a = Task(sum_each_col)
task_a.register_args(a=10, b=10)
task_b = Task(sum_series)
```

pipeline register step and execute

```python
pipeline = Pipeline()

step_a = Step('step_a', [task_a], '')
step_b = Step('step_b', [task_b], '')

pipeline.register_step(step_a)
pipeline.register_step(step_b)
pipeline.execute(dn)
```

know the steps and its tasks from start to end

```python
pipeline.get_step_task_mapping()
>> > {0: ('test', ['sum_each_col']), 1: ('test1', ['sum_series'])}
```

know the snapshot result in each steps, each tasks, friendly to data scientist for debugging

```markdown
pipeline.get_pipeline_snapshot_res(step_id=1,tasks_id=0)
> > > array([197.70351007])
```

# Evaluation (under development)

- see module: [evaluation](pyemits/core/evaluation)
- backtesting
- model evaluation

# Ensemble (under development)

- blending
- stacking
- voting
- by combo package
    - moa
    - aom
    - average
    - median
    - maximization

# IO

- db connection and manipulation

```python
from pyemits.common.io.db import DBConnectionBase

db = DBConnectionBase.from_full_db_path('sqlite:///test.db')

db.execute('CREATE TABLE abcc(c1 int, c2 int, c3 int)')

db.execute('INSERT INTO abcc(c1, c2, c3) VALUES (10, 10, 10)', always_commit=True)

db.execute('SELECT * FROM abcc', fetch=10)
db.execute('SELECT * FROM abcc', fetch='all')

schemas = db.get_schemas()

schemas['main']['abcc']
>> [{'name': 'c1',
     'type': INTEGER(),
     'nullable': True,
     'default': None,
     'autoincrement': 'auto',
     'primary_key': 0},
    {'name': 'c2',
     'type': INTEGER(),
     'nullable': True,
     'default': None,
     'autoincrement': 'auto',
     'primary_key': 0},
    {'name': 'c3',
     'type': INTEGER(),
     'nullable': True,
     'default': None,
     'autoincrement': 'auto',
     'primary_key': 0}]
```

- local

# dashboard ???

# other miscellaneous feature

- continuous evaluation
- aggregation
- dimensional reduction
- data profile (intensive data overview)

# to be confirmed

....

# References

the following libraries gave me some idea/insight

1. greykit
    1. changepoint detection
    2. model summary
    3. seaonality
2. pytorch-forecasting
3. darts
4. pyaf
5. orbit
6. kats/prophets by facebook
7. sktime
8. gluon ts
9. tslearn
10. pyts
11. luminaries
12. tods
13. autots
14. pyodds
15. scikit-hts


