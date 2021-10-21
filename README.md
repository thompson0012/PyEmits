![Project Icon](./assets/icon.png)

PyEmits, a python package for easy manipulation in time-series data. Time-series data is very common in real life.

- Engineering
- FSI industry (Financial Services Industry)
- FMCG (Fast Moving Consumer Good)

Data scientist's work consists of:
- forecasting
- prediction/simulation
- data prepration
- cleansing
- anomaly detection
- descriptive data analysis/exploratory data analysis 

each new business unit shall build the following wheels again and again
1. data pipeline
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
   6. deep learning model in easy way
   7. ensemble modelling
3. exploratory data analysis
   1. descriptive data analysis
   2. ...

That's why I create this project, also for fun. haha

This project is under active development, free to use (Apache 2.0)
I am happy to see anyone can contribute for more advancement on features

# Install
```shell
pip install pyemits
```

# Features highlight

1. Easy training

```python
import numpy as np

from pyemits.core.ml.regression.trainer import RegTrainer, RegressionDataModel

X = np.random.randint(1, 100, size=(1000, 10))
y = np.random.randint(1, 100, size=(1000, 1))

raw_data_model = RegressionDataModel(X, y)
trainer = RegTrainer(['XGBoost'], [None], raw_data_model)
trainer.fit()

```

2. Accept neural network as model

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
                     {'fit_config' : [dict(epochs=10, batch_size=32)]})
trainer.fit()
```
PyTorch, MXNet under development
you can leave me a message if you want to contribute

3. MultiOutput training
```python
import numpy as np 

from pyemits.core.ml.regression.trainer import RegressionDataModel, MultiOutputRegTrainer
from pyemits.core.preprocessing.splitting import SlidingWindowSplitter

X = np.random.randint(1, 100, size=(10000, 1))
y = np.random.randint(1, 100, size=(10000, 1))

# when use auto-regressive like MultiOutput, pls set ravel = True
# ravel = False, when you are using LSTM which support multiple dimension
splitter = SlidingWindowSplitter(24,24,ravel=True)
X, y = splitter.split(X, y)

raw_data_model = RegressionDataModel(X,y)
trainer = MultiOutputRegTrainer(['XGBoost'], [None], raw_data_model)
trainer.fit()
```
4. Parallel training
   - provide fast training using parallel job
   - use RegTrainer as base, but add Parallel running
```python
import numpy as np 

from pyemits.core.ml.regression.trainer import RegressionDataModel, ParallelRegTrainer

X = np.random.randint(1, 100, size=(10000, 1))
y = np.random.randint(1, 100, size=(10000, 1))

raw_data_model = RegressionDataModel(X,y)
trainer = ParallelRegTrainer(['XGBoost', 'LightGBM'], [None, None], raw_data_model)
trainer.fit()
```

or you can use RegTrainer for multiple model, but it is not in Parallel job
```python
import numpy as np 

from pyemits.core.ml.regression.trainer import RegressionDataModel,  RegTrainer

X = np.random.randint(1, 100, size=(10000, 1))
y = np.random.randint(1, 100, size=(10000, 1))

raw_data_model = RegressionDataModel(X,y)
trainer = RegTrainer(['XGBoost', 'LightGBM'], [None, None], raw_data_model)
trainer.fit()
```
5. KFold training
   - KFoldConfig is global config, will apply to all
```python
import numpy as np 

from pyemits.core.ml.regression.trainer import RegressionDataModel,  KFoldCVTrainer
from pyemits.common.config_model import KFoldConfig

X = np.random.randint(1, 100, size=(10000, 1))
y = np.random.randint(1, 100, size=(10000, 1))

raw_data_model = RegressionDataModel(X,y)
trainer = KFoldCVTrainer(['XGBoost', 'LightGBM'], [None, None], raw_data_model, {'kfold_config':KFoldConfig(n_splits=10)})
trainer.fit()
```
6. Easy prediction
```python
import numpy as np 
from pyemits.core.ml.regression.trainer import RegressionDataModel,  RegTrainer
from pyemits.core.ml.regression.predictor import RegPredictor

X = np.random.randint(1, 100, size=(10000, 1))
y = np.random.randint(1, 100, size=(10000, 1))

raw_data_model = RegressionDataModel(X,y)
trainer = RegTrainer(['XGBoost', 'LightGBM'], [None, None], raw_data_model)
trainer.fit()

predictor = RegPredictor(trainer.clf_models, 'RegTrainer')
predictor.predict(RegressionDataModel(X))

```
7. Forecast at scale
   - see examples: [forecast at scale.ipynb](./examples/forecast%20at%20scale.ipynb)
8. Data Model
```python
from pyemits.common.data_model import RegressionDataModel
import numpy as np
X = np.random.randint(1, 100, size=(1000,10,10))
y = np.random.randint(1, 100, size=(1000, 1))

data_model = RegressionDataModel(X, y)

data_model._update_variable('X_shape', (1000,10,10))
data_model.X_shape

data_model.add_meta_data('X_shape', (1000,10,10))
data_model.meta_data

```
9. Anomaly detection (under development)
   - see module: [anomaly detection](./pyemits/core/ml/anomaly_detection)
   - Kalman filter
10. Evaluation (under development)
    - see module: [evaluation](./pyemits/evaluation)
    - backtesting
    - model evaluation
11. Ensemble (under development) 
    - blending
    - stacking
    - voting
    - by combo package
      - moa
      - aom
      - average
      - median
      - maximization
12. IO 
    - db connection
    - local
13. dashboard ???
14. other miscellaneous feature
    - continuous evaluation
    - aggregation
    - dimensional reduction
    - data profile (intensive data overview)
15. to be confirmed

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


