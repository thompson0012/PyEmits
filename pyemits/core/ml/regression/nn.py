

from pyemits.core.ml.base import WrapperBase, NeuralNetworkWrapperBase


class KerasWrapper(NeuralNetworkWrapperBase):
    def __init__(self, keras_model_obj, nickname=None):
        super(KerasWrapper, self).__init__(keras_model_obj, nickname)

    @classmethod
    def from_blank_model(cls, nickname=None):
        from keras import Sequential
        model = Sequential()
        return cls(model, nickname)

    @classmethod
    def from_simple_lstm_model(cls, train_data_shape, output_shape):
        """

        Parameters
        ----------
        train_data_shape
        output_shape

        Returns
        -------
        KerasWrapper()

        Notes
        -----
        if your rnn model have loss: nan, which mean you have inf,-inf, nan value which makes the explosion gradient

        """
        from keras import Sequential
        from keras.layers import Dense, Dropout, LSTM
        model = Sequential()
        model.add(LSTM(128,
                       activation='softmax',
                       input_shape=train_data_shape,
                       ))
        model.add(Dropout(0.1))
        model.add(Dense(output_shape))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return cls(model)

    def _fit(self, *args, **kwargs):
        return self.model_obj.fit(*args, **kwargs)

    def _predict(self, *args, **kwargs):
        return self.model_obj.predict(*args, **kwargs)

    def __str__(self):
        return f"KerasWrapper_{self._nickname}"


# TODO
class PyTorchWrapper(NeuralNetworkWrapperBase):
    def __init__(self, pytorch_model_obj, nickname=None):
        super(PyTorchWrapper, self).__init__(pytorch_model_obj, nickname)

    def _fit(self, *args, **kwargs):
        import pytorch_lightning as pl
        trainer = pl.Trainer()
        return trainer.fit(self.model_obj, *args, **kwargs)

    def fit(self, *args, **kwargs):
        return self._fit(*args, **kwargs)

    def __str__(self):
        return f"PyTorchWrapper_{self._nickname}"


# TODO
class MXNetWrapper(NeuralNetworkWrapperBase):
    def __init__(self, mxnet_model_obj, nickname=None):
        super(MXNetWrapper, self).__init__(mxnet_model_obj, nickname)
