import numpy as np

from pyemits.common.validation import raise_if_value_not_contains, raise_if_not_all_value_contains
from pyemits.core.ml.base import BaseWrapper, NeuralNetworkWrapperBase


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
class TorchLightningWrapper(NeuralNetworkWrapperBase):
    """
    for each customized torch lightning model
    you have to achieve the following method:
    1. forward
    2. configure_optimizers
    3. training_step
    4. validation_step

    with the best use of pytorch_lighting, it automates:
    1. Epoch and batch iteration
    2. Calling of optimizer.step(), backward, zero_grad()
    3. Calling of .eval(), enabling/disabling grads
    4. Saving and loading weights
    5. Tensorboard (see Loggers options)
    6. Multi-GPU training support
    7. TPU support
    8. 16-bit training support
    """

    def __init__(self, torch_lightning_model_obj, nickname=None):
        super(TorchLightningWrapper, self).__init__(torch_lightning_model_obj, nickname)
        self._is_model_valid()

    def _fit(self, *args, **kwargs):
        import pytorch_lightning as pl
        trainer = pl.Trainer()
        return trainer.fit(self.model_obj, *args, **kwargs)

    def fit(self, *args, **kwargs):
        return self._fit(*args, **kwargs)

    def _predict(self, *args, **kwargs):
        return self._model_obj.forward(*args, **kwargs)

    def __str__(self):
        return f"PyTorchWrapper_{self._nickname}"

    def _is_model_valid(self):
        from pytorch_lightning import LightningModule
        from pyemits.common.utils.misc_utils import get_class_attributes
        self._model_obj: LightningModule
        raise_if_not_all_value_contains(['forward', 'configure_optimizers', 'training_step', 'validation_step'],
                                        get_class_attributes(self._model_obj))

    def add_layer2blank_model(self, *args, **kwargs):
        from torch import nn
        self._model_obj.seq.add_module(*args, **kwargs)

    @classmethod
    def from_blank_model(cls, nickname=None):
        import pytorch_lightning as pl
        from torch import nn
        import torch.nn.functional as F
        import torch

        class BlankSeqModel(pl.LightningModule):
            def __init__(self):
                super(BlankSeqModel, self).__init__()
                self.seq = nn.Sequential()

            def forward(self, x):
                x = self.seq(x)
                return x

            def training_step(self, batch, batch_idx):
                x, y = batch
                predictions = self.seq(x)
                loss = F.mse_loss(predictions, y)
                self.log('train_loss', loss)

            def validation_step(self, batch, batch_idx):
                x, y = batch
                val_prediction = self.seq(x)
                loss = F.mse_loss(val_prediction, y)
                self.log('val_loss', loss)

            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
                return optimizer

        return cls(BlankSeqModel(), 'torch_lightning_sequence')

    @classmethod
    def from_simple_linear_model(cls, input_dim, output_dim=1):
        from pytorch_lightning import LightningModule
        import pytorch_lightning as pl
        from torch import nn
        import torch.nn.functional as F
        import torch

        class LinearModel(LightningModule):
            def __init__(self):
                super(LinearModel, self).__init__()
                self.linear1 = nn.Linear(input_dim, input_dim * 2)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(input_dim * 2, input_dim)
                self.linear3 = nn.Linear(input_dim, output_dim)
                self.sigmoid = nn.Sigmoid()

                self.linear_relu_stack = nn.Sequential(self.linear1, self.relu,
                                                       self.linear2, self.relu,
                                                       self.linear3, self.sigmoid)

            def forward(self, x):
                x = self.linear_relu_stack(x)
                return x

            def training_step(self, batch, batch_idx):
                x, y = batch
                predictions = self.linear_relu_stack(x)
                loss = F.mse_loss(predictions, y)
                self.log('train_loss', loss)

            def validation_step(self, batch, batch_idx):
                x, y = batch
                val_prediction = self.linear_relu_stack(x)
                loss = F.mse_loss(val_prediction, y)
                self.log('val_loss', loss)

            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
                return optimizer

        return cls(LinearModel(), 'torch_lightning_simple_linear')

    @classmethod
    def from_simple_autoencoder(cls, input_length, output_length):
        import pytorch_lightning as pl
        from torch import nn
        import torch.nn.functional as F
        import torch

        class AutoEncoder(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_length, 64),
                    nn.ReLU(),
                    nn.Linear(64, 3)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(3, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_length)
                )

            def forward(self, x):
                # in lightning, forward defines the prediction(预测)/inference(推理) actions
                embedding = self.encoder(x)
                return embedding

            def training_step(self, batch, batch_idx):
                # training_step defined the train loop.
                # It is independent of forward
                x, y = batch
                x = x.view(x.size(0), -1)
                z = self.encoder(x)
                x_hat = self.decoder(z)
                loss = F.mse_loss(x_hat, x)
                # Logging to TensorBoard by default
                self.log('train_loss', loss)
                # self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                x = x.view(x.size(0), -1)
                z = self.encoder(x)
                x_hat = self.decoder(z)
                loss = F.mse_loss(x_hat, x)
                self.log('val_loss', loss)

            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
                return optimizer

        return cls(AutoEncoder(), "torch_lightning_auto_encoder")


def torchlighting_data_helper(X: np.ndarray,
                              y: np.ndarray,
                              shuffle: bool = True,
                              batch_size: int = 64,
                              train_size: float = 0.9):
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset
    from sklearn.model_selection import train_test_split
    import torch
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    dl_train = DataLoader(TensorDataset(torch.tensor(X_train).float(),
                                        torch.tensor(y_train).float()),
                          shuffle=shuffle,
                          batch_size=batch_size)
    dl_val = DataLoader(TensorDataset(torch.tensor(X_test).float(),
                                      torch.tensor(y_test).float()),
                        shuffle=shuffle,
                        batch_size=batch_size)
    return dl_train, dl_val


# TODO
class MXNetWrapper(NeuralNetworkWrapperBase):
    def __init__(self, mxnet_model_obj, nickname=None):
        super(MXNetWrapper, self).__init__(mxnet_model_obj, nickname)
