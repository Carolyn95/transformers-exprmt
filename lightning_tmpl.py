import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


# Pytorch Datasets Template
class MyDataset(Dataset):

  def __init__(self, **args):
    self.dataset = []
    self._build()

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    return self.dataset[index]

  def _build(self):
    self._build_examples_from_files(**args)

  def _build_examples_from_files(self, **args):
    print()


# Define LightningModule
class LitAutoEncoder(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(),
                                 nn.Linear(64, 3))
    self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                                 nn.Linear(64, 28 * 28))

  def forward(self, x):
    # in lightning, forward defines the prediction/inference actions
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
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

  def backward(self, loss, optimizer, optimizer_idx):
    loss.backward()


# Use forward for inference (predicting).
# Use training_step for training.

# Fit with Lightning Trainer
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)
# init model
autoencoder = LitAutoEncoder()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
trainer = pl.Trainer()
trainer.fit(autoencoder, train_loader)
"""
Trainer automates:
- epoch and batch iteration
- calling of optimizer.step(), backward, zero_grad()
- calling of eval(), enabling/disabling grads
- saving and loading weights
- tensorboard
- multi-gpu support
- tpu support 
- 16-bit training support
"""


# ------ RAPID prototyping
class LitAutoEncoder(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(),
                                 nn.Linear(128, 3))
    self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(),
                                 nn.Linear(128, 28 * 28))

  def training_step(self, batch, batch_idx):
    # --------------------------
    # REPLACE WITH YOUR OWN
    x, y = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x_hat, x)
    self.log('train_loss', loss)
    return loss
    # --------------------------

  def validation_step(self, batch, batch_idx):
    # --------------------------
    # REPLACE WITH YOUR OWN
    x, y = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x_hat, x)
    self.log('val_loss', loss)
    # --------------------------

  def test_step(self, batch, batch_idx):
    # --------------------------
    # REPLACE WITH YOUR OWN
    x, y = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x_hat, x)
    self.log('test_loss', loss)
    # --------------------------

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
