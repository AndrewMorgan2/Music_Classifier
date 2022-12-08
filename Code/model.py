#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import seaborn as sn 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import argparse
from pathlib import Path

import dataset
import evalutation
import random

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=5e-5, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=200,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=1,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=1000,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=1000,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=1,
    type=int,
    help="Number of worker processes used to load data.",
)
##Custom parser
parser.add_argument(
    "--deep",
    default=False,
    type=bool,
    help="If deep is used",
)
parser.add_argument(
    "--noise",
    default=False,
    type=bool,
    help="If noise is used",
)
parser.add_argument(
    "--reduced",
    default=False,
    type=bool,
    help="If reduced noise is used",
)
parser.add_argument(
    "--pixel",
    default=False,
    type=bool,
    help="If pixel drop out is used",
)
##Global variables
deepBool= False
noiseBool = False
pixelBool = False
reducedBool = False

class AudioShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    global deepBool 
    deepBool = args.deep
    global nosieBool 
    nosieBool = args.noise
    global reducedBool 
    reducedBool = args.reduced
    global pixelBool 
    pixelBool = args.pixel

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    train_path = '../data/base/train.pkl'
    val_path = '../data/base/val.pkl'

    if args.noise == True:
        train_path = '../data/train_noise.pkl'
        val_path = '../data/val_aug/val_new.pkl'
    if args.reduced == True:
        train_path = '../data/train_reduce.pkl'
        val_path = '../data/val_aug/val_new.pkl'
    ##Add a if both then else into these two from there
    if args.reduced == True and args.noise:
        train_path = '../data/train_reduce_noise.pkl'
    train_dataset = dataset.GTZAN(train_path).dataset
    test_dataset = dataset.GTZAN(val_path).dataset

    print(len(train_dataset))

    if args.pixel == True:
        # Random dropout of pixels (10%)
        train_dataset_aug = train_dataset
        for data in train_dataset:
            newspectogram = data[1].numpy()
            for x in range(0, len(newspectogram[0])):
                for y in range(0, len(newspectogram[0][x])):
                    if random.random() <= 0.1:
                        newspectogram[0][x][y] = 0
            train_dataset_aug.append((data[0], torch.tensor(newspectogram), data[2], data[3]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
     
    model = None
    if args.deep == True:
        model = SNN(height=80, width=80, channels=1, class_count=10)
    else:
        model = DNN(height=80, width=80, channels=1, class_count=10)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

##Shallow neural network
class SNN(nn.Module):
    def __init__(self, height: int=80, width: int=80, channels: int=1, class_count: int=10):
        super().__init__()
        self.input_shape = AudioShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.conv1Left = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=16,
            kernel_size=(10, 23),
            padding='same',
        )

        self.initialise_layer(self.conv1Left)

        self.pool1Left = nn.MaxPool2d(kernel_size=(1, 20))

        self.conv1Right = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=16,
            kernel_size=(21, 20),
            padding='same',
        )
        self.initialise_layer(self.conv1Right)

        self.pool1Right = nn.MaxPool2d(kernel_size=(20, 1))

        self.fc = nn.Linear(10240, 200)
        self.initialise_layer(self.fc)

        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(200, 10)
        self.initialise_layer(self.fc1)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        ##Left 
        left_side = F.leaky_relu(self.conv1Left(spectrogram), negative_slope = 0.3)
        left_side = self.pool1Left(left_side)
        left_side = torch.flatten(left_side, start_dim=1)

        ##Right
        right_side = F.leaky_relu(self.conv1Right(spectrogram), negative_slope = 0.3)
        right_side = self.pool1Right(right_side)
        right_side = torch.flatten(right_side, start_dim=1)

        ##Merge and drop out before fully connected layer
        concatonated = torch.cat((left_side, right_side), axis = 1)

        concatonated = F.leaky_relu(self.fc(concatonated), negative_slope = 0.3)
        dropped_out = self.dropout(concatonated)

        fully_connected_layer = self.fc1(dropped_out)
        return fully_connected_layer

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class DNN(nn.Module):
    def __init__(self, height: int=80, width: int=80, channels: int=1, class_count: int=10):
        super().__init__()
        self.input_shape = AudioShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.conv1Left = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=16,
            kernel_size=(10, 23),
            padding='same',
        )
        self.initialise_layer(self.conv1Left)

        self.conv2Left = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(5, 11),
            padding='same',
        )
        self.initialise_layer(self.conv2Left)

        self.conv3Left = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 5),
            padding='same',
        )
        self.initialise_layer(self.conv3Left)

        self.conv4Left = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(2, 4),
            padding='same',
        )
        self.initialise_layer(self.conv4Left)

        self.conv1Right = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=16,
            kernel_size=(21, 10),
            padding='same',
        )
        self.initialise_layer(self.conv1Right)

        self.conv2Right = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(10, 5),
            padding='same',
        )
        self.initialise_layer(self.conv2Right)

        self.conv3Right = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 3),
            padding='same',
        )
        self.initialise_layer(self.conv3Right)

        self.conv4Right = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(4, 2),
            padding='same',
        )
        self.initialise_layer(self.conv4Right)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2Left = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool2Right = nn.MaxPool2d(kernel_size=(5, 1))

        self.fc = nn.Linear(5120, 200)
        self.initialise_layer(self.fc)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(200, 10)
        self.initialise_layer(self.fc1)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:

        ##Left side 
        left_side = F.leaky_relu(self.conv1Left(spectrogram), negative_slope = 0.3)
        left_side = self.pool1(left_side)
        left_side = F.leaky_relu(self.conv2Left(left_side), negative_slope = 0.3)
        left_side = self.pool1(left_side)
        left_side = F.leaky_relu(self.conv3Left(left_side), negative_slope = 0.3)
        left_side = self.pool1(left_side)
        left_side = F.leaky_relu(self.conv4Left(left_side), negative_slope = 0.3)
        left_side = self.pool2Left(left_side)

        left_side_last = torch.flatten(left_side, start_dim=1)

        ##Right side
        right_side = F.leaky_relu(self.conv1Right(spectrogram), negative_slope = 0.3)
        right_side = self.pool1(right_side)
        right_side = F.leaky_relu(self.conv2Right(right_side), negative_slope = 0.3)
        right_side = self.pool1(right_side)
        right_side = F.leaky_relu(self.conv3Right(right_side), negative_slope = 0.3)
        right_side = self.pool1(right_side)
        right_side = F.leaky_relu(self.conv4Right(right_side), negative_slope = 0.3)
        right_side = self.pool2Right(right_side)

        ##Merge and drop out before fully connected layer
        right_side_last = torch.flatten(right_side, start_dim=1)

        merged = torch.cat((left_side_last, right_side_last), axis = 1)

        merged = F.leaky_relu(self.fc(merged), negative_slope = 0.3)
        final = self.dropout(merged)

        final = self.fc1(final)
        return final

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight) # He initialisation

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for _, batch, labels, _ in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                logits = self.model.forward(batch)

                loss = self.criterion(logits, labels)

                loss.backward()

                self.optimizer.step() 
                self.optimizer.zero_grad() 
                
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        preds_for_graph = []
        list_preds = []
        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for _, batch, labels, _ in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                results["preds"].extend(list(preds.cpu().numpy()))
                results["labels"].extend(list(labels.cpu().numpy()))
                preds_for_graph.extend(list(preds.cpu().numpy()))
                list_preds.extend(list(logits))

        ##Code to make conf_matrix 
        ##This breaks the validate so only true when we want to look at the matrix
        if False:
            labels_music = ["blues", "classic", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
            conf = confusion_matrix(y_true = results["labels"], y_pred = preds_for_graph)
            conf_norm = conf.astype('float') / conf.sum(axis = 1)[:, np.newaxis]
            conf_norm_df = pd.DataFrame(conf_norm, index=labels_music, columns = labels_music)

            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = labels_music)

            cm_display.plot()
            plt.show()
            #plt.axes().set_xlabel("Actual")
            #plt.axes().set_ylabel("Preps")
            filename = "Confusion_Matrix_" + str(deepBool) + "=deep_" + str(noiseBool) + "=noise_" + str(reducedBool) + "=reduced_" + str(pixelBool) + "=pixel"
            #axis.set_title("Confusion_Matrix")
            plt.rcParams["figure.figsize"] = (16,16)
            plt.savefig(filename)

        accuracy = evalutation.evaluate(list_preds, "../data/base/val.pkl")
        if noiseBool == True or reducedBool == True:
            accuracy = evalutation.evaluate(list_preds, "../data/val_aug/val_new.pkl")

        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'deep={args.deep}_noise={args.noise}_pixel={args.pixel}_reduced={args.reduced}_bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())