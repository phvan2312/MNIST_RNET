from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, RandomApply, CenterCrop
from torch.utils.data import DataLoader

import copy
from models.model import MnistResNet
from dataset import MyDataset

epochs = 10

def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)

def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

import json, os
def get_data_loaders_from_file(train_batch_size, val_batch_size,
                               train_root, train_path,
                               valid_root, valid_path):
    train_root_2 = "/home/vanph/Desktop/pets/Resnet_MNIST/data/MNIST/"
    train_root_3 = ""
    train_root_4 = ""
    train_root_5 = "/home/vanph/Desktop/prjs/dataloader/out/yen_train/"

    valid_root_2 = "/home/vanph/Desktop/pets/Resnet_MNIST/data/MNIST/"
    valid_root_3 = "/home/vanph/Desktop/prjs/dataloader/out/yen_valid/"

    train_dct = json.load(open(train_path,'r'))
    train_dct_2 = json.load(open("/home/vanph/Desktop/pets/Resnet_MNIST/data/MNIST/train/labels.json","r"))
    train_dct_3 = json.load(open("/home/vanph/Desktop/pets/Resnet_MNIST/data/CH/train/labels.json","r"))
    train_dct_4 = json.load(open("/home/vanph/Desktop/pets/Resnet_MNIST/data/hardcode/train/labels.json","r"))
    train_dct_5 = json.load(open("/home/vanph/Desktop/prjs/dataloader/out/yen_train/labels.json","r"))

    valid_dct = json.load(open(valid_path,'r'))
    valid_dct_2 = json.load(open('/home/vanph/Desktop/pets/Resnet_MNIST/data/MNIST/test/labels.json','r'))
    valid_dct_3 = json.load(open('/home/vanph/Desktop/prjs/dataloader/out/yen_valid/labels.json','r'))

    train_data_transform = Compose([
        RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        Resize((28, 28)),
        ToTensor(),
    ])

    train_filenames = [os.path.join(train_root, _src) for _src in train_dct.keys()]
    train_filenames_2 = [os.path.join(train_root_2, _src) for _src in train_dct_2.keys()]
    train_filenames_3 = [os.path.join(train_root_3, _src) for _src in train_dct_3.keys()]
    train_filenames_4 = [os.path.join(train_root_4, _src) for _src in train_dct_4.keys()]
    train_filenames_5 = [os.path.join(train_root_5, _src) for _src in train_dct_5.keys()]

    train_filenames += train_filenames_2
    train_filenames += train_filenames_3
    train_filenames += train_filenames_4
    train_filenames += train_filenames_5

    train_labels = list(train_dct.values()) + list(train_dct_2.values()) + list(train_dct_3.values()) \
                   + list(train_dct_4.values()) + list(train_dct_5.values())

    train_dataloader = DataLoader(MyDataset(filenames=train_filenames, labels=train_labels,
                                            transform=train_data_transform,mode='train'),
                                  batch_size=train_batch_size, shuffle=True, num_workers=4)

    valid_data_transform = Compose([
        # RandomAffine(degrees=6, translate=(0.15, 0.15)),
        Resize((28, 28)),
        ToTensor(),
    ])

    valid_filenames = [os.path.join(valid_root, _src) for _src in valid_dct.keys()]
    valid_filenames_2 = [os.path.join(valid_root_2, _src) for _src in valid_dct_2.keys()]
    valid_filenames_3 = [os.path.join(valid_root_3, _src) for _src in valid_dct_3.keys()]

    valid_filenames += valid_filenames_2
    valid_filenames += valid_filenames_3

    valid_labels = list(valid_dct.values()) + list(valid_dct_2.values()) + list(valid_dct_3.values())

    valid_dataloader = DataLoader(MyDataset(filenames=valid_filenames, labels=valid_labels,
                                            transform=valid_data_transform,mode='valid'),
                                  batch_size=val_batch_size, shuffle=False)

    return train_dataloader, valid_dataloader

def get_data_loaders(train_batch_size, val_batch_size):
    mnist = MNIST(download=False, train=True, root="./raw/").train_data.float()

    train_data_transform = Compose([
        RandomAffine(degrees=30,translate=(0.15,0.15),scale=(0.9,1.1),shear=10),
        Resize((48, 48)),
        ToTensor(),
        Normalize((mnist.mean() / 255,), (mnist.std() / 255,))
    ])

    valid_data_transform = Compose([
        #RandomAffine(degrees=6, translate=(0.15, 0.15)),
        Resize((48, 48)),
        ToTensor(),
        Normalize((mnist.mean() / 255,), (mnist.std() / 255,))
    ])

    train_loader = DataLoader(MNIST(download=False, root=".", transform=train_data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=valid_data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

def save_model(model, epoch, output_path='./save'):
    save_path = os.path.join(output_path, "model_ep_%d.t7" % epoch)

    torch.save(model, save_path)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MnistResNet()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

    train_loader, val_loader = get_data_loaders_from_file(train_batch_size=32,val_batch_size=32,
                                                          train_root="/home/vanph/Desktop/prjs/dataloader/out/number_train",
                                                          train_path="/home/vanph/Desktop/prjs/dataloader/out/number_train/labels.json",
                                                          valid_root="/home/vanph/Desktop/prjs/dataloader/out/number_valid",
                                                          valid_path="/home/vanph/Desktop/prjs/dataloader/out/number_valid/labels.json") #get_data_loaders(8,8)

    #train_loader, val_loader = get_data_loaders(8,8)

    start_ts = time.time()

    model.to(device)

    losses = []
    batches = len(train_loader)
    val_batches = len(val_loader)

    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        total_loss = 0

        # progress bar (works in Jupyter notebook too!)
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

        # ----------------- TRAINING  --------------------
        # set model to training
        model.train()

        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)

            # training step for single batch
            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            # updating progress bar
            progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))

        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----------------- VALIDATION  -----------------
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []

        # set model to evaluating (testing)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)

                outputs = model(X)  # this get's the prediction from the network

                val_losses += loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1]  # get class from network's prediction

                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy),
                                       (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )

            print ('Saving model')
            save_model(copy.deepcopy(model),epoch= epoch + 1)

        print(
            f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
        print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss / batches)  # for plotting learning curve
    print(f"Training time: {time.time()-start_ts}s")

if __name__ == "__main__":
    main()
