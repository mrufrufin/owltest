import torch
import torchaudio as TA
from birdset import BirdSet
from torch.utils.data import DataLoader
from torch import cuda
def get_data():
    train_ds = BirdSet(set_type="train")
    test_ds = BirdSet(set_type="test")
    return train_ds, test_ds


def runner(to_train = True):
    device = 'cpu'
    dstx, dste = get_data()
    if torch.cuda.is_available() == True:
        device = 'cuda'
    trainer(dstx)

def trainer_batch(epoch_idx, dloader):
    for batch_idx, (ci,cl) in enumerate(dloader):
        print(ci)
        print(cl)

def trainer(train_data, bs = 16, epochs = 1):
    tdload = DataLoader(train_data, shuffle=True, batch_size = bs)
    for batch_idx in range(epochs):
        trainer_batch(batch_idx, tdload)

runner()
