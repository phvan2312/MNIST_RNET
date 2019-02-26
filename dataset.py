from PIL import Image
import cv2, random
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, filenames, labels, transform=None, mode='train'):
        assert len(filenames) == len(labels), "Number of files != number of labels"
        self.fns = filenames
        self.lbs = labels
        self.transform = transform
        self.mode = mode

        self.char2idx = {
            '0':0,
            '1':1,
            '2':2,
            '3':3,
            '4':4,
            '5':5,
            '6':6,
            '7':7,
            '8':8,
            '9':9,
            '¥':10
        }

    def __len__(self):
        return len(self.fns)

    def char_to_id(self, char):
        return self.char2idx[char]

    def __getitem__(self, idx):

        image = cv2.imread(self.fns[idx], 0) #Image.open(self.fns[idx]).convert('L')
        h,w = image.shape

        if self.mode == 'train'and 'dataloader' in self.fns[idx]:
            # if self.lbs[idx] == '¥':
            #     if random.randint(0,10) < 3:
            #         image = image[random.randint(0,5):h-random.randint(0,5),
            #                 random.randint(0,5):w-random.randint(0,5)]

            image = cv2.copyMakeBorder(image, random.randint(0,10), random.randint(0,10),
                                       random.randint(0,10), random.randint(0,10),borderType=cv2.BORDER_CONSTANT,
                                       value=(255,255,255))
        #print (self.fns[idx])
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, self.char2idx[self.lbs[idx]]