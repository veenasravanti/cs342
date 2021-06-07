import torchvision
from PIL import Image
import os


from torch.utils.data import Dataset, DataLoader

import csv
LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        print(LABEL_NAMES[0])
        os.chdir("..")
        currentpath=os.getcwd()
        dataset_path=currentpath+"\\"+dataset_path

        with open(dataset_path+'\labels.csv') as f:
            data=csv.reader(f)
            next(data)
            self.data =list(data)
            self.path=dataset_path


    def __len__(self):
         return len(self.data)



    def __getitem__(self, idx):
        pic = Image.open(self.path+"\\"+self.data[idx][0])
        tensor = torchvision.transforms.ToTensor()(pic)
        label= {
            LABEL_NAMES[0]: 0,
            LABEL_NAMES[1]: 1,
            LABEL_NAMES[2]:2,
            LABEL_NAMES[3]: 3,
            LABEL_NAMES[4]: 4,
            LABEL_NAMES[5]: 5,

        }.get(self.data[idx][1])
        return (tensor,label)




def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
