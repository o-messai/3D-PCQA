import torch, os
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
from utils import get_processed_patches_rgb, normalize_point_cloud 


class PointCloudDataset(Dataset):
    def __init__(self, dataset, config, mode= 'valid'):
        super().__init__()
        self.mode = mode
        self.patch_size = config['patch_size'] 
        self.point_size = config['point_size'] 
        self.input_channel = config['input_channel'] 
        self.input_size = config['input_size'] 
        
        if mode == "train":
            self._load_folder = config[dataset]['train_dir'] 
            self.name_mos_txt = config[dataset]['train_txt'] 
            self.input_data = []
            self.label = []

            with open(self.name_mos_txt, 'r') as f:
                for line in f:
                    temp = line.strip().split(',')
                    input_data, label = temp[1], float(temp[2])
                    self.input_data.append((input_data))
                    self.label.append((label))
            #self.label = normalize_mos(self.label)
            
        elif mode == "test":
            self._load_folder  = config[dataset]['test_dir'] 
            self.name_mos_txt = config[dataset]['test_txt'] 
            self.input_data = []
            self.label = []

            with open(self.name_mos_txt, 'r') as f:
                for line in f:
                    temp = line.strip().split(',')
                    input_data, label = temp[1], float(temp[2])
                    self.input_data.append((input_data))
                    self.label.append((label))
            #self.label = normalize_mos(self.label)
            
        elif mode == "valid":
            
            self._load_folder  = config[dataset]['valid_dir']  
            self.input_data = sorted(os.listdir(self._load_folder))

        else: 
            print('Error in loading folder!') 

          
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        objpath = self._load_folder + str(self.input_data[idx])
        cloud = PyntCloud.from_file(objpath)
        points = normalize_point_cloud(torch.FloatTensor(cloud.xyz).to("cpu"))
        cloud.xyz = points
        rgb = cloud.points[['red', 'green', 'blue']].values / 255.0
        rgb  = torch.FloatTensor(rgb).to("cpu")
        input = get_processed_patches_rgb(points, rgb, self.patch_size, self.point_size, 
                                      self.input_channel, self.input_size, add_freq=True)
            
        if self.mode =="valid":
            file_name = self.input_data[idx]
            return input, file_name
        return input, torch.tensor(self.label[idx]).view(-1) 
    
if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    train_loader = PointCloudDataset('BASICS', config, mode='train')
    for i, (input, labels) in enumerate(train_loader):    
        print(len(input))
        print("label", labels)
        print(input[0].size())
        print(labels.size())
        break
        
        