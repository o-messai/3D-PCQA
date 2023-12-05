from model   import CoAtNet
from data_load import PointCloudDataset
import torch, csv, yaml, sys, warnings  
import time

warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', UserWarning)

if __name__ == '__main__':
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('#==> Using ',device,'....')
    
    model_file =  sys.argv[1]       #"./model/model_weights.pth"
    dataset_folder = sys.argv[2]    #"./dataset/ppc/"  
    config["BASICS"]['valid_dir'] = dataset_folder

    # model
    input_size = 32
    input_channel = 9
    num_blocks = [2, 2, 3, 3, 2]             # L
    channels = [128, 128, 128, 256, 1024]    # D
    model = CoAtNet((input_size, input_size), input_channel, num_blocks, channels, num_classes=1).to(device)  
    model.load_state_dict(torch.load(model_file, map_location=device))
    
    #test phase
    model.eval()
    pred_quality = []
    names = []
    valid_dataset = PointCloudDataset("BASICS", config, mode='valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False)
    st = time.time()
    with torch.no_grad():
        for i, (patches, file_name) in enumerate(valid_loader):
            patches_in = torch.stack(patches).to(device)
            patches_in = torch.transpose(patches_in, dim0=1, dim1=0)
            patches_in = patches_in.squeeze(0).to(device)
            outputs = model(patches_in)
            pred_quality.append(outputs.mean().item())
            names.append(file_name[0])
            #print("Processing >>", file_name[0])
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')        
    header = ['ppc', 'predictions']   
    with open('results.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        for i in range(len(pred_quality)):
            data = [names[i][:-4], pred_quality[i]]
            writer.writerow(data)