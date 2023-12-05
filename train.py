from tensorboardX import SummaryWriter 
from torchsummary import summary
from datetime     import datetime
from argparse     import ArgumentParser
from model   import CoAtNet
from data_load import PointCloudDataset
from utils     import ensure_dir, metric_eval
import torch.nn as nn
import numpy as np
import torch, os, shutil, yaml, warnings
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', UserWarning)

if __name__ == '__main__':
    # Training settings
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="ICIP20")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    args = parser.parse_args()
    
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # logger for tracking experiments
    if os.path.exists('/results/performance_logs.txt'):
        append_write = 'a'
    else:
        append_write = 'w' 
    ensure_dir('results')
    f = open('./results/performance_logs.txt', 'a+') 

    now = datetime.now()
    print("#==> Experiment date and time = ", now)
    f.write("\n \n #============================== Experiment date and time = %s.==============================#" % now)
    f.write("\n dataset = {:s} epochs = {:d}, batch_size = {:d}, lr = {:f}, weight_decay= {:f}".format(args.dataset, 
                                                                                                       args.epochs, 
                                                                                                       args.batch_size, 
                                                                                                       args.lr, 
                                                                                                       args.weight_decay))
    f.write("\n %s" % config)
    # seed = random.randint(10000000, 99999999)   
    seed = 14732152 # Fixed seed for test
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("#==> Seed:", seed)
    f.write("\n Seed : {:d}".format(seed))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        print('#==> Using GPU device:', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('#==> Using CPU....')
  
    print("#==> Dataset: " + args.dataset)
    ensure_dir('./results/model/' + args.dataset)
    save_model = "./results/model/"+ args.dataset +"/best_mesh.pth" 
    model_dir = "./results/model/"+ args.dataset +"/"
        
    # remove the folder 'visualize/tensorboard' and create another one empty
    shutil.rmtree('./results/visualize', ignore_errors=True)
    ensure_dir('./results/visualize/tensorboard')
    writer = SummaryWriter('./results/visualize/tensorboard')
    dataset = args.dataset

    train_dataset = PointCloudDataset(dataset, config, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)
    
    test_dataset = PointCloudDataset(dataset, config, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset)
    # model
    input_size = config['input_size']
    input_channel = config['input_channel'] 
    num_blocks = [3, 3, 6, 14, 2]
    channels = [64, 96, 128, 128, 512]
 
    model = CoAtNet((input_size, input_size), input_channel, num_blocks, channels, num_classes=1).to(device)
    #summary(model, input_size = [(input_channel, input_size, input_size)])   
     
    f.write("\n num_blocks = %s" % num_blocks)
    f.write("\n channels = %s" % channels)
    f.write("\n %s" % summary(model, input_size = [(input_channel, input_size, input_size)]))
    #model.load_state_dict(torch.load("./results/model/"+ args.dataset +"/best_mesh.pth"))
    
    criterion = nn.L1Loss() 
    #criterion = nn.MSELoss() 
    #criterion = nn.SmoothL1Loss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.8)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, last_epoch=-1)
    best_performance = -1
          
    # training phase
    #accum_iter = args.batch_size 
    
    for epoch in range(args.epochs):
        model.train()
        LOSS_all = 0
        for batch_idx, (patches, label) in enumerate(train_loader):    
                   
            for j in range(len(patches)):
                patches_in = patches[j].to(device)      
                label = label.to(device)
                patches_in = patches_in.squeeze(0)
                Quality = model(patches_in)
                loss = criterion(Quality, label)
                loss = loss # / accum_iter 
                LOSS_all = LOSS_all + loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                #if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                #    optimizer.step()
                #    optimizer.zero_grad()
                    
        train_loss_all = LOSS_all / (batch_idx + 1)
        print ('#==> Training_loss : ',train_loss_all)
        f = open('./results/performance_logs.txt', 'a+') 
        f.write('\n #==> Training_loss : %f' % train_loss_all)
        writer.add_scalar('train/total_loss',train_loss_all, epoch * len(train_loader) + batch_idx)  
        #torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(epoch)))
        
        # test phase
        model.eval()
        #testnum = len(os.listdir(config[dataset]['test_dir']))
        dataset = args.dataset
        testnum = sum(1 for line in open(config[dataset]['test_txt']))

        pred_quality = np.zeros(testnum)
        true_quality = np.zeros(testnum)
        LOSS_all_test = 0

        with torch.no_grad():
            for i, (patches, label) in enumerate(test_loader):
                
                patches_in = torch.stack(patches).to(device)
                patches_in = torch.transpose(patches_in, dim0=1, dim1=0)
                patches_in = patches_in.squeeze(0)

                label = label.to(device)
                true_quality[i] = label.item()
                outputs = model(patches_in)
                pred_quality[i] = outputs.mean()
                #print("pred_quality", pred_quality[i])
                #print("true_quality", true_quality[i])

        #LOSS_all_test = criterion(pred_quality, true_quality)
        test_loss = LOSS_all_test / (i + 1)
        SROCC_test, PLCC_test, KROCC_test, RMSE_test = metric_eval(pred_quality, true_quality)
        
        print("#==> "+dataset+" Epoch {} Test Results : loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(epoch,
                                                                                                            test_loss,
                                                                                                            SROCC_test,
                                                                                                            PLCC_test,
                                                                                                            KROCC_test,
                                                                                                            RMSE_test))
        f.write("\n #==> "+dataset+" Epoch {} Test Results : loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(epoch,
                                                                                                            test_loss,
                                                                                                            SROCC_test,
                                                                                                            PLCC_test,
                                                                                                            KROCC_test,
                                                                                                            RMSE_test))
        plt.scatter(pred_quality, true_quality, color='blue',marker='o',edgecolors='red')
        plt.xlabel("Prediction")
        plt.ylabel("MOS")
        plt.savefig('0_MOS_vs_pred.png')
        plt.clf()
        
        test_performance = (PLCC_test + SROCC_test)/2
        if best_performance <  test_performance:
            print("#==> ******** Update Epoch {} best valid RMSE ********".format(epoch))
            f.write("\n #==> ******** Update Epoch {} best valid RMSE "+dataset+" ********".format(epoch))
            #torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(epoch)))
            torch.save(model.state_dict(), save_model)
            writer.add_scalars('test/best score:', {'SROCC': SROCC_test, 'PLCC': PLCC_test,'KROCC': KROCC_test,}, epoch * len(test_loader) + i)
            #best_PLCC = PLCC
            best_performance = test_performance 
            
        #################################### Test on other databses ####################################
        """
        dataset = "PointXR"
        #testnum = len(os.listdir(config[dataset]['test_dir']))
        testnum = sum(1 for line in open(config[dataset]['test_txt']))
  
        pred_quality = np.zeros(testnum)
        true_quality = np.zeros(testnum)
        LOSS_all_test = 0
        test_dataset_new = PointCloudDataset(dataset, config, mode='test')
        test_loader_new = torch.utils.data.DataLoader(test_dataset_new)
        with torch.no_grad():
            for i, (patches, label) in enumerate(test_loader_new):
                
                patches_in = torch.stack(patches).to(device)
                patches_in = torch.transpose(patches_in, dim0=1, dim1=0)
                patches_in = patches_in.squeeze(0)
                #print("patches_in", patches_in.shape)
                label = label.to(device)
                true_quality[i] = label.item()
                outputs = model(patches_in)
                pred_quality[i] = outputs.mean()

        #LOSS_all_test = criterion(pred_quality, true_quality)
        test_loss = LOSS_all_test / (i + 1)
        SROCC_test, PLCC_test, KROCC_test, RMSE_test = metric_eval(pred_quality, true_quality)
        
        print("#==> "+dataset+" Epoch {} Test Results : loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(epoch,
                                                                                                            test_loss,
                                                                                                            SROCC_test,
                                                                                                            PLCC_test,
                                                                                                            KROCC_test,
                                                                                                            RMSE_test))
        f.write("\n #==> "+dataset+" Epoch {} Test Results : loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(epoch,
                                                                                                            test_loss,
                                                                                                            SROCC_test,
                                                                                                            PLCC_test,
                                                                                                            KROCC_test,
                                                                                                            RMSE_test))
        """
