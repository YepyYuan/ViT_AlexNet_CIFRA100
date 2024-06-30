import torch
import torch.utils
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from datetime import datetime
from torchvision.transforms.v2 import CutMix
import os
from tqdm import tqdm
import numpy as np

def split_cutmix_label(labels_cutmix):
    
    device = labels_cutmix.device
    lam = torch.min(labels_cutmix[labels_cutmix!=0])
    nonzero_index = torch.nonzero(labels_cutmix)

    target_A = []
    target_B = []

    index_save = False

    for index in nonzero_index:
        index_lam = labels_cutmix[index[0],index[1]]

        if torch.abs(lam-0.5) < 1e-6:
            if index_lam == 1.0:
                target_A.append(index[1].cpu().numpy())
                target_B.append(index[1].cpu().numpy())
            else:
                if not index_save:
                    target_A.append(index[1].cpu().numpy())
                    index_save = True
                else:
                    target_B.append(index[1].cpu().numpy())
                    index_save = False
        else:
            if index_lam == 1.0:
                target_A.append(index[1].cpu().numpy())
                target_B.append(index[1].cpu().numpy())
            elif torch.abs(index_lam - lam) < 1e-5:
                target_B.append(index[1].cpu().numpy())
            else:
                target_A.append(index[1].cpu().numpy())

    target_A = torch.from_numpy(np.array(target_A))
    target_B = torch.from_numpy(np.array(target_B))

    return target_A.to(device), target_B.to(device), lam.to(device)


def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, num_epoches, cutmix_flag=False):
    cur_datetime = datetime.now()
    date_str = '{:0>4d}{:0>2d}{:0>2d}'.format(cur_datetime.year, cur_datetime.month, cur_datetime.day)
    time_str = cur_datetime.strftime('%H%M%S')

    model_save_path = './models/' + date_str + '_' + time_str
    if os.path.exists(model_save_path) == False:
        os.makedirs(model_save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter('runs/'+ date_str + '_' + time_str)

    best_test_acc = 0

    for epoch in tqdm(range(num_epoches)):

        torch.cuda.empty_cache()
        itrs = 0

        train_loss_sum = 0.0
        # train_acc = 0.0
        # correct = 0
        # total = 0
        cutmix = None

        if cutmix_flag:
            cutmix = CutMix(num_classes= 100)
        ## train
        model.train()

        for data in train_dataloader:
            images, labels = data
            if cutmix:
                images, labels = Variable(images.to(device)), Variable(labels.to(device))
                images_cutmix, labels_cutmix = cutmix(images, labels)
                output = model(images_cutmix)
                labels_a, labels_b, lam = split_cutmix_label(labels_cutmix)
                # if labels_a.shape[0] != output.shape[0] or labels_b.shape[0] != output.shape[0] :
                #     print(lam)
                #     print(labels_cutmix[0] - lam)

                loss = (1-lam) * criterion(output, labels_a) + lam * criterion(output, labels_b)
            else:
                images, labels = Variable(images.to(device)), Variable(labels.to(device))
                output = model(images)
                loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            # _,prediction = torch.max(output, dim=1)

            # total += labels.size(0)
            # correct += prediction.eq(labels).cpu().sum()
            itrs += 1

            train_loss_avg = train_loss_sum / itrs
            # train_acc = correct / total
        
        model_weights = model.state_dict()
        if (epoch+1) % 5 == 0:
            torch.save(model_weights, model_save_path + '/model_epoch_{:0>3d}.pth'.format(epoch+1))

        print('Epoch %d| %d training complete!' %(epoch+1, num_epoches))
        print('-'*30)
        print('train loss: %.6f' %(train_loss_avg))

        test_correct = 0
        test_total = 0
        test_loss = []

        ## test
        model.eval()

        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            loss = criterion(output, labels)

            _,prediction = torch.max(output, dim=1)

            test_total += labels.size(0)
            test_correct += prediction.eq(labels).cpu().sum()
            test_acc = test_correct / test_total
            test_loss.append(loss.item())
        
        test_loss = sum(test_loss)/len(test_loss)

        if test_acc > best_test_acc:
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, model_save_path + '/best_model.pth')

            best_test_acc = test_acc


        writer.add_scalars('loss', {'train': train_loss_avg, 'test':test_loss} , epoch)
        writer.add_scalars('acc', {'test':test_acc}, epoch)
        
        print('test loss: %.6f acc: %.3f' %(test_loss, test_acc))

    writer.close()