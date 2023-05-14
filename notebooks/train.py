import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm
import dataloader
from torch.utils.checkpoint import checkpoint
import time
import warnings
from torchvision import transforms
warnings.filterwarnings('ignore')
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import cv2
from pcgrad import PCGrad
from segment_anything.utils.transforms import ResizeLongestSide
data_transform = {
    'train': transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
        # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
train_data_IDRiD, valid_data_IDRiD = dataloader.load_data()
train_data_IDRiD_od, valid_data_IDRiD_od = dataloader.load_data2()
# resnet50 = torch.load("models/DeepDR_history.pt")
def prepare_image(image, transform, device):
    image = image.cpu().numpy()
    image = transform.apply_image(image)
    image = torch.as_tensor(image,device=device.device)
    return image.permute(2,0,1).contiguous()
class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet50(pretrained=True)
        self.net.fc1 = nn.Linear(1000, 5)
        # self.net.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        feature = self.net(x)
        class1 = self.net.fc1(feature)
        # class2 = self.net.fc2(feature)
        return class1#, class2
# resnet50 = models.resnet50(pretrained=True)

sam_checkpoint = "./models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
resnet50 = HydraNet()
for param in resnet50.parameters():
    param.requires_grad = True
for param in sam.parameters():
    param.requires_grad = True

sam = sam.to(device)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
predictor = SamPredictor(sam)
# print(device)
# print(torch.cuda.device_count())
resnet50 = resnet50.to(device)
# predictor = predictor.to(device=device)

dataset = 'IDRiD'

def dice(pred, target):
    # pred = np.array(pred)
    # target = np.array(target)
    intersect = torch.logical_and(pred, target)
    return (2 * torch.sum(torch.sum(intersect))) / (torch.sum(torch.sum(pred)) + torch.sum(torch.sum(target)))

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        score = dice(logits, targets)
        return 1 - score
    
    
loss_func1 = nn.CrossEntropyLoss()
loss_func2 = nn.MSELoss()
loss_func3 = SoftDiceLoss()
# optimizer = torch.optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)
optimizer = PCGrad(torch.optim.SGD([{'params': sam.parameters()},{'params': resnet50.parameters()}], lr=0.001, momentum=0.9, weight_decay=5e-4))
def train_and_valid(model, model2, loss_function1, loss_function2, loss_function3, optimizer, epochs=25):

    history = []
    best_acc = 0.0
    best_epoch = 0
    iterator_classification_train = iter(train_data_IDRiD)
    iterator_classification_valid = iter(valid_data_IDRiD)
    iterator_segmentation_train = iter(train_data_IDRiD_od)
    iterator_segmentation_valid = iter(valid_data_IDRiD_od)
    train_loss = 0.0
    valid_loss = 0.0
    
    correct_counts_train = 0
    correct_counts_valid = 0
    train_data_size = 0
    valid_data_size = 0

    train_loss2 = 0.0
    valid_loss2 = 0.0
    
    train_loss3 = 0.0
    valid_loss3 = 0.0
    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch, epochs))

        model.train()

        
        try:
            classification_train = iterator_classification_train.next()
        except StopIteration:
            iterator_classification_train = iter(train_data_IDRiD)
            classification_train = iterator_classification_train.next()

        try:
            segmentation_train = iterator_segmentation_train.next()
        except StopIteration:
            iterator_segmentation_train = iter(train_data_IDRiD_od)
            segmentation_train = iterator_segmentation_train.next()       
            
        image_classification_train, labels_classification_train, coordinates_classification_train = classification_train
        image_segmentation_train, image_od_segmentation_train, coordinates_segmentation_train = segmentation_train

        image_classification_train = image_classification_train.to(device)
        labels_classification_train = labels_classification_train.to(device)
        coordinates_classification_train = coordinates_classification_train.to(device)
        image_segmentation_train = image_segmentation_train.to(device)
        image_od_segmentation_train = image_od_segmentation_train.to(device)
        coordinates_segmentation_train = coordinates_segmentation_train.to(device)
        # print(coordinates_classification_train)
        # image_classification_train = image_classification_train.to(device)
        # labels_classification_train = labels_classification_train.to(device)
        # image_segmentation_train = image_segmentation_train.to(device)
        # image_od_segmentation_train = image_od_segmentation_train.to(device)
        
        # input_label = np.array([1])
        input_label = torch.tensor([[1]]).to(device)
        # image_segmentation_train = np.array(image_segmentation_train.cpu())
        # print(image_segmentation_train.shape)
        # model2.set_image(np.array(image_segmentation_train[0]))
        

        batch_input = []
        for i in range(len(image_segmentation_train)):
            temp = {}
            # print(image_segmentation_train[i].shape)
            temp['image'] = prepare_image(image_segmentation_train[i],resize_transform,model2)
            x = torch.unsqueeze(coordinates_segmentation_train[i], 0)
            x = torch.unsqueeze(x, 0)
            temp['point_coords'] = resize_transform.apply_coords_torch(x,image_segmentation_train[i].shape[:2])
            # print(torch.tensor([[coordinates_segmentation_train[i]]]))
            temp['point_labels'] = input_label
            temp['original_size'] = image_segmentation_train[i].shape[:2]
            batch_input.append(temp)

        # batch_output = checkpoint(model2, batch_input, True)
        batch_output = model2(batch_input, multimask_output=True)
        # model2.set_torch_image(prepare_image(image_segmentation_train[0],resize_transform,sam), original_image_size=image_segmentation_train[0].shape[:2])
        # print(coordinates_segmentation_train)
        # masks, scores, logits = model2.predict_torch(
        #     point_coords=coordinates_segmentation_train,
        #     point_labels=input_label,
        #     multimask_output=True,
        # )
        loss3 = 0
        for i in range(len(batch_output)):
            j = 0
            min_dice = 100
            for k in range(3):
                if min_dice > dice(batch_output[i]['masks'][0][k], image_od_segmentation_train[i]):
                    min_dice = dice(batch_output[i]['masks'][0][k], image_od_segmentation_train[i])
                    j = k
            loss3_temp = loss_function3(batch_output[i]['masks'][0][j], image_od_segmentation_train[i])
            loss3_temp.requires_grad_(True)
            loss3 += loss3_temp
        # for i, (mask, score) in enumerate(zip(masks, scores)):
        #     if min_dice > dice(mask, image_od_segmentation_train[0]):
        #         min_dice = dice(mask, image_od_segmentation_train[0])
        #         j = i
        # mask = masks[j]
        # loss3 = loss_function3(mask, image_od_segmentation_train[0])

        batch_input = []
        for i in range(len(image_classification_train)):
            temp = {}
            # print(image_segmentation_train[i].shape)
            temp['image'] = prepare_image(image_classification_train[i],resize_transform,model2)
            x = torch.unsqueeze(coordinates_classification_train[i], 0)
            x = torch.unsqueeze(x, 0)
            temp['point_coords'] = resize_transform.apply_coords_torch(x,image_classification_train[i].shape[:2])
            # print(torch.tensor([[coordinates_segmentation_train[i]]]))
            temp['point_labels'] = input_label
            temp['original_size'] = image_classification_train[i].shape[:2]
            batch_input.append(temp)

        # batch_output = checkpoint(model2, batch_input, True)
        batch_output = model2(batch_input, multimask_output=True)
        # masks, scores, logits = predictor.predict_torch(
        #     point_coords=coordinates_classification_train,
        #     point_labels=input_label,
        #     multimask_output=True,
        # )
        loss2 = 0
        center_t = []
        for i in range(len(batch_output)):
            j = 0
            min_dis = 99999999
            for k in range(3):
                # print(batch_output[i]['masks'][0][k].shape)
                temp = batch_output[i]['masks'][0][k].cpu().numpy()
                gray = np.array(temp).astype('uint8')
                gray *= 76
                # temp = cv2.bitwise_not(temp)
                # threshold = 200
                # ret, binary = cv2.threshold(temp, threshold, 255, cv2.THRESH_BINARY)
                # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # x, y, w, h = cv2.boundingRect(contours[1])
                # gray = np.array(temp).astype('uint8')
                th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) 
                th = cv2.bitwise_not(th)
                x, y, w, h = cv2.boundingRect(th)
                center = torch.tensor([x+w/2, y+h/2]).to(device)
                # print(torch.norm(coordinates_classification_train[i] - center, 2))
                if min_dis > torch.norm(coordinates_classification_train[i] - center, 2):
                    min_dis = torch.norm(coordinates_classification_train[i] - center, 2)
                    j = k
                    opt_x, opt_y, opt_w, opt_h = x, y, w, h
                    opt_center = center
            # mask.append(batch_output[i]['masks'][0][j])
            center_t.append([opt_x, opt_y, opt_w, opt_h])
            loss2_temp = loss_function2(opt_center, coordinates_classification_train[i])
            loss2_temp.requires_grad_(True)
            loss2 += loss2_temp

        # dis = 99999999999999999
        # # opt_x, opt_y, opt_w, opt_h = 0, 0, 0, 0
        # for i, (mask, score) in enumerate(zip(masks, scores)):
        #     mask=mask+1-1
        #     mask=mask*76
        #     gray=np.array(mask).astype('uint8')
        #     th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) 
        #     th = cv2.bitwise_not(th)
        #     x, y, w, h = cv2.boundingRect(th)
        #     center = torch.tensor([x+w/2, y+h/2])
        #     if dis > np.linalg.norm(coordinates_classification_train[0] - center):
        #         dis = np.linalg.norm(coordinates_classification_train[0] - center)
        #         j=i
        #         opt_x, opt_y, opt_w, opt_h = x, y, w, h
        #         opt_center = center
        # mask = masks[j]
        # # print(opt_center)
        # # print(coordinates_classification_train[0])
        # loss2 = loss_function2(opt_center, torch.tensor(coordinates_classification_train[0]))
        # print(image_classification_train.shape)
        loss1 = 0
        t_image = None
        for i in range(len(image_classification_train)):
            # print(image_classification_train[i].shape)
            img_transpose = image_classification_train[i].permute(2, 0, 1)
            temp = torchvision.transforms.functional.to_pil_image(img_transpose)
            image_classification_train_crop = temp.crop((center_t[i][0], center_t[i][1], center_t[i][0] + center_t[i][2], center_t[i][1] + center_t[i][3]))
            image_classification_train_crop = data_transform['train'](image_classification_train_crop)
            image_classification_train_crop = torch.unsqueeze(image_classification_train_crop, 0)
            if t_image == None:
                t_image = image_classification_train_crop
            else:
                t_image = torch.cat((t_image, image_classification_train_crop), dim=0)
            
            # print(image_classification_train_crop.shape)

        output = model(t_image.to(device))
        # print(labels_classification_train[i])
        loss1 += loss_function1(output, labels_classification_train)
        ret, predictions = torch.max(output.data, 1)
        # print(predictions)
        # print(labels_classification_train)
        # print(torch.sum(predictions == labels_classification_train.data))
        correct_counts_train += torch.sum(predictions.data == labels_classification_train.data)
        train_data_size += len(labels_classification_train)
        # print(loss1)
        # print(loss2)
        # print(loss3)
        train_loss += loss1.item()
        train_loss2 += loss2.item()
        train_loss3 += loss3.item()
        optimizer.zero_grad()
        # loss = loss1 + loss2 + loss3
        task_loss = [loss1, loss2, loss3]
        assert len(task_loss) == 3
        optimizer.pc_backward(task_loss) 
        # loss.backward()
        optimizer.step() 

        with torch.no_grad():
#             model.eval()
            model.train()
            
            try:
                classification_valid = iterator_classification_valid.next()
            except StopIteration:
                iterator_classification_valid = iter(valid_data_IDRiD)
                classification_valid = iterator_classification_valid.next()

            try:
                segmentation_valid = iterator_segmentation_valid.next()
            except StopIteration:
                iterator_segmentation_valid = iter(valid_data_IDRiD_od)
                segmentation_valid = iterator_segmentation_valid.next()       
                
            image_classification_valid, labels_classification_valid, coordinates_classification_valid = classification_valid
            image_segmentation_valid, image_od_segmentation_valid, coordinates_segmentation_valid = segmentation_valid

            image_classification_valid = image_classification_valid.to(device)
            labels_classification_valid = labels_classification_valid.to(device)
            coordinates_classification_valid = coordinates_classification_valid.to(device)
            image_segmentation_valid = image_segmentation_valid.to(device)
            image_od_segmentation_valid = image_od_segmentation_valid.to(device)
            coordinates_segmentation_valid = coordinates_segmentation_valid.to(device)
            # print(coordinates_classification_valid)
            # image_classification_valid = image_classification_valid.to(device)
            # labels_classification_valid = labels_classification_valid.to(device)
            # image_segmentation_valid = image_segmentation_valid.to(device)
            # image_od_segmentation_valid = image_od_segmentation_valid.to(device)
            
            # input_label = np.array([1])
            input_label = torch.tensor([[1]]).to(device)
            # image_segmentation_valid = np.array(image_segmentation_valid.cpu())
            # print(image_segmentation_valid.shape)
            # model2.set_image(np.array(image_segmentation_valid[0]))
            

            batch_input = []
            for i in range(len(image_segmentation_valid)):
                temp = {}
                # print(image_segmentation_valid[i].shape)
                temp['image'] = prepare_image(image_segmentation_valid[i],resize_transform,model2)
                x = torch.unsqueeze(coordinates_segmentation_valid[i], 0)
                x = torch.unsqueeze(x, 0)
                temp['point_coords'] = resize_transform.apply_coords_torch(x,image_segmentation_valid[i].shape[:2])
                # print(torch.tensor([[coordinates_segmentation_valid[i]]]))
                temp['point_labels'] = input_label
                temp['original_size'] = image_segmentation_valid[i].shape[:2]
                batch_input.append(temp)

            # batch_output = checkpoint(model2, batch_input, True)
            batch_output = model2(batch_input, multimask_output=True)
            # model2.set_torch_image(prepare_image(image_segmentation_valid[0],resize_transform,sam), original_image_size=image_segmentation_valid[0].shape[:2])
            # print(coordinates_segmentation_valid)
            # masks, scores, logits = model2.predict_torch(
            #     point_coords=coordinates_segmentation_valid,
            #     point_labels=input_label,
            #     multimask_output=True,
            # )
            loss3 = 0
            for i in range(len(batch_output)):
                j = 0
                min_dice = 100
                for k in range(3):
                    if min_dice > dice(batch_output[i]['masks'][0][k], image_od_segmentation_valid[i]):
                        min_dice = dice(batch_output[i]['masks'][0][k], image_od_segmentation_valid[i])
                        j = k
                loss3_temp = loss_function3(batch_output[i]['masks'][0][j], image_od_segmentation_valid[i])
                loss3_temp.requires_grad_(True)
                loss3 += loss3_temp
            # for i, (mask, score) in enumerate(zip(masks, scores)):
            #     if min_dice > dice(mask, image_od_segmentation_valid[0]):
            #         min_dice = dice(mask, image_od_segmentation_valid[0])
            #         j = i
            # mask = masks[j]
            # loss3 = loss_function3(mask, image_od_segmentation_valid[0])

            batch_input = []
            for i in range(len(image_classification_valid)):
                temp = {}
                # print(image_segmentation_valid[i].shape)
                temp['image'] = prepare_image(image_classification_valid[i],resize_transform,model2)
                x = torch.unsqueeze(coordinates_classification_valid[i], 0)
                x = torch.unsqueeze(x, 0)
                temp['point_coords'] = resize_transform.apply_coords_torch(x,image_classification_valid[i].shape[:2])
                # print(torch.tensor([[coordinates_segmentation_valid[i]]]))
                temp['point_labels'] = input_label
                temp['original_size'] = image_classification_valid[i].shape[:2]
                batch_input.append(temp)

            # batch_output = checkpoint(model2, batch_input, True)
            batch_output = model2(batch_input, multimask_output=True)
            # masks, scores, logits = predictor.predict_torch(
            #     point_coords=coordinates_classification_valid,
            #     point_labels=input_label,
            #     multimask_output=True,
            # )
            loss2 = 0
            center_t = []
            for i in range(len(batch_output)):
                j = 0
                min_dis = 99999999
                for k in range(3):
                    # print(batch_output[i]['masks'][0][k].shape)
                    temp = batch_output[i]['masks'][0][k].cpu().numpy()
                    gray = np.array(temp).astype('uint8')
                    gray *= 76
                    # temp = cv2.bitwise_not(temp)
                    # threshold = 200
                    # ret, binary = cv2.threshold(temp, threshold, 255, cv2.THRESH_BINARY)
                    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # x, y, w, h = cv2.boundingRect(contours[1])
                    # gray = np.array(temp).astype('uint8')
                    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) 
                    th = cv2.bitwise_not(th)
                    x, y, w, h = cv2.boundingRect(th)
                    center = torch.tensor([x+w/2, y+h/2]).to(device)
                    # print(torch.norm(coordinates_classification_valid[i] - center, 2))
                    if min_dis > torch.norm(coordinates_classification_valid[i] - center, 2):
                        min_dis = torch.norm(coordinates_classification_valid[i] - center, 2)
                        j = k
                        opt_x, opt_y, opt_w, opt_h = x, y, w, h
                        opt_center = center
                # mask.append(batch_output[i]['masks'][0][j])
                center_t.append([x,y,w,h])
                loss2_temp = loss_function2(opt_center, coordinates_classification_valid[i])
                loss2_temp.requires_grad_(True)
                loss2 += loss2_temp

            # dis = 99999999999999999
            # # opt_x, opt_y, opt_w, opt_h = 0, 0, 0, 0
            # for i, (mask, score) in enumerate(zip(masks, scores)):
            #     mask=mask+1-1
            #     mask=mask*76
            #     gray=np.array(mask).astype('uint8')
            #     th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) 
            #     th = cv2.bitwise_not(th)
            #     x, y, w, h = cv2.boundingRect(th)
            #     center = torch.tensor([x+w/2, y+h/2])
            #     if dis > np.linalg.norm(coordinates_classification_valid[0] - center):
            #         dis = np.linalg.norm(coordinates_classification_valid[0] - center)
            #         j=i
            #         opt_x, opt_y, opt_w, opt_h = x, y, w, h
            #         opt_center = center
            # mask = masks[j]
            # # print(opt_center)
            # # print(coordinates_classification_valid[0])
            # loss2 = loss_function2(opt_center, torch.tensor(coordinates_classification_valid[0]))
            # print(image_classification_valid.shape)
            loss1 = 0
            t_image = None
            for i in range(len(image_classification_valid)):
                # print(image_classification_valid[i].shape)
                img_transpose = image_classification_valid[i].permute(2, 0, 1)
                temp = torchvision.transforms.functional.to_pil_image(img_transpose)
                image_classification_valid_crop = temp.crop((center_t[i][0], center_t[i][1], center_t[i][0] + center_t[i][2], center_t[i][1] + center_t[i][3]))
                image_classification_valid_crop = data_transform['valid'](image_classification_valid_crop)
                image_classification_valid_crop = torch.unsqueeze(image_classification_valid_crop, 0)
                if t_image == None:
                    t_image = image_classification_valid_crop
                else:
                    t_image = torch.cat((t_image, image_classification_valid_crop), dim=0)
                
                # print(image_classification_valid_crop.shape)

            output = model(t_image.to(device))
            # print(labels_classification_valid[i])
            loss1 += loss_function1(output, labels_classification_valid)
            ret, predictions = torch.max(output.data, 1)
            # print(predictions)
            # print(labels_classification_valid)
            # print(torch.sum(predictions == labels_classification_valid.data))
            correct_counts_valid += torch.sum(predictions.data == labels_classification_valid.data)
            valid_data_size += len(labels_classification_valid)
            # print(loss1)
            # print(loss2)
            # print(loss3)
            valid_loss += loss1.item()
            valid_loss2 += loss2.item()
            valid_loss3 += loss3.item()



        if epoch%50==0:       
            avg_train_loss = train_loss/train_data_size
            avg_train_acc = correct_counts_train/train_data_size

            avg_valid_loss = valid_loss/valid_data_size
            avg_valid_acc = correct_counts_valid/valid_data_size

            history.append([avg_train_loss, avg_valid_loss,
                        avg_train_acc.cpu(), avg_valid_acc.cpu()])

            if best_acc < avg_valid_acc:
                best_acc = avg_valid_acc
                best_epoch = epoch

            # epoch_end = time.time()

            print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%".format(
                epoch, avg_train_loss, avg_train_acc *
                100, avg_valid_loss, avg_valid_acc*100
            ))
            print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(
                best_acc, best_epoch))

            print("Coordinates Training Loss: {:.4f},  Coordinates Validation Loss: {:.4f}".format(
                train_loss2/train_data_size, valid_loss2/valid_data_size))  
            
            print("Dice Training Loss: {:.4f},  Dice Validation Loss: {:.4f}".format(
                train_loss3/train_data_size, valid_loss3/valid_data_size))  
            train_loss = 0.0
            valid_loss = 0.0
            
            correct_counts_train = 0
            correct_counts_valid = 0
            train_data_size = 0
            valid_data_size = 0

            train_loss2 = 0.0
            valid_loss2 = 0.0
            
            train_loss3 = 0.0
            valid_loss3 = 0.0
            # torch.save(model, 'models/'+dataset+'_model_'+str(epoch+1)+'.pt')
    return model, model2, history


num_epochs = 10000
trained_model, trained_model2, history = train_and_valid(resnet50, sam, loss_func1, loss_func2, loss_func3, optimizer, num_epochs)
torch.save(trained_model, 'checkpoints/'+dataset+'.pth')

history = np.array(history)
plt.subplot(211)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
# plt.ylim(0, 1)
# plt.savefig(dataset+'_loss_curve.png')
# plt.show()
plt.subplot(212)
plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
# plt.ylim(0, 1)
plt.savefig('result_images/'+dataset+'.png')
plt.show()
