from shutil import copyfile
import os
import random
def split_data(path,des_path,train_percent=8,validation_percent=1,test_percent=1):
    #防呆
    if train_percent+validation_percent+test_percent!=10:
        print("train_percent+validation_percent+test_percent要等於10")
        return False
    #建立所需資料夾
    os.mkdir(des_path)
    os.mkdir(os.path.join(des_path,"train"))
    os.mkdir(os.path.join(des_path,"validation"))
    os.mkdir(os.path.join(des_path,"test"))
    
    for i in os.listdir(path):
        os.mkdir(os.path.join(des_path,"train",i))
        os.mkdir(os.path.join(des_path,"validation",i))
        os.mkdir(os.path.join(des_path,"test",i))
        
        #每個圖片的檔名list(不包含路徑)
        pic_path=os.listdir(os.path.join(path,i)) 
        #洗牌
        random.shuffle(pic_path) #洗牌
        #算比例
        train_num=len(pic_path)//10*train_percent
        validation_num=len(pic_path)//10*validation_percent
        test_num=len(pic_path)//10*test_percent

        
        for j in pic_path[:train_num]:
            copyfile(os.path.join(path,i,j),os.path.join(des_path,"train",i,j))
        for j in pic_path[train_num:train_num+validation_num]:
            copyfile(os.path.join(path,i,j),os.path.join(des_path,"validation",i,j))
        for j in pic_path[train_num+validation_num:train_num+validation_num+test_num]:
            copyfile(os.path.join(path,i,j),os.path.join(des_path,"test",i,j))

IMAGE_GEN_DIR="C:\\Experiment\\Mask_RCNN\\cvc612"
DEST_PATH="C:\\Experiment\\Mask_RCNN\\cvc612_data_split"
split_data(IMAGE_GEN_DIR,DEST_PATH,8,1,1)