import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
import cv2
import PIL
import numpy as np

from model import convnext_base as create_model


def main(folderpath,k):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 6
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # img_path = folderpath
    file_list = [f for f in os.listdir(folderpath) if f.endswith('.nii.gz')]
    prob_list = [['0','1','2','3','4','5','class']]
    # prob_list = prob_list.append ([] * 6)
    for img_path in file_list:
        # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img_path = os.path.join(folderpath, img_path)
        img0 = sitk.ReadImage(img_path)
        imgnp = sitk.GetArrayFromImage(img0)
        imgnp = imgnp[0, :, :]
        imgnp2 = ((imgnp - imgnp.min()) / (imgnp.max() - imgnp.min())) * 255
        imgnp3 = imgnp2.astype("uint8")
        img = cv2.cvtColor(imgnp3, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        # img = Image.open(img_path)
        # plt.imshow(img)
        # # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)



        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = create_model(num_classes=num_classes).to(device)
        # load model weights
        model_weight_path = "./weights3/best_model.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        # plt.title(print_res)
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()))

        prob_list.append(np.append(predict.numpy(), predict_cla))
        # plt.show()


    # 转换为 Pandas DataFrame
    df = pd.DataFrame(prob_list[1:], columns=prob_list[0])

    # 保存到 CSV 文件

    csv_file_path = os.path.join('./predict_result3', str(k)+'.csv')
    df.to_csv(csv_file_path, index=False)


if __name__ == '__main__':
    oripath = '/home/sx91/Transformation/deep-learning-for-image-processing-master/data_set/Cer/test'
    if os.path.exists("./predict_result3") is False:
        os.makedirs("./predict_result3")

    for k in range(6):
        folderpath =os.path.join(oripath, str(k))
        main(folderpath, k)
