import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import torchvision
from model.test_model import model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def SMDR_IS_map(underwater_path, SMDR_IS):
    original = Image.open(underwater_path).convert('RGB')
    enhance_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    with torch.no_grad():
        original = enhance_transforms(original)
        original = original.cuda().unsqueeze(0)
        torch.cuda.synchronize()
        start = time.time()
        enhanced = SMDR_IS(original)
        torch.cuda.synchronize()
        end = time.time()

    return enhanced, original, end - start


if __name__ == '__main__':

    test_path=r'data/input'

    pth_path= r'checkpoint/checkpoint.ckpt'
    checkpoint = torch.load(pth_path)
    Model = model()
    Model.load_state_dict(checkpoint['state_dict'])
    Model = Model.cuda()

    test_list = os.listdir(test_path)

    fps_avg = []
    time_avg = []
    for i, image in enumerate(test_list):
        print(image)
        if image.split('.')[-1] == 'png' or image.split('.')[-1] == 'jpg':
            enhanced, original, time_num = SMDR_IS_map(os.path.join(test_path, image), Model)
            torchvision.utils.save_image(enhanced, r'data/result/' + image.replace('.tif', '.jpg'))
            fps_avg.append(1 / time_num)
            time_avg.append(time_num)

    print('avg fps:', np.mean(fps_avg))
    print('avg fps:', np.mean(time_avg))