import torch
import torch.nn as nn
from torchvision.transforms import ToTensor,ToPILImage
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from PIL import Image
import random
import matplotlib.pyplot as plt
def add_gaussian_noise(data,mean=0,std=0.1):
    """
    添加高斯模糊
    """
    device = data.device
    # 和torch.randn功能类似，默认标准差是1; 区别是torch.normal可以定制mean和std
    noise = torch.normal(mean,std,size=data.size()).to(device)
    noisy_data = data + noise
    # 限制取值在0，1范围内 
    noisy_data = torch.clamp(noisy_data, 0, 1)
    return noisy_data
#
# AutoEncoder
class AE(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.ReLU(),
        )

    def forward(self, inputs):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat


if __name__ == "__main__":

    batch_size = 32
    hidden_size = 128
    output_size = 28 * 28
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-3
    epoch = 1
    
    ds_test = MNIST(root='F:\VsConde-Python\chen\data', download=False, train=False, transform=ToTensor())
    dl_train = DataLoader(ds_test, batch_size=batch_size)

    # for images, labels in dl_train:
    #     noisy_images= add_gaussian_noise(images)
    #     # ToPILImage()(noisy_images.squeeze()).show() # 查看图片
    #     break
    
    ae = AE(hidden_size, output_size)
    ae = ae.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    for epoch in range(epoch):
        tpbar = tqdm(dl_train)

        for imgs, _ in tpbar:
            imgs = imgs.to(device)
            imgs = imgs.reshape(imgs.shape[0], -1)  # 将图像展平为向量
            noise_img = add_gaussian_noise(imgs)  # 添加高斯噪声
            logits = ae(noise_img)  # 通过自编码器得到输出
            loss = loss_fn(logits, imgs)  # 计算损失

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            ae.zero_grad()  # 清空梯度

            tpbar.set_description(f'epoch: {epoch+1} train_loss: {loss.item():.4f}')

    # 模型测试
    img_index = random.choice(list(range(len(ds_test))))
    img_real, label = ds_test[img_index]
    
    # 搬到GPU上
    img_real = img_real.to(device)
    img_noise = add_gaussian_noise(img_real)

    # 通过自编码器进行预测
    predict = ae(img_noise.reshape(1, -1))
    predict = torch.clamp(predict, 0, 1)  # 限制预测值在 [0, 1] 范围内

    # 将图像 reshape 为 28x28
    # 搬回CPU, 因为plt.imshow 函数需要的是 CPU 上的 NumPy 数组
    real_img = img_real.reshape(28, 28).to("cpu")
    pred_img = predict.reshape(28, 28).to("cpu")
    noise_img = img_noise.reshape(28, 28).to("cpu")

    # 显示图像
    plt.subplot(1, 3, 1)
    plt.title('real')
    plt.imshow(real_img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('noise')
    plt.imshow(noise_img, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('pred')
    plt.imshow(pred_img.detach().numpy(), cmap='gray')

    plt.show()