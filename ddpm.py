'''
    文本生成可以看成n个字的生成 词表大小为m 那么一个句子可以看成 m*n 的矩阵 最后查表就可以解码
'''
import os
import torch
import torch.nn as nn
# from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
# from utils import *
from my_transformer import WfTransformer
import logging
# from torch.utils.tensorboard import SummaryWriter
import json
from transformers import BertTokenizer,BertTokenizerFast,T5Tokenizer
from dataset import CurDataset
from torch.utils.data import DataLoader
import numpy as np
import random

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    '''
        noise_steps : 添加噪声的步骤
        beta_start : 开始时的高斯噪声的beta
        beta_end : 结束时的高斯噪声的beta
        seq_len : 句子长度
        vocabulary_size : 字典大小
    '''
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, seq_len=128, vocabulary_size=30522, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # self.img_size = img_size
        self.seq_len = seq_len
        self.vocabulary_size = vocabulary_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # 累计乘法

    def prepare_noise_schedule(self):
        '''
            生成一个噪声序列，总共的噪声步数是noise_steps
        '''
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None] # 将alpha 扩展成为(b,n,m) 方便对一个批次的文本做乘法
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)) # 为每个batch 的数据随机一个时间点

    def sample(self, model, n):
        logging.info(f"Sampling {n} new sentences....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.seq_len, self.vocabulary_size)).to(self.device) # 随机生成长度为 128 字典大小为 25535的句子
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 1).type(torch.uint8)
        return x

def save_seq(seqlis,path):
    pass

def read_data(dirname):
    all_data = []
    for file in os.listdir(dirname):
        if(file.split('.')[-1] != 'jsonl'):
            continue
        doc_seq_data = []
        with open(os.path.join(dirname,file),'r') as f:
            for lin in f.readlines():
                doc_seq_data.append(json.loads(lin.strip()))
        all_data+=doc_seq_data
    return all_data

def get_data():
    '''
        获取文本
        这里直接使用bert的tokenizer来做
    '''
    # data = read_data('/opt/data/private/sxu/fwang/idea3/code/data/json/clear_seq_train_data/')    
    data = read_data('train')
    c_dataset = CurDataset(data)

    diff_dataloader = DataLoader(c_dataset, batch_size=4,
                              shuffle=True, collate_fn=c_dataset.collate_fn)

    return diff_dataloader
def train(device,lr,epochs):
    # setup_logging(args.run_name)
    # dataloader = get_data(args)
    dataloader = get_data()
    # model = UNet().to(device)
    model = WfTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, batchdata in enumerate(pbar):
            sentences = batchdata['seq_tok']
            # images = images.to(device) 
            t = diffusion.sample_timesteps(sentences.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(sentences, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        pred_seq = diffusion.sample(model, n=4) # 生成的句子
        # print(pred_seq)
        # pred_seq_path = 'xxx'
        # save_seq(pred_seq,pred_seq_path)
        # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def seed_everything(seed=1226):
    '''
    设置整个开发环境的seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True 

if __name__ == '__main__':
    seed_everything()
    # diffusion = Diffusion(device='cpu')
    # sentences = torch.randn((3,3,5))
    # print(sentences)
    # t = diffusion.sample_timesteps(sentences.shape[0])
    # x_t, noise = diffusion.noise_images(sentences, t)
    # model = WfTransformer(c_in=5,c_out=5)
    # predicted_noise = model(x_t, t)
    # print(predicted_noise)
    train('cuda',1e-5,100)