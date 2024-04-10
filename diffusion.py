import pickle
import torch
import torch.nn as nn
import numpy as np
from transformer import ResNet

class diffusion(nn.Module):
    def __init__(self, config, device):
        super().__init__() # ensure that the diffusion model properly inherits all the functionality of a pytorch module
        self.device = device # where will the model run (cpu or gpu)
        self.config_diff = config['diffusion'] # settings for diffusion model 
        var, _ = pickle.load(open('preprocess/data/var.pkl', 'rb')) # preprocessing 
        self.lv = len(var)
        self.res_model = ResNet(self.config_diff, self.device) # part of the transformer for sequence processing
        # parameters for diffusion model
        self.num_steps = self.config_diff['num_steps'] # number of steps
        self.beta = np.linspace(self.config_diff['beta_start'] ** 0.5, self.config_diff['beta_end'] ** 0.5, self.num_steps) ** 2 # generation of beta values
        self.alpha_hat = 1 - self.beta # scale of the noise added at each step 
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1)  # convertion to pytorch tensor 

    def process(self, batch): # prepare the input batch data for further processing by the model 
        samples_x = batch['samples_x'].to(self.device).float() # transfers batch data samples x to device and converts to float 
        samples_y = batch['samples_y'].to(self.device).float() # transfers batch data samples y to device and converts to float 
        info = batch['info'].to(self.device) # transfers batch info data to device
        
        return samples_x, samples_y, info

    def forward(self, batch, size_x, size_y): # forward pass of the model 
        samples_x, samples_y, info = self.process(batch) # process input batch 
        t = torch.randint(0, self.num_steps, [len(samples_x)]).to(self.device) # generate tensor t of random ints (time steps for diffusion process) and the length of samples x
        current_alpha = self.alpha_torch[t] # selection of corresponding alpha value for each sample to scale samples y 
        noise = torch.randn((len(samples_x), size_y)).to(samples_y.device) # generation of random noise added to samples y ans ensured zthat its on the same device
        mask_x = samples_x[:, 3] # indicate which elements in samples x and y should be considered during certain operations
        mask_y = samples_y[:, 3]
        samples_x[:, 0] = torch.where(mask_x == 1, samples_x[:, 0], self.lv) # handling missing data 
        samples_x[:, 1] = torch.where(mask_x == 1, samples_x[:, 1], -1)
        samples_y[:, 0] = torch.where(mask_y == 1, samples_y[:, 0], self.lv)
        samples_y[:, 1] = torch.where(mask_y == 1, samples_y[:, 1], -1)
        samples_y[:, 2] = ((current_alpha ** 0.5) * samples_y[:, 2] + ((1.0 - current_alpha) ** 0.5) * noise) * mask_y # weighted sum of the original samples y and the generated noise (eq. page 5 last line) modified by mask
        predicted = self.res_model(samples_x, samples_y, info, t) # prediction
        residual = torch.where(mask_y == 1, noise - predicted, 0) # 
        loss = (residual ** 2).sum() / info[:, 2].sum() # 

        return loss

    def forecast(self, samples_x, samples_y, info, n_samples): # reverse diffusion process / prediction over n samples 
        generation = torch.zeros(n_samples, samples_y.shape[0], samples_y.shape[-1]).to(self.device) # storing the gerated forecasts 
        for i in range(n_samples): # iterate thorugh every sample
            samples_y[:, 2] = torch.randn_like(samples_y[:, 2]) * samples_y[:, 3] # initializing with random noise 
            for t in range(self.num_steps - 1, -1, -1): # iterate over diffusion steps , reverse diffusion process
                mask_x = samples_x[:, 3]
                mask_y = samples_y[:, 3]
                samples_x[:, 0] = torch.where(mask_x == 1, samples_x[:, 0], self.lv)
                samples_x[:, 1] = torch.where(mask_x == 1, samples_x[:, 1], -1)
                samples_y[:, 0] = torch.where(mask_y == 1, samples_y[:, 0], self.lv)
                samples_y[:, 1] = torch.where(mask_y == 1, samples_y[:, 1], -1)
                predicted = self.res_model(samples_x, samples_y, info, torch.tensor([t]).to(self.device)) # prediction
                # Denoising operation 
                coeff1 = 1 / self.alpha_hat[t] ** 0.5 # coefficient equation (3)
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5 # coefficient equation (3) with beta = 1 - alpha_hat
                samples_y[:, 2] = coeff1 * (samples_y[:, 2] - coeff2 * predicted) * samples_y[:, 3] # equation (3)
                if t > 0:
                    noise = torch.randn_like(samples_y[:, 2]) * samples_y[:, 3] # epsilon = N(0,I)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5 # equation (4)
                    samples_y[:, 2] += sigma * noise

            generation[i] = samples_y[:, 2].detach()
            
        return generation.permute(1, 2, 0)

    def evaluate(self, batch, n_samples):
        samples_x, samples_y, info = self.process(batch)
        with torch.no_grad():
            generation = self.forecast(samples_x, samples_y, info, n_samples)
            
        return generation, samples_y, samples_x