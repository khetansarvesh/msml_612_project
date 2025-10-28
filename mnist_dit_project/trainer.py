class Trainer:
    def __init__(self, model, data_loader, optimizer, device, num_epochs=40, checkpoint_path='mnist_dit_ckpt.pth'):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            losses = []
            for im, _ in tqdm(self.data_loader):
                self.optimizer.zero_grad()
                im = im.float().to(self.device)
                noise = torch.randn_like(im).to(self.device)
                t = torch.randint(low=0, high=1000, size=(im.shape[0],)).to(self.device)
                noisy_im = self.add_noise(im, noise, t)
                noise_pred = self.model(noisy_im, t)
                loss = torch.nn.MSELoss()(noise_pred, noise)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            print(f'Finished epoch: {epoch + 1} | Loss: {np.mean(losses)}')
            self.save_checkpoint()

    def add_noise(self, im, noise, t):
        betas = torch.linspace(0.0001, 0.02, 1000).to(self.device)
        alpha_cum_prod = torch.cumprod(1. - betas, dim=0).to(self.device)
        return torch.sqrt(alpha_cum_prod[t])[:, None, None, None] * im + torch.sqrt(1 - alpha_cum_prod[t])[:, None, None, None] * noise

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)