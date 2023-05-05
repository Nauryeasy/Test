from aiogram.dispatcher.filters.state import StatesGroup, State
import pickle
import torch
import torch.nn as nn

TOKEN = '5983098073:AAGCUbZSdcrdZpmqszj2_5xih039V28Zyt4'


class Autoencoder_Working(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(len(unique_emojies), 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(68, 16, 3, stride=2, padding=1),  # 64
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 64, 3, stride=2, padding=1),  # 32
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, stride=2, padding=1),  # 16
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, 3, stride=2, padding=1),  # 8
            nn.InstanceNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 4096, 3, stride=2, padding=1),  # 4
            nn.InstanceNorm2d(4096),
            nn.Conv2d(4096, 4096, 3, stride=2, padding=1),  # 2
            nn.InstanceNorm2d(4096)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048 + 64, 4096, 4, stride=2, padding=1),  # 4
            nn.InstanceNorm2d(4096),
            nn.ReLU(),
            nn.ConvTranspose2d(4096, 1024, 4, stride=2, padding=1),  # 8
            nn.InstanceNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1),  # 16
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 32, 4, stride=2, padding=1),  # 32
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 8, 4, stride=2, padding=1),  # 64
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),  # (n,4,128,128)
            nn.InstanceNorm2d(4),
            nn.Sigmoid()
        )

    def Encoder_func(self, x, label_encode):
        label_encode = label_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, label_encode], 1)

        encoded = self.encoder(x)  # свертка

        return encoded

    def Decoder_func(self, x, label_decode):
        label_decode = label_decode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, label_decode], 1)
        decoded = self.decoder(x)

        return decoded

    def _sample_latent(self, h_enc):
        mu = h_enc[:, :2048]
        log_sigma = h_enc[:, 2048:]
        sigma = torch.exp(log_sigma)

        return mu + sigma * torch.randn_like(sigma), mu, sigma  # Reparameterization trick

    def latent_loss(self, mu, sigma):
        mean_sq = mu ** 2
        stddev_sq = sigma ** 2
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def forward(self, tensor, label_encode, label_decode):
        label_encode, label_decode = torch.tensor(label_encode, dtype=torch.long), torch.tensor(label_decode,
                                                                                                dtype=torch.long)

        # КОДИРОВНИЕ
        encoded = self.Encoder_func(tensor, self.embeddings(label_encode))
        # print(encoded.size())

        # LOSS
        encoded, mu, sigma = self._sample_latent(encoded)
        loss = self.latent_loss(mu, sigma)

        # ДЕКОДИРОВАНИЕ
        decoded = self.Decoder_func(encoded, self.embeddings(label_decode))
        # print(decoded.size())
        return decoded, loss


class Form(StatesGroup):
    sticker_got = State()
    sticker_not_got = State()
    make_sticker_pack = State()


with open("SaveModel/metadata.pkl", "rb") as metadata:
    emojies = pickle.load(metadata)

for key, value in emojies.items():
    lst = []
    for _, value_im in value.items():
        lst.append(value_im["emoji"])

    emojies[key] = lst

unique_emojies = []

for key in emojies:
    unique_emojies.extend(emojies[key])
unique_emojies = list(set(unique_emojies))

# try:
with open("SaveModel/model.pkl", "rb") as file:
    model = pickle.load(file)
print('Model is load!')
# except
print('Load Model Error!')
