
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms

import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=3, padding=1),  # b, 64, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 64, 5, 5
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # b, 64, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 64, 2, 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2),  # b, 64, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 5, stride=3,
                               padding=1),  # b, 64, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,
                                          weight_decay=1e-8)

    def forward(self, x):
        x = self.encoder(x)
        # Apply the transform to each image in the batch

        x = self.decoder(x)
        return x

    def train(self, x_train_augmented, x_train, batch_size, epochs):
        train_loss = []

        # Wrap your loop with tqdm to display a progress bar
        for epoch in range(epochs):
            for i in tqdm(range(0, len(x_train), batch_size), desc=f'Epoch {epoch+1}/{epochs}'):
                batch_augmented = x_train_augmented[i:i+batch_size]
                batch = x_train[i:i+batch_size]
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                output = self.forward(batch_augmented)
                loss = self.criterion(output, batch)
                loss.backward()
                self.optimizer.step()
            train_loss.append(loss.detach().numpy())
            print(
                f'Epoch {epoch+1}/{epochs}, Train Loss: {loss:.4f}')

        # Defining the Plot Style
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Plotting the last 100 values
        plt.plot(train_loss[-epochs:], label='Train Loss')
        plt.title('Autoencoder Model Loss')
        plt.legend()
        plt.show()

        print('Finished Training')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def extract_features(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return x

    def summary(self):
        print(self)
