
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_output):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.ReLU()
        )

        self.amplitude_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 512, 1024),    
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_output)
        )

        self.width_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 512, 1024),    
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_output)
        )

        self.type_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 512, 1024),    
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_output),
            nn.Sigmoid()  
        )

    def forward(self, spectrum):
        conv_out = self.encoder(spectrum)
        reconstructed_spectrum = self.decoder(conv_out)
        amplitude = self.amplitude_predictor(conv_out)
        width = self.width_predictor(conv_out)
        type = self.type_predictor(conv_out)
        return amplitude, width, reconstructed_spectrum, type
