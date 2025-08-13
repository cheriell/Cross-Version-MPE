import torch
import torch.nn as nn
import torch.nn.functional as F


# Model parameters
min_pitch = 24
n_in = 6
n_ch = [70, 70, 50, 10]
n_prefilt_layers = 5
n_bins_in = 216
n_bins_out = 72
a_lrelu = 0.3
p_dropout = 0.2
last_kernel_size = n_bins_in // 3 + 1 - n_bins_out


class aoe_cnn_model(torch.nn.Module):
    """Here, we extend the basic ResCNN to the AoECNN model.
    We use the default model hyperparameters from the basic ResCNN model, to ensure that the models are comparable.

    The model architecutre is descirbed in notes/experimental_notes_02_Baselines.md, where the [Encoder CNN], together with the [MPE Model Backend], make up the ResCNN model.

    Model parameters (copied form the ts config files):
    "model_params": {
        "min_pitch": 24,
        "n_chan_input": 6,
        "n_chan_layers": [70, 70, 50, 10],
        "n_prefilt_layers": 5,
        "residual": true,
        "n_bins_in": 216,
        "n_bins_in_comment": "num_octaves(6)*12*3",
        "n_bins_out": 72,
        "a_lrelu": 0.3,
        "p_dropout": 0.2
    }    
    """

    def __init__(self, ae_layers):
        super().__init__()

        self.ae_layers = ae_layers

        assert ae_layers >=2 and ae_layers <= n_prefilt_layers - 1

        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])
        self.encoder = encoder_cnn(ae_layers)
        self.decoder = decoder_cnn(ae_layers)
        self.mpe_model_backend = mpe_model_backend(ae_layers)

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x_enc = self.encoder(x_norm)
        x_recon = self.decoder(x_enc)
        y_pred = self.mpe_model_backend(x_enc)
        return x_recon, y_pred
    
    def predict(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x_enc = self.encoder(x_norm)
        y_pred = self.mpe_model_backend(x_enc)
        y_pred = F.sigmoid(y_pred)
        return y_pred


class aoe_cnn_model_no_recon(torch.nn.Module):

    def __init__(self, ae_layers):
        super().__init__()

        self.ae_layers = ae_layers

        assert ae_layers >=2 and ae_layers <= n_prefilt_layers - 1

        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])
        self.encoder = encoder_cnn(ae_layers)
        self.mpe_model_backend = mpe_model_backend(ae_layers)

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x_enc = self.encoder(x_norm)
        y_pred = self.mpe_model_backend(x_enc)
        return y_pred

    def predict(self, x):
        y_pred = self(x)
        y_pred = F.sigmoid(y_pred)
        return y_pred
    

class encoder_cnn(torch.nn.Module):

    def __init__(self, ae_layers):
        """This is the Encoder CNN part of the AoECNN model.

        Parameters
        ----------
        ae_layers : int
            The number of layers in the autoencoder.
        """
        super(encoder_cnn, self).__init__()

        self.ae_layers = ae_layers

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_ch[0], kernel_size=(15, 15), padding=(7, 7), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Dropout(p=p_dropout)
        )
        self.prefilt_list = nn.ModuleList()
        for _ in range(1, ae_layers):
            self.prefilt_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[0], kernel_size=(15,15), padding=(7,7), stride=(1,1)),
                nn.LeakyReLU(negative_slope=a_lrelu),
                nn.MaxPool2d(kernel_size=(3,1), stride=(1,1), padding=(1,0)),
                nn.Dropout(p=p_dropout)
            ))

    def forward(self, x):
        x = self.conv(x)
        for i in range(0, self.ae_layers - 1):
            x_new = self.prefilt_list[i](x)
            x = x_new + x
        return x
    

class decoder_cnn(torch.nn.Module):

    def __init__(self, ae_layers):
        """This is the Decoder CNN part of the AoECNN model.

        Parameters
        ----------
        ae_layers : int
            The number of layers in the autoencoder.
        """
        super(decoder_cnn, self).__init__()

        self.ae_layers = ae_layers

        self.prefilt_list = nn.ModuleList()
        for _ in range(1, ae_layers):
            self.prefilt_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[0], kernel_size=(15, 15), padding=(7, 7), stride=(1, 1)),
                nn.LeakyReLU(negative_slope=a_lrelu),
                nn.Dropout(p=p_dropout)
            ))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_in, kernel_size=(15, 15), padding=(7, 7), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
        )

    def forward(self, x):
        for i in range(0, self.ae_layers - 1):
            x_new = self.prefilt_list[i](x)
            x = x_new + x
        x_recon = self.conv(x)
        return x_recon
    

class mpe_model_backend(torch.nn.Module):

    def __init__(self, ae_layers):
        """This is the MPE Model Backend part of the AoECNN model.
        
        Parameters
        ----------
        ae_layers : int
            The number of layers in the autoencoder.
        """
        super(mpe_model_backend, self).__init__()

        self.ae_layers = ae_layers

        self.prefilt_list = nn.ModuleList()
        for _ in range(ae_layers, n_prefilt_layers):
            self.prefilt_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[0], kernel_size=(15,15), padding=(7,7), stride=(1,1)),
                nn.LeakyReLU(negative_slope=a_lrelu),
                nn.MaxPool2d(kernel_size=(3,1), stride=(1,1), padding=(1,0)),
                nn.Dropout(p=p_dropout)
            ))
        
        # Binding to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            # nn.Sigmoid()  # this is no longer needed with BCEWithLogitsLoss
        )

    def forward(self, x):
        for i in range(len(self.prefilt_list)):
            x_new = self.prefilt_list[i](x)
            x = x_new + x
        x = self.conv2(x)
        x = self.conv3(x)
        y_pred = self.conv4(x)
        return y_pred