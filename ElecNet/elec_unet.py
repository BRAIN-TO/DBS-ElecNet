import torch
import torch.nn as nn
import os


class DoubleConv(nn.Module):
    """(Conv3d => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super(UNet3D, self).__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(base_filters*2, base_filters*4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = DoubleConv(base_filters*4, base_filters*8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_filters*8, base_filters*16)

        # Decoder
        self.up4 = nn.ConvTranspose3d(base_filters*16, base_filters*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_filters*16, base_filters*8)
        self.up3 = nn.ConvTranspose3d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_filters*8, base_filters*4)
        self.up2 = nn.ConvTranspose3d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_filters*4, base_filters*2)
        self.up1 = nn.ConvTranspose3d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_filters*2, base_filters)

        # Output
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      
        p1 = self.pool1(e1)    
        e2 = self.enc2(p1)     
        p2 = self.pool2(e2)    
        e3 = self.enc3(p2)     
        p3 = self.pool3(e3)    
        e4 = self.enc4(p3)     
        p4 = self.pool4(e4)    

        # Bottleneck
        b = self.bottleneck(p4) 

        # Decoder
        u4 = self.up4(b)        
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)       
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)       
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)       
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.final_conv(d1)  # (B, 1, 128, 128, 128)
        return out

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Network initialized --------------')
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        if verbose:
            print(self)
        print('Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')
    
    def save(self, epoch, save_dir):
        """Save model weights, and current epoch."""
        save_filename = 'unet_epoch_%s.pth' % epoch
        save_path = os.path.join(save_dir, save_filename)
        device = next(self.parameters()).device
        torch.save(self.to("cpu").state_dict(), save_path)
        self.to(device)
        print(f"Model saved to: {save_path}")
