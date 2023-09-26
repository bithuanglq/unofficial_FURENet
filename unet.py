import torch 
import torch.nn as nn




class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_ch=10) -> None:
        super().__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

    def forward(self, x):
        '''
            x: [bsz, 10, 256, 256]
        '''
        e1 = self.Conv1(x)      # [1,64,256,256]

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)     # [1,128,128,128]

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)     # [1,256,64,64]

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)     # [1,512,32,32]
        return e1, e2, e3, e4



class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class csSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse


class UNet(nn.Module):
    def __init__(self, img_ch=10, output_ch=10):
        super(UNet, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]  # [64, 128, 256, 512]

        self.Encoder_Zh = Encoder(img_ch)
        self.Encoder_Zdr = Encoder(img_ch)
        self.Encoder_Kdp = Encoder(img_ch)

        self.SE_block = csSE(512*3)

        self.Up4 = up_conv(filters[3]*3, filters[2])
        self.Up_conv4 = conv_block(filters[3]*2, filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2]*2, filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1]*2, filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()


    def forward(self, Zh, Zdr, Kdp):
        '''
            Zh, Zdr, Kdp: [batchsize, 10, 256, 256]
        '''

        encoder_Zh = self.Encoder_Zh(Zh)
        encoder_Zdr = self.Encoder_Zdr(Zdr)
        encoder_kdp = self.Encoder_Kdp(Kdp)

        e4 = torch.concat((encoder_Zdr[3], encoder_kdp[3], encoder_Zh[3]), dim=1)    # [bsz, 512*3, 32, 32]
        e4 = self.SE_block(e4)

        d4 = self.Up4(e4)    # [bsz, 256, 64, 64]   
        d4 = torch.cat((encoder_Zdr[2], encoder_kdp[2], encoder_Zh[2], d4), dim=1)  # [bsz, 256*4, 64, 64]
        d4 = self.Up_conv4(d4)      # [bsz,256,64,64]

        d3 = self.Up3(d4)    # [bsz, 128, 128, 128]
        d3 = torch.cat((encoder_Zdr[1], encoder_kdp[1], encoder_Zh[1], d3), dim=1)  # [bsz, 128*4, 128,128]
        d3 = self.Up_conv3(d3)      # [bsz,128,128,128]

        d2 = self.Up2(d3)    # [bsz, 64, 256, 256]
        d2 = torch.cat((encoder_Zdr[0], encoder_kdp[0], encoder_Zh[0], d2), dim=1)  # [bsz, 64*4, 256, 256]
        d2 = self.Up_conv2(d2)      # [bsz,64,256,256]

        out = self.Conv(d2)         # [bsz,10,256,256]

        out = self.active(out)      # [0,1]

        return out





if __name__=='__main__':
    if False:
        net = UNet(img_ch=10, output_ch=10)
        input_tensor1 = torch.rand((1, 10, 256, 256))
        input_tensor2 = torch.rand((1, 10, 256, 256))
        input_tensor3 = torch.rand((1, 10, 256, 256))
        output_tensor = net(input_tensor1, input_tensor2, input_tensor3)
        print(output_tensor.shape)      # torch.Size([1, 10, 256, 256])
