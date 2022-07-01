import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, input_filters, num_filters, down_sampling=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_filters, num_filters, 3, stride=(1 if down_sampling is False else 2), padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.conv3 = nn.Conv2d(input_filters, num_filters, 1, stride=(1 if down_sampling is False else 2))
        self.input_filters = input_filters
        self.num_filters = num_filters
        self.down_sampling = down_sampling

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if (self.num_filters != self.input_filters or (self.down_sampling is True)):
            identity = self.conv3(x)

        out = out + identity

        return self.relu(out)


class HandSegNet(nn.Module):
    '''
    Perform segmentation for a hand RoI. 
    Input: size (Bs, 3, 256, 256). RoI (the hand rigion is much larger than the background. )
    Output: size (Bs, 2, 64, 64). Instance level segmentation. 
    1st chanel for left hand; 2nd channel for right hand. 
    '''
    def __init__(self):
        super(HandSegNet, self).__init__()
        out_channels = 1 # left, right, back-ground
        self.conv0_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv0_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=2, dilation=2)
        self.lrelu = nn.LeakyReLU(inplace=True) # Rectifier Nonlinearities Improve Neural Network Acoustic Models, ICML2013
        self.block_s64 = nn.Sequential(BasicBlock(64, 128, True),BasicBlock(128, 128))
        self.block_s32 = nn.Sequential(BasicBlock(128, 256, True),BasicBlock(256, 256))
        self.block_s16 = nn.Sequential(BasicBlock(256, 256, True),BasicBlock(256, 256))
        self.block_s8 = nn.Sequential(BasicBlock(256, 256, True),BasicBlock(256, 256))
        self.block_s4 = nn.Sequential(BasicBlock(256, 256, True),BasicBlock(256, 256))

        self.block_decoder_s1 = nn.Sequential(BasicBlock(512, 256),BasicBlock(256, 128))
        self.block_decoder_s2 = nn.Sequential(BasicBlock(384, 256),BasicBlock(256, 128))
        self.block_decoder_s3 = nn.Sequential(BasicBlock(256 + out_channels, 256),BasicBlock(256, 128))
        self.block_decoder_s4 = nn.Sequential(BasicBlock(128 + out_channels, 128),BasicBlock(128, 128))
        self.conv = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.softmax_along_channels = nn.Softmax(dim = 1)

    def forward(self, x):
        header = torch.cat((self.conv0_0(x),self.conv0_1(x)), 1)
        header = self.lrelu(header)
        # encoder
        block1 = self.block_s64(header)
        block2 = self.block_s32(block1)
        block3 = self.block_s16(block2)
        block4 = self.block_s8(block3)
        block5 = self.block_s4(block4)
        
        #decoder
        decoder_1 = torch.cat((self.upsample(block5), block4), 1)
        decoder_s8 = self.block_decoder_s1(decoder_1)

        decoder_2 = torch.cat((self.upsample(decoder_s8), block3), 1)
        heat_map0 = self.conv(self.block_decoder_s2(decoder_2))
        heat_map0 = self.lrelu(heat_map0)

        decoder_3 = torch.cat((self.upsample(heat_map0), block2), 1)
        heat_map1 = self.conv(self.block_decoder_s3(decoder_3))
        heat_map1 = self.lrelu(heat_map1)

        decoder_4 = torch.cat((self.upsample(heat_map1), block1), 1)
        heat_map2 = self.conv(self.block_decoder_s4(decoder_4))
        heat_map2 = self.lrelu(heat_map2)

        return (heat_map0, heat_map1, heat_map2)


class HandSegNet_ScaleUp(HandSegNet):
    def __init__(self):
        super(HandSegNet_ScaleUp, self).__init__()
        out_channels = 1
        self.block_decoder_s5 = BasicBlock(64 + out_channels, 16)
        self.final_conv = nn.Sequential(BasicBlock(16, out_channels), nn.LeakyReLU())
    def forward(self, x):
        header = torch.cat((self.conv0_0(x),self.conv0_1(x)), 1)
        header = self.lrelu(header)
        # encoder
        block1 = self.block_s64(header)
        block2 = self.block_s32(block1)
        block3 = self.block_s16(block2)
        block4 = self.block_s8(block3)
        block5 = self.block_s4(block4)
        
        #decoder
        decoder_1 = torch.cat((self.upsample(block5), block4), 1)
        decoder_s8 = self.block_decoder_s1(decoder_1)

        decoder_2 = torch.cat((self.upsample(decoder_s8), block3), 1)
        heat_map0 = self.conv(self.block_decoder_s2(decoder_2))
        heat_map0 = self.lrelu(heat_map0)

        decoder_3 = torch.cat((self.upsample(heat_map0), block2), 1)
        heat_map1 = self.conv(self.block_decoder_s3(decoder_3))
        heat_map1 = self.lrelu(heat_map1)

        decoder_4 = torch.cat((self.upsample(heat_map1), block1), 1)
        heat_map2 = self.conv(self.block_decoder_s4(decoder_4))
        heat_map2 = self.lrelu(heat_map2)

        decoder_5 = torch.cat((self.upsample(heat_map2), header), 1)
        heat_map3_wide = self.block_decoder_s5(decoder_5)
        heat_map4 = self.final_conv(self.upsample(heat_map3_wide))

        return (heat_map1, heat_map2, heat_map4)


class HandSegNet_ScaleUp__Deploy(HandSegNet_ScaleUp):
    def forward(self, x):
        header = torch.cat((self.conv0_0(x),self.conv0_1(x)), 1)
        header = self.lrelu(header)
        # encoder
        block1 = self.block_s64(header)
        block2 = self.block_s32(block1)
        block3 = self.block_s16(block2)
        block4 = self.block_s8(block3)
        block5 = self.block_s4(block4)
        
        #decoder
        decoder_1 = torch.cat((self.upsample(block5), block4), 1)
        decoder_s8 = self.block_decoder_s1(decoder_1)

        decoder_2 = torch.cat((self.upsample(decoder_s8), block3), 1)
        heat_map0 = self.conv(self.block_decoder_s2(decoder_2))
        heat_map0 = self.lrelu(heat_map0)

        decoder_3 = torch.cat((self.upsample(heat_map0), block2), 1)
        heat_map1 = self.conv(self.block_decoder_s3(decoder_3))
        heat_map1 = self.lrelu(heat_map1)

        decoder_4 = torch.cat((self.upsample(heat_map1), block1), 1)
        heat_map2 = self.conv(self.block_decoder_s4(decoder_4))
        heat_map2 = self.lrelu(heat_map2)

        decoder_5 = torch.cat((self.upsample(heat_map2), header), 1)
        heat_map3_wide = self.block_decoder_s5(decoder_5)
        heat_map4 = self.final_conv(self.upsample(heat_map3_wide))

        return heat_map4


class HandLRRegressNet(nn.Module):
    def __init__(self):
        super(HandLRRegressNet, self).__init__()
        # input: (3, 64, 64) image * mask
        # output: classification left(1, 0) /right(0, 1)
        import torchvision.models as models
        self.backbone = models.resnet18(num_classes = 2)
    
    def forward(self, x):
        x = self.backbone(x)
        return x

        

################ Unit Test ################
def io_test():
    input_Tsor = torch.randn(7, 3, 256, 256)
    model = HandSegNet()
    output_Tsors = model(input_Tsor)
    print(output_Tsors[-1].shape) # (1, 2, 64, 64)
    print(output_Tsors[-2].shape)
    print(output_Tsors[-3].shape)

def io_test_scale_up():
    import torch.nn.functional as F
    input_Tsor = torch.randn(7, 3, 256, 256)
    model = HandSegNet_ScaleUp()
    # model = HandSegNet()
    output_Tsors = model(input_Tsor)

    gt_0 = torch.rand_like(output_Tsors[0])
    gt_1 = torch.rand_like(output_Tsors[1])
    gt_2 = torch.rand_like(output_Tsors[2])
    loss = F.mse_loss(output_Tsors[0], gt_0) + F.mse_loss(output_Tsors[1], gt_1) + F.mse_loss(output_Tsors[2], gt_2)
    loss.backward()
    print(output_Tsors[-1].shape) # (1, 2, 64, 64)
    print(output_Tsors[-2].shape)
    print(output_Tsors[-3].shape)
    # F.mse_loss(output_Tsors[2], gt_2).backward()

def io_test_regress():
    input_Tsor = torch.randn(7, 3, 64, 64)
    model = HandLRRegressNet()
    output_Tsors = model(input_Tsor)
   

if __name__ == "__main__":
    # io_test()
    # io_test_scale_up()
    io_test_regress()
