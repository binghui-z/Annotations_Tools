import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    '''
    a basic res block: 
    x -> 3x3 -> 3x3-->
      ---------------
    '''
    def __init__(self, input_filters, num_filters, down_sampling=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = input_filters, out_channels = num_filters, 
                    kernel_size = 3, stride=(1 if down_sampling is False else 2), padding=1)
        
        self.conv2 = nn.Conv2d(in_channels = num_filters, out_channels = num_filters, 
                    kernel_size = 3, padding=1)

        self.conv3 = nn.Conv2d(input_filters, num_filters, 1, stride=(1 if down_sampling is False else 2))

        self.input_filters = input_filters
        self.num_filters = num_filters
        self.down_sampling = down_sampling

        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)

        if self.num_filters != self.input_filters or self.down_sampling:
            # in-out channel mismatch / the detailed info has been missed
            identity = self.conv3(x)
        
        return self.activation(out + identity)
    

class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        
        # shared back-bone
        self.conv0_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1) # (32, 128, 128) -
        self.conv0_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=2, dilation=2) # (32, 128, 128) --> cat --> (64, 128, 128)
        self.lrelu = nn.LeakyReLU(inplace=True)

        self.block_s64 = nn.Sequential(BasicBlock(64, 128, True),BasicBlock(128, 128)) # (128, 64, 64) -> (128, 64, 64) ->
        self.block_s32 = nn.Sequential(BasicBlock(128, 256, True),BasicBlock(256, 256)) # (256, 32, 32) -> (256, 32, 32) ->
        self.block_s16 = nn.Sequential(BasicBlock(256, 256, True),BasicBlock(256, 256)) # (256, 16, 16) -> (256, 16, 16) ->
        self.block_s8 = nn.Sequential(BasicBlock(256, 256, True),BasicBlock(256, 256)) # (256, 8, 8) -> (256, 8, 8) ->
        self.block_s4 = nn.Sequential(BasicBlock(256, 256, True),BasicBlock(256, 256)) # (256, 4, 4) -> (256, 4, 4) ->

        self.block_decoder_s1 = nn.Sequential(BasicBlock(512, 256),BasicBlock(256, 128)) # up[(256, 4, 4)] (+) (256, 8, 8) -> (512, 8, 8); (512, 8, 8)---> (128, 8, 8)
        self.block_decoder_s2 = nn.Sequential(BasicBlock(384, 256),BasicBlock(256, 128)) # up[(128, 8, 8)] (+) (256, 16, 16) -> (384, 16, 16); (384, 16, 16)--->(128, 16, 16)->(36, 16, 16)
        self.block_decoder_s3 = nn.Sequential(BasicBlock(292, 256),BasicBlock(256, 128)) # up[(36, 16, 16)] (+) (256, 32, 32) -> (292, 32, 32); (292, 32, 32)--->(128, 32, 32)->(36, 32, 32)
        self.block_decoder_s4 = nn.Sequential(BasicBlock(164, 128),BasicBlock(128, 128)) # up[(36, 32, 32)] (+) (128, 64, 64) -> (164, 64, 64); (164, 64, 64)--->(128, 64, 64)->(36, 64, 64)

        self.conv = nn.Conv2d(in_channels=128, out_channels=36, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

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
        decoder_s8 = self.block_decoder_s1(decoder_1) # (128, 8, 8)

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


if __name__ == "__main__":
    
    gm_net = SRNet()
    gm_x = torch.rand(5, 3, 64, 64)
    gm_y0, gm_y1, gm_y2 = gm_net(gm_x)

    print(gm_y0.shape)
    print(gm_y1.shape)
    print(gm_y2.shape)