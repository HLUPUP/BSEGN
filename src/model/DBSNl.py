import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_model


@regist_model


class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    '''
    def __init__(self, in_ch=16, out_ch=1, base_ch=128, num_module=9):#hlin_ch=3, out_ch=3, base_ch=128, num_module=9
        
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"
        out_ch=1# hl 1
        in_ch=7#hl 7
        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)

        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

        #hl
        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=(1,2*6-1), stride=1, padding=(0,6-1)) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, 1,     kernel_size=1) ]
        
        
        self.tail_gaussian = nn.Sequential(*ly)


    def forward(self, x):
        
        x = self.head(x) #24 3 140 140 - 24 128 140 140
        br1 = self.branch1(x) #24 128 140 140
        br2 = self.branch2(x) #24 128 140 140
        x = torch.cat([br1, br2], dim=1)
        x1 = self.tail(x)
        return x1
        

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=(1,2*stride-1), stride=1, padding=(0,stride-1)) ]
    
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        
        return self.body(x)

class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=(1,3), stride=1, padding=(0,stride),dilation=(1,stride)) ]
        
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        
        return x + self.body(x)

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        
        _, _, kH, kW = self.weight.size()# hl get network structure weight size 
        
        self.mask.fill_(1)
        self.mask[:, :, :, kW//2] = 0
        
        

    def forward(self, x):
        
        self.weight.data *= self.mask
        
        
        return super().forward(x)



class DBSN_gaussian(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    '''
    def __init__(self,base_ch=64):#hlin_ch=3, out_ch=3, base_ch=128, num_module=9
        
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        # assert base_ch%2 == 0, "base channel should be divided with 2"
        out_ch=1# hl 1
        in_ch=7#hl 7
        ly = []
        ly += [ nn.Conv2d(1, 16, kernel_size=(1,21), stride=1, padding=(0,10),dilation=(1,1)) ]#1,3
        ly += [ nn.Conv2d(16, base_ch, kernel_size=(1,5), stride=1, padding=(0,2),dilation=(1,1)) ]#1,3
        ly += [ nn.Conv2d(base_ch, base_ch, kernel_size=(1,5), stride=1, padding=(0,2),dilation=(1,1)) ]#1,3
        self.head_hl=nn.Sequential(*ly)

        

        ly = []
        ly += [ nn.Conv2d(base_ch,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        
        self.tail = nn.Sequential(*ly)
        
        


    def forward(self, x):
        x = self.head_hl(x)      
        x = self.tail(x)
        return x
       

    def _initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DBSN_correction(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    '''
    def __init__(self, in_ch=16, out_ch=1, base_ch=128, num_module=9):#hlin_ch=3, out_ch=3, base_ch=128, num_module=9
        
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"
        out_ch=7# hl 1
        in_ch=7#hl 7
        ly = []
        ly += [ nn.Conv2d(7, 1, kernel_size=(1,5), stride=1, padding=(0,2),dilation=(1,1)) ]#1,3
        self.head_hl=nn.Sequential(*ly)

        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl_correction(2, base_ch, num_module)
        self.branch2 = DC_branchl_correction(3, base_ch, num_module)

        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        
        self.tail = nn.Sequential(*ly)
        
        


    def forward(self, x):
        x = self.head_hl(x)
        x1 = self.head(x) #24 3 140 140 - 24 128 140 140

        br1 = self.branch1(x1) #24 128 140 140
        br2 = self.branch2(x1) #24 128 140 140

        x1 = torch.cat([br1, br2], dim=1)
        x1 = self.tail(x1)
       

        return x1
       

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

class DC_branchl_correction(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=(1,2*stride-1), stride=1, padding=(0,stride-1)) ]
        # self.ceter_hl = nn.Sequential(*ly)

        
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]#这两行注释过


        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)









 