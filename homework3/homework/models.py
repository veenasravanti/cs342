import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1): 
          super().__init__()   
          self.net = torch.nn.Sequential(
                  #Adding Batch Normalization
                  #torch.nn.BatchNorm2d(n_input)
                  torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride,bias=False),
                  torch.nn.BatchNorm2d(n_output), #if using after set baias to false
                  #can use LayerNorm, InstanceNorm or groupNorm
                  #or After
                  torch.nn.ReLU(),
                  torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                  torch.nn.ReLU()
                )
          #torch.nn.init.kaiming_normal_(self.net[0].weight)
          #torch.nn.init.constant_(self.net[0].bias,0.01)
          #Wont have to use initialization in pytorch since it has inbuilt initialization

        def forward(self, x):
              return(self.net(x))

    def __init__(self, layers=[32,64,128], n_input_channels=3):
            super().__init__()
            L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2,bias=False),
                 torch.nn.BatchNorm2d(32),
                 torch.nn.ReLU(),
                 torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            c = 32
            for l in layers:
                L.append(self.Block(c, l, stride=2))
                c = l
            self.network = torch.nn.Sequential(*L)
            self.classifier = torch.nn.Linear(c, 6)
            #torch.nn.init.zeros(self.classifier.weight)

    def forward(self, x):
        # @x: torch.Tensor((B,3,64,64))
        #@return: torch.Tensor((B,6))
        #Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        z = self.network(x)

        # Global average pooling\n",
        z = z.mean(dim=[2,3])
        # Classify\n",
        return self.classifier(z)


class FCN(torch.nn.Module):
      class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1): 
          super().__init__()   
          self.net = torch.nn.Sequential(
                  #Adding Batch Normalization
                  #torch.nn.BatchNorm2d(n_input)
                  torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride,bias=False),
                  torch.nn.BatchNorm2d(n_output), #if using after set baias to false
                  #can use LayerNorm, InstanceNorm or groupNorm
                  #or After
                  torch.nn.ReLU(),
                  torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                  torch.nn.ReLU()
                )
          #torch.nn.init.kaiming_normal_(self.net[0].weight)
          #torch.nn.init.constant_(self.net[0].bias,0.01)
          #Wont have to use initialization in pytorch since it has inbuilt initialization

        def forward(self, x):
              return(self.net(x))

  class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


  class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Block(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = Block(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

  def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
   



model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
