import torch
import torch.nn.functional as F
import numpy


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)
class Block(torch.nn.Module):
      def __init__(self, n_input, n_output, stride=1): 
            super().__init__()   

            self.net = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride,bias=False),
                    torch.nn.BatchNorm2d(n_output), #if using after set baias to false
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1,bias=False),
                    torch.nn.ReLU()
                  )
           
            #print("Exit Block n_input=",n_input)
            #print("Exit Block n_output=",n_output)
      def forward(self, x):
          return(self.net(x))



          
class Down(torch.nn.Module):
    #Downscaling with maxpool then double conv

      def __init__(self, in_channels, out_channels):
          super().__init__()
          
          self.maxpool_conv = torch.nn.Sequential(
              torch.nn.MaxPool2d(2,padding=1),
              Block(in_channels, out_channels)
          )

      def forward(self, x):
          #print("down forward")
          return self.maxpool_conv(x)

class OutConv(torch.nn.Module):
      def __init__(self, in_channels, out_channels):
          super(OutConv,self).__init__()
          self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
          

      def forward(self, x): 
          return self.conv(x)

class Up(torch.nn.Module):
      #Upscaling then double conv

      def __init__(self, in_channels, out_channels):
          super().__init__()
          self.up = torch.nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
          self.conv = Block(in_channels, out_channels)


      def forward(self, x1, x2):
          x1 = self.up(x1)
          diffY = x2.size()[2] - x1.size()[2]
          diffX = x2.size()[3] - x1.size()[3]

          x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
          
          x = torch.cat([x2, x1], dim=1)
          return self.conv(x)



class Planner(torch.nn.Module):
     def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])
              
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inc = Block(in_channels,64 ) #first output
          
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)           
        self.up1 = Up(256, 128)
        self.up2 = Up(128,64)
        self.outc = OutConv(64, out_channels)

       
     def forward(self, x):
          z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
          x1 = self.inc(z)
          x2 = self.down1(x1)
          x3 = self.down2(x2)
          x = self.up1(x3, x2)
          x = self.up2(x, x1)
          output=torch.mean(self.outc(x), 1)
          output=spatial_argmax(output)
          return output

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
