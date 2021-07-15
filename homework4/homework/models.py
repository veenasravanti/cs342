import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    mp=torch.nn.MaxPool2d(kernel_size=max_pool_ks,padding=max_pool_ks//2,stride=1)
    X_max = mp(heatmap[None, None])[0,0]
    #possible_det=heatmap - (X_max>heatmap).float()*100 # to match score and heatmap # score does not match heatmap
    if max_det > X_max.numel():
      max_det =X_max.numel()
    heatmap_upd = (heatmap == X_max) * heatmap #  Replace all other values in heatmap to 0 except for max values 
    output_tensor = torch.flatten(heatmap_upd) #selected index k out of range
    score, location = torch.topk(output_tensor, max_det)  # should get the score and indices #use that to find cx and cy 
    #print("score",score)
    lst = []
    for sc, loc in list(zip(score, location)):
     #print(sc)
     if sc > min_score:
        cx = loc % heatmap.size(1)
        cy = loc // heatmap.size(1)
        lst.append([float(sc),cx,cy])
    return lst



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


class Detector(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
              
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inc = Block(in_channels,64 ) #first output
          
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256) 
          
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.outc = OutConv(64, out_channels)

       
    def forward(self, x):
      """
         Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
      """
      x1 = self.inc(x)
      x2 = self.down1(x1)
      x3 = self.down2(x2)
      x = self.up1(x3, x2)
      x = self.up2(x, x1)
      #print(x1)
      #print(" main forward")
      logits = self.outc(x)
      return logits    


    def detect(self, image):
      """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
      """
      heatmap = self.forward(image[None])
      zero_arr=[0,0]

      kart = list(extract_peak(heatmap[0,0],max_det = 30))
      for i in kart:
         i.append(float(0))
         i.append(float(0))
      kart = tuple(kart)
        # print(kart[0])
      
      bomb = list(extract_peak(heatmap[0,1], max_det = 30))
      for i in bomb:
         i.append(float(0))
         i.append(float(0))
      bomb = tuple(bomb)
        # print(len(bomb))

      pickup = list(extract_peak(heatmap[0,2], max_det = 30))
      for i in pickup:
         i.append(float(0))
         i.append(float(0))
      pickup = tuple(pickup)
        # print(len(pickup))
        # output = [kart, bomb, pickup] # convert into float, int, int and then add two zeros in the final two positions
      output = [kart, bomb, pickup]
      return output

       


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
