from utils import *
from PCDARTS import *


class NASSurrogate():
    def __init__(self, init_channels=16, layers=8, n_channels=3, gpu_id=0, num_class=10, softmax=False):
        self.criterion = torch.nn.MSELoss()
        self.device = torch.device(('cuda:%d' % gpu_id) if torch.cuda.is_available() else 'cpu')
        self.surrogate = NASNetwork(C=init_channels, num_classes=num_class, layers=layers, n_channels=n_channels, criterion=self.criterion, softmax=softmax, device=self.device)
        self.surrogate = self.surrogate.to(self.device)
        self.architect = Architect(self.surrogate, momentum=0.9, weight_decay=3e-4, arch_learning_rate=6e-4, arch_weight_decay=1e-3)
        
        self.optimizer = torch.optim.SGD(self.surrogate.parameters(), lr=0.1, momentum=0.9, weight_decay=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 5.0, eta_min=0.0)
        self.num_class = num_class
        self.iter_train = -1

    def __call__(self, img, no_grad=True):
        # img : B * C * H * W  0~1 torch.Tensor
        # return: B * P  torch.Tensor
        if no_grad: 
            self.surrogate.eval()
            with torch.no_grad(): return self.surrogate(img.to(self.device))
        else:
            return self.surrogate(img.to(self.device))

    def train(self, attacker_id, sampler, batch_size, iter_train, log_file_path):
        log_file = open(log_file_path, 'a')

        img_ori, lbl_ori = sampler.generate_training_batch(batch_size)
        img_ori = torch.Tensor(img_ori).to(self.device)
        img_ori.requires_grad = True
        lbl_ori = torch.Tensor(lbl_ori).to(self.device)

        self.lr = self.scheduler.get_lr()[0]
        self.surrogate.train()
        self.optimizer.zero_grad()

        # NAS
        img_ori_search, lbl_ori_search = sampler.generate_training_batch(batch_size)
        img_ori_search = torch.tensor(img_ori_search, dtype=torch.float32, requires_grad=False).to(self.device)
        lbl_ori_search = torch.tensor(lbl_ori_search, dtype=torch.float32, requires_grad=False).to(self.device)
        self.architect.step(img_ori, lbl_ori, img_ori_search, lbl_ori_search, self.lr, self.optimizer, unrolled=False)
        # if epoch > 15 in official implementation
        
        # normal
        lbl = self.__call__(img_ori, no_grad=False)
        loss = self.criterion(lbl, lbl_ori)
        acc = (lbl.argmax(axis=1).int() == lbl_ori.argmax(axis=1).int()).float().mean().detach().cpu().numpy()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.surrogate.parameters(), 5)###
        self.optimizer.step()

        if iter_train != self.iter_train:
            self.scheduler.step()
            self.iter_train = iter_train
        
        output({'Surrogate': attacker_id, 'Batch': iter_train, 'Loss': '%.5f' % loss.detach(), 'Acc': round(acc*100, 2)}, end='\r', stream=log_file)
        log_file.close()
        return loss.detach()

    def save(self, save_name):
        self.surrogate.eval()
        torch.save(self.surrogate.state_dict(), save_name)
    
    def load(self, model_path):
        print('Load surrogate from', model_path)
        self.surrogate.load_state_dict(state_dict=torch.load(model_path, map_location={'cuda:4': 'cuda', 'cuda:1': 'cuda'}))


class Surrogate(): # fixed architecture, using class Net for wrapping
    def __init__(self, surrogate_name, gpu_id=0, num_class=10, softmax=False):
        self.surrogate = Net(net_name=surrogate_name, num_class=num_class, load_pretrained=False, pretrained=(num_class==1000), softmax=softmax) # load pretrained surrogates for ImageNet
        self.device = torch.device(('cuda:%d' % gpu_id) if torch.cuda.is_available() else 'cpu')
        self.surrogate = self.surrogate.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=0.00005)
        self.num_class = num_class

    def __call__(self, img, no_grad=True):
        # img : B * C * H * W  0~1 torch.Tensor
        # return: B * P  torch.Tensor
        if no_grad: 
            self.surrogate.eval()
            with torch.no_grad(): return self.surrogate(img.to(self.device))
        else:
            return self.surrogate(img.to(self.device))

    def train(self, attacker_id, sampler, batch_size, iter_train, **kwargs):
        if self.num_class == 1000: return 0 # do not finetune for ImageNet
        img_ori, lbl_ori = sampler.generate_training_batch(batch_size)

        img_ori = torch.Tensor(img_ori).to(self.device)
        img_ori.requires_grad = True
        lbl_ori = torch.Tensor(lbl_ori).to(self.device)

        self.surrogate.train()
        self.optimizer.zero_grad()
        
        lbl = self.__call__(img_ori, no_grad=False)
        loss = self.criterion(lbl, lbl_ori)
        acc = (lbl.argmax(axis=1).int() == lbl_ori.argmax(axis=1).int()).float().mean().detach().cpu().numpy()
        loss.backward()
        self.optimizer.step()
        
        output({'Surrogate': attacker_id, 'Batch': iter_train, 'Loss': '%.5f' % loss.detach(), 'Acc': round(acc*100, 2)}, end='\r')#, stream=log_file
        return loss.detach()

    def save(self, save_name):
        self.surrogate.eval()
        torch.save(self.surrogate.state_dict(), save_name)
    
    def load(self, model_path):
        print('Load surrogate from', model_path)
        self.surrogate.load_state_dict(state_dict=torch.load(model_path))


class Net(nn.Module):
    def __init__(self, net_name, load_pretrained=True, pretrained=False, num_class=10, softmax=False):
        pretrained_model_path = None
        if isinstance(load_pretrained, str):
            pretrained_model_path = load_pretrained
            load_pretrained = False
        super(Net, self).__init__()
        self.net_name = net_name
        if   net_name == 'ResNet50':      self.model = torchvision.models.resnet50(pretrained=pretrained)      # success load
        elif net_name == 'ResNet101':     self.model =  torchvision.models.resnet101(pretrained=pretrained)     # success load
        elif net_name == 'ResNet152':     self.model =  torchvision.models.resnet152(pretrained=pretrained)     # success load
        elif net_name == 'VGG16':         self.model =  torchvision.models.vgg16(pretrained=pretrained)         # success load
        elif net_name == 'VGG19':         self.model =  torchvision.models.vgg19(pretrained=pretrained)         # success load
        elif net_name == 'DenseNet121':   self.model =  torchvision.models.densenet121(pretrained=pretrained)   # success load
        elif net_name == 'DenseNet169':   self.model =  torchvision.models.densenet169(pretrained=pretrained)   # success load
        elif net_name == 'DenseNet201':   self.model =  torchvision.models.densenet201(pretrained=pretrained)   # success load
        elif net_name == 'InceptionV3':   self.model =  torchvision.models.inception_v3(pretrained=pretrained)  # success load
        else: raise ValueError('Invalid Network Name!')
        self.num_class = num_class
        self.softmax = softmax
        if self.num_class == 1000: return # load imagenet model would cause error if execuating below

        if pretrained_model_path is not None:
            state_dict = torch.load(pretrained_model_path)
            self.model.load_state_dict(state_dict=state_dict)
        elif load_pretrained:
            state_dict = torch.load(model_url[net_name])
            if 'DenseNet' in net_name:
                state_dict = transfer_state_dict(state_dict)
            self.model.load_state_dict(state_dict=state_dict)
        
        self.ELU = nn.ELU()
        self.fc = nn.Linear(1000, num_class)
        

    def forward(self, x):
        x = self.model(x)
        if self.num_class != 1000:
            x = self.ELU(x)
            x = self.fc(x)
        if self.softmax: x = nn.functional.relu(x)
        return x