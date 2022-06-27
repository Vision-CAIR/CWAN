import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms
from model import Discriminator

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        self.std = self.std.to(tensor.device)
        self.mean = self.mean.to(tensor.device)
        tensor.mul_(self.std[:, None, None]).add_(self.mean[:, None, None])
        return tensor

wikiart_denormalize = DeNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])

def get_nearest_mean_examples(dset, key,  module, examples_per_class=10, batch_size=64, verbose=False, diverse=False): 
    """
    1. Get samples nearest to the the actual mean of the class 
    2. Get samples nearest to the the actual mean of the class but are diverse.
    """   
    proto_files = {}

    ## Changing resolution to 256 for proper calculation of protos
    current_res = dset.resolution
    dset.resolution = 256


    labels = getattr(dset, key)
    classes, counts = np.unique(labels, return_counts=True)
    for itr, (clas, count) in enumerate(zip(classes, counts)):
        low = 0
        labels = torch.tensor(labels)
        indexes,  = (labels[:, 0] == clas).nonzero(as_tuple=True)
        output_tensor = []
        high = count
        for i in range(low, count, batch_size):
            range_value = indexes[i:i+batch_size if i+batch_size < high else high]
            input_tensor = [imagenet_normalize(wikiart_denormalize(dset[k]['img']))[None] for k in range_value]
            with torch.no_grad():
                output_tensor.append(module(torch.cat(input_tensor, dim=0)))   

        class_features = torch.cat(output_tensor, dim=0)
        mean_class_features = class_features.mean(dim=0)
        dist = torch.norm(class_features - mean_class_features[None], dim=1, p=None)
        
        if diverse:
            samples = get_diverse_samples(class_features, mean_class_features, count, examples_per_class, verbose=verbose)
            proto_files[itr] = indexes[samples].tolist()
        else:
            knn = dist.topk(examples_per_class, largest=False)
            if verbose:
                print("norm: ", torch.norm(class_features[knn.indices].mean(dim=0) - mean_class_features, dim=0, p=None))
            proto_files[itr] = indexes[knn.indices].tolist()                
            
        if verbose:
            print('\n', itr, proto_files[itr])
            print('\n')
    
    ## setting the current resolution back.
    dset.resolution = current_res   
    return proto_files

def get_diverse_samples(class_features, mean_class_features, count, examples_per_class, top_k=100, verbose=False):
    total_features, feature_size = class_features.shape
    K = 5000
    indices = torch.cat([torch.tensor(np.random.choice(np.arange(total_features), size=examples_per_class, replace=False)) for i in range(K)])
    Y = class_features[indices].view(K, examples_per_class, feature_size).contiguous()
    indices = indices.view(K, examples_per_class)
    mean_Y = Y.mean(dim=1)
    dist = torch.norm(mean_Y - mean_class_features[None], dim=1, p=None)
    topk = dist.topk(top_k, largest=False)
    Ys = Y[topk.indices]
    indices = indices[topk.indices]
    if verbose:
        print(indices)
        print(dist[topk.indices])
    return indices[torch.randint(0, len(indices), size=(1,))][0]

def get_random_examples(dset, key, examples_per_class=10):
    low = 0
    proto_files = {}
    labels = getattr(dset, key)
    classes, counts = np.unique(labels, return_counts=True) 
    for itr, (clas, count) in enumerate(zip(classes, counts)):
        labels = torch.tensor(labels)
        indexes,  = (labels[:, 0] == clas).nonzero(as_tuple=True)
        proto_files[itr] = indexes[torch.randint(0, int(count), (examples_per_class,))].tolist()
    return proto_files

def compute_protos(proto_files, key, module, dset, grad_proto, module_kwargs, verbose=False):
    protos = []
    for clas, idxs in proto_files.items():
        input_tensors = []
        for idx in idxs:
            item = dset[idx]
            if verbose:
                print(item['img'].shape, item['img'].min(), item['img'].max(), item['img'].mean((1,2)))
            input_tensors.append(item['img'][None])
        input_batch = torch.cat(input_tensors, dim=0).cuda()
        with torch.no_grad():
            if key == 'pruned_style_class':
                ### 10 images on single GPU.
                if type(module) == Discriminator:
                    _, features = module(input_batch, **module_kwargs)
                else:
                    _, features = module.module(input_batch, **module_kwargs)
        protos.append(features.mean(0)[None])
    return torch.cat(protos, dim=0)

def imitative_loss(input, binary=False, unique_cv=None):
    current_device = input.device
    eye = torch.eye(input.shape[0], input.shape[0]).to(current_device)
    if unique_cv is not None:
        if binary:
            return F.binary_cross_entropy(input[unique_cv].view(-1, 1), eye[unique_cv].view(-1, 1))
        else:
            return F.kl_div(input[unique_cv].log(), eye[unique_cv])        
    else:
        if binary:
            return F.binary_cross_entropy(input.view(-1, 1), eye.view(-1, 1))
        else:
            return F.kl_div(input.log(), eye)

def creative_loss(input, binary=False):
    current_device = input.device    
    target = torch.ones(input.shape[0], input.shape[0]).to(current_device) / input.shape[0]
    if binary:
        return F.binary_cross_entropy(input.view(-1, 1), target.view(-1, 1))
    else:
        return F.kl_div(input.log(), target)

def visiting_loss(input):
    current_device = input.device    
    target = torch.ones_like(input).to(current_device)/ input.shape[0]
    return F.kl_div(input.log().view(1,-1), target.view(1,-1))

class RWLoss(nn.Module):
    def __init__(self,
                 tau=3,
                 alpha=0.7,
                 binary=False,
                 graph_smoothing=0.1,
                 proto_method='random_once',
                 running_mean_factor=None,
                 feature_extractor=None,
                 opt=None
                ):
        """
        n_classes: total number of classes will be required for mean proto.
        tau: number of hops on unlabelled points
        alpha: decay factor for loss
        binary: If true, it will comput binary cross entropy loss.
        graph_smoothing: As in TF implementation.
        proto_method:'random_once', 'random_all', 'nearest_mean'
        """
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.binary = binary
        self.c = graph_smoothing
        self.proto_method = proto_method
        self.running_mean_factor = running_mean_factor
        self.opt = opt
        self.feature_extractor = feature_extractor
        self.proto_examples = {}
        self.protos = {}
        if opt.normalize_protos_scale is not None:
            print(f"Prototypes and features will be normalized with value {opt.normalize_protos_scale}")
            self.normalize_protos = True
            self.normalize_scale = opt.normalize_protos_scale
        else:
            print("Prototypes and features will not be normalized")
            self.normalize_protos = False

    def forward(self, 
                features, 
                labels, 
                discriminator, 
                mode, 
                dataset,
                key,
                **kwargs):
        """
        features: BXF
        labels: N (tensor of integers)
        discriminator: nn.Module to compute protos
        mode: string "creative" or "imitative"
        dataset: dataset object
        """
        verbose = kwargs.get('verbose', False)
        self.mode = mode
        N, F = features.shape
        if labels is not None:
            unique_cv = labels.unique()
        else:
            unique_cv = None
        
        module_kwargs = kwargs.get('module_kwargs', {})

        if self.proto_method == 'random_once':
            if hasattr(self, 'proto_examples'):
                if key not in self.proto_examples.keys():
                    self.proto_examples[key] = get_random_examples(dataset, key, 10)
        elif self.proto_method == 'random_all':
            self.proto_examples = get_random_examples(dataset, 10)
        elif self.proto_method == 'nearest_mean_once':
            if hasattr(self, 'proto_examples'): # compute best options using self.feature_extractor
                if key not in self.proto_examples.keys():
                    self.proto_examples[key] = get_nearest_mean_examples(dataset, key, self.feature_extractor, 10, verbose=verbose)  
        elif self.proto_method == 'nearest_mean_once_diverse':
            if hasattr(self, 'proto_examples'): # compute best options using self.feature_extractor
                if key not in self.proto_examples.keys():
                    self.proto_examples[key] = get_nearest_mean_examples(dataset, key, self.feature_extractor, 10, verbose=verbose, diverse=True)                
        
        if verbose:
            print('computing protos')

        protos = compute_protos(self.proto_examples[key], key, discriminator, dataset, self.opt.RW_grad_proto, module_kwargs)
        self.protos[key] = protos

        if verbose:
            print('protos shape ', protos.shape)
            

        current_device = self.protos[key].device
        if self.normalize_protos:
            features = (features * self.normalize_scale) / torch.norm(features, dim=-1, keepdim=True).detach()
            self.protos[key] = (self.protos[key] * self.normalize_scale) / torch.norm(self.protos[key], dim=-1, keepdim=True).detach()

        A = torch.norm(self.protos[key].unsqueeze(1) - features, dim=-1).pow(2).mul(-1)
        A_t = (1 - self.c) * A.t().softmax(-1) + (self.c * torch.ones_like(A.t())/len(self.protos[key])) 
        A = A.softmax(-1)
        A = (1 - self.c) * A + (self.c * torch.ones_like(A) / len(self.protos[key]))
        A_t = (1 - self.c) * A_t + (self.c * torch.ones_like(A_t) / N)
        B = ((torch.norm(features.unsqueeze(1) - features, dim=-1).pow(2).mul(-1)) - torch.eye(N,N).to(current_device) * 1000).softmax(-1)  ## Adding -1000 at diagonals  to avoid self loops
        
        ## Random Walk.
        landing_probs = []
        T0 = A@A_t
        landing_probs.append(T0)
        T = A
        for i in range(self.tau - 1):
            T = T@B
            landing_probs.append(T@A_t)
            
        loss = 0.            
        for i, p in enumerate(landing_probs):
            if self.mode == 'creative':
                loss += (self.alpha ** i) * creative_loss(p, self.binary)
            elif self.mode == 'imitative':
                loss += (self.alpha ** i) * imitative_loss(p, self.binary, unique_cv)
            else:
                raise Exception("Wrong Mode")
                
        if verbose:
            print(loss, visiting_loss(A))        
        loss += visiting_loss(A)

        return loss
