import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from random_walk_loss import RWLoss

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator
import os
from datetime import datetime
# from networksRW_pruned import RWLoss
from torchvision import models
import random
# from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    dataloader = loader
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        if not args.no_pbar:
            pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)


    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    # Initialize RW loss here.
    if args.use_RWLoss:
        rw_loss_func = RWLoss(tau=args.RW_tau,
                         alpha=args.RW_alpha,
                         binary=args.RW_use_BCE,
                         proto_method=args.RW_proto_method,
                         running_mean_factor=args.RW_proto_rm,
                         feature_extractor=getattr(models, args.RW_feature_extractor)(pretrained=True),
                         opt=args
                        )
    criterion_classif = nn.NLLLoss().to(device=device)
    get_logits = args.use_RWLoss or args.use_KL    
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        batch_input = next(loader)
        real_img = batch_input['img'].to(device)
        style_target = batch_input['art_style'].long().to(device)       
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred  = discriminator(fake_img, return_logits=False)
        if get_logits:
            real_pred, style_logits_real = discriminator(real_img_aug, return_logits=True)
            style_logits_prob = F.log_softmax(style_logits_real)
            classif_loss = criterion_classif(style_logits_prob, style_target)
        else:  
            real_pred = discriminator(real_img_aug, return_logits=False)

        d_loss = d_logistic_loss(real_pred, fake_pred)
        loss_dict["d"] = d_loss
        if get_logits:
            loss_dict["classif_loss"] = classif_loss
            d_loss += classif_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_augment_data = torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment += reduce_sum(ada_augment_data)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img, return_logits=False)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        if get_logits:
            op = discriminator(fake_img, return_logits=args.use_KL, return_features=args.use_RWLoss)
            if len(op) == 3:
                fake_pred, style_logits_fake, style_features_fake = op
            elif args.use_KL:
                fake_pred, style_logits_fake = op
            elif args.use_RWLoss:
                fake_pred, style_features_fake = op

        if args.use_KL:
            # computeKL loss
            style_logits_fake_probs = F.log_softmax(style_logits_fake)
            loss_KL = -args.KL_weight * (style_logits_fake_probs / style_logits_fake_probs.size(1)).mean()
        if args.use_RWLoss:
            # compute RW loss
            loss_RW = args.RW_weight * rw_loss_func(features=style_features_fake, 
                                    labels=style_target, 
                                    discriminator=discriminator, 
                                    mode='creative',
                                    dataset=dataloader.dataset,
                                    key='pruned_style_class',
                                    module_kwargs={'return_logits':False,
                                                    'return_features':True}
                                    )
        else:
            fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss
        if args.use_KL:
            loss_dict["KL_loss"] = loss_KL
            g_loss += loss_KL
        # Compute RW loss here.
        if args.use_RWLoss:
            loss_dict["RW_loss"] = loss_RW
            g_loss += loss_RW
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()
        if get_logits:
            classf_loss_val = loss_reduced["classif_loss"].mean().item()
        if args.use_KL:
            KL_loss_val = loss_reduced["KL_loss"].mean().item()
        ## Condition for RW loss.
        if args.use_RWLoss:
            RW_loss_val = loss_reduced["RW_loss"].mean().item()


        if get_rank() == 0:
            description = (f'Iteration: {i}; '
                          f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                          f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                          f"augment: {ada_aug_p:.4f} ")
            if get_logits:
                description += f"classf_loss: {classf_loss_val:0.2f}; "
            if args.use_KL:
                description += f"KL: {KL_loss_val:.2f}; "
            ## Condition for RW loss
            if args.use_RWLoss:
                description += f"RW: {RW_loss_val:.2f}; "  
            if args.no_pbar:
                if (i + 1) % 1000 == 0 or i==0:
                    print(description)
            else:
                pbar.set_description(description)

            if wandb and args.wandb:
                log_dict = {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                if get_logits:
                    log_dict['classification_loss'] = classf_loss_val
                if args.use_KL:
                    log_dict['KL loss'] = KL_loss_val
                ## add RW log here               
                if args.use_RWLoss:
                    log_dict['RW loss'] = RW_loss_val
                wandb.log(log_dict)

            if i % 1000 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    utils.save_image(
                        sample,
                        os.path.join(args.checkpoint_path, f"sample/{str(i).zfill(6)}.png"),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 5000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    os.path.join(args.checkpoint_path, f"checkpoint/{str(i).zfill(6)}.pt"),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )


    parser.add_argument('--data_aug', type=str, default='matlab', choices=['matlab', 'simclr'], help='matlab add noise file or simclr',)
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for dataloader')  
    parser.add_argument('--use_KL', action='store_true', help='use KL loss for the generator')
    parser.add_argument('--KL_weight', type=float, default=1.0, help='weight of KL loss for the generator')
    parser.add_argument('--no_style', action='store_true', help='dont use style classification')
    parser.add_argument('--no_genre', action='store_true', help='dont use genre classification')   
    parser.add_argument('--no_artist', action='store_true', help='dont use genre classification') 
    parser.add_argument('--creativity_label', type=str, default='style_genre_artist' ,help='use CAN Loss loss for the generator')
    parser.add_argument('--name_suffix', type=str, default='test' ,help='name_suffix of the checkpoint folder')
    parser.add_argument('--checkpoint_folder', type=str, default='./' ,help='checkpoint folder')
    parser.add_argument('--batch_mult', type=int, default=1 ,help='batch size multiplier')
    parser.add_argument('--no_pbar', action='store_true')

    ## Random Walk params
    parser.add_argument('--use_RWLoss', action='store_true',  help='use random walk loss')
    parser.add_argument('--RW_tau', type=int, default=3, help='number of hops between points')
    parser.add_argument('--RW_alpha', type=int, default=0.7, help='decay factor to calculate RW loss')
    parser.add_argument('--RW_use_BCE', action='store_true', help='whether to use binary cross entropy or not')   
    parser.add_argument('--RW_weight', type=float, default=1.0, help='weight to use for RW loss')      
    parser.add_argument('--RW_proto_method', type=str, default='random_once', help='weight to use for RW loss allowed(random_once, random_all, nearest_mean_once, nearest_mean_once_diverse)')
    parser.add_argument('--RW_proto_rm', type=float, default=None, help='running mean factor while proto_method is random_all')   
    parser.add_argument('--RW_grad_proto',  action='store_true', help='whether to use gradients of discriminator while calculating the protos')  ## Not used
    parser.add_argument('--RW_feature_extractor', type=str, default='resnet18', help='torchvision model to be used to extract features useful only in case of nearest_mean')   ## Good enough
    parser.add_argument('--classify_creative', action='store_true', help='add additional classification for creative vs imitative mode.')
    parser.add_argument('--KL_creative', action='store_true', help='use KL only in creative mode.')
    parser.add_argument('--normalize_protos_scale', default=None, type=float,  help='whether to L2-normalize-scale prototypes or not')
    parser.add_argument('--only_creative', action='store_true',  help='genreator only in creative mode')
    parser.add_argument('--only_imitative', action='store_true',  help='generator only in imitative mode')    
    parser.add_argument('--verbose', action='store_true', help='print logs')
    parser.add_argument('--add_mode', action='store_true', help='add modes to noise.')
    parser.add_argument('--KL_mode', action='store_true', help='in imitative mode KL loss is negated.')
    parser.add_argument('--disc_imitative', action='store_true', help='imitative mode for real data.')
    parser.add_argument('--lr_scale', type=float, default=1.0, help='imitative mode for real data.')   
    parser.add_argument('--vanilla', action='store_true',  help='uses vanilla styleGAN')
    args = parser.parse_args()
    print(args)
    now = datetime.now()
    ff_n = now.strftime("%m-%d-%y-%H-%M-%S")

    checkpoint_path = os.path.join(args.checkpoint_folder, args.name_suffix, ff_n)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_path, 'sample'), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_path, 'checkpoint'), exist_ok=True)
    args.checkpoint_path = checkpoint_path

    print("Checkpoint path: ", checkpoint_path)
    with open(os.path.join(checkpoint_path, 'args.txt'), 'w') as f:
        f.write(str(args))    

    from wikiart_dataset_pruned import wikiart_dataset_HR
    dataset = wikiart_dataset_HR(args)
    # import pdb; pdb.set_trace();
    dataset.resolution = args.size

    n_discs = []
    if not args.vanilla:
        n_discs.append(args.n_styles)


    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print("WORLD_SIZE", os.environ["WORLD_SIZE"])
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, ndiscs=n_discs
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    # dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=f"stylegan2-{args.name_suffix}")

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
