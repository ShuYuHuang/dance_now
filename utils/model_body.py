
import numpy as np
import torch
from torch import nn
import os
import functools
from . nnmodels import *
##############################################################################
# Main Model
##############################################################################
class Pix2PixHDModel(nn.Module):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg,\
                        d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,\
                                        d_real,d_fake),flags) if f]
        return loss_filter
    
    def __init__(self, label_nc, output_nc):
        super(Pix2PixHDModel, self).__init__() 
        self.isTrain=True
        self.Tensor= torch.cuda.FloatTensor
        
        self.gpu_ids=[0,1,2,3]
        self.resize_or_crop='scale_width'
        if self.resize_or_crop != 'none' or not self.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.use_features = False
        self.gen_features = False
        self.load_features= False
        self.no_instance=True
        self.feat_num=3
        self.data_type=32
        self.label_nc=label_nc
        self.BODY_SIZE=512
        ##### define networks        
        # Generator network
        netG_input_nc = label_nc        
        if not self.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += self.feat_num
            
        norm_layer0 = functools.partial(nn.InstanceNorm2d, affine=False)
        
        self.netG =  GlobalGenerator(netG_input_nc, output_nc, norm_layer=norm_layer0).cuda()
        self.netG = nn.DataParallel(self.netG)
        self.netG.apply(weights_init)
        self.device=torch.device('cuda')
        
        # Discriminator network
        self.n_layers_D=3
        self.num_D=2
        if self.isTrain:
            use_sigmoid = False
            
            netD_input_nc = output_nc+label_nc
            
            if not self.no_instance:
                netD_input_nc += 1
            norm_layer2 = functools.partial(nn.InstanceNorm2d, affine=False)
            self.netD =  MultiscaleDiscriminator(netD_input_nc, 64, self.n_layers_D,norm_layer2, use_sigmoid,self.num_D).cuda()
            self.netD.apply(weights_init)
            self.netD = nn.DataParallel(self.netD)
        
        self.verbose=False
        
        # set loss functions and optimizers
        self.pool_size=0
        self.lr=2e-4
        self.no_ganFeat_loss=False
        self.no_vgg_loss=False
        self.beta1=0.5
        self.no_lsgan=False
        if self.isTrain:
            if self.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool =  ImagePool(self.pool_size)
            self.old_lr = self.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not self.no_ganFeat_loss, not self.no_vgg_loss)
            
            self.criterionGAN =  GANLoss(use_lsgan=not self.no_lsgan, tensor=self.Tensor).cuda()
            self.criterionGAN = nn.DataParallel(self.criterionGAN)
            
            self.criterionFeat = torch.nn.L1Loss()
            
            if not self.no_vgg_loss:             
                self.criterionVGG =  VGGLoss().cuda()
                self.criterionVGG=nn.DataParallel(self.criterionVGG)
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG',\
                                               'D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            self.niter_fix_global=0
            self.n_local_enhancers=1
            if self.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(self.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % self.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
            
            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
    
    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD(fake_query)
        else:
            return self.netD(input_concat)

    def forward(self, input_label, real_image, feat_map, infer=False):
        # Fake Generation
        #---------------------------------Body Generation-----------------------------------------
        fake_image = self.netG(input_label)
        
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD(torch.cat((input_label, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        self.lambda_feat=10.0
        loss_G_GAN_Feat = 0
        if not self.no_ganFeat_loss:
            feat_weights = 4.0 / (self.n_layers_D + 1)
            D_weights = 1.0 / self.num_D
            for i in range(self.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.lambda_feat
        loss_G_VGG = 0
        if not self.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.lambda_feat            
                    
        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN,loss_G_GAN_Feat, loss_G_VGG,\
                                  loss_D_real, loss_D_fake ), None if not infer else fake_image ]
#---------------------------------syh----2020.08.14---------------------------
    def inference(self, label, inst):
        # Encode Inputs        
        input_label, inst_map = label, inst

        # Fake Generation
        if self.use_features:       
            # sample clusters from precomputed features             
            feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label        
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG(input_concat)
        else:
            print(input_concat.shape,input_concat.device)
            fake_image = self.netG(input_concat)
        return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        self.checkpoints_dir="./checkpoints"
        self.name="target"
        self.cluster_path="cluster"
        cluster_path = os.path.join(self.checkpoints_dir, self.name, self.cluster_path)        
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
        if self.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        self.niter_decay=20
        lrd = self.lr / self.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr