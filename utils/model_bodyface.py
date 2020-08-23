
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
        flags = (True, use_gan_feat_loss, use_gan_feat_loss, use_vgg_loss, True, True,True,True)
        def loss_filter(g_gan, g_gan_feat,g_gan_face, g_vgg,\
                        d_real, d_fake, d_face_real, d_face_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_gan_face,g_vgg,\
                                        d_real,d_fake, d_face_real, d_face_fake),flags) if f]
        return loss_filter
    
    def __init__(self, label_nc,head_label_nc, output_nc):
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
        self.head_label_nc=head_label_nc
        self.HEAD_SIZE=64
        self.HALF_HEAD=32
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
        self.device=torch.device('cuda')
        
        # Face Enhancer net
        norm_layer1 = functools.partial(nn.InstanceNorm2d, affine=False)
        self.facenetG =  GlobalGenerator(3+head_label_nc, output_nc, norm_layer=norm_layer1).cuda()
        self.facenetG = nn.DataParallel(self.facenetG)
        
        # Discriminator network
        self.n_layers_D=3
        self.num_D=2
        if self.isTrain:
            use_sigmoid = False
            
            netD_input_nc = output_nc+label_nc
            
            if not self.no_instance:
                netD_input_nc += 1
            norm_layer2 = functools.partial(nn.InstanceNorm2d, affine=False)
            self.netD =  MultiscaleDiscriminator(netD_input_nc, 32, self.n_layers_D,norm_layer2, use_sigmoid,self.num_D).cuda()
            self.netD = nn.DataParallel(self.netD)
            
            norm_layer3 = functools.partial(nn.InstanceNorm2d, affine=False)
            self.facenetD =  MultiscaleDiscriminator(3+3+head_label_nc, 32, self.n_layers_D,norm_layer3, use_sigmoid,self.num_D).cuda()
            self.facenetD = nn.DataParallel(self.facenetD)
        
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
            
            self.criterionGAN =  GANLoss(use_lsgan=not self.no_lsgan, tensor=self.Tensor)
            
            self.criterionFeat = torch.nn.L1Loss()
            
            if not self.no_vgg_loss:             
                self.criterionVGG =  VGGLoss().cuda()
                self.criterionVGG=nn.DataParallel(self.criterionVGG)
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_GAN_Face','G_VGG',\
                                               'D_real', 'D_fake','D_face_real', 'D_face_fake')

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
                params = list(self.netG.parameters())+list(self.facenetG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
            
            # optimizer D                        
            params = list(self.netD.parameters())+list(self.facenetD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
    
    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD(fake_query)
        else:
            return self.netD(input_concat)
    def discriminate_face(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.facenetD(fake_query)
        else:
            return self.facenetD(input_concat)

    def forward(self, label, head_lbl,head_cent, image, feat, infer=False):
        # Encode Inputs
        input_label, real_image, feat_map = label, image, feat
        # Fake Generation
        input_concat = input_label
        #---------------------------------Body Generation-----------------------------------------
        fake_image = self.netG(input_concat)
        
        
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
        loss_G_GAN_Face=0
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
                    
        #---------------------------------Head Enhancement-----------------------------------------
        head_input=torch.zeros(len(head_lbl),3+self.head_label_nc,self.HEAD_SIZE,self.HEAD_SIZE,dtype=torch.float32,device=torch.device('cuda:0'))
        
        head_level1_img=torch.zeros(len(head_lbl),3,self.HEAD_SIZE,self.HEAD_SIZE,dtype=torch.float32,device=torch.device('cuda:0'))
        fake_head=head_level1_img.detach()
        real_head=head_level1_img.detach()
        head_level2_img=head_level1_img.detach()
        for bb in range(len(head_lbl)):
            print(head_cent[bb][1]-self.HALF_HEAD,head_cent[bb][1]+self.HALF_HEAD)
            print(head_cent[bb][0]-self.HALF_HEAD,head_cent[bb][0]+self.HALF_HEAD)
            head_level2_img[bb]=fake_image[bb,:,head_cent[bb][1]-self.HALF_HEAD:\
                                                     head_cent[bb][1]+self.HALF_HEAD,\
                                                     head_cent[bb][0]-self.HALF_HEAD:\
                                                     head_cent[bb][0]+self.HALF_HEAD].detach()
            head_input[bb,...]=torch.cat((head_level2_img[bb], head_lbl[bb]), dim=0)
            real_head[bb,...]=real_image[bb,:,head_cent[bb][1]-self.HALF_HEAD:\
                                         head_cent[bb][1]+self.HALF_HEAD,\
                                         head_cent[bb][0]-self.HALF_HEAD:\
                                         head_cent[bb][0]+self.HALF_HEAD]
            
        
        fake_head=self.facenetG(head_input)
        head_level2_img+=fake_head
        
        for bb in range(len(head_lbl)):
            fake_image[bb,:,head_cent[bb][1]-self.HALF_HEAD:\
                                                     head_cent[bb][1]+self.HALF_HEAD,\
                                                     head_cent[bb][0]-self.HALF_HEAD:\
                                                     head_cent[bb][0]+self.HALF_HEAD]\
            +=fake_head[bb]
         # Fake Detection and Loss
        pred_fake_head_pool = self.discriminate_face(head_input, head_level2_img, use_pool=True)
        loss_D_head_fake = self.criterionGAN(pred_fake_head_pool, False)        

        # Real Detection and Loss        
        pred_head_real = self.discriminate_face(head_input, real_head)
        loss_D_head_real = self.criterionGAN(pred_head_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_head_fake = self.facenetD(torch.cat((head_input, head_level2_img), dim=1))        
        loss_G_GAN_face = self.criterionGAN(pred_head_fake, True)  
                   
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat,loss_G_GAN_face, loss_G_VGG,\
                                  loss_D_real, loss_D_fake,loss_D_head_real, loss_D_head_fake ), None if not infer else fake_image ]
#---------------------------------syh----2020.08.06---------------------------
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