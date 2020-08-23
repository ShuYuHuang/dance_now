
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
    
    def init_loss_filter(self,G_GAN_Face=False,G_VGG_face=False,D_face_real=False, D_face_fake=False):
        flags = (G_GAN_Face,G_VGG_face,D_face_real, D_face_fake)
        def loss_filter(g_gan_face,g_vgg_face,d_face_real, d_face_fake):
            return [l for (l,f) in zip((g_gan_face,g_vgg_face,d_face_real, d_face_fake),flags) if f]
        return loss_filter
    
    def __init__(self, head_label_nc, output_nc):
        super(Pix2PixHDModel, self).__init__() 
        self.isTrain=True
        self.Tensor= torch.cuda.FloatTensor
        
        self.gpu_ids=[0,1,2,3]
        self.resize_or_crop='scale_width'
        if self.resize_or_crop != 'none' or not self.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.no_instance=True
        
        self.data_type=32
        self.head_label_nc=head_label_nc
        self.feat_num=3+head_label_nc
        #self.feat_num=head_label_nc
        self.output_nc=output_nc
        self.HEAD_SIZE=64
        self.HALF_HEAD=32
        self.BODY_SIZE=512
        ##### define networks    
        
        # Face Enhancer net
        #norm_layer1 = functools.partial(nn.InstanceNorm2d, affine=False)
        norm_layer1=nn.BatchNorm2d
        self.facenetG =  GlobalGenerator(self.feat_num, self.output_nc, norm_layer=norm_layer1).cuda()
        self.facenetG.apply(weights_init)
        self.facenetG = nn.DataParallel(self.facenetG)
        
        # Discriminator network
        self.n_layers_D=3
        self.num_D=2
        if self.isTrain:
            use_sigmoid = False
            
            
            self.facenetD =  FaceNLayerDiscriminator(self.output_nc, 64, self.n_layers_D,use_sigmoid=use_sigmoid).cuda()
            self.facenetD.apply(weights_init)
            self.facenetD = nn.DataParallel(self.facenetD)
        
        self.verbose=False
        
        # set loss functions and optimizers
        self.pool_size=0
        self.lr=2e-4
        self.beta1=0.5
        self.no_lsgan=False
        if self.isTrain:
            if self.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool =  ImagePool(self.pool_size)
            self.old_lr = self.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(True,True,True,True)
            
            # GAN loss
            #self.criterionGAN =  GANLoss(use_lsgan=not self.no_lsgan, tensor=self.Tensor).cuda()
            self.Gloss=nn.MSELoss()
        
            # Discriminator loss
            self.Dloss=nn.MSELoss()
            
            # VGG loss
            self.criterionVGG =  VGGLoss().cuda()
            self.criterionVGG=nn.DataParallel(self.criterionVGG)
            
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN_Face',"G_VGG_face",\
                                               'D_face_real', 'D_face_fake')

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

                params_dict = dict(self.facenetG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(self.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % self.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.facenetG.parameters()) 
            self.optimizer_G_face = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
            
            
            # optimizer D                        
            params = list(self.facenetD.parameters())    
            self.optimizer_D_face = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
            
    def discriminate_face(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.facenetD(fake_query)
        else:
            return self.facenetD(input_concat)

    def forward(self, head_lbl,fake_head, real_head, infer=False):
        #---------------------------------Head Enhancement----------------------------------------
        
        #---------------------------------Train Generator----------------------------------------
        head_input=torch.cat((fake_head, head_lbl), dim=1)
        head_buff=self.facenetG(head_input)
        #enhanced_head=head_buff
        enhanced_head=head_buff+fake_head
            
        # GAN loss (Fake Passability Loss)     
        fake_features = self.facenetD.module.extract_features(enhanced_head)
        with torch.no_grad():
            real_features = self.facenetD.module.extract_features(real_head)
        loss_G_GAN_face = self.Gloss(fake_features, real_features)
        
        self.lambda_feat=2.0
        
        loss_G_VGG_face =self.criterionVGG(enhanced_head, real_head) * self.lambda_feat 
        
        loss_G=loss_G_GAN_face+loss_G_VGG_face
        loss_Gavg = torch.mean(loss_G)
                ############### Backward Pass ####################
        # update generator weights
        self.optimizer_G_face.zero_grad()
        loss_Gavg.backward()
        self.optimizer_G_face.step()
        #---------------------------------Train Discriminator----------------------------------------
        
        
        reapeat_time=4
        for _ in range(reapeat_time):
            with torch.no_grad():
                head_input=torch.cat((fake_head, head_lbl), dim=1)
                head_buff=self.facenetG(head_input)
                enhanced_head=head_buff+fake_head
                #enhanced_head=head_buff
            # Real Detection and Loss 
            real_labels = self.facenetD(real_head)
            with torch.no_grad():
                ones = torch.ones_like(real_labels)
                zeros = torch.zeros_like(real_labels)

                # one sided label smoothing for vanilla gan
                ones.uniform_(.9, 1.1)
                zeros.uniform_(-.1, .1)

            loss_D_head_real = self.Dloss(real_labels, ones)

            fake_labels = self.facenetD(enhanced_head)
            loss_D_head_fake = self.Dloss(fake_labels, zeros)     

            loss_D=loss_D_head_real+loss_D_head_fake
            loss_Davg = torch.mean(loss_D)
            ############### Backward Pass ####################
            # update discriminator weights
            self.optimizer_D_face.zero_grad()

            loss_Davg.backward()
            self.optimizer_D_face.step()
        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN_face,loss_G_VGG_face,loss_D_head_real, loss_D_head_fake),None if not infer else enhanced_head ]
#---------------------------------syh----2020.08.06---------------------------
    def inference(self, head_lbl,fake_head):
        head_input=torch.cat((fake_head, head_lbl), dim=1)
        with torch.no_grad():
            enhanced_head=self.facenetG(head_input)
            enhanced_head+=fake_head
        return enhanced_head

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

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())         
        self.optimizer_G_face = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
        if self.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        self.niter_decay=20
        lrd = self.lr / self.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D_face.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_face.param_groups:
            param_group['lr'] = lr
        if self.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr