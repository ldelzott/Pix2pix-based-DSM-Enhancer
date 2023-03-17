import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = Unet256(input_nc, output_nc, ngf, use_dropout=use_dropout)
    #net = ResUnet(input_nc, output_nc, ngf, use_dropout=use_dropout)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer) 
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'basic_rewritten':
        print("Initializing basic_rewritten discriminator")
        net = RewrittenNLayerDiscriminator(input_nc, ndf, norm_layer=norm_layer) # Default configuration: input_nc=2, ndf=64, n_layer = 3
    elif netD == 'custom':
        print("Initializing custom discriminator")
        net = CustomDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
    



class Unet256(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False):
        # Construct a Unet generator
        # Parameters:
        #    input_nc (int)  -- the number of channels in input images
        #    output_nc (int) -- the number of channels in output images
        #    ngf (int)       -- the number of filters in the last conv layer
        #   norm_layer      -- normalization layer


        super(Unet256, self).__init__()
        
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        self.downblock_8 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False))
        
        self.downblock_7 = nn.Sequential(nn.LeakyReLU(0.2, True), 
                                         nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=False), 
                                         norm_layer(ngf*2))
        
        self.downblock_6 = nn.Sequential(nn.LeakyReLU(0.2, True), 
                                         nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, bias=False), 
                                         norm_layer(ngf*4))
        
        self.downblock_5 = nn.Sequential(nn.LeakyReLU(0.2, True), 
                                         nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, bias=False), 
                                         norm_layer(ngf*8))
        
        self.downblock_4 = nn.Sequential(nn.LeakyReLU(0.2, True), 
                                         nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=False), 
                                         norm_layer(ngf*8))
        
        self.downblock_3 = nn.Sequential(nn.LeakyReLU(0.2, True), 
                                         nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=False), 
                                         norm_layer(ngf*8))
        
        self.downblock_2 = nn.Sequential(nn.LeakyReLU(0.2, True), 
                                         nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=False), 
                                         norm_layer(ngf*8))
        
        self.downblock_1 = nn.Sequential(nn.LeakyReLU(0.2, True), 
                                         nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=False))
        
        
        self.upblock_1 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf*8, ngf*8,kernel_size=4, stride=2, padding=1, bias=False), 
                                       norm_layer(ngf*8))
        
        if use_dropout:
        
            self.upblock_2 = nn.Sequential(nn.ReLU(True), 
                                           nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=False), 
                                           norm_layer(ngf*8), 
                                           nn.Dropout(0.5))
            self.upblock_3 = nn.Sequential(nn.ReLU(True), 
                                           nn.ConvTranspose2d(ngf*8, ngf*8,kernel_size=4, stride=2, padding=1, bias=False), 
                                           norm_layer(ngf*8), 
                                           nn.Dropout(0.5))
            self.upblock_4 = nn.Sequential(nn.ReLU(True), 
                                           nn.ConvTranspose2d(ngf*8, ngf*8,kernel_size=4, stride=2,padding=1, bias=False), 
                                           norm_layer(ngf*8), 
                                           nn.Dropout(0.5))
        else:
            self.upblock_2 = nn.Sequential(nn.ReLU(True), 
                                           nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=False), 
                                           norm_layer(ngf*8))
            self.upblock_3 = nn.Sequential(nn.ReLU(True), 
                                           nn.ConvTranspose2d(ngf*8, ngf*8,kernel_size=4, stride=2, padding=1, bias=False), 
                                           norm_layer(ngf*8))
            self.upblock_4 = nn.Sequential(nn.ReLU(True), 
                                           nn.ConvTranspose2d(ngf*8, ngf*8,kernel_size=4, stride=2,padding=1, bias=False), 
                                           norm_layer(ngf*8))
        
        self.upblock_5 = nn.Sequential(nn.ReLU(True), 
                                       nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False), 
                                       norm_layer(ngf*4))
        
        self.upblock_6 = nn.Sequential(nn.ReLU(True), 
                                       nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False), 
                                       norm_layer(ngf*2))
        
        self.upblock_7 = nn.Sequential(nn.ReLU(True), 
                                       nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False), 
                                       norm_layer(ngf))
        
        self.upblock_8 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1),
                                       nn.Tanh(),)
        
        

  
    def forward(self, input):
        
        d8 = self.downblock_8(input)                        # Input [1,1,256,256]
        d7 = self.downblock_7(d8)                           # Input [1,64,128,128]
        d6 = self.downblock_6(d7)                           # Input [1,128,64,64]
        d5 = self.downblock_5(d6)                           # Input [1,256,32,32]
        d4 = self.downblock_4(d5)                           # Input [1,512,16,16]
        d3 = self.downblock_3(d4)                           # Input [1,512,8,8]
        d2 = self.downblock_2(d3)                           # Input [1,512,4,4]
        d1 = self.downblock_1(d2)                           # Input [1,512,2,2]
        
        #u1_cat=torch.cat([d2, self.upblock_1(d1)], 1)       # Output [1,512,2,2]
        u1_cat=torch.add(d2, self.upblock_1(d1)) 
        #u2_cat=torch.cat([d3, self.upblock_2(u1_cat)], 1)   # Output [1,1024,4,4]
        u2_cat=torch.add(d3, self.upblock_2(u1_cat))
        #u3_cat=torch.cat([d4, self.upblock_3(u2_cat)], 1)   # Output [1,1024,8,8]
        u3_cat=torch.add(d4, self.upblock_3(u2_cat))
        #u4_cat=torch.cat([d5, self.upblock_4(u3_cat)], 1)   # Output [1,1024,16,16]
        u4_cat=torch.add(d5, self.upblock_4(u3_cat))
        #u5_cat=torch.cat([d6, self.upblock_5(u4_cat)], 1)   # Output [1,512,32,32]
        u5_cat=torch.add(d6, self.upblock_5(u4_cat))
        #u6_cat=torch.cat([d7, self.upblock_6(u5_cat)], 1)   # Output [1,256,64,64]
        u6_cat=torch.add(d7, self.upblock_6(u5_cat))
        #u7_cat=torch.cat([d8, self.upblock_7(u6_cat)], 1)   # Output [1,128,128,128]
        u7_cat=torch.add(d8, self.upblock_7(u6_cat))
        u8_cat = self.upblock_8(u7_cat)                     # Output [1,1,256,256]
        
        return u8_cat
        



# The following code was taken from https://github.com/rishikksh20/ResUnet/blob/master/core/res_unet.py
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2
    
class ResUnet(nn.Module):
    def __init__(self,input_nc, output_nc, ngf=64, use_dropout=False, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(input_nc, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(input_nc, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], output_nc, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output


###########################################################################################



class RewrittenNLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):  # Default configuration: input_nc=2, ndf=64, n_layer = 3
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(RewrittenNLayerDiscriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), # [2,256,256]->[64,128,128]
                                   nn.LeakyReLU(0.2, True), 
                                   
                                   nn.Conv2d(ndf * 1, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), #[64,128,128]->[128,64,64]
                                   norm_layer(ndf * 2), 
                                   nn.LeakyReLU(0.2, True), 
                                   
                                   nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), # [128,64,64] -> [256,32,32]
                                   norm_layer(ndf * 4),
                                   nn.LeakyReLU(0.2, True), 
                                   
                                   nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False), # [256,32,32] -> [512,32,32]
                                   norm_layer(ndf * 8),
                                   nn.LeakyReLU(0.2, True),
                                   
                                   nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)) # [512, 32, 32] -> [1, 32, 32]

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
#tensor_rand = torch.rand((1,2,256,256))
#model = CustomDiscriminator(2, ndf=64, n_layers=3,)
#print(model(tensor_rand).shape)

class CustomDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):  # Default configuration: input_nc=2, ndf=64, n_layer = 3
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(CustomDiscriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(input_nc, 32, kernel_size=4, stride=2, padding=1), 
                                   nn.LeakyReLU(0.2, True), 
                                   
                                   nn.Conv2d(32 * 1, 32 * 2, kernel_size=4, stride=2, padding=1, bias=False), 
                                   norm_layer(32 * 2), 
                                   nn.LeakyReLU(0.2, True), 
                                   
                                   nn.Conv2d(32 * 2, 32 * 4, kernel_size=4, stride=2, padding=1, bias=False),
                                   norm_layer(32 * 4),
                                   nn.LeakyReLU(0.2, True), 
                                   
                                   nn.Conv2d(32 * 4, 32 * 8, kernel_size=4, stride=1, padding=1, bias=False),
                                   norm_layer(32 * 8),
                                   nn.LeakyReLU(0.2, True),
                                   
                                   nn.Conv2d(32 * 8, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
#tensor_rand = torch.rand((1,2,256,256))
#model = CustomDiscriminator(2, ndf=64, n_layers=3,)
#print(model(tensor_rand).shape)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d): 
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
