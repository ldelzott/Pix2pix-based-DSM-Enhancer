# Pix2pix based DSM-enhancer

This repository hosts an image enhancement tool, based on a Generative Adversarial Network (GAN). The GAN was trained on the altimetric data of the city of Bruxelles, from which thousands of samples were extracted: each sample contains several images of 256x256 pixels. One image contains low-quality altimetric data, another image contains higher-quality altimetric data, and a third image contains the NDVI view of the geographic region depicted by the first two images.

<img width="1165" alt="HighQuality_LowQuality_NDVI-Sample" src="https://github.com/ldelzott/Pix2pix-based-DSM-Enhancer/assets/78430576/21056d41-19ec-4fe3-92c0-718460455e24">

After the training process, the generator part of the GAN can be used to enhance the lower-quality altimetric data into higher-quality data. According to the results obtained for the data of Bruxelles, the model can learn the desired mapping for the training set: i.e. when a low-quality tile from the training set is given as an input, the GAN provides a (learned) enhanced version as an output. A visual inspection showed convincing generalization capabilities for the validation data. However, those performances could be largely influenced by the relative scarcity of the samples extracted from the considered dataset (Bruxelles). One could perform a proper evaluation of the generalization performances of the model by testing the model on a world region made of visual features distinct from those learned from Bruxelles. 

  

## To get started

Please consider the notebook "Pix2pix_predictions_and_assessment.ipynb" to get started with the provided code. 

### ESFPNet
Please consider the .pt and .pth files associated with the ESFPNet model. These files should be placed in the 'docker_ESFPNet' folder.

https://www.dropbox.com/s/2c16izfdmi2e5ns/ESFPNet_DSM_2021_segmentation.pt?dl=0

https://www.dropbox.com/s/f6rxufubx6mbfnf/mit_b4.pth?dl=0

### PIX2PIX
Please consider the two .pth files associated with the Pix2pix model. These files should be placed in the 'docker_pix2pix' folder.

https://www.dropbox.com/s/use4e95lbh5t4l6/latest_net_D.pth?dl=0

https://www.dropbox.com/s/cs08npups6u1vut/latest_net_G.pth?dl=0



## Main reference
The present work arose from an attempt to provide an improved version of the solution proposed in the paper 
**AUTOMATIC LARGE-SCALE 3D BUILDING SHAPE REFINEMENT USING CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS** by K. Bittner, P. d’Angelo,
, M. Korner and, P. Reinartz 
(https://isprs-archives.copernicus.org/articles/XLII-2/103/2018/isprs-archives-XLII-2-103-2018.pdf).

The container implementation uses the docker-related code from https://github.com/fsalmasri/Sustain_docker 

