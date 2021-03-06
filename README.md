# Denoise_ML
##Introduction
![G_L1](MDense-net.png)
###Dense-net model
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape           Param #     Connected to                     
==================================================================================================
input (InputLayer)              (None, None, None, 1)   0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, None, None, 32)  320         input[0][0]                      
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, None, None, 32)  0           input[0][0]  conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, None, None, 32)  9536        concatenate_1[0][0]              
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, None, None, 32)  128         conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, None, None, 32)  16384       batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, None, None, 64)  18496       conv2d_3[0][0]                   
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, None, None, 96)  0           conv2d_3[0][0]                   
                                                                 conv2d_4[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, None, None, 64)  55296       concatenate_2[0][0]              
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, None, None, 64)  256         conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, None, None, 64)  65536       batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, None, None, 128) 73856       conv2d_6[0][0]                   
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, None, None, 192) 0           conv2d_6[0][0]  conv2d_7[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, None, None, 128) 221312      concatenate_3[0][0]              
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, None, None, 128) 512         conv2d_8[0][0]                   
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, None, None, 128) 0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, None, None, 192) 0           up_sampling2d_1[0][0]  batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, None, None, 64)  110656      concatenate_4[0][0]              
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, None, None, 256) 0           concatenate_4[0][0]              
                                                                 conv2d_9[0][0]                   
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, None, None, 64)  147520      concatenate_5[0][0]              
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, None, None, 64)  256         conv2d_10[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, None, None, 64)  0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, None, None, 96)  0           up_sampling2d_2[0][0]  batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, None, None, 32)  27680       concatenate_6[0][0]              
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, None, None, 128) 0           concatenate_6[0][0]    conv2d_11[0][0]                  
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, None, None, 32)  36896       concatenate_7[0][0]              
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, None, None, 32)  128         conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, None, None, 32)  9248        batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, None, None, 32)  9248        conv2d_13[0][0]                  
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, None, None, 1)   289         conv2d_14[0][0]                  
==================================================================================================
```

###Dependencies 
```
python3
keras
numpy
argparse
random
string

```

### train model
#####step1. ready the train data in train_data/noisy & train_data/clean
#####step2. execute 
######         $ python densenet_denoisy.py
        

###1. test wave
```
$ python densenet_denoise.py test_data/p226_110.wav
```

![](test_data/p226_110.png)  ![](test_data/p226_110_densenet.png)
