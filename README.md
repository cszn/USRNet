# Deep unfolding network for image super-resolution
Kai Zhang, [Luc Van Gool](https://vision.ee.ethz.ch/people-details.OTAyMzM=.TGlzdC8zMjQ4LC0xOTcxNDY1MTc4.html), [Radu Timofte](http://people.ee.ethz.ch/~timofter/)  
_[Computer Vision Lab](https://vision.ee.ethz.ch/the-institute.html), ETH Zurich, Switzerland_

[paper]

___________

* [Classical SISR degradation model](#classical-sisr-degradation-model)
* [Motivation](#motivation)
* [Unfolding algorithm](#unfolding-algorithm)
* [Deep unfolding SR network](#deep-unfolding-sr-network)
* [Models](#models)
* [Codes](#codes)
* [Visual results](#visual-results)
* [Approximated bicubic kernel under classical SR degradation model assumption](#approximated-bicubic-kernel-under-classical-sr-degradation-model-assumption)
* [Results on bicubicly degradated LR images](#results-on-bicubicly-degradated-lr-images)
* [Generalizability](#generalizability)
* [Real image SR](#real-image-sr)
* [References](#references)

Classical SISR degradation model
----------
<img src="figs/classical_degradation_model.png" width="440px"/> 


Motivation
----------
<img src="figs/category.png" width="536px"/>

Learning-based single image super-resolution (SISR)
methods are continuously showing superior effectiveness
and efficiency over traditional model-based methods, largely
due to the end-to-end training. However, different from
model-based methods that can handle the SISR problem
with different scale factors, blur kernels and noise levels
under a unified MAP (maximum a posteriori) framework,
learning-based methods (e.g., SRMD [3]) generally lack such flexibility.

```
[1] "Learning deep CNN denoiser prior for image restoration." CVPR, 2017.
[2] "Deep plug-and-play super-resolution for arbitrary blur kernels." CVPR, 2019.
[3] "Learning a single convolutional super-resolution network for multiple degradations." CVPR, 2018.
```

<img src="figs/fig1.png" width="440px"/>


While the classical degradation model can result in various LR images for an HR image, with different blur kernels, scale factors and noise, the study of learning *`a single end-to-end trained deep model`* to invert all such LR images to HR image is still lacking.


Unfolding algorithm
----------
By unfolding the MAP inference via a half-quadratic splitting
algorithm, a fixed number of iterations consisting of alternately solving a `data subproblem` and a `prior subproblem`
can be obtained.

#TODO

Deep unfolding SR network
----------
We proposes an end-to-end trainable unfolding network which leverages both learning-based
methods and model-based methods. 


USRNet inherits the `flexibility of model-based methods` to super-resolve
blurry, noisy images for different scale factors via `a single
model`, while maintaining the `advantages of learning-based methods`.

#TODO

Models
----------

- USRNet

- USRGAN

- USRNet-tiny

Codes
----------

#TODO

Visual results
----------
<img src="figs/test_50_x4_k81.png" width="50px"/> <img src="figs/test_50_k8_s4_mse.png" width="200px"/> <img src="figs/test_50_k8_s4_gan.png" width="200px"/> 

<img src="figs/test_36_L1.png" width="50px"/> <img src="figs/test_36_k1_s4_mse.png" width="200px"/> <img src="figs/test_36_k1_s4_gan.png" width="200px"/> 

<img src="figs/test_18_x4_k121.png" width="50px"/> <img src="figs/test_18_k12_s4_mse.png" width="200px"/> <img src="figs/test_18_k12_s4_gan.png" width="200px"/> 

From left to right: `LR image`; `result of USRNet(x4)`; `result of USRGAN(x4)`

Approximated bicubic kernel under classical SR degradation model assumption
----------
<img src="figs/bicubic_kernelx2.png" width="450px"/> 
<img src="figs/bicubic_kernelx3.png" width="450px"/> 
<img src="figs/bicubic_kernelx4.png" width="450px"/>

Results on bicubicly degradated LR images
----------
<img src="figs/test_19_x4_k1_L1.png" width="88px"/> <img src="figs/test_19_x4_k1_E.png" width="352px"/> 

<img src="figs/test_35_x4_k1_L.png" width="88px"/> <img src="figs/test_35_x4_k1_E.png" width="352px"/> 

<img src="figs/test_42_x4_k1_L.png" width="88px"/> <img src="figs/test_42_x4_k1_E.png" width="352px"/> 

From left to right: `LR image`; `result of USRGAN(x4)`


Generalizability
----------
<img src="figs/g1_LR.png" width="110px"/> <img src="figs/g1_HR.png" width="330px"/> 

`(a) Result of USRNet(x3) for kernel size 67x67`  
Even trained with kernel size 25x25, USRNet generalizes well to much larger kernel size.

<img src="figs/g2_LR.png" width="110px"/> <img src="figs/g2_HR.png" width="330px"/> 

`(b) Result of USRGAN(x3) for kernel size 70x70`  
Even trained with kernel size 25x25 and scale factor 4, USRGAN generalizes well to much larger kernel size and another scale factor 3.

Real image SR
----------
<img src="figs/chip.png" width="88px"/> `LR image, 108x52`

<img src="figs/chip_x1.png" width="88px"/> `USRNet(x1), deblurring, 108x52`

<img src="figs/chip_x2.png" width="176px"/> `USRNet(x2), 216x104`

<img src="figs/chip_x3.png" width="264px"/> `USRNet(x3), 324x156`

<img src="figs/chip_x4.png" width="352px"/> `USRNet(x4), 432x208`

<img src="figs/chip_x5.png" width="440px"/> `USRNet(x5), 540x260`

<img src="figs/chip.png" width="440x"/> `Zoomed LR image`

The above results are obtained via `a single USRNet model` by setting different scale factors (x1, x2, x3, x4, x5) and Gaussian blur kernels (with width 0.6, 1.0, 1.8, 2.4, 3).

<img src="figs/butterfly.bmp" width="440x"/> `Zoomed LR image, 256x256`

<img src="figs/butterfly_x2.bmp" width="440x"/> `USRNet(x2), 512x512`

<img src="figs/comic.png" width="440x"/> `Zoomed LR image, 250x361`

<img src="figs/comic_x2.png" width="440x"/> `USRNet(x2), 500x722`


References
----------
```
@inproceedings{zhang2020deep, % USRNet
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}

```
