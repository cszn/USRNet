# USRNet

USRNet can handle `super-resolution (x2, x3 and x4) and deblurring` for various blur kernels via `a single model`!

Classical (traditional) SISR degradation model
----------
<img src="figs/classical_degradation_model.png" width="440px"/> 


Motivation
----------
<img src="figs/fig1.png" width="440px"/>


While the classical degradation model can result in various LR images for an HR image, with different blur kernels, scale factors and noise, the study of learning `a single end-to-end trained deep model` to invert all such LR images to HR image is still lacking.

Unfolding algorithm
----------
#TODO

Deep unfolding SR network (USRNet)
----------

#TODO

Approximated bicubic kernel under classical SR degradation model assumption
----------
<img src="figs/bicubic_kernelx2.png" width="450px"/> 
<img src="figs/bicubic_kernelx3.png" width="450px"/> 
<img src="figs/bicubic_kernelx4.png" width="450px"/>



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
