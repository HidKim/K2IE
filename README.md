# Python Code for Kernel Method-based Kernel Intensity Estimator
This library provides kernel method-based kernel intensity estimator (K<sup>2</sup>IE) implemented in Tensorflow. K<sup>2</sup>IE is a kernel method model to estimate intensity functions with the least squares loss functionals. For details, see our ICML2025 paper [1].

The code was tested on Python 3.10.8, tensorflow-deps 2.10.0, tensorflow-macos 2.10.0, and tensorflow-metal 0.6.0.

# Installation
To install latest version:
```
pip install git+https://github.com/HidKim/K2IE
```

# Basic Usage
Import random Fourier Feature map (RFM) kernel class:
```
from HidKim_K2IE import kernels_rfm
```
Initialize RFM kernel
```
ker = kernels_rfm(n_dim, kernel='gaussian', n_rand_feature=500, seed=0, n_dim=2, qmc=True)
```
- `n_dim`:  *int* <br>
  >The dimensionality of inputs.
- `kernel`: *string, default='gaussian'* <br> 
  >The kernel function: 'gaussian', 'laplace', and 'cauchy'.
- `n_rand_feature`:  *int, default=0* <br>
  >The number of random Fourier features.  
- `seed`:  *int, default=0* <br>
  >The seed for sampling Fourier features.

Import K<sup>2</sup>IE class:
```
from HidKim_K2IE import k2_intensity_estimator
```
Initialize K<sup>2</sup>IE:
```
k2ie = k2_intensity_estimator(kernel=ker)
```
- `kernel`: *kernels_rfm instance* <br> 
  
Fit K<sup>2</sup>IE with data:
```
time = model.fit(d_spk, d_region, a, b)
```
- `d_spk`: *ndarray of shape (n_points, dim_points)* <br>
  > The training point data.  
- `d_region`: *ndarray of shape (n_subregion, dim_points, 2)*  <br>
  >The observation region. e.g.) [ [[0,1],[0,1]], [[1,3],[0,1]] ] represents that there are two adjacent subdomains: one is a unit square, and the other is a rectangle with a length of 2 in the x-direction and a length of 1 in the y-direction.
- `a`: *float* <br>
  >The amplitude hyper-parameter for shift-invariant kernel function, or the regularlization hyper-parameter '\gamma' in ICML2025 paper.
- `b`:  *float*  <br>
  >The scale hyper-parameter for shift-invariant kernel function. 

Evaluate the integral of the squared intensity function over a specified domain (used for closs-validation of hyper-parameter):
```
y = k2ie.predict_integral_squared(region)
```
- `region`: *ndarray of shape (n_subregion, dim_points, 2)* <br>
  > The region for integral.  
- **Return**: *float* <br>
  >The evaluated itengral of the squared intensity function.

Predict intensity function on specified inputs:
```
r_est = model.predict(y)
```
- `y`: *ndarray of shape (n_points, dim_points)* <br> 
  >The points on input space for evaluating intensity values.
- **Return**: *ndarray of shape (n_points,)* <br>
  >The predicted values of intensity function at the specified points.

# Reference
1. Hideaki Kim, Tomoharu Iwata, Akinori Fujino. "K<sup>2</sup>IE: Kernel Method-based Kernel Intensity Estimators for Inhomogeneous Poisson Processes", *International Conference on Machine Learning*, 2025.
```
@inproceedings{kim2025k2ie,
  title={K$^2$IE: Kernel Method-based Kernel Intensity Estimators for Inhomogeneous Poisson Processes},
  author={Kim, Hideaki and Iwata, Tomoharu and Fujino, Akinori},
  booktitle={International Conference on Machine Learning},
  volume={*},
  pages={*--*},
  year={2025}
}
``` 

# License
Released under "SOFTWARE LICENSE AGREEMENT FOR EVALUATION". Be sure to read it.

# Contact
Feel free to contact the author Hideaki Kim (hideaki.kin@ntt.com).