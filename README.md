# Python Code for Kernel Method-based Kernel Intensity Estimator (*Under Construction*)
This library provides kernel method-based kernel intensity estimator (K<sup>2</sup>IE) implemented in Tensorflow. K<sup>2</sup>IE is a kernel method model to estimate intensity functions with the least squares loss functionals. For details, see our ICML2025 paper [1].

The code was tested on Python 3.10.8, tensorflow-deps 2.10.0, tensorflow-macos 2.10.0, and tensorflow-metal 0.6.0.

# Installation
To install latest version:
```
pip install git+https://github.com/HidKim/K2IE
```

# Basic Usage
Import RFM kernel class:
```
from HidKim_K2IE import kernels_rfm
```
Initialize RFM kernel
```
ker = kernels_rfm(kernel='gaussian', n_rand_feature=500, seed=0, n_dim=2, qmc=True)
```
Import K<sup>2</sup>IE class:
```
from HidKim_K2IE import k2_intensity_estimator
```
Initialize K<sup>2</sup>IE:
```
k2ie = k2_intensity_estimator(kernel=ker)
```
- `kernel`: *string, default='Gaussian'* <br> 
  >The kernel function for Gaussian process. Only 'Gaussian' is available now.
- `eq_kernel`:  *string, default='RFM'* <br>
  >The approach to constructing equivalent kernel. Only 'RFM' is available now.  
- `eq_kernel_options`:  *dict, default={'n_rfm':500}* <br>
  >The options for constructing equivalent kernel. `'n_rfm'` specifies the number of feature samples for the random feature map approach to constructing equivalent kernel.
  
Fit SurvPP with data:
```
time = model.fit(formula, df, set_par, lr=0.05, display=True)
```
- `formula`: *string in the form 'Surv(Start, Stop, Event) ~ cov1 + cov2 + ...'* <br> 
  >Identify the column labels of start time (Start), stop time (Stop), event indicator (Event, 1 represents that an event occurred at stop time, and 0 represents that observation is right-censored), and covariates used for survival analysis.
- `df`:  *pandas.DataFrame*  <br>
  > The survival data in counting process format. Each row represents a sub-interval for a subject, and should contain start and end time points, event indicator (0/1), and values of covariates in the sub-interval.  
- `set_par`:  *ndarray of shape (n_candidates, dim_hyperparameter)*  <br>
  >The set of hyper-parameters for hyper-parameter optimization. The optimization is performed by maximizing the marginal likelihood.
- `lr`: *float, default=0.05* <br>
  >The learning rate for gradient descent algorithm (Adam).
- `display`:  *bool, default=True*  <br>
  >Display the summary of the data and the fitting. 
- **Return**: *float* <br>
  >The execution time.

Predict hazard function on specified covariate values:
```
r_est = model.predict(y, conf_int=[0.025,0.5,0.975])
```
- `y`: *ndarray of shape (n_points, dim_covariate)* <br> 
  >The points on covariate domain for evaluating intensity values.
- `conf_int`:  *ndarray of shape (n_quantiles,), default=[.025, .5, .975]*  <br>
  > The quantiles for predicted hazard function.
- **Return**: *ndarray of shape (n_quantiles, n_points)* <br>
  >The predicted values of hazard function at the specified points.

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