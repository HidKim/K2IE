import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import dill, re
import tensorflow as tf
import numpy as np
from pylab import *
from HidKim_K2IE import kernels_rfm, k2_intensity_estimator, \
     permanental_process, kernel_intensity_estimator

tf.config.set_visible_devices([], 'GPU')

def mc_integration(func,region,n_sample=int(1e4)):
     np.random.seed(0)
     p = array([np.random.uniform(r[0],r[1],n_sample) for r in region]).T
     return np.mean(func(p))

# Load Data #############################################################
dfile = 'data/2D_p10.dill' # Table 2 in ICML2025 paper, 'p' is the prob. of domain selection
data = dill.load(open(dfile,'rb'))
d_spk, f_true, region = data['spk'], data['f'], data['region']
d_region_tr, d_region_tr_out = data['region_train'], data['region_train_out']

# Make Directory for Result ############################################# 
dir_result = 'result/'+re.split(r'[/.]+',dfile)[1]+'/'
if not os.path.exists(dir_result):
    os.makedirs(dir_result)

# Set Candidates of Hyper-parameter #####################################
set_a = exp(linspace(log(0.1),log(100),10))
set_b = exp(linspace(log(0.1),log(100),10))/(region[0][0][1]-region[0][0][0])
set_par = [[x,y] for x in set_a for y in set_b]

# Kernel function with Random Fourier Features ##########################
ker = kernels_rfm(kernel='gaussian', n_rand_feature=500,
                  seed=0, n_dim=2, qmc=True)

# Setting for Cross-validation ##########################################
n_split = 5
p_prob = 0.6

ise = {'KIE':[], 'K2IE':[], 'PP':[]} # Integrate squared error
iae = {'KIE':[], 'K2IE':[], 'PP':[]} # Integrated absolute error
cpu = {'KIE':[], 'K2IE':[], 'PP':[]} # CPU time for estimation
est = {'KIE':[], 'K2IE':[], 'PP':[]} # Estimated intensity 

for ii, (spk,region_tr,region_tr_out) in enumerate(zip(d_spk,d_region_tr,d_region_tr_out)):
    print('Trial#:',ii)

    # Plot True Intensity Function ######################################
    subplot(2,2,1)
    grid_x = linspace(region[0][0][0],region[0][0][1],500)
    grid_y = linspace(region[0][1][0],region[0][1][1],500)
    xx,yy = meshgrid(grid_x,grid_y)
    x_ev = vstack((xx.flatten(),yy.flatten())).T
    y = f_true(x_ev)
    vmax = max(y)*1.2
    imshow(y.reshape(len(grid_x),len(grid_y)), extent=[grid_x[0],grid_x[-1],grid_y[0],grid_y[-1]],
           origin='lower', aspect='auto', cmap='jet',vmin=0, vmax=vmax) 
    for s in spk:
         plot(s[0],s[1],'k.',markersize=1.5)
    for x in region_tr_out:
         fill_between(x[0], [x[1][1]]*2, [x[1][0]]*2, fc='w', hatch="//") 
    
    # Split Data into Training and Test #################################
    d_train, d_test = [], []
    np.random.seed(0)
    for jj in range(n_split):
         indx = np.random.binomial(1, p_prob, size=len(spk))
         d_train.append(spk[indx==1])
         d_test.append(spk[indx!=1])     
         
    # Kernel Intensity Estimator (KIE) ##################################
    #####################################################################
    model = 'KIE'
    kie = kernel_intensity_estimator(ker)
    score, par = [], []

    # Hyper-parameter Optimization: Start
    for b in set_b:
        sc = []
        for d_tr, d_te in zip(d_train, d_test):
            
            _ = kie.fit(d_tr,region_tr,b)
            factor = (1.-p_prob)/p_prob
            rate = lambda x: kie.predict(x).numpy()*factor
            tlog = sum(log(rate(d_te)))
            for reg in region_tr:
                tlog -= mc_integration(rate,reg,n_sample=1000)
            sc.append(tlog)
        
        score.append(mean(sc))
        par.append(b)
    opt_b = array(par)[argmax(array(score))]
    # Hyper-parameter Optimization: End
    
    elapse_t = kie.fit(spk,region_tr,opt_b)
    y = kie.predict(x_ev).numpy()
    subplot(2,2,2)
    imshow(y.reshape(len(grid_x),len(grid_y)), extent=[grid_x[0],grid_x[-1],grid_y[0],grid_y[-1]],
           origin='lower', aspect='auto', cmap='jet',vmin=0, vmax=vmax)
    for s in spk:
         plot(s[0],s[1],'k.',markersize=1.5)
    for x in region_tr_out:
         fill_between(x[0], [x[1][1]]*2, [x[1][0]]*2, fc='w', hatch="//")
    
    func = lambda x: (kie.predict(x)-f_true(x))**2
    int_sq_err = mean([mc_integration(func,reg) for reg in region_tr])
    ise[model].append(int_sq_err)
    func = lambda x: abs(kie.predict(x)-f_true(x))
    int_ab_err = mean([mc_integration(func,reg) for reg in region_tr])
    iae[model].append(int_ab_err)
    cpu[model].append(elapse_t)
    est[model].append(y)
    
    # Kernel Method-based Kernel Intensity Estimator (K2IE) #############
    #####################################################################
    model = 'K2IE'
    k2ie = k2_intensity_estimator(ker)
    score, par = [], []

    # Hyper-parameter Optimization: Start
    for [a,b] in set_par:
        sc = []
        for d_tr, d_te in zip(d_train, d_test):
            
            factor = (1.-p_prob)/p_prob
            _ = k2ie.fit(d_tr,region_tr,a/factor,b)
            rate = lambda x: k2ie.predict(x).numpy()*factor
            tls = 2.*sum(rate(d_te)) \
                - k2ie.predict_integral_squared(region_tr)*factor**2
            sc.append(tls)
        
        score.append(mean(sc))
        par.append([a,b])
    [opt_a, opt_b] = array(par)[np.nanargmax(array(score))]
    # Hyper-parameter Optimization: End
    
    elapse_t = k2ie.fit(spk,region_tr,opt_a,opt_b)
    y = k2ie.predict(x_ev).numpy()
    subplot(2,2,3)
    imshow(y.reshape(len(grid_x),len(grid_y)), extent=[grid_x[0],grid_x[-1],grid_y[0],grid_y[-1]],
           origin='lower', aspect='auto', cmap='jet',vmin=0, vmax=vmax)
    for s in spk:
         plot(s[0],s[1],'k.',markersize=1.5)
    for x in region_tr_out:
         fill_between(x[0], [x[1][1]]*2, [x[1][0]]*2, fc='w', hatch="//")
    
    func = lambda x: (k2ie.predict(x)-f_true(x))**2
    int_sq_err = mean([mc_integration(func,reg) for reg in region_tr])
    ise[model].append(int_sq_err)
    func = lambda x: abs(k2ie.predict(x)-f_true(x))
    int_ab_err = mean([mc_integration(func,reg) for reg in region_tr])
    iae[model].append(int_ab_err)
    cpu[model].append(elapse_t)
    est[model].append(y)

    # Flaxman's model with Permanental Process (PP) #####################
    #####################################################################
    model = 'PP'
    pp = permanental_process(ker)
    score, par = [], []

    # Hyper-parameter Optimization: Start
    for [a,b] in set_par:
        sc = []
        
        for d_tr, d_te in zip(d_train, d_test):

            factor = (1.-p_prob)/p_prob
            _ = pp.fit(d_tr,region_tr,a/factor,b,lr=0.01,display=False)
            rate = lambda x: pp.predict(x).numpy()*factor
            tlog = sum(log(rate(d_te))) \
                - pp.predict_integral(region_tr)*factor
            sc.append(tlog)
            
        score.append(mean(sc))
        par.append([a,b])
    [opt_a, opt_b] = array(par)[argmax(array(score))]
    # Hyper-parameter Optimization: End
    
    elapse_t = pp.fit(spk,region_tr,opt_a,opt_b,display=False)
    y = pp.predict(x_ev).numpy()
    subplot(2,2,4)
    imshow(y.reshape(len(grid_x),len(grid_y)), extent=[grid_x[0],grid_x[-1],grid_y[0],grid_y[-1]],
           origin='lower', aspect='auto', cmap='jet',vmin=0, vmax=vmax)
    for s in spk:
         plot(s[0],s[1],'k.',markersize=1.5)
    for x in region_tr_out:
         fill_between(x[0], [x[1][1]]*2, [x[1][0]]*2, fc='w', hatch="//")
    
    func = lambda x: (pp.predict(x)-f_true(x))**2
    int_sq_err = mean([mc_integration(func,reg) for reg in region_tr])
    ise[model].append(int_sq_err)
    func = lambda x: abs(pp.predict(x)-f_true(x))
    int_ab_err = mean([mc_integration(func,reg) for reg in region_tr])
    iae[model].append(int_ab_err)
    cpu[model].append(elapse_t)
    est[model].append(y)
    
    for iii in range(1,5):
         subplot(2,2,iii)
         xlim(grid_x[0],grid_x[-1])
         ylim(grid_y[0],grid_y[-1])
         xticks(linspace(grid_x[0],grid_x[-1],6))
         yticks(linspace(grid_x[0],grid_x[-1],6))
         axis('square')
    
    # Save Results ######################################################
    dill.dump((ise,iae,cpu,est),open(dir_result+'perf.dill','wb'))
    savefig(dir_result+str(ii+1).zfill(3)+'.pdf')
    close('all')

    print(ise)
    
# Display Summary of Results ############################################
print('')
print('**ISE**')
q = ise
print('KIE :',mean(q['KIE']),std(q['KIE']))
print('K2IE:',mean(q['K2IE']),std(q['K2IE']))
print('PP  :',mean(q['PP']),std(q['PP']))
print('')
print('**IAE**')
q = iae
print('KIE :',mean(q['KIE']),std(q['KIE']))
print('K2IE:',mean(q['K2IE']),std(q['K2IE']))
print('PP  :',mean(q['PP']),std(q['PP']))
print('')
print('**CPU**')
q = cpu
print('KIE :',mean(q['KIE']),std(q['KIE']))
print('K2IE:',mean(q['K2IE']),std(q['K2IE']))
print('PP  :',mean(q['PP']),std(q['PP']))
