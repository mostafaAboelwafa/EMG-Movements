
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib notebook')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy.signal import butter,lfilter,filtfilt
from scipy.io import loadmat 
get_ipython().magic(u'matplotlib inline')


# In[2]:

def extract_data(src):
    data = loadmat(src)
    unused_keys = ['__globals__','__header__','__version__']
    for key,value in list(data.iteritems()):

        if key in unused_keys:
             del data[key]
        else:
             data[key] = pd.DataFrame(value)

    return data


# In[3]:

def visualization (Data,window_size,zooming_from_to=(0,5)):

        x_axis=np.linspace(0,window_size,Data.shape[0]) 
        y_axis=Data#rows value in the intended electrode number
        plt.plot(x_axis,y_axis)
        plt.xlim(zooming_from_to)
        plt.show()


# In[4]:

def filtration (data,sample_rate,cut_off,order=5,type='highpass',):
    nyq = .5 * sample_rate
    b,a=butter(order,cut_off/nyq,btype=type)
    filtered_Data= lfilter(b,a,data)
    return filtered_Data


# In[5]:

def window_rms(D,w_length):
    D2 = np.power(D,2)
    window = np.ones(w_length)/float(w_length)
    return np.sqrt(np.convolve(D2, window, 'valid'))


# In[6]:


# Error: the array is too large!!! 
'''def envelope_plot (s):
    D=np.array(s)
    upper_env_x=[0,]
    upper_env_y=[D[0],]
    lower_env_x=[0,]
    lower_env_y=[D[0],]
    
    for k in xrange(1,len(D)-1):
        if ((D[k] - D[k-1]>0) and (D[k]-D[k+1]>0)):
            upper_env_x.append(k)
            upper_env_y.append(D[k])
        if ((D[k] - D[k-1]<0) and (D[k]-D[k+1]<0)):
            lower_env_x.append(k)
            lower_env_y.append(D[k])
    
    upper_env_x.append(len(D)-1)
    upper_env_y.append(D[-1])
    lower_env_x.append(len(D)-1)
    lower_env_y.append(D[-1])
    
    upper_xy = interp1d(upper_env_x,upper_env_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    lower_xy = interp1d(lower_env_x,lower_env_y,kind = 'cubic',bounds_error = False, fill_value=0.0)
    
    q_upper = zeros(D.shape)
    q_lower = zeros(D.shape)
    for k in xrange(0,len(D)):
        q_upper[k] = upper_xy(k)
        q_lower[k] = lower_xy(k)
        
    
    return q_upper, q_lower '''


def E_try (fs,rms_D,order=4):
    nyq=0.5*fs
    d,c=butter(order,10/nyq,btype='lowpass')
    E=filtfilt(d,c,rms_D)
    return E

