
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from matplotlib import rc
#fileName = './data/raw_window_512.txt'

def plot_pulse(fileName):
#vadjn = list(range(2600,2700,5))
#vadjn = list(np.zeros(10))

    #### Data parameters
    rango = 1  # number of steps in delay values for the waveform generator
    repeticiones =50 # Number of waveforms for the same delay value
    
    
    
    df = pd.read_csv ( fileName, sep=" ", header=None, skiprows=1 )
    sstoutfb = pd.read_csv ( fileName, sep=" ", header=None,nrows=1 )
    print(df)
    
    total= int(rango*repeticiones)
    startWindow=0
    #totalWindows=1
    #print(pd.DataFrame(df, columns=[0,3]))
    #nmbrWindows = 1
    print(len(df[0]))
    
   # print(np.arange(startWindow*32,totalWindows*32,1 ))
    #df['xx'] = np.arange(startWindow*32,totalWindows*32,1 )
    #df[0].plot(x='xx')
    fig= plt.figure(num=None, figsize=(8,6), dpi=80)
    #fig(num=None, figsize=(8,6), dpi=80)
    fig.subplots_adjust(hspace=1, wspace=0.4)
    fontsizeAxis=28
    std3windowsList = list()
    maxList = list()
    maxPos = list()
    print(df)
    
    
    lblsize =16
    plt.rc('xtick', labelsize= lblsize)
    plt.rc('ytick', labelsize=lblsize)
    
    ped_raw = df.iloc[:,0:48]
    ped= ped_raw.mean(axis=1)
    ped_subtracted = df[49]-ped
    #
    maximums= pd.DataFrame()
    
    for i in range(0,rango,1):
        ax = fig.add_subplot(1,1,1+i)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) 
        for	k in range(i,total,rango):
            std_3windows = 0
            plt.plot(list(df.index), df[k], '-o', markersize=4) 
             
        		
    
    	#plt.xticks(np.arange(200,384,10))       
         #   plt.xlim(210, 375)
         #   plt.ylim(-200,200)
            
         #   plt.yticks(np.arange(-100,301,100))       
            plt.grid(True, linestyle='--', linewidth=1)
            
            maxPos.append( df[k].idxmax() )
            maxList.append( df[k].max())
            
         
           # std_3windows += np.std(df[k][0:96])
            #plt.title('delay = {} ns'.format(i), fontsize=18, color='b')
            plt.title('Raw data for pedestals, 50 avg', fontsize=18, color='b')
            #for j in range(0,int(32*nmbrWindows*40),32):
             #   plt.axvline(j-1, color='k', linewidth=2)
        std3windowsList.append(std_3windows/repeticiones)
        textstr = 'std3windows={:10.2f}'.format(std_3windows/repeticiones)
       # plt.plot(list(df.index), ped_subtracted)
       # plt.show()
        return ped
#    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
#print('std3list={}'.format(np.asarray(std3windowsList).fig.text(0.5, 0.04, 'Time [ns]', ha='center', fontsize=fontsizeAxis)
#print('std3list={}'.format(np.asarray(std3windowsList).fig.text(0.08, 0.5, 'ADC counts', va='center', rotation='vertical', fontsize=fontsizeAxis)
#fig.text(.4, .95, 'Pedestal substracted data, same delay, 10 times', ha='center', fontsize=16)
# Option 2
# TkAgg backend
#manager = plt.get_current_fig_manager()
#manager.resize(*manager.window.maxsize())
#plot_pulse(fileName)
#Sinewave fit
"""
def sinefit(x,a,b,c,d):
    return a*np.sin(b*x + c) + d

print(df.shape)
x = np.arange(220,340,1)
freqlist = list()

for i in range(0,repeticiones,1):
    popt, pocv = curve_fit(sinefit, x, df[i][220:340].values, bounds = ((160,0,-np.inf,-np.inf),(np.inf,0.3,np.inf,np.inf)) )
    freqlist.append(popt[1]/(2*np.pi))

print(np.std(freqlist))
plt.figure()
rc('font', size = 14)
plt.title('100 trials of a .03GHz (30MHz) pulse')
plt.hist(freqlist,6)
plt.xlabel('Frequency [GHz]')
plt.ylabel('# instances')
"""


#Display sine data + fit
"""
popt, pocv = curve_fit(sinefit, x, df[0][220:340].values, bounds = ((160,0,-np.inf,-np.inf),(np.inf,0.3,np.inf,np.inf)) )

ans = popt[0]*np.sin(popt[1]*(df[1][220:340].index)+ popt[2] ) +popt[3]

rc('font', size=14)
plt.figure()
plt.title('Data VS Fit trial0')
plt.plot(x, ans, label = 'Fit')
plt.plot( x, df[0][220:340].values, 'o-', label = 'Data')
plt.yticks(np.arange(-200,200,100))
plt.ylabel('ADC counts')
plt.xticks(np.arange(220,340,20))
plt.xlabel('Time [ns]') 
plt.legend()
"""
### Histograms

# Max positions
#plt.figure()
#plt.hist(maxPos,bins=np.arange(384)-0.5)
#plt.plot(np.arange(0,rango,1), maxList, '-o')
#plt.xlabel('Delay [ns]', fontsize=16)
#plt.ylabel('Max Value [ns]', fontsize=16)
#plt.title('max2')
#plt.yticks(np.arange(0,300,10))
#plt.xticks(np.arange(210,260,2)) 

#
#
#
## Max values
#plt.figure()
#plt.hist(maxList,bins=np.arange(384)-0.5)
#plt.title('maxList')
#plt.xlabel('Max value', fontsize=16)
#plt.grid(True, linestyle='--', linewidth=1)
##plt.yticks(np.arange(0,100,5))
#plt.xticks(np.arange(260,300,2)) 
#
#
#df['std']= df.std(axis=1)
#df['mean'] = df.mean(axis=1)
#

# Mean and std for cases when rango = 1.

#plt.figure()
#plt.errorbar(list(df.index), df['mean'], yerr= df['std'], markersize=6, capsize=4, linewidth=2 )
#plt.xticks(np.arange(0,500,10))       
##plt.xlim(205, 235)
##plt.ylim(-10,300)
#
#plt.yticks(np.arange(-10,301,10))       
#plt.grid(True, linestyle='--', linewidth=1)
#plt.xlabel('Time [ns]', fontsize=28)
#plt.ylabel('ADC counts', fontsize=28)
#
#plt.figure()
#plt.plot(list(df.index), df['mean'], '-ok', linewidth=3, markersize=12)
#plt.fill_between(list(df.index), df['mean']- df['std'], df['mean']+ df['std'], color= 'yellow')
#
##plt.xlim(205, 235)
##plt.ylim(-10,300)
##plt.yticks(np.arange(-10,301,10))       
#plt.grid(True, linestyle='--', linewidth=1)
#plt.xlabel('Time [ns]', fontsize=30)
#plt.ylabel('ADC counts', fontsize=30)


#print(df[0].std())
#print(df[0].mean())

#plt.show() 


