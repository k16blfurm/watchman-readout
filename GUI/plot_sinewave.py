
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from matplotlib import rc
fileName = 'sinewave_sstoutfb_57.txt'
df = pd.read_csv ( fileName, sep=" ", header=None, skiprows=1 )

#fileName = '/home/idlab-52/elise_fork/watchman-readout/GUI/data/sinewave_sstoutfb_58.txt'
fileName = 'sinewave_sstoutfb_58.txt'
df1 = pd.read_csv ( fileName, sep=" ", header=None, skiprows=1 )

fileName = 'sinewave_sstoutfb_59.txt'
df2 = pd.read_csv ( fileName, sep=" ", header=None, skiprows=1 )

fileName = 'sinewave_sstoutfb_60.txt'
df3 = pd.read_csv ( fileName, sep=" ", header=None, skiprows=1 )

fileName = 'sinewave_sstoutfb_61.txt'
df4 = pd.read_csv ( fileName, sep=" ", header=None, skiprows=1 )

fileName = 'sinewave_sstoutfb_62.txt'
df5 = pd.read_csv ( fileName, sep=" ", header=None, skiprows=1 )

fileName = 'sinewave_sstoutfb_63.txt'
df6 = pd.read_csv ( fileName, sep=" ", header=None, skiprows=1 )


vadjn = list(range(2600,2700,5))
#vadjn = list(np.zeros(10))

#### Data parameters
rango = 1  # number of steps in delay values for the waveform generator
repeticiones = 100 # Number of waveforms for the same delay value


#df = pd.read_csv ( fileName, sep=" ", header=None, skiprows=1 )
sstoutfb = pd.read_csv ( fileName, sep=" ", header=None,nrows=1 )
print(df)


total= int(rango*repeticiones)
startWindow=0
totalWindows=12
#print(pd.DataFrame(df, columns=[0,3]))
nmbrWindows = 4
print(len(df[0]))

print(np.arange(startWindow*32,totalWindows*32,1 ))
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

maximums= pd.DataFrame()

for i in range(0,rango,1):
    ax = fig.add_subplot(1,1,1+i)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) 
    for	k in range(i,total,rango):
        std_3windows = 0
        plt.plot(list(df.index), df4[k], '-o', markersize=4) 
        


	#plt.xticks(np.arange(200,384,10))       
        plt.xlim(210, 375)
        plt.ylim(-200,200)
        
        plt.yticks(np.arange(-100,301,100))       
        plt.grid(True, linestyle='--', linewidth=1)
        
        maxPos.append( df[k].idxmax() )
        maxList.append( df[k].max())
        
     
       # std_3windows += np.std(df[k][0:96])
        #plt.title('delay = {} ns'.format(i), fontsize=18, color='b')
        for j in range(0,int(32*nmbrWindows*10),32):
            plt.axvline(j-1, color='k', linewidth=2)
    std3windowsList.append(std_3windows/repeticiones)
    textstr = 'std3windows={:10.2f}'.format(std_3windows/repeticiones)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
#print('std3list={}'.format(np.asarray(std3windowsList).fig.text(0.5, 0.04, 'Time [ns]', ha='center', fontsize=fontsizeAxis)
#print('std3list={}'.format(np.asarray(std3windowsList).fig.text(0.08, 0.5, 'ADC counts', va='center', rotation='vertical', fontsize=fontsizeAxis)
fig.text(.4, .95, 'Pedestal substracted data, same delay, 10 times', ha='center', fontsize=16)
# Option 2
# TkAgg backend
#manager = plt.get_current_fig_manager()
#manager.resize(*manager.window.maxsize())


"""
#Mean plot all (ugly way)
x = np.arange(4)
rc('font', size = 14)
plt.figure()
plt.title('Mean Frequencies Fit & STD of Varied SSTOUTFB')
plt.ylabel('Frequency [GHz]')
plt.xticks(x, ('', '59', '60', '61'))
plt.errorbar(1, 0.03004196998121797 , yerr = 0.00013175499441022215, fmt = 'o', color = 'r')
plt.errorbar(2, 0.03006969087792514, yerr = 0.00013615269995500586, fmt = 'o', color = 'r')
plt.errorbar(3, 0.030055780732232994, yerr = 0.00013903505800511964, fmt = 'o', color = 'r')

plt.figure()
rc('font', size = 14)
plt.title('Mean Frequencies Zero Crossing & STD of Varied SSTOUTFB')
plt.ylabel('Frequency [GHz]')
plt.xticks(x, ('', '59', '60', '61'))
plt.errorbar(1, 0.030335648148148143, yerr = 0.001031317239105197, fmt = 'o', color = 'b')
plt.errorbar(2, 0.03026606753812636, yerr = 0.00097363334796474490, fmt = 'o', color = 'b')
plt.errorbar(3, 0.030343477668845313, yerr = 0.0010147102588984284, fmt = 'o', color = 'b')

"""
#Sinewave fit
def sinefit(x,a,b,c,d):
    return a*np.sin(b*x + c) + d

print(df.shape)

win_len=32
s_start=255 #sample start fit
s_end=s_start+win_len*3   #sample end fit

x = np.arange(s_start,s_end,1)
nobs= len(x)
y_uncert=1.358 #uncertainty (nosie) in adc, comes from the stdev of the pedestal distribution of whole buffer, good first approximation
sig=np.zeros(nobs) + 1.385 #assuming constant uncertainty for every sample in y data
freqlistf = list()
periodlist = list()

#for i in range(0,repeticiones,1):
for i in range(0,1,1):
    popt, pocv = curve_fit(sinefit, x, df2[i][s_start:s_end].values, bounds = ((160,0,-np.inf,-np.inf),(np.inf,0.4,np.inf,np.inf)), sigma=sig, absolute_sigma=True)
    freqlistf.append(popt[1]*1000/(2*np.pi))
    periodlist.append(1/(popt[1]/(2*np.pi)))
meanf =  np.mean(freqlistf)
meanp = np.mean(periodlist)
print('mean fit freq [MHz] = ', meanf)
print('mean fit per [ns] =', meanp)
errorp = np.std(periodlist)
errorf = (np.std(freqlistf))
print('error period [ns] =',errorf)
print('error frequency [GHz] =',errorp)

chi = ( df2[i][s_start:s_end].values - sinefit(x, *popt)) / y_uncert
chi2 = (chi ** 2).sum()
dof = len(x) - len(popt)
factor = (chi2 / dof)
pcov_sigma = pocv / factor
print('chi^2 = ',chi2)
print('degres of freedom = ',dof)
print('chi^2/dof = ',factor)
print('pcov /(chi^2/dof) = ',np.sqrt(pcov_sigma[0, 0]))

plt.figure()
rc('font', size = 14)
plt.title('Frequency Histogram (sstoutfb = 63)')
plt.hist(freqlistf,10)
plt.xlabel('Frequency [MHz]')
plt.ylabel('# instances')

plt.figure()
rc('font', size = 14)
plt.title('Period Histogram (sstoutfb = 63)')
plt.hist(periodlist,10)
plt.xlabel('Period [ns]')
plt.ylabel('# instances')
plt.show()



"""

#Sinewave zero crossing
freqlist = np.zeros(shape = (1,7), dtype = float)
for j in range(100):
    freqlist0 = []
    sindata = df[j][220:340]
    zerc_xing = np.where(np.diff(np.signbit(sindata.values)))[0]
    for i in range(zero_xing.size-1):
        freqlist0.append(0.5/(sindata.index[zero_xing[i+1]]-sindata.index[zero_xing[i]]))
    freqlist0.append(np.mean(freqlist0))
    if (j == 0):
        freqlist = freqlist0
    else:
        freqlist = np.vstack((freqlist, freqlist0))
meanzx = np.mean(freqlist[:,6])
print('mean zero xing freq [GHz] =', meanzx)

plt.figure()
rc('font', size = 14)
plt.title('100 trials .03GHz pulse zero xing method (sstoutfb = 60)')
plt.hist(freqlist[:,0],bins=10, alpha = 0.5, label='t1')
plt.hist(freqlist[:,1],bins=10, alpha = 0.5, label='t2')
plt.hist(freqlist[:,2],bins=10, alpha = 0.5, label='t3')
plt.hist(freqlist[:,3],bins=10, alpha = 0.5, label='t4')
plt.hist(freqlist[:,4],bins=10, alpha = 0.5, label='t5')
plt.hist(freqlist[:,5],bins=10, alpha = 0.5, label='t6')

plt.legend()
plt.xlabel('Frequency [GHz]')
plt.ylabel('# instances')

errorzx = (np.std(freqlist))
print(errorzx)


#Fit VS zero xing mean freq
plt.figure()
plt.title('Fit VS Zero X-ing Frequencies (sstoutfb = 60)')
plt.ylabel('f [GHz]')
plt.errorbar(1, meanf, yerr = errorf, fmt = 'o', label = 'fit')
plt.errorbar(2, meanzx, yerr = errorzx, fmt = 'o', label = 'zero xing')
plt.legend(loc = 'upper left')

#Display sine data + fit

def sinefit(x,a,b,c,d):
     return a*np.sin(b*x + c) + d
x = np.arange(220,340,1)

dflist = [df,df1,df2,df3,df4,df5,df6]

rc('font', size=14)
plt.figure()
plt.title('Varied SSTOUTFB')
freqlistdf = []
r = []

n = 57
for j in dflist:
    popt, pocv = curve_fit(sinefit, x, j[0][220:340].values, bounds = ((160,0,-np.inf,-np.inf),(np.inf,0.4,np.inf,np.inf)) )
    ans = popt[0]*np.sin(popt[1]*(j[0][220:340].index)+ popt[2] ) +popt[3]
    #if n == 60:
        #plt.plot(x, ans, label = 'Fit df %.2f' %n, c = 'k', linewidth = 3.0)
    plt.plot( x, j[0][220:340].values, 'o-', label = 'SSTOUTFB %.2f' %n, alpha = 0.6)
    freqlistdf = np.append(freqlistdf, (popt[1]/(2*np.pi)))
    r = np.append(r,np.sum(np.abs(j[0][220:340].values - ans)))
    n = n+1
    
#r = np.sum(np.abs(df[0][220:340].values - ans))
#print('Error 59:', r)

print('Fits Frequencies:',freqlistdf)
print('Errors:', r)

plt.grid(True, linestyle='--', linewidth=1)
for j in range(0,int(32*nmbrWindows*10),32):
        plt.axvline(j-1, color='k', linewidth=2)
plt.xlim(210, 375)
plt.ylim(-200,200)
plt.yticks(np.arange(-200,200,100))
plt.ylabel('ADC counts')
plt.xticks(np.arange(220,340,20))
plt.xlabel('Time [ns]') 
plt.legend()

"""
#Plot all (slightly nicer way)
def sinefit (x,a,b,c,d):
    return a*np.sin(b*x+c) + d
x = np.arange(s_start, s_end, 1)
ticks1 = np.arange(8)
rc('font', size=15)
plt.figure()
plt.title('Mean Frequencies and STD of Varied SSTOUTFB')
plt.xlabel('SSTOUTFB Value')
plt.ylabel('Frequency [MHz]')
plt.xticks(ticks1, ('', '57', '58', '59', '60', '61', '62', '63'))
dflist = [df,df1,df2,df3,df4,df5,df6]
n = 57
track = 1
for j in dflist:
    freqlist1 = []
    for i in range(0,repeticiones,1):
        popt, pocv = curve_fit(sinefit, x, j[i][s_start:s_end].values, bounds = ((160,0,-np.inf,-np.inf),(np.inf,0.4,np.inf,np.inf)), sigma=sig, absolute_sigma=True )
        freqlist1 = np.append(freqlist1, popt[1]*1000/(2*np.pi))
    plt.errorbar(track, np.std(freqlist1) , yerr = 0, fmt = 'o', label = 'SSTOUTFB %.2f' %n)
    n = n+1
    track = track+1
plt.legend
plt.show() 



