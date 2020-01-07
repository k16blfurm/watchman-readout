from functools import partial
from threading import Thread, Timer
import time
import sys
import socket
import optparse
import random
#import binary2text as b2t
import numpy as np
#from watchman_nogui import Watchman_main_window
import matplotlib.pyplot as plt
import waveform_gen_33600 as wv_gen
from waveform_gen_33600 import wave_gen
import os
import targetc as targetc
import pandas as pd
from plot_delays_max import plot_pulse

tc = targetc.targetc()

    
lblsize =16
plt.rc('xtick', labelsize= lblsize)
plt.rc('ytick', labelsize=lblsize)


fileName= '/home/salvador/Desktop/trigger.txt'
df_Data = pd.read_csv ( fileName, sep=" ", names='p', skiprows=1 )

print(df_Data)
ped_512=plot_pulse('./data/raw_window_512.txt')


#ped_512=data.iloc[0:95]

pedSubtractedData= df_Data-ped_512[0:95]

df_Data['ped']= ped_512[0:95]

print(df_Data.iloc[0:10])

df_Data_window = df_Data.iloc[32:64,:]
df_Data_window['pedSub']= df_Data_window['p']-df_Data_window['ped']



print(df_Data_window)



plt.figure()

plt.plot(df_Data_window['ped'], label='Pedestal, 50 avg')
plt.plot(df_Data_window['p'], label='Raw PMT pulse')
plt.grid(True, linestyle='--', linewidth=1)
plt.legend(fontsize=25,loc= 'upper center' )
plt.ylabel('ADC counts', fontsize=lblsize)
plt.xlabel('Time [ns]', fontsize=lblsize)
plt.figure()

plt.plot(df_Data_window['pedSub'], '-ro')
plt.grid(True, linestyle='--', linewidth=1)
plt.ylabel('ADC counts', fontsize=lblsize)
plt.xlabel('Time [ns]', fontsize=lblsize)
plt.title('PMT pulse with TARGETC, wdo_number=1', fontsize=18, color='b')

#plt.plot(pedSubtractedData [31:63], '-ro')
#plt.plot(ped_512[31:63]-1280)
#plt.plot(df_Data[31:63]-1280)
plt.show()

