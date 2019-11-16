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

tc = targetc.targetc()
#wave_gen().Output1(out=False)

rango = list((range(10,11,1)))  # number of steps in delay values for the waveform generator

repeticiones = list( range(0,1,1)   )
fileToSave = './data/sinewave_sstoutfb_59.txt'

#print ('Setting ssToutFB')
#regID = 65
#regValue = 59
#time.sleep(2)

nmbrWindows = 4
tc.send_command(9,50,nmbrWindows) # pedestal
time.sleep(3)

time.sleep(1)
regID = 151
regValue = 0
tc.send_command(8,regID,regValue) # first window
time.sleep(1)

regID = 152

regValue = nmbrWindows
tc.send_command(8, regID, regValue) # nmbrWindows
time.sleep(1)



startWindow=0
totalWindows=12

Windows512 = np.zeros((totalWindows*31))

Windows512_delays= list()


WindowsSum = np.zeros((totalWindows*31))



#wave_gen().Output1(out=True)


for j in repeticiones: # # Number of waveforms for the same delay value

    print(j)
   

    for i in rango:
    
 #      wave_gen().trigDelay(i*.000000001)
       time.sleep(0.5)
       Windows512 = tc.get_512_windows(startWindow,totalWindows,nmbrWindows,2)       
       Windows512 = [int(i)] + Windows512.tolist()
       Windows512_delays.append(Windows512)
       time.sleep(0.5)

np.savetxt(os.path.abspath(fileToSave), np.array(Windows512_delays).T, fmt='%5.3f')
time.sleep(1)

Windows512_delays = Windows512_delays * 0 
    

#wave_gen().Output1(out=False)


