from functools import partial
from threading import Thread, Timer
import time
import sys
import socket
import optparse
import random
import numpy as np
#from watchman_nogui import Watchman_main_window
import matplotlib.pyplot as plt
import waveform_gen_33600 as wv_gen
from waveform_gen_33600 import wave_gen
import os
import targetc as targetc
import pandas as pd

"""
Script for modifyng montiming register 

"""

tc = targetc.targetc()





print ('SSPIN')
regID = 76
regValue = 48
tc.send_command(8, regID, regValue) 
time.sleep(5)

print ('ssTout')
regID = 76
regValue = 16
tc.send_command(8, regID, regValue) 
time.sleep(5)

print ('ssToutFB')
regID = 76
regValue = 32
tc.send_command(8, regID, regValue) 
time.sleep(5)

print ('WR_STRB1')
regID = 76
regValue = 64
tc.send_command(8, regID, regValue) 
time.sleep(5)

print ('WR_ADDR_INCR')
regID = 76
regValue = 80
tc.send_command(8, regID, regValue) 
time.sleep(5)

print ('WR_STRB2')
regID = 76
regValue = 96
tc.send_command(8, regID, regValue) 
time.sleep(5)

print ('WR_ADDR_INCR2')
regID = 76
regValue = 112
tc.send_command(8, regID, regValue) 
time.sleep(5)




wave_gen().Output1(out=False)



