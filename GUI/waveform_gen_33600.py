#!/usr/bin/env python

"""
Functions for controlling the waveform generator 33600a
based on https://github.com/samdejong86/Agilent33600
"""
import visa
from time import sleep

class wave_gen(object):
    def __init__(self):
        self.address =  "192.168.153.214"
        #load the VISA resource manager
        rm = visa.ResourceManager('@py')
        
        #connect to the device
        self.inst = rm.open_resource("TCPIP::"+self.address+"::INSTR")
        
#        print(self.inst.query("*IDN?"))
 
    def Output1(self,out=True):
        if out == True:
            cmd = "OUTPUT1 ON"
        else:
            cmd = "OUTPUT1 OFF"
            
        message="Controlling\nRemotely"
        self.inst.write("DISP:TEXT '"+message+"'")
        
        #sent commands to device
        self.inst.write(cmd)
        print(cmd)
        #
        #clear message
        self.inst.write("DISP:TEXT ''")
        #close device connection
        self.inst.close()
    
    def trigDelay(self,trig_delay):
        cmd = "TRIG1:DEL {:.9f}".format(trig_delay)
        
        message="Controlling\nRemotely"
        self.inst.write("DISP:TEXT '"+message+"'")
        
        #sent commands to device
        
        self.inst.write(cmd)
        print(cmd)
        
        #clear message
        self.inst.write("DISP:TEXT ''")
        
        #close device connection
        self.inst.close()
    
