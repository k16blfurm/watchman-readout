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
from tempfile import TemporaryFile
from plot_delays_max import plot_pulse
from waiting import wait
class targetc():

    def __init__(self):

        self.UDP_IP = '192.168.1.10'
##     Contain the port number for UDP communication
        self.UDP_PORT = 7
        self.UDP_PORT_data = 8
        self.UDP_PORT_trigger= 9

        self.init_UDP_connection_cmd()
        #self.init_UDP_connection_trigger_mode()
        self.flag_data = []
##     Contain the zynq's ip
##     List of all the commands
##     Flag which indicates if the streaming is running
        self.flag_transfer_done = False
        self.cmd = ['write_all_reg', 'read_all_reg', 'ping', 'trigger_mode', 'stop_uC', 'settime', 'recover_data', 'get_windows','write_register','pedestal', 'get_windows_raw', 'restartAll']
        self.stream_flag = False
        ## Flag which indicates that the user want to close the GUI (to avoid problem when accessing graphical object after "WM_DELETE_WINDOW" event)
        self.destroy_flag = False 
        ## Flag which indicates if the graphical window is open or not
        self.toplevel_flag = False
        ## Flag which indicates to the main thread if it needs to stop
        #self.run_flag = False
        self.timer_thread_flag = False
        self.recover_data_flag = False
        self.timer_thread_flag_2 = False
        self.get_windows_flag = False
        self.thread_cmd=Thread(target=self.thread_cmd_int, args=())
        self.run_flag = True
        self.thread_cmd.start()

       ## Socket object used to established the UDP connection with the zynq
    def init_UDP_connection_data(self):
   # global sock_dat
        self.sock_data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_data.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2097152) # change the size of the socket buffer
        self.sock_data.bind(('', self.UDP_PORT_data))
        self.sock_data.settimeout(0.1) # method sock.recvfrom return after maximum 0.1sec if no data are received
       # return sock_data

        ## Socket object used to established the UDP connection with the zynq
    
    def close_UDP_connection_data(self):
            #self.sock_data.shutdown(socket.SHUT_RDWR)
        self.sock_data.close()

    def init_UDP_connection_cmd(self):
        #global sock
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', self.UDP_PORT))
        print('socket initialized, Port:',self.UDP_PORT) 
        self.sock.settimeout(0.1) # method sock.recvfrom return after maximum 0.1sec if no data are received
            #return sock
    
            ## Socket object used to established the UDP connection with the zynq
    
    def close_UDP_connection_cmd(self):
        #self.sock_data.shutdown(socket.SHUT_RDWR)
        self.sock.close()
   
    def init_UDP_connection_trigger_mode(self):
       # global sock_data
        self.sock_trigger = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_trigger.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2097152) # change the size of the socket buffer
        self.sock_trigger.bind(('', self.UDP_PORT_trigger))
        self.sock_trigger.settimeout(0.1) # method sock.recvfrom return after maximum 0.1sec if no data are received
       # return sock_data
    
        ## Socket object used to established the UDP connection with the zynq
    
    def close_UDP_connection_trigger_mode(self):
        #self.sock_data.shutdown(socket.SHUT_RDWR)
        self.sock_trigger.close()
    
    def send_command(self,comando,param1,param2):
       print("received command {}".format(self.cmd[comando]))
       # Build the frame
       payload = bytearray()
       payload.append(int("0x55", 0)) # frame's start code 0x55AA
       payload.append(int("0xAA", 0))
       payload.append(comando)  # then then command ID
       payload.append(random.randrange(0,255)) # then a random number to put an "id" on every frame
    
    #   if(cmd[comando] == 'write_all_reg'): # if the command is write register, add the register's value
    #       for reg in regs:
    #           numb = int(reg.get())
    #           payload.append(int(numb / 256))
    #           payload.append(int(numb % 256))
    #         #  print(int(numb / 256))
    #         #  print(int(numb % 256))
       if(self.cmd[comando] == 'restartAll'): # restart main()
           dummy = param1 
       if(self.cmd[comando] == 'trigger_mode'): # restart main() 
          #self.init_UDP_connection_trigger_mode()
          payload.append(int("0x33", 0)) # frame's end code 0x33CC
          payload.append(int("0xCC", 0))
          self.init_UDP_connection_data()
        
          self.thread_trigger_obj=Thread(target=self.thread_trigger, args=())
        #  #thread_timer_2=Timer(10,thread_timer_int_2)
          self.thread_trigger_obj.start()
          self.get_windows_flag = True
        #  self.thread_user_mode_obj=Thread(target=self.thread_user_mode, args=())
          #thread_timer_2=Timer(10,thread_timer_int_2)
         # self.thread_user_mode_obj.start()
         # self.get_windows_flag = True

          payload.append(int("0x33", 0)) # frame's end code 0x33CC
          payload.append(int("0xCC", 0))
          print("stream command sent")
       if(self.cmd[comando] == 'write_register'): # if the command is write register, add the register's value
                                                
           payload.append(param1) # regID
           payload.append(int(param2 / 256)) # regValue
           payload.append(int(param2 % 256))
           payload.append(int("0x33", 0)) # frame's end code 0x33CC
           payload.append(int("0xCC", 0))
       if(self.cmd[comando] == 'pedestal'): # if the command is write register, add the register's value
                                                
           payload.append(param1) # pedestal Voltage not tested yet 06/19/2019
           payload.append(param2) # number of windows
           payload.append(int("0x33", 0)) # frame's end code 0x33CC
           payload.append(int("0xCC", 0))
   
      
       if(self.cmd[comando] == 'get_windows' or self.cmd[comando] == 'get_windows_raw'):
   #       payload.append(param1)
   #       payload.append(int(param2 / 256)) 
   #       payload.append(int(param2 % 256))
          payload.append(int("0x33", 0)) # frame's end code 0x33CC
          payload.append(int("0xCC", 0))
          self.init_UDP_connection_data()
          self.thread_user_mode_obj=Thread(target=self.thread_user_mode, args=())
          #thread_timer_2=Timer(10,thread_timer_int_2)
          self.thread_user_mode_obj.start()
          self.get_windows_flag = True
      # print("Tx: " + cmd[comando] + " rand=" + str(payload[3])) 
       #payload.append(int("0x33", 0)) # frame's end code 0x33CC
       #payload.append(int("0xCC", 0))
       self.sock.sendto(payload, (self.UDP_IP, self.UDP_PORT)) 
      # for i in range(0,len(payload)):
      #     print(hex(payload[i]))
 
    def get_sorted_data(self,binData):
        data = [item for sublist in binData for item in sublist] 
        self.windowSize = int(len(data[0])/2)
               
        self.int_array =  np.zeros((self.totalWindows-self.startWindow, self.windowSize  ))

        for i in range(len(data)):
            self.int_array[i] = np.fromstring(data[i],dtype=np.uint16)
        
        #print('lEN', len(self.int_array))
        print("all channels, window 0",self.int_array)
        self.Vped=0
        self.windowsNumbers = [self.int_array[x][1] for x in range(0,len(self.int_array)) ] # create a list with the window numbers, byte 1 from each window
        print('windowsNumbers',self.windowsNumbers) 
        self.numberofWindows = len(self.windowsNumbers)
        pedestal_avg = 0.
        offset_avoid_negative=0.
        #payload = [self.int_array[x:x + 512] for x in range( 2,len(self.int_array), int(self.windowSize/2) ) ] # get the data from each window asumming a self.windowSize, payload[window][data]
 
        payload = [self.int_array[i][2:514]  for i in range(0,len(self.int_array))]
        #print("payload len",len(payload_integers))
        #payload = [ [float('{}.{}'.format(int(payload_integers[i][j*2]), int(payload_integers[i][j*2+1]))) for j in range(int(len(payload_integers[i])/2))] for i in range(len(payload_integers)) ]
        # Subtracting offset for UINT32_T in PS
        payload = [ [ (payload[i][j] ) for j in range(len(payload[i]))] for i in range(len(payload)) ]
        windows_and_channels = [ [ payload[i][ x:x + 32] for x in range(0,len(payload[i]),32) ] for i in range(self.numberofWindows)] # create a nested list from the payload, windows_and_channels[window][channel][sample]
        self.data_by_channel = list()
        for i in range(len(windows_and_channels[0])): 
            self.data_by_channel.append( self.same_channel(i,self.numberofWindows,windows_and_channels).tolist() ) 
        self.windowsData = np.array(self.data_by_channel).T-self.Vped
        return self.windowsData
    
    def same_channel(self, channel, numberWindows,list_to_flat):
        """
         Get the same channel from different windows
        """
        sameChannel=np.array([])
        for i in range(numberWindows):
            sameChannel = np.concatenate((sameChannel,list_to_flat[i][channel] ),axis = None  )
        return sameChannel


    def thread_user_mode(self):  
        print("starting thread")
        flag_tmp = True
        self.timer_thread_flag_2 = False
        maxWindows = self.stepWindows
        cntWindows = 0
        windowsList= list()
        #window_array = np.zeros((32*16))
        while(cntWindows < maxWindows):
            try:
                data, adress = self.sock_data.recvfrom(1031) # wait on data
                # process the data received
                if(adress[0] == self.UDP_IP): # test the emitter's ip
                    if((data[0] == int("0x55", 0)) and (data[1] == int("0xAA", 0))): # for every command look for start code
                        # if((data[2052] == int("0x33", 0)) and (data[2053] == int("0xCC", 0))):
                         if((data[1028] == int("0x33", 0)) and (data[1029] == int("0xCC", 0))):
                            #self.data=data
                            windowsList.append(data)
                           # print('DATA',data)                           
                            cntWindows += 1
                            print(cntWindows)
                         else:
                            # error: no end code
                            print(data[0],data[1020:1030], len(data))
                            print("Rx: ERROR end of data")
                            cntWindows = maxWindows
                            flag_tmp = False
                    else:
                        # error: no start code
                        print("Rx: ERROR start of data")
                        cntWindows = maxWindows
                        flag_tmp = False
                else:
                    # error: wrong emitter's ip
                    print("Rx: ERROR ip of data")
                    cntWindows = maxWindows
                    flag_tmp = False
            # socket exception: no data for received before timeout
            except socket.timeout:
                time.sleep(0.1)
            # socket exception: problem during execution of socket.recvfrom
            except socket.error:
                dummy = 0 # dummy execution to catch the exception
        
#        self.get_sorted_data(np.array(windowsList))
        self.allWindows.append(windowsList)
       # print("windowsList", windowsList)
        
       # print("windowsList", windowsList[1][0])
        windowsList = windowsList*0 
       # windowsData_int = np.zeros((32*4))
        self.close_UDP_connection_data()
        
    def thread_trigger(self):  
        print("starting thread")
        flag_tmp = True
        self.timer_thread_flag_2 = False
        maxWindows = self.stepWindows
        cntWindows = 0
        windowsList= list()
        #window_array = np.zeros((32*16))
        while(cntWindows < maxWindows):
            try:
                data, adress = self.sock_data.recvfrom(1031) # wait on data
                # process the data received
                if(adress[0] == self.UDP_IP): # test the emitter's ip
                    if((data[0] == int("0x55", 0)) and (data[1] == int("0xAA", 0))): # for every command look for start code
                        # if((data[2052] == int("0x33", 0)) and (data[2053] == int("0xCC", 0))):
                         if((data[1028] == int("0x33", 0)) and (data[1029] == int("0xCC", 0))):
                            #self.data=data
                            windowsList.append(data)
                           # print('DATA',data)                           
                            cntWindows += 1
                            print(cntWindows)
                         else:
                            # error: no end code
                            print(data[0],data[1020:1030], len(data))
                            print("Rx: ERROR end of data")
                            cntWindows = maxWindows
                            flag_tmp = False
                    else:
                        # error: no start code
                        print("Rx: ERROR start of data")
                        cntWindows = maxWindows
                        flag_tmp = False
                else:
                    # error: wrong emitter's ip
                    print("Rx: ERROR ip of data")
                    cntWindows = maxWindows
                    flag_tmp = False
            # socket exception: no data for received before timeout
            except socket.timeout:
                time.sleep(0.1)
            # socket exception: problem during execution of socket.recvfrom
            except socket.error:
                dummy = 0 # dummy execution to catch the exception
            print("trying")
#        self.get_sorted_data(np.array(windowsList))
        self.allWindows.append(windowsList)
       # print("windowsList", windowsList)
        
       # print("windowsList", windowsList[1][0])
        windowsList = windowsList*0 
        self.flag_transfer_done= True
       # windowsData_int = np.zeros((32*4))
       # self.close_UDP_connection_data()
        self.WindowsData_toSave= np.zeros((32*self.stepWindows))
        #print(self.WindowsData_toSave)
        #print(self.allWindows)
        self.WindowsData_toSave = self.get_sorted_data(self.allWindows)[:,self.channel]       
        WindowsData_toSave_flat = self.WindowsData_toSave.flatten() 
        WindowsData_txt = [1]+ WindowsData_toSave_flat.tolist() 
        np.savetxt(os.path.abspath(self.fileToSave), np.array(WindowsData_txt).T, fmt='%5.3f')
        print("dataSaved") 
        self.flag_transfer_done=False
        plot_pulse(self.fileToSave)
        

    ## Method thread to process the command received by UDP (running all the time)
    # @param : The object pointer
    def thread_cmd_int(self):
        while self.run_flag: # running flag
            try:
                data = bytearray()
                data, adress = self.sock.recvfrom(2*128+20) # wait on data
                # process the data received (echo command)
                if(adress[0] == self.UDP_IP): # test the emitter's ip
                    if((data[0] == int("0x55", 0)) and (data[1] == int("0xAA", 0))): # for every command look for start code
                        offset = 0
                        # stop/start command
                        if(self.cmd[data[2]] == 'start_stop_stream'):  
                            if(self.stream_flag): # stop streaming
                                if(self.destroy_flag == False):
                                    if(self.toplevel_flag): # if the 2nd window is open, print number of data received and lost
                                        print("total of frame received =" + str(window_data.count))
                                        print("LostCnt:"+str(window_data.lostcnt))
                                self.stream_flag = False
                            else: # start streaming
                                if(self.destroy_flag == False):
                                    if(self.toplevel_flag): # if the 2nd window is open, reset number of data received and lost
                                        window_data.count = 0
                                        window_data.lostcnt = 0
                                self.stream_flag = True
                        # stop uC command
                        if(self.cmd[data[2]] == 'stop_uC'):
                            self.stream_flag = False
                        # read all registers command
                        if(self.cmd[data[2]] == 'read_all_reg'): # adapt index to find the frame's end code
                            offset = 128*2
                        # for every command look for the end code
                        if((data[4+offset] == int("0x33", 0)) and (data[5+offset] == int("0xCC", 0))):
       #                     if(destroy_flag == False):
       #                         print("Rx: " + cmd[data[2]] + " rand=" + str(data[3]))
                                if(offset == 128*2): # in case of read all register command
                                    count = 4
                                    for reg in regs: # update the value registers value
                                        reg.delete(0,END)
                                        reg.insert(END, str(data[count]*256 + data[count+1]))
                                        count += 2
                        else:
                            # error: no end code
                            if(self.destroy_flag == False): 
                                print("Rx: ERROR end of frame")
                    else:
                        # error: no start code
                        if(self.destroy_flag == False):
                            print("Rx: ERROR start of frame")
                else:
                    # error: wrong emitter's ip
                    if(self.destroy_flag == False):
                        print("Rx: ERROR ip of frame (" + adress[0] + ")")
            # socket exception: no data for received before timeout
            except socket.timeout:
                time.sleep(0.1)
            # socket exception: problem during execution of socket.recvfrom
            except socket.error:
                dummy = 0 # dummy execution to catch the exception
            #time.sleep(0.5)
       # print("end of command thread", file=sys.stderr)
       # return         
    
    
    
    def get_512_windows(self, startWindow,totalWindows,stepWindows, channel):
        self.allWindows=list()
        self.stepWindows = stepWindows 
        self.startWindow = startWindow
        self.totalWindows= totalWindows
        regID = 151    
        nmbrIterations= len(range(startWindow,totalWindows+ stepWindows,stepWindows))
        WindowsData_toSave= np.zeros(( nmbrIterations, 32*self.stepWindows))
        #count=0
        regValue=0
        for j in range(startWindow,totalWindows,self.stepWindows):#change totalWindows to 511 for the whole ASIC buffer # change t0 28 for 25 windowsi # last test 12
            regValue= j
          #  print('START WINDOW', regValue)         
#   count = count
            self.send_command(8,regID,regValue) # change the start window
            time.sleep(0.5)
            self.send_command(7,self.stepWindows,0) # get windows
            time.sleep(0.5) #.5
           

        #WindowsData_toSave[count]= self.windowsData[:,2]
        #self.windowsData= self.windowsData*0
        #count+=1
        WindowsData_toSave = self.get_sorted_data(self.allWindows)[:,channel]       
   
#     self.windowsData= self.windowsData*0
        return WindowsData_toSave.flatten() 
    
    #############################################
    
    def trigger_mode(self, stepWindows,channel, fileToSave):
        self.stepWindows = stepWindows 
        self.allWindows=list()        
        self.totalWindows=stepWindows
        self.startWindow=0
        self.send_command(3,0,0)# send command to PS to start trigger mode
        self.channel=channel  
        self.fileToSave=fileToSave
        
        return 
