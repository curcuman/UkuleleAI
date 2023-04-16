#!/usr/bin/env python3

import serial
from serial.serialutil import SerialException
import sys
import time
import os
label = "C_"
name = 0

def main(dev: str='COM13', baudrate: int=921600, timeout: int=10):
        name = 0
        f = open('data/'+label+str(name)+'.pcm', 'wb')
        print(f'Waiting on {dev}…')
        ser = None
        while ser is None:
                #if os.path.exists(dev):
                try:
                        ser = serial.Serial(dev, baudrate, timeout=timeout)
                except:
                        pass
                time.sleep(0.1)
        print("STARTING!")
        while True:
                try:
                        #r = ser.readline() # Read one line of data

                        #f.write(r[:-2]) # Write raw data, discarding \r\n
                        r = ser.read(96) # Read one line of data
                        if b'abcd' in r:
                                print("received 1 second")
                                r = r[:-32]
                                name+=1
                                f = open('data/'+label+str(name)+'.pcm', 'wb')
                        f.write(r) # Write raw data, discarding \r\n
                        


                except SerialException: # Handle serial port disconnect, try to reconnect
                        ser.close()
                        ser = None

                        print(f'Waiting on {dev}…')
                        while ser is None:
                                if os.path.exists(dev):
                                        try:
                                                ser = serial.Serial(dev, baudrate, timeout=timeout)
                                        except:
                                                pass
                                time.sleep(0.1)

if __name__ == '__main__':
	main(*sys.argv[1:])
