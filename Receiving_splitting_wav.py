#!/usr/bin/env python3

import serial
from serial.serialutil import SerialException
import sys
import time
import os
import wave

label = "D/D_"
name = 0
channels = 1
sample_width = 2
sample_rate = 16000

def main(dev: str='COM13', baudrate: int=921600, timeout: int=10):
    name = 0
    wav_file = wave.open('data/'+label+str(name)+'.wav', 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(sample_width)
    wav_file.setframerate(sample_rate)
    print(f'Waiting on {dev}…')
    ser = None
    while ser is None:
        try:
            ser = serial.Serial(dev, baudrate, timeout=timeout)
        except:
            pass
        time.sleep(0.1)
    print("STARTING in 0 seconds!")
    while True:
        try:
            r = ser.read(96) # Read one line of data
            if b'abcd' in r:
                print("received ",name+1," samples")
                r = r[:-32]
                name += 1
                wav_file = wave.open('data/'+label+str(name)+'.wav', 'wb')
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
            wav_file.writeframes(r)
                
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
        except KeyboardInterrupt:
            wav_file.close()

if __name__ == '__main__':
    main(*sys.argv[1:])
