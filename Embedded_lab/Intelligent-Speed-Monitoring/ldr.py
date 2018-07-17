import RPi.GPIO as GPIO
import time
from subprocess import call
import numpy as np

mpin=17
tpin=27
GPIO.setmode(GPIO.BCM)
cap=0.000001
adj=2.130620985
i=0
t=0
lim = 5000

dur = 1.5

f = 0

while True:
    GPIO.setup(mpin, GPIO.OUT)
    GPIO.setup(tpin, GPIO.OUT)

    GPIO.output(mpin, False)
    GPIO.output(tpin, False)

    time.sleep(0.1)
    GPIO.setup(mpin, GPIO.IN)
    time.sleep(0.1)

    GPIO.output(tpin, True)

    starttime = time.time()
    endtime = [time.time(), time.time()]

    while (GPIO.input(mpin) == GPIO.LOW):
        endtime=time.time()

    measureresistance=endtime-starttime

    res=(measureresistance/cap)*adj

    i=i+1
    t=t+res
    if i==1:
            t=t/i
            #print(res)
            if f==0 and res>lim:
                s = time.time()
                f = 1
            if f == 1 and res<lim:
                e = time.time()
                f = 0
                if e-s < 1.5:
                    call(["python", "img.py"])
            i=0
            t=0
