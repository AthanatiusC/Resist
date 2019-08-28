import os
from datetime import datetime

def sys(message):
    print(" [ SYSTEM ]: {}".format(message))

def log(message):
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    print(" [{}]: {}".format(time,message))

def warn(message):
    print(" [ WARNING ]: {}".format(message))

def err(message):
    print(" [ERROR]: {}".format(message))