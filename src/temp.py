import os
import time

print("sleeping for 3 hours")
time.sleep(3600*3)
os.system("CUDA_VISIBLE_DEVICES=2 bash train.sh")