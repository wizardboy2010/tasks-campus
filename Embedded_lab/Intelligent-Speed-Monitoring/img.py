import os
import subprocess

os.system("fswebcam /home/pi/webcam/img.jpg")

'''cmd = ["scp", "/home/pi/webcam/img.jpg", "user@10.9.19.86:/media/user/9C2C23682C233D20/Users/user/Desktop/acads/sem6/Embedded-lab/Capstone/rasp"]

output = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0]

print(output)'''

subprocess.call(["scp", "/home/pi/webcam/img.jpg", "user@10.9.19.86:/media/user/9C2C23682C233D20/Users/user/Desktop/acads/sem6/Embedded-lab/Capstone/rasp"])

