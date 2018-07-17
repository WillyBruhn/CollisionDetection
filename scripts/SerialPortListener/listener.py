##############
## Script listens to serial port and writes contents into a file
##############
## requires pySerial to be installed 
import serial
import os
import time
import datetime


serial_port = '/dev/ttyACM0';
#baud_rate = 9600; #In arduino, Serial.begin(baud_rate)

#baud_rate = 115200;
baud_rate = 2000000;
write_to_file_path = "output.txt";

output_file = open(write_to_file_path, "w+");
ser = serial.Serial(serial_port, baud_rate)

numOfLoggedVals = 12

collisionInCorrupted = False
try:
	while True:
	    line = ser.readline();
	    line = line.decode("utf-8") #ser.readline returns a binary, convert to string
	    
	    ts = time.time()
	    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S:%f')[:-3]
	    line = line.rstrip()
	    line+= " " + st+ "\n"
	    print(line);
	    numOfVals = len(line.split())

		# there was some corrupted data, lets note the collision now then
	    if collisionInCorrupted:
	    	if numOfVals == 9:
	    		words=line.split(" ")
	    		words[8]="1"
	    		" ".join(words)
	    		collisionInCorrupted = False

	    if "ovf" not in line:
	    	if numOfVals == numOfLoggedVals:
	    		output_file.write(line);
	    	else:
	    		print("corrupted data")
	    	print("ovf detected")
	    	collisionInCorrupted = True

except KeyboardInterrupt:
    pass

output_file.close()


#os.system("/home/willy/10.Semester/Robotik/MyRobot/SerialPortListener/./evluate.R")
