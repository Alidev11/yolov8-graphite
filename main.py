from IPython.display import display
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.utils import LOGGER
import cv2
import re
import time
from graphyte import init, Sender
import threading
from collections import deque
import os
import psutil

sender = Sender(host='localhost', port=2003)
# Initialize the Graphite client
init('localhost', 2003)

pid = os.getpid()
process = psutil.Process(pid)
start_time = 0.00
def yolo_predict():
    model = YOLO("runs/detect/train3/weights/best_nano.pt")
    #model = YOLO("yolov8n.pt")
    results = model.predict(source="0", show=True)

# Get the process ID of the current Python program
def send_metrics():
    filename = 'logs/my_log_file.log'
    state = True
    while state:
        with open(filename) as f:
            if os.path.getsize(filename) != 0:
                last_line = deque(f,1).pop().strip()
                m = re.search(r"\d+\.\d+ms", last_line)
                if m:
                    x = re.search(r"\d+", m.group())
                    if x:
                        if "Speed:" in last_line:
                            state = False
                            exit()
                        sender.send('inference_time2', int(x.group()))
                        display(int(x.group()))

        cpu_percent = process.cpu_percent(interval=1)
        memory_info = process.memory_info().rss / 1024 / 1024
        sender.send('cpu_usage_yolo', int(cpu_percent))
        sender.send('mem_usage_yolo', int(memory_info))
        time.sleep(0.5)



# create threads
thread1 = threading.Thread(target=yolo_predict)
thread2 = threading.Thread(target=send_metrics)

# start threads
thread2.start()
thread1.start()

# wait for threads to finish
thread2.join()
thread1.join()


if cv2.waitKey(1) & 0xFF == ord('q'):
    exit()