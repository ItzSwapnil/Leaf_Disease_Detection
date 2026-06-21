
import sys
import time

for i in range(3):
    print("hello", i, flush=True)
    sys.stderr.write("world\r")
    sys.stderr.flush()
    time.sleep(1)
