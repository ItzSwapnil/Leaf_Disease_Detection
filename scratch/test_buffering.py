import subprocess
import sys

child = """
import sys
import time
for i in range(3):
    print("hello", i, flush=True)
    sys.stderr.write("world\\r")
    sys.stderr.flush()
    time.sleep(1)
"""
with open("child.py", "w") as f:
    f.write(child)

p = subprocess.Popen(
    [sys.executable, "child.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
assert p.stdout is not None
for line in p.stdout:
    print("PARENT GOT:", repr(line))
