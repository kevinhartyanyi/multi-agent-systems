from subprocess import Popen, PIPE
import time
from reinforce_agent import *
process = Popen(["java", "-jar", "massim-2019-2.0/server/server-2019-2.1-jar-with-dependencies.jar", "--monitor", "8000",
                 "-conf", "massim-2019-2.0/server/conf/SampleConfig-Deliverable1.json"],
      stdout=PIPE, stderr=PIPE, stdin=PIPE)

print("First wait")
time.sleep(10)
print("Enter")
process.stdin.write(b'\n')
process.stdin.flush()
print("Wait")
time.sleep(16)
print("Kill")
process.kill()
print("Dead")

for i in process.communicate():
    print(i)
