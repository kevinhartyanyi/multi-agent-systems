from subprocess import Popen, PIPE
import time

process = Popen(["java", "-jar", "massim-2019-2.0/server/server-2019-2.1-jar-with-dependencies.jar",
                 "-conf", "massim-2019-2.0/server/conf/SampleConfig-Deliverable1.json"],
      stdout=PIPE, stderr=PIPE, stdin=PIPE)

time.sleep(1)
process.communicate(input=b'\n')
print(process.communicate())
