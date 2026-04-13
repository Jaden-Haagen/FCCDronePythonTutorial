from pysimverse import Drone
import time


drone = Drone()
drone.connect()
drone.take_off()




drone.land()
time.sleep(1)