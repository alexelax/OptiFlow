import os
import subprocess
import sys
import threading
from Settings import Settings
from simulationProcess import simulationProcess
import traci
import time
from PortManager import *




def main():

    settings = Settings()


    portManager = PortManager(threading.Lock())
    portManager.addPort(settings.ports)


    processes=[]

    #sharedList=multiprocessing.Manager().list()
    #sharedList.append(portManager)
    # Create a separate process for each simulation
    for i in range(0,2):
        processes.append(simulationProcess(portManager,i,None,settings))
    

    # Start both processes in parallel
    for p in processes:
        p.start()


    while True:
        finish=True
        time.sleep(1)
        for p in processes:
            if p.is_alive():
                finish=False
        if finish:
            break

    # Wait for both processes to finish
    #process1.join()
    #process2.join()

    for p in processes:
        print(p.retVal)

   

if __name__ == "__main__":
    main()
