import random
import subprocess
from PortManager import PortManager
import time
from singleton import Singleton
import traci
from threading import Thread
import os 

class simulationProcess(Thread):

    def __init__(self, portMngr:PortManager,id:int,genome,Settings, name=None, args=(), kwargs={}, daemon=None):
        super().__init__( group=None, name=name, args=args, kwargs=kwargs, daemon=daemon )
        # `args` and `kwargs` are stored as `self._args` and `self._kwargs`
        self.portMngr=portMngr
        self.id = id
        self.genome = genome
        self.Settings=Settings
        self.retVal=None
        

    def run(self) :
        # prendo la porta da usare per la connessione tcp
        port = None
        while port == None:
            time.sleep(1)
            port = self.portMngr.lockPort()

        self.retVal  = simulationProcess.simulate(self.id,   self.genome,   self.Settings,port)

        #rilascio la porta
        self.portMngr.releasePort(port)


    def simulate(id:int,   genome,  Settings,  Port=None,GUI:bool=False):

        # Path to the SUMO binary
        sumo_binary = "sumo" 
        if(GUI):
            sumo_binary = "sumo-gui"       
      
        # Path to the SUMO configuration file (.sumocfg)
        sumo_config_file = "data/test.sumocfg"

       

        if Port==None:
            Port=12345
            #TODO: trovare una porta libera!

        Singleton().print("SUMO ",id," - port: ",Port)
        sumo_cmd = [sumo_binary, "-c", sumo_config_file,"--remote-port", str(Port)]     
        if GUI:  
            sumo_cmd.append("--start")
        #se si usa il parametro "waiting_time", aggiungere ai cmq questo ->         ,"--waiting-time-memory","3600"
        #serve per far si che il waiting_time non venga resettato ogni 100s

        #,"--step-length","0.5"      -> modifica quanto dura uno step ( di default è a 1 )

        subprocess.Popen(sumo_cmd)


        #sumo_process = traci.start(sumo_cmd)
        sumo = traci.connect(Port)

        Singleton().print("SUMO ",id," connesso")
        vehicles={}
        i =0 
        # Main simulation loop
        while sumo.simulation.getMinExpectedNumber() > 0:
            sumo.simulationStep()

            # Replace "intersection_id" with the ID of the intersection you want to monitor
            intersection_id = "J2"

            # Get the number of vehicles for each lane at the specified intersection
            num_vehicles_per_lane = {}
            #lanes = traci.trafficlight.getControlledLanes(intersection_id)
            for lane in Settings.laneDetectors:
                num_vehicles_per_lane[lane] = sumo.lanearea.getLastStepVehicleNumber(lane)
                #TODO: c'è una giunzione a destra, devo trovare il modo di prendere quei veicoli??? bha





            light_definition=sumo.trafficlight.getAllProgramLogics(Settings.trafficLightID)
            currentPhase= light_definition[0].currentPhaseIndex
            #currentPhase=traci.trafficlight.getPhase("J2")        #altro modo

            allPhase= light_definition[0].getPhases()


            for vehicle_id in sumo.vehicle.getIDList():

                #waiting_time dovrebbe essere il tempo perso fermo
                #waiting_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                
                #time_loss, considerando il tempo ottimale per la tratta, è il tempo perso tra stare fermo e i rallentamenti ( non ha problemi di salvataggio)
                time_loss=sumo.vehicle.getTimeLoss(vehicle_id)
                vehicles[vehicle_id]=time_loss






            #TODO: LOGICA QUA!
            if int.from_bytes(os.urandom(1), 'big')%2:
                sumo.trafficlight.setPhase("J2", (currentPhase+1) % len(allPhase))













          
          
            

        # Get the simulation report
        total_simulation_time = sumo.simulation.getTime()
        max_time_loss = 0
        avg=0
        for vehicle_id in vehicles:
            time_loss=vehicles[vehicle_id]
            avg+=time_loss

            max_time_loss = max(max_time_loss, time_loss)

        avg/=len(vehicles)



        sumo.close()

        
        return (total_simulation_time, max_time_loss , avg)
         