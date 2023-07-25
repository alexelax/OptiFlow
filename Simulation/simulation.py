import random
import subprocess
from PortManager import PortManager
import time
from Settings import Settings
import traci
from multiprocessing import Process, Value
from threading import Thread
import os 
import neat
import numpy as np
from util import util
import pandas as pd




class SimulationThread(Thread):

    def __init__(self, id:int,network:neat.nn.FeedForwardNetwork,Settings,port=None, name=None, args=(), kwargs={}, daemon=None,GUI:bool=False):
        super().__init__( group=None, name=name, args=args, kwargs=kwargs, daemon=daemon )
        # `args` and `kwargs` are stored as `self._args` and `self._kwargs`
        
        assert len(args) == 2, "passare 2 parametri ad args: un manager.dict() ( usato per il ritorno dei valori) e Value('f', 0.0) ( usata per aggiornare il tempo della simulazione ) "
        
        self.id = id
        self.network = network
        self.settings=Settings
        self.retVal=args[0]
        if self.retVal == None:
            self.retVal= {}         #dizionario non gestito da un manager ma fa niente

        self.GUI=GUI
        self.port=port
        self.secondsFromStart=args[1]
        if self.secondsFromStart == None:
            self.secondsFromStart = Value('f',0.0)

    def setSecondFromStart(self,sec):
        self.secondsFromStart.value=sec

    def getSecondFromStart(self):
        return self.secondsFromStart.value


    def run(self) :
        ret  = Simulation.simulate(self)

        self.retVal["total_simulation_time"]=ret[0]
        self.retVal["max_time_loss"]=ret[1]
        self.retVal["avg"]=ret[2]


class SimulationProcess(Process):

    def __init__(self, id:int,network:neat.nn.FeedForwardNetwork,Settings,port=None, name=None, args=(), kwargs={}, daemon=None,GUI:bool=False):
        super().__init__( group=None, name=name, args=args, kwargs=kwargs, daemon=daemon )
        # `args` and `kwargs` are stored as `self._args` and `self._kwargs`
        
        assert len(args) == 2, "passare 2 parametri ad args: un manager.dict() ( usato per il ritorno dei valori) e Value('f', 0.0) ( usata per aggiornare il tempo della simulazione ) "
        
        self.id = id
        self.network = network
        self.settings=Settings
        self.retVal=args[0]
        if self.retVal == None:
            self.retVal= {}         #dizionario non gestito da un manager ma fa niente

        self.GUI=GUI
        self.port=port
        self.secondsFromStart=args[1]
        if self.secondsFromStart == None:
            self.secondsFromStart = Value('f',0.0)

    def setSecondFromStart(self,sec):
        self.secondsFromStart.value=sec

    def getSecondFromStart(self):
        return self.secondsFromStart.value


    def run(self) :

        ret  = Simulation.simulate(self)

        self.retVal["total_simulation_time"]=ret[0]
        self.retVal["max_time_loss"]=ret[1]
        self.retVal["avg"]=ret[2]



class Simulation: 

    def simulate(callerClass:SimulationProcess | SimulationThread):
        self=callerClass

        network = self.network
        settings = self.settings
        port=self.port
        GUI= self.GUI

        # Path to the SUMO binary
        sumo_binary = "sumo" 
        if(GUI):
            sumo_binary = "sumo-gui"       
      
        # Path to the SUMO configuration file (.sumocfg)
        sumo_config_file = "data/test.sumocfg"

       

        if port==None:
            port = util.getFreeTCPPort()
            

        
        sumo_cmd = [sumo_binary, "-c", sumo_config_file,"--remote-port", str(port),"--step-length",str(settings.step_length),"--no-warnings","true","--no-step-log","true"]     
        if GUI:  
            sumo_cmd.append("--start")
            sumo_cmd.append("--quit-on-end")
            
        #se si usa il parametro "waiting_time", aggiungere ai cmq questo ->         ,"--waiting-time-memory","3600"
        #serve per far si che il waiting_time non venga resettato ogni 100s


        subprocess.Popen(sumo_cmd)


        #sumo_process = traci.start(sumo_cmd)
        sumo = traci.connect(port)

        
        vehicles={}
        
        pendingPhase=None
        secondsFromAllRed=0
        secondsFromGreen=0


        laneState={
            "E0_0":{"redTime":0, "lastState":"r","index":9},        #da sx -> dritto / gira in basso
            "E0_1":{"redTime":0, "lastState":"r","index":11},        #da sx -> gira in alto
            "-E4_0":{"redTime":0, "lastState":"r","index":6},       #da basso 
            "-E5_0":{"redTime":0, "lastState":"r","index":0},       #da alto 
            "E2_0":{"redTime":0, "lastState":"r","index":3},        #da destra -> dritto / gira in alto
            "E2_1":{"redTime":0, "lastState":"r","index":5},        #da destra -> gira in basso
        }
        # Main simulation loop
        while sumo.simulation.getMinExpectedNumber() > 0:
            sumo.simulationStep()

            self.setSecondFromStart(self.getSecondFromStart()+settings.step_length)

            if self.getSecondFromStart()>settings.maxSimulationTime:
                #termina
                break
            

            signal_state = sumo.trafficlight.getRedYellowGreenState(settings.traffic.trafficLightID)
            
            for lane in laneState:
                current = signal_state[ laneState[lane]["index"]]
                if laneState[lane]["lastState"] == 'r': #se prima era rosso
                    if current == 'r' :  #e adesso è rosso 
                        laneState[lane]["redTime"]+=settings.step_length    #aumento il tempo
                    elif  current == 'G' :      #se è verde
                        laneState[lane]["redTime"]=0    #lo resetto

                laneState[lane]["lastState"]=current
            


            # Get the number of vehicles for each lane at the specified intersection
            num_vehicles_per_lane = {}
            #lanes = traci.trafficlight.getControlledLanes(intersection_id)
            for lane in settings.traffic.laneDetectors:
                num_vehicles_per_lane[lane] = sumo.lanearea.getLastStepVehicleNumber(lane)





            #light_definition=sumo.trafficlight.getAllProgramLogics(settings.traffic.trafficLightID)
            #currentPhase= light_definition[0].currentPhaseIndex
            currentPhase=sumo.trafficlight.getPhase(settings.traffic.trafficLightID)        #altro modo

            #allPhase= light_definition[0].getPhases()


            for vehicle_id in sumo.vehicle.getIDList():

                #waiting_time dovrebbe essere il tempo perso fermo
                #waiting_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                
                #time_loss, considerando il tempo ottimale per la tratta, è il tempo perso tra stare fermo e i rallentamenti ( non ha problemi di salvataggio)
                time_loss=sumo.vehicle.getTimeLoss(vehicle_id)
                vehicles[vehicle_id]=time_loss

            #-------------------------------------
            #--------------- LOGICA ---------------
            #-------------------------------------
            #se sono in "rosso" e non c'è nessuno stato da attivare 
                # NN

            #sono in "rosso" (  settings.traffic.stateIndex_Red ) e c'è uno stato da attivare 
                #controllo se sono passati settings.traffic.redAllSeconds dalla passaggio a allRed (  settings.traffic.stateIndex_Red )
                    #si -> imposto lo stato in "pending"
                    #no -> attendo 
                
            #se sono in "giallo" ( ovvero in un indice successivo a tutti quelli prensenti dentro settings.traffic.trafficLightStates / quindi NON presente nella lista)
                #attendo -> in automatico dovrebbe andare a rosso e di conseguenza andare nel punto 1/2
            
            #se sono in "verde"
                # se sono passati MENO di settings.traffic.minTimeBeforeChange
                    #attendo -> per "sicurezza" non posso cambiare troppo spesso il semaforo quindi devo aspettare
                # se sono passati PIU di settings.traffic.minTimeBeforeChange
                    # NN

            
            #se ho lavorato con la NN
                #se la NN da una action che porta ad uno stato UGUALE a quello attuale 
                    #non faccio niente
                # ...................................... stato DIVERSO da quello attuale 
                    #imposto il giallo ( stato successivo al corrente ) e imposto nel "pending" lo stato da impostare                     
                    


            toNN=False

            if currentPhase == settings.traffic.stateIndex_Red:     #allRed
                secondsFromAllRed+=settings.step_length     #passato uno step, quindi un tot di tempo
                if pendingPhase==None:
                    toNN=True
                else:       #se c'è uno stato da attivare
                    if secondsFromAllRed >= settings.traffic.redAllSeconds: #e sono passati abbastanza secondi di allRed
                        sumo.trafficlight.setPhase(settings.traffic.trafficLightID, pendingPhase)
                        pendingPhase=None
                        secondsFromAllRed=0
                        secondsFromGreen=0
                    
            elif currentPhase not in settings.traffic.trafficLightPhases:     #giallo
                pass #non serve questo if... ma lo lascio per debug


            elif currentPhase in settings.traffic.trafficLightPhases:     #verde
                secondsFromGreen+=settings.step_length                      
                if secondsFromGreen > settings.traffic.minTimeBeforeChange:   #passato un po dall'ultimo cambio          
                    toNN=True                                                 #chiedo a NN


            
           


            if toNN:
                
                #input_data = np.random.rand(len(network.input_nodes))     #TODO: sostituire con dati reali!

                sensoreDestroAgg=num_vehicles_per_lane["E2_0D"]
                if sensoreDestroAgg>0:
                    sensoreDestroAgg+=1

                input_data_numVehicle=[
                    num_vehicles_per_lane["E0_0D"],
                    num_vehicles_per_lane["E0_1D"],
                    num_vehicles_per_lane["-E4_0D"],
                    num_vehicles_per_lane["-E5_0D"],
                    num_vehicles_per_lane["E2_0D"] + sensoreDestroAgg/2,      #immagino che metà vanno dritti 
                    num_vehicles_per_lane["E2_1D"] + sensoreDestroAgg/2,      #e l'altra metà giri
                ]

                input_data_numVehicle = Simulation.normalize(input_data_numVehicle)


                input_data_redTime=[]
                for lane in laneState:
                    input_data_redTime.append(laneState[lane]["redTime"])

                #input_data_redTime = Simulation.normalize(input_data_redTime)
                input_data_redTime = Simulation.normalize(input_data_redTime,0,settings.maxSimulationTime)


                input_data = np.concatenate((input_data_numVehicle, input_data_redTime))

                #input_data =  np.random.rand(12)
                #input_data=[ i for i in input_data ]

                output = network.activate(input_data)
                action=pd.Series(output).idxmax()

                nextPhase=settings.traffic.trafficLightPhases[action]
                if nextPhase!=currentPhase:
                    pendingPhase=nextPhase   
                    prevPhase=currentPhase
                    if currentPhase != settings.traffic.stateIndex_Red:   #primo caso, quando parte che è allRed, non devo portare a Giallo
                        sumo.trafficlight.setPhase(settings.traffic.trafficLightID, currentPhase+1)













          
          
            

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
         



    def normalize(data,min_val=None,max_val=None):

        if min_val == None:
            min_val = min(data)
        if max_val == None:    
            max_val = max(data)
            
        if min_val == max_val:
            normalized_data = [.5]*len(data)
        else:
            normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
        return normalized_data
