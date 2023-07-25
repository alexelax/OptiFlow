import random
import subprocess
from PortManager import PortManager
import time
from Settings import Settings
import traci
from multiprocessing import Process, Manager,Value
from threading import Thread
import os 
import neat
import numpy as np
from util import util
import pandas as pd


class SimulationParallel():
    def __init__(
            self, id:int,
            network:neat.nn.FeedForwardNetwork,
            settings,
            port=None,
            GUI:bool=False,
            name=None, args=(), kwargs={}, daemon=None
            ):
        super().__init__( group=None, name=name, args=args, kwargs=kwargs, daemon=daemon )
        # `args` and `kwargs` are stored as `self._args` and `self._kwargs`
        
        assert len(args) == 2, "passare 2 parametri ad args: un manager.dict() ( usato per il ritorno dei valori) e Value('f', 0.0) ( usata per aggiornare il tempo della simulazione ) "
        
        self.id = id
        self.network = network
        self.settings=settings
        self.retVal=args[0]
        if self.retVal == None:
            self.retVal= Manager().dict()        

        self.GUI=GUI
        self.port=port
        self.secondsFromStart=args[1]
        if self.secondsFromStart == None:
            self.secondsFromStart = Value('f',0.0)

    def setSecondFromStart(self,sec):
        self.secondsFromStart.value=sec

    def getSecondFromStart(self):
        return self.secondsFromStart.value

    
    def runAndWait(self):
        self.start()
        self.join()
        return (self.retVal["total_simulation_time"],self.retVal["max_time_loss"],self.retVal["avg"])


class SimulationThread(SimulationParallel,Thread):
    def __init__(self, id:int,
            network:neat.nn.FeedForwardNetwork,
            settings:Settings,
            port=None,
            GUI:bool=False,
            name=None, args=()):  
        super().__init__(id,network,settings,port=port,GUI=GUI,name=name,args=args)
        self.settings : Settings
        

    def run(self):
        ret  = self.settings.SimulationEngine.simulate(self)

        self.retVal["total_simulation_time"]=ret[0]
        self.retVal["max_time_loss"]=ret[1]
        self.retVal["avg"]=ret[2]



class SimulationProcess(SimulationParallel,Process):
    def __init__(self, id:int,
            network:neat.nn.FeedForwardNetwork,
            settings:Settings,
            port=None,
            GUI:bool=False,
            name=None, args=()):  
        super().__init__(id,network,settings,port=port,GUI=GUI,name=name,args=args)
        self.settings : Settings
        

    def run(self):
        ret  = self.settings.SimulationEngine.simulate(self)

        self.retVal["total_simulation_time"]=ret[0]
        self.retVal["max_time_loss"]=ret[1]
        self.retVal["avg"]=ret[2]




class SimulationAgent_AI:
    def __init__(self,setting: Settings) -> None:
        self.setting=setting
        
    def run(data):
        #ritorna un array di priorità per ogni lane
        pass

class SimulationAgent_HAND:
    def __init__(self,setting: Settings) -> None:
        self.setting=setting
        
    def run(data):
        pass 


class Simulation: 
    def simulate(callerClass:SimulationProcess | SimulationThread, simulationAgent: SimulationAgent_AI | SimulationAgent_HAND):
        self=callerClass

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
            "E0_0":{"redTime":0, "lastState":"r","index":9,"numVehicle":0},        #da sx -> dritto / gira in basso
            "E0_1":{"redTime":0, "lastState":"r","index":11,"numVehicle":0},        #da sx -> gira in alto
            "-E4_0":{"redTime":0, "lastState":"r","index":6,"numVehicle":0},       #da basso 
            "-E5_0":{"redTime":0, "lastState":"r","index":0,"numVehicle":0},       #da alto 
            "E2_0":{"redTime":0, "lastState":"r","index":3,"numVehicle":0},        #da destra -> dritto / gira in alto
            "E2_1":{"redTime":0, "lastState":"r","index":5,"numVehicle":0},        #da destra -> gira in basso
        }



        maxRedTime=60   #secondi massimi per il semaforo rosso 

        while sumo.simulation.getMinExpectedNumber() > 0:
            sumo.simulationStep()

            #controllo se ci mette troppo -> termino
            self.setSecondFromStart(self.getSecondFromStart()+settings.step_length)
            if self.getSecondFromStart()>settings.maxSimulationTime:
                break      
            
            #vedo lo stato attuale e modifico i tempi di rosso dei vari semafori
            signal_state = sumo.trafficlight.getRedYellowGreenState(settings.traffic.trafficLightID)            
            for lane in laneState:
                current = signal_state[ laneState[lane]["index"]]
                if laneState[lane]["lastState"] == 'r': #se prima era rosso
                    if current == 'r' :  #e adesso è rosso 
                        laneState[lane]["redTime"]+=settings.step_length    #aumento il tempo
                    elif  current == 'G' :      #se è verde
                        laneState[lane]["redTime"]=0    #lo resetto

                laneState[lane]["lastState"]=current
            

            #recupero il numero di veicoli per lane 
            num_vehicles_per_lane = {}
            for detector in settings.traffic.laneDetectors:
                lane=settings.traffic.laneDetectors[detector]["lane"]
                num_vehicles_per_lane[lane] = sumo.lanearea.getLastStepVehicleNumber(detector)
                

            for lane in laneState:
                laneState[lane]["numVehicle"]=num_vehicles_per_lane[lane]

            #aggiustamento per un sensore
            sensoreDestroAgg=num_vehicles_per_lane["-E1_0"]
            if sensoreDestroAgg>0:
                sensoreDestroAgg+=1
            laneState["E2_0"]["numVehicle"] += sensoreDestroAgg/2    #immagino che metà vanno dritti 
            laneState["E2_1"]["numVehicle"] += sensoreDestroAgg/2    #e l'altra metà giri


            #TODO: ottimizzabile se ristrutturo i 2 pezzi di codice precendete
            #se ci sono 0 auto su una lane, porto a 0 il tempo del rosso ( non c'è nessuna auto in attesa!! )
            for lane in laneState:
                if laneState[lane]["numVehicle"]==0:        
                    laneState[lane]["redTime"]=0




            #Phase corrente del semaforo
            currentPhase=sumo.trafficlight.getPhase(settings.traffic.trafficLightID)    

            #aggiorno i tempi persi per ogni veicolo
            for vehicle_id in sumo.vehicle.getIDList():
                time_loss=sumo.vehicle.getTimeLoss(vehicle_id)
                vehicles[vehicle_id]=time_loss

            #-------------------------------------
            #--------------- LOGICA ---------------
            #-------------------------------------
            #se sono in "rosso" e non c'è nessuno stato da attivare 
                # ALGORITMO

            #sono in "rosso" (  settings.traffic.stateIndex_Red ) e c'è uno stato da attivare 
                #si -> imposto lo stato che c'è in "pending"
                    
                
            #se sono in "giallo" ( ovvero in un indice successivo a tutti quelli prensenti dentro settings.traffic.trafficLightStates / quindi NON presente nella lista)
                #attendo -> in automatico dovrebbe andare a rosso e di conseguenza andare nel punto 1/2
            
            #se sono in "verde"
                # se sono passati MENO di settings.traffic.minTimeBeforeChange
                    #attendo -> per "sicurezza" non posso cambiare troppo spesso il semaforo quindi devo aspettare
                # se sono passati PIU di settings.traffic.minTimeBeforeChange
                    # ALGORITMO

            
            #se ho lavorato con la ALGORITMO
                #se la ALGORITMO da una action che porta ad uno stato UGUALE a quello attuale 
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
                
                #recupero il numero di veicoli per lane e li normalizzo
                input_data_numVehicle=[ laneState[lane]["numVehicle"] for lane in laneState ]
                input_data_numVehicle = SimulationUtil.normalize(input_data_numVehicle,min_val=0)
                i=0
                for lane in laneState:
                    laneState[lane]["numVehicle_norm"]=input_data_numVehicle[i]
                    i+=1
                
                #prendo il tempo del rosso e lo normalizzo
                input_data_redTime=[ laneState[lane]["redTime"] for lane in laneState ]
                input_data_redTime = SimulationUtil.normalize(input_data_redTime,0,settings.maxSimulationTime)
                i=0
                for lane in laneState:
                    laneState[lane]["redTime_norm"]=input_data_numVehicle[i]
                    i+=1

                #aggiungo il bonus
                for lane in laneState:
                    if laneState[lane]["redTime"] > maxRedTime:
                        laneState[lane]["redTime_norm"]+=   (laneState[lane]["redTime"] - maxRedTime)/5
                        #bonus calcolato empiricamente
                        #ogni secondo oltre il maxRedTime va ad aggiungere un punteggio

                
                #calcolo il punteggio per lane
                for lane in laneState:
                    laneState[lane]["out"] = laneState[lane]["numVehicle_norm"]*laneState[lane]["redTime_norm"]

                

                #calcolo la priorita per Phase
                PhasePriority={}
                for Phase in settings.traffic.trafficLightPhases_withLane:
                    priority=0
                    for lane in settings.traffic.trafficLightPhases_withLane[Phase]:
                        priority+=laneState[lane]["out"] 
                    PhasePriority[Phase]=priority

                
                #TODO: setto lane per lane
                #sumo.trafficlight.setRedYellowGreenState(settings.traffic.trafficLightID,"GGGGGGGGGGGG")

                #ottengo la Phase con il punteggio maggiore
                #Phase=max(PhasePriority, key=lambda x:PhasePriority[x])


                #network.
                

                nextPhase=Phase
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








class SimulationUtil:
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
    