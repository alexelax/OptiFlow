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
import copy

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
        
    def __call__(self,data):
        #ritorna un array di priorità per ogni lane
        pass

class SimulationAgent_HAND:
    def __init__(self,setting: Settings) -> None:
        self.setting=setting
        
    def __call__(self, data):
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
        
        lanes=copy.deepcopy(settings.traffic.lanes)
        #non necessario ma per pulizia e "documentazione" assegno le variabili ad ogni lane
        for lane in lanes:
            lanes[lane]["currentState"]='r'
            lanes[lane]["numVehicle"]=0
            lanes[lane]["redTime"]=0
            lanes[lane]["greenTime"]=0
            lanes[lane]["yellowTime"]=0
            lanes[lane]["lockUntill"]=-1        #se è settata, il valore della lane non può essere modificata fino a che il tempo (self.getSecondFromStart()) non supera questo valore
            lanes[lane]["nextState"]=''

        nextLanesToSetGreen=[]
        while sumo.simulation.getMinExpectedNumber() > 0:
            sumo.simulationStep()

            #controllo se ci mette troppo -> termino
            self.setSecondFromStart(self.getSecondFromStart()+settings.step_length)
            if self.getSecondFromStart()>settings.maxSimulationTime:
                break      
            
            #vedo lo stato attuale e modifico i tempi di rosso dei vari semafori
            #TODO: potrei usare più cicli ma forse così è più leggibile
            signal_state = sumo.trafficlight.getRedYellowGreenState(settings.traffic.trafficLightID)            
            for lane in lanes:
                current = signal_state[ lanes[lane]["strIndexes"][0] ]      #per ogni index nel signal_state dovrei trovare lo stesso carattere/stato, controllo solo il primo
                if lanes[lane]["currentState"] == 'r': #se prima era rosso
                    if current == 'r' :                                     #e adesso è rosso 
                        lanes[lane]["redTime"]+=settings.step_length                                #aumento il tempo
                    elif  current == 'y' :                                  #e adesso è è giallo
                        lanes[lane]["redTime"]=0                                                    #lo resetto
                    elif  current == 'G' :                                  #e adesso è è verde
                        lanes[lane]["redTime"]=0                                                    #lo resetto


                elif lanes[lane]["currentState"] == 'y': #se prima era giallo
                    if current == 'r' :                                     #e adesso è rosso 
                        lanes[lane]["yellowTime"]=0                                                 #lo resetto
                    elif  current == 'y' :                                  #e adesso è è giallo
                        lanes[lane]["yellowTime"]+=settings.step_length                             #aumento il tempo
                    elif  current == 'G' :                                  #e adesso è è verde
                        lanes[lane]["yellowTime"]=0                                                 #lo resetto



                elif lanes[lane]["currentState"] == 'G': #se prima era verde
                    if current == 'r' :                                     #e adesso è rosso 
                        lanes[lane]["greenTime"]=0                                                  #lo resetto
                    elif  current == 'y' :                                  #e adesso è è giallo
                        lanes[lane]["greenTime"]=0                                                  #lo resetto
                    elif  current == 'G' :                                  #e adesso è è verde
                        lanes[lane]["greenTime"]+=settings.step_length                              #aumento il tempo

                lanes[lane]["currentState"]=current
            

            #recupero il numero di veicoli per lane 
            for lane in lanes:
                detector = lanes[lane]["detector"]
                lanes[lane]["numVehicle"] = sumo.lanearea.getLastStepVehicleNumber(detector)
                

            #aggiustamento per un sensore
            sensoreDestroAgg=sumo.lanearea.getLastStepVehicleNumber["-E1_0D"]
            if sensoreDestroAgg>0:
                sensoreDestroAgg+=1
            lanes["E2_0"]["numVehicle"] += sensoreDestroAgg/2    #immagino che metà vanno dritti 
            lanes["E2_1"]["numVehicle"] += sensoreDestroAgg/2    #e l'altra metà giri





            #TODO: ottimizzabile se ristrutturo i 2 pezzi di codice precendete
            #se ci sono 0 auto su una lane, porto a 0 il tempo del rosso ( non c'è nessuna auto in attesa!! )
            for lane in lanes:
                if lanes[lane]["numVehicle"]==0:
                    lanes[lane]["redTime"]=0



            #aggiorno i tempi persi per ogni veicolo
            for vehicle_id in sumo.vehicle.getIDList():
                time_loss=sumo.vehicle.getTimeLoss(vehicle_id)
                vehicles[vehicle_id]=time_loss


            #controllo ogni lane e verifico i lock, se i lock sono terminati li setto a -1
            for lane in lanes:
                if lanes[lane]["lockUntill"]!= -1 and self.getSecondFromStart() > lanes[lane]["lockUntill"]:
                    lanes[lane]["lockUntill"]=-1

            #controllo ogni lane e verifico il giallo, se yellowTime >= self.traffic.yellowTime
                #si -> porto lo stato a rosso e setto yellowTime = 0
                #no -> incremento yellowTime
            for lane in lanes:
                if lanes[lane]["currentState"]=='y' and lanes[lane]["yellowTime"]>=self.traffic.yellowTime:
                    lanes[lane]["nextState"]='r'
                    lanes[lane]["yellowTime"]=0

                    #TODO: non so se farlo qua o fidarmi del controllo precedente unito al set del lock quando setto il giallo
                    #lanes[lane]["lockUntill"]=self.getSecondFromStart()+2  #per almeno due secondi deve rimanere rosso??



            #-------------------------------------
            #--------------- LOGICA ---------------
            #-------------------------------------

            


            #ad ogni step lancio l'AGENT che mi dice quali sono le lane che dovrebbero essere attivate in ordine di priorità
                #per ogni lane
                    #controllo se è compatibile con quelle prima ( no -> vado alla successiva )
                    #verifico se la lane è già verde  ( si -> vado alla successiva )
                    #controllo se la lane è bloccata (si -> vado alla successiva )
                    #controllo se la lane è incompatibile con altre lane già verdi
                        #se è incompatibile con qualcuna 
                            #per tutte le lane incompatibili controllo se sono bloccate -> se anche 1 è bloccata ( e verde implicito ) -> vado alla successiva ( ha lei la priorità )
                            #se NESSUNA è bloccata, allora le setto a gialle ( dovrò aspettare che diventino rosse per poter settare la mia )
                                            #TODO: implemento un valore nelle lanes che mi dice che deve essere attivata una lane dopo che un altra diventa gialla? 
                                            #oppure mi affido al fatto che quando la lane che in questo momento è incompatibile diventa rossa, l'altra dovrebbe avere una 
                                            #priorità ancora alta?? ( al momento provo la seconda)
                        #se compatibile
                            #la setto a verde


            laneProbability = simulationAgent(lanes)
            laneProbability=[
                ["E0_0",0.9],
                ["E0_1",0.7],
                ["-E4_0",0.5],
                ["-E5_0",0.3],
                ["E2_0",0.2],
                ["E2_1",0.1],
            ]
               
            acceptedLanes=[]
            greenLanes=[lane for lane in lanes if lanes[lane]["currentState"]=='G']

            for el in laneProbability:
                lane = el[0]
                isAccepted=False

                #scorro tutte le linee accettate ( da portare a verdi ), se ce ne sono alcune NON presenti nelle compatibili della corrente, le conto
                incompatibilities= len([ lane for lane in acceptedLanes if lane not in lanes[lane]["laneCompatibility"] ])
                if incompatibilities > 0: continue  #ci sono incompatibilità -> successivo
                if lanes[lane]["currentState"]=='G':    #se è gia verde -> ok 
                    isAccepted=True
                else:
                    if lanes[lane]["lockUntill"]==-1: continue  #se è bloccata (e NON è verde ) -> successivo
                    incompatibilities= [ lane for lane in greenLanes if lane not in lanes[lane]["laneCompatibility"] ]
                    if len(incompatibilities)>0:
                        incompatibilities_Locked= [lane for lane in incompatibilities if lanes[lane]["lockUntill"]!=-1]
                        if len(incompatibilities_Locked) > 0: continue  #ci sono incompatibilità bloccate-> hanno loro la priorità -> successivo
                        for lane in incompatibilities:
                            lanes[lane]["nextState"]='y'
                    else:
                        isAccepted=True

                if isAccepted:
                    acceptedLanes.append(lane)
                    lanes[lane]["nextState"]='G'
                
            #se non ci sono modifiche, setto lo stato successivo a quello corrente
            for lane in lanes:
                if lanes[lane]["nextState"]=='':
                    lanes[lane]["nextState"]=lanes[lane]["currentState"]


            #creo la stringa per settare i semafori
            stateStrLen=max([ max(lanes[lane]["strIndexes"]) for lane in lanes ])
            stateStr=" "*stateStrLen
            for lane in lanes:
                for i in lanes[lane]["strIndexes"]:
                    stateStr[i]=lanes[lane]["nextState"]


            #setto lo stato del semaforo corrente
            sumo.trafficlight.setRedYellowGreenState(settings.traffic.trafficLightID,stateStr)

            #ottengo la Phase con il punteggio maggiore
            #Phase=max(PhasePriority, key=lambda x:PhasePriority[x])













          
          
            

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
    