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

class SimulationAgent_AI:
    def __init__(self,setting: Settings,args=() ) -> None:
        self.setting=setting
        if type(args) is tuple:
            self.network= args[0]
        else:
            self.network= args




    def __call__(self,data):

        greenTime =[ data[line]["greenTime"] for line in data]
        redTime =[ data[line]["redTime"] for line in data]
        autoPerLane =[ data[line]["numVehicle"] for line in data]

        greenTime = SimulationUtil.normalize(greenTime,0,self.setting.maxSimulationTime)
        redTime = SimulationUtil.normalize(redTime,0,self.setting.maxSimulationTime)
        autoPerLane = SimulationUtil.normalize(autoPerLane)

        input_data = np.concatenate((autoPerLane,greenTime, redTime))
        tinput_data = [i for i in input_data]

        output = self.network.activate(input_data)
        


        o= [ a for a in zip(data.keys(),output)]
        o.sort(key=lambda a: a[1],reverse = True)

        return o


"""
 maxGreenTime=self.params[0]
        maxRedTime=self.params[1]
        redFactor= self.params[2]
        redNorMin= self.params[3]
        redNorMax= self.params[4]
        greenNorMin= self.params[5]
        greenNorMax= self.params[6]
[60,60,0.25,0,720,-0.01,180]
"""

class SimulationAgent_HAND:
    def __init__(self,setting: Settings,args=()) -> None:
        self.setting=setting
        
    def __call__(self, data):

        maxGreenTime=60
        maxRedTime=self.setting.traffic.maxRedTime

        autoPerLane =[ data[line]["numVehicle"] for line in data]
        autoPerLane = SimulationUtil.normalize(autoPerLane)

        redTime =[ data[line]["redTime"] for line in data]
        redTimeBonus =[ 0 if data[line]["redTime"] < maxRedTime else (data[line]["redTime"]-maxRedTime)/4 for line in data]
        redTime = SimulationUtil.normalize(redTime,0,self.setting.maxSimulationTime)             #valmax normalizzazione empirico
        redTime = [ a[0]+a[1] for a in zip(redTime,redTimeBonus)]                                   


        greenTime =[ data[line]["greenTime"] for line in data]
        greenTime =[ maxGreenTime-t if t < maxGreenTime else 0 for t in greenTime]
        greenTime = SimulationUtil.normalize(greenTime,-0.01,self.setting.maxSimulationTime/4)      #val/minvalmax normalizzazione empirico
        
        
        output= [ a for a in zip(data.keys(),autoPerLane,redTime,greenTime)]
        out=[]
        for o in output:
            out.append([ o[0],(o[1]*o[2])+ (o[1]*o[3])])
 

        out.sort(key=lambda a: a[1],reverse = True)

        return out


class SimulationAgent_HAND_GA:
    def __init__(self,setting: Settings,args=()) -> None:
        self.setting=setting
        self.params=args[0]
        
    def __call__(self, data):

        

        maxGreenTime=self.params[0]
        maxRedTime=self.params[1]
        redFactor= self.params[2]
        redNorMin= self.params[3]
        redNorMax= self.params[4]
        greenNorMin= self.params[5]
        greenNorMax= self.params[6]




        autoPerLane =[ data[line]["numVehicle"] for line in data]
        autoPerLane = SimulationUtil.normalize(autoPerLane)

        redTime =[ data[line]["redTime"] for line in data]
        redTimeBonus =[ 0 if data[line]["redTime"] < maxRedTime else (data[line]["redTime"]-maxRedTime)*redFactor for line in data]
        redTime = SimulationUtil.normalize(redTime,redNorMin,redNorMax)             
        redTime = [ a[0]+a[1] for a in zip(redTime,redTimeBonus)]                                   


        greenTime =[ data[line]["greenTime"] for line in data]
        greenTime =[ maxGreenTime-t if t < maxGreenTime else 0 for t in greenTime]
        greenTime = SimulationUtil.normalize(greenTime,greenNorMin,greenNorMax)      #val/minvalmax normalizzazione empirico
        
        
        output= [ a for a in zip(data.keys(),autoPerLane,redTime,greenTime)]
        out=[]
        for o in output:
            out.append([ o[0],(o[1]*o[2])+ (o[1]*o[3])])
 

        out.sort(key=lambda a: a[1],reverse = True)

        return out



class SimulationParallel():
    def __init__(
            self, id:int,
            agent: SimulationAgent_AI | SimulationAgent_HAND ,
            settings,
            port=None,
            GUI:bool=False,
            name=None, args=(), kwargs={}, daemon=None
            ):
        super().__init__( group=None, name=name, args=args, kwargs=kwargs, daemon=daemon )
        # `args` and `kwargs` are stored as `self._args` and `self._kwargs`
        
        assert len(args) == 2, "passare 2 parametri ad args: un manager.dict() ( usato per il ritorno dei valori) e Value('f', 0.0) ( usata per aggiornare il tempo della simulazione ) "
        
        self.id = id
        self.agent = agent
        self.settings=settings
        self.retVal=args[0]
        if self.retVal == None:
            self.retVal= Manager().dict()        

        self.GUI=GUI
        self.port=port
        self.secondsFromStart=args[1]
        if self.secondsFromStart == None:
            self.secondsFromStart = Value('f',0.0)

    
    def runAndWait(self):
        self.start()
        self.join()
        return (self.retVal["total_simulation_time"],self.retVal["max_time_loss"],self.retVal["avg"],self.retVal["exitCode"])


class SimulationThread(SimulationParallel,Thread):
    def __init__(self, id:int,
            agent:SimulationAgent_AI | SimulationAgent_HAND ,
            settings:Settings,
            port=None,
            GUI:bool=False,
            name=None, args=()):  
        super().__init__(id,agent,settings,port=port,GUI=GUI,name=name,args=args)
        self.settings : Settings
        

    def run(self):
        ret  = Simulation.simulate(self.settings,self.secondsFromStart,self.agent,self.port,self.GUI)

        self.retVal["total_simulation_time"]=ret[0]
        self.retVal["max_time_loss"]=ret[1]
        self.retVal["avg"]=ret[2]
        self.retVal["exitCode"]=ret[3]

class SimulationProcess(SimulationParallel,Process):
    def __init__(self, id:int,
            agent:SimulationAgent_AI | SimulationAgent_HAND ,
            settings:Settings,
            port=None,
            GUI:bool=False,
            name=None, args=()):  
        super().__init__(id,agent,settings,port=port,GUI=GUI,name=name,args=args)
        self.settings : Settings
        

    def run(self):

        ret  = Simulation.simulate(self.settings,self.secondsFromStart,self.agent,self.port,self.GUI)

        self.retVal["total_simulation_time"]=ret[0]
        self.retVal["max_time_loss"]=ret[1]
        self.retVal["avg"]=ret[2]
        self.retVal["exitCode"]=ret[3]




class Simulation: 
    def simulate(settings: Settings, secondFromStart:Value, simulationAgent: SimulationAgent_AI | SimulationAgent_HAND,port:int=None,GUI:bool=False):
        
    
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

        #laststateStr=""
        #timeLastChange=0
        #breakForWasteTime=False

        exitCode=0
        
        while sumo.simulation.getMinExpectedNumber() > 0:
            sumo.simulationStep()

            #azzero i valori precedenti che possono dare problemi 
            for lane in lanes:
                lanes[lane]["nextState"]=''

                
            #controllo se ci mette troppo -> termino
            secondFromStart.value+=settings.step_length
            if secondFromStart.value>settings.maxSimulationTime:
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
            sensoreDestroAgg=sumo.lanearea.getLastStepVehicleNumber("-E1_0D")
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
                if lanes[lane]["lockUntill"]!= -1 and secondFromStart.value > lanes[lane]["lockUntill"]:
                    lanes[lane]["lockUntill"]=-1

            #controllo ogni lane e verifico il giallo, se yellowTime >= self.traffic.yellowTime
                #si -> porto lo stato a rosso e setto yellowTime = 0
                #no -> incremento yellowTime
            for lane in lanes:
                if lanes[lane]["currentState"]=='y' and lanes[lane]["yellowTime"]>=settings.traffic.yellowTime:
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
           
               
            acceptedLanes=[]
            greenLanes=[lane for lane in lanes if lanes[lane]["currentState"]=='G']
            yellowLanes=[lane for lane in lanes if lanes[lane]["currentState"]=='y']

            for el in laneProbability:
                lane = el[0]
                isAccepted=False

                #scorro tutte le linee accettate ( da portare a verdi ), se ce ne sono alcune NON presenti nelle compatibili della corrente, le conto
                incompatibilities= len([ l for l in acceptedLanes if l not in lanes[lane]["laneCompatibility"] ])
                if incompatibilities > 0: continue  #ci sono incompatibilità -> successivo
                elif lanes[lane]["nextState"]=='y': continue   #se qualcuno ha già impostato che il next deve essere y ( quindi era verde) ma deve spegnersi, non puo essere accettato
                elif lanes[lane]["currentState"]=='G':
                    isAccepted=True
                else:
                    if lanes[lane]["lockUntill"]!=-1: continue  #se è bloccata (e NON è verde ) -> successivo
                    incompatibilities= [ l for l in greenLanes if l not in lanes[lane]["laneCompatibility"] ]
                    incompatibilities.extend([ l for l in yellowLanes if l not in lanes[lane]["laneCompatibility"] ])

                    if len(incompatibilities)>0:
                        incompatibilities_Locked= [l for l in incompatibilities if lanes[l]["lockUntill"]!=-1]
                        if len(incompatibilities_Locked) > 0: continue  #ci sono incompatibilità bloccate-> hanno loro la priorità -> successivo
                        for lane in incompatibilities:
                            lanes[lane]["nextState"]='y'
                            lanes[lane]["lockUntill"]=secondFromStart.value+settings.traffic.yellowTime+2       #per il rosso? ( già gestito anche da un altra parte)
                    else:
                        isAccepted=True

                if isAccepted:
                    acceptedLanes.append(lane)
                    lanes[lane]["nextState"]='G'
                    if lanes[lane]["currentState"]!='G':
                        lanes[lane]["lockUntill"]=secondFromStart.value+settings.traffic.minGreenTime
                
            #se non ci sono modifiche, setto lo stato successivo a quello corrente
            for lane in lanes:
                if lanes[lane]["nextState"]=='':
                    lanes[lane]["nextState"]=lanes[lane]["currentState"]


            #creo la stringa per settare i semafori
            stateStrLen=max([ max(lanes[lane]["strIndexes"]) for lane in lanes ])+1
            
            stateStr=['']*stateStrLen
            for lane in lanes:
                for i in lanes[lane]["strIndexes"]:
                    stateStr[i]=lanes[lane]["nextState"]

            stateStr="".join(stateStr)

            #setto lo stato del semaforo corrente
            sumo.trafficlight.setRedYellowGreenState(settings.traffic.trafficLightID,stateStr)

            
            #file1 = open("myfile.txt", "a")  # append mode
            #file1.write(stateStr+"\n")
            #file1.close()

            """
            if laststateStr==stateStr:
                timeLastChange+=1
            else:
                timeLastChange=0
                laststateStr=stateStr

            if timeLastChange > 120:    #2 minuti? bha proviamo ( molto limitante )
                exitCode=-1                                            
                break         
            """
          
          
            
            
        total_simulation_time = sumo.simulation.getTime()
        #trucchetto per dirgli che ha finito la simulazione "sprecando" tempo e deve essere valutata 0
        #if breakForWasteTime:
        #    total_simulation_time=settings.maxSimulationTime


        max_time_loss = 0
        avg=0
        for vehicle_id in vehicles:
            time_loss=vehicles[vehicle_id]
            avg+=time_loss

            max_time_loss = max(max_time_loss, time_loss)

        avg/=len(vehicles)



        sumo.close()



        
        return (total_simulation_time, max_time_loss , avg,exitCode)
        #exitCode
        #-1 -> interrotto prima della fine xke non cambia nulla







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
    