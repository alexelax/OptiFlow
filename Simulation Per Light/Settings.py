
from container import container
import os

class Settings:
    def __init__(self) -> None:
        
        from simulation import SimulationProcess,SimulationThread,SimulationAI,SimulationByHand
        self.OnlyRunWinner=True
        #self.random_seed=42
        self.random_seed=int.from_bytes(os.urandom(4), 'big')


        self.neat_config_ini='neat-config-feedforward.ini'
        self.winningPath='__winner.pkl'
        self.checkPointPath='__neat_checkpoint.pkl'

        #numero di generazioni per run
        self.neat_num_generations=300

        #tempo ( secondi ) massimi prima di terminare la simulazione 
        self.maxSimulationTime=360*2      #il doppio del tempo teorico della simulaizone...       


        #tipo di parallelizzazione per la simulazione
        #    SimulationProcess -> più performanti ( usare massimo 12 porte / processi in contemporanea )
        #    SimulationThread  -> meno performante ( ho provato fino a 60 thread )
        self.SimulationParallelClass :SimulationProcess | SimulationThread =SimulationThread #mettere SimulationThread per usare i thread
        
        #engine da usare per effettuare la simulazione
        #   SimulationAI -> usa la network allenata 
        #   SimulationByHand -> usa una logica programmata
        self.SimulationEngine: SimulationAI | SimulationByHand = SimulationByHand

        #numero di thread/processi per il training
        self.maxThread_Process=8
       
        self.ports=[
            8813,8814,8815,8816   ,8817,8818,8819,8820,  8821,8822,8823,8824,  8825,8826,8827,8828  , 8829,8830,8831,8832,
            8833, 8834, 8835, 8836, 8837, 8838, 8839, 8840, 8841, 8842, 8843, 8844, 8845, 8846, 8847, 8848, 8849, 8850, 8851, 8852, 
            8853, 8854, 8855, 8856, 8857, 8858, 8859, 8860, 8861, 8862, 8863, 8864, 8865, 8866, 8867, 8868, 8869, 8870, 8871, 8872, 
        ]   #60 

        assert self.maxThread_Process < len(self.ports), "maxThread_Process deve essere inferiore al numero di porte inserite nell'array 'ports'"
        assert self.SimulationParallelClass == SimulationProcess or self.SimulationParallelClass == SimulationThread, "Classe non valida per 'SimulationParallelClass'"
        assert self.SimulationEngine == SimulationAI or self.SimulationEngine == SimulationByHand, "Classe non valida per 'SimulationTipe'"
        
        
        self.ports=self.ports[:self.maxThread_Process]







        #--------------------------------------
        #--------------------------------------
        #-------------- TRAFFICO --------------
        #--------------------------------------
        #--------------------------------------

    
        #numero di secondi per ogni step ( sono ammessi anche valori con la virgola )
        self.step_length=1  


        self.traffic=container()

        #strIndexes -> index all'interno della stringa 'rrrGGrrrrGGr'
        self.traffic.lanes={
            "E0_0":{"strIndexes":[9,10],"detector":"E0_0D"},        #da sx -> dritto / gira in basso
            "E0_1":{"strIndexes":[11],"detector":"E0_1D"},        #da sx -> gira in alto
            "-E4_0":{"strIndexes":[6,7,8],"detector":"-E4_0D"},       #da basso 
            "-E5_0":{"strIndexes":[0,1,2],"detector":"-E5_0D"},       #da alto 
            "E2_0":{"strIndexes":[3,4],"detector":"E2_0"},        #da destra -> dritto / gira in alto
            "E2_1":{"strIndexes":[5],"detector":"E2_1"},        #da destra -> gira in basso
        }
        #TODO: ricordarsi nel codice di aggiungere questo
        # "-E1_0D":{"lane":"-E1_0"},       #da destra -> prima di unione corsie


 
        #nome del semaforo
        self.traffic.trafficLightID="J2"

       
        
        #tempo "massimo" di rosso per ogni lane, oltre il quale dovrebbe scattare un "bonus"
        self.traffic.maxRedTime=60

        #tempo per il giallo
        self.traffic.yellowTime=5

        #tempo minimo per il vede
        self.traffic.minGreenTime=10



    def stepToSecond(self,steps):
        return steps*self.step_length

    def secondToStep(self,seconds):
        return seconds / self.step_length