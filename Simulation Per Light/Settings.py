
from container import container
import os



class Settings:
    def __init__(self) -> None:
        
        from simulation import SimulationProcess,SimulationThread,SimulationAgent_AI , SimulationAgent_HAND
        self.OnlyRunWinner=False
        #self.random_seed=42
        self.random_seed=int.from_bytes(os.urandom(4), 'big')


        self.neat_config_ini='neat-config-feedforward.ini'
        self.winningPath='__winner.pkl'
        self.checkPointPath='__neat_checkpoint.pkl'

        #numero di generazioni per run
        self.neat_num_generations=300

        self.neat_overridePopSize=None       #None per disabilitarlo

        #tempo ( secondi ) massimi prima di terminare la simulazione 
        self.maxSimulationTime=360*2      #il doppio del tempo teorico della simulaizone...       
    


        #tipo di parallelizzazione per la simulazione
        #    SimulationProcess -> più performanti ( usare massimo 12 porte / processi in contemporanea )
        #    SimulationThread  -> meno performante ( ho provato fino a 60 thread )
        self.SimulationParallelClass :SimulationProcess | SimulationThread =SimulationProcess #mettere SimulationThread per usare i thread
        
        #engine da usare per effettuare la simulazione
        #   SimulationAI -> usa la network allenata 
        #   SimulationByHand -> usa una logica programmata
        self.SimulationAgent: SimulationAgent_AI | SimulationAgent_HAND = SimulationAgent_AI

        #numero di thread/processi per il training
        self.maxThread_Process=8            #60 thread | 12 process

        self.train_GUI=False        #ATTENZIONE!! a true con più processi potrebbe essere pericoloso!!
       
        self.ports=[
            8813,8814,8815,8816   ,8817,8818,8819,8820,  8821,8822,8823,8824,  8825,8826,8827,8828  , 8829,8830,8831,8832,
            8833, 8834, 8835, 8836, 8837, 8838, 8839, 8840, 8841, 8842, 8843, 8844, 8845, 8846, 8847, 8848, 8849, 8850, 8851, 8852, 
            8853, 8854, 8855, 8856, 8857, 8858, 8859, 8860, 8861, 8862, 8863, 8864, 8865, 8866, 8867, 8868, 8869, 8870, 8871, 8872, 
        ]   #60 

        assert self.maxThread_Process <= len(self.ports), "maxThread_Process deve essere inferiore al numero di porte inserite nell'array 'ports'"
        assert self.SimulationParallelClass == SimulationProcess or self.SimulationParallelClass == SimulationThread, "Classe non valida per 'SimulationParallelClass'"
        assert self.SimulationAgent == SimulationAgent_AI or self.SimulationAgent == SimulationAgent_HAND, "Classe non valida per 'SimulationAgent'"
        
        
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
            "E0_0":{                    #da sx -> dritto / gira in basso
                "strIndexes":[9,10],
                "detector":"E0_0D",
                "laneCompatibility":["E0_1","E2_0"]
            },       
            "E0_1":{                    #da sx -> gira in alto
                "strIndexes":[11],
                "detector":"E0_1D",
                "laneCompatibility":["E0_0","E2_1"]
                },        
            "-E4_0":{                   #da basso 
                "strIndexes":[6,7,8],
                "detector":"-E4_0D",
                "laneCompatibility":["-E5_0"]
                },       
            "-E5_0":{                    #da alto 
                "strIndexes":[0,1,2],
                "detector":"-E5_0D",
                "laneCompatibility":["-E4_0"]
                },      
            "E2_0":{                    #da destra -> dritto / gira in alto
                "strIndexes":[3,4],
                "detector":"E2_0D",
                "laneCompatibility":["E0_0","E2_1"]
                },        
            "E2_1":{                    #da destra -> gira in basso
                "strIndexes":[5],
                "detector":"E2_1D",
                "laneCompatibility":["E0_1","E2_0"]
                },       
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