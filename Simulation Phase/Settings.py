
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
       
        


       

        #numero di secondi per ogni step ( sono ammessi anche valori con la virgola )
        self.step_length=1  


        self.traffic=container()

        #nome dei detector nelle "lane" ( per trovare quante auto ci sono su ogni corsia )
        self.traffic.laneDetectors={
            "E0_0D":{"lane":"E0_0"},        #da sx -> dritto / gira in basso
            "E0_1D":{"lane":"E0_1"},        #da sx -> gira in alto
            "-E4_0D":{"lane":"-E4_0"},       #da basso 
            "-E5_0D":{"lane":"-E5_0"},       #da alto 
            "E2_0D":{"lane":"E2_0"},        #da destra -> dritto / gira in alto
            "E2_1D":{"lane":"E2_1"},        #da destra -> gira in basso
            "-E1_0D":{"lane":"-E1_0"},       #da destra -> prima di unione corsie
        }


        


        #nome del semaforo
        self.traffic.trafficLightID="J2"

        #stati possibili del semaforo ( il numero successivo corrisponde al semaforo "giallo" per lo stato )
        self.traffic.trafficLightPhases=[
            1,
            3,
            5,
            7,
            9
        ]
        self.traffic.trafficLightPhases_withLane={
            1:["E0_0","E2_0"],
            3:["E0_1","E2_1"],
            5:["E0_0","E0_1"],
            7:["E2_0","E2_1"],
            9:["-E4_0","-E5_0"]
        }

        #numero di secondi in cui mantere il rosso per tutti
        self.traffic.redAllSeconds=2

        #secondi che devono passare tra un cambio di stato ed un altro
        self.traffic.minTimeBeforeChange=10

        #index per lo stato "tutto rosso"
        self.traffic.stateIndex_Red=0



         
        self.ports=[
            8813,8814,8815,8816   ,8817,8818,8819,8820,  8821,8822,8823,8824,  8825,8826,8827,8828  , 8829,8830,8831,8832,
            8833, 8834, 8835, 8836, 8837, 8838, 8839, 8840, 8841, 8842, 8843, 8844, 8845, 8846, 8847, 8848, 8849, 8850, 8851, 8852, 
            8853, 8854, 8855, 8856, 8857, 8858, 8859, 8860, 8861, 8862, 8863, 8864, 8865, 8866, 8867, 8868, 8869, 8870, 8871, 8872, 
        ]   #60 

        assert self.maxThread_Process < len(self.ports), "maxThread_Process deve essere inferiore al numero di porte inserite nell'array 'ports'"
        assert self.SimulationParallelClass == SimulationProcess or self.SimulationParallelClass == SimulationThread, "Classe non valida per 'SimulationParallelClass'"
        assert self.SimulationEngine == SimulationAI or self.SimulationEngine == SimulationByHand, "Classe non valida per 'SimulationTipe'"
        
        
        
        self.ports=self.ports[:self.maxThread_Process]

    def stepToSecond(self,steps):
        return steps*self.step_length

    def secondToStep(self,seconds):
        return seconds / self.step_length