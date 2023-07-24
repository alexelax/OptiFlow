from container import container
import os

class Settings:
    def __init__(self) -> None:

        #self.random_seed=42
        self.random_seed=int.from_bytes(os.urandom(4), 'big')


        self.neat_config_ini='neat-config-feedforward.ini'
        self.winningPath='__winner.pkl'
        self.checkPointPath='__neat_checkpoint.pkl'

        self.neat_num_generations=300
        #tempo ( secondi ) massimi prima di terminare la simulazione 
        self.maxSimulationTime=360*2      #il doppio del tempo teorico della simulaizone...       

        #ATTENZIONE! aggiungere una porta qua vuol dire anche aggiungere  un nuovo thread durante l'esecuzione
        self.ports=[
            8813,8814,8815,8816   ,8817,8818,8819,8820,  8821,8822,8823,8824,  8825,8826,8827,8828  , 8829,8830,8831,8832,
            8833, 8834, 8835, 8836, 8837, 8838, 8839, 8840, 8841, 8842, 8843, 8844, 8845, 8846, 8847, 8848, 8849, 8850, 8851, 8852, 
            8853, 8854, 8855, 8856, 8857, 8858, 8859, 8860, 8861, 8862, 8863, 8864, 8865, 8866, 8867, 8868, 8869, 8870, 8871, 8872, 
        ]   #60
        #self.ports=[8813]
        self.ports=[ 8813,8814,8815,8816,8817,8818,8819,8820,8821,8822,8823,8824]       #multiprocessing
        #self.ports=[ 8813,8814,8815]
        


       


        self.step_length=1  #numero di secondi per ogni step ( sono ammessi anche valori con la virgola )
        self.traffic=container()

        #nome dei detector nelle "lane" ( per trovare quante auto ci sono su ogni corsia )
        self.traffic.laneDetectors=[
            "E0_0D",        #da sx -> dritto / gira in basso
            "E0_1D",        #da sx -> gira in alto
            "-E4_0D",       #da basso 
            "-E5_0D",       #da alto 
            "E2_0D",        #da destra -> dritto / gira in alto
            "E2_1D",        #da destra -> gira in basso
            "-E1_0D",       #da destra -> prima di unione corsie
        ]

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


        #numero di secondi in cui mantere il rosso per tutti
        self.traffic.redAllSeconds=2

        #secondi che devono passare tra un cambio di stato ed un altro
        self.traffic.minTimeBeforeChange=10

        #index per lo stato "tutto rosso"
        self.traffic.stateIndex_Red=0



    def stepToSecond(self,steps):
        return steps*self.step_length

    def secondToStep(self,seconds):
        return seconds / self.step_length