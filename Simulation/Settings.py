class Settings:
    def __init__(self) -> None:

        #nome dei detector nelle "lane" ( per trovare quante auto ci sono su ogni corsia )
        self.laneDetectors=[
            "E0_0D",
            "E0_1D",
            "-E4_0D",
            "-E5_0D",
            "E2_0D",
            "E2_1D",
            "-E1_0D",
        ]

        #nome del semaforo
        self.trafficLightID="J2"

        #stati possibili del semaforo ( il numero successivo corrisponde al semaforo "giallo" per lo stato )
        self.trafficLightStates={
            1,
            3,
            5,
        }

        #numero di secondi/step per il giallo
        self.yellowSeconds=5

        #step ( secondi? ) che devono passare tra un cambio di stato ed un altro
        self.minTimeBeforeChange=10      

        #step massimi prima di terminare la simulazione
        self.maxSimulationStep=7000      

        #ATTENZIONE! aggiungere una porta qua vuol dire anche aggiungere  un nuovo thread durante l'esecuzione
        #self.ports=[8813,8814,8815,8816]
        self.ports=[8813]


        self.neat_num_generations=10


        self.random_seed=42

