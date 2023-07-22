import random
import threading
import neat
import numpy as np
from Settings import Settings
from simulationProcess import simulationProcess
import time
from PortManager import *
import pickle
import pandas as pd
      



def eval_genome(genome, config):

    #TODO: creo un modo per avviare e gestire thread 


    genomeToTest = [g for  g in genome]
    

    for genome_id, g in genome:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        # Genera input casuali di dimensione 10
        #input_data = np.random.rand(10)
        # Esegui la rete neurale e ottieni 4 output come probabilità
        output = net.activate(input_data)
        # Esempio: somma degli output per normalizzare in modo che abbiano una somma di 1 (probabilità)
        #output_sum = sum(output)
        #if output_sum==0:
        #    g.fitness=0
        #else:
        #    probabilities = [x / output_sum for x in output]
        #    # Esempio di obiettivo: massimizzare la probabilità del primo output
        #    g.fitness = probabilities[0]

        action=pd.Series(output).idxmax()
        
        


def main():

    settings = Settings()
    random.seed(settings.random_seed)
    np.random.seed(settings.random_seed)


    portManager = PortManager(threading.Lock())
    portManager.addPort(settings.ports)


    # Configurazione NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,'neat-config-feedforward.ini')
    config.genome_config.seed = settings.random_seed


    
    # Personalizzazione della struttura della rete neurale

    # Aggiungi layer nascosti (se necessario)
    #config.genome_config.add_hidden_layer(5)  # Aggiunge un layer nascosto con 5 nodi

    # Creazione del popolazione di reti neurali
    population = neat.Population(config)

    # Report sui progressi dell'addestramento
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    
    winner=None

    # Eseguire l'evoluzione per un numero di generazioni
    winner = population.run(eval_genome, settings.neat_num_generations)
    with open('neat_population.pkl', 'wb') as f:
        pickle.dump(population, f)

    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)



    # Visualizzare il miglior individuo (rete neurale)
    print('\nMiglior individuo:\n{!s}'.format(winner))

    # Utilizzare il vincitore per effettuare predizioni
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for i in range(1,100):
        input_data = np.random.rand(10)  # Input casuali di dimensione 10
        output = winner_net.activate(input_data)
        #print('Input:', input_data)
        print('Probabilità di output:', output)




    processes=[]

    #sharedList=multiprocessing.Manager().list()
    #sharedList.append(portManager)
    # Create a separate process for each simulation
    for i in range(0,2):
        processes.append(simulationProcess(portManager,i,None,settings))
    

    # Start both processes in parallel
    for p in processes:
        p.start()


    while True:
        finish=True
        time.sleep(1)
        for p in processes:
            if p.is_alive():
                finish=False
        if finish:
            break

    # Wait for both processes to finish
    #process1.join()
    #process2.join()

    for p in processes:
        print(p.retVal)

   

if __name__ == "__main__":
    main()
