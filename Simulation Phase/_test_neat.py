import neat
import numpy as np
import random
import pickle 

random_seed = 42            
random.seed(random_seed)
np.random.seed(random_seed)



# Funzione di fitness per valutare la rete neurale su un problema specifico
def eval_genome(genome, config):
    for genome_id, g in genome:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        # Genera input casuali di dimensione 10
        input_data = np.random.rand(10)
        #input_data = [ i for i in input_data]
        # Esegui la rete neurale e ottieni 4 output come probabilità
        output = net.activate(input_data)
        # Esempio: somma degli output per normalizzare in modo che abbiano una somma di 1 (probabilità)
        output_sum = sum(output)
        if output_sum==0:
            g.fitness=0
        else:
            probabilities = [x / output_sum for x in output]
            # Esempio di obiettivo: massimizzare la probabilità del primo output
            g.fitness = probabilities[0]
        
        
#per ciascun genoma, devo far eseguire la simulazione
# a fine simulazione imposto il fitness  ( g.fitness = ...) inversamente proporzionale al tempo medio di attesa ( o altri parametri... )
#       il fitness più è alto meglio è


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,'_test-config-feedforward copy.ini')
config.genome_config.seed = random_seed



# Personalizzazione della struttura della rete neurale

# Aggiungi layer nascosti (se necessario)
#config.genome_config.add_hidden_layer(5)  # Aggiunge un layer nascosto con 5 nodi

# Creazione del popolazione di reti neurali
population = neat.Population(config)

# Report sui progressi dell'addestramento
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

num_generations=10
winner=None

# Eseguire l'evoluzione per un numero di generazioni
winner = population.run(eval_genome, num_generations)
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
