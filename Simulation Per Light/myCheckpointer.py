"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""

import gzip
import pickle
import random
import time

from neat.population import Population
from neat.reporting import BaseReporter
from itertools import count

class myCheckpointer(BaseReporter):
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """

    def __init__(self, generation_interval=100, time_interval_seconds=300,
                 checkpointPath='neat-checkpoint.pkl',winnerPath='winner.pkl'):
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds
        self.checkpointPath = checkpointPath
        self.winnerPath = winnerPath

        self.current_generation = None
        self.last_generation_checkpoint = -1
        self.last_time_checkpoint = time.time()

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        checkpoint_due = False

        if self.time_interval_seconds is not None:
            dt = time.time() - self.last_time_checkpoint
            if dt >= self.time_interval_seconds:
                checkpoint_due = True

        if (checkpoint_due is False) and (self.generation_interval is not None):
            dg = self.current_generation - self.last_generation_checkpoint
            if dg >= self.generation_interval:
                checkpoint_due = True

        if checkpoint_due:
            self.save_checkpoint(config, population, species_set, self.current_generation)
            self.last_generation_checkpoint = self.current_generation
            self.last_time_checkpoint = time.time()

    def save_checkpoint(self, config, population, species_set, generation):
        """ Save the current simulation state. """
        #filename = '{0}{1}'.format(self.filename_name, generation)
        #print("Saving checkpoint to {0}".format(filename))

        with gzip.open(self.checkpointPath, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


        #trovo il winnerID ( chi ha il fitness più alto )
        t = {p:population[p].fitness for p in population if population[p].fitness!=None }
        if len(t)>0:        #se non ce ne sono, non salvo ( tutti i fitness a 0 )
            winnerID=max(t, key=lambda x:t[x])

            winner = population[winnerID]  #--> non so se questo comando potrebbe dare problemi se non c'è un winner ( tutti a None / 0 )
            with open(self.winnerPath, 'wb') as f:     #salvo il winner
                pickle.dump(winner, f)
        



    @staticmethod
    def restore_checkpoint(filename):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            pop =  Population(config, (population, species_set, generation+1))
            #imposto l'indexer ( contatore che da l'id al prossimo genoma creato) uguale al numero massimo presente nella popolazione +1
            #altrimenti quando riprendo il train da una situazione esistente, gli id vengono generati da 1 e vanno a sovrascrivere i vecchi genomi!
            pop.reproduction.genome_indexer = count(max(population.keys())+1)       
            return pop
