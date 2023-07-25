import random
import threading
import neat
import numpy as np
from Settings import Settings
from myCheckpointer import myCheckpointer
#from simulationProcess import simulationProcess
import time
from PortManager import *
import pickle
import pandas as pd
import os
from ColorTextLib import *
import math
import visualize
from simulation import SimulationProcess,SimulationAI,SimulationThread
from multiprocessing import Process, Manager,Value

settings = Settings()
portManager = PortManager(threading.Lock())
portManager.addPort(settings.ports)

manager = None 

def printMatrix(matrix,length=None):

    s = [[str(e) for e in row] for row in matrix]
    if length==None:
        lens = [max(map(len, col)) for col in zip(*s)]
    else: 
        lens = [length for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def toString(genomes,processes,deleteBefore=True):
    #stampo tutti i genomi con relativo fitness
    #per ciascuno stampo anche se è già stato processato o no
    tmp={}
    for genome_id, g in genomes:
        tmp[genome_id]={
            "genome":g,
            "time":0,
            "fit": "/", 
            "status":"/",
            "port":"/"
        }
        if g.fitness == 0:
            tmp[genome_id]["fit"]=0
        elif g.fitness != None: 
            tmp[genome_id]["fit"]="{:.4f}".format(g.fitness)

    for p in processes:
        tmp[p["genome_id"]]["time"]=p["secondsFromStart"].value #p["process"].secondsFromStart
        tmp[p["genome_id"]]["status"]="r" if not p["finish"] else "f"
        tmp[p["genome_id"]]["port"]="r" if not p["process"].port else p["process"].port

    def initMatrix():
        return [
            ["genoma"],
            ["time"],
            ["status"],
            ["fit"],
            #["port"],
            ]
        
    pMatrix = initMatrix()

    
    maxElPerRow=20
    
    #pulisco quelle prima
    if deleteBefore:
        numBlock=math.ceil(len(tmp)/maxElPerRow)
       
        rowPerBlock=4
        numLines=(numBlock*rowPerBlock)+(numBlock)        #(numBlock)  -> per cancellare le linee vuote
        cursor_up = '\x1b[1A'
        #erase_line = '\x1b[2K'         #non serve in quanto la sovrascrivo ogni volta  
        print(((cursor_up)*numLines) +cursor_up)
    

    i=0
    for id in tmp:
        t = tmp[id]
        pMatrix[0].append(id)
        pMatrix[1].append(t["time"])
        if t["status"]=="/":
            pMatrix[2].append(colorText(t["status"],bcolors.RED))
        elif t["status"]=="r":
            pMatrix[2].append(colorText(t["status"],bcolors.YELLOW))
        elif t["status"]=="f":
            pMatrix[2].append(colorText(t["status"],bcolors.GREEN))
        else:
            pMatrix[2].append(t["status"])


        pMatrix[3].append(t["fit"])
        #pMatrix[4].append(t["port"])

        i+=1

        if i >=maxElPerRow:
            printMatrix(pMatrix,6)
            print("")
            i=0
            pMatrix = initMatrix()

    if len(pMatrix[0])>1:
        printMatrix(pMatrix,6)
        print("")
   



    #esempio
    #genoma:    1   2   3   4       
    #time:      0   0   100 1080
    #status:    /   /   r   f      ( running giallo, finish verde, not start rosso)
    #fit:       1   2   3   50
    #port       813 814 815 816



def eval_genome(genome, config):

    genomeToTest = [ (genome_id, g) for  genome_id, g in genome]


    processesRunning=[]
    processesAll=[]
    justStarted=True
    
    while len(genomeToTest) > 0 or len(processesRunning)>0:
        port=None

        sleepTime=1

        if len(genomeToTest)>0: #se ce ne sono ancora
            freeCount=portManager.getFreeCount()
            #se ho più di un posto, allora diminuisco lo sleep per riempire più velocemente
            if freeCount > 1:   
                sleepTime=0.1

            if freeCount > 0:
                port= portManager.lockPort()
                genome_id, g=genomeToTest.pop()
                net = neat.nn.FeedForwardNetwork.create(g, config)  
                retVal = manager.dict()
                secondsFromStart = Value('f', 0.0)

                
                proc =settings.SimulationParallelClass(genome_id,net,settings,port=port,GUI=False,args=(retVal,secondsFromStart))  


                simData={
                        "port":port,
                        "process":proc,
                        "genome":g,
                        "genome_id":genome_id,
                        "retVal":retVal,
                        "secondsFromStart":secondsFromStart,
                        "finish":False
                    }

                processesRunning.append(simData)
                processesAll.append(simData)
                proc.start()

            

                

           
        #controllo i processi 
        processesJustEnd = [ p for p in processesRunning if not p["process"].is_alive()]
        genomeIds=[ p["genome_id"]  for p in processesJustEnd ]
        processesRunning = [ p for p in processesRunning if p["genome_id"] not in genomeIds]


        

        for p in processesJustEnd:
            
            total_simulation_time, max_time_loss , avg = p["retVal"]["total_simulation_time"],p["retVal"]["max_time_loss"],p["retVal"]["avg"]
            p["finish"]=True


            g=p["genome"]

            portManager.releasePort(p["port"])
            
           
            
            g.fitness = fitness(total_simulation_time,max_time_loss,avg,settings)
           

            #se qualche processo ha finito, vuol dire che ci sono posti liberi e aspetto poco
            sleepTime=0.1


      
        time.sleep(sleepTime)
        if justStarted:
            justStarted=False
            toString(genome,processesAll,False)
        else:
            toString(genome,processesAll)


def fitness(total_simulation_time,max_time_loss,avg,settings:Settings):

    #più il valore è alto, più vuol dire che sta andando bene ( controllare nell'ini il fitness_threshold, ovvero il valore considerato "massimo" del fitness )
    if total_simulation_time>=settings.maxSimulationTime:
        fitness=0
    elif avg==0:
        fitness=1   #perfetto, nessuno ha perso tempo
    else:
        fitness=1/((avg+max_time_loss)/2)
        #incluso anche il max_time_loss ->  media tra l'avg e il max_time_loss
        #oppure aggiungere delle penalità più è alta la differenza tra avg e max_time_loss
        #oppure escludere direttamente qualsiasi generazione che ha un max_time_loss troppo alto ( tipo il doppio / triplo dell'avg)
    return fitness

def main():

    random.seed(settings.random_seed)
    np.random.seed(settings.random_seed)


    if os.path.exists(settings.checkPointPath):
        population = myCheckpointer.restore_checkpoint(settings.checkPointPath)
        config = population.config

    else:
        # Configurazione NEAT
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,settings.neat_config_ini)
        config.genome_config.seed = settings.random_seed
        
        # Creazione del popolazione di reti neurali
        population = neat.Population(config)

    assert config.genome_config.num_outputs==len(settings.traffic.trafficLightPhases), "Numero di ouput nel 'config.ini' non corrisponde al numero di 'trafficLightPhases' nei settings"
        


    # Report sui progressi dell'addestramento
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(myCheckpointer(1, 5,settings.checkPointPath,settings.winningPath))   #reporter per salvare la rete neurale
    
    winner=None

    if not settings.OnlyRunWinner:
        # Eseguire l'evoluzione per un numero di generazioni
        for i in range (0,settings.neat_num_generations):
            winner = population.run(eval_genome, 1)

            #salva il winner
            #with open(settings.winningPath, 'wb') as f:
            #    pickle.dump(winner, f)

            

        

    

    


    #test del migliore / salvato
    with open(settings.winningPath, 'rb') as f:
        winner = pickle.load(f)
    # Visualizzare il miglior individuo (rete neurale)
    print('\nMiglior individuo:\n{!s}'.format(winner))




    node_names = {-1: 'E0_0D', -2: 'E0_1D',-3: '-E4_0D',-4: '-E5_0D',-5: 'E2_0D',-6: 'E2_1D',-7: 'T1',-8: '-T2',-9: 'T3',-10: 'T4',-11: 'T5',-12: 'T6', 0: 'Phase 1',1: 'Phase 3',2: 'Phase 5',3: 'Phase 7',4: 'Phase 9'}
    #visualize.draw_net(config, winner,True, filename="__deleteMe" ,node_names=node_names)
    #visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    

    if not settings.OnlyRunWinner:
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)


    # Utilizzare il vincitore per effettuare predizioni
    winner_net = neat.nn.FeedForwardNetwork.create(winner, population.config)
    

    caller = SimulationThread(winner.key,winner_net,settings,GUI=True,args=(manager.dict(),None))
    total_simulation_time, max_time_loss , avg=caller.runAndWait()      


    print("total_simulation_time: ",total_simulation_time)
    print("max_time_loss: ",max_time_loss)
    print("avg: ",avg)
    print("fitness: ",fitness(total_simulation_time,max_time_loss,avg,settings) )




   

if __name__ == "__main__":

    manager= Manager()
    main()
