import os
import neat
import gzip
import random
import pickle
import psutil
import logging
import importlib
import subprocess
from pytocl.protocol import Client
import my_driver
from my_driver_swarm import MyDriver
from threading import Thread
from time import sleep

logger = logging.getLogger(__name__)

class BestGenomeReporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        with open('network_best_swarm.pickle', 'wb') as net_out:
            pickle.dump(net, net_out)

        print_evaluation(best_genome.evaluation)

        print('Finished genomes:', sum([1 for genome_idx in population 
            if hasattr(population[genome_idx], 'evaluation') and population[genome_idx].evaluation[6]]))

def run(checkpoint):
    """create a population, evolve and show the result"""
    # load or create the population, which is the top-level object for a NEAT run.
    if checkpoint:
        if checkpoint == -1:
            file = max(os.listdir('checkpoints_swarm/'), key=lambda f: int(f.split('-')[-1]))
        else:
            file = 'neat-checkpoint-' + str(checkpoint)

        with gzip.open('checkpoints_swarm/' + file) as f:
            generation, config_prev, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            population = neat.Population(config, (population, species_set, generation))
        # population = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-' + str(checkpoint))
    else:
        population = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(BestGenomeReporter())
    population.add_reporter(neat.Checkpointer(1, None, 'checkpoints_swarm/neat-checkpoint-'))

    # Run for up to 30 generations.
    winner = population.run(eval_genomes, 50)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('network_winner_swarm.pickle', 'wb') as net_out:
        pickle.dump(net, net_out)

    # Display the winning genome.
    print('\nWinning genome:\n{!s}'.format(winner))



def eval_genomes(genomes, config):
    importlib.reload(my_driver)
    from my_driver import MyDriver

    for idx, item in enumerate(genomes):
        print('Genome: ', idx)

        print('start server')        
        server_proc = subprocess.Popen(["torcs", "-r", torcs_config_file], stdout=subprocess.PIPE)

        print('start clients')
        [os.remove(file) for file in os.listdir() if file.startswith('pos_')]        
        genome_idx, genome = item
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        result1 = []
        driver1 = Thread(target=run_client, args=(0, genome, net, result1))
        driver1.start()

        result2 = []
        driver2 = Thread(target=run_client, args=(1, genome, net, result2))
        driver2.start()

        try:
            server_proc.wait(timeout=60)
            print('\nserver stopped\n')
        except subprocess.TimeoutExpired as ex:
            process = psutil.Process(server_proc.pid)
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
            print('\nserver expired\n')
        driver1.join()
        driver2.join()

        if result1[0] > result2[0]:
            print('client:    1')
            genome.fitness = result1[0]
            genome.evaluation = result1
            print_evaluation(result1)
        else:
            print('client:    2')
            genome.fitness = result2[0]
            genome.evaluation = result2
            print_evaluation(result2)

        print('clients done\n')        

def run_client(client_id, genome, network, evaluation):
    driver = MyDriver(network=network, logdata=False)
    client = Client(driver=driver, port=3001+client_id)
    
    print(client_id, 'driving...')
    client.run()

    evaluation.extend(driver.eval(2587.54)) # track length for aalborg
    
    
def print_evaluation(evaluation):
    print('fitness:   ', evaluation[0], '\n',
          'T_out:     ', evaluation[1], '\n', 
          'damage:    ', evaluation[2], '\n', 
          'distance:  ', evaluation[3], '\n', 
          'ticks:     ', evaluation[4], '\n',
          'speed:     ', evaluation[5], '\n',
          'prev time: ', evaluation[6], '\n',
          'cur time:  ', evaluation[7], '\n',
          'position:  ', evaluation[8], '\n',
          'start:     ', evaluation[9],
          sep='',
          end='\n\n')
# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

if __name__ == '__main__':
    # Parse command line arguments
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('torcs_config_file', help="YAML file with torcs configuration")
    parser.add_argument('neat_file', help="file with neat configuration")
    parser.add_argument('-c', '--checkpoint', type=int,
                        help="checkpoint number to use")
    args = parser.parse_args()

    torcs_config_file = os.path.abspath(args.torcs_config_file)
        # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         args.neat_file)
    logging.basicConfig(level=logging.ERROR)
    run(args.checkpoint)
