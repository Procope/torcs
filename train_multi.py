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
from my_driver import MyDriver
from threading import Thread, Semaphore
from time import sleep

_logger = logging.getLogger(__name__)

class BestGenomeReporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        with open('network_best_multi.pickle', 'wb') as net_out:
            pickle.dump(net, net_out)

def run(checkpoint):
    """create a population, evolve and show the result"""
    # load or create the population, which is the top-level object for a NEAT run.
    if checkpoint:
        if checkpoint == -1:
            file = max(os.listdir('checkpoints_multi/'), key=lambda f: int(f.split('-')[-1]))
        else:
            file = 'neat-checkpoint-' + str(checkpoint)

        with gzip.open('checkpoints_multi/' + file) as f:
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
    population.add_reporter(neat.Checkpointer(1, None, 'checkpoints_multi/neat-checkpoint-'))

    # Run for up to 30 generations.
    winner = population.run(eval_genomes, 50)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('network_winner_multi.pickle', 'wb') as net_out:
        pickle.dump(net, net_out)

    # Display the winning genome.
    print('\nWinning genome:\n{!s}'.format(winner))



def eval_genomes(genomes, config):
    importlib.reload(my_driver)
    from my_driver import MyDriver

    lock = Semaphore(value=1)
    best_time = float('inf')
    finished = 0

    for batch_idx, batch in enumerate(chunks(genomes, 10)):
        print('batch idx:', batch_idx)

        print('start server')        
        server_proc = subprocess.Popen(["torcs", "-r", torcs_config_file], stdout=subprocess.PIPE)

        print('start clients')        
        clients = []        
        for idx, item in enumerate(batch):
            genome_idx, genome = item
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            client = Thread(target=run_client, args=(idx, genome, net, lock))
            clients.append(client)
            client.start()

        try:
            server_proc.wait(timeout=180)
        except subprocess.TimeoutExpired as ex:
            process = psutil.Process(server_proc.pid)
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
        print('\nserver stopped\n')

        for client in clients:
            client.join()
        print('clients done\n')        
    print('Best time: ', best_time)
    print('Finished races: ', finished)

def run_client(client_id, genome, network, lock):
    driver = MyDriver(network=network, logdata=False)
    client = Client(driver=driver, port=3001+client_id)
    
    print(client_id, 'driving...')
    client.run()

    evaluation = driver.eval(2587.54) # track length for aalborg
    lock.acquire()
    print('Genome:    ', client_id, '\n',
          'fitness:   ', evaluation[0], '\n',
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
    lock.release()
    
    genome.fitness = evaluation[0]

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
