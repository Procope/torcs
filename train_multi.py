import os
import neat
import gzip
import random
import pickle
import psutil
import importlib
import subprocess
from pytocl.protocol import Client
import my_driver
from my_driver import MyDriver
import asyncio

class BestGenomeReporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        with open('network_best.pickle', 'wb') as net_out:
            pickle.dump(net, net_out)

def run(config_file, checkpoint):
    """load the config, create a population, evolve and show the result"""
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

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
    with open('network_winner.pickle', 'wb') as net_out:
        pickle.dump(net, net_out)

    # Display the winning genome.
    print('\nWinning genome:\n{!s}'.format(winner))



def eval_genomes(genomes, config):
    importlib.reload(my_driver)
    from my_driver import MyDriver

    best_time = float('inf')
    finished = 0

    for batch_idx, batch in enumerate(chunks(genomes, 10)):
        print('batch idx:', idx)

        print('start server')        
        server_proc = subprocess.Popen(["torcs", "-r", torcs_config_file], stdout=subprocess.PIPE)
        
        clients = []        
        for idx, item in enumerate(batch):
            clients.append(run_client(idx, *item))

        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.gather(*clients))
        loop.close()
        
        for 
        try:
            server_proc.wait(timeout=20)
        except subprocess.TimeoutExpired as ex:
            process = psutil.Process(server_proc.pid)
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
        
    print('Best time: ', best_time)
    print('Finished races: ', finished)

async def run_client(client_id, genome_id, genome):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    driver = MyDriver(network=net, logdata=False)
    client = Client(driver=driver, port=3001+client_id)
    
    print(client_id, 'driving ...')
    client.run()

    genome.fitness = driver.eval(2587.54) # track length for aalborg

    if driver.prev_state.last_lap_time:
        finished += 1
        if driver.prev_state.last_lap_time < best_time:
            best_time = driver.prev_state.last_lap_time

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

    run(args.neat_file, args.checkpoint)
