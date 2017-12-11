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

class BestGenomeReporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        with open('single_driver/network_best.pickle', 'wb') as net_out:
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
            file = max(os.listdir('single_driver/checkpoints/'), key=lambda f: int(f.split('-')[-1]))
        else:
            file = 'neat-checkpoint-' + str(checkpoint)

        with gzip.open('single_driver/checkpoints/' + file) as f:
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
    population.add_reporter(neat.Checkpointer(1, None, 'single_driver/checkpoints/neat-checkpoint-'))

    # Run for up to 30 generations.
    winner = population.run(eval_genomes, 50)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('single_driver/network_winner.pickle', 'wb') as net_out:
        pickle.dump(net, net_out)

    # Display the winning genome.
    print('\nWinning genome:\n{!s}'.format(winner))



def eval_genomes(genomes, config):
    importlib.reload(my_driver)
    from my_driver import MyDriver

    best_time = float('inf')
    finished = 0

    for idx, item in enumerate(genomes):
        print('idx:', idx)

        genome_id, genome = item
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        print('start server')        
        server_proc = subprocess.Popen(["torcs", "-r", torcs_config_file], stdout=subprocess.PIPE)

        driver = MyDriver(network=net, logdata=False)
        client = Client(driver=driver)
        
        print('driving...')
        client.run()

        try:
            server_proc.wait(timeout=20)
        except subprocess.TimeoutExpired as ex:
            process = psutil.Process(server_proc.pid)
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
        
        genome.fitness = driver.eval(2057.56) # track length for speedway
        # genome.fitness = driver.eval(6355.65) # track length for alpine 1
        # genome.fitness = driver.eval(3274.20) # track length for ruudskogen
        print('fitness:   ', genome.fitness, '\n')


        if driver.prev_state.last_lap_time:
            finished += 1
            if driver.prev_state.last_lap_time < best_time:
                best_time = driver.prev_state.last_lap_time

    print('Best time: ', best_time)
    print('Finished races: ', finished)

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
