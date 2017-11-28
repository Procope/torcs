import os
import neat
import pickle
import subprocess
from pytocl.protocol import Client
from my_driver import MyDriver

def run(config_file, checkpoint):
    """load the config, create a population, evolve and show the result"""
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # load or create the population, which is the top-level object for a NEAT run.
    if checkpoint:
        population = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-' + str(checkpoint))
    else:
        population = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1, None, 'checkpoints/neat-checkpoint-'))

    # Run for up to 30 generations.
    winner = population.run(eval_genomes, 30)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('network_winner.pickle', 'wb') as net_out:
        pickle.dump(net, net_out)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))



def eval_genomes(genomes, config):
    best_time = float('inf')
    best_genome = max(genomes, key=lambda genome: genome[1].fitness if genome[1].fitness else float('-inf'))[1]

    for idx, item in enumerate(genomes):
        print('idx: ', idx)

        genome_id, genome = item
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        print('start server')        
        server_proc = subprocess.Popen(["torcs", "-r", torcs_config_file], stdout=subprocess.PIPE)

        driver = MyDriver(network=net, logdata=False)
        client = Client(driver=driver)
        
        print('driving...')
        T_out, distance, ticks = client.run()

        time = get_time(server_proc)
        if time < best_time:
            best_time = time
        
        speed_avg = distance/ticks
        eta = 1000
        beta = 1000
        genome.fitness = eta - T_out + speed_avg*beta + distance
        
        print('fitness:   ', genome.fitness)
        print('T_out:     ', T_out)
        print('distance:  ', distance)
        print('ticks:     ', ticks)
        print('speed:     ', speed_avg)
        print('time:      ', time, '\n')

        if genome.fitness > best_genome.fitness:
            net = neat.nn.FeedForwardNetwork.create(genomes.best_genome, config)
            with open('network_best.pickle', 'wb') as net_out:
                pickle.dump(net, net_out)

    print('Best time: ', best_time)

def get_time(process):
    try:
        sim, _, time, *_ = process.communicate()[0].splitlines()[-1].split() #Sim Time: 60.47 [s], Leader Laps: 1, Leader Distance: 2.058 [km]
        if sim.decode('UTF-8') == 'Sim':  
            time = float(time)
            print(time)
            return time
    except IndexError as ex:
        pass
    except ValueError as ex:
        pass
    
    return float('inf')


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
