import neat
from torcs_tournament import *
import pickle

def run(config_file, checkpoint):
    """load the config, create a population, evolve and show the result"""
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if checkpoint:
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-' + str(checkpoint))
    else:
        p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, None))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 1)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('network_winner.pickle', 'wb') as net_out:
        pickle.dump(net, net_out)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        with open('network.pickle', 'wb') as net_out:
            pickle.dump(net, net_out)

        controller.race_and_save(simulate=args.simulate)

        with open('../torcs-client/eval.pickle', 'rb') as eval_in:
            T_out, distance, ticks = pickle.load(eval_in)
        
        speed_avg = distance/ticks
        eta = 1000
        beta = 1000
        genome.fitness = eta - T_out + speed_avg*beta + distance
        print('fitness:  ', genome.fitness)
        print('T_out:    ', T_out)
        print('distance: ', distance)
        print('ticks:    ', ticks)
        print('speed:    ', speed_avg)

if __name__ == '__main__':
    # Parse command line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('config_file', help="YAML file with torcs configuration")
    parser.add_argument('neat_file', help="file with neat configuration")
    parser.add_argument('-c', '--checkpoint', type=int,
                        help="checkpoint number to use")
    parser.add_argument('-l', '--level', default='INFO', type=log_level_type,
                        help="Logging level to use")
    parser.add_argument(
        '-s',
        '--simulate',
        action='store_true',
        help="Attempts to mimic a full run without starting child processes."
        " May fail if no old TORCS output files are present in the expected"
        " directory.")

    args = parser.parse_args()

    # Initialise logging
    logging.basicConfig(level=args.level)

    # Race
    controller = Controller.load_config(args.config_file)

    run(args.neat_file, args.checkpoint)
    logger.info("Done!")
