import pickle
import neat
import os
import gzip
from train_swarm import BestGenomeReporter
import random
from neat.six_util import iteritems, itervalues
from pprint import pprint
import matplotlib.pyplot as plt

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

def get_statistics(directory):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

    stats = []
    for file in sorted(os.listdir(directory), key=lambda x: int(x[16:])):
        if file.startswith("neat-checkpoint-"): 
            with gzip.open(os.path.abspath(directory + file)) as f:
                generation, config_prev, population, species_set, rndstate = pickle.load(f)
                random.setstate(rndstate)
                population = neat.Population(config, (population, species_set, generation))
            best = None
            for g in itervalues(population.population):
                if best is None or (g.fitness and best.fitness and g.fitness > best.fitness):
                    best = g
            if best.fitness is None:
                best = stats[-1][1]
            stats.append((generation, best))
            continue
        else:
            continue

    return stats

if __name__ == '__main__':
    stats = []
    directory = 'single_driver/checkpoints/'
    stats.extend(get_statistics(directory))
    directory = 'checkpoints_swarm/'
    stats.extend(get_statistics(directory))

    for generation, genome in stats:
        print(generation, end=' ')
        print(genome.fitness, end=' ')
        if hasattr(genome, 'evaluation'):
            print(genome.evaluation, end='')
            print('speed', genome.evaluation[3]/genome.evaluation[6], end='')
        print()

    x1 = [generation for generation, genome in stats[:87]]
    y1 = [genome.fitness for generation, genome in stats[:87]]
    x2 = [generation for generation, genome in stats[87:124]]
    y2 = [genome.fitness for generation, genome in stats[87:124]]
    x3 = [generation for generation, genome in stats[125:174]]
    y3 = [genome.fitness for generation, genome in stats[125:174]]

    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(x1, y1, 'x', linestyle='-', markersize=4)
    ax1.set_ylabel('fitness')
    ax1.set_ylim([1500,2100])
    ax1.set_xlim([-3,90])

    ax2.plot(x2, y2, 'x', linestyle='-', markersize=4)
    ax2.set_ylabel('fitness')
    ax2.set_xlabel('generation')
    ax2.set_ylim([6180,6200])
    ax2.set_xlim([85,125])

    # ax3.plot(x3, y3, 'bo', markersize=1)
    # ax3.set_ylim([2440,2460])

    plt.subplots_adjust(wspace=.5)

    plt.savefig('fitness.png')
    plt.show()

