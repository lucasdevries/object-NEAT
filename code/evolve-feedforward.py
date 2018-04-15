"""
Object recognition NEAT
"""

from __future__ import print_function
import os
import neat
import visualize
import load_images
import time

xor_inputs, test_inputs , xor_outputs, test_outputs = load_images.load_data()

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        fitness_start = 0
        # fitness_pos = 0
        # fitness_neg = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)

            if abs(output[0] - xo[0]) < 0.1:
                fitness_start += 1
            else:
                fitness_start += 0
        genome.fitness = fitness_start / float(len(xor_outputs))


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(300))
    
    # Run for up to 300 generations.
    start_time = time.time()


    winner = p.run(eval_genomes, 5000)

    stats.save()

    print(stats.best_genomes(5))
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input , expected output {!r}, got {!r}".format(xo, output))
    # test on testset
    fitness_test = 0

    for xi_test, xo_test in zip(test_inputs, test_outputs):
        output_test = winner_net.activate(xi_test)

        if abs(output_test[0] - xo_test[0]) < 0.1:
            fitness_test += 1
        else:
            fitness_test += 0

    test_evaluation = fitness_test / float(len(test_outputs))
    print("\nTest evaluation: {!r}".format(test_evaluation))
    print("--- %s seconds ---" % (time.time() - start_time))

    node_names = {0:'Output'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
 
    
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)
    


# def eval_genomes_test(genomes, config):
        
#         # fitness_pos = 0
#         # fitness_neg = 0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         for xi_test, xo_test in zip(test_inputs, test_outputs):
#             output_test = net.activate(xi_test)

#             if abs(output_test[0] - xo_test[0]) < 0.1:
#                 fitness_test += 1
#             else:
#                 fitness_test += 0
#         test_evaluation = fitness_test / float(len(test_outputs))
#         print("\nTest evaluation: {!r}".format(test_evaluation))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)