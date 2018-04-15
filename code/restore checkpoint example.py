def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    checkpoint_file='neat-checkpoint-3210'
    p = neat.Checkpointer().restore_checkpoint(checkpoint_file)
    # p = neat.Population(config, (population, species_set, generation))
    # p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(30))
    
    # Run for up to 300 generations.
    start_time = time.time()

    # p.__resume_checkpoint('neat-checkpoint-3210')

    winner = p.run(eval_genomes, 4000)

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
        output_test = softmax(output)
        if ((output_test[int(xo_test[0])] == max(output_test)) and (abs(output_test[int(xo_test[0])] - 1.0) < 0.1)):
                # print(output)
                # print(output[int(xo[0])])
                # print(xo[0])
            fitness_test += 1
        else:
            fitness_test += 0

    test_evaluation = fitness_test / float(len(test_outputs))
    print("\nTest evaluation: {!r}".format(test_evaluation))
    print("--- %s seconds ---" % (time.time() - start_time))

    node_names = {}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
 
    
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-3210')
    p.run(eval_genomes, 10)
    