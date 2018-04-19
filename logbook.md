## Log book
# Wednesday, April 18
* Softmax yields values between [0.27,0.73] when the prediction itself is [0,1], tried to fix this and implement Cross Entropy as error/fitness. This has to be minimalized, but NEAT is not working well for minimalizing functions. So I decided to continue with softmax and the dot product as fitness.
* Als added penelizing for wrong classifications to steer NEAT even more in the right direction.
* Implemented Tournament selection by altering the source code. Initial was random selection.
* Ran an experiment with 1000 generations, but fitness was decreasing. Slight increase in the beginning and drop after generation 70. Might be due to drop in speciation due to stagnation and species elitism 3. A lot of bad (the worst, fitness -.5) individuals are still reproduced. See the figures below. Must be due to the reproduction threshold. For the next experiments, this fraction is decreased.
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/indivduals-1000gens-18april-tourn.png?raw=true)
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/avg-18april-tourn.png?raw=true)


# Tuesday, April 17
* Results: test accuracy 0.80. 
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/avg-17april.png?raw=true)
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/indivduals-5000gens-17april.png?raw=true)
* Interesting: average fitness of population is decreasing.
* Implemented NEAT for the larger images (600x800px) to see if a normal field of view images works better than cropped images. 
* Talked to Gongjin. We found out that neat-python does not use tournament selection but random selection. Tomorrow I will implement Tournament selection and check PEAS (python evolutionary algoritms) if it is also suitable for object recognition. 


# Monday, April 16
* Changed the sobel filter: Using a larger stride (3), pooling is not needed anymore.
* Implemented soft-max layer in the evaluation to predict final output.
* Run experiment, 5000 generations with the new sobel operator and softmax layer. Testing on both positive and negative samples. Reduced compatability threshold further to 2.4 to get more speciation. 

# Thursday, April 12
* MNIST crashed, no space left on hard disk. 
* Restore checkpoint function was nog working as is should have worked. Debugged and found a solution to the problem.
* Same github problem, I have to get rid of the checkpoints upload but gitignore is not working... Old (unfinished) commits are still in queue.
* Strangly, the best indiviual jumps from fitness 0 to ~0.2 to 0.5 to 0.83, no gradual improvements.
* Talk with Finn and Gongjin: next steps: try sobel with larger stride, so pooling is not neccessary. Test only for positive samples.w

# Wednesday, April 11
* Analyzed the 5000 gens run from last night.
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/indivduals-5000gens-11april.png?raw=true)
* There were just two species, so for the next run I changed the compatibility_threshold to 2.6, species_elitism to 3 and elitism to 1 with a survival_threshold of 0.8. This should infer more diversity and more reproduction, while the best 3 individuals of the different species are conserved.
* mnist is still running. 
* Reviewed the positive and negative examples and I got rid off all images for which it was difficult (even for a human) to classify it as positive or negative.

# Tuesday, April 10
* Extracted the network parameters for the n best individuals.
* Implemented a python script to load the MNIST data set and modify NEAT to work with this data.
* Ran new experiment with 6000 generations to use and to track the fitness back to generation zero. 
* Reviewed fitness function for mnist data and implemented a new version for mnist.
* Runs: mnist 2500 gens, evolve 5000 gens.


# Monday, April 9
* Plotted all individuals: fitness vs. generations. Therefore I had to extract all innovation numbers and generation numbers. Modified a lot of code. 
* A lot of networks have fitness 0.5: expected as the output is binary. 
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/individuals.png?raw=true)

# Friday, April 5
* Ran a few experiments with 2500 and 6000 generations, got 0.83 test accuracy. Still not the network with a lot complexity.
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/6000gens.png?raw=true)
* Next step: track individual over all generations and see how it changes over time to get more insights.
* For next experiment: reduce elitism to achieve higher diversity. 

# Thursday, April 5
* Made logbook to track progress. 
* Played with parameters, got 0.81 test accuracy. 
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/300gens.png?raw=true)



# Wednesday, April 4
* Solved fitness problem: integer should have been divided by a float in order to get a float.
```python
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
```
* Build a separate file for data loading `load_images.py` and evolving the network `evolve-feedforward.py`.
* Implemented a larger data set and evaluation for the test set for the best individual.
````python
		winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi_test, xo_test in zip(test_inputs, test_outputs):
        output_test = winner_net.activate(xi_test)

        if abs(output_test[0] - xo_test[0]) < 0.1:
            fitness_test += 1
        else:
            fitness_test += 0

    test_evaluation = fitness_test / float(len(test_outputs))
    print("\nTest evaluation: {!r}".format(test_evaluation))
````
* Test accuracy for data set with 400 positive and 400 negative examples (train 75%, test 25%) is 0.74.
* Some hidden nodes, but not enough and not real complex networks. Need to figure out what the speciation parameters are doing and how reproduction is defined to create more complex networks.

# Tuesday, April 3
* Started to modify the XOR example to have more input nodes.
* Defined the `sobel` operator and performed this operation on a subset of 10 positive and 10 negative examples.
```python
def sobel_op(image):
    dx = ndimage.sobel(image, 0)  # horizontal derivative
    dy = ndimage.sobel(image, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag = mag / np.max(mag) # normalize (Q&D)
    return mag
```
* After the `sobel` operator, the images gets (max)pooled once to reduce the image size: number of input nodes is 762.
* It seems that NEAT is learning and all 20 examples are classified correctly, but no hidden nodes are created and the network is not complex at all. Might be due to the arbitrary fitness function or too small sample size.
* Changed fitness function to the fraction of correctly classified  examples. Correct if the difference between output and expected output is less than 0.1.
```python
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
        genome.fitness = fitness_start / len(xor_outputs)
```
Now, no learning happens at all. Genome fitness is always zero.
