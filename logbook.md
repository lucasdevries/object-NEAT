## Log book
# Sunday,May 6 to Tuesday, May 8
* Transferred part of the neat-python code to MultiNEAT to do object recognition.
* Did a lot of different test runs to see if MultiNEAT is working accordingly.
* Wrote additional code to plot the individuals, save statistics and make MultiNEAT complete for experimentation. 
* Runs about 10 times faster than neat-python, great!
* Still figuring out how some python bindings work.
* I think this is a great step. Experimenting with neat-python gave me the ability to learn all about the parameters and how NEAT works and this implementation is great for experimentation.

# Saturday,May 5
* Spent all day trying to install MultiNEAT. Istalled Xcode, and after a few hours I got it working (magicically..). Ran some tests with a simple XOR example for NEAT and HyperNEAT. Seems to be working.


# Friday,May 4
* Gongjin and I installed CLion and OpenCV to give me the ability to build a larger data set with new samples from different robots.
* Another attempt to install MultiNEAT, because I still have the feeling that neat-python is not the right choice. I have to alter too much of the source code and see no real improvements. Also, it is really slow. 

# Thursday,May 3
* Meeting with Finn was postponed. No further developments.

# Wednesday,May 2
* Discussed the greater goal of the research and my thesis as a component of this research.

# Tuesday, May 1
* Implemented the computation time in the fitness function. Weight for normal fitness is now 0.95, while weight for computation time is 0.05, and both are normalized such that the time dependance is still small.
* Wrote down all questions for meeting with Finn on Wednesday. 
* Started to build a TensorFlow pipeline to get a Neural Network using backpropagation to compare to my evolve NN. Test evaluation: 50%…

# Monday, April 30
* Tried to implement calculation time dependancy. Seems counterintuitive to me: shorter calculation time means a less complex network. But we want both short calculation time and high complexity.
* Gongjin and I had a discussion with Jacqueline. Got some interesting questions and wrote these down to discuss with Finn. 
* Started to experiment with more individuals and lower compatibility to increase diversity.

# Thursday, April 26
* Gongjin explained why we just care about the small image sample and not the fullsize images. Discussed the hog filter.
* Skimage test accuracy: 0.845. I stopped the CV2 experiment because the fitness did not go above 0.0.
* Played with the PEAS implementation of NEAT. Modified it for object recognition. Each generation is much slower (takes about 10 times as much time). I will discuss this with Finn. 

# Wednesday, April 25
* Experiments with the a Hog filter. Tried two implementation: one with CV2 (336 inputs) and one from Skimage (624 inputs). CV2 gives me the opportunity to define stride, Skimage is more shallow. Visually, the difference between a positive and negative example for Skimage is large. But for CV2 with the parameters from Gongjins research, the difference looks very small.
* Set up 2 experiments with both implementations.

# Tuesday, April 24
* Attended a funeral, so I was not able to work on the thesis.

# Monday, April 23
* Both experiments wit multiple or just one structural mutation have a high test accuracy (0.88 and 0.885 resp.). The single structural mutation is reaches a higher accuracy, but it takes longer to get an individual with high fitness. Training fitness is 0.92 for both. 
* Implemented the new fitness function.
* Run: 1000 gens with new fitness function. 
* Implemented a seperate test python file. Now I can change some criteria or do testing after the evolving is done. Also, the model is saved to a pickle file and could be loaded.

# Friday, April 20
* Set up another experiment with 2500 generations, now with a maximum of 1 stuctural mutation on each child. Still using the old fitness function to compare it to the previous run.

One structural mutation:
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/test4-avg.png?raw=true)
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/test4-indi.png?raw=true)

# Thursday, April 19
* Started to experiment with 2500 generations and the lower (0.01) enable connection rate. A lot of mutations happen on each child. Interesting to lower this amount.
* Had a talk with Gongjin on the fitness function. By penalizing and rewarding outputs that are very much right or wrong, a 'bonus' is added to the dot product: the difference between the predicted values. So: [0.49, 0.51] would have less bonus than [0.01, 0.99] for a negative example, but [0.51,0.49] is also penalized less than [0.99, 0.01].
Multiple structural mutations:
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/test3-avg.png?raw=true)
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/test3-indi.png?raw=true)

# Wednesday, April 18
* Softmax yields values between [0.27,0.73] when the prediction itself is [0,1], tried to fix this and implement Cross Entropy as error/fitness. This has to be minimized, but NEAT is not working well for minimizing functions. So I decided to continue with softmax and the dot product as fitness.
* Als added penalizing for wrong classifications to steer NEAT even more in the right direction.
* Implemented Tournament selection by altering the source code. Initial was random selection.
* Ran an experiment with 1000 generations, but fitness was decreasing. Slight increase in the beginning and drop after generation 70. Might be due to drop in speciation due to stagnation and species elitism 3. A lot of bad (the worst, fitness -.5) individuals are still reproduced. See the figures below. Must be due to the reproduction threshold. For the next experiments, this fraction is decreased.
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/indivduals-1000gens-18april-tourn.png?raw=true)
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/avg-18april-tourn.png?raw=true)
* Another test with lower enable connection rate (now 0.01), yields complexer model and better performance even after 100 generations.
* ![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/test-1-avg-100gens-19april-test1?raw=true)
![alt text](https://github.com/lucasdevries/object-NEAT/blob/master/images/test-1-indivduals-100gens-19april-test1.png?raw=true)
* Increasing average fitness!!!!


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
