# One Max Problem
# This is the first complete example built with DEAP. It will help new users to overview some of the framework’s possibilities and illustrate the potential of evolutionary algorithms in general. 
# The problem itself is both very simple and widely used in the evolutionary computational community. 
# We will create a population of individuals consisting of integer vectors randomly filled with 0 and 1. 
# Then we let our population evolve until one of its members contains only 1 and no 0 anymore.

# Setting Things Up
# In order to solve the One Max problem, we need a bunch of ingredients. First we have to define our individuals, which will be lists of integer values, and to generate a population using them. Then we will add some functions and operators taking care of the evaluation and evolution of our population and finally put everything together in script.
# But first of all, we need to import some modules.

import random

from deap import base
from deap import creator
from deap import tools


# Since the actual structure of the required individuals in genetic algorithms does strongly depend on the task at hand, 
# DEAP does not contain any explicit structure. 
# It will rather provide a convenient method for creating containers of attributes, associated with fitnesses, called the deap.creator. 
# Using this method we can create custom individuals in a very simple way.

# The creator is a class factory that can build new classes at run-time. 
# It will be called with first the desired name of the new class, 
# second the base class it will inherit, and in addition any subsequent 
# arguments you want to become attributes of your class. 
# This allows us to build new and complex structures of any type of container from lists to n-ary trees.

# First we will define the class FitnessMax. 
# It will inherit the Fitness class of the deap.base module and contain an additional attribute called weights. 
# Please mind the value of weights to be the tuple (1.0,). This way we will be maximizing a single objective fitness. 
# We can’t repeat it enough, in DEAP single objectives is a special case of multi objectives.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Now we will use our custom classes to create types representing our individuals as well as our whole population.

# All the objects we will use on our way, an individual, the population, as well as all functions,
# operators, and arguments will be stored in a DEAP container called Toolbox. 
# It contains two methods for adding and removing content, register() and unregister().

# In this code block we register a generation function toolbox.attr_bool() and two initialization ones individual() 
# and population(). toolbox.attr_bool(), when called, will draw a random integer between 0 and 1. 
# The two initializers, on the other hand, will instantiate an individual or population.

# The registration of the tools to the toolbox only associates aliases to the already existing functions 
# and freezes part of their arguments. 
# This allows us to fix an arbitrary amount of argument at certain values so we only have to specify the 
# remaining ones when calling the method. 
# For example, the attr_bool() generator is made from the randint() function that takes two arguments 
# a and b, with a <= n <= b, where n is the returned integer. Here, we fix a = 0 and b = 1.

# Our individuals will be generated using the function initRepeat(). 
# Its first argument is a container class, in our example the Individual one we defined in the previous section. 
# This container will be filled using the method attr_bool(), provided as second argument, 
# and will contain 100 integers, as specified using the third argument. When called, the individual() method will 
# thus return an individual initialized with what would be returned by calling the attr_bool() method 100 times. 
# Finally, the population() method uses the same paradigm, but we don’t fix the number of individuals 
# that it should contain.
toolbox = base.Toolbox()
# Attribute generator 
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# The evaluation function is pretty simple in our example. We just need to count the number of ones in an individual.
# The returned value must be an iterable of a length equal to the number of objectives (weights).
def evalOneMax(individual):
    return sum(individual),

# The Genetic Operators
# Within DEAP there are two ways of using operators. 
# We can either simply call a function from the tools module or register it with its arguments in a toolbox, 
# as we have already seen for our initialization methods. 
# The most convenient way, however, is to register them in the toolbox, because this allows us to 
# easily switch between the operators if desired. The toolbox method is also used when working with the algorithms module. See the One Max Problem: Short Version for an example

# Registering the genetic operators required for the evolution in our One Max problem and their default arguments in the toolbox is done as follows.

# The evaluation will be performed by calling the alias evaluate. 
# It is important to not fix its argument in here. We will need it later on to apply the function to each separate 
# individual in our population. The mutation, on the other hand, needs an argument to be fixed 
# (the independent probability of each attribute to be mutated indpb).
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Evolving the Population
# Once the representation and the genetic operators are chosen, we will define an algorithm combining all the individual 
# parts and performing the evolution of our population until the One Max problem is solved. 
# It is good style in programming to do so within a function, generally named main().
def main():
    # pop will be a list composed of 300 individuals. Since we left the parameter n open during the registration of the population() method in our toolbox, we are free to create populations of arbitrary size.
    pop = toolbox.population(n=300)
    # The next thing to do is to evaluate our entire new population.
    # We map() the evaluation function to every individual and then assign their respective fitness. 
    # Note that the order in fitnesses and population is the same.
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Before we go on, this is the time to define some constants we will use later on.
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    # Performing the Evolution
    # The evolution of the population is the final step we have to accomplish. 
    # Recall, our individuals consist of 100 integer numbers and we want to evolve our population until 
    # we got at least one individual consisting of only 1 and no 0. 
    # So all we have to do is to obtain the fitness values of the individuals
    fits = [ind.fitness.values[0] for ind in pop]

    # and evolve our population until one of them reaches 100 or the number of generations reaches 1000.
    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while max(fits) < 100 and g < 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # The evolution itself will be performed by selecting, mating, and mutating the individuals in our population.
        # In our simple example of a genetic algorithm, the first step is to select the next generation.
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # This will creates an offspring list, which is an exact copy of the selected individuals. 
        # The toolbox.clone() method ensure that we don’t use a reference to the individuals 
        # but an completely independent instance. 
        # This is of utter importance since the genetic operators in toolbox will modify the provided objects in-place.

        # Next, we will perform both the crossover (mating) and the mutation of the produced children 
        # with a certain probability of CXPB and MUTPB. The del statement will invalidate the fitness 
        # of the modified offspring.

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # The crossover (or mating) and mutation operators, provided within DEAP, usually take respectively 
        # 2 or 1 individual(s) as input and return 2 or 1 modified individual(s). 
        # In addition they modify those individuals within the toolbox container and we do not need to reassign their results.

        # Since the content of some of our offspring changed during the last step, we now need to re-evaluate 
        # their fitnesses. To save time and resources, we just map those offspring which fitnesses were marked invalid.

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # And last but not least, we replace the old population by the offspring.
        pop[:] = offspring

        # To check the performance of the evolution, we will calculate and print the minimal, maximal, 
        # and mean values of the fitnesses of all individuals in our population as well as their standard deviations.

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

# This evolution will now run until at least one of the individuals will be filled with 1 exclusively.
main()
# A Statistics object is available within DEAP to facilitate the gathering of the evolution’s statistics.
 
    