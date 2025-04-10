import random
import time
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np



class GeneticAlgorithm:
    """Main class for Genetic Algorithm implementation"""

    def __init__(self, target="Hello world!", pop_size=2048, max_iter=16384,
                 elite_rate=0.10, mutation_rate=0.25, crossover_type ="single", use_crossover=True, use_mutation=True, fitness_type="ORIGINAL", lcs_bonus=5):
        # Algorithm parameters
        self.target = target
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.elite_rate = elite_rate
        self.mutation_rate = mutation_rate

        # Added for section 6
        self.use_crossover = use_crossover  # New parameter to enable/disable crossover
        self.use_mutation = use_mutation  # New parameter to enable/disable mutation

        # Section 7
        self.fitness_type = fitness_type  # "ORIGINAL" or "LCS"
        self.lcs_bonus = lcs_bonus  # Bonus for exact matches in LCS (5)

        # Statistics tracking
        self.avg_fitness_history = []
        self.std_dev_history = []
        self.best_fitness_history = []
        self.worst_fitness_history = []
        self.fitness_range_history = []

        # Timing tracking
        self.generation_times = []
        self.generation_cpu_times = []
        self.total_elapsed_times = []

        self.crossover_type = crossover_type  # Can be "single", "two", or "uniform"
        # Validate crossover type
        valid_crossovers = ["SINGLE", "TWO", "UNIFORM"]
        if self.crossover_type not in valid_crossovers:
            print(f"Warning: Invalid crossover type '{crossover_type}'. Defaulting to 'SINGLE'.")
            self.crossover_type = "SINGLE"

        # Adding lists to track selection pressure metrics (section 8)
        self.fitness_variance_history = []
        self.top_avg_ratio_history = []



    class Individual:
        """Represents an individual in the population (equivalent to ga_struct)"""

        def __init__(self, chromosome=""):
            self.chromosome = chromosome  # String representation
            self.fitness = 0  # Fitness value

        def __str__(self):
            return f"{self.chromosome} (Fitness: {self.fitness})"

    def init_population(self):
        """Initialize the population with random individuals"""
        population = []
        buffer = []

        tsize = len(self.target)
        for _ in range(self.pop_size):
            rand_str = "".join(chr(random.randint(32, 32 + 89)) for _ in range(tsize))
            individual = self.Individual(rand_str)
            population.append(individual)

        # Initialize buffer with empty individuals
        for _ in range(self.pop_size):
            buffer.append(self.Individual())

        return population, buffer

    # Section 7 changes (vcreating new methods to choose from)
    # def calc_fitness(self, population):
    #     """Calculate fitness for all individuals in the population"""
    #     tsize = len(self.target)
    #     for individual in population:
    #         fitness_val = 0
    #         for j in range(tsize):
    #             fitness_val += abs(ord(individual.chromosome[j]) - ord(self.target[j]))
    #         individual.fitness = fitness_val
    def original_fitness(self, individual_str, target_str):
        """Original fitness calculation using character-by-character distance"""
        fitness = 0
        for i in range(len(target_str)):
            fitness += abs(ord(individual_str[i]) - ord(target_str[i]))
        return fitness

    def calc_fitness(self, population):
        """Calculate fitness for all individuals using the selected method"""
        for individual in population:
            if self.fitness_type == "ORIGINAL":
                individual.fitness = self.original_fitness(individual.chromosome, self.target)
            elif self.fitness_type == "LCS":
                individual.fitness = self.lcs_fitness(individual.chromosome, self.target, self.lcs_bonus)


    def sort_by_fitness(self, population):
        """Sort the population by fitness (ascending)"""
        population.sort(key=lambda x: x.fitness)

    def elitism(self, population, buffer, esize):
        """Preserve the best individuals"""
        for i in range(esize):
            buffer[i].chromosome = population[i].chromosome
            buffer[i].fitness = population[i].fitness

    def mutate(self, member):
        """Apply mutation to an individual"""
        tsize = len(self.target)
        ipos = random.randint(0, tsize - 1)
        delta = random.randint(32, 32 + 89)

        chars = list(member.chromosome)
        new_val = (ord(chars[ipos]) + delta) % 122
        if new_val < 32:
            new_val = 32
        chars[ipos] = chr(new_val)
        member.chromosome = "".join(chars)

    def mate(self, population, buffer):
        """Mate individuals to create the next generation"""
        esize = int(self.pop_size * self.elite_rate)

        # copy top individuals (elitism)
        self.elitism(population, buffer, esize)

        # mate the rest
        for i in range(esize, self.pop_size):
            # Parent selection
            i1 = random.randint(0, (self.pop_size // 2) - 1)
            i2 = random.randint(0, (self.pop_size // 2) - 1)

            # Get parent chromosomes
            parent1 = population[i1].chromosome
            parent2 = population[i2].chromosome

            if self.use_crossover:
                # Apply crossover based on selected method
                if self.crossover_type == "SINGLE":
                    child = self.single_point_crossover(parent1, parent2)
                elif self.crossover_type == "TWO":
                    child = self.two_point_crossover(parent1, parent2)
                elif self.crossover_type == "UNIFORM":
                    child = self.uniform_crossover(parent1, parent2)
            else:
                # No crossover - just copy one parent (random selection)
                child = parent1 if random.random() < 0.5 else parent2

            # Assign child to buffer
            buffer[i].chromosome = child
            buffer[i].fitness = 0

            # Apply mutation with probability mutation_rate
            if self.use_mutation and random.random() < self.mutation_rate:
                self.mutate(buffer[i])
################################### Section 1 ###################################
    def calc_population_stats(self, population, generation):
        """Calculate and output statistics for the population (Section 1)"""
        best_fitness = population[0].fitness
        worst_fitness = population[-1].fitness
        avg_fitness = sum(ind.fitness for ind in population) / self.pop_size

        variance = sum((ind.fitness - avg_fitness) ** 2 for ind in population) / self.pop_size
        stdev_fitness = math.sqrt(variance)

        diff_best_worst = worst_fitness - best_fitness

        # Store statistics
        self.avg_fitness_history.append(avg_fitness)
        self.std_dev_history.append(stdev_fitness)
        self.best_fitness_history.append(best_fitness)
        self.worst_fitness_history.append(worst_fitness)
        self.fitness_range_history.append(diff_best_worst)

        print(f"Generation: {generation}")
        print(f"  Avg fitness: {avg_fitness:.2f}")
        print(f"  Worst fitness: {worst_fitness}")
        print(f"  Std dev: {stdev_fitness:.2f}")
        print(f"  Range (best..worst): {best_fitness}..{worst_fitness} [diff={diff_best_worst}]")

    ################################### Section 2 ###################################
    def measure_generation_time(self, gen_start_time, gen_start_clock, algorithm_start_time):
        """Measure and output timing information for a generation (Section 2)"""
        current_time = time.time()
        current_clock = time.process_time()

        gen_elapsed = current_time - gen_start_time
        total_elapsed = current_time - algorithm_start_time

        gen_cpu = current_clock - gen_start_clock

        # Store timing data
        self.generation_times.append(gen_elapsed)
        self.generation_cpu_times.append(gen_cpu)
        self.total_elapsed_times.append(total_elapsed)

        print(f"  Timing:")
        print(f"    CPU Clock Ticks: {gen_cpu:.6f} seconds")
        print(f"    Elapsed Time: {gen_elapsed:.6f} seconds")
        print(f"    Total Elapsed: {total_elapsed:.6f} seconds")

        # Format elapsed time in a more readable way
        elapsed_formatted = str(datetime.timedelta(seconds=int(total_elapsed)))
        print(f"    Formatted Total Elapsed: {elapsed_formatted}")

    def print_best(self, population):
        """Print the best individual in the population"""
        best = population[0]
        print(f"Best: {best.chromosome} ({best.fitness})\n")

    def print_summary(self, generations, converged, total_time, total_cpu):
        """Print a summary of the algorithm's performance"""
        if converged:
            print("\nSolution found!")
        else:
            print("\nMaximum iterations reached without finding solution.")

        print(f"Total generations: {generations}")
        print(f"Total elapsed time: {total_time:.6f} seconds")
        print(f"Total CPU time: {total_cpu:.6f} seconds")

        if len(self.generation_times) > 0:
            avg_gen_time = sum(self.generation_times) / len(self.generation_times)
            avg_gen_cpu = sum(self.generation_cpu_times) / len(self.generation_cpu_times)

            print(f"Average generation time: {avg_gen_time:.6f} seconds")
            print(f"Average CPU time per generation: {avg_gen_cpu:.6f} seconds")

    ################################### Section 3 ###################################
    def plot_fitness_history(self):
        """Create and save a line graph showing fitness evolution (best, avg, worst)"""
        plt.figure(figsize=(12, 6))

        generations = range(len(self.avg_fitness_history))

        # Plot the three lines
        plt.plot(generations, self.best_fitness_history, 'g-', label='Best Fitness')
        plt.plot(generations, self.avg_fitness_history, 'b-', label='Average Fitness')
        plt.plot(generations, self.worst_fitness_history, 'r-', label='Worst Fitness')

        plt.title('Fitness Evolution Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.savefig('fitness_evolution.png')
        plt.show()

    def plot_fitness_boxplot(self):
        """Create and save box plots showing fitness distribution by generation"""
        # For a true boxplot, we'd need all fitness values for each generation
        # Since we only have summary statistics, we'll create a simplified version

        # Use only every Nth generation to make the plot readable
        step = max(1, len(self.avg_fitness_history) // 10)
        selected_gens = range(0, len(self.avg_fitness_history), step)

        # Create data for boxplot - approximate from our statistics
        # For each generation, create an array representing the distribution
        boxplot_data = []
        labels = []

        for i in selected_gens:
            mean = self.avg_fitness_history[i]
            std_dev = self.std_dev_history[i]
            min_val = self.best_fitness_history[i]
            max_val = self.worst_fitness_history[i]

            # Create a synthetic distribution based on our statistics
            # This is an approximation for visualization purposes
            q1 = max(min_val, mean - std_dev)
            q3 = min(max_val, mean + std_dev)

            # Store the five-number summary (min, Q1, median, Q3, max)
            boxplot_data.append([min_val, q1, mean, q3, max_val])
            labels.append(f"Gen {i}")

        plt.figure(figsize=(12, 6))

        # Create a boxplot from our data
        plt.boxplot(boxplot_data, labels=labels, showmeans=True)

        plt.title('Fitness Distribution by Generation')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)

        # Save the figure
        plt.savefig('fitness_boxplot.png')
        plt.show()

    ################################### Section 4 ###################################
    # Also changed __init__ to include a crossover_type parameter, and the mate method to use the selected crossover type
    def single_point_crossover(self, parent1, parent2):
        """Perform single point crossover between two parents"""
        tsize = len(self.target)
        spos = random.randint(0, tsize - 1)
        return parent1[:spos] + parent2[spos:]

    def two_point_crossover(self, parent1, parent2):
        """Perform two point crossover between two parents"""
        tsize = len(self.target)

        # Ensure point1 < point2
        point1 = random.randint(0, tsize - 2)
        point2 = random.randint(point1 + 1, tsize - 1)

        # Take beginning from parent1, middle from parent2, end from parent1
        return parent1[:point1] + parent2[point1:point2] + parent1[point2:]

    def uniform_crossover(self, parent1, parent2):
        """Perform uniform crossover between two parents"""
        tsize = len(self.target)
        child = ""

        # For each position, randomly choose from either parent
        for i in range(tsize):
            if random.random() < 0.5:
                child += parent1[i]
            else:
                child += parent2[i]

        return child

    ################################### Section 7 ###################################
    def lcs_fitness(self, individual_str, target_str, bonus_for_match=5):
        """
        individual_str: The individual's chromosome, target_str: The target string, bonus_for_match: Bonus points for characters in correct positions
        Returns: Fitness score
        """
        m = len(individual_str)
        n = len(target_str)
        # Create LCS table using dynamic programming
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # Fill the dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if individual_str[i - 1] == target_str[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # LCS length
        lcs_length = dp[m][n]

        # Calculate exact matches bonus
        exact_matches = 0
        for i in range(m):
            if i < n and individual_str[i] == target_str[i]:
                exact_matches += 1

        # The fitness is: max possible score - (LCS length + bonus * exact matches)
        # We subtract from the max possible score to make it a minimization problem
        max_possible_score = len(target_str) + bonus_for_match * len(target_str)
        fitness = max_possible_score - (lcs_length + bonus_for_match * exact_matches)

        return fitness

    ################################### Section 8 ###################################
    def calculate_selection_probabilities(self, population):
        """Calculate selection probabilities for each individual using fitness-proportionate selection"""
        # Get all fitness values
        fitness_values = [ind.fitness for ind in population]

        # For minimization problems (like ours), we need to transform fitness values
        # We use max_fitness + 1 - fitness so higher values are better
        if len(set(fitness_values)) == 1:  # All fitnesses are identical
            return [1.0 / len(population)] * len(population)

        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)

        # Adjust the transformation to create more distinction
        # If max == min, this prevents division by zero
        if max_fitness == min_fitness:
            transformed_fitness = [1.0] * len(fitness_values)
        else:
            # Use a different transformation that creates more distinction
            transformed_fitness = [(max_fitness + 1 - fitness) / (max_fitness - min_fitness + 1)
                                   for fitness in fitness_values]

        # Calculate total fitness
        total_fitness = sum(transformed_fitness)

        # Calculate selection probabilities
        selection_probs = [tf / total_fitness for tf in transformed_fitness]
        return selection_probs

    def calculate_fitness_variance(self, population):
        """Calculate variance of selection probabilities (Fitness Variance)"""
        selection_probs = self.calculate_selection_probabilities(population)

        # Mean probability and Variance calculations
        mean_prob = sum(selection_probs) / len(selection_probs)
        variance = sum((p - mean_prob) ** 2 for p in selection_probs) / len(selection_probs)

        return variance

    def calculate_top_avg_ratio(self, population, top_fraction=0.1):
        """Calculate Top-Average Selection Probability Ratio"""
        selection_probs = self.calculate_selection_probabilities(population)

        # Sort selection probabilities (higher is better)
        sorted_probs = sorted(selection_probs, reverse=True)

        # Calculate number of top individuals
        top_count = max(1, int(len(population) * top_fraction))

        # Calculate average probability for top individuals
        top_avg = sum(sorted_probs[:top_count]) / top_count

        # Calculate overall average
        overall_avg = sum(selection_probs) / len(selection_probs)

        # Calculate ratio (and also handle the division by zero)
        if overall_avg == 0:
            return 1.0

        return top_avg / overall_avg

    # Adding plotting methods to visualize selection pressure
    def plot_selection_pressure(self):
        """Plot selection pressure metrics over generations and calling this in the end of the run function"""
        plt.figure(figsize=(12, 6))

        generations = range(len(self.fitness_variance_history))

        # Plot the two metrics
        plt.subplot(1, 2, 1)
        plt.plot(generations, self.fitness_variance_history, 'b-')
        plt.title('Fitness Variance Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Variance of Selection Probabilities')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(generations, self.top_avg_ratio_history, 'r-')
        plt.title('Top-Average Ratio Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Top/Avg Selection Probability Ratio')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('selection_pressure.png')
        plt.show()

    def run(self):
        """Run the genetic algorithm"""
        # Initialize populations
        population, buffer = self.init_population()

        # Start timing for the entire algorithm
        algorithm_start_time = time.time()
        algorithm_start_clock = time.process_time()

        # Run the algorithm
        for i in range(self.max_iter):
            # Start timing for this generation
            gen_start_time = time.time()
            gen_start_clock = time.process_time()

            # Calculate fitness and sort
            self.calc_fitness(population)
            self.sort_by_fitness(population)

            # Calculate and print statistics (Section 1)
            self.calc_population_stats(population, i)

            # Measure and print timing (Section 2)
            self.measure_generation_time(gen_start_time, gen_start_clock, algorithm_start_time)

            # Adding for section 8, Calculate and store selection pressure metrics
            fitness_variance = self.calculate_fitness_variance(population)
            top_avg_ratio = self.calculate_top_avg_ratio(population)
            self.fitness_variance_history.append(fitness_variance)
            self.top_avg_ratio_history.append(top_avg_ratio)
            print(f"  Selection Pressure:")
            print(f"    Fitness Variance: {fitness_variance:.10f}")
            print(f"    Top-Avg Ratio: {top_avg_ratio:.6f}")


            # Print best individual
            self.print_best(population)

            # Check for convergence
            if population[0].fitness == 0:
                total_time = time.time() - algorithm_start_time
                total_cpu = time.process_time() - algorithm_start_clock
                self.print_summary(i + 1, True, total_time, total_cpu)

                # Generate plots (added for Section 3)
                self.plot_fitness_history()
                self.plot_fitness_boxplot()

                self.plot_selection_pressure()

                return population[0], i + 1  # Return best individual and generations


            # Create next generation
            self.mate(population, buffer)
            population, buffer = buffer, population

        # If no convergence achieved
        total_time = time.time() - algorithm_start_time
        total_cpu = time.process_time() - algorithm_start_clock
        self.print_summary(self.max_iter, False, total_time, total_cpu)

        # Always generate plots(added for Section 3)
        self.plot_fitness_history()
        self.plot_fitness_boxplot()

        # Generate plots section 8
        self.plot_selection_pressure()


        return population[0], self.max_iter  # Return best individual and max generations


# Main execution
def main():
    """Main function"""
    # Set random seed
    random.seed(int(time.time()))

    # Create and run the genetic algorithm
    #ga = GeneticAlgorithm(crossover_type = "SINGLE") # Choose one crossover type to use Can be "SINGLE", "TWO", or "UNIFORM"

    # Added for section 6
    # Configuration 1: Crossover Only
    #ga = GeneticAlgorithm(crossover_type="SINGLE", use_crossover=True, use_mutation=False)

    # Configuration 2: Mutation Only
    #ga = GeneticAlgorithm(crossover_type="SINGLE", use_crossover=False, use_mutation=True)

    # Configuration 3: Both Crossover and Mutation
    #ga = GeneticAlgorithm(crossover_type="SINGLE", use_crossover=True, use_mutation=True)

    ga = GeneticAlgorithm(crossover_type="SINGLE", use_crossover=True, use_mutation=True, fitness_type="ORIGINAL")
    #ga = GeneticAlgorithm(crossover_type="TWO", use_crossover=True, use_mutation=True, fitness_type="LCS", lcs_bonus=5)

    best_solution, generations = ga.run()

    print(f"\nFinal solution: {best_solution.chromosome} with fitness {best_solution.fitness}")
    print(f"Found in {generations} generations")
    return 0


# Entry point
if __name__ == "__main__":
    main()