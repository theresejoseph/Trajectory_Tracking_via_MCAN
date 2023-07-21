import random
import numpy as np
from CAN import attractorNetwork2D, attractorNetwork, activityDecoding,activityDecodingAngle,headDirectionAndPlaceNoWrapNet
import matplotlib.pyplot as plt
import math
import multiprocessing
from multiprocessing import freeze_support
from functools import partial
import logging
plt.style.use(['science', 'no-latex'])


def headDirectionFitness(angVel,genome):
    N=360
    num_links=int(genome[0]) #int
    excite=int(genome[1]) #int
    activity_mag=genome[2] #uni
    inhibit_scale=genome[3] #uni
    iterations=int(genome[4])


    theta_weights=np.zeros(N)
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    theta_weights[net.activation(0)]=net.full_weights(num_links)

    output=[0]
    expected=[0]
    # angVel=np.random.uniform(-45,45, 400)
    for i in range(len(angVel)):
        for j in range(iterations):
            theta_weights=net.update_weights_dynamics(theta_weights,angVel[i])
            theta_weights[theta_weights<0]=0
    
        output.append(activityDecodingAngle(theta_weights,10,N))
        expected.append((expected[-1]+angVel[i])%N)
    
    return (np.sum(abs(np.array(expected)-np.array(output))))*-1


def attractorGridcell_fitness(vel, angVel, genome):
    N=100
    num_links=int(genome[0]) #int
    excite=int(genome[1]) #int
    activity_mag=genome[2] #uni
    inhibit_scale=genome[3] #uni
    iterations=int(genome[4])

    prev_weights=np.zeros((N,N))
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
    prev_weights=network.excitations(0,0)
    x,y=0,0
    x_integ, y_integ=[],[]
    x_grid, y_grid=[], []

    x_grid_expect,y_grid_expect=0,0
    for i in range(len(vel)):
        for j in range(iterations):
            prev_weights,wrap_rows, wrap_cols=network.update_weights_dynamics(prev_weights, vel[i], angVel[i])
            
            x_grid_expect+=wrap_cols*N
            y_grid_expect+=wrap_rows*N

        
        #grid cell output 
        maxXPerScale, maxYPerScale = np.argmax(np.max(prev_weights, axis=1)),np.argmax(np.max(prev_weights, axis=0))
        x_grid.append(activityDecoding(prev_weights[maxXPerScale,:],5,N)+x_grid_expect)
        y_grid.append(activityDecoding(prev_weights[:,maxYPerScale],5,N)+y_grid_expect)

      
        #integrated output
        x,y=x+(vel[i]*np.cos(np.deg2rad(angVel[i]))), y+(vel[i]*np.sin(np.deg2rad(angVel[i])))
        x_integ.append(x)
        y_integ.append(y)


    x_error=np.sum(np.abs(np.array(x_grid) - np.array(x_integ)))
    y_error=np.sum(np.abs(np.array(y_grid) - np.array(y_integ)))

    return (x_error+y_error)*-1


class GeneticAlgorithm:
    def __init__(self,num_gens,population_size,filename,fitnessFunc, ranges,scales, angVels, vels, dim,numProcess):
        self.num_gens=num_gens
        self.population_size=population_size
        self.filename=filename
        self.fitnessFunc=fitnessFunc
        self.ranges=ranges
        self.scales=scales
        self.angVels=angVels 
        self.vels=vels 
        self.numProcess=numProcess
        self.dim=dim

    def rand(self,range_idx,intOruni):
        # return random integer or float from uniform distribution  within the allowed range of each parameter 
        if intOruni=='int':
            return random.randint(self.ranges[range_idx][0],self.ranges[range_idx][1])
        elif intOruni=='uni':
            return random.uniform(self.ranges[range_idx][0],self.ranges[range_idx][1])

    def initlisePopulation(self,numRandGenomes):
        population=[]
        for i in range(numRandGenomes):
            # genome=[self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int'), self.rand(5,'int'),self.rand(6,'uni'),self.rand(7,'uni')]
            genome=[self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int')] #no wrap iters
            # genome=[self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int'), self.rand(5,'int')] #2d 
            population.append(genome)
        return population 

    def mutate(self, genome):
        # if random value is greater than 1-probabilty, then mutate the gene
        # if no genes are mutated then require one (pick randomly)
        # amount of mutation = value + gaussian (with varience)
        if self.dim=='1D':
            mutate_amount=np.array([int(np.random.uniform(0,1)), int(np.random.uniform(0,1)), np.random.uniform(0,0.05), np.random.uniform(0,0.05), int(np.random.uniform(0,1))]) #head direction
        elif self.dim=='2D':
            mutate_amount=np.array([int(np.random.uniform(0,1)), int(np.random.uniform(0,1)), np.random.uniform(0,0.01), np.random.uniform(0,0.001), int(np.random.uniform(0,1))]) #atractor grid cell 

        mutate_prob=np.array([random.random() for i in range(len(genome))])
        mutate_indexs=np.argwhere(mutate_prob<=0.2)
        
        new_genome=np.array(genome)
        new_genome[mutate_indexs]+=mutate_amount[mutate_indexs]
        return new_genome 
    
    def checkMutation(self,genome):
        # check mutated genome exists within the specified range 
        g=self.mutate(genome)
        while([self.ranges[i][0] <= g[i] <= self.ranges[i][1] for i in range(len(genome))]!=[True]*len(genome)):
            g=self.mutate(genome)
        return g
    
    def process_fitness(self, i, population):
        # fit=self.fitnessFunc(self.vels, self.angVels, genome=population[i])
        # return fit
        try:
            if self.dim=='1D':
                fit=self.fitnessFunc(self.angVels, population[i])
            elif self.dim=='2D':
                fit=self.fitnessFunc(self.vels, self.angVels, genome=population[i])
            
            return fit
        except Exception as e:
            return -10000000000

    def sortByFitness(self,population,topK):
        # fitness for each genome 
        # sort genomes by fitness
        fitness=np.zeros(len(population))

        with multiprocessing.Pool(processes=self.numProcess) as pool:
            fitness = pool.map(partial(self.process_fitness, population=population), range(len(population)))
        
        fitness=np.array(fitness)
        idxs = np.argsort(fitness)[::-1]
        return fitness[idxs[:topK]],idxs[:topK]
    
    def selection(self,population):
        # parent = take the best 5 (add to new population)
        # for every parent  make 3 children and add to new population 
        num_parents=self.population_size//4
        num_children_perParent=(self.population_size-num_parents)//num_parents

        fitnesses,indexes=self.sortByFitness(population,num_parents)
        print('Finsihed Checking Fitness of old population, now mutating parents to make a new generation')

        '''Keep the fittest genomes as parents'''
        new_population=[population[indexes[i]] for i in range(num_parents)] #parents are added to the new population 

        '''Make 15 Children from the fittest parents'''
        for i in range(num_parents):
            for j in range(num_children_perParent):
                new_genome_inrange=self.checkMutation(population[indexes[i]])
                new_population.append(new_genome_inrange)
        return new_population
    
    def implimentGA(self):
        population=self.initlisePopulation(self.population_size)
        print(population)
        fitnesses=np.zeros((self.population_size,1))
        order_population=np.zeros((self.num_gens,self.population_size,len(population[0])+1))

        '''iterate through generations'''
        for i in range(self.num_gens):
            print(f'Current Generation {i}')
            population=np.array(self.selection(population))
            print('Finsihed making new populaiton  through mutation, now evaluting fitness and sorting')
            fitnesses,indexes=self.sortByFitness(population,self.population_size)
            order_population[i,:,:] = np.hstack((np.array(population[indexes]), fitnesses[:,None]))

            '''Stop the GA if fitness hasnt improved for <stop_val> generations'''
            current_fitnesses=[max(fit) for fit in np.array(order_population)[:,:,-1]]
            stop_val=self.num_gens
            if i>=stop_val and all(element == current_fitnesses[i] for element in current_fitnesses[i-stop_val:i]):
                break
            print(fitnesses)

            with open(self.filename, 'wb') as f:
                np.save(f, np.array(order_population))



def runGA(run1D=False, run2D=False, plotting1D=False, plotting2D=False):
    if run2D==True:
        filename=f'./Results/GA_Experiment_Output/TuningMultiscale.npy'
        # mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.005), np.random.normal(0,0.00005), int(np.random.normal(0,1)), int(np.random.normal(0,1))])
        # ranges = [[1,20],[1,20],[0,1],[0.00001, 0.005],[1,5], [1,5]]
        ranges = [[1,10],[1,10],[0,1],[0,0.005],[1,2]]
        fitnessFunc=  attractorGridcell_fitness
        num_gens=20
        population_size=8
        scales=[0.25,1,4,16]
        test_length=50
        # np.random.seed(5)
        vels=np.random.uniform(0,5,test_length)
        dirs=np.arange(0,360, 360//test_length)
        GeneticAlgorithm(num_gens,population_size,filename,fitnessFunc,ranges,scales, dirs, vels, dim='2D',numProcess=8).implimentGA()

    if plotting2D==True:
        filename=f'./Results/GA_Experiment_Output/TuningMultiscale.npy'
        savePath=f'./Results/GA_Experiment_Output/GA_multiscale_Plot.png'
        with open(filename, 'rb') as f:
            data = np.load(f)
        mean_1=np.array([np.mean(fit) for fit in data[:,:,-1]])
        std_1=np.array([np.std(fit) for fit in data[:,:,-1]])

        x = np.arange(len(mean_1))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, mean_1, color='teal', label='2D Network Tuning with Genetic Algorithm')
        ax.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='teal', alpha=0.2)
        ax.set_xlabel('Generation')    
        ax.set_ylabel('Fitness [-SAD]')
        ax.set_title('2D Network Tuning')
        plt.tight_layout()
        plt.savefig(savePath)
        print(data[-1,0,:])


    if run1D==True:
        filename=f'./Results/RandomData/GA_Experiment_Output/TuningHeadDirection.npy'
        # mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.005), np.random.normal(0,0.00005), int(np.random.normal(0,1)), int(np.random.normal(0,1))])
        ranges =  [[1,20],[1,20],[0.05,4],[0,0.1],[1,2]]
        fitnessFunc=  headDirectionFitness
        num_gens=20
        population_size=12
        scales=[0.25,1,4,16]
        test_length=200
        # np.random.seed(5)
        vels=np.random.uniform(0,5,test_length)
        angVels=np.random.uniform(-45,45, test_length)

        GeneticAlgorithm(num_gens,population_size,filename,fitnessFunc,ranges,scales, angVels, vels, dim='1D', numProcess=4).implimentGA()

    if plotting1D==True:
        filename=f'./Results/GA_Experiment_Output/TuningHeadDirection.npy'
        savePath=f'./Results/GA_Experiment_Output/GA_HeadDirection_Plot.png'
        with open(filename, 'rb') as f:
            data = np.load(f)
        mean_1=np.array([np.mean(fit) for fit in data[:,:,-1]])
        std_1=np.array([np.std(fit) for fit in data[:,:,-1]])

        x = np.arange(len(mean_1))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, mean_1, color='g', label='1D Network Tuning with Genetic Algorithm')
        ax.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='g', alpha=0.2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness [-SAD]')
        ax.set_title('1D Network Tuning')
        plt.tight_layout()
        plt.savefig(savePath)
        print(data[-1,0,:])
        

def plotAllGA():
    filename=f'./Results/GA_Experiment_Output/TuningHeadDirection.npy'
    with open(filename, 'rb') as f:
        data = np.load(f)
    mean_1=np.array([np.mean(fit) for fit in data[:,:,-1]])
    std_1=np.array([np.std(fit) for fit in data[:,:,-1]])

    filename2=f'./Results/GA_Experiment_Output/TuningMultiscale.npy'
    with open(filename2, 'rb') as f:
        data = np.load(f)
        mean_2=np.array([np.mean(fit) for fit in data[:,:,-1]])
        std_2=np.array([np.std(fit) for fit in data[:,:,-1]])

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(4, 2),sharey='row')
    plt.tight_layout()
    x1 = np.arange(len(mean_1))
    ax1.plot(x1, mean_1, 'g-')
    ax1.fill_between(x1, mean_1 - std_1, mean_1 + std_1, color='g', alpha=0.2)
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Fitness [-SAD]')
    ax1.set_title('1D Network Tuning')

    x2 = np.arange(len(mean_2))
    ax2.plot(x2, mean_2, color='teal')
    ax2.fill_between(x2, mean_2 - std_2, mean_2 + std_2, color='teal', alpha=0.2)
    ax2.set_xlabel('Generations')
    ax2.set_title('2D Network Tuning')
    plt.savefig('./Results/GA_Experiment_Output/1D_2D_NetworkTuning.png')


if __name__ == '__main__':
    freeze_support()
    plotAllGA()
    runGA(run2D= False, plotting2D=True)
    runGA(run1D= False, plotting1D=True)


