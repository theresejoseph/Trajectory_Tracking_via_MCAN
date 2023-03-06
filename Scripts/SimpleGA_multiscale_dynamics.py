import random
from cv2 import Algorithm
import numpy as np
import sys
sys.path.append('../scripts')


from CAN import attractorNetwork2D, attractorNetwork
import CAN as can
# from TwoModesofMultiscale import scale_selection, hierarchicalNetwork
# from SelectiveMultiScalewithWraparound2D import  hierarchicalNetwork2D, hierarchicalNetwork2DGrid
# from DataHandling import saveOrLoadNp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import math
import multiprocessing
from multiprocessing import freeze_support
from functools import partial
import logging
# import scienceplots
# plt.rcParams.update(plt.rcParamsDefault)
plt.style.use(['science', 'no-latex'])
# plt.rcParams['text.usetex'] = True
# print(plt.style.available)

'''GA Fitness Functions'''

# head direction 
def headDirectionFitness(genome):
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
    angVel=np.random.uniform(-45,45, 400)
    for i in range(len(angVel)):
        for j in range(iterations):
            theta_weights=net.update_weights_dynamics(theta_weights,angVel[i])
            theta_weights[theta_weights<0]=0
    
        output.append(can.activityDecodingAngle(theta_weights,10,N))
        expected.append((expected[-1]+angVel[i])%N)
    
    return (np.sum(abs(np.array(expected)-np.array(output))))*-1

#grid cell 
def attractorGridcell_fitness(genome):
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
    dirs=np.arange(0,360)
    speeds=np.random.uniform(0,5,360)
    x_integ, y_integ=[],[]
    x_grid, y_grid=[], []

    x_grid_expect,y_grid_expect=0,0
    for i in range(len(speeds)):
        for j in range(iterations):
            prev_weights,wrap_rows, wrap_cols=network.update_weights_dynamics(prev_weights, dirs[i], speeds[i])
            
            x_grid_expect+=wrap_cols*N
            y_grid_expect+=wrap_rows*N
        
        #grid cell output 
        maxXPerScale, maxYPerScale = np.argmax(np.max(prev_weights, axis=1)),np.argmax(np.max(prev_weights, axis=0))
        x_grid.append(can.activityDecoding(prev_weights[maxXPerScale,:],5,N)+x_grid_expect)
        y_grid.append(can.activityDecoding(prev_weights[:,maxYPerScale],5,N)+y_grid_expect)

      
        #integrated output
        x,y=x+(speeds[i]*np.cos(np.deg2rad(dirs[i]))), y+(speeds[i]*np.sin(np.deg2rad(dirs[i])))
        x_integ.append(x)
        y_integ.append(y)

    # plt.plot(x_integ,y_integ,'g.')
    # plt.plot(x_grid,y_grid,'m.')
    # plt.show()
    x_error=np.sum(np.abs(np.array(x_grid) - np.array(x_integ)))
    y_error=np.sum(np.abs(np.array(y_grid) - np.array(y_integ)))


    return (x_error+y_error)*-1


def scale_selection(input,scales):
    swap_val=1
    if input<=scales[0]*swap_val:
        scale_idx=0
    elif input>scales[0]*swap_val and input<=scales[1]*swap_val:
        scale_idx=1
    elif input>scales[1]*swap_val and input<=scales[2]*swap_val:
        scale_idx=2
    elif input>scales[2]*swap_val and input<=scales[3]*swap_val:
        scale_idx=3
    elif input>scales[3]*swap_val:
        scale_idx=4
    return scale_idx

def headDirection(theta_weights, angVel, init_angle):
    global theata_called_iters
    N=360
    num_links,excite,activity_mag,inhibit_scale, iterations=16, 17, 2.26818183,  0.0281834545, 2
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    
    if theata_called_iters==0:
        theta_weights[net.activation(init_angle)]=net.full_weights(num_links)
        theata_called_iters+=1
    for j in range(iterations):
        theta_weights=net.update_weights_dynamics(theta_weights,angVel)
        theta_weights[theta_weights<0]=0

    # plt.bar(np.arange(N), theta_weights)
    # plt.show()

    return theta_weights


def hierarchicalNetwork2DGridNew(prev_weights, net,N, vel, direction, iterations, wrap_iterations, x_grid_expect, y_grid_expect,scales):
    '''Select scale and initilise wrap storage'''
    delta = [(vel/scales[0]), (vel/scales[1]), (vel/scales[2]), (vel/scales[3]), (vel/scales[4])]
    cs_idx=scale_selection(vel,scales)
    wrap_rows=np.zeros((len(scales)))
    wrap_cols=np.zeros((len(scales)))

    '''Update selected scale'''
    del_x_cs, del_y_cs= delta[cs_idx]*np.cos(np.deg2rad(direction)), delta[cs_idx]*np.sin(np.deg2rad(direction))
    x_grid_expect[cs_idx]=(x_grid_expect[cs_idx]+(del_x_cs *scales[cs_idx]))%(N*scales[cs_idx])
    y_grid_expect[cs_idx]=(y_grid_expect[cs_idx]+(del_y_cs *scales[cs_idx]))%(N*scales[cs_idx])
    for i in range(iterations):
        prev_weights[cs_idx][:], wrap_rows_cs, wrap_cols_cs= net.update_weights_dynamics(prev_weights[cs_idx][:],direction, delta[cs_idx])
        prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
        wrap_rows[cs_idx]+=wrap_rows_cs
        wrap_cols[cs_idx]+=wrap_cols_cs
    
    '''Update the 16 scale based on wraparound in 0.25 scale'''
    if (cs_idx==0 and (wrap_rows[cs_idx]!=0 or wrap_cols[cs_idx]!=0 )):
        del_rows_100, del_cols_100=(wrap_rows[cs_idx]*scales[cs_idx]*N)/scales[3], (wrap_cols[cs_idx]*scales[cs_idx]*N)/scales[3]  
        direction_100=np.rad2deg(math.atan2(del_rows_100, del_cols_100))
        distance_100=math.sqrt(del_cols_100**2 + del_rows_100**2)

        x_grid_expect[3]=(x_grid_expect[3]+(del_cols_100 *scales[3]))%(N*scales[3])
        y_grid_expect[3]=(y_grid_expect[3]+(del_rows_100 *scales[3]))%(N*scales[3])
        # wraparound[4]=(can.activityDecoding(prev_weights[4][:],4,N) + update_amount)//(N-1)
        for i in range(wrap_iterations):
            prev_weights[3][:], wrap_rows_100, wrap_cols_100= net.update_weights_dynamics(prev_weights[3][:],direction_100, distance_100)
            prev_weights[3][prev_weights[3][:]<0]=0
            wrap_rows[3]+=wrap_rows_100
            wrap_cols[3]+=wrap_cols_100

    '''Update the 100 scale based on wraparound in any of the previous scales'''
    if (cs_idx!=0 and (wrap_rows[cs_idx]!=0 or wrap_cols[cs_idx]!=0 )): 
        # tunedParms=3,5,1,0.000865888565,1
        # net100=attractorNetwork2D(N,N,tunedParms[0],tunedParms[1], tunedParms[2],tunedParms[3])
        del_rows_100, del_cols_100=(wrap_rows[cs_idx]*scales[cs_idx]*N)/scales[4], (wrap_cols[cs_idx]*scales[cs_idx]*N)/scales[4]  
        direction_100=np.rad2deg(math.atan2(del_rows_100, del_cols_100))
        distance_100=math.sqrt(del_cols_100**2 + del_rows_100**2)

        x_grid_expect[4]=(x_grid_expect[4]+(del_cols_100 *scales[4]))%(N*scales[4])
        y_grid_expect[4]=(y_grid_expect[4]+(del_rows_100 *scales[4]))%(N*scales[4])
        # wraparound[4]=(can.activityDecoding(prev_weights[4][:],4,N) + update_amount)//(N-1)
        for i in range(wrap_iterations):
            prev_weights[4][:], wrap_rows_100, wrap_cols_100= net.update_weights_dynamics(prev_weights[4][:],direction_100, distance_100)
            prev_weights[4][prev_weights[4][:]<0]=0
            wrap_rows[4]+=wrap_rows_100
            wrap_cols[4]+=wrap_cols_100

    '''Update the 10000 scale based on wraparound in the 100 scale'''
    if (wrap_rows[-2]!=0 or wrap_cols[-2]!=0 ):
        del_rows_10000, del_cols_10000=(wrap_rows[-2]*scales[-2]*N)/scales[5], (wrap_cols[-2]*scales[-2]*N)/scales[5]  
        direction_10000=np.rad2deg(math.atan2(del_rows_10000, del_cols_10000))
        distance_10000=math.sqrt(del_cols_10000**2 + del_rows_10000**2)

        for i in range(wrap_iterations):
            prev_weights[-1][:], wrap_rows[-1], wrap_cols[-1]= net.update_weights_dynamics(prev_weights[-1][:],direction_10000, distance_10000)
            prev_weights[-1][prev_weights[-1][:]<0]=0
        
    wrap=0
    # if np.any(wrap_cols!=0):
    #     wrap=1
    #     print(f"------------------------------------------------------------------------------------------------------wrap_cols {wrap_cols}")
    # if np.any(wrap_rows!=0):
    #     wrap=1
    #     print(f"------------------------------------------------------------------------------------------------------wrap_rows {wrap_rows}")

       
    return prev_weights, wrap, x_grid_expect, y_grid_expect

def headDirectionAndPlaceNew(genome):
    global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, x_grid_expect, y_grid_expect 

    num_links=int(genome[0]) #int
    excite=int(genome[1]) #int
    activity_mag=genome[2] #uni
    inhibit_scale=genome[3] #uni
    iterations=int(genome[4])
    wrap_iterations=int(genome[5])
    N=100
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)

    '''__________________________Load stored velocities______________________________'''
    index=16
    outfile=f'../results/TestEnvironmentFiles/TraverseInfo/BerlineEnvPath{index}.npz'
    traverseInfo=np.load(outfile, allow_pickle=True)
    vel,angVel=traverseInfo['speeds'], traverseInfo['angVel']


    '''__________________________Storage and initilisation parameters______________________________'''
    scales=[0.25,1,4,16,100,10000]
    theta_weights=np.zeros(360)
    theata_called_iters=0
    start_x, start_y=(50*scales[3])+(50*scales[4])+(50*scales[5]),(50*scales[3])+(50*scales[4])+(50*scales[5])
    wrap_counter=[0,0,0,0,0,0]
    x_grid, y_grid=[], []
    x_grid_expect, y_grid_expect =[0,0,0,50*scales[3],50*scales[4],50*scales[5]],[0,0,0,50*scales[3],50*scales[4],50*scales[5]]
    x_integ, y_integ=[],[]
    q=[start_x,start_y,0]


    '''__________________________Initilising scales in the center and at the edge_____________________________'''
    prev_weights=[np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)),np.zeros((N,N)), np.zeros((N,N))]
    for n in range(2):
        prev_weights[n]=network.excitations(0,0)
        prev_weights[n]=network.update_weights_dynamics_row_col(prev_weights[n][:], 0, 0)
        prev_weights[n][prev_weights[n][:]<0]=0
    
    for start_idx in range(3,6):
        prev_weights[start_idx]=network.excitations(50,50)
        prev_weights[start_idx][:]= network.update_weights_dynamics_row_col(prev_weights[start_idx][:],0,0)
        prev_weights[start_idx][prev_weights[start_idx][:]<0]=0


    '''_______________________________Iterating through simulation velocities_______________________________'''
    
    for i in range(300):   
        N_dir=360
        theta_weights=headDirection(theta_weights, np.rad2deg(angVel[i]), 0)
        direction=np.argmax(theta_weights)
        # hD_x,hD_y=(theta_weights*np.cos(np.deg2rad(np.arange(N_dir)*360/N)))[::3], (theta_weights*np.sin(np.deg2rad(np.arange(N_dir)*360/N)))[::3]
        prev_weights, wrap, x_grid_expect, y_grid_expect= hierarchicalNetwork2DGridNew(prev_weights, network, N, vel[i], direction, iterations,wrap_iterations, x_grid_expect, y_grid_expect, scales)

        '''1D method for decoding'''
        maxXPerScale, maxYPerScale = np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))]), np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])
        decodedXPerScale=[can.activityDecoding(prev_weights[m][maxXPerScale[m], :],5,N)*scales[m] for m in range(len(scales))]
        decodedYPerScale=[can.activityDecoding(prev_weights[m][:,maxYPerScale[m]],5,N)*scales[m] for m in range(len(scales))]
        x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale), np.sum(decodedYPerScale)
        # print(f'decoded: {decodedXPerScale}, {decodedYPerScale}')
        # print(f'expected: {x_grid_expect}, {y_grid_expect}')
        x_grid.append(x_multiscale_grid-start_x)
        y_grid.append(y_multiscale_grid-start_y)

        '''Path integration'''
        q[2]+=angVel[i]
        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        x_integ.append(q[0]-start_x)
        y_integ.append(q[1]-start_y)
    

    '''_______________________________Return error_______________________________'''
    x_error=np.sum(np.abs(np.array(x_grid) - np.array(x_integ)))
    y_error=np.sum(np.abs(np.array(y_grid) - np.array(y_integ)))
    return (x_error+y_error)*-1  

'''Implementation'''
class GeneticAlgorithm:
    def __init__(self,num_gens,population_size,filename,fitnessFunc, ranges):
        self.num_gens=num_gens
        self.population_size=population_size
        self.filename=filename
        self.fitnessFunc=fitnessFunc
        self.ranges=ranges
        # self.mutate_amount=mutate_amount

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
            # genome=[self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int'), self.rand(5,'int'), \
            # self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int'), self.rand(5,'int'), \
            # self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int'), self.rand(5,'int'), \
            # self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int'), self.rand(5,'int'), \
            # self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int'), self.rand(5,'int'), \
            # self.rand(0,'int'), self.rand(1,'int'),self.rand(2,'uni'),self.rand(3,'uni'), self.rand(4,'int'), self.rand(5,'int')] #head direction and grid cell 
            population.append(genome)
        return population 

    def mutate(self, genome):
        # if random value is greater than 1-probabilty, then mutate the gene
        # if no genes are mutated then require one (pick randomly)
        # amount of mutation = value + gaussian (with varience)
        mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.1), np.random.normal(0,0.0005), int(np.random.normal(0,1))])
    
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
    
    def process_element(self, i, population):
        # return self.fitnessFunc(population[i])
        try:
            fit=self.fitnessFunc(population[i])
            print(i, fit)
            return fit
        except Exception as e:
            return -10000000000


    def sortByFitness(self,population,topK):
        # fitness for each genome 
        # sort genomes by fitness
        fitness=np.zeros(len(population))

        with multiprocessing.Pool(processes=14) as pool:
            fitness = pool.map(partial(self.process_element, population=population), range(len(population)))
        
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
        # new_population+=self.initlisePopulation(num_parents-1)
        # '''Add 5 random genomes into the population'''
        # new_population=self.initlisePopulation(num_parents)

        '''Make 15 Children from the fittest parents'''
        for i in range(num_parents):
            for j in range(num_children_perParent):
                new_genome_inrange=self.checkMutation(population[indexes[i]])
                # print(population[indexes[i]], new_genome_inrange)
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

def runGA1D(plot=False):
    #[num_links, excitation width, activity magnitude,inhibition scale]
    # filename=f'../results/GA_MultiScale/tuningGridNew1.npy'
    filename=f'../results/GA_MultiScale/headDirection_randomInput.npy'
    filename=f'../results/GA_MultiScale/place_randomInput.npy'

    # mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.05), int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.05)])
    # ranges = [[1,10],[1,10],[0.1,4],[0,0.1],[1,10],[1,10],[0.1,4],[0,0.1]]
    # fitnessFunc=CAN_tuningShiftAccuracywithWraparound

    # mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.03), np.random.normal(0,0.03), int(np.random.normal(0,1))])
    # ranges = [[1,10],[1,10],[0.05,3],[0,0.2],[1,10]]
    # fitnessFunc=CAN_tuningShiftAccuracy

    # mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.05), int(np.random.normal(0,1)), int(np.random.normal(0,1))])
    # ranges = [[1,10],[1,10],[0.1,1],[0,0.1],[1,10],[1,10]]
    # fitnessFunc=MultiResolutionFeedthrough2D

    # mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.05), np.random.normal(0,0.05), int(np.random.normal(0,1))])
    # ranges = [[1,20],[1,20],[0.05,4],[0,0.1],[1,2]]
    # fitnessFunc=headDirectionFitness

    ranges = [[1,20],[1,20],[0.05,4],[0,0.005],[1,4]]
    fitnessFunc= attractorGridcell_fitness

    # mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.005), np.random.normal(0,0.00005), int(np.random.normal(0,1)), int(np.random.normal(0,1))])
    # ranges = [[1,10],[1,10],[0,1],[0,0.0005],[1,5], [1,5]]
    # fitnessFunc=headDirectionAndPlaceNew

    # mutate_amount=np.array([int(np.random.normal(0,1)), int(np.random.normal(0,1)), np.random.normal(0,0.005), np.random.normal(0,0.00005), int(np.random.normal(0,1))])
    # ranges = [[1,10],[1,10],[0,1],[0,0.0008],[1,5]]
    # fitnessFunc=errorFilterNetwork


    num_gens=20
    population_size=28

    if plot==True:
        with open(filename, 'rb') as f:
            data = np.load(f)
        mean_1=np.array([np.mean(fit) for fit in data[:,:,-1]])
        std_1=np.array([np.std(fit) for fit in data[:,:,-1]])

        x = np.arange(len(mean_1))
        plt.plot(x, mean_1, color='teal', label='2D Network Tuning with Genetic Algorithm')
        plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='teal', alpha=0.2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness [-SAD]')
        plt.title('2D Network Tuning')
        plt.tight_layout()
        plt.savefig('../results/GA_MultiScale/2DTuning.pdf')
        print(data[-1,0,:])
    else:
        GeneticAlgorithm(num_gens,population_size,filename,fitnessFunc,ranges).implimentGA()

if __name__ == '__main__':
    freeze_support()
    # runGA1D(plot=False)
    runGA1D(plot=True)

filename=f'../results/GA_MultiScale/headDirection_randomInput.npy'
with open(filename, 'rb') as f:
    data = np.load(f)
    mean_1=np.array([np.mean(fit) for fit in data[:,:,-1]])
    std_1=np.array([np.std(fit) for fit in data[:,:,-1]])

    

filename=f'../results/GA_MultiScale/place_randomInput.npy'
with open(filename, 'rb') as f:
    data = np.load(f)
    mean_2=np.array([np.mean(fit) for fit in data[:15,:,-1]])
    std_2=np.array([np.std(fit) for fit in data[:15,:,-1]])

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(4, 2),sharey='row')
plt.tight_layout()


x1 = np.arange(len(mean_1))
ax1.plot(x1, mean_1, 'g-', label='Head Direction Network Tuning with Genetic Algorithm')
ax1.fill_between(x1, mean_1 - std_1, mean_1 + std_1, color='g', alpha=0.2)
ax1.set_xlabel('Generations')
ax1.set_ylabel('Fitness [-SAD]')
ax1.set_title('Head Direction Network Tuning')
# ax1.tight_layout()

x2 = np.arange(len(mean_2))
ax2.plot(x2, mean_2, color='teal', label='Head Direction Network Tuning with Genetic Algorithm')
ax2.fill_between(x2, mean_2 - std_2, mean_2 + std_2, color='teal', alpha=0.2)
ax2.set_xlabel('Generations')
# ax2.set_ylabel('Fitness [-SAD]')
ax2.set_title('Multiscale Network Tuning')
# ax2.tight_layout()

plt.savefig('../results/PaperFigures/HeadDirectionandMultiscaleNetworkTuning1.png')