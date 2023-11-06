import math
import numpy as np 
import time
import random  
import pandas as pd
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D 

from tqdm import tqdm


'''CAN networks'''
class attractorNetwork:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, N, num_links, excite_radius, activity_mag,inhibit_scale):
        self.excite_radius=excite_radius
        self.N=N
        self.num_links=num_links
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale

    def inhibitions(self,id):
        ''' each nueuron inhibits all other nueurons but itself'''
        return np.delete(np.arange(self.N),self.excitations(id))

    def excitations(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.excite_radius,self.excite_radius+1):
            excite.append((id + i) % self.N)
        return np.array(excite)

    def activation(self,id):
        '''each neuron excites itself and num_links neurons left and right with wraparound connections'''
        excite=[]
        for i in range(-self.num_links,self.num_links+1):
            excite.append((int(id) + i) % self.N)
        return np.array(excite)

    def full_weights(self,radius):
        x=np.arange(-radius,radius+1)
        return 1/(np.std(x) * np.sqrt(2 * np.pi)) * np.exp( - (x - np.mean(x))**2 / (2 * np.std(x)**2))  

    def fractional_weights(self,non_zero_prev_weights,delta):
        frac=delta%1
        if frac == 0:
            return non_zero_prev_weights
        else: 
            inv_frac=1-frac
            frac_weights=np.zeros((len(non_zero_prev_weights)))
            frac_weights[0]=non_zero_prev_weights[0]*inv_frac
            for i in range(1,len(non_zero_prev_weights)):
                frac_weights[i]=non_zero_prev_weights[i-1]*frac + non_zero_prev_weights[i]*inv_frac
            return frac_weights

    def frac_weights_1D(self, prev_weights, delta):
        mysign=lambda x: 1 if x > 0 else -1
        frac=delta - int(delta)
        inv_frac=1-abs(frac)

        non_zero_idxs=np.nonzero(prev_weights)[0]
        shifted_weights=np.zeros(self.N)
        shifted_weights[(non_zero_idxs+mysign(frac))%self.N]=prev_weights[non_zero_idxs]
        return prev_weights*inv_frac + shifted_weights*abs(frac)

    def update_weights_dynamics(self,prev_weights, delta, moreResults=None, cross=None):
        
        indexes,non_zero_weights,full_shift,inhibit_val=np.arange(self.N),np.zeros(self.N),np.zeros(self.N),0
        non_zero_idxs=indexes[prev_weights>0] # indexes of non zero prev_weights
        '''copied and shifted activity'''
        full_shift_amount=lambda x: np.floor(x) if x > 0 else np.ceil(x)
        full_shift[(non_zero_idxs + int(full_shift_amount(delta)))%self.N]=prev_weights[non_zero_idxs]*self.activity_mag
        # print(int(np.floor(delta)))
        shift=self.frac_weights_1D(full_shift,delta)  #non zero weights shifted by delta
        copy_shift=shift+prev_weights
        shifted_indexes=np.nonzero(copy_shift)[0]
        '''excitation'''
        excitations_store=np.zeros((len(shifted_indexes),self.N))
        excitation_array,excite=np.zeros(self.N),np.zeros(self.N)
        for i in range(len(shifted_indexes)):
            excitation_array[self.excitations(shifted_indexes[i])]=self.full_weights(self.excite_radius)*prev_weights[shifted_indexes[i]]
            excitations_store[i,:]=excitation_array
            excite[self.excitations(shifted_indexes[i])]+=self.full_weights(self.excite_radius)*prev_weights[shifted_indexes[i]]
        '''inhibit'''
        # shift_excite=copy_shift
        non_zero_inhibit=np.nonzero(excite) 
        for idx in non_zero_inhibit[0]:
            inhibit_val+=excite[idx]*self.inhibit_scale
        '''update activity'''
        prev_weights+=(copy_shift+excite-inhibit_val)
        prev_weights=prev_weights/np.linalg.norm(prev_weights)

        if moreResults==True:
           return prev_weights/np.linalg.norm(prev_weights), non_zero_weights,[inhibit_val]*self.N, excitations_store
        else: 
            return prev_weights/np.linalg.norm(prev_weights)


class attractorNetwork2D:
    '''defines 1D attractor network with N neurons, angles associated with each neurons 
    along with inhitory and excitatory connections to update the weights'''
    def __init__(self, N1, N2, num_links, excite_radius, activity_mag,inhibit_scale):
        self.excite_radius=int(excite_radius)
        self.num_links=int(num_links)
        self.N1=N1
        self.N2=N2  
        self.N=(N1,N2)
        self.activity_mag=activity_mag
        self.inhibit_scale=inhibit_scale

    def full_weights(self,radius, mx=0, my=0):
        len=(radius*2)+1
        x, y = np.meshgrid(np.linspace(-1,1,len), np.linspace(-1,1,len))
        sigma = 1.0
        return np.exp(-( ((x-mx)**2 + (y-my)**2) / ( 2.0 * sigma**2 ) ) )
    

    def inhibitions(self,weights):
        ''' constant inhibition scaled by amount of active neurons'''
        return np.sum(weights[weights>0]*self.inhibit_scale)

    def excitations(self, idx, idy, scale=1):
        '''A scaled 2D gaussian with excite radius is created at given neuron position with wraparound '''

        excite_rowvals = np.arange(-self.excite_radius, self.excite_radius+1)
        excite_colvals = np.arange(-self.excite_radius, self.excite_radius+1)
        excite_rowvals = (idx + excite_rowvals) % self.N[0]
        excite_colvals = (idy + excite_colvals) % self.N[1]

        gauss = self.full_weights(self.excite_radius)  # 2D gaussian scaled
        excite = np.zeros((self.N[0], self.N[1]))  # empty excite array
        excite[excite_rowvals[:, None], excite_colvals] = gauss
        return excite * scale

    def neuron_activation(self,idx,idy):
        '''A scaled 2D gaussian with excite radius is created at given neruon position with wraparound '''
        excite_rowvals=[] #wrap around row values 
        excite_colvals=[] #wrap around column values shifted_col_ids
        for i in range(-self.num_links,self.num_links+1):
            excite_rowvals.append((idx + i) % self.N1)
            excite_colvals.append((idy + i) % self.N2)
         

        gauss=self.full_weights(self.num_links)# 2D gaussian scaled 
        excite=np.zeros((self.N1,self.N2)) # empty excite array 
        for i,r in enumerate(excite_rowvals):
            for j,c in enumerate(excite_colvals):
                excite[r,c]=gauss[i,j]
        return excite

    def fractional_weights(self,full_shift,delta_row,delta_col):
        mysign=lambda x: 1 if x > 0 else -1
        frac_row, frac_col=delta_row - int(delta_row),delta_col - int(delta_col)
        inv_frac_row, inv_frac_col=[1-(delta_row%1),1-(delta_col%1)]

        non_zero_weights=np.nonzero(full_shift)
        shifted_row, shifted_col=np.zeros((self.N1,self.N2)), np.zeros((self.N1,self.N2))
        shifted_col[non_zero_weights[0],(non_zero_weights[1]+mysign(frac_col))%self.N2]=full_shift[non_zero_weights]
        shifted_row[(non_zero_weights[0]+mysign(frac_row))%self.N1, non_zero_weights[1]]=full_shift[non_zero_weights]
        
        if frac_row != 0 and frac_col !=0:
            shifted_rowThencol, shifted_colThenrow=np.zeros((self.N1,self.N2)), np.zeros((self.N1,self.N2))
            non_zero_col, non_zero_row=np.nonzero(shifted_col), np.nonzero(shifted_row)
            shifted_colThenrow[(non_zero_col[0]+ mysign(frac_row))%self.N1, non_zero_col[1]]=shifted_col[non_zero_col]
            shifted_rowThencol[non_zero_row[0], (non_zero_row[1]+ mysign(frac_col))%self.N2]=shifted_row[non_zero_row]

            col=full_shift*inv_frac_col + shifted_col*abs(frac_col)
            colRow=col*inv_frac_row + shifted_colThenrow*abs(frac_row)

            row=full_shift*inv_frac_row + shifted_row*abs(frac_row)
            rowCol=row*inv_frac_col + shifted_rowThencol*abs(frac_col)

            return (rowCol + colRow)/2
        
        elif frac_row == 0 and frac_col !=0:
            return shifted_col
        
        elif frac_row != 0 and frac_col ==0:
            return shifted_row
        
        else:
            return full_shift
    
    def fractional_shift(self, M, delta_row, delta_col):
        M_new=np.zeros((self.N[0], self.N[1]))
        axiss=[1,0]
        for idx, delta in enumerate([delta_col,delta_row]):
            
            frac = delta % 1
            if frac == 0.0:
                M_new=M
                continue
            
            shift= 1 if delta > 0 else -1
            frac = frac if delta > 0 else (1-frac)
            M_new += (1 - frac) * M + frac * np.roll(M, shift, axis=axiss[idx])

        return M_new / (np.linalg.norm(M_new))
 
    def update_weights_dynamics_row_col(self,prev_weights, delta_row, delta_col):
        non_zero_rows, non_zero_cols=np.nonzero(prev_weights) # indexes of non zero prev_weights
        prev_max_col,prev_max_row=np.argmax(np.max(prev_weights, axis=0)),np.argmax(np.max(prev_weights, axis=1))

        func = lambda x: int(math.ceil(x)) if x < 0 else int(math.floor(x))

        '''copied and shifted activity'''
        full_shift=np.zeros((self.N1,self.N2))
        shifted_row_ids, shifted_col_ids=(non_zero_rows +func(delta_row))%self.N1, (non_zero_cols+ func(delta_col))%self.N2
        full_shift[shifted_row_ids, shifted_col_ids]=prev_weights[non_zero_rows, non_zero_cols]
        copy_shift=self.fractional_shift(full_shift,delta_row,delta_col)*self.activity_mag


        '''excitation'''
        copyPaste=copy_shift
        non_zero_copyPaste=np.nonzero(copyPaste)  
        # print(len(non_zero_copyPaste[0]))
        excited=np.zeros((self.N1,self.N2))
        # t=time.time()
        for row, col in zip(non_zero_copyPaste[0], non_zero_copyPaste[1]):
            excited+=self.excitations(row,col,copyPaste[row,col])
        # print(time.time()-t)
        
        # excited=np.sum(excited_array, axis=0)
        # print(np.shape(excited_array), np.shape(excited))
        '''inhibitions'''
        inhibit_val=0
        shift_excite=copy_shift+prev_weights+excited
        non_zero_inhibit=np.nonzero(shift_excite) 
        for row, col in zip(non_zero_inhibit[0], non_zero_inhibit[1]):
            inhibit_val+=shift_excite[row,col]*self.inhibit_scale
        inhibit_array=np.tile(inhibit_val,(self.N1,self.N2))

        '''update activity'''

        prev_weights+=copy_shift+excited-inhibit_val
        prev_weights[prev_weights<0]=0


        
        return prev_weights/np.linalg.norm(prev_weights) if np.sum(prev_weights) > 0 else [np.nan]
    
    def update_weights_dynamics(self,prev_weights, direction, speed, moreResults=None):
        non_zero_rows, non_zero_cols=np.nonzero(prev_weights) # indexes of non zero prev_weights
        # maxXPerScale, maxYPerScale=np.argmax(np.max(prev_weights, axis=0)),np.argmax(np.max(prev_weights, axis=1))
        prev_maxXPerScale, prev_maxYPerScale = np.argmax(np.max(prev_weights, axis=1)) , np.argmax(np.max(prev_weights, axis=0))
        prev_max_col=round(activityDecoding(prev_weights[prev_maxXPerScale, :],5,self.N2),0)
        prev_max_row=round(activityDecoding(prev_weights[:,prev_maxYPerScale],5,self.N1),0)

        delta_row=np.round(speed*np.sin(np.deg2rad(direction)),6)
        delta_col=np.round(speed*np.cos(np.deg2rad(direction)),6)
        
        func = lambda x: int(math.ceil(x)) if x < 0 else int(math.floor(x))

        '''copied and shifted activity'''
        full_shift=np.zeros((self.N1,self.N2))
        shifted_row_ids, shifted_col_ids=(non_zero_rows +func(delta_row))%self.N1, (non_zero_cols+ func(delta_col))%self.N2
        full_shift[shifted_row_ids, shifted_col_ids]=prev_weights[non_zero_rows, non_zero_cols]
        copy_shift=self.fractional_shift(full_shift,delta_row,delta_col)*self.activity_mag


        '''excitation'''
        copyPaste=copy_shift
        non_zero_copyPaste=np.nonzero(copyPaste)  
        # print(len(non_zero_copyPaste[0]))
        excited=np.zeros((self.N1,self.N2))
        # t=time.time()
        for row, col in zip(non_zero_copyPaste[0], non_zero_copyPaste[1]):
            excited+=self.excitations(row,col,copyPaste[row,col])
        # print(time.time()-t)
        
        # excited=np.sum(excited_array, axis=0)
        # print(np.shape(excited_array), np.shape(excited))
        '''inhibitions'''
        inhibit_val=0
        shift_excite=copy_shift+prev_weights+excited
        non_zero_inhibit=np.nonzero(shift_excite) 
        for row, col in zip(non_zero_inhibit[0], non_zero_inhibit[1]):
            inhibit_val+=shift_excite[row,col]*self.inhibit_scale
        inhibit_array=np.tile(inhibit_val,(self.N1,self.N2))

        '''update activity'''

        prev_weights+=copy_shift+excited-inhibit_val
        prev_weights[prev_weights<0]=0

 
        '''wrap around'''
        # maxXPerScale, maxYPerScale=np.argmax(np.max(prev_weights, axis=0)),np.argmax(np.max(prev_weights, axis=1))
        maxXPerScale, maxYPerScale = np.argmax(np.max(prev_weights, axis=1)) , np.argmax(np.max(prev_weights, axis=0))
        max_col=round(activityDecoding(prev_weights[maxXPerScale, :],5,self.N2),0)
        max_row=round(activityDecoding(prev_weights[:,maxYPerScale],5,self.N1),0)
        
        # print(f"col_prev_current {prev_max_col, max_col} row_prev_current {prev_max_row, max_row}")
        wrap_cols=0 
        
        if prev_max_col>max_col and (direction<=90 or direction>=270): #right 
            wrap_cols=1
            # print(f'{direction}, prev col {prev_max_col}, curr_col {max_col}')
        elif prev_max_col<max_col and (direction>=90 and direction<=270): #left
            wrap_cols=-1
            # print(f'{direction}, prevCol {prev_max_col}, currCol {max_col}')

        
        wrap_rows=0 
 
        if prev_max_row>max_row and (direction>=0 and direction<=180) : #up 
            wrap_rows=1
        elif prev_max_row<max_row and (direction>=180 and direction<=360) : #down 
            wrap_rows=-1
        
            


        # actual_delta_col=max_col-prev_max_col
        # actual_delta_row=max_row-prev_max_row

        # print(prev_max_row,max_row, actual_delta_row)

        # if prev_max_col+actual_delta_col> self.N2-1 :
        #     wrap_cols=1
        #     wrap_counter[current_scale]+=1
        # elif prev_max_col+actual_delta_col <= 0 :
        #     wrap_cols=-1
        #     wrap_counter[current_scale]+=1
        # else:
        #     wrap_cols=0 

        # if prev_max_row+actual_delta_row> self.N1-1 :
        #     wrap_rows=1
        #     wrap_counter[current_scale]+=1
        # elif prev_max_row+actual_delta_row <= 0 :
        #     wrap_rows=-1
        #     wrap_counter[current_scale]+=1
        # else:
        #     wrap_rows=0 
        
        # if wrap_counter[current_scale]==1 : # identifying first instance of wrap around where the change is only 0.5 
        #     wrap_cols*=0.5
        #     wrap_rows*=0.5
        # elif wrap_counter[current_scale]==2 and any(direction == diag for diag in [45, 135, 225, 315]): # identifying diagonals where row and column are crossed together 
        #     wrap_cols*=0.5
        #     wrap_rows*=0.5


        if moreResults==True:
            return prev_weights/np.linalg.norm(prev_weights),copy_shift,excited,inhibit_array
        else:
            return prev_weights/np.linalg.norm(prev_weights) if np.sum(prev_weights) > 0 else [np.nan], wrap_rows, wrap_cols
        
    def shiftedCell(self, placeWeights, direction, speed):
        pass


'''Decoding Pose Helper Fucntions'''
def activityDecoding(prev_weights,radius,N):
    '''Isolating activity at a radius around the peak to decode position'''
    # if np.argmax(prev_weights)==0:
    #     return 0
    # else:
    neurons=np.arange(N)
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]

    x,y=local_activity*np.cos(np.deg2rad(neurons*360/N)), local_activity*np.sin(np.deg2rad(neurons*360/N))
    vect_sum=np.rad2deg(math.atan2(sum(y),sum(x))) % 360
    weighted_sum = N*(vect_sum/360)

    if weighted_sum>(N-1):
        weighted_sum=0

    return weighted_sum


def activityDecodingAngle(prev_weights,radius,N):
    '''Isolating activity at a radius around the peak to decode direction'''
    neurons = np.arange(N)
    peak=np.argmax(prev_weights) 
    local_activity=np.zeros(N)
    local_activity_idx=[]
    for i in range(-radius,radius+1):
        local_activity_idx.append((peak + i) % N)
    local_activity[local_activity_idx]=prev_weights[local_activity_idx]

    x,y=local_activity*np.cos(np.deg2rad(neurons*360/N)), local_activity*np.sin(np.deg2rad(neurons*360/N))
    vect_sum=np.rad2deg(math.atan2(sum(y),sum(x))) % 360
    return vect_sum



'''Multiscale CAN Helper Functions'''
def positionToVel2D(data_x,data_y):
    vel,angVel=[],[]
    for i in range(0,len(data_x)):
        x0=data_x[i-2]
        x1=data_x[i-1]
        x2=data_x[i]
        y0=data_y[i-2]
        y1=data_y[i-1]
        y2=data_y[i]

        vel.append(np.sqrt(((x2-x1)**2)+((y2-y1)**2))) #translation
        angVel.append((math.atan2(y2-y1,x2-x1)) - (math.atan2(y1-y0,x1-x0))) 
    
    return np.array(vel),np.array(angVel)


def pathIntegration(speed, angVel):
    q=[0,0,0]
    x_integ,y_integ=[],[]
    for i in range(len(speed)):
        q[0],q[1]=q[0]+speed[i]*np.cos(q[2]), q[1]+speed[i]*np.sin(q[2])
        q[2]+=angVel[i]
        x_integ.append(round(q[0],4))
        y_integ.append(round(q[1],4))

    return x_integ, y_integ


def errorTwoCoordinateLists(x_pi, y_pi, x2, y2, errDistri=False):
    '''RMSE error'''
    err=[]
    x_err_sum,y_err_sum=0,0
    for i in range(len(x2)):
        x_err_sum+=(x_pi[i]-x2[i])**2
        y_err_sum+=(y_pi[i]-y2[i])**2
        if errDistri== True:
            err.append(np.abs(x_pi[i]-x2[i])+ np.abs(y_pi[i]-y2[i]))

    x_error=np.sqrt(x_err_sum/len(x2))
    y_error=np.sqrt(y_err_sum/len(y2))

    if errDistri==True:
        return np.array(err)#np.cumsum(np.array(err))
    else:
        return (x_error+y_error)


def scale_selection(input,scales, swap_val=1):
    if len(scales)==1:
        scale_idx=0
    else: 

        if input<=scales[0]*swap_val:
            scale_idx=0

        for i in range(len(scales)-1):
            if input>scales[i]*swap_val and input<=scales[i+1]*swap_val:
                scale_idx=i+1
        
        # if input>scales[-2]*swap_val:
        #     scale_idx=len(scales)-1
        
        if input>scales[-1]*swap_val:
            scale_idx=len(scales)-1

    return scale_idx


def headDirection(theta_weights, angVel, init_angle):
    global theata_called_iters
    N=360
    # num_links,excite,activity_mag,inhibit_scale, iterations=16, 17, 2.16818183,  0.0281834545, 2
    # num_links,excite,activity_mag,inhibit_scale, iterations=16, 17, 2.16818183,  0.0381834545, 2
    num_links,excite,activity_mag,inhibit_scale, iterations=13,4,2.70983783e+00,4.84668851e-02,2
    net=attractorNetwork(N,num_links,excite, activity_mag,inhibit_scale)
    
    if theata_called_iters==0:
        theta_weights[net.activation(init_angle)]=net.full_weights(num_links)
        theata_called_iters+=1

    for j in range(iterations):
        theta_weights=net.update_weights_dynamics(theta_weights,angVel)
        theta_weights[theta_weights<0]=0
    
    
    return theta_weights


def hierarchicalNetwork2DGridNowrapNet(prev_weights, net,N, vel, direction, iterations, wrap_iterations, x_grid_expect, y_grid_expect,scales):
    '''Select scale and initilise wrap storage'''
    delta = [(vel/scales[i]) for i in range(len(scales))]
    cs_idx=scale_selection(vel,scales)
    # print(vel, scales, cs_idx)
    wrap_rows=np.zeros((len(scales)))
    wrap_cols=np.zeros((len(scales)))

    '''Update selected scale'''

    for i in range(iterations):
        prev_weights[cs_idx][:], wrap_rows_cs, wrap_cols_cs= net.update_weights_dynamics(prev_weights[cs_idx][:],direction, delta[cs_idx])
        prev_weights[cs_idx][prev_weights[cs_idx][:]<0]=0
        x_grid_expect+=wrap_cols_cs*N*scales[cs_idx]
        y_grid_expect+=wrap_rows_cs*N*scales[cs_idx]
    


    
        # if np.any(wrap_cols_cs!=0):
        #     print(f"------------------------------------------------------------------------------------------------------wrap_cols {wrap_cols_cs}, {scales[cs_idx]}")
        # if np.any(wrap_rows_cs!=0):
        #     print(f"------------------------------------------------------------------------------------------------------wrap_rows {wrap_rows_cs}, {scales[cs_idx]}")

    wrap=0   
    return prev_weights, wrap, x_grid_expect, y_grid_expect


def headDirectionAndPlaceNoWrapNet(scales, vel, angVel,savePath, printing=False, N=100, returnTypes=None, genome=None):
    global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 

    if genome is not None: 
        num_links=int(genome[0]) #int
        excite=int(genome[1]) #int
        activity_mag=genome[2] #uni
        inhibit_scale=genome[3] #uni
        iterations=int(genome[4])
        wrap_iterations=int(genome[5])
     
    else:
        num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,6.51431074e-04,3,2 #with decimals 200 iters fitness -395 modified
        # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=11,8,5.09182735e-01,2.78709739e-04,5,2
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)

    

    '''__________________________Storage and initilisation parameters______________________________'''
    # scales=[0.25,1,4,16]
    theta_weights=np.zeros(360)
    theata_called_iters=0
    # start_x, start_y=(50*scales[3])+(50*scales[4])+(50*scales[5]),(50*scales[3])+(50*scales[4])+(50*scales[5])
    wrap_counter=[0,0,0,0,0,0]
    x_grid, y_grid=[], []
    x_grid_expect, y_grid_expect =0,0
    x_integ, y_integ=[],[]
    q=[0,0,0]
    x_integ_err, y_integ_err=[],[]
    q_err=[0,0,0]

    '''__________________________Initilising scales in the center and at the edge_____________________________'''
    prev_weights=[np.zeros((N,N)) for _ in range(len(scales))]
    for n in range(len(scales)):
        for m in range(iterations):
            prev_weights[n]=network.excitations(0,0)
            prev_weights[n]=network.update_weights_dynamics_row_col(prev_weights[n][:], 0, 0)
            prev_weights[n][prev_weights[n][:]<0]=0
    

    '''_______________________________Iterating through simulation velocities_______________________________'''
    # for i in range(1,len(vel)):   
    tbar = tqdm(range(1,len(vel)))
    for i in tbar:
        '''Path integration'''
        q[2]+=angVel[i]
        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        x_integ.append(q[0])
        y_integ.append(q[1])


        '''Mutliscale CAN update'''
        N_dir=360
        theta_weights=headDirection(theta_weights, np.rad2deg(angVel[i]), 0)
        direction=activityDecodingAngle(theta_weights,5,N_dir)
        prev_weights, wrap, x_grid_expect, y_grid_expect= hierarchicalNetwork2DGridNowrapNet(prev_weights, network, N, vel[i], direction, iterations,wrap_iterations, x_grid_expect, y_grid_expect, scales)

        '''1D method for decoding'''
        maxXPerScale, maxYPerScale = np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))]), np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])
        decodedXPerScale=[activityDecoding(prev_weights[m][maxXPerScale[m], :],5,N)*scales[m] for m in range(len(scales))]
        decodedYPerScale=[activityDecoding(prev_weights[m][:,maxYPerScale[m]],5,N)*scales[m] for m in range(len(scales))]
        x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale), np.sum(decodedYPerScale)
        # x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale[0:3]+x_grid_expect[3:6]), np.sum(decodedYPerScale[0:3]+y_grid_expect[3:6])
        x_grid.append(x_multiscale_grid+x_grid_expect)
        y_grid.append(y_multiscale_grid+y_grid_expect)

        '''Error integrated path'''
        q_err[2]+=angVel[i]
        q_err[0],q_err[1]=q_err[0]+vel[i]*np.cos(np.deg2rad(direction)), q_err[1]+vel[i]*np.sin(np.deg2rad(direction))
        x_integ_err.append(q_err[0])
        y_integ_err.append(q_err[1])

        if printing==True:
            print(f'dir: {np.rad2deg(q[2])}, {direction}')
            print(f'vel: {vel[i]}')
            print(f'decoded: {decodedXPerScale}, {decodedYPerScale}')
            print(f'expected: {x_grid_expect}, {y_grid_expect}')
            print(f'integ: {x_integ[-1]}, {y_integ[-1]}')
            print(f'CAN: {x_grid[-1]}, {y_grid[-1]}')
            print('')

    if savePath != None:
        np.save(savePath, np.array([x_grid, y_grid, x_integ, y_integ, x_integ_err, y_integ_err]))
    
    print(f'CAN error: {errorTwoCoordinateLists(x_integ,y_integ, x_grid, y_grid)}') 
        
    

    if returnTypes=='Error':
        return errorTwoCoordinateLists(x_integ,y_integ, x_grid, y_grid)
    elif returnTypes=='PlotShow':
        plt.plot(x_integ, y_integ, 'g.')
        # plt.plot(x_integ_err, y_integ_err, 'y.')
        plt.plot(x_grid, y_grid, 'b.')
        plt.axis('equal')
        plt.title('Test Environment 2D space')
        plt.legend(('Path Integration without Error','Multiscale Grid Decoding'))
        plt.show()
    elif returnTypes=='posInteg+CAN':
        return x_integ,y_integ, x_grid, y_grid


def headDirectionAndPlaceNoWrapNetAnimate(scales, test_length, vel, angVel,savePath, plot=False, printing=True, N=100):
    global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 

    num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,6.51431074e-04,3,2 #with decimals 200 iters fitness -395 modified
    num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,10,1,0.0008,2,1
    network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)

    

    '''__________________________Storage and initilisation parameters______________________________'''
    # scales=[0.25,1,4,16]
    theta_weights=np.zeros(360)
    theata_called_iters=0
    # start_x, start_y=(50*scales[3])+(50*scales[4])+(50*scales[5]),(50*scales[3])+(50*scales[4])+(50*scales[5])
    wrap_counter=[0,0,0,0,0,0]
    x_grid, y_grid=[], []
    x_grid_expect, y_grid_expect =0,0
    x_integ, y_integ=[],[]
    q=[0,0,0]
    x_integ_err, y_integ_err=[],[]
    q_err=[0,0,0]

    '''__________________________Initilising scales in the center and at the edge_____________________________'''
    prev_weights=[np.zeros((N,N)) for _ in range(len(scales))]
    for n in range(len(scales)):
        for m in range(iterations):
            prev_weights[n]=network.excitations(0,0)
            prev_weights[n]=network.update_weights_dynamics_row_col(prev_weights[n][:], 0, 0)
            prev_weights[n][prev_weights[n][:]<0]=0
    

    '''_______________________________Iterating through simulation velocities_______________________________'''
    fig, axs = plt.subplots(1,4,figsize=(5, 3)) 
    def animate(i):  
        global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 

        '''Path integration'''
        q[2]+=angVel[i]
        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        x_integ.append(q[0])
        y_integ.append(q[1])


        '''Mutliscale CAN update'''
        N_dir=360
        theta_weights=headDirection(theta_weights, np.rad2deg(angVel[i]), 0)
        direction=activityDecodingAngle(theta_weights,5,N_dir)
        prev_weights, wrap, x_grid_expect, y_grid_expect= hierarchicalNetwork2DGridNowrapNet(prev_weights, network, N, vel[i], direction, iterations,wrap_iterations, x_grid_expect, y_grid_expect, scales)

        '''1D method for decoding'''
        maxXPerScale, maxYPerScale = np.array([np.argmax(np.max(prev_weights[m], axis=1)) for m in range(len(scales))]), np.array([np.argmax(np.max(prev_weights[m], axis=0)) for m in range(len(scales))])
        decodedXPerScale=[activityDecoding(prev_weights[m][maxXPerScale[m], :],5,N)*scales[m] for m in range(len(scales))]
        decodedYPerScale=[activityDecoding(prev_weights[m][:,maxYPerScale[m]],5,N)*scales[m] for m in range(len(scales))]
        x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale), np.sum(decodedYPerScale)
        # x_multiscale_grid, y_multiscale_grid=np.sum(decodedXPerScale[0:3]+x_grid_expect[3:6]), np.sum(decodedYPerScale[0:3]+y_grid_expect[3:6])
        x_grid.append(x_multiscale_grid+x_grid_expect)
        y_grid.append(y_multiscale_grid+y_grid_expect)

        '''Error integrated path'''
        q_err[2]+=angVel[i]
        q_err[0],q_err[1]=q_err[0]+vel[i]*np.cos(np.deg2rad(direction)), q_err[1]+vel[i]*np.sin(np.deg2rad(direction))
        x_integ_err.append(q_err[0])
        y_integ_err.append(q_err[1])

        if printing==True:
            print(f'dir: {np.rad2deg(q[2])}, {direction}')
            print(f'vel: {vel[i]}')
            print(f'decoded: {decodedXPerScale}, {decodedYPerScale}')
            print(f'expected: {x_grid_expect}, {y_grid_expect}')
            print(f'integ: {x_integ[-1]}, {y_integ[-1]}')
            print(f'CAN: {x_grid[-1]}, {y_grid[-1]}')
            print('')

        for k in range(4):
            axs[k].clear()
            axs[k].imshow(prev_weights[k][:][:], cmap='jet')#(np.arange(N),prev_weights[k][:],color=colors[k])
            axs[k].spines[['top', 'left', 'right']].set_visible(False)
            axs[k].invert_yaxis()

    ani = FuncAnimation(fig, animate, interval=1,frames=test_length,repeat=False)
    # plt.show()

    writergif = animation.PillowWriter(fps=30) 
    ani.save(savePath, writer=writergif)

    print(f'CAN error: {errorTwoCoordinateLists(x_integ,y_integ, x_grid, y_grid)}')

