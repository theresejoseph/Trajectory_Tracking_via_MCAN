import math
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

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

    def excitations(self,idx,idy,scale=1):
        '''A scaled 2D gaussian with excite radius is created at given neruon position with wraparound '''
        
        excite_rowvals=[] #wrap around row values 
        excite_colvals=[] #wrap around column values 
        for i in range(-self.excite_radius,self.excite_radius+1):
            excite_rowvals.append((idx + i) % self.N1)
            excite_colvals.append((idy + i) % self.N2)
         

        gauss=self.full_weights(self.excite_radius)# 2D gaussian scaled 
        excite=np.zeros((self.N1,self.N2)) # empty excite array 
        for i,r in enumerate(excite_rowvals):
            for j,c in enumerate(excite_colvals):
                excite[r,c]=gauss[i,j]
        return excite*scale 

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
    
    def fractional_shift(self, M,delta_row,delta_col):
        M_row, M_col = np.zeros((self.N1, self.N2)), np.zeros((self.N1, self.N2))
        
        mysign=lambda x: 1 if x > 0 else -1
        whole_shift_row, whole_shift_col = np.floor(delta_row), np.floor(delta_col)
        frac_row, frac_col=delta_row%1,delta_col%1
        inv_frac_row, inv_frac_col=[1-(delta_row%1),1-(delta_col%1)]

        for i in range(self.N1):
            for j in range(self.N2):
                # M_row[int((i+whole_shift_row)%self.N1),j]=inv_frac_row*prev_weights[int((i+whole_shift_row)%self.N1),j] + frac_row*prev_weights[int((i+whole_shift_row+mysign(frac_row))%self.N1), j]
                # M_col[i,int((j+whole_shift_col)%self.N2)]=inv_frac_col*prev_weights[i,int((j+whole_shift_col)%self.N2)] + frac_col*prev_weights[i,int((j+whole_shift_col+mysign(frac_col))%self.N2)]
                if frac_row == 0.0:
                    M_row=M
                elif delta_row>0:
                    M_row[i,j]=(1-frac_row)*M[i,j] + frac_row*M[int((i-1)%self.N1), j]
                else:
                    M_row[i,j]=(frac_row)*M[i,j] + (1-frac_row)*M[int((i+1)%self.N1), j]

                if frac_col == 0.0:
                    M_col=M
                elif delta_col>0:    
                    M_col[i,j]=(1-frac_col)*M[i,j] + frac_col*M[i,int((j-1)%self.N2)]
                else:
                    M_col[i,j]=(frac_col)*M[i,j] + (1-frac_col)*M[i,int((j+1)%self.N2)]
        return (M_row+M_col) /np.linalg.norm((M_row+M_col))
 
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

'''Tester Functions'''
def visulaiseFractionalWeights():
    fig = plt.figure(figsize=(5, 6))
    nrows=5
    ax0 = fig.add_subplot(nrows, 1, 1)
    ax1 = fig.add_subplot(nrows, 1, 2)
    ax2 = fig.add_subplot(nrows, 1, 3)
    ax3 = fig.add_subplot(nrows, 1, 4)
    ax4 = fig.add_subplot(nrows, 1, 5)
    fig.tight_layout()

    weights=np.array([0,0,0,0,0,2,3,4,5,4,3,2,0,0,0,0])
    N1,num_links,excite_radius,activity_mag,inhibit_scale= len(weights),6, 4, 1, 0.01
    net=attractorNetwork( N1,num_links,excite_radius,activity_mag,inhibit_scale)


    idx=np.nonzero(weights)[0]
    shifted_right,shifted_left=np.zeros(len(weights)),np.zeros(len(weights))
    shifted_right[idx+1]=weights[idx]
    shifted_left[idx-1]=weights[idx]
    frac=weights*0.25 + shifted_right*0.75
    frac2=weights*0.01 + shifted_right*0.99


    ax0.bar(np.arange(len(weights)),weights,color='r')
    ax0.set_ylim([0,10])
    ax0.set_title('Orginal')

    ax1.bar(np.arange(len(weights)),net.frac_weights_1D(weights,0.012),color='m')
    ax1.set_title('0.25 unit copy paste')
    ax1.set_ylim([0,10])

    ax2.bar(np.arange(len(weights)),net.frac_weights_1D(weights,0.5),color='m')
    ax2.set_title('0.5 unit copy paste')
    ax2.set_ylim([0,10])

    
    ax3.bar(np.arange(len(weights)),net.frac_weights_1D(weights,0.75), color='m')
    ax3.set_title('0.75 unit copy paste')
    ax3.set_ylim([0,10])

    ax4.bar(np.arange(len(weights)), shifted_right, color='b')
    ax4.set_title('1 unit Right')
    ax4.set_ylim([0,10])
    plt.show()

def visulaiseDeconstructed2DAttractor():
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 4, 1)
    ax1 = fig.add_subplot(1, 4, 2)
    ax2 = fig.add_subplot(1, 4, 3)
    ax3 = fig.add_subplot(1, 4, 4)
    fig.tight_layout()

    N1,N2,excite_radius,activity_mag,inhibit_scale=  10, 10, 1, 1, 0.01
    delta_row, delta_col = 0.167, 0.33
    net=attractorNetwork2D( N1,N2,excite_radius,activity_mag,inhibit_scale)
    old_weights=net.excitations(0,9)

    ax0.imshow(old_weights)
    ax0.set_title('Previous Activity')

    old_weights, copy, excite,inhibit_array = net.update_weights_dynamics(old_weights,delta_row,delta_col,moreResults=True)
    ax1.imshow(copy)
    non_zero_copy=np.nonzero(copy)
    print(copy[non_zero_copy[0],non_zero_copy[1]])
    ax1.set_title('Copied and Shifted Activity')

    ax2.imshow(excite)
    ax2.set_title('Exctied Activity')

    ax3.imshow(old_weights)
    ax3.set_title('Inhibited Activity')
    plt.show()

def visulaise2DFractions(prev_weights, another_prev_weights):
    fig = plt.figure(figsize=(13, 4))
    ax0 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2)
    fig.tight_layout()

    N1,N2,num_links,excite_radius,activity_mag,inhibit_scale=  100, 100, 1, 1, 1, 0.0005
    net=attractorNetwork2D( N1,N2,num_links,excite_radius,activity_mag,inhibit_scale)


    ax0.imshow(prev_weights)
    ax0.set_title('Previous Activity')
    ax0.invert_yaxis()

   
    another_prev_weights=net.update_weights_dynamics(another_prev_weights,0,0)
    ax0.imshow(another_prev_weights)
    ax0.set_title('Copied and Shifted 0.9 Column 0.9 Row')
    ax0.invert_yaxis()

    def animate(i):
        global prev_weights
        prev_weights=net.fractional_shift(prev_weights,0.7,0.7)
        ax3.imshow(prev_weights)
        ax3.set_title('Copied and Shifted 0.1 Column 0.1 Row')
        ax3.invert_yaxis()

    ani = FuncAnimation(fig, animate, interval=1,frames=100,repeat=False)
    plt.show()


# visulaiseFractionalWeights()
# visulaiseDeconstructed2DAttractor()

# N1,N2,num_links,excite_radius,activity_mag,inhibit_scale=  100, 100, 1, 1, 1, 0.0005
# net=attractorNetwork2D( N1,N2,num_links,excite_radius,activity_mag,inhibit_scale)
# prev_weights=net.neuron_activation(50,50)
# another_prev_weights=net.neuron_activation(50,50)
# visulaise2DFractions(prev_weights, another_prev_weights)

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


