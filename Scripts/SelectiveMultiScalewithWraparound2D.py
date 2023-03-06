
import matplotlib.pyplot as plt
import numpy as np
import random  
import math
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D 
from scipy import signal
from scipy import ndimage
import time 
from os import listdir
import sys
sys.path.append('./scripts')
import CAN
from CAN import attractorNetwork2D, attractorNetwork, activityDecodingAngle, activityDecoding
import CAN as can
# import pykitti
import json 
from DataHandling import saveOrLoadNp  
# import scienceplots
plt.style.use(['science','ieee'])
# plt.style.use(['science','no-latex'])



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
        

        # elif input>scales[0]*swap_val and input<=scales[1]*swap_val:
        #         scale_idx=1 
        
        # elif input>scales[1]*swap_val and input<=scales[1]*swap_val:
        #         scale_idx=2
        
        # elif input>scales[2]*swap_val and input<=scales[1]*swap_val:
        #         scale_idx=3
        
        # elif input>scales[3]*swap_val and input<=scales[1]*swap_val:
        #         scale_idx=4
        for i in range(len(scales)-2):
            if input>scales[i]*swap_val and input<=scales[i+1]*swap_val:
                scale_idx=i+1
        
        if input>scales[-2]*swap_val:
            scale_idx=len(scales)-1
    # elif input>scales[2]*swap_val and input<=scales[3]*swap_val:
    #     scale_idx=3
    # elif input>scales[3]*swap_val:
    #     scale_idx=4
    return scale_idx

def headDirection(theta_weights, angVel, init_angle):
    global theata_called_iters
    N=360
    # num_links,excite,activity_mag,inhibit_scale, iterations=16, 17, 2.16818183,  0.0281834545, 2
    num_links,excite,activity_mag,inhibit_scale, iterations=16, 17, 2.16818183,  0.0381834545, 2
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
        # wrap_rows[cs_idx]+=wrap_rows_cs
        # wrap_cols[cs_idx]+=wrap_cols_cs
        x_grid_expect+=wrap_cols_cs*N*scales[cs_idx]
        y_grid_expect+=wrap_rows_cs*N*scales[cs_idx]
    
    # for i in range(len(wrap_cols)):
    #     x_grid_expect+=wrap_cols[i]*N*scales[i]
    # for i in range(len(wrap_rows)):
    #     y_grid_expect+=wrap_rows[i]*N*scales[i]

    
        if np.any(wrap_cols_cs!=0):
            print(f"------------------------------------------------------------------------------------------------------wrap_cols {wrap_cols_cs}, {scales[cs_idx]}")
        if np.any(wrap_rows_cs!=0):
            print(f"------------------------------------------------------------------------------------------------------wrap_rows {wrap_rows_cs}, {scales[cs_idx]}")

    wrap=0   
    return prev_weights, wrap, x_grid_expect, y_grid_expect

def headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,savePath, plot=False, printing=True, N=100):
    global theata_called_iters,theta_weights, prev_weights, q, wrap_counter, current_i, x_grid_expect, y_grid_expect 

    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=7,8,5.47157578e-01 ,3.62745653e-04, 2, 2 #good only at small scale
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=72,1,9.05078199e-01,7.85317908e-04,4,1
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=6,1,3.89338335e-01,1.60376324e-04, 3,3  #improved at larger scale 

    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=7,1,2.59532708e-01 ,2.84252467e-04,4,3 #without decimals 1000 iters fitness -5000
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,6.51431074e-04,3,2 #with decimals 200 iters fitness -395
    num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,2,1.10262708e-01,6.51431074e-04,3,2 #with decimals 200 iters fitness -395 modified
    
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=10,10,1,0.0008,2,1 #with decimals 200 iters fitness -395 modified
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=5,7,9.59471889e-01,2.93846361e-04,1,1 #tuned to reduce error 
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.015,0.000865888565,1,1 #0.25 scale input, np.random.uniform(0,1,1) error
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.05,0.000565888565,1,1 #16 scale input, np.random.uniform(10,20,1) error
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.04,0.000965888565,1,1 #1 scale input, np.random.uniform(1,2,1) error
    # num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.05,0.000665888565,1,1 #4 scale input, np.random.uniform(2,10,1) error
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
    for i in range(1,test_length):   
        '''Path integration'''
        q[2]+=angVel[i]
        q[0],q[1]=q[0]+vel[i]*np.cos(q[2]), q[1]+vel[i]*np.sin(q[2])
        x_integ.append(q[0])
        y_integ.append(q[1])

        '''Dynamic network tuning'''
        # swap_val=5
        # if vel[i]<=scales[0]*swap_val:
        #     num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.015,0.000865888565,1,1 #0.25 scale input, np.random.uniform(0,1,1) error
        #     network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
        # elif vel[i]>scales[0]*swap_val and vel[i]<=scales[1]*swap_val:
        #     num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.04,0.000865888565,1,1 #1 scale input, np.random.uniform(1,2,1) error
        #     network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
        # elif vel[i]>scales[1]*swap_val and vel[i]<=scales[2]*swap_val:
        #     num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.05,0.000665888565,1,1 #4 scale input, np.random.uniform(2,10,1) error
        #     network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)
        # elif vel[i]>scales[2]*swap_val:
        #     num_links,excite,activity_mag,inhibit_scale, iterations, wrap_iterations=3,5,0.05,0.000565888565,1,1 #16 scale input, np.random.uniform(10,20,1) error
        #     network=attractorNetwork2D(N,N,num_links,excite, activity_mag,inhibit_scale)   

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

    if plot ==True:    
        plt.plot(x_integ, y_integ, 'g.')
        # plt.plot(x_integ_err, y_integ_err, 'y.')
        plt.plot(x_grid, y_grid, 'b.')
        plt.axis('equal')
        plt.title('Test Environment 2D space')
        plt.legend(('Path Integration without Error','Multiscale Grid Decoding'))
        plt.show()
    else:
        return x_grid, y_grid

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



def plotFromSavedArray(outfile,savePath):
    x_grid,y_grid, x_integ, y_integ, x_integ_err, y_integ_err= np.load(outfile)
    '''Compute distance'''
    dist=np.sum(vel)
   
    '''Compute error'''
    error=errorTwoCoordinateLists(x_integ, y_integ,x_grid,y_grid)

    '''Plot'''
    fig, axs = plt.subplots(1,1,figsize=(4, 3))
    plt.title('Kitti Dataset Trajectory Tracking')
    plt.plot(x_integ, y_integ, 'g--')
    plt.plot(x_grid, y_grid, 'm-')
    # plt.plot(wrap_x, wrap_y,'r*')
    plt.axis('equal')
    # plt.title(f'Distance:{round(dist)}m   Error:{round(error)}m/iter   Iterations:{len(x_integ)}')
    plt.legend(('Path Integration', 'Multiscale CAN'))
    plt.savefig(savePath)

def plotSavedMultiplePaths(length,outfile,pathfile,savepath,figrows,figcols,randomSeedVariation):
    fig, axs = plt.subplots(figrows,figcols,figsize=(4, 4))
    fig.legend(['MultiscaleCAN', 'Grid'])
    fig.tight_layout(pad=1)
    fig.suptitle('Tracking Simulated Trajectories through Berlin')
    axs=axs.ravel()
    errors,distance=[],[]
    for i in range(length):
        '''error'''
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePath{i}.npy'         
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathChangedFeedthrough_{i}.npy'
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathTuned_{i}.npy'
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathLonger_{i}.npy'
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathwithUniformErr_{i}.npy'
        # outfile=f'./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathwithSpeeds0to20_{i}.npy'
        x_grid,y_grid,x_integ, y_integ, x_integ_err, y_integ_err = np.load(pathfile+f'{i}.npy')

        '''distance'''
        # traverseInfo=np.load(outfile+f'{i}.npz', allow_pickle=True)
        # dist=np.sum(traverseInfo['speeds'])
        np.random.seed(i*randomSeedVariation)
        vel=np.random.uniform(0,20,len(x_grid)) 
        dist=np.sum(vel)
 
        # print(f'RMSE:{errorTwoCoordinateLists(x_integ, y_integ,x_grid,y_grid)}, Distance {dist}')
        print(errorTwoCoordinateLists(x_integ, y_integ,x_grid,y_grid))
        # print(dist)


        '''plot'''
        l1,=axs[i].plot(x_grid,y_grid, 'm-',label='MultiscaleCAN')
        l2,=axs[i].plot(x_integ, y_integ, 'g--')
        # axs[i].plot(x_integ_err, y_integ_err, 'r.')
        axs[i].axis('equal')
        # axs[i].set_title(f'CAN Err:{round(errorCAN)}m   Integ Err:{round(errorPathIntegration)}')
        # axs[i].legend(['MultiscaleCAN', 'Naiive Integration'])
    # print(np.array(errors).T)
    # print('')
    # print(np.array(distance).T)
    # print('')
    plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(top=0.93)
    fig.legend((l1, l2), ('Multiscale CAN', 'Ground Truth'),loc="lower center", ncol=2)
    plt.savefig(savepath)




# kinemVelFile='./results/TestEnvironmentFiles/TraverseInfo/testEnvPathVelocities2.npy'
# kinemAngVelFile='./results/TestEnvironmentFiles/TraverseInfo/testEnvPathAngVelocities2.npy'
# vel,angVel=np.load(kinemVelFile), np.load(kinemAngVelFile)
# vel=np.concatenate([np.linspace(0,scales[0]*5,test_length//5), np.linspace(scales[0]*5,scales[1]*5,test_length//5), np.linspace(scales[1]*5,scales[2]*5,test_length//5), np.linspace(scales[2]*5,scales[3]*5,test_length//5), np.linspace(scales[3]*5,scales[4]*5,test_length//5)])

index=0
outfilePart='./results/TestEnvironmentFiles/TraverseInfo/BerlineEnvPath'
outfile=outfilePart+f'{index}.npz'
traverseInfo=np.load(outfile, allow_pickle=True)
vel,angVel,truePos, startPose=traverseInfo['speeds'], traverseInfo['angVel'], traverseInfo['truePos'], traverseInfo['startPose']

scales=[0.25,1,4,16]
# scales=[1]
if len(vel)<100:
    test_length=len(vel)
else:
    test_length=100

savePath="./results/GIFs/BerlinPathMultiscaleAttractor4scales2.gif" 
np.random.seed(index)
angVel=[np.deg2rad(3.6)]*test_length
# vel=np.concatenate((np.random.uniform(0,2,test_length//2) , np.random.uniform(4,40,test_length//2)))
vel=np.concatenate(([0.4]*(test_length//4),[1.75]*(test_length//4),[7]*(test_length//4),[32]*(test_length//4)))
headDirectionAndPlaceNoWrapNetAnimate(scales, test_length, vel, angVel,savePath)

'''Running 18 paths with Multiscale CAN'''
def runningAllPathsFromACity(length,outfilePart,pathFile,randomSeedVariation):
    for index in range(length):
        outfile=outfilePart+f'{index}.npz'
        traverseInfo=np.load(outfile, allow_pickle=True)
        vel,angVel,truePos, startPose=traverseInfo['speeds'], traverseInfo['angVel'], traverseInfo['truePos'], traverseInfo['startPose']

        # scales=[0.25,1,4,16]
        scales=[1]
        if len(vel)<1000:
            test_length=len(vel)
        else:
            test_length=1000

        # iterPerScale=int(np.ceil(test_length/4))
        # vel=np.concatenate([np.linspace(0,scales[0]*5,iterPerScale), np.linspace(scales[0]*5,scales[1]*5,iterPerScale), np.linspace(scales[1]*5,scales[2]*5,iterPerScale), np.linspace(scales[2]*5,scales[3]*5,iterPerScale)])
        np.random.seed(index*randomSeedVariation)
        vel=np.random.uniform(0,20,test_length) 
        headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,pathfile+f'{index}.npy', printing=False)

# Brisbane = 7, Japan =4, NYC=7, Berlin=9

length=18
outfile='./results/TestEnvironmentFiles/TraverseInfo/BerlineEnvPath'
pathfile='./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathwithSpeeds0to20_'
savepath='./results/PaperFigures/MultipathTrackingSpeeds0to20_Berlin.pdf'
# pathfile='./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathwithSpeedsOG_'
# savepath='./results/TestEnvironmentFiles/MultipathTrackingSpeedsOG_Berlin.png'
# pathfile='./results/TestEnvironmentFiles/SinglescaleCAN/TestMultiscalePathwithSpeeds_'
# savepath='./results/TestEnvironmentFiles/SinglepathTrackingSpeeds_Berlin.png'
# runningAllPathsFromACity(length,outfile,pathfile,1)
# plotSavedMultiplePaths(length,outfile,pathfile,savepath,6,3,1)

length=7
outfile='./results/TestEnvironmentFiles/TraverseInfo/NYC'
pathfile='./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscaleNYC'
savepath='./results/TestEnvironmentFiles/MultipathTrackingSpeeds0to20_NYC.png'
# pathfile='./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscaleNYC_OG'
# savepath='./results/TestEnvironmentFiles/MultipathTrackingOG_NYC.png'
# pathfile='./results/TestEnvironmentFiles/SinglescaleCAN/TestMultiscaleNYC_'
# savepath='./results/TestEnvironmentFiles/SinglepathTracking_NYC.png'
# runningAllPathsFromACity(length,outfile,pathfile,2)
# plotSavedMultiplePaths(length,outfile,pathfile,savepath,3,3,2)

length=4
outfile='./results/TestEnvironmentFiles/TraverseInfo/Japan'
pathfile='./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscaleJapan'
savepath='./results/TestEnvironmentFiles/MultipathTrackingSpeeds0to20_Japan.png'
# pathfile='./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscaleJapanOG'
# savepath='./results/TestEnvironmentFiles/MultipathTrackingSpeedsOG_Japan.png'
# pathfile='./results/TestEnvironmentFiles/SinglescaleCAN/TestMultiscaleJapan'
# savepath='./results/TestEnvironmentFiles/SinglepathTrackingSpeeds_Japan.png'
# runningAllPathsFromACity(length,outfile,pathfile,3)
# plotSavedMultiplePaths(length,outfile,pathfile,savepath,2,2,3)

length=7
outfile='./results/TestEnvironmentFiles/TraverseInfo/Brisbane'
pathfile='./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscaleBrisbane'
savepath='./results/TestEnvironmentFiles/MultipathTrackingSpeeds0to20_Brisbane.png'
# pathfile='./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscaleBrisbaneOG'
# savepath='./results/TestEnvironmentFiles/MultipathTrackingSpeedsOG_Brisbane.png'
# pathfile='./results/TestEnvironmentFiles/SinglescaleCAN/TestMultiscaleBrisbane'
# savepath='./results/TestEnvironmentFiles/SinglepathTrackingSpeeds_Brisbane.png'
# runningAllPathsFromACity(length,outfile,pathfile,4)
# plotSavedMultiplePaths(length,outfile,pathfile,savepath,3,3,4)

'''Cumalitive error Distribution'''
# outfile=f'./results/TestEnvironmentFiles/TraverseInfo/BerlineEnvPath{1}.npz'
# traverseInfo=np.load(outfile, allow_pickle=True)
# angVel= traverseInfo['angVel']
# if len(angVel)<1000:
#     test_length=len(angVel)
# else:
#     test_length=1000
# vel=np.random.uniform(0,20,test_length)

scales=[1]
singlePath='./results/TestEnvironmentFiles/CumaliSingle_20speed_berlin1.npy'
# single_x,single_y=headDirectionAndPlaceNoWrapNet(scales,test_length, vel, angVel,singlePath,plot=False, printing=False)

scales=[0.25,1,4,16]
multiPath='./results/TestEnvironmentFiles/CumaliMulti_20speed_berlin1.npy'
# multi_x,multi_y=headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,multiPath,plot=False, printing=False)

x_gridM,y_gridM, x_integM, y_integM, x_integ_err, y_integ_err= np.load(multiPath)
x_gridS,y_gridS, x_integS, y_integS, x_integ_err, y_integ_err= np.load(singlePath)
multipleError=errorTwoCoordinateLists(x_integM, y_integM,x_gridM,y_gridM,errDistri=True)
singleError=errorTwoCoordinateLists(x_integS, y_integS,x_gridS,y_gridS,errDistri=True)

# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(3.2, 1.7))
# fig.legend(['MultiscaleCAN', 'Grid'])
# fig.tight_layout()
# fig.suptitle('ATE Error Over Time',y=1.07)
# ax1.bar(np.arange(999),singleError, color='royalblue', width=1)
# ax1.set_xlabel('Berlin Trajectories')
# ax1.set_ylabel('ATE [m]')
# ax2.bar(np.arange(999),multipleError,  color='mediumorchid',width=1)
# ax2.set_xlabel('Berlin Trajectories')
# plt.subplots_adjust(top=0.9)
# fig.legend(('Single scale','Multiscale'),loc='upper center', bbox_to_anchor=(0.5,1.03),ncol=2)
# plt.savefig('./results/PaperFigures/errorOverTim.pdf')

'''Local Error segments'''
def plotMultiplePathsErrorDistribution(length,pathfileSingle,pathfileMulti,savepath,randomSeedVariation):
    fig = plt.subplots(1,1,figsize=(3.2, 1.8))
    # fig.legend(['MultiscaleCAN', 'Grid'])
    # fig.tight_layout()
    # fig.suptitle('ATE within 18 Trajectories through Berlin',y=1.05)

    errorSingle,erroMulti=[],[]
    for i in range(length):
        x_grid,y_grid,x_integ, y_integ, x_integ_err, y_integ_err = np.load(pathfileSingle+f'{i}.npy')
        errorSingle.append(errorTwoCoordinateLists(x_integ, y_integ,x_grid,y_grid))
 
    for i in range(length):
        x_grid,y_grid,x_integ, y_integ, x_integ_err, y_integ_err = np.load(pathfileMulti+f'{i}.npy')
        erroMulti.append(errorTwoCoordinateLists(x_integ, y_integ,x_grid,y_grid))
 
    # ax1.bar(np.arange(length),errorSingle, color='royalblue', width=1)
    # ax1.set_xlabel('Berlin Trajectories')
    # ax1.set_ylabel('ATE [m]')
    # ax2.bar(np.arange(length),erroMulti,  color='mediumorchid',width=1)
    # ax2.set_xlabel('Berlin Trajectories')
    # plt.subplots_adjust(top=0.9)
    # fig.legend(('Single scale','Multiscale'),loc='upper center', bbox_to_anchor=(0.5,1.01),ncol=2)
    # plt.figure(figsize=(2.7,2))
    plt.subplots_adjust(bottom=0.2)
    plt.bar(np.arange(length),errorSingle,color='royalblue')
    plt.bar(np.arange(length),erroMulti, color='mediumorchid')
    plt.legend(['Single-scale', 'Multiscale'],ncol=2,loc='best')
    plt.xlabel('Berlin Trajectories',y=0)
    plt.ylabel('ATE [m]')
    plt.ylim([0,12000])
    plt.title('ATE within 18 Trajectories through Berlin')
    # plt.tight_layout()
    
    plt.savefig(savepath)


# length=18
# pathfileMulti='./results/TestEnvironmentFiles/MultiscaleCAN/TestMultiscalePathwithSpeeds0to20_'
# pathfileSingle='./results/TestEnvironmentFiles/SinglescaleCAN/TestMultiscalePathwithSpeeds_'
# savepath='./results/PaperFigures/ErrorDistributionLocalSegmentsOnefig.pdf'
# plotMultiplePathsErrorDistribution(length,pathfileSingle,pathfileMulti,savepath,1)



''' Benefits of Multiscale '''
def mutliVs_single(filepath, index, desiredTestLength):
    outfile=f'./results/TestEnvironmentFiles/TraverseInfo/BerlineEnvPath{index}.npz'
    traverseInfo=np.load(outfile, allow_pickle=True)
    angVel= traverseInfo['angVel']
  
    if len(angVel)<desiredTestLength:
        test_length=len(angVel)
    else:
        test_length=desiredTestLength

    errors=[]
    for i in range(1,21):
        vel=np.random.uniform(0,i,test_length)
        true_x,true_y=pathIntegration(vel,angVel)

        scales=[1]
        single_x,single_y=headDirectionAndPlaceNoWrapNet(scales,test_length, vel, angVel,None,plot=False, printing=False)
        singleError=errorTwoCoordinateLists(true_x,true_y, single_x, single_y)

        scales=[0.25,1,4,16]
        multi_x,multi_y=headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,None,plot=False, printing=False)
        multipleError=errorTwoCoordinateLists(true_x,true_y, multi_x, multi_y)

        errors.append([singleError,multipleError])

    np.save(filepath, errors)

# index=0
# filepath=f'./results/TestEnvironmentFiles/MultiscaleVersus SingleScale/Path{index}_singleVSmultiErrors3.npy'
# # mutliVs_single(filepath, index, 500)
# plt.figure(figsize=(2.7,2))
# singleErrors, multipleErrors = zip(*np.load(filepath))
# plt.plot(singleErrors, 'b')
# plt.plot(multipleErrors, 'm')
# plt.legend(['Single-scale', 'Multiscale'])
# plt.xlabel('Maximum velocity within Test Trajectory')
# plt.ylabel('ATE [m] ')
# plt.title('Network Perfomance over Large Velocity Ranges')
# plt.tight_layout()
# plt.savefig('./results/PaperFigures/SingleVsMultiNetworkRMSE.pdf')



''' Kitti Odometry'''
def data_processing(index):
    poses = pd.read_csv(f'./data/dataset/poses/'+index, delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    return gt

def testing_Conversion(sparse_gt):
    # length=len(sparse_gt[:,:,3][:,0])
    data_x=sparse_gt[:, :, 3][:,0]#[:200]
    data_y=sparse_gt[:, :, 3][:,2]#[:200]
    delta1,delta2=[],[]
    for i in range(1,len(data_x)):
        x0=data_x[i-2]
        x1=data_x[i-1]
        x2=data_x[i]
        y0=data_y[i-2]
        y1=data_y[i-1]
        y2=data_y[i]

        delta1.append(np.sqrt(((x2-x1)**2)+((y2-y1)**2))) #translation
        delta2.append((math.atan2(y2-y1,x2-x1)) - (math.atan2(y1-y0,x1-x0)))  

    print(np.array(delta1).shape,np.array(delta2).shape)

    # print(dirs)
    # delta2=np.transpose(np.diff(dirs))
    # print(delta1,delta2)
    np.save('./results/TestEnvironmentFiles/kittiVels.npy', np.array([delta1,delta2]))

# for i in range(10):
sparse_gt=data_processing(f'00.txt')#[0::4]
testing_Conversion(sparse_gt)
# scales=[1,2,4,8,16]
# scales=[0.25,1,4,16]
# scales=[1]
# vel,angVel=np.load('./results/TestEnvironmentFiles/kittiVels.npy')
# if len(vel)<500:
#     test_length=len(vel)
# else:
#     test_length=500

# test_length=len(vel)
# headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,f'./results/TestEnvironmentFiles/kittiPath_nosparse_singleScale.npy', printing=False)

# plotFromSavedArray(f'./results/TestEnvironmentFiles/kittiPath_nosparse.npy','./results/TestEnvironmentFiles/KittiPath7_nosparse_scaleMultipier2.png')

'''PLOTTING SIGNLE vs MULTI'''
# multiPath=f'./results/TestEnvironmentFiles/kittiPath_nosparse.npy'
# singlePath=f'./results/TestEnvironmentFiles/kittiPath_nosparse_singleScale.npy'

# x_gridM,y_gridM, x_integM, y_integM, x_integ_err, y_integ_err= np.load(multiPath)
# x_gridS,y_gridS, x_integS, y_integS, x_integ_err, y_integ_err= np.load(singlePath)

# print(errorTwoCoordinateLists(x_integS, y_integS,x_gridS,y_gridS))
# print(np.sum(vel))

# fig, (ax1,ax2) = plt.subplots(1,2,figsize=(3.4, 1.9))
# fig.suptitle('Multiscale vs. Single Scale Kitti Odometry Path')
# plt.subplots_adjust(bottom=0.2)
# l2,=ax1.plot(x_gridM, y_gridM, 'm-')
# l1,=ax1.plot(x_integM, y_integM, 'g--')
# ax1.axis('equal')

# l3,=ax2.plot(x_gridS, y_gridS, 'b-')
# l4,=ax2.plot(x_integS, y_integS, 'g--')
# ax2.axis('equal')

# fig.legend((l2, l3, l4), ('Multiscale CAN', 'Single scale CAN','Ground Truth'),loc='lower center',ncol=3)
# plt.savefig('./results/TestEnvironmentFiles/KittiSinglevsMulti.pdf')

