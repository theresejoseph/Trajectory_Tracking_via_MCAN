
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
# from DataHandling import saveOrLoadNp  
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

        for i in range(len(scales)-2):
            if input>scales[i]*swap_val and input<=scales[i+1]*swap_val:
                scale_idx=i+1
        
        if input>scales[-2]*swap_val:
            scale_idx=len(scales)-1

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
        x_grid_expect+=wrap_cols_cs*N*scales[cs_idx]
        y_grid_expect+=wrap_rows_cs*N*scales[cs_idx]
    


    
        # if np.any(wrap_cols_cs!=0):
        #     print(f"------------------------------------------------------------------------------------------------------wrap_cols {wrap_cols_cs}, {scales[cs_idx]}")
        # if np.any(wrap_rows_cs!=0):
        #     print(f"------------------------------------------------------------------------------------------------------wrap_rows {wrap_rows_cs}, {scales[cs_idx]}")

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



'''####################################################################################################################################'''

'''Running All Paths in a City SingleScale and MultiScale'''
def runningAllPathsFromACity(City, scaleType, run=False, plotting=False):
    #scaleType = Single or Multi; 
    #City = Berlin or Japan or Brisbane or NewYork;
    if City=='Berlin':
        length=18
        outfilePart='./Datasets/CityScaleSimulatorVelocities/Berlin/BerlineEnvPath'
        pathfile=f'./Results/Berlin/CAN_Experiment_Output_{scaleType}/TestingTrackswithSpeeds0to20_Path'
        savepath=f'./Results/Berlin/TestingTrackswithSpeeds0to20_{scaleType}scale.png'
        figrows,figcols=6,3
        randomSeedVariation=1

    elif City=='Japan':
        length=4
        outfilePart='./Datasets/CityScaleSimulatorVelocities/Tokyo/Japan'
        pathfile=f'./Results/Tokyo/CAN_Experiment_Output_{scaleType}/TestingTrackswithSpeeds0to20_Path'
        savepath=f'./Results/Tokyo/TestingTrackswithSpeeds0to20_{scaleType}scale.png'
        figrows,figcols=2,2
        randomSeedVariation=2
            
    elif City=='Brisbane':
        length=7
        outfilePart='./Datasets/CityScaleSimulatorVelocities/Brisbane/Brisbane'
        pathfile=f'./Results/Brisbane/CAN_Experiment_Output_{scaleType}/TestingTrackswithSpeeds0to20_Path'
        savepath=f'./Results/Brisbane/TestingTrackswithSpeeds0to20_{scaleType}scale.png'
        figrows,figcols=3,3
        randomSeedVariation=3

    elif City=='NewYork':
        length=7
        outfilePart='./Datasets/CityScaleSimulatorVelocities/NewYork/NYC'
        pathfile=f'./Results/NewYork/CAN_Experiment_Output_{scaleType}/TestingTrackswithSpeeds0to20_Path'
        savepath=f'./Results/NewYork/TestingTrackswithSpeeds0to20_{scaleType}scale.png'
        figrows,figcols=3,3
        randomSeedVariation=4

    if run==True:
        for index in range(length):
            outfile=outfilePart+f'{index}.npz'
            traverseInfo=np.load(outfile, allow_pickle=True)
            vel,angVel,truePos, startPose=traverseInfo['speeds'], traverseInfo['angVel'], traverseInfo['truePos'], traverseInfo['startPose']

            if scaleType=='Multi':
                scales=[0.25,1,4,16]
            elif scaleType=='Single':
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
    
    if plotting==True:
        fig, axs = plt.subplots(figrows,figcols,figsize=(4, 4))
        fig.legend([f'{scaleType}scaleCAN', 'Grid'])
        fig.tight_layout(pad=1)
        fig.suptitle(f'{scaleType}scale Trajectory Tracking through {City} with CAN')
        axs=axs.ravel()
        for i in range(length):
            '''error'''
            x_grid,y_grid,x_integ, y_integ, x_integ_err, y_integ_err = np.load(pathfile+f'{i}.npy')

            '''distance'''
            # traverseInfo=np.load(outfile+f'{i}.npz', allow_pickle=True)
            # dist=np.sum(traverseInfo['speeds'])
            np.random.seed(i*randomSeedVariation)
            vel=np.random.uniform(0,20,len(x_grid)) 
            dist=np.sum(vel)
            print(errorTwoCoordinateLists(x_integ, y_integ,x_grid,y_grid))

            '''color dictionary'''
            color={'Multi': 'm-', 'Single': 'b-'}

            '''plot'''
            l1,=axs[i].plot(x_grid,y_grid, color[scaleType],label=f'{scaleType}scaleCAN')
            l2,=axs[i].plot(x_integ, y_integ, 'g--')
            axs[i].axis('equal')
            
        plt.subplots_adjust(bottom=0.1)
        plt.subplots_adjust(top=0.93)
        fig.legend((l1, l2), (f'{scaleType}scale CAN', 'Ground Truth'),loc="lower center", ncol=2)
        plt.savefig(savepath)

# scaleType='Single'
# runningAllPathsFromACity('Japan', scaleType, run=False, plotting=True)
# runningAllPathsFromACity('NewYork', scaleType, run=False, plotting=True)
# runningAllPathsFromACity('Brisbane', scaleType,run=False, plotting=True)
# runningAllPathsFromACity('Berlin', scaleType, run=False, plotting=True)

# scaleType='Multi'
# runningAllPathsFromACity('Japan', scaleType, run=False, plotting=True)
# runningAllPathsFromACity('NewYork', scaleType, run=False, plotting=True)
# runningAllPathsFromACity('Brisbane', scaleType,run=False, plotting=True)
# runningAllPathsFromACity('Berlin', scaleType, run=False, plotting=True)


''' Multi versus Single over Large Velocity Range'''
def mutliVs_single(filepath, index, desiredTestLength, run=False, plotting=False ):
    outfile=f'./Datasets/CityScaleSimulatorVelocities/Berlin/BerlineEnvPath{index}.npz'
    traverseInfo=np.load(outfile, allow_pickle=True)
    angVel= traverseInfo['angVel']
  
    if len(angVel)<desiredTestLength:
        test_length=len(angVel)
    else:
        test_length=desiredTestLength

    if run==True:
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

    if plotting==True:
        plt.figure(figsize=(2.7,2))
        singleErrors, multipleErrors = zip(*np.load(filepath))
        plt.plot(singleErrors, 'b')
        plt.plot(multipleErrors, 'm')
        plt.legend(['Single-scale', 'Multiscale'])
        plt.xlabel('Maximum velocity within Test Trajectory')
        plt.ylabel('ATE [m] ')
        plt.title('Network Perfomance over Large Velocity Ranges')
        plt.tight_layout()
        plt.savefig(f'./Results/Berlin/MultivsSingleErrors_Path{index}.png')

# index=0
# filepath=f'./Results/Berlin/MultivsSingleErrors_Path{index}.npy'
# mutliVs_single(filepath, index, 500, run=False, plotting=True) 




'''Cumalitive error Distribution Single vs Multi'''
def CumalativeError_SinglevsMulti(singlePath, multiPath, run=False, plotting=False):
    outfile=f'./Datasets/CityScaleSimulatorVelocities/Berlin/BerlineEnvPath{1}.npz'
    traverseInfo=np.load(outfile, allow_pickle=True)
    angVel= traverseInfo['angVel']
    if len(angVel)<1000:
        test_length=len(angVel)
    else:
        test_length=1000
    vel=np.random.uniform(0,20,test_length)

    if run==True:
        scales=[1]
        single_x,single_y=headDirectionAndPlaceNoWrapNet(scales,test_length, vel, angVel,singlePath,plot=False, printing=False)
        scales=[0.25,1,4,16]
        multi_x,multi_y=headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,multiPath,plot=False, printing=False)
    if plotting==True:
        x_gridM,y_gridM, x_integM, y_integM, x_integ_err, y_integ_err= np.load(multiPath)
        x_gridS,y_gridS, x_integS, y_integS, x_integ_err, y_integ_err= np.load(singlePath)
        multipleError=errorTwoCoordinateLists(x_integM, y_integM,x_gridM,y_gridM,errDistri=True)
        singleError=errorTwoCoordinateLists(x_integS, y_integS,x_gridS,y_gridS,errDistri=True)
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(3.2, 1.7))
        fig.legend(['MultiscaleCAN', 'Grid'])
        fig.tight_layout()
        fig.suptitle('ATE Error Over Time',y=1.07)
        ax1.bar(np.arange(999),singleError, color='royalblue', width=1)
        ax1.set_xlabel('Berlin Trajectories')
        ax1.set_ylabel('ATE [m]')
        ax2.bar(np.arange(999),multipleError,  color='mediumorchid',width=1)
        ax2.set_xlabel('Berlin Trajectories')
        plt.subplots_adjust(top=0.9)
        fig.legend(('Single scale','Multiscale'),loc='upper center', bbox_to_anchor=(0.5,1.03),ncol=2)
        plt.savefig('./Results/Berlin/CumalitiveError_Path1_SinglevsMulti.png')

# singlePath='./Results/Berlin/CumalativeError_Path1_SingleScale.npy'
# multiPath='./Results/Berlin/CumalativeError_Path1_MultiScale.npy'
# CumalativeError_SinglevsMulti(singlePath, multiPath, run=False, plotting=True)




'''Local Error segments Berlin'''
def plotMultiplePathsErrorDistribution():
    length=18
    pathfileSingle=f'./Results/Berlin/CAN_Experiment_Output_Single/TestingTrackswithSpeeds0to20_Path'
    pathfileMulti=f'./Results/Berlin/CAN_Experiment_Output_Multi/TestingTrackswithSpeeds0to20_Path'


    fig,axs = plt.subplots(1,1,figsize=(3.2, 1.8)) 
    errorSingle,erroMulti=[],[]
    for i in range(length):
        x_grid,y_grid,x_integ, y_integ, x_integ_err, y_integ_err = np.load(pathfileSingle+f'{i}.npy')
        errorSingle.append(errorTwoCoordinateLists(x_integ, y_integ,x_grid,y_grid))
 
    for i in range(length):
        x_grid,y_grid,x_integ, y_integ, x_integ_err, y_integ_err = np.load(pathfileMulti+f'{i}.npy')
        erroMulti.append(errorTwoCoordinateLists(x_integ, y_integ,x_grid,y_grid))
 
    plt.subplots_adjust(bottom=0.2)
    axs.bar(np.arange(length),errorSingle,color='royalblue')
    axs.bar(np.arange(length),erroMulti, color='mediumorchid')
    axs.legend(['Single-scale', 'Multiscale'],ncol=2,loc='best')
    axs.set_xlabel('Berlin Trajectories',y=0)
    axs.set_ylabel('ATE [m]')
    # axs.set_ylim([0,12000])
    axs.set_title('ATE within 18 Trajectories through Berlin')
    
    savepath=f'./Results/Berlin/LocalSegmentError_AllPaths_SinglevsMulti.png'
    plt.savefig(savepath)

# plotMultiplePathsErrorDistribution()




''' Kitti GT Poses'''
def data_processing(index):
    
    poses = pd.read_csv(f'./Datasets/kittiOdometryPoses/{index}.txt', delimiter=' ', header=None)
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
    
    # extracting velocities from poses 
    sparse_gt=gt
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

    np.save(f'./Datasets/kittiVelocities/kittiVels_{index}.npy', np.array([delta1,delta2]))

# for i in range(10):
#     data_processing(f'0{i}')
# data_processing('10')


def runningAllPathsFromKittiGT(length, scaleType, run=False, plotting=False):
    #scaleType = Single or Multi; 
    
    pathfile=f'./Results/Kitti/CAN_Experiment_Output_{scaleType}/TestingTracksfromGTpose_'
    savepath=f'./Results/Kitti/TestingTracksfromGTpose_{scaleType}scale.png'

    if run==True:
        for index in range(length):
            if index==10:
                velFile=f'./Datasets/kittiVelocities/kittiVels_{index}.npy'
            else: 
                velFile=f'./Datasets/kittiVelocities/kittiVels_0{index}.npy'
            vel,angVel=np.load(velFile)

            if scaleType=='Multi':
                scales=[0.25,1,4,16]
            elif scaleType=='Single':
                scales=[1]

            if len(vel)<1000:
                test_length=len(vel)
            else:
                test_length=1000
            test_length=len(vel)

            headDirectionAndPlaceNoWrapNet(scales, test_length, vel, angVel,pathfile+f'{index}.npy', printing=False)
            print(f'Finished vels {index}')
    
    if plotting==True:
        fig, axs = plt.subplots(4,3,figsize=(4, 4))
        fig.legend([f'{scaleType}scaleCAN', 'Grid'])
        fig.tight_layout(pad=1)
        fig.suptitle(f'{scaleType}scale Trajectory Tracking for KittiGT_poses with CAN')
        axs=axs.ravel()
        for i in range(length):
            '''error'''
            x_grid,y_grid,x_integ, y_integ, x_integ_err, y_integ_err = np.load(pathfile+f'{i}.npy')

            '''color dictionary'''
            color={'Multi': 'm-', 'Single': 'b-'}

            '''plot'''
            l1,=axs[i].plot(x_grid,y_grid, color[scaleType],label=f'{scaleType}scaleCAN')
            l2,=axs[i].plot(x_integ, y_integ, 'g--')
            axs[i].axis('equal')
            
        plt.subplots_adjust(bottom=0.1)
        plt.subplots_adjust(top=0.93)
        fig.legend((l1, l2), (f'{scaleType}scale CAN', 'Ground Truth'),loc="lower center", ncol=2)
        plt.savefig(savepath)

# runningAllPathsFromKittiGT(11, 'Multi', run=True, plotting=True)
# runningAllPathsFromKittiGT(11, 'Single', run=True, plotting=True)


def plotKittiGT_singlevsMulti(index):
    multiPath=f'./Results/Kitti/CAN_Experiment_Output_Multi/TestingTracksfromGTpose_{index}.npy'
    singlePath=f'./Results/Kitti/CAN_Experiment_Output_Multi/TestingTracksfromGTpose_{index}.npy'
    x_gridM,y_gridM, x_integM, y_integM, x_integ_err, y_integ_err= np.load(multiPath)
    x_gridS,y_gridS, x_integS, y_integS, x_integ_err, y_integ_err= np.load(singlePath)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(3.4, 1.9))
    fig.suptitle('Multiscale vs. Single Scale Kitti Odometry Path')
    plt.subplots_adjust(bottom=0.2)
    l2,=ax1.plot(x_gridM, y_gridM, 'm-')
    l1,=ax1.plot(x_integM, y_integM, 'g--')
    ax1.axis('equal')

    l3,=ax2.plot(x_gridS, y_gridS, 'b-')
    l4,=ax2.plot(x_integS, y_integS, 'g--')
    ax2.axis('equal')

    fig.legend((l2, l3, l4), ('Multiscale CAN', 'Single scale CAN','Ground Truth'),loc='lower center',ncol=3)
    plt.savefig(f'./Results/Kitti/KittiSinglevsMulti_{index}.png')

# plotKittiGT_singlevsMulti(0)