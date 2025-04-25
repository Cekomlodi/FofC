# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 22:53:31 2025

@author: cekom
"""

def PostAnim(df, animate = True, Plot_title='Plot Title', downsample = 1, 
             VideoName = 'PosteriorAnimation.mp4', titlesize = 15, fps = 30, 
             cmap = 'BuPu', linecolor = 'darkorchid'):
    '''

    Parameters
    ----------
    df : Pandas DataFrame containing all desired parameters already altered to
        desired units and scales
        
    animate : Do you want an animation or just a plot? The default is True which animates it.
    
    Plot_title : The default is 'Plot Title'.
    
    downsample : How many iterations do you want to skip for the purpose of animation
        The default is 1, which takes every point as an individual frame.
        
    VideoName : The default is 'PosteriorAnimation.mp4'. .gif is also acceptable
    
    titlesize : Sometimes plots get really big and the title sizes need to increase too. 
        The default is 15.
    fps : Custom fps of mp4 output. The default is 30.
    
    cmap : Colormap for all corner 2d histograms. The default is 'BuPu'.
    
    linecolor : Color for all chains and diagonal histograms. The default is 'darkorchid'.

    Returns
    -------
    Outputs static plot to default viewer, saves mp4 to working location. If mp4 name
    matches another file, it will rewrite it!

    '''
    
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FFMpegWriter

    
    num_chains = len(df.columns)
    column_list = list(df.columns)
    
    #Setting up the graph environment
    width_ratios = np.concatenate(([4], np.ones(num_chains)))
    height_ratios = np.ones(num_chains)
    gridspec_height = num_chains
    gridspec_width = num_chains + 1
    total_entries = gridspec_height*gridspec_width
    arr = np.arange(1,total_entries+1, 1)
    general_layout = arr.reshape(gridspec_height, gridspec_width)
    
    
    #Change each entry to a string so we can make the corner
    general_string = general_layout.astype(str)
    
    #make the corner :(
    L = len(general_string[0])-2
    
    for i in range(0,len(general_string)):
        for k in range(0,L):
            general_string[i][k+i+2] = '.'
        L = L-1
        

    layout_pattern = general_string
    print(layout_pattern)
    
    
    gs_kw = dict(width_ratios= width_ratios, height_ratios= height_ratios)
    fig, axd = plt.subplot_mosaic(layout_pattern,
                                  gridspec_kw=gs_kw, figsize=(4*num_chains, 2*num_chains), constrained_layout=True)
    
    #static scary corner plots with chains
        
        #All of the chain x and y parameters to avoid growing plots in the future
    for i in range(0, num_chains): 
        
        axd[str(layout_pattern[i][0])].set_xlim(0, len(df[column_list[i]]))
        axd[str(layout_pattern[i][0])].set_ylim(min(df[column_list[i]]), max(df[column_list[i]]))
        
        if i < num_chains-1:
            axd[str(layout_pattern[i][0])].set_xticklabels([])
            #axd[str(layout_pattern[i][0])].tick_params(bottom = False)
            
        
    axd[layout_pattern[num_chains-1][0]].set_xlabel('Chain Iteration', fontsize = titlesize)
    fig.suptitle(Plot_title, horizontalalignment = 'right', verticalalignment = 'top', fontsize = titlesize)
    
    #Set axis labels
    for i in range(0, num_chains):
        axd[str(layout_pattern[i][0])].plot(np.arange(0,len(df[column_list[i]]), 1), df[column_list[i]], linewidth = .5, color = linecolor)
        print('Chain {} done'.format(i+1))
        axd[layout_pattern[num_chains-1][i+1]].set_xlabel('{0}'.format(column_list[i]), fontsize = titlesize, labelpad=15)
        axd[layout_pattern[i][0]].set_ylabel('{0}'.format(column_list[i]), fontsize = titlesize)
        
    #Plot all of the diagonal Histograms
    Corner_index = 0
    for i in range(0, num_chains):
        hist, bins = np.histogram(df[column_list[i]], density = True)
        axd[layout_pattern[i][Corner_index+1]].stairs(hist, bins, color = linecolor, fill = True)
        axd[layout_pattern[i][Corner_index+1]].tick_params(left = False)
    
        print('Histogram {} done'.format(i+1))
        
        if i < num_chains-1:
            axd[layout_pattern[i][Corner_index+1]].set_xticklabels([])
            axd[layout_pattern[i][Corner_index+1]].set_yticklabels([])
        
        Corner_index = Corner_index+1
        
    #Plot the inner histograms
    for i in range(1, num_chains):
        for k in range(1, num_chains):
            if k <= i:
                axd[layout_pattern[i][k]].hist2d(df[column_list[k-1]], df[column_list[i]], cmap = cmap, density = True)
            
                if k > 0:
                        axd[layout_pattern[i][k+1]].set_yticklabels([])
                        #axd[layout_pattern[i][k]].tick_params(left = False)
            
        if i < num_chains-1:
            for k in range(1, num_chains):
                if k <= i:
                    axd[layout_pattern[i][k]].set_xticklabels([])
                    #axd[layout_pattern[i][k]].tick_params(bottom = False)

            
        
        print('2dHist Column {} Done'.format(i))
    
    plt.show()        
    
    ##########################################################################
    #Start animating all of the things...
   ##########################################################################
   
    if animate == True:
        #set up the environment
        num_chains = len(df.columns)
        column_list = list(df.columns)
        
        #Setting up the graph environment
        width_ratios = np.concatenate(([4], np.ones(num_chains)))
        height_ratios = np.ones(num_chains)
        gridspec_height = num_chains
        gridspec_width = num_chains + 1
        total_entries = gridspec_height*gridspec_width
        arr = np.arange(1,total_entries+1, 1)
        general_layout = arr.reshape(gridspec_height, gridspec_width)
        
        
        #Change each entry to a string so we can make the corner
        general_string = general_layout.astype(str)
        
        #make the corner:
        L = len(general_string[0])-2
        
        for i in range(0,len(general_string)):
            for k in range(0,L):
                general_string[i][k+i+2] = '.'
            L = L-1
            

        layout_pattern = general_string
        
        
        gs_kw = dict(width_ratios= width_ratios, height_ratios= height_ratios)
        fig, axd = plt.subplot_mosaic(layout_pattern,
                                      gridspec_kw=gs_kw, figsize=(4*num_chains, 2*num_chains), constrained_layout=True)
        
        
        
        #start making frames
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'))
        
        from tqdm import tqdm
        
        with writer.saving(fig, VideoName, 100):
        #Try without Setting Up blank graph lines and plots
        
            for q in tqdm (range(0,len(df[column_list[0]])+1)):
                
                if q%downsample == 0:
                    
                    axd[layout_pattern[num_chains-1][0]].set_xlabel('Chain Iteration')
                    fig.suptitle(Plot_title, horizontalalignment = 'right', verticalalignment = 'top')
                    Corner_index = 0
                    
                    #Set axis labels
                    for i in range(0, num_chains):
                        axd[layout_pattern[num_chains-1][i+1]].set_xlabel('{0}'.format(column_list[i]), labelpad=15)
                        axd[layout_pattern[i][0]].set_ylabel('{0}'.format(column_list[i]))

                    #Setting lengths for the chains, no q iteration
                    for i in range(0, num_chains):
                        axd[str(layout_pattern[i][0])].set_xlim(0, len(df[column_list[i]]))
                        axd[str(layout_pattern[i][0])].set_ylim(min(df[column_list[i]]), max(df[column_list[i]]))
                        
                        if i < num_chains-1:
                            axd[str(layout_pattern[i][0])].set_xticklabels([])
                            #axd[str(layout_pattern[i][0])].tick_params(bottom = False)
                            
                        #plots with q dependence
                        #Chains
                        axd[str(layout_pattern[i][0])].plot(np.arange(0,len(df[column_list[i]].head(q)), 1), df[column_list[i]].head(q), linewidth = .5, color = linecolor)
                        
                        #Corner Histogram overall parameters
                        hist, bins = np.histogram(df[column_list[i]])
                        norm = hist/np.absolute((sum(hist)*(bins[1]-bins[0])))
                        
                        axd[layout_pattern[i][Corner_index+1]].set_xlim(min(bins), max(bins))
                        axd[layout_pattern[i][Corner_index+1]].set_ylim(0, max(norm)+.02)
                        
                        #Plot all of the diagonal Histograms
                        finalhist, finalbins = np.histogram(df[column_list[i]].head(q))                    
                        normal = finalhist/np.absolute((sum(finalhist)*(finalbins[1]-finalbins[0])))
                        axd[layout_pattern[i][Corner_index+1]].stairs(normal, finalbins, color = linecolor, fill = True)
                        axd[layout_pattern[i][Corner_index+1]].tick_params(left = False)
                        
                        if i < num_chains-1:
                            axd[layout_pattern[i][Corner_index+1]].set_xticklabels([])
                            axd[layout_pattern[i][Corner_index+1]].set_yticklabels([])
                            #axd[layout_pattern[i][Corner_index+1]].tick_params(bottom = False)
                        
                        Corner_index = Corner_index+1
                    
                    #Plot all of the 2dhistograms
                    for i in range(1, num_chains):
                        for k in range(1, num_chains):
                            if k <= i:   
                                axd[layout_pattern[i][k]].hist2d(df[column_list[k-1]].head(q), df[column_list[i]].head(q), cmap = cmap)
                            
                                if k > 0:
                                    axd[layout_pattern[i][k+1]].set_yticklabels([])
                        
                        if i < num_chains-1:
                            for k in range(1, num_chains):
                                if k <= i:
                                    axd[layout_pattern[i][k]].set_xticklabels([])
                                    #axd[layout_pattern[i][k]].tick_params(bottom = False)
                                    
                    writer.grab_frame()
                    
                    #Clear the stair histogram plots because they're tempermental? Yeah.
                    Corner_Index = 0
                    for i in range(0, num_chains):
                        axd[layout_pattern[i][Corner_Index+1]].clear()
                        Corner_Index = Corner_Index+1
                        
                    
                else:
                    continue
        
    return    
            

#%%Read in LisaCatTools

from lisacattools.catalog import GWCatalog
from lisacattools.catalog import GWCatalogs
from lisacattools.catalog import GWCatalogType
import corner
import numpy as np

#define path names
catpath = '/Users/corinnekomlodi/Desktop/Sangria_v3/drive-download-20250417T184349Z-001'
filename = "sangria_v3_ucb.h5"

#Call in catalogs
catalogs = GWCatalogs.create(GWCatalogType.UCB, catpath, filename)
final_catalog = catalogs.get_last_catalog()
detections_attr = final_catalog.get_attr_detections()
detections =  final_catalog.get_detections(detections_attr)

sourceId = detections.index.values[
    np.argmin(np.abs(np.array(detections["SNR"])-detections["SNR"].max()))]
detections.loc[[sourceId], ["SNR", "Frequency"]]

samples = final_catalog.get_source_samples(sourceId)

#Define desired parameters
Parameters = ["Frequency", "Frequency Derivative", "Amplitude", "Inclination"]

#Show standard corner for camparrison
fig = corner.corner(samples[Parameters])

#define desired Dataframe
df_LISA = samples[Parameters]

#%% Run the Function

#note, only df is required, while at least downsample and videoname is recommended
PostAnim(df_LISA, animate = False, downsample = 5, titlesize=15, VideoName = 'test_axis.gif', Plot_title = 'LDC0017949155')

 