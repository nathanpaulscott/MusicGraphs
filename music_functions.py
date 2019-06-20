import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import pandas as pd
from itertools import zip_longest
import music21 as ms
from music21 import *
import pickle
import glob
import os
import time
import re
import math
from copy import deepcopy

def pre_process_mid(dir, 
                    recursive=False, 
                    force_cleaning=False, 
                    lower=0, upper=127, 
                    prune=-1,ch_off=[],
                    note_cnt_thd=0.8, 
                    ave_note_thd=1.5, 
                    pitch_window_thd=0.8, 
                    single_scale_confidence_thd=0.85, 
                    force_recalc=False,
                    use_hard_coded_tranpose_file=False,
                    scale_option='normal'):
    #this will go through the midis in the /midi/ dir, parse them to a .pkl with a common key (C)
    
    if use_hard_coded_tranpose_file:
        df = pd.read_csv(dir + 'transpose.csv')
        transpose_dict = {row[1]:row[2] for row in df.itertuples()}

    if force_cleaning and not force_recalc:
        files = [f.replace('\\','/') for f in glob.iglob(dir + '*.pkl',recursive=True) if os.path.isfile(f)]
    else:
        files = [f.replace('\\','/') for f in glob.iglob(dir + '*.mid',recursive=True) if os.path.isfile(f)]
        
    #testing
    #with open('test.csv','w') as fd:
    #    fd.write('song key data\n')

    for f in files:
        dir = os.path.split(f)[0] + '/'
        file = os.path.split(f)[1][:-4]
        print('{0}'.format(file))
        #skip if you get the dummy file
        if file == 'zz.delete' or file[-4:] == '-adj': 
            continue 

        #parse the .mid to .pkl if it has not been done or forced to
        if not os.path.exists(dir + file + '-adj.pkl') or force_recalc or force_cleaning:
            #check if the base .pkl file is there, if it is do not re convert it unless forced to
            if not os.path.exists(dir + file + '.pkl') or force_recalc:
                result = parse_n_save_mid_comb_v2(dir, file)
                if result == 'bad_file':
                    continue
                    
                if not use_hard_coded_tranpose_file:
                    #estimate the scale, must do it from the processed .pkl file, not form the .mid file!!!  not good if from .mid file
                    scale = get_note_freq_from_pkl(dir, file, lower=0, upper=127, prune=-1, ch_off=[], 
                                                   plot=True, scale_option=scale_option)

                    #determine the pitch shift to move it to C
                    shift = {'C':0, 'D-':-1, 'D':-2, 'E-':-3, 'E':-4, 'F':-5, 'G-':-6, 'G':5, 'A-':4, 'A':3, 'B-':2, 'B':1}

                    #testing
                    #with open('test.csv','a') as fd:
                    #    temp = str('scale = {0}, key = {1}, ps = {2}, conf = {3:04.4f}\n'.format(scale['scale'], scale['key'], shift[scale['scale']], scale['conf']))
                    #    fd.write(file + ": " + temp)


                    print('######################################################')
                    print('######################################################')
                    print('key = {0}, ps = {1}, conf = {2:04.4f}'.format(scale['scale'],shift[scale['scale']], scale['conf']))
                    print('######################################################')
                    print('######################################################')

                    #skip file if conf is lower than confidence threshold, i.e. the peice of music is too disparat, 
                    #possibly using non major scales or possibly changing key during the song too much, these cause major issues
                    #for this algorithm, so do not use
                    if single_scale_confidence_thd == -1:
                        #check for the case we do not need to transpose, do nothing
                        pass

                    elif scale['conf'] < single_scale_confidence_thd:
                        #skip file
                        print("skipping file as the peice doesn't fit into the major scale well enough")
                        #write a blank .pkl file
                        adjust_pkl(dir, file, -9999)
                        continue

                    else:
                        #load and transpose the .pkl file
                        adjust_pkl(dir, file, shift[scale['scale']], lower=0, upper=127, prune=-1, ch_off=[])

                else:
                    #load and transpose the .pkl file using the hard coded shift from teh transpose file
                    print("doing hard coded transposing")
                    adjust_pkl(dir, file, transpose_dict[file], lower=0, upper=127, prune=-1, ch_off=[])
            
            #clean the pkl file and save as '-adj.pkl'
            pkl_cleaning(dir, file, lower, upper, note_cnt_thd, ave_note_thd, pitch_window_thd)
            
def build_graph(dir, G, use_raw=False, type='poly', lower=0, upper=127, prune=-1, ch_off=[], truncate=-1, edge_dur_qn_max_abs=12, edge_dur_qn_max_rel=12, pitch_delta_max=12, use_highest_note_only_flag=False, ignore_edge_duration=False):
    #this will go through the adjusted .pkl files in the /midi/ dirand build a graph from them
    #files = [f.replace('\\','/') for f in glob.iglob(dir + '**/' + '*.mid',recursive=True) if os.path.isfile(f)]
    files = [f.replace('\\','/') for f in glob.iglob(dir + '*-adj.pkl', recursive=True) if os.path.isfile(f)]
    for f in files:
        dir = os.path.split(f)[0] + '/'
        file = os.path.split(f)[1][:-8]
        print('{0} ... {1}'.format(dir,file))
        if type == 'poly':
            make_graph_comb(dir, file, G, lower, upper, prune, use_raw)
        else:
            make_graph_single_note_v3(dir, file, G, lower, upper, 
                                      ch_off, use_raw, truncate=truncate, 
                                      edge_dur_qn_max_abs=edge_dur_qn_max_abs, 
                                      edge_dur_qn_max_rel=edge_dur_qn_max_rel, 
                                      pitch_delta_max=pitch_delta_max,
                                      use_highest_note_only_flag=use_highest_note_only_flag,
                                      ignore_edge_duration=ignore_edge_duration)
                                      


## Scale Estimation Code

#utility functions to determine the closest major scale to the scales used in the piece of music
#it also returns the TPQN
#inputs the processed .pkl file, outputs the best match scale in a dict
#to call: get_note_freq(dir, file, lower=0, upper=127, prune = 100, ch_off = [])
#NOTE: filename should have no extension
#NOTE: dir should have forward slashes and should end with a forward slash

#-------------------------------------------------------------------------------------------------
def find_best_us_penta_key(notes):
    #this outputs the most likely us blues scale for keys for the given song notes
    #input => expects a list of tuples with proportions with the keys in the following order
    # A,B-,B,C,D-,D,E-,E,F,G-,G,A-
    rt = 1   #root boost
    scales = {'A' :[rt,0,1,0,1,0,0,1,0,rt,0,0],
              'B-':[0,rt,0,1,0,1,0,0,1,0,rt,0],
              'B' :[0,0,rt,0,1,0,1,0,0,1,0,rt],
              'C' :[rt,0,0,rt,0,1,0,1,0,0,1,0], 
              'D-':[0,rt,0,0,rt,0,1,0,1,0,0,1], 
              'D' :[1,0,rt,0,0,rt,0,1,0,1,0,0],
              'E-':[0,1,0,rt,0,0,rt,0,1,0,1,0],
              'E' :[0,0,1,0,rt,0,0,rt,0,1,0,1],
              'F' :[1,0,0,1,0,rt,0,0,rt,0,1,0],
              'G-':[0,1,0,0,1,0,rt,0,0,rt,0,1],
              'G' :[1,0,1,0,0,1,0,rt,0,0,rt,0],
              'A-':[0,1,0,1,0,0,1,0,rt,0,0,rt]}
    cor = [(x,0.5 + np.corrcoef(y,[x[1] for x in notes])[0][1]/2) for x,y in scales.items()]
    cor = sorted(cor, key = lambda x: -x[1])
    return cor


def find_best_indo_penta_key(notes):
    #this outputs the most likely japanese scale for keys for the given song notes
    #input => expects a list of tuples with proportions with the keys in the following order
    # A,B-,B,C,D-,D,E-,E,F,G-,G,A-
    rt = 1.2   #root boost
    scales = {'A' :[rt,0,1,0,1,0,0,1,0,1,0,0], 
              'B-':[0,rt,0,1,0,1,0,0,1,0,1,0], 
              'B' :[0,0,rt,0,1,0,1,0,0,1,0,1],
              'C' :[1,0,0,rt,0,1,0,1,0,0,1,0],
              'D-':[0,1,0,0,rt,0,1,0,1,0,0,1],
              'D' :[1,0,1,0,0,rt,0,1,0,1,0,0],
              'E-':[0,1,0,1,0,0,rt,0,1,0,1,0],
              'E' :[0,0,1,0,1,0,0,rt,0,1,0,1],
              'F' :[1,0,0,1,0,1,0,0,rt,0,1,0],
              'G-':[0,1,0,0,1,0,1,0,0,rt,0,1],
              'G' :[1,0,1,0,0,1,0,1,0,0,rt,0],
              'A-':[0,1,0,1,0,0,1,0,1,0,0,rt]}
    cor = [(x,0.5 + np.corrcoef(y,[x[1] for x in notes])[0][1]/2) for x,y in scales.items()]
    cor = sorted(cor, key = lambda x: -x[1])
    return cor


def find_best_indo_hepta_key(notes):
    #this outputs the most likely japanese scale for keys for the given song notes
    #input => expects a list of tuples with proportions with the keys in the following order
    # A,B-,B,C,D-,D,E-,E,F,G-,G,A-
    rt = 1.2   #root boost
    scales = {'A' :[rt,1,0,0,1,0,1,1,1,0,0,1], 
              'B-':[1,rt,1,0,0,1,0,1,1,1,0,0], 
              'B' :[0,1,rt,1,0,0,1,0,1,1,1,0],
              'C' :[0,0,1,rt,1,0,0,1,0,1,1,1],
              'D-':[1,0,0,1,rt,1,0,0,1,0,1,1],
              'D' :[1,1,0,0,1,rt,1,0,0,1,0,1],
              'E-':[1,1,1,0,0,1,rt,1,0,0,1,0],
              'E' :[0,1,1,1,0,0,1,rt,1,0,0,1],
              'F' :[1,0,1,1,1,0,0,1,rt,1,0,0],
              'G-':[0,1,0,1,1,1,0,0,1,rt,1,0],
              'G' :[0,0,1,0,1,1,1,0,0,1,rt,1],
              'A-':[1,0,0,1,0,1,1,1,0,0,1,rt]}
    cor = [(x,0.5 + np.corrcoef(y,[x[1] for x in notes])[0][1]/2) for x,y in scales.items()]
    cor = sorted(cor, key = lambda x: -x[1])
    return cor


def find_best_japanese_penta_key(notes):
    #this outputs the most likely japanese scale for keys for the given song notes
    #input => expects a list of tuples with proportions with the keys in the following order
    # A,B-,B,C,D-,D,E-,E,F,G-,G,A-
    rt = 1.2   #root boost
    scales = {'A' :[rt,1,0,0,0,1,0,1,1,0,0,0], 
              'B-':[0,rt,1,0,0,0,1,0,1,1,0,0], 
              'B' :[0,0,rt,1,0,0,0,1,0,1,1,0],
              'C' :[0,0,0,rt,1,0,0,0,1,0,1,1],
              'D-':[1,0,0,0,rt,1,0,0,0,1,0,1],
              'D' :[1,1,0,0,0,rt,1,0,0,0,1,0],
              'E-':[0,1,1,0,0,0,rt,1,0,0,0,1],
              'E' :[1,0,1,1,0,0,0,rt,1,0,0,0],
              'F' :[0,1,0,1,1,0,0,0,rt,1,0,0],
              'G-':[0,0,1,0,1,1,0,0,0,rt,1,0],
              'G' :[0,0,0,1,0,1,1,0,0,0,rt,1],
              'A-':[1,0,0,0,1,0,1,1,0,0,0,rt]}
    cor = [(x,0.5 + np.corrcoef(y,[x[1] for x in notes])[0][1]/2) for x,y in scales.items()]
    cor = sorted(cor, key = lambda x: -x[1])
    return cor

def find_best_japanese_hexa_key(notes):
    #this outputs the most likely japanese scale for keys for the given song notes
    #input => expects a list of tuples with proportions with the keys in the following order
    # A,B-,B,C,D-,D,E-,E,F,G-,G,A-
    rt = 1.2   #root boost
    scales = {'A' :[rt,0,1,1,0,1,0,1,0,0,1,0], 
              'B-':[0,rt,0,1,1,0,1,0,1,0,0,1],
              'B' :[1,0,rt,0,1,1,0,1,0,1,0,0],
              'C' :[0,1,0,rt,0,1,1,0,1,0,1,0],
              'D-':[0,0,1,0,rt,0,1,1,0,1,0,1],
              'D' :[1,0,0,1,0,rt,0,1,1,0,1,0],
              'E-':[0,1,0,0,1,0,rt,0,1,1,0,1],
              'E' :[1,0,1,0,0,1,0,rt,0,1,1,0],
              'F' :[0,1,0,1,0,0,1,0,rt,0,1,1],
              'G-':[1,0,1,0,1,0,0,1,0,rt,0,1],
              'G' :[1,1,0,1,0,1,0,0,1,0,rt,0],
              'A-':[0,1,1,0,1,0,1,0,0,1,0,rt]}
    cor = [(x,0.5 + np.corrcoef(y,[x[1] for x in notes])[0][1]/2) for x,y in scales.items()]
    cor = sorted(cor, key = lambda x: -x[1])
    return cor


def find_best_major_key(notes):
    #this outputs the most likely major scale for major keys for the given song notes
    #input => expects a list of tuples with proportions with the keys in the following order
    # A,B-,B,C,D-,D,E-,E,F,G-,G,A-
    rt = 1.2   #major scale root boost
    major_scales = {'A' :[rt,0,1,0,1,1,0,1,0,1,0,1], 
                    'B-':[1,rt,0,1,0,1,1,0,1,0,1,0], 
                    'B' :[0,1,rt,0,1,0,1,1,0,1,0,1],
                    'C' :[1,0,1,rt,0,1,0,1,1,0,1,0], 
                    'D-':[0,1,0,1,rt,0,1,0,1,1,0,1], 
                    'D' :[1,0,1,0,1,rt,0,1,0,1,1,0], 
                    'E-':[0,1,0,1,0,1,rt,0,1,0,1,1], 
                    'E' :[1,0,1,0,1,0,1,rt,0,1,0,1], 
                    'F' :[1,1,0,1,0,1,0,1,rt,0,1,0], 
                    'G-':[0,1,1,0,1,0,1,0,1,rt,0,1], 
                    'G' :[1,0,1,1,0,1,0,1,0,1,rt,0], 
                    'A-':[0,1,0,1,1,0,1,0,1,0,1,rt]}
    cor = [(x,0.5 + np.corrcoef(y,[x[1] for x in notes])[0][1]/2) for x,y in major_scales.items()]
    cor = sorted(cor, key = lambda x: -x[1])
    return cor


def find_best_minor_key(notes):
    #this outputs the most likely major scale for minor keys for the given song notes
    #input => expects a list of tuples with proportions with the keys in the following order
    # A,B-,B,C,D-,D,E-,E,F,G-,G,A-
    mir = 1.2   #minor scale root boost
    major_scales = {'A' :[1,0,1,0,1,1,0,1,0,mir,0,1], 
                    'B-':[1,1,0,1,0,1,1,0,1,0,mir,0], 
                    'B' :[0,1,1,0,1,0,1,1,0,1,0,mir],
                    'C' :[mir,0,1,1,0,1,0,1,1,0,1,0], 
                    'D-':[0,mir,0,1,1,0,1,0,1,1,0,1], 
                    'D' :[1,0,mir,0,1,1,0,1,0,1,1,0], 
                    'E-':[0,1,0,mir,0,1,1,0,1,0,1,1], 
                    'E' :[1,0,1,0,mir,0,1,1,0,1,0,1], 
                    'F' :[1,1,0,1,0,mir,0,1,1,0,1,0], 
                    'G-':[0,1,1,0,1,0,mir,0,1,1,0,1], 
                    'G' :[1,0,1,1,0,1,0,mir,0,1,1,0], 
                    'A-':[0,1,0,1,1,0,1,0,mir,0,1,1]}
    cor = [(x,0.5 + np.corrcoef(y,[x[1] for x in notes])[0][1]/2) for x,y in major_scales.items()]
    cor = sorted(cor, key = lambda x: -x[1])
    return cor

    
def get_note_freq_from_pkl(dir, file, lower=0, upper=127, prune=-1, ch_off=[], plot=True, scale_option='normal'):
    #this loads the .pkl file (previously processed .mid file)
    #it then does any filtering as per the args:
    #ch_off gives a list of channels to exclude when it builds the combined sequence, it already excludes ch 10  (drums) 
    #lower, upper tell it to remove notes outside of those bounds
    #prune tells it to limit the max number of notes per tick in the combined set (all channels) to that number, 
    #it just cuts the combined set, doesn't do it intelligently
    #--------------------------------------------
    #it returns the estimate scale of the song:
    #-----------------------------------------------
    #it also produces a bar plot showing the frequency of each of the 12 notes C, D, Eb, E, F, Gb, G, Ab, A, Bb, B
    #-----------------------------------------------
    
    #load previously created ch_seq data
    with open(dir + file + '.pkl', 'rb') as fp:
        ch_seq = pickle.load(fp)
    
    '''
    #removing unwanted ch
    if ch_off != []:
        print('recombining')
        ch_remove = ch_off + [10]        #add the drums to be removed just incase they forgot to remove
        del ch_seq['comb']
        #rebuilds the combined sequence
        ch_seq['comb'] = get_comb_ch(ch_seq, ch_remove)
        print('done')
    
    #getting rid of notes out of our fitering range
    if lower != 0 or upper != 127:
        print('filtering notes')
        for i in range(len(ch_seq['comb'])):
            ch_seq['comb'][i] = {x for x in ch_seq['comb'][i] if x >= lower and x <= upper}
        print('done')

    #cut down the note set size to max prune notes per tick
    if prune != -1:
        print('pruning notes')
        for i in range(len(ch_seq['comb'])):
            ch_seq['comb'][i] = set(sorted(ch_seq['comb'][i])[0:prune])
        print('done')
    '''
    
    #create the freq dict
    note_dict = {}
    for i in range(128):
        note_str = ms.note.Note(i).name
        if note_str == 'C#': note_str = 'D-'
        if note_str == 'F#': note_str = 'G-' 
        if note_str == 'G#': note_str = 'A-' 
        note_dict[i] = note_str
    
    #create the freq data
    #use all ch's
    note_cnt = {}
    ch_remove = set(ch_off + [10])
    for ch in set(ch_seq) - {'Header', 'Tempo', 'Key', 'comb_add', 'comb'} - ch_remove:
        for note_set in ch_seq[ch]:
            #for note in note_set:
            for note in note_set[1]:
                note_cnt[note] = note_cnt.get(note,0) + 1
    '''
    #the old way .... use comb ch
    #for note_set in ch_seq['comb']:
    #    for note in note_set:
    #        note_cnt[note] = note_cnt.get(note,0) + 1
    '''
    
    #convert to 12 note scale
    note_name_cnt = {'A':0, 'B-':0, 'B':0, 'C':0, 'D-':0, 'D':0, 'E-':0, 'E':0, 'F':0, 'G-':0, 'G':0, 'A-':0}
    for note in note_cnt:
        note_name = note_dict[note]
        note_name_cnt[note_name] = note_name_cnt.get(note_name,0) + note_cnt[note]
    
    total = sum([y for x,y in note_name_cnt.items()])
    notes = [(x, y/total if total != 0 else 0) for x,y in note_name_cnt.items()]
    #reorder it
    custom_order = ['A', 'B-', 'B', 'C', 'D-', 'D', 'E-', 'E', 'F', 'G-', 'G', 'A-']
    custom_order_dict = {y:x for x,y in enumerate(custom_order)}
    notes = sorted(notes, key = lambda x: custom_order_dict[x[0]])
   
    if scale_option == 'normal':
        #does a correlation analysis vs 12 major and 12 minor scale positions
        cor_maj = find_best_major_key(notes)
        cor_min = find_best_minor_key(notes)
        if cor_maj[0][1] >= cor_min[0][1]:
            song_scale = {'scale':cor_maj[0][0], 'key': (cor_maj[0][0],'Major'), 'conf':cor_maj[0][1]}
        else:
            maj_min_dict = {'A':'G-', 'B-':'G', 'B':'A-', 'C':'A', 'D-':'B-', 'D':'B', 'E-':'C', 'E':'D-', 'F':'D', 'G-':'E-', 'G':'E', 'A-':'F'}
            song_scale = {'scale':cor_min[0][0], 'key': (maj_min_dict[cor_min[0][0]],'Minor'), 'conf':cor_min[0][1]}
    
    elif scale_option == 'jap_penta':
        #does a correlation analysis vs 12 japanese scales
        cor = find_best_japanese_penta_key(notes)
        song_scale = {'scale':cor[0][0], 'key': (cor[0][0],'NA'), 'conf':cor[0][1]}

    elif scale_option == 'jap_hexa':
        #does a correlation analysis vs 12 japanese scales
        cor = find_best_japanese_hexa_key(notes)
        song_scale = {'scale':cor[0][0], 'key': (cor[0][0],'NA'), 'conf':cor[0][1]}

    elif scale_option == 'indo_penta':
        #does a correlation analysis vs 12 japanese scales
        cor = find_best_indo_penta_key(notes)
        song_scale = {'scale':cor[0][0], 'key': (cor[0][0],'NA'), 'conf':cor[0][1]}

    elif scale_option == 'indo_hepta':
        #does a correlation analysis vs 12 japanese scales
        cor = find_best_indo_hepta_key(notes)
        song_scale = {'scale':cor[0][0], 'key': (cor[0][0],'NA'), 'conf':cor[0][1]}

    elif scale_option == 'us_penta':
        #does a correlation analysis vs 12 penta major and 12 penta minor scale positions
        cor_min = find_best_us_penta_key(notes)
        song_scale = {'scale':cor_min[0][0], 'key': (cor_min[0][0],'Major'), 'conf':cor_min[0][1]}

    #specific scale searching options
    elif scale_option == 'single_maj':
        #does a correlation analysis vs 12 japanese scales
        cor = find_best_major_key(notes)
        song_scale = {'scale':cor[0][0], 'key': (cor[0][0],'NA'), 'conf':cor[0][1]}
    elif scale_option == 'single_min':
        #does a correlation analysis vs 12 japanese scales
        cor = find_best_minor_key(notes)
        song_scale = {'scale':cor[0][0], 'key': (cor[0][0],'NA'), 'conf':cor[0][1]}
    elif scale_option == 'single_maj_penta':
        #does a correlation analysis vs 12 japanese scales
        cor = find_best_us_penta_maj_key(notes)
        song_scale = {'scale':cor[0][0], 'key': (cor[0][0],'NA'), 'conf':cor[0][1]}
    elif scale_option == 'single_min_penta':
        #does a correlation analysis vs 12 japanese scales
        cor = find_best_us_penta_min_key(notes)
        song_scale = {'scale':cor[0][0], 'key': (cor[0][0],'NA'), 'conf':cor[0][1]}
    elif scale_option == 'single_jap_penta':
        #does a correlation analysis vs 12 japanese scales
        cor = find_best_japanese_penta_key(notes)
        song_scale = {'scale':cor[0][0], 'key': (cor[0][0],'NA'), 'conf':cor[0][1]}
    
    if plot:
        plt.bar([x[0] for x in notes], [x[1] for x in notes], alpha=0.3)
        plt.title(file + ' key:' + song_scale['key'][0] + ' ' + song_scale['key'][1] + '(' + str(round(song_scale['conf'],2))+')')
        plt.xlabel('Notes')
        plt.ylabel('Freq')
        plt.show()
    
    return song_scale



def get_TPQN_from_mid(dir, file):
    #reads the TPQN from the file header
    mf = midi.MidiFile()
    mf.open(dir + file + '.mid')
    mf.read()
    mf.close()
    return mf.ticksPerQuarterNote


## pkl File Adjusting Code

#this code is for .pkl file adjusting
#transpose pitch with pitch_delta (semitone shift)
#to call: adjust_pkl(dir, file, ps, lower=0, upper=127, prune = -1, ch_off = [])
#NOTE: filename should have no extension
#NOTE: dir should have forward slashes and should end with a forward slash

def adjust_pkl(dir, file, ps=0, lower=0, upper=127, prune = 100, ch_off = []):
    #load previously created ch_seq data
    with open(dir + file + '.pkl', 'rb') as fp:
        ch_seq = pickle.load(fp)
    
    '''
    #removing unwanted ch
    if ch_off != []:
        print('recombining')
        ch_off = ch_off + [10]        #add the drums to be removed just incase they forgot to remove
        del ch_seq['comb']
        #rebuilds the combined sequence
        ch_seq['comb'] = get_comb_ch(ch_seq, ch_off)
        print('done')
    
    #getting rid of notes out of our fitering range
    if lower != 0 or upper != 127:
        print('filtering notes')
        for i in range(len(ch_seq['comb'])):
            ch_seq['comb'][i] = {x for x in ch_seq['comb'][i] if x >= lower and x <= upper}
        print('done')

    #cut down the note set size to max prune notes per tick
    if prune != -1:
        print('pruning notes')
        for i in range(len(ch_seq['comb'])):
            ch_seq['comb'][i] = set(sorted(ch_seq['comb'][i])[0:prune])
        print('done')
    '''
    
    #ready to transpose, will transpose all ch and comb
    #check for the turn off function to create a blank file
    if ps == -9999:
        ch_seq = {x:y for x,y in ch_seq.items() if x in ['Header', 'Tempo', 'Key']}
    #do the regular transposing if the shift is not 0
    elif ps != 0:
        print('transposing')
        ch_remove = set(ch_off + [10])
        for ch in set(ch_seq) - {'Header', 'Tempo', 'Key', 'comb_add', 'comb'} - ch_remove:
            for i in range(len(ch_seq[ch])):
                ch_seq[ch][i] = ({x+ps for x in ch_seq[ch][i][0] if x+ps>=0 and x+ps<=127}, \
                                 {x+ps for x in ch_seq[ch][i][1] if x+ps>=0 and x+ps<=127})
        for ch in ['comb_add', 'comb']:
            for i in range(len(ch_seq[ch])):
                ch_seq[ch][i] = {x+ps for x in ch_seq[ch][i] if x+ps>=0 and x+ps<=127}
        print('done')
        
    #overwrite the .pkl file
    with open(dir + file + '.pkl', 'wb') as fp:
        pickle.dump(ch_seq, fp)
        

    
def pkl_cleaning(dir, file, lower=40, upper=80, note_cnt_thd=0.8, ave_note_thd=1.5, pitch_window_thd=0.8):
    #note, you must input file as the filename without the '.mid'
    #this takes in a .pkl file determines which ch:
    #NOTE: total notes for a ch is the sum of the ch set size over all ticks
    #1) best fit into the lower-upper pitch window => notes_in_window/total_notes > pitch_window_thd
    #2) have the most notes total_notes > total_notes_max*note_cnt_thd
    #3) have an ave_note_set_size < ave_note_thd
    #NOTE IF NO CHS MEET YOUR CRITERIA THEN THE 'COMB' CH WILL BE BLANK, BUT IT IS OK, IT WILL NOT BE WRITTEN TO THE GRAPH
    
    print('cleaning')
    #load previously created ch_seq data
    with open(dir + file + '.pkl', 'rb') as fp:
        ch_seq = pickle.load(fp)
    
    max_notes = 0
    ave_notes = 0
    ch_off = ['Header', 'Tempo', 'Key', 'comb_add', 'comb', 10]
    
    #if the .pkl is empty remove any old '-adj.pkl' files and exit
    if len(set(ch_seq) - set(ch_off)) == 0:
        if os.path.exists(dir + file + '-adj.pkl'):
            os.remove(dir + file + '-adj.pkl')
        return
    
    ch_stats = {}
    for ch in [x for x in ch_seq if x not in ch_off]:
        ch_stats[ch] = {'tot_notes':0, 'ave_notes':0, 'window':0}
        for tick_note_set in ch_seq[ch]:
            ch_stats[ch]['tot_notes'] += len(tick_note_set[1])
            ch_stats[ch]['window'] += len([x for x in tick_note_set[1] if x>=lower and x<=upper])
        ch_stats[ch]['ave_notes'] = ch_stats[ch]['tot_notes']/len([x[1] for x in ch_seq[ch] if len(x[1])>0])
        ch_stats[ch]['window'] = ch_stats[ch]['window']/ch_stats[ch]['tot_notes']
    
    print(len(ch_stats))
    
    #find the max tot notes
    max_notes=0
    for ch in ch_stats: 
        if ch_stats[ch]['tot_notes'] > max_notes: 
            max_notes = ch_stats[ch]['tot_notes']
    if max_notes == 0:
        print('Can not clean, there are no notes!!!')
        return

    #testing
    print('####################################')
    print('####################################')
    print('####################################')
    for ch in [x for x in ch_stats.keys() if x not in ch_off]:
        print('for ch {0}, tot_note% = {1:04.4f}, ave_note% = {2:04.4f}, window% = {3:04.4f}'.format(
            ch,
            ch_stats[ch]['tot_notes']/max_notes,
            ch_stats[ch]['ave_notes'],
            ch_stats[ch]['window']))

    print('####################################')
    print('####################################')

    #concatenate the accepted ch to a new .pkl file (name-fin.pkl)
    ch_cnt = 0
    ch_seq['comb_add'] = []
    ch_seq['comb'] = []
    for ch in [x for x in ch_stats.keys() if ch_stats[x]['tot_notes']/max_notes > note_cnt_thd and
                                             ch_stats[x]['ave_notes'] < ave_note_thd and
                                             ch_stats[x]['window'] > pitch_window_thd]:
        ch_cnt+=1
        for tick_note_set in ch_seq[ch]:
            ch_seq['comb_add'].append(tick_note_set[0])
            ch_seq['comb'].append(tick_note_set[1])

    print('kept = {0} chs'.format(ch_cnt))

    #remove all chs and comb
    ch_keep = ['Header', 'Tempo', 'Key', 'comb_add', 'comb']
    for ch in [x for x in ch_seq if x not in ch_keep]:
        del ch_seq[ch]

    #write it to disk
    with open(dir + file + '-adj.pkl', 'wb') as fp:
        pickle.dump(ch_seq, fp)

    print('done cleaning')

    

## .mid to .pkl processing code

#this code parses .mid files to a per tick sequence of note sets for each ch (except ch10 which is drums),
#it also creates a combined sequence of note sets that includes notes from ALL channels
#it saves the output python data structure to disk as a .pkl file
#to call:  parse_n_save_mid_comb(dir,filename)
#NOTE: filename should have no extension
#NOTE: dir should have forward slashes and should end with a forward slash


###########################################################################################
#NOTE: WATCH YOUR NESTED COLLECTIONS, YOU NEED TO COPY ALL SUBCOLLECTIONS FOR IT TO MAKE A DEEPCOPY, OR JUST USE DEEPCOPY, BUT THAT IS SLOW
#i.e. if you have a list of lists, you cant just write y = x.copy(), all the elements of x will still be pointers
#you need to write y = [item.copy() for item in x]  ....   this is safe!!!!!!  and if you have more nestsed collections, you need to go down to that level
#or, as I said, just use deepcopy which does a recursive search

def get_comb_ch_v2(ch_seq, ch_off = [10]):
    #add combined seq of all instrument ch except ch.10 which is drums
    #note: drums are always ch 10 in music21, in my other software it ch numbers are all 1 less
    tick_len = 0
    channels = set()
    for ch in ch_seq:
        if ch not in ['Header','Tempo','Key'] + ch_off:
            channels.add(ch)
            if len(ch_seq[ch]) > tick_len:
                tick_len = len(ch_seq[ch])
    combo_add = []
    combo = []
    ch_len = {x:len(ch_seq[x]) for x in channels}
    for i in range(tick_len):
        add_notes = set()
        notes = set()
        for ch in ch_len:
            if i < ch_len[ch]: 
                add_notes = add_notes.union(ch_seq[ch][i][0])
                notes = notes.union(ch_seq[ch][i][1])
        combo_add.append(add_notes)
        combo.append(notes)
    #here combo is a list of tuples of sets, one list item per tick
    return combo_add, combo
    
def parse_n_save_mid_comb_v2(dir, file):
    #this code, opens a midi and converts it to a time stream data format, storing it in a .pkl file
    #---------------------------------------------------------------------------------------------------
    print('start')
    # to open and parse a midi file do this
    mf = midi.MidiFile()
    mf.open(dir + file + '.mid')
    try:
        mf.read()
    except Exception as e:
        print(e)
        print('files was still loaded, so just continuing...')
    mf.close()

    #if you want a stream
    #s=converter.parse("file.mid")
    #s.write('midi', fp='./0a.output.mid')
    
    '''
    #show header
    print(mf.format)
    print(mf.ticksPerQuarterNote)
    print(mf.ticksPerSecond)
    '''
    
    #Deal with the header info
    error_msg = ''
    ch_seq = {'Header':{'Format':-1,'TPQN':-1,'TPS':-1}, 'Tempo':{}, 'Key':{}}
    #get the format, TPQN or TPS info
    if mf.format is not None:
        ch_seq['Header']['Format'] = mf.format
    if mf.ticksPerQuarterNote is not None:
        ch_seq['Header']['TPQN'] = mf.ticksPerQuarterNote
    elif mf.ticksPerSecond is not None:
        ch_seq['Header']['TPS'] = mf.ticksPerQuarterNote
    if ch_seq['Header']['Format'] == -1:
        print('Header Issue: no Format info in file, can not process file...')
        return 'bad_file'
    elif ch_seq['Header']['TPQN'] == -1 and ch_seq['Header']['TPS'] == -1:
        print('Header Issue: no TPQN or TPS info in file, can not process file...')
        return 'bad_file'
    elif ch_seq['Header']['TPQN'] > 2000:
        print('Header Issue: TPQN too high, can not process file...')
        return 'bad_file'
        

    #look for the tempo info: it stores the tempo track info, usually in track 1, sometimes it could be in each track, this will have a dict of dicts. Each track that has a tempo setting will be here with the abs time the tempo starts at
    for track in mf.tracks:   #go through each track
        tick = 0
        ch_seq['Tempo'][track.index] = {}
        for event in track.events:  #go through each event in each track
            #go through the track and look for tempo events
            if event.type == 'DeltaTime':
                tick += event.time
            if event.type == 'SET_TEMPO':
                ch_seq['Tempo'][track.index][tick] = ms.midi.getNumber(event.data,3)[0]

        if len(ch_seq['Tempo'][track.index]) == 0:
            del ch_seq['Tempo'][track.index]


    #output the time info to the screen for checking
    print('############################################################')
    print('TPQN = {0}, TPS = {1}'.format(ch_seq['Header']['TPQN'],ch_seq['Header']['TPS']))
    print('Tempo List: {0}'.format(ch_seq['Tempo']))
    
    #add chs to the ch_seq object and notes dict
             #notes = {}
    for track in mf.tracks:   #go through each track
        for event in track.events:  #go through each event in each track
            if event.type == 'NOTE_ON' and event.channel not in ch_seq:
                ch_seq[event.channel] = []
                #notes[event.channel] = set()
    
    if len(set(ch_seq) - {'Header', 'Tempo', 'Key'}) == 0:
        #we have no note channels, bad file, so exit
        print('no channels, can not process file...')
        return 'bad_file'
    
    test = 0
    #Start the main processing, go through each track
    for track in mf.tracks:
        ch_seq_temp = {}
        ch_temp = set()
        for event in track.events:  #go through each event in each track
            #go through the track and list the ch it contains
            if event.type == "NOTE_ON" or event.type == "NOTE_OFF":
                ch_temp.add(event.channel)
        ch_seq_temp = deepcopy({x:[(set(),set())] for x in ch_temp})

        #now ch_seq_temp will have one key per ch that was found in the track, i.e. ch_seq_temp = {4:[],8:[]} => so the track contains ch 4 and 8
        tick_old = 0      #this is the absolute time counter in ticks
        tick_now = 0      #this is the absolute time counter in ticks
        #each element in the ch list will be a tick
        for event in track.events:
            #analyse the event:
            if event.type=='DeltaTime':
                #increase the time counter => tick_new = tick_old + tick_delta
                delta = event.time
                if delta > 0:
                    tick_old = tick_now
                    tick_now = tick_old + delta
                    #copy the current ch notes_set (notes[ch_id]) to all elements in the ch seq from the old tick up to the current tick
                    for ch in ch_seq_temp:
                        ch_seq_temp[ch].extend([(set(), ch_seq_temp[ch][tick_old][1].copy()) for i in range(delta)])
            
            elif event.type == 'NOTE_ON':
                ch = event.channel
                if event.velocity == 0:   #treat as off, some files do this instead of using 'NOTE_OFF', also some encode drums with velocity = 0
                    if event.pitch in ch_seq_temp[ch][tick_now][1]:
                        ch_seq_temp[ch][tick_now][1].remove(event.pitch)
                else:
                    #notes[ch].add(event.pitch)
                    ch_seq_temp[ch][tick_now][0].add(event.pitch)
                    ch_seq_temp[ch][tick_now][1].add(event.pitch)
            
            elif event.type == 'NOTE_OFF':
                ch = event.channel
                if event.pitch in ch_seq_temp[ch][tick_now][1]:
                    ch_seq_temp[ch][tick_now][1].remove(event.pitch)

        #merge the chs with the main DB
        for ch in ch_seq_temp:
            if ch in ch_seq and len(ch_seq[ch]) > 0:
                if len(ch_seq_temp[ch]) > len(ch_seq[ch]):
                    for i in range(len(ch_seq[ch])):
                        ch_seq[ch][i] = (ch_seq[ch][i][0].union(ch_seq_temp[ch][i][0]),
                                         ch_seq[ch][i][1].union(ch_seq_temp[ch][i][1]))
                    for i in range(len(ch_seq[ch]),len(ch_seq_temp[ch])):
                        ch_seq[ch].append((set().union(ch_seq_temp[ch][i][0]), 
                                           set().union(ch_seq_temp[ch][i][1])))
                else:
                    for i in range(len(ch_seq_temp[ch])):
                        ch_seq[ch][i] = (ch_seq[ch][i][0].union(ch_seq_temp[ch][i][0]),
                                         ch_seq[ch][i][1].union(ch_seq_temp[ch][i][1]))
            else:
                ch_seq[ch] = deepcopy(ch_seq_temp[ch])
    
    #add the combined ch
    ch_seq['comb_add'], ch_seq['comb'] = get_comb_ch_v2(ch_seq, ch_off = [10])
    
    #at this point ch_seq[ch] contains a list of tuples of sets, index is tick
    for i in range(30):
        if i in ch_seq:
            print('ch {0} has tick len {1} and {2} notes'.format(i,len(ch_seq[i]),sum([len(x[1]) for x in ch_seq[i]])))        

    #write it to disk
    #dump
    with open(dir + file + '.pkl', 'wb') as fp:
        pickle.dump(ch_seq, fp)
    
    print('done')
    
    return ''

## Make Single Note Graph Code V3

def quantize_time_qn(duration, cuts, TPQN):
    #input 'duration' is in ticks, cut intervals are in QN, TPQN is from the header
    #output 'duration' is in quantised quarter notes eg. 0.125(1/32), 0.25(1/16), 0.375(dot1/16), 0.5(1/8), 1(1/4), 2(1/2)
    #first convert duration to QN
    duration = duration/TPQN
    if duration >= cuts[-1]: 
        duration = cuts[-1]
    elif duration == 0: 
        duration = 0
    else: 
        duration = cuts[np.digitize(duration, cuts, right=False)]
    return duration


def make_graph_single_note_v3(dir, file, G, lower=0, upper=127, ch_off=[], 
                              use_raw=False, truncate=-1, 
                              edge_dur_qn_max_abs=12, 
                              edge_dur_qn_max_rel=12, 
                              pitch_delta_max=12,
                              use_highest_note_only_flag = False,
                              ignore_edge_duration=False):

    
    def add_node(G, node):
        #add note as node to the graph if not there
        if not G.has_node(node):
            G.add_node(node, weight = 1)
        elif 'weight' not in G.nodes[node]:
            G.nodes[node]['weight'] = 1
        else:
            G.nodes[node]['weight'] += 1
        #increment the total note counter
        G.name += 1
        if truncate != -1 and G.name >= truncate:
            print('getting out early due to note truncation')
            print('graph has {0} nodes and {1} edges.'.format(G.number_of_nodes(),G.number_of_edges()))
            return 'exit'
    
    
    #build graph
    #follows the paper
    #each node is a note (pitch and duration in fractions of QN)
    #each edge is comprised of source and target notes and a duration (start time of target - start time of source)
    
    #load
    print('reading')
    if use_raw:
        with open(dir + file + '.pkl', 'rb') as fp:
            ch_seq = pickle.load(fp)
    else:
        with open(dir + file + '-adj.pkl', 'rb') as fp:
            ch_seq = pickle.load(fp)
    print('done')

    #getting rid of notes out of our fitering range
    if lower != 0 or upper != 127:
        print('filtering notes')
        for i in range(len(ch_seq['comb'])):
            ch_seq['comb_add'][i] = {x for x in ch_seq['comb_add'][i] if x >= lower and x <= upper}
            ch_seq['comb'][i] = {x for x in ch_seq['comb'][i] if x >= lower and x <= upper}
        print('done')

    #these are the duration cuts in proportion of QN (quarter notes)
    cuts = [0.015625,0.0234375,0.03125,0.046875,0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.75,1,1.5,2,3,4,6,8,12]
    TPQN = ch_seq['Header']['TPQN']
    if TPQN <= 0:
        print('file has no TPQN, skipping...')
        return
    
    #this only keeps the highest note
    if use_highest_note_only_flag:
        for i in range(len(ch_seq['comb'])):
            if ch_seq['comb'][i] != set():
                ch_seq['comb'][i] = {max(ch_seq['comb'][i])}
                if ch_seq['comb'][i].issubset(ch_seq['comb_add'][i]):
                    ch_seq['comb_add'][i] = ch_seq['comb'][i]
                else:
                    ch_seq['comb_add'][i] = set()
            else:
                ch_seq['comb_add'][i] = set()
                
    print('starting')
    active_set = {}
    last_notes = set()
    last_tick = len(ch_seq['comb'])
    special_cases = 0
    for tick in range(last_tick):
        add_notes = ch_seq['comb_add'][tick]
        notes = ch_seq['comb'][tick]
        #get the notes we have lost since last tick
        lost_notes = last_notes.difference(notes)
        #get the notes we have gained since last tick
        gain_notes = notes.difference(last_notes)
        #check for the case the same note went off and on at this tick
        if add_notes != set():
            for note in add_notes:
                if note in last_notes and note in notes:
                    special_cases+=1
                    lost_notes.add(note)
                    gain_notes.add(note)
        #we only perform actions if there is a change in the note set
        if lost_notes != set():   #we have lost some notes
            for note in lost_notes:
                #calc and update the note duration
                note_duration_qn = -1
                #update active_set where the note is a target
                for item in [x for x in active_set if x[1] == note and active_set[x]['target_duration'][0] == 's']:
                    note_duration_qn = quantize_time_qn(tick - active_set[item]['target_duration'][1], cuts, TPQN)
                    active_set[item]['target_duration'] = ('d', note_duration_qn)
                #update active_set where the note is a source
                for item in [x for x in active_set if x[0] == note and active_set[x]['source_duration'][0] == 's']:
                    note_duration_qn = quantize_time_qn(tick - active_set[item]['source_duration'][1], cuts, TPQN)
                    active_set[item]['source_duration'] = ('d', note_duration_qn)
                    #add note as node to the graph if not there
                    node_to_add = str(note) + '_' + str(note_duration_qn)
                    if add_node(G, node_to_add) == 'exit':
                        return
                
            #add entries that are complete to graph and delete from active_set
            for item in [x for x in active_set if \
                         active_set[x]['source_duration'][0] == 'd' and \
                         active_set[x]['target_duration'][0] == 'd' and \
                         active_set[x]['edge_duration'][0] == 'd']:
                #add edge to graph if the edge meets our criteria
                edge_duration_qn = active_set[item]['edge_duration'][1]
                if edge_duration_qn <= edge_dur_qn_max_abs and \
                edge_duration_qn <= edge_dur_qn_max_rel*active_set[item]['source_duration'][1] and \
                abs(item[0]-item[1]) <= pitch_delta_max:
                    node1 = str(item[0])+'_'+str(active_set[item]['source_duration'][1])
                    node2 = str(item[1])+'_'+str(active_set[item]['target_duration'][1])
                    if ignore_edge_duration:
                        if G.has_edge(node1, node2):
                            G.edges[node1, node2]['weight'] += 1
                        else:
                            G.add_edge(node1, node2, weight = 1)
                    else:
                        edge_duration_qn = max(1/4,edge_duration_qn)
                        if G.has_edge(node1, node2, edge_duration_qn):
                            G.edges[node1, node2, edge_duration_qn]['weight'] += 1
                        else:
                            G.add_edge(node1, node2, edge_duration_qn, weight = 1)
                #remove the active_set item as we are done with it
                del active_set[item]
                
        if gain_notes != set():   #we have gained some notes
            for note in gain_notes:
                #add the note as a target to the currently unassigned source notes in the active_set 
                for item in [x for x in active_set if x[1] == '']:
                    active_set[(item[0],note)] = {}
                    active_set[(item[0],note)]['source_duration'] = active_set[item]['source_duration']
                    active_set[(item[0],note)]['target_duration'] = ('s',tick)
                    active_set[(item[0],note)]['edge_duration'] = ('d',quantize_time_qn(tick - active_set[item]['edge_duration'][1], cuts, TPQN))
            #delete the entries with target = '' as we have assigned them all, must do it here, outside the for loops
            for item in [x for x in active_set if x[1] == '']:
                del active_set[item]
            #add the new note and it's start time to the active_set
            for note in gain_notes:
                active_set[(note,'')] = {'source_duration':('s',tick), 'target_duration':('s',-1), 'edge_duration':('s',tick)}

        last_notes = notes
    
    print('finished')
    print('graph has {0} nodes and {1} edges.'.format(G.number_of_nodes(),G.number_of_edges()))
    print('GOT {0} special cases!!!'.format(special_cases))

## Make Multi Note Graph Code

def make_graph_comb(dir, file, G, lower=0, upper=127, prune=-1, use_raw=True):
    #build graph, if no graph is passed, it starts a new graph
    #option: this makes polyphonic nodes, the ticks held on each node are recorded as weight, the edge count to a
    #particular target is the edge weight, edges are directed
    #each node has an on/off flag => the weight for on_flag is the num ticks held with th enote on, the weight for the on_flag = F
    #is the num ticks we are on the node but the notes are all off.
    #it does not include ch10, but all other ch are included
    #input is the dir and file with the processed .pkl file, there is no output, it just updates the graph that was given to it
    #graph is nathan format

    #load
    print('reading')
    if use_raw:
        with open(dir + file + '.pkl', 'rb') as fp:
            ch_seq = pickle.load(fp)
    else:
        with open(dir + file + '-adj.pkl', 'rb') as fp:
            ch_seq = pickle.load(fp)
    print('done')

    TPQN = ch_seq['Header']['TPQN']
    if TPQN <= 0:
        print('file has no TPQN, skipping...')
        return

    #getting rid of notes out of our fitering range
    if lower != 0 or upper != 127:
        print('filtering notes')
        for i in range(len(ch_seq['comb'])):
            ch_seq['comb_add'][i] = {x for x in ch_seq['comb_add'][i] if x >= lower and x <= upper}
            ch_seq['comb'][i] = {x for x in ch_seq['comb'][i] if x >= lower and x <= upper}
        print('done')
        
    #cut down the note set size to max prune notes per tick
    print('pruning notes')
    if prune != -1:
        for i in range(len(ch_seq['comb'])):
            ch_seq['comb'][i] = '_'.join([str(x) for x in sorted(ch_seq['comb'][i])[0:prune]])
    else:
        for i in range(len(ch_seq['comb'])):
            ch_seq['comb'][i] = '_'.join([str(x) for x in sorted(ch_seq['comb'][i])])
    print('done')
       
    print('starting')
    #set the rest and active counters
    off_cnt = 0
    on_cnt = 0
    last_node = 'start'
    #go through each tick
    for tick in range(len(ch_seq['comb'])):
        node = ch_seq['comb'][tick]

        ##############################3
        #testing
        if last_node == '':
            print('faark me, got an error')
            raise KeyboardInterrupt
        ##############################3
            
        #NOTE: last node is the current node that the graph is camped on, it will always be something, never ''
        #we have a repeat of the last node
        if node == last_node:  
            if last_node != 'start':
                on_cnt += 1
                #just add weight to node
                G.nodes[last_node]['on_weight'] += 1

        #we have a change in the node
        elif node != last_node:
            #we have a rest, so keep old note and update off weight
            if node == '' and last_node != 'start':    
                off_cnt += 1
                G.nodes[last_node]['off_weight'] += 1
        
            #we have a change in the node to a new node
            elif node != '':
                #add or update node
                if G.has_node(node): 
                    G.nodes[node]['on_weight'] += 1
                else: 
                    G.add_node(node, on_weight=1, off_weight=0)
                
                #add or update edge only if the rest was not too long
                if off_cnt < 4*on_cnt: 
                    if G.has_edge(last_node, node): 
                        G.edges[last_node, node]['weight'] += 1
                    elif last_node != 'start': 
                        G.add_edge(last_node, node, weight=1)
                    else:
                        pass
                    
                #overwrite last_node 
                last_node = node
                #resset the active/rest counters
                off_cnt = 0
                on_cnt = 1

    print('finished')
    print('graph has {0} nodes and {1} edges.'.format(G.number_of_nodes(),G.number_of_edges()))


## Music Creation Code

def t2qn(tick):
    return midi.translate.midiToDuration(tick,TPQN)._qtrLength

def create_polyphonic_music(G, use_raw=False, instrument=ms.instrument.Violin()):
    #this creates your music from the given graph (nathan format)
    #output is a midi stream
    TPQN = 192/4   #this basically scales the speed of the peice
    #min max levels in QN => 1/8 is an 1/32nd note
    min_note_len = 1/1000   #1/8
    max_note_len = 1000    #8

    #these are the duration cuts in proportion of QN (quarter notes)
    cuts = [0.015625,0.0234375,0.03125,0.046875,0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.75,1,1.5,2,3,4,6,8,12]
    
    node_current = str(np.random.choice(G.nodes()))
    node_current_base = node_current
    node_new = ''
    tick_old = 0
    s = ms.stream.Stream()
    p = ms.stream.Part()
    p.insert(instrument)
    for tick in range(10000):
        #node_current is the current node state, node_current_base is the current base node state (on state)
        #and node_new is the newly chosen node, if = '' it means it is a rest (off state) and we keep the current base
        if tick%1000 == 0:
            print(tick)
        #choose only from the outgoing edges
        nbr_names = [x for x in G.neighbors(node_current_base)]   #this only chooses outgoing edges, doesn't include incoming
        nbr_weights = [G.edges[node_current_base, nbr]['weight'] for nbr in nbr_names]
        #build the next tick choices based on whether we are in on state or off state
        #in off state, remove the option to go back to the on state
        if node_current == '':
            #in the case we are rested, then do not include the option to go back to the active state, only include the edges
            nbr_weights.extend([G.nodes[node_current_base]['off_weight']])
            nbr_names.extend([''])
        else:
            nbr_weights.extend([G.nodes[node_current_base]['on_weight'],G.nodes[node_current_base]['off_weight']])
            nbr_names.extend([node_current_base,''])
        nbr_weights = [float(i)/sum(nbr_weights) for i in nbr_weights]
        node_new = np.random.choice(nbr_names, p=nbr_weights, replace=False)
        #check for a change in node, o/w we just let the tick counter increase by not updating it
        if node_new != node_current:
            #we got a change so write the current node as we now know the duration
            #but only if it is a non rest, i.e. not '', in that case, do not write anything
            if node_current != '':
                #delta is the tick delta
                delta_qn =  quantize_time_qn(tick - tick_old, cuts, TPQN)
                notes = [int(x) for x in node_current.split('_')]
                if delta_qn > 0:
                    p.append(ms.chord.Chord(notes, duration=ms.duration.Duration(min(max_note_len, max(min_note_len, delta_qn)))))
            #reset tick counter
            tick_old = tick
            #take new value for the current node
            node_current = node_new
            #take the new value for the current_base if the new node is not a rest ('')
            if node_new != '':
                node_current_base = node_new
    s.append(p)
    return s


def create_monophonic_music_v2(G, instrument=ms.instrument.Violin()):
    #this creates your music for the single note shit
    #input is a graph of the same format as the paper
    #output is a midi stream

    #time mult
    tn = 0.75
    tr = tn
    #min max levels in QNs => 1/8 => 1/32nd note, 1/1000 and 1000 mean I am not using these limits
    min_rest_len = 1/1000
    max_rest_len = 1000
    min_note_len = 1/1000
    max_note_len = 1000
    
    node_weights = []
    node_choice = list(G.nodes())
    for node in node_choice:
        if 'weight' not in G.nodes[node]: G.nodes[node]['weight'] = 1
        node_weights.append(G.nodes[node]['weight'])
    node_weight_sum = sum(node_weights)
    node_weights = [x/node_weight_sum for x in node_weights]
    while True:
        node_old = np.random.choice(node_choice, p=node_weights)
        if len(G[node_old]) > 0: break
    node_new = ''
    
    s = ms.stream.Stream()
    p = ms.stream.Part()
    p.insert(instrument)
    
    #This puts in 4 low notes as a starting indicator
    ########################################################
    for i in range(4):
        n = ms.note.Note(12, duration=ms.duration.Duration(1))
        n.volume.velocity=90
        p.append(n)
    ########################################################
    
    for i in range(2000):
        if i%1000 == 0:
            print(i)
        nbr_names = []
        nbr_weights = []
        #go through the nbrs => each nbr is n1, n2, edge delta
        for target in G[node_old]:
            for edge_delta in G[node_old][target]:
                nbr_names.append(target + '_edge_delta' + str(edge_delta))
                nbr_weights.append(G[node_old][target][edge_delta]['weight'])
        #normalise weights
        nbr_weights = [float(i)/sum(nbr_weights) for i in nbr_weights]
        if len(nbr_names) > 0:
            choice = np.random.choice(nbr_names, p=nbr_weights, replace=False)
            node_new, edge_delta = choice.split('_edge_delta')
            #only write something if a new node is chosen
            if node_new != node_old:
                if node_old != '':
                    note_delta = float(node_old.split('_')[1])
                    rest_delta = float(edge_delta) - note_delta
                    rest_delta = max(0,rest_delta)   #make sure the rest is not -ve
                    note = int(float(node_old.split('_')[0]))
                    if note_delta > 0:
                        p.append(ms.note.Note(note, duration=ms.duration.Duration(tn*min(max_note_len, max(min_note_len, note_delta)))))
                    if rest_delta > 0:
                        p.append(ms.note.Rest(duration=ms.duration.Duration(tr*min(max_rest_len, max(min_rest_len, rest_delta)))))

                node_old = node_new
        else:
            #for j in [12,14,12,15,20,25,30,25,20,15]:
            #    p.append(ms.note.Note(j, duration=ms.duration.Duration(1/8)))
            print('got no nbrs for {0}, doing a random jump back in...'.format(node_old))
            while True:
                node_old = np.random.choice(node_choice, p=node_weights)
                if len(G[node_old]) > 0: break
    s.append(p)
    return s


def create_monophonic_music_v3(G, instrument=ms.instrument.Violin()):
    #this creates your music for the single note shit
    #input is a graph of the same format as the paper
    #output is a midi stream
    #this is the case when we ignore edge duration

    #time mult
    tn = 0.75
    tr = tn
    #min max levels in QNs => 1/8 => 1/32nd note, 1/1000 and 1000 mean I am not using these limits
    min_rest_len = 1/1000
    max_rest_len = 1000
    min_note_len = 1/1000
    max_note_len = 1000
    
    node_weights = []
    node_choice = list(G.nodes())
    for node in node_choice:
        if 'weight' not in G.nodes[node]: G.nodes[node]['weight'] = 1
        node_weights.append(G.nodes[node]['weight'])
    node_weight_sum = sum(node_weights)
    node_weights = [x/node_weight_sum for x in node_weights]
    while True:
        node_old = np.random.choice(node_choice, p=node_weights)
        if len(G[node_old]) > 0: break
    node_new = ''
    
    s = ms.stream.Stream()
    p = ms.stream.Part()
    p.insert(instrument)
    
    #This puts in 4 low notes as a starting indicator
    ########################################################
    for i in range(4):
        n = ms.note.Note(12, duration=ms.duration.Duration(1))
        n.volume.velocity=90
        p.append(n)
    ########################################################
    
    for i in range(1000):
        if i%1000 == 0:
            print(i)
        nbr_names = []
        nbr_weights = []
        #go through the nbrs => each nbr is n1, n2, edge delta
        for target in G[node_old]:
            nbr_names.append(target)
            nbr_weights.append(G[node_old][target]['weight'])
        #normalise weights
        nbr_weights = [float(i)/sum(nbr_weights) for i in nbr_weights]
        if len(nbr_names) > 0:
            choice = np.random.choice(nbr_names, p=nbr_weights, replace=False)
            node_new = choice
            #only write something if a new node is chosen
            if node_new != node_old:
                if node_old != '':
                    note_delta = float(node_old.split('_')[1])
                    rest_delta = 0
                    note = int(float(node_old.split('_')[0]))
                    if note_delta > 0:
                        p.append(ms.note.Note(note, duration=ms.duration.Duration(tn*min(max_note_len, max(min_note_len, note_delta)))))

                node_old = node_new
        else:
            print('got no nbrs for {0}, doing a random jump back in...'.format(node_old))
            while True:
                node_old = np.random.choice(node_choice, p=node_weights)
                if len(G[node_old]) > 0: break
    s.append(p)
    return s
    
    
    
    
     
#stats
###################################################################
def get_nbr_cnt(G,node):
    n_nbrs = 0
    for x in G[node]:
        for y in G[node][x]:
            n_nbrs += 1
    return n_nbrs
    
def mean_shortest_path_sample(G,sampling_rate=0.1):
    #this calculates the representative mean shortest path, by sampling to minimise the time
    shortest_path_lengths = []
    for sg in nx.algorithms.components.connected_components(G):
        subgraph = G.subgraph(sg)
        total_nodes = subgraph.number_of_nodes()
        total_pairs = total_nodes*(total_nodes-1)/2
        sample_cnt = int(total_pairs*sampling_rate)
        sample = np.random.choice(list(subgraph.nodes()), (sample_cnt, 2))
        sample = [(x,y) for x,y in sample if x != y and (y,x) not in sample]
        shortest_path_lengths.extend([nx.shortest_path_length(subgraph, x, y) for x, y in sample])
    return np.mean(shortest_path_lengths)

def all_pairs(nodes):
    """Generates all pairs of nodes."""
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j:
                yield u, v
                
def degrees(G, type='both'):
    #returns the node degree list for both dirn, out or in
    if type == 'both':
        output = [G.degree(u) for u in G]
    elif type == 'out':
        output = [G.out_degree(u) for u in G]
    elif type == 'in':
        output = [G.in_degree(u) for u in G]
    return output

def edge_weights_mdg(G):
    #returns the edge weights list for both dirn, out or in
    weights = []
    for s in G:
        for t in G[s]:
            for link in G[s][t]:
                weights.append(G[s][t][link]['weight'])
    return weights


def edge_weights_dg(G):
    #returns the edge weights list for both dirn, out or in
    weights = []
    for s in G:
        for t in G[s]:
            weights.append(G[s][t]['weight'])
    return weights


def node_strength_mdg(G, type='both'):
    #returns the node strength list for both dirn, out or in
    weights = {}
    for s in G:
        weights[s] = 0
        if type == 'both':
            for t in G.predecessors(s):
                for link in G[t][s]:
                    weights[s] += G[t][s][link]['weight']
            for t in G.successors(s):
                for link in G[s][t]:
                    weights[s] += G[s][t][link]['weight']
        elif type == 'out':
            for t in G.successors(s):
                for link in G[s][t]:
                    weights[s] += G[s][t][link]['weight']
        elif type == 'in':
            for t in G.predecessors(s):
                for link in G[t][s]:
                    weights[s] += G[t][s][link]['weight']
    return [x for x in weights.values()]


def node_strength_dg(G, type='both'):
    #returns the node strength list for both dirn, out or in
    weights = {}
    for s in G:
        weights[s] = 0
        if type == 'both':
            for t in G.predecessors(s):
                weights[s] += G[t][s]['weight']
            for t in G.successors(s):
                weights[s] += G[s][t]['weight']
        elif type == 'out':
            for t in G.successors(s):
                weights[s] += G[s][t]['weight']
        elif type == 'in':
            for t in G.predecessors(s):
                weights[s] += G[t][s]['weight']
    return [x for x in weights.values()]