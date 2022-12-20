import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from env import user, password, host
from scipy.stats import levene, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import math
import numpy as np
import os
import pandas as pd
import requests
import seaborn as sns
import statsmodels.api as sm
import doctest
import pdb
import warnings
warnings.filterwarnings("ignore")

#######################################################################################################
def get_prep22(df):
    global df22
    df22=df22.rename(columns={'AGGREGATION LEVEL': 'agg_level', 'CAMPUS':'campus_number', 
                              'REGION':'region','DISTRICT NAME AND NUMBER': 'dist_name_num',
                              'CHARTER_STATUS':'charter_status','CAMPUS NAME AND NUMBER': 
                              'campus_name_num', 'SECTION': 'section','HEADING':'heading',
                              'HEADING NAME': 'heading_name', 'YR22':'student_count'})
    df22['charter_encoded'] = df22.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 
                                                       'TRADITIONAL ISD/CSD':0})
    df22=df22[(df22.heading == 'A01') | (df22.heading ==  'A03')]
    df22=df22[df22['student_count'] != '-999']
    df22['student_count']= df22['student_count'].str.replace("<", "")
    df22['student_count'] = df22['student_count'].astype(float)
    dfpivot=df22.pivot(index='campus_number', columns='heading', values= 'student_count').dropna()
    df22=df22.merge(dfpivot,how= 'right', on= 'campus_number')
    df22=df22.rename(columns={'A01': 'student_enrollment', 'A03':'discipline_count'})
    df22['discipline_percent']= ((df22['discipline_count']/df22['student_enrollment'])*100)
    df22=df22.round({'discipline_percent': 0})
    df22=df22.drop(columns=['agg_level', 'campus_number', 'region', 'charter_status', 
                            'dist_name_num', 'student_count','section', 'heading',
                            'heading_name', 'student_count'])
    df22.dropna()
    df22=df22.drop_duplicates()
    df22=df22.reset_index(drop=True)
    
    
def get_prep21(df):
    global df21
    df21=df21.rename(columns={'AGGREGATION LEVEL': 'agg_level', 'CAMPUS':'campus_number', 
                              'REGION':'region','DISTRICT NAME AND NUMBER': 'dist_name_num',
                              'CHARTER_STATUS':'charter_status','CAMPUS NAME AND NUMBER': 
                              'campus_name_num', 'SECTION': 'section','HEADING':'heading',
                              'HEADING NAME': 'heading_name', 'YR21':'student_count'})
    df21['charter_encoded'] = df21.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 
                                                       'TRADITIONAL ISD/CSD':0})
    df21=df21[(df21.heading == 'A01') | (df21.heading ==  'A03')]
    df21=df21[df21['student_count'] != '-999']
    df21['student_count']= df21['student_count'].str.replace("<", "")
    df21['student_count'] = df21['student_count'].astype(float)
    dfpivot=df21.pivot(index='campus_number', columns='heading', values= 'student_count').dropna()
    df21=df21.merge(dfpivot,how= 'right', on= 'campus_number')
    df21=df21.rename(columns={'A01': 'student_enrollment', 'A03':'discipline_count'})
    df21['discipline_percent']= ((df21['discipline_count']/df21['student_enrollment'])*100)
    df21=df21.round({'discipline_percent': 0})
    df21=df21.drop(columns=['agg_level', 'campus_number', 'region', 'charter_status', 
                            'dist_name_num', 'student_count','section', 'heading',
                            'heading_name', 'student_count'])
    df21.dropna()
    df21=df21.drop_duplicates()
    df21=df21.reset_index(drop=True)
    
    
def get_prep20(df):
    global df20
    df20=df20.rename(columns={'AGGREGATION LEVEL': 'agg_level', 'CAMPUS':'campus_number', 
                              'REGION':'region','DISTRICT NAME AND NUMBER': 'dist_name_num',
                              'CHARTER_STATUS':'charter_status','CAMPUS NAME AND NUMBER': 
                              'campus_name_num', 'SECTION': 'section','HEADING':'heading',
                              'HEADING NAME': 'heading_name', 'YR20':'student_count'})
    df20['charter_encoded'] = df20.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 
                                                       'TRADITIONAL ISD/CSD':0})
    df20=df20[(df20.heading == 'A01') | (df20.heading ==  'A03')]
    df20=df20[df20['student_count'] != '-999']
    df20['student_count']= df20['student_count'].str.replace("<", "")
    df20['student_count'] = df20['student_count'].astype(float)
    dfpivot=df20.pivot(index='campus_number', columns='heading', values= 'student_count').dropna()
    df20=df20.merge(dfpivot,how= 'right', on= 'campus_number')
    df20=df20.rename(columns={'A01': 'student_enrollment', 'A03':'discipline_count'})
    df20['discipline_percent']= ((df20['discipline_count']/df20['student_enrollment'])*100)
    df20=df20.round({'discipline_percent': 0})
    df20=df20.drop(columns=['agg_level', 'campus_number', 'region', 'charter_status', 
                            'dist_name_num', 'student_count','section', 'heading',
                            'heading_name', 'student_count'])
    df20.dropna()
    df20=df20.drop_duplicates()
    df20=df20.reset_index(drop=True)
    
    
def get_prep19(df):
    global df19
    df19=df19.rename(columns={'AGGREGATION LEVEL': 'agg_level', 'CAMPUS':'campus_number', 
                              'REGION':'region','DISTRICT NAME AND NUMBER': 'dist_name_num',
                              'CHARTER_STATUS':'charter_status','CAMPUS NAME AND NUMBER': 
                              'campus_name_num', 'SECTION': 'section','HEADING':'heading',
                              'HEADING NAME': 'heading_name', 'YR19':'student_count'})
    df19['charter_encoded'] = df19.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 
                                                       'TRADITIONAL ISD/CSD':0})
    df19=df19[(df19.heading == 'A01') | (df19.heading ==  'A03')]
    df19=df19[df19['student_count'] != '-999']
    df19['student_count']= df19['student_count'].str.replace("<", "")
    df19['student_count'] = df19['student_count'].astype(float)
    dfpivot=df19.pivot(index='campus_number', columns='heading', values= 'student_count').dropna()
    df19=df19.merge(dfpivot,how= 'right', on= 'campus_number')
    df19=df19.rename(columns={'A01': 'student_enrollment', 'A03':'discipline_count'})
    df19['discipline_percent']= ((df19['discipline_count']/df19['student_enrollment'])*100)
    df19=df19.round({'discipline_percent': 0})
    df19=df19.drop(columns=['agg_level', 'campus_number', 'region', 'charter_status', 
                            'dist_name_num', 'student_count','section', 'heading',
                            'heading_name', 'student_count'])
    df19.dropna()
    df19=df19.drop_duplicates()
    df19=df19.reset_index(drop=True)
    
    
    
def get_prep18(df18):
    df18=df18.rename(columns={'AGGREGATION LEVEL': 'agg_level', 'CAMPUS':'campus_number', 
                              'REGION':'region','DISTRICT NAME AND NUMBER': 'dist_name_num',
                              'CHARTER_STATUS':'charter_status','CAMPUS NAME AND NUMBER': 
                              'campus_name_num', 'SECTION': 'section','HEADING':'heading',
                              'HEADING NAME': 'heading_name', 'YR18':'student_count'})
    df18['charter_encoded'] = df18.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 
                                                       'TRADITIONAL ISD/CSD':0})
    df18=df18[(df18.heading == 'A01') | (df18.heading ==  'A03')]
    df18=df18[df18['student_count'] != '-999']
    df18['student_count']= df18['student_count'].str.replace("<", "")
    df18['student_count'] = df18['student_count'].astype(float)
    dfpivot=df18.pivot(index='campus_number', columns='heading', values= 'student_count').dropna()
    df18=df18.merge(dfpivot,how= 'right', on= 'campus_number')
    df18=df18.rename(columns={'A01': 'student_enrollment', 'A03':'discipline_count'})
    df18['discipline_percent']= ((df18['discipline_count']/df18['student_enrollment'])*100)
    df18=df18.round({'discipline_percent': 0})
    df18=df18.drop(columns=['agg_level', 'campus_number', 'region', 'charter_status', 
                            'dist_name_num', 'student_count','section', 'heading',
                            'heading_name', 'student_count'])
    df18.dropna()
    df18=df18.drop_duplicates()
    df18=df18.reset_index(drop=True)