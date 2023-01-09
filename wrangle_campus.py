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
import warnings
warnings.filterwarnings("ignore")


###################################################################################
#################################### ACQUIRE DATA #################################
###################################################################################

#go to : https://rptsvr1.tea.texas.gov/adhocrpt/Disciplinary_Data_Products/Download_All_Districts.html
#download the csv for 2018/19, 2019/20, 2020/21, and 2021/22 School Years
#upload csv files:

df22 = pd.read_csv('CAMPUS_summary_22.csv')
df21 = pd.read_csv('CAMPUS_summary_21.csv')
df20 = pd.read_csv('CAMPUS_summary_20.csv')
df19 = pd.read_csv('CAMPUS_summary_19.csv')
df18 = pd.read_csv('CAMPUS_summary_18.csv')
    
###################################################################################
##################################### PREP DATA ###################################
###################################################################################

# function takes in a dataset and returns a prepped dataframe ready to explore
def campus_prep(df):
    #rename the columns for ease of use with python omitting any capitilization and spaces
    df=df.rename(columns={'AGGREGATION LEVEL': 'agg_level',
                          'CAMPUS':'campus_number',
                          'REGION':'region',
                          'DISTRICT NAME AND NUMBER':'dist_name_num',
                          'CHARTER_STATUS':'charter_status',
                          'CAMPUS NAME AND NUMBER':'campus_name_num',
                          'SECTION': 'section',
                          'HEADING':'heading',
                          'HEADING NAME': 'heading_name',
                          'YR22':'student_count',
                          'YR21':'student_count',
                          'YR20':'student_count',
                          'YR19':'student_count',
                          'YR18':'student_count'})

    #maps the column charter_status, so that charter and tradfitional schools are numeric, 
    df['charter_encoded'] = df.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 
                                                       'TRADITIONAL ISD/CSD':0})
    
    #filters the dataset, so that only A01 and A03 are listed
    df=df[(df.heading == 'A01') | (df.heading ==  'A03') | (df.heading ==  'H06')]

    # filters out any student count that is -999 which is TEA's masked count
    df=df[df['student_count'] != '-999']

    #removes and replaces < that some student counts have
    df['student_count']= df['student_count'].str.replace("<", "")
    
    #changes data type to float
    df['student_count'] = df['student_count'].astype(float)
    
    #creates a pivot table with campus number as index and creating heading columns with student count and drops any nulls
    pivot=df.pivot(index='campus_number', columns='heading', values= 'student_count').dropna()
    
    #merges pivot table onto original df, but keeps the values of all the pivot table  
    df=df.merge(pivot,how= 'right', on= 'campus_number')

    #renames A01 and A03
    df=df.rename(columns={'A01': 'student_enrollment', 'A03':'discipline_count', 'H06':'iss'})

    #creates a new column by dividing the discipline by enrolled
    df['discipline_percent']= ((df['discipline_count']/df['student_enrollment'])*100)

    #creates a new column based on iss divided by student enrollment
    df['iss_percent']= ((df['iss']/df['student_enrollment'])*100)
    
    #rounds the percent
    df=df.round({'discipline_percent': 0})
    df=df.round({'iss_percent':0})
    
    # removes unnecessary columns
    df=df.drop(columns=['agg_level', 'campus_number', 'region', 'charter_status', 
                                'dist_name_num', 'student_count','section', 'heading',
                                'heading_name', 'student_count'])
    #removes any nulls
    df.dropna()

    #drops the duplicate rows that were created by the merge
    df=df.drop_duplicates()

    #reset the index
    df=df.reset_index(drop=True)
    
    #returns a fresh and prepped df
    return df

def df_combine(cs18,cs19,cs20,cs21,cs22):
    df=pd.concat([cs18,cs19,cs20,cs21,cs22], ignore_index=True)
    return(df)
###################################################################################
#################################### SPLIT DATA ###################################
###################################################################################

#Step 5: Test and train dataset split
def split_campus_data(df):
    '''
    This function performs split on tea data, stratify charter_encoded.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.charter_encoded)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.charter_encoded)
    return train, validate, test

#train, validate, test= split_tea_data(df) 