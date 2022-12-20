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
import warnings
warnings.filterwarnings("ignore")


###################################################################################
#################################### ACQUIRE DATA #################################
###################################################################################

#go to : https://rptsvr1.tea.texas.gov/adhocrpt/Disciplinary_Data_Products/Download_All_Districts.html
#download the csv for 2018/19, 2019/20, 2020/21, and 2021/22 School Years
#upload csv files:
df22 = pd.read_csv('DISTRICT_summary_22.csv')
df21 = pd.read_csv('DISTRICT_summary_21.csv')
df20 = pd.read_csv('DISTRICT_summary_20.csv')
df19 = pd.read_csv('DISTRICT_summary_19.csv')

###################################################################################
##################################### PREP DATA ###################################
###################################################################################


#for each school year
def prep22(df):
    global df22
    df22=df22.rename(columns={'AGGREGATION LEVEL':'agg_level', 'REGION':'region', 'DISTNAME':'dist_name', 
                              'DISTRICT':'district_num', 'CHARTER_STATUS':'charter_status', 'SECTION':'section', 
                              'HEADING':'heading', 'HEADING NAME':'heading_name'})
    df22['charter_encoded'] = df22.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 'TRADITIONAL ISD/CSD':0})
    df22=df22[(df22.heading == 'A01') | (df22.heading ==  'A03')]
    df22 = df22[df22['count'] != -999]
    df22.dropna()
    df22=df22.drop_duplicates()
    df22pivot=df22.pivot(index='district_num', columns='heading_name', values= 'count').dropna()
    df22=df22.merge(df22pivot,how= 'right', on= 'district_num')
    df22=df22.drop(columns=['heading', 'heading_name','count', 'agg_level', 'region', 'district_num', 'section',
                            'charter_status'])
    df22=df22.rename(columns={'DISTRICT CUMULATIVE YEAR END ENROLLMENT': 'enrollment', 
                              'DISTRICT DISCIPLINE RECORD COUNT':'disciplined'})
    df22=df22.drop_duplicates()
    df22.dropna()
    df22=df22.reset_index(drop=True)
    df22['discipline_percent']= ((df22['disciplined']/df22['enrollment'])*100)
    df22=df22.round({'discipline_percent': 0})

#call function with: prep22(df22)

def prep21(df):
    global df21
    df21=df21.rename(columns={'AGGREGATION LEVEL':'agg_level', 'REGION':'region', 'DISTNAME':'dist_name', 
                              'DISTRICT':'district_num', 'CHARTER_STATUS':'charter_status', 'SECTION':'section', 
                              'HEADING':'heading', 'HEADING NAME':'heading_name'})
    df21['charter_encoded'] = df21.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 'TRADITIONAL ISD/CSD':0})
    df21=df21[(df21.heading == 'A01') | (df21.heading ==  'A03')]
    df21 = df21[df21['count'] != -999]
    df21.dropna()
    df21=df21.drop_duplicates()
    df21pivot=df21.pivot(index='district_num', columns='heading_name', values= 'count').dropna()
    df21=df21.merge(df21pivot,how= 'right', on= 'district_num')
    df21=df21.drop(columns=['heading', 'heading_name','count', 'agg_level', 'region', 'district_num', 
                            'section','charter_status'])
    df21=df21.rename(columns={'DISTRICT CUMULATIVE YEAR END ENROLLMENT': 'enrollment', 
                              'DISTRICT DISCIPLINE RECORD COUNT':'disciplined'})
    df21=df21.drop_duplicates()
    df21.dropna()
    df21=df21.reset_index(drop=True)
    df21['discipline_percent']= ((df21['disciplined']/df21['enrollment'])*100)
    df21=df21.round({'discipline_percent': 0})

#call function with: prep21(df21)

def prep20(df):
    global df20
    df20=df20.rename(columns={'AGGREGATION LEVEL':'agg_level', 'REGION':'region', 'DISTNAME':'dist_name', 
                              'DISTRICT':'district_num', 'CHARTER_STATUS':'charter_status', 'SECTION':'section', 
                              'HEADING':'heading', 'HEADING NAME':'heading_name'})
    df20['charter_encoded'] = df20.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 'TRADITIONAL ISD/CSD':0})
    df20=df20[(df20.heading == 'A01') | (df20.heading ==  'A03')]
    df20 = df20[df20['count'] != -999]
    df20.dropna()
    df20=df20.drop_duplicates()
    df20pivot=df20.pivot(index='district_num', columns='heading_name', values= 'count').dropna()
    df20=df20.merge(df20pivot,how= 'right', on= 'district_num')
    df20=df20.drop(columns=['heading', 'heading_name','count', 'agg_level', 'region', 'district_num', 
                            'section','charter_status'])
    df20=df20.rename(columns={'DISTRICT CUMULATIVE YEAR END ENROLLMENT': 'enrollment', 
                              'DISTRICT DISCIPLINE RECORD COUNT':'disciplined'})
    df20=df20.drop_duplicates()
    df20.dropna()
    df20=df20.reset_index(drop=True)
    df20['discipline_percent']= ((df20['disciplined']/df20['enrollment'])*100)
    df20=df20.round({'discipline_percent': 0})

#call function with: prep20(df20)

def prep19(df):
    global df19
    df19=df19.rename(columns={'AGGREGATION LEVEL':'agg_level', 'REGION':'region', 'DISTNAME':'dist_name',                                                     'DISTRICT':'district_num', 'CHARTER_STATUS':'charter_status', 'SECTION':'section',                                               'HEADING':'heading', 'HEADING NAME':'heading_name'})
    df19['charter_encoded'] = df19.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 'TRADITIONAL ISD/CSD':0})
    df19=df19[(df19.heading == 'A01') | (df19.heading ==  'A03')]
    df19 = df19[df19['count'] != -999]
    df19.dropna()
    df19=df19.drop_duplicates()
    df19pivot=df19.pivot(index='district_num', columns='heading_name', values= 'count').dropna()
    df19=df19.merge(df19pivot,how= 'right', on= 'district_num')
    df19=df19.drop(columns=['heading', 'heading_name','count', 'agg_level', 'region', 'district_num',                                                       'section','charter_status'])
    df19=df19.rename(columns={'DISTRICT CUMULATIVE YEAR END ENROLLMENT': 'enrollment', 
                          'DISTRICT DISCIPLINE RECORD COUNT':'disciplined'})
    df19=df19.drop_duplicates()
    df19.dropna()
    df19=df19.reset_index(drop=True)
    df19['discipline_percent']= ((df19['disciplined']/df19['enrollment'])*100)
    df19=df19.round({'discipline_percent': 0})

#call function with: prep19(df19)

def df_combine(a,b,c,d):
    df=pd.concat([df19,df20,df21,df22], ignore_index=True)
    return(df)

#call function with: df_combine(df19,df20,df21,df22)



#combine the files
df= pd.concat([df19,df20,df21,df22], ignore_index=True)

###################################################################################
#################################### SPLIT DATA ###################################
###################################################################################

#Step 5: Test and train dataset split
def split_tea_data(df):
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

