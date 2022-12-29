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

df22 = pd.read_csv('CAMPUS_summary_22.csv')
df21 = pd.read_csv('CAMPUS_summary_21.csv')
df20 = pd.read_csv('CAMPUS_summary_20.csv')
df19 = pd.read_csv('CAMPUS_summary_19.csv')
df18 = pd.read_csv('CAMPUS_summary_18.csv')

###################################################################################
##################################### PREP DATA ###################################
###################################################################################
# for each school year
def prep22(df):
    global df22
    df22['charter_encoded'] = df22.charter_status.map({'OPEN ENROLLMENT CHARTER':1, 'TRADITIONAL ISD/CSD':0})
    df22=df22[(df22.heading == 'A01') | (df22.heading == 'A03')]
    df22=df22[df22['student_count'] != -999]
    df22['student_count']= df22['student_count'].str.replace("<", "")
    df22['student_count'] = df22['student_count'].astype(float)
    df22.dropna()
    df22=df22.drop_duplicates()
    df22pivot=df22.pivot(index='campus_number', columns='heading', values='student_count').dropna()
    df22=df22.merge(df22pivot,how= 'right', on= 'campus_number')
    df22=df22.drop(columns=['agg_level', 'campus_number', 'region', 'charter_status', 
                            'dist_name_num', 'student_count','section', 'heading',
                            'heading_name', 'student_count'])
    df22=df22.rename(columns={'A01': 'student_enrollment', 'A03':'discipline_count'})
    df22=df22.drop_duplicates()
    df22.dropna()
    df22=df22.reset_index(drop=True)
    df22['discipline_percent']= ((df22['discipline_count']/df22['student_enrollment'])*100)
    df22=df22.round({'discipline_percent': 0})