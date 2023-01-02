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


#for each school year
def prep22(df22):
    df22.rename(columns={'AGGREGATION LEVEL': 'agg_level', 'CAMPUS':'campus_number', 
                              'REGION':'region','DISTRICT NAME AND NUMBER': 'dist_name_num',
                              'CHARTER_STATUS':'charter_status','CAMPUS NAME AND NUMBER': 
                              'campus_name_num', 'SECTION': 'section','HEADING':'heading',
                              'HEADING NAME': 'heading_name', 'YR22':'student_count'})
    
    return(df22)
