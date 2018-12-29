
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import basic_functions as bf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import os
import csv
import datetime
import sys

import warnings
warnings.filterwarnings("ignore") 


# In[2]:


path = '../namesbystate/'
state = 'merged'
full_path = path+state+'.csv'
data_set = pd.read_csv(full_path)

    
# # Sélection du modèle

# In[21]:


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries["Occurence"].rolling(15).mean()
    rolstd =timeseries["Occurence"].rolling(15).std()

    #Plot rolling statistics:
    figure()
    plt.plot(timeseries["Year"],timeseries["Occurence"], color='blue',label='Original')
    plt.plot(timeseries["Year"],rolmean, color='red', label='Rolling Mean')
    plt.plot(timeseries["Year"],rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.close()

    #Perform Dickey-Fuller test:
    #print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries["Occurence"], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print (dfoutput)
    return dfoutput


#On peut démontrer la relation par récurrence. Sk = Diffk + 2*S_k-1 - S_k-2
#TODO à vérifier. Ca donne les valeurs très élevées de prédiction à l'échelle d'origine. Faire la fonction avec récursion pour comparer
def originalScaleOrder2(timeseries):
    ts=pd.Series(timeseries, copy=True)
    for k in range(2,len(timeseries)):
        s=0
        for i in range(2,k+1):
            s+=(k-i+1)*timeseries.iloc[i]
        ts.iloc[k]=(s+2*(k-1)*timeseries.iloc[1]-(k-1)*timeseries.iloc[0])
    return ts

# In[56]:

def arimaModel(dataM,save_path,transformation):
    _pow = np.power
    if transformation=="log":
        trans_func = np.log
        trans_inv = np.exp
    elif transformation=="sqrt":
        trans_func = np.sqrt
        trans_inv = lambda x : _pow(x, 2)
    elif transformation=="curt": #cube root
        trans_func = lambda x : _pow(x, 1/3)
        trans_inv = lambda x : _pow(x, 3)
    elif transformation=="square":
        trans_func= lambda x : _pow(x, 2)
        trans_inv = np.sqrt
        
    ts_log = trans_func(dataM["Occurence"])
    
    #plt.plot(dataM["Year"],ts_log, color = 'blue',label="Transformation")
    #plt.close()
    
    #Differentiation ordre 1
    ts_log_diff = ts_log.diff() #ts_log - ts_log.shift()
    
    # Differentiation ordre 2
    ts_log_diff_2 = ts_log - 2 * ts_log.shift() + ts_log.shift(2)
    
    figure()
    plt.plot(dataM["Year"],ts_log_diff, color = 'blue',label="Diff order 1")
    plt.plot(dataM["Year"],ts_log_diff_2, color = 'green',label="Diff order 2")
    plt.legend(loc='best')
    plt.title('Comparison differenciation order 1 and 2')
    plt.savefig(save_path+'Differentiation_orders.png')
    plt.close()
    
    #Recreate timeseries with result of diff order 1
    new_ts=pd.concat([dataM["Year"],ts_log_diff],axis=1)
    new_ts.dropna(inplace=True)
    
    result=test_stationarity(new_ts)
    with open(save_path+"Results.csv","w",newline="") as file:
        spamwriter=csv.writer(file,delimiter=";")
        spamwriter.writerow(["Test Statistic",'Critical Value (1%)','Critical Value (5%)','Critical Value (10%)'])        
        spamwriter.writerow([str(result["Test Statistic"]),str(result['Critical Value (1%)']),str(result['Critical Value (5%)']),str(result['Critical Value (10%)'])])
    
    if (result["Test Statistic"]>result['Critical Value (1%)']): #and result["Test Statistic"]>result['Critical Value (5%)']): 
    #MAIS déjà que c'est moyens, alors si on diminue la confiance de stationnarité, les résultats sont encore moins bons
        print ("La courbe n'est pas stationnaire ! Différentiation à l'ordre 2.")
    
        #Recreate timeseries with result of diff order 2
        new_ts=pd.concat([dataM["Year"],ts_log_diff_2],axis=1)
        new_ts.dropna(inplace=True)
    
        result=test_stationarity(new_ts)
        
        with open(save_path+"Results.csv","a",newline="") as file:
            spamwriter=csv.writer(file,delimiter=";")
            spamwriter.writerow(["Test Statistic",'Critical Value (1%)','Critical Value (5%)','Critical Value (10%)'])        
            spamwriter.writerow([str(result["Test Statistic"]),str(result['Critical Value (1%)']),str(result['Critical Value (5%)']),str(result['Critical Value (10%)'])])

        
        if (result["Test Statistic"]>result['Critical Value (1%)']):
            raise ValueError("La courbe n'est toujours pas stationnaire ! ")
        else:
            d = 2
    else:
        d = 1
        
    #Stationnaire avec confiance de 99%
    
    
    # In[48]:
    
    
    lag_acf = acf(new_ts["Occurence"], nlags=20)
    lag_pacf = pacf(new_ts["Occurence"], nlags=20, method='ols')
    
    #Plot ACF: 
    plt.figure()
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(new_ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(new_ts)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    
    # Obtenir p = premier x tel que la courbe coupe l'interval supérieur de confiance
    p=0
    while lag_acf[p]>1.96/np.sqrt(len(new_ts)) and p<len(lag_acf):
        p+=1
    
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(new_ts)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(new_ts)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.savefig(save_path+'ACF_PACF.png')
    plt.close()
    
    # Obtenir q = premier x tel que la courbe coupe l'interval supérieur de confiance
    q=0
    while lag_pacf[q]>1.96/np.sqrt(len(new_ts)) and q<len(lag_pacf):
        q+=1
    
    #TODO voir pourquoi ces valeurs 1.96 etc.
        
    # # Training
    
    # In[49]:
    
    met="css"
    
    def modeling(ts_log,p,d,q,save_path,met):
    
        try:
            model = ARIMA(ts_log, order=(p, d, q)) #TODO check if AR and MA models better, even if i don't think so
            results_AR = model.fit(disp=-1,method=met) #Seems that it also works without this ? TODO Check
            print ("Choosing p={}, q={} and d={}".format(p,q,d))
            with open(save_path+"Results.csv","a",newline="") as file:
                spamwriter=csv.writer(file,delimiter=";")
                spamwriter.writerow(["p","q","d"])        
                spamwriter.writerow([str(p),str(q),str(d)])
            flag=1
        except:
            print ("Warning: Chosen p={} and q={} don't work!! Choosing p={}, q={}, d={}".format(p,q,p,q-1,d))
            model = ARIMA(ts_log, order=(p, d, q-1))
            with open(save_path+"Results.csv","a",newline="") as file:
                spamwriter=csv.writer(file,delimiter=";")
                spamwriter.writerow(["p","q","d_corrected"])        
                spamwriter.writerow([str(p),str(q),str(d-1)])
            flag=2
            results_AR = model.fit(disp=-1,method=met) #Uses Kalman filter to fit     
            
        return results_AR,flag
    
    results_AR_mle,flag_mle = modeling(ts_log,p,d,q,save_path,"mle")
    rss_mle=sum((results_AR_mle.fittedvalues-new_ts["Occurence"])**2)
    
    results_AR_cssmle,flag_cssmle = modeling(ts_log,p,d,q,save_path,"css-mle")
    rss_cssmle=sum((results_AR_cssmle.fittedvalues-new_ts["Occurence"])**2)
    if rss_mle<rss_cssmle:
        rss=rss_mle
        met="mle"
        results_AR=results_AR_mle
        flag=flag_mle
    else:
        rss=rss_cssmle
        met="css-mle"
        results_AR=results_AR_cssmle
        flag=flag_cssmle
    
    #Prédictions futures
    forecast_nb=5
    forecast_year=[dataM["Year"].iloc[len(dataM["Year"])-1]+i for i in range(1,forecast_nb+1)]
    forecast=results_AR.predict(start=len(dataM["Year"]),end=len(dataM["Year"])+forecast_nb-1)
    
    new_ts.dropna(inplace=True)
    plt.figure()
    plt.plot(new_ts["Year"],new_ts["Occurence"],label="Original",color='blue')
    plt.plot(dataM["Year"][d:],results_AR.fittedvalues, color='red',label="prediction") #d car nombre de NaN dépend de d.
    plt.plot(forecast_year,forecast,label="Future prediction",color='brown')
    plt.legend(loc="best")
    
    plt.title('RSS: %.4f'% rss)
    plt.savefig(save_path+'Model_Prediction_Transformed.png')
    plt.close()
    
    with open(save_path+"Results.csv","a",newline="") as file:
        spamwriter=csv.writer(file,delimiter=";")
        spamwriter.writerow(["rss"])        
        spamwriter.writerow([str(rss)])
    

    # Traitement inverse des données
    
    # In[51]:
    
    
    full_prediction = results_AR.fittedvalues.append(forecast)
    full_prediction_year = dataM["Year"].append(pd.Series(forecast_year))
    
    #Fait les opérations inverses pour avoir la prédiction avec trend et seasonality
    predictions_ARIMA_diff = pd.Series(full_prediction, copy=True) #pq recreer une autre série ?
    if (d==1):
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index) #à changer quand on met la prédiction future ? possible ?
        predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
        predictions_ARIMA = trans_inv(predictions_ARIMA_log)
        #print(predictions_ARIMA)
        
        #Recréer une timeseries de prédiction  pour rendre plus compréhensible
        figure()
        plt.plot(dataM["Year"],dataM["Occurence"],color="blue",label="Original")
        plt.plot(full_prediction_year[0:len(full_prediction_year)-forecast_nb],predictions_ARIMA[:len(predictions_ARIMA)-forecast_nb],color="Red",label="Prediction")
        plt.plot(full_prediction_year[len(full_prediction_year)-forecast_nb:],predictions_ARIMA[len(predictions_ARIMA)-forecast_nb:],color="Violet",label="Future Prediction")
        plt.legend(loc="best")
        plt.xlim(1910,2020)
        
        rmse=np.sqrt(sum((predictions_ARIMA[forecast_nb:]-dataM["Occurence"])**2)/len(dataM["Occurence"]))
        plt.title('RMSE: %.4f'% rmse)
        plt.savefig(save_path+'Model_Prediction.png')
        plt.close()
        with open(save_path+"Results.csv","a",newline="") as file:
            spamwriter=csv.writer(file,delimiter=";")
            spamwriter.writerow(["rmse"])        
            spamwriter.writerow([str(rmse)])
        
    elif (d==2):
        predictions_ARIMA_log = originalScaleOrder2(predictions_ARIMA_diff)
    
        predictions_ARIMA = trans_inv(predictions_ARIMA_log) #np.exp(predictions_ARIMA_log)
        
        
        #XXX Valeurs trop élevées !!!!
        #TODO pourquoi taille décalée de 1 quand d=2 ? Il y a bien 2 valeurs au début qui ne sont pas calculées, mais quand même. Ca devrait être pris en compte
        figure()
        plt.plot(dataM["Year"],dataM["Occurence"],color="blue",label="Original")
        plt.close()
        figure()
        plt.plot(full_prediction_year[d:len(full_prediction_year)-forecast_nb],predictions_ARIMA[0:len(predictions_ARIMA)-forecast_nb],color="Red",label="Prediction")
        plt.plot(full_prediction_year[len(full_prediction_year)-forecast_nb:],predictions_ARIMA[len(predictions_ARIMA)-forecast_nb:],color="Violet",label="Future Prediction")
        plt.legend(loc="best")
        
        rmse=np.sqrt(sum((predictions_ARIMA[0:len(predictions_ARIMA)-forecast_nb]-dataM["Occurence"])**2)/len(dataM["Occurence"]))
        
        plt.title('RMSE: %.4f'% rmse)
        plt.savefig(save_path+'Model_Prediction.png')
        plt.close()
        
        with open(save_path+"Results.csv","a",newline="") as file:
            spamwriter=csv.writer(file,delimiter=";")
            spamwriter.writerow(["rmse"])        
            spamwriter.writerow([str(rmse)])
    
    #print(full_prediction)
    #print(predictions_ARIMA_log)
    
    return rss,rmse,flag,met


# In[3]:
    
print("Please input a number of names to evaluate: ")
nb_names=int(input())
#☺print("Please input a name to evaluate: ")
#name_list = [input()]
name_list = bf.get_list_names(nb_names) 
state_list= bf.get_list_states()
states_number=len(state_list)
#print("Please input the gender of the name: ")

rss_list=[]
rmse_list=[]
total_count=0
c_notStationary=0
c_correction=0
transformation="log"
time= datetime.datetime.now()
main_path="../results_arima/{}/".format(time.strftime("%Y-%m-%d-%H-%M"))
if not os.path.exists(main_path):
    os.mkdir(main_path)

with open(main_path+"Results.csv","a",newline="") as file:
    spamwriter=csv.writer(file,delimiter=";")
    spamwriter.writerow(["name","state","gender","rss","rmse","transformation","fit_method"])  
for name in name_list:
    for state in state_list[:states_number]:
        data=bf.get_year(state,name)
        #for gender in gender_list:     
        dataM=data[data["Gender"]=="M"]
        dataF=data[data["Gender"]=="F"]  
        gender="M"

        if len(dataM)<len(dataF): #(len(dataM)==0 or (len(dataF)>0 & dataM["Occurence"].iloc[0]<dataF["Occurence"].iloc[0])):
            dataM=dataF
            gender="F"
            
        if len(dataM)==0:
            print("No timeseries")
            continue
        else:
            total_count+=1
            save_path=main_path+'{}-{}-{}/'.format(state,name,gender)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
    
            figure()
            plt.plot(dataM["Year"],dataM["Occurence"])
            plt.savefig(save_path+'Original.png')
            plt.close()
            
            try:
                rss,rmse,flag,met=arimaModel(dataM,save_path,transformation)
                rss_list.append(rss)
                rmse_list.append(rmse)
                if flag==2:
                    c_correction+=1
            except:
                c_notStationary+=1
                #print("Not stationary!")
                continue
            
            with open(main_path+"Results.csv","a",newline="") as file:
                spamwriter=csv.writer(file,delimiter=";")     
                spamwriter.writerow([name,state,gender,str(rss),str(rmse),transformation,met])
                
rmse_nan=pd.Series(rmse_list)

with open(main_path+"Results.csv","a",newline="") as file:
    spamwriter=csv.writer(file,delimiter=";")    
    spamwriter.writerow(["countTotal","count_corrected","count_notStationary","mean_rss","nb_nan","mean_rmse"])
    spamwriter.writerow([str(total_count),str(c_correction),str(c_notStationary),str(np.mean(rss_list)),str(rmse_nan.isna().sum()),str(np.mean(rmse_nan))])
print("Total evaluations: {},  number of corrections of q: {},    number of not starionary ts: {}".format(str(total_count),str(c_correction),str(c_notStationary)))

#name="John"
#state="AK"
#gender="M"
#data=bf.get_year(state,name)
#dataM=data[data["Gender"]==gender]
#save_path='../results_arima/{}-{}-{}/'.format(state,name,gender)
#if not os.path.exists(save_path):
#    os.mkdir(save_path)
#model(dataM,save_path,transformation)


# # Remarques:
# 
# Marche plus ou moins bien suivant les prénoms.
# 
# Texas:
# - John ok
# - Lucas ok
# - Bob non
# - Jaime non
# 
# CA:
# - John ok
# - Sophie non
# - Marie bof: tendance, mais courbes éloignées
# - Catherine bof: idem :'(
# 
# NY:
# - Jon: ordre 2. A améliorer
# 
# Même si la confiance est de 99% pour la stationnarité, la prédiction n'est pas géniale parfois.
# 
# En utilisant ACF et PACF, la plupart du temps, on devrait choisir p=1, q=1, mais ça bugue. Il faut mettre ARIMA(ts_log, order=(2, 1, 2)). J'ai pu mettre 1 et 1 pour Jaime, mais les résultats étaient mauvais.
# 
# ====> A revérifier parce qu'il y avait un bug
# 
# Quand ça suit plus ou moins la tendance, c'est quand même décalé...
# 
# Diff ordre 2 ne marche pas... Parce que pas assez de données ?
# 
# prendre en compte les gdes amplitudes, pas prendre forcément log, ms aussi sqrt etc ?
# aussi prendre en compte les petites
# 
# enlever décalage
# 
# 
# ## Paramètres à changer éventuellement et autres à faire:
# - p et q et d
# - time window de 15 années ====> par exemple, valeur trop grande quand c'est un nouveau prénom utilisé que récemment (genre Khaleesi). En même temps, pour ces prénoms, c'est encore plus dur de prédire leur tendance et il y a très peu de chance qu'ils deviennent les prénoms les plus populaires.
# => Et en fait, changer cette valeur ne change pas la conclusion sur la stationnarité ? Donc **peut prendre une valeur plus faible **
# - les prénoms et states
# - **méthodes pour rendre stationnaire car différentiation ne marche pas toujours!!!!!**
# - Vérifier pour plus de courbes si même si mle ou css-mle pour fit fait diminuer rmle ? on dirait pas pour john...
# - Fit pour moins de valeur puis prédire pour 2015 à 2020 ? Peut comparer pour 2015 à 2017


# Vérifier pourquoi results_AR.forecast ne marche pas, mais results_AR.predict marche mieux ? Ce n'est pas censé faire la même chose ?
# 
# ---
# 
# # References
# 
# https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/  
# https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
# 
# 
# https://otexts.org/fpp2/stationarity.html  
# https://people.duke.edu/~rnau/411arim2.htm  
# https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/  
# https://machinelearningmastery.com/time-series-data-stationary-python/
