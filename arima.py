
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import basic_functions as bf
from statsmodels.tsa.stattools import adfuller, acf, pacf,kpss
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def adf_test(timeseries):
    
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
    dftest = adfuller(timeseries["Occurence"], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput


def kpss_test(timeseries):
    kpsstest = kpss(timeseries["Occurence"], regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    
    return kpss_output


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
    elif transformation=="nothing":
        def initial(ts):
            return ts
        
        trans_func= initial
        trans_inv = initial
        
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
    
    result=adf_test(new_ts)
    with open(save_path+"Results.csv","w",newline="") as file:
        spamwriter=csv.writer(file,delimiter=";")
        spamwriter.writerow(["ADF Test","Test Statistic",'Critical Value (1%)','Critical Value (5%)','Critical Value (10%)'])        
        spamwriter.writerow(["",str(result["Test Statistic"]),str(result['Critical Value (1%)']),str(result['Critical Value (5%)']),str(result['Critical Value (10%)'])])
    
    if (result["Test Statistic"]>result['Critical Value (1%)']): #and result["Test Statistic"]>result['Critical Value (5%)']): 
    #MAIS déjà que c'est moyens, alors si on diminue la confiance de stationnarité, les résultats sont encore moins bons
        print ("La courbe n'est pas stationnaire ! Différentiation à l'ordre 2.")
    
        #Recreate timeseries with result of diff order 2
        new_ts=pd.concat([dataM["Year"],ts_log_diff_2],axis=1)
        new_ts.dropna(inplace=True)
    
        result=adf_test(new_ts)
        
        with open(save_path+"Results.csv","a",newline="") as file:
            spamwriter=csv.writer(file,delimiter=";")
            spamwriter.writerow(["ADF Test","Test Statistic",'Critical Value (1%)','Critical Value (5%)','Critical Value (10%)'])        
            spamwriter.writerow(["",str(result["Test Statistic"]),str(result['Critical Value (1%)']),str(result['Critical Value (5%)']),str(result['Critical Value (10%)'])])

        
        if (result["Test Statistic"]>result['Critical Value (1%)']):
            raise ValueError("La courbe n'est toujours pas stationnaire ! ")
        else:
            #KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test
            results_kpss = kpss_test(new_ts)
            
            with open(save_path+"Results.csv","a",newline="") as file:
                spamwriter=csv.writer(file,delimiter=";")
                spamwriter.writerow(["KPSS Test","Test Statistic",'Critical Value (1%)','Critical Value (5%)','Critical Value (10%)'])        
                spamwriter.writerow(["",str(results_kpss["Test Statistic"]),str(results_kpss['Critical Value (1%)']),str(results_kpss['Critical Value (5%)']),str(results_kpss['Critical Value (10%)'])])
            
            if (results_kpss["Test Statistic"]<results_kpss['Critical Value (2.5%)']):
                raise ValueError("Test KPSS - La courbe à l'ordre 2 n'est pas trend-stationnaire ! ")
                
            d = 2
    else:
        #KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test
        results_kpss = kpss_test(new_ts)
        #print(results_kpss)
        
        with open(save_path+"Results.csv","a",newline="") as file:
            spamwriter=csv.writer(file,delimiter=";")
            spamwriter.writerow(["KPSS Test","Test Statistic",'Critical Value (1%)','Critical Value (5%)','Critical Value (10%)'])        
            spamwriter.writerow(["",str(results_kpss["Test Statistic"]),str(results_kpss['Critical Value (1%)']),str(results_kpss['Critical Value (5%)']),str(results_kpss['Critical Value (10%)'])])
      
        if (results_kpss["Test Statistic"]<results_kpss['Critical Value (2.5%)']):
            raise ValueError("Test KPSS - La courbe n'est pas trend-stationnaire ! ")
        
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
    
    met="mle"
    
    def modelingBrut(ts_log,p,d,q,met):
        model = ARIMA(ts_log, order=(p, d, q)) #TODO check if AR and MA models better, even if i don't think so
        results_AR = model.fit(disp=-1,method=met) #Seems that it also works without this ? TODO Check
        flag=3
        return results_AR,flag
    
    def modeling(ts_log,p,d,q,save_path,met):
    
        try:
            model = ARIMA(ts_log, order=(p, d, q)) #TODO check if AR and MA models better, even if i don't think so
            results_AR = model.fit(disp=-1,method=met) #Seems that it also works without this ? TODO Check
            flag=1
        except:
            model = ARIMA(ts_log, order=(p, d, q-1))
            flag=2
            results_AR = model.fit(disp=-1,method=met) #Uses Kalman filter to fit    
            q=q-1
            
        return results_AR,flag,q
    
    #Brut force
    p_brut=0
    q_brut=0
    min_rss=1000
    for p_it in range(6):
        for q_it in range(6):
            try:
                results_AR_brut,flag_brut=modelingBrut(ts_log,p_it,d,q_it,"mle")
                rss=sum((results_AR_brut.fittedvalues-new_ts["Occurence"])**2)
                #print("{}  {}  {}".format(str(p_it),str(q_it),str(rss)))
                if rss<min_rss:
                    min_rss=rss
                    p_brut=p_it
                    q_brut=q_it
            except:
                break
    
    if (p_brut!=p or q_brut!=q):
        print("Brut choosing different: p={} and q={}".format(p_brut,q_brut))
        with open(save_path+"Results.csv","a",newline="") as file:
            spamwriter=csv.writer(file,delimiter=";")
            spamwriter.writerow(["p_brut","q_brut"])        
            spamwriter.writerow([str(p_brut),str(q_brut)])
    
    
        #Using ACF, PACF analysis
        results_AR_mle,flag_mle,q_cor = modeling(ts_log,p,d,q,save_path,"mle")
        rss_mle=sum((results_AR_mle.fittedvalues-new_ts["Occurence"])**2)
        
        results_AR_cssmle,flag_cssmle,q_cor = modeling(ts_log,p,d,q,save_path,"css-mle")
        rss_cssmle=sum((results_AR_cssmle.fittedvalues-new_ts["Occurence"])**2)
        
        if flag_mle==1:
            print ("Choosing p={}, q={} and d={}".format(p,q,d))
        elif flag_mle==2:
            print ("Warning: Chosen p={} and q={} don't work!! Choosing p={}, q={}, d={}".format(p,q,p,q-1,d))
    
        with open(save_path+"Results.csv","a",newline="") as file:
            spamwriter=csv.writer(file,delimiter=";")
            if q==q_cor:
                spamwriter.writerow(["p","q","d"])        
                
            else:
                spamwriter.writerow(["p","q_cor","d"])  
            spamwriter.writerow([str(p),str(q_cor),str(d)])
                
        if rss_mle==min(rss_mle,min_rss,rss_cssmle):
            print("BETTER MLE")
            rss=rss_mle
            met="mle"
            results_AR=results_AR_mle
            flag=flag_mle
        elif rss_cssmle==min(rss_mle,min_rss,rss_cssmle):
            print("BETTER CSS MLE")
            rss=rss_cssmle
            met="css-mle"
            results_AR=results_AR_cssmle
            flag=flag_cssmle
        else:
            rss=min_rss
            met="css-mle"
            results_AR=results_AR_brut
            flag=flag_brut
    else:
        rss=min_rss
        met="css-mle"
        results_AR=results_AR_brut
        flag=1

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
    
    return rss,rmse,flag,met,d,p,q,p_brut,q_brut


# In[3]:
    
print("Please input a number of names to evaluate: ")
nb_names=int(input())
#print("Please input a name to evaluate: ")
#name_list = [input()]
name_list = bf.get_list_names(nb_names) 
state_list= bf.get_list_states()
states_number=len(state_list)

rss_list=[]
rmse_list=[]
total_count=0
c_notStationary=0
c_correction=0
c_brut=0
transformation="log" #sqrt marche moyennement que pour John TX
time= datetime.datetime.now()
main_path="../results_arima/{}/".format(time.strftime("%Y-%m-%d-%H-%M"))
if not os.path.exists(main_path):
    os.mkdir(main_path)

with open(main_path+"main_results.csv","a",newline="") as file:
    spamwriter=csv.writer(file,delimiter=";")
    spamwriter.writerow(["name","state","gender","rss","rmse","transformation","fit_method","differentiation order","p","q","p_brut","q_brut"])  
for name in name_list:
    for state in state_list[:states_number]:
        print("{} - {}".format(name,state))
        data=bf.get_year(state,name)
        #for gender in gender_list:     
        dataM=data[data["Gender"]=="M"]
        dataF=data[data["Gender"]=="F"]  
        gender="M"

        if len(dataM)<len(dataF): #(len(dataM)==0 or (len(dataF)>0 & dataM["Occurence"].iloc[0]<dataF["Occurence"].iloc[0])):
            dataM=dataF
            gender="F"
            
        if len(dataM)==0:
            print("No timeseries.")
            continue
        else:
            total_count+=1
            save_path=main_path+'{}-{}-{}/'.format(name,state,gender)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
    
            figure()
            plt.plot(dataM["Year"],dataM["Occurence"])
            plt.savefig(save_path+'Original.png')
            plt.close()
            
            try:
                rss,rmse,flag,met,d,p,q,p_brut,q_brut=arimaModel(dataM,save_path,transformation)
                rss_list.append(rss)
                rmse_list.append(rmse)
                if flag==2:
                    c_correction+=1
                elif flag==3:
                    c_brut+=1
            except:
                c_notStationary+=1
                print("Not stationary!")
                os.rename(save_path,main_path+"{}-{}-{}_notStat/".format(name,state,gender))
                continue
            
            with open(main_path+"main_results.csv","a",newline="") as file:
                spamwriter=csv.writer(file,delimiter=";")     
                spamwriter.writerow([name,state,gender,rss,rmse,transformation,met,d,p,q,p_brut,q_brut])
                
rmse_nan=pd.Series(rmse_list)

with open(main_path+"main_results.csv","a",newline="") as file:
    spamwriter=csv.writer(file,delimiter=";")    
    spamwriter.writerow(["countTotal","count_corrected","count_notStationary","count_brut","mean_rss","nb_nan","mean_rmse"])
    spamwriter.writerow([str(total_count),str(c_correction),str(c_notStationary),str(c_brut),str(np.mean(rss_list)),str(rmse_nan.isna().sum()),str(np.mean(rmse_nan))])
print("Total eval: {},  nb of corr of q: {},    nb of not starionary ts: {},  nb model: {},    nb brut better: {}".format(str(total_count),str(c_correction),str(c_notStationary),str(total_count-c_notStationary),str(c_brut)))

#name="John"
#state="CA"
#gender="M"
#data=bf.get_year(state,name)
#dataM=data[data["Gender"]==gender]
#save_path='../results_arima/{}-{}-{}/'.format(state,name,gender)
#if not os.path.exists(save_path):
#    os.mkdir(save_path)
#arimaModel(dataM,save_path,transformation)

#RQ : ne marche pas quand les valeurs sont plus faibles. ex occurence qui est <2000 au max.
