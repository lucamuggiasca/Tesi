import pandas as pd
import yfinance as yf
import numpy as np
import Sentiment
import technical_indicators_lib as til
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime,timedelta,timezone
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error, mean_squared_log_error, median_absolute_error, explained_variance_score



'''
#Definisco il Normalizzatore
class Normalizer():
    # una gaussiana ha 2 parametri: mu (indica la media, il centro, x del picco) e la variazione std
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x): 
        self.mu = np.mean(x, axis=(0)) # mean() fa la media dei dati sull'asse specificato
        self.sd = np.std(x, axis=(0)) # deviazione std
        normalized_x = (x - self.mu)/self.sd
        print(self.mu.shape, self.sd.shape)
        return normalized_x
        
    def inverse_transform(self, x):
        return (x*self.sd) + self.mu
    
    def inverse_transform_lin(self, x):
        return (x*self.sd.iloc[0]) + self.mu.iloc[0]
'''
#Funzione per convertire array di timestamps in float con fuso orario italiano    
def timestamps_to_floats(timestamps):

    # Imposta il fuso orario desiderato(UTC+2()
    tz = timezone(timedelta(hours=2))

    # Converte gli oggetti datetime nel fuso orario specifico e ottiene i timestamp float
    float_array = [timestamp.astimezone(tz).timestamp() for timestamp in timestamps]

    return np.array(float_array)

#Funzione per convertire array di float in datestamp con fuso orario italiano 
def float_to_date(float_dates):
    # Imposta il fuso orario desiderato(UTC+2)
    tz = timezone(timedelta(hours=2))
    # Converti i float in oggetti datetime con il fuso orario specificato
    date_objects = [datetime.fromtimestamp(ts, tz=tz) for ts in float_dates]
    # Formatta le date come stringhe nel formato 'Y-M-D'
    date_strings = [dt.strftime('%Y-%m-%d') for dt in date_objects]

    return date_strings

#Funzione per aggiungere la data successiva al grafico
def add_next_dates(date_array):

    last_date_float = date_array[-1]
    last_date_datetime= datetime.utcfromtimestamp(last_date_float)
    day_of_week=last_date_datetime.weekday()
    next_date_float = last_date_float
    
    if day_of_week==4:
        next_date_float += 86400 * 3  # Se è un venerdì, aggiungo tre giorni per ottenere il lunedì successivo
    else:
        next_date_float += 86400

    extended_dates = np.append(date_array, next_date_float)
    return extended_dates

#Funzione per smoothing dei dati per singola colonna tramite media mobile
def apply_rolling_mean(data, column, window_size):
    data[column] = data[column].rolling(window=window_size).mean()
    data = data.dropna()  # Rimuovi i valori mancanti generati dalla media mobile
    return data


sentiment = 1
initial_price_plot = 0

# Carica i dati tramite api yfinance
TITMI = yf.Ticker("MONC.MI")
data= TITMI.history(period="1Y",actions=False) 
data_sent=data

#Reset dell'indice
data.reset_index(inplace=True)

# Converti le colonne da stringa a float
data['Date'] = timestamps_to_floats(data['Date']) 

#Real close prices plot
if initial_price_plot == 1:
    dates = data['Date']
    first_date = dates.iloc[0]
    last_date= dates.iloc[-1]
    num_ticks = 5
    date_intervals = (last_date - first_date) / num_ticks
    x_ticks = [first_date + i * date_intervals for i in range(num_ticks)]
    x_ticks.append(last_date)
    x_labels = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in x_ticks] 
    x_labels[0] = (datetime.utcfromtimestamp(x_ticks[0]) + timedelta(days=1)).strftime('%Y-%m-%d')
    x_labels[5] = (datetime.utcfromtimestamp(x_ticks[5]) + timedelta(days=1)).strftime('%Y-%m-%d')
    plt.figure(figsize=(12, 6))
    plt.plot(dates, data['Close'], color='blue', linewidth=2, label='Close prices from 2022-10-13 to 2023-10-12')
    plt.title('Actual close prices')
    plt.xlabel('Data')
    plt.ylabel('Valore di chiusura nel mercato (€)')
    plt.xticks(x_ticks, x_labels)  # Imposta le ticks dell'asse x
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True)
    plt.show()


#Applico smoothing ai prezzi di chiusura
data = apply_rolling_mean(data, 'Close', 5)

# Converto i nomi delle colonne in minsucolo perchè richiesti così dalla libreria technical_indicators_lib
data = data.rename(columns={"Open": "open", "High": "high", "Low":"low", "Close": "close", "Volume": "volume"}) 

#Indicatori
data = til.SMA().get_value_df(data, 7)
data = data.rename(columns={'SMA':'SMA7'})
data = til.SMA().get_value_df(data, 14)
data = data.rename(columns={'SMA':'SMA14'})

data = til.EMA().get_value_df(data, 7)
data = data.rename(columns={'EMA':'EMA7'})
data = til.EMA().get_value_df(data, 14)
data = data.rename(columns={'EMA':'EMA14'})

data = til.RSI().get_value_df(data, 14)
data = til.CCI().get_value_df(data, 14)
#data = til.ADI().get_value_df(data) inutilizzabile per problema libreria
data = til.StochasticKAndD().get_value_df(data, 14)
data = til.MACD().get_value_df(data)
data = til.ATR().get_value_df(data)

percorso_file_csv = r"C:\\Users\\lucam\\OneDrive\\Desktop\\test.csv"
data.to_csv(percorso_file_csv, index=False)

#ALL_POSSIBLE_FEATURES:'open','high','low','close','volume','SMA7','SMA14','EMA7','EMA14','RSI','CCI','MACD','stoc_k','stoc_d','MACD_signal_line','ATR','polarity'

#Features impostate per il training
train_test_nextpred_features=['open','high','low','close','volume']

# Calcola il prezzo di chiusura del giorno successivo e aggiungi come target
data['Next_Close']=data['close'].shift(-1)

# Ordina il DataFrame in base alle date
data.sort_values(by='Date', inplace=True)

#Elimino tuple con null
data=data.dropna()

#Split in X e y
X_notnormalize=data[train_test_nextpred_features]
y_notnormalize=data['Next_Close']

#Normalizzazione
ss = StandardScaler()
mm = MinMaxScaler()
X_trans = ss.fit_transform(X_notnormalize)
y_trans =mm.fit_transform(y_notnormalize.values.reshape(-1,1))

# Calcola l'indice per dividere i dati
split_index = int(0.7 * len(data))

# Dividi i dati in set di addestramento, test e new
X_train = X_trans[:split_index]
Y_train_norm = y_trans[:split_index]

split_new = int(0.66 * len(data[split_index:]))

X_test = (X_trans[split_index:])[:split_new]
Y_test_norm = (y_trans[split_index:])[:split_new]

X_new = (X_trans[split_index:])[split_new:]
Y_new_norm = (y_trans[split_index:])[split_new:]

#Applico il regressore AdaBoost
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=100, learning_rate=1, random_state=42)
ada.fit(X_train, Y_train_norm)
Y_pred_train_norm = ada.predict(X_train)
Y_pred_test_norm = ada.predict(X_test)

#Previsione su dati mai visti
Y_pred_new_norm = ada.predict(X_new)

#Denormalizzo
Y_train = mm.inverse_transform(Y_train_norm.reshape(-1,1))
Y_pred_train = mm.inverse_transform(Y_pred_train_norm.reshape(-1,1))
Y_test = mm.inverse_transform(Y_test_norm.reshape(-1,1))
Y_pred_test = mm.inverse_transform(Y_pred_test_norm.reshape(-1,1))
Y_new = mm.inverse_transform(Y_new_norm.reshape(-1,1))
Y_pred_new = mm.inverse_transform(Y_pred_new_norm.reshape(-1,1))

#Sentiment
if sentiment == 1:
    for i in range(new_cutoff-5):
        idx1 = new_cutoff - i
        idx2 = idx1 + 5
        # print("data: ",data.index[-idx1], " - ", data.index[-idx2])
        Sent = mySentiment.Sentiment_Analysis(query=query,
                                              start=data.index[-idx2], 
                                              end=data.index[-idx1])
        s_a = Sent.do_Analysis(True)
        data_sa = np.zeros((new_cutoff-5,2))
        data_sa[i][0] = test_predict[i+5]
        data_sa[i][1] = s_a
    print(data_sa)

#Metriche di valutazione

#Calcolo Cross Validation
#Calcolo MSE
#print("MSE train: %f" % mean_squared_error(Y_train, Y_pred_train))
print("MSE test: %f" % mean_squared_error(Y_test, Y_pred_test))
#Calcolo MAPE
#print("MAPE train: %f" % (mean_absolute_percentage_error(Y_train, Y_pred_train)*100))
print("MAPE test: %f" % (mean_absolute_percentage_error(Y_test, Y_pred_test)*100))
#Calcolo MSE new
print("MSE new test: %f" % mean_squared_error(Y_new, Y_pred_new))
#Calcolo MAPE new
print("MAPE new test: %f" % (mean_absolute_percentage_error(Y_new, Y_pred_new)*100))
'''
scores = cross_val_score(ada, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
mean_mse_train = -scores.mean()
scores = cross_val_score(ada, X_test, Y_test, cv=5, scoring='neg_mean_squared_error')
mean_mse_test = -scores.mean()
print ("CV MSE train: ", mean_mse_train)
print ("CV MSE test: ", mean_mse_test)

#Calcolo R2
print("R2 train: %f" % r2_score(Y_train, Y_pred_train))
print("R2 test: %f" % r2_score(Y_test, Y_pred_test))
#Calcolo MAE
print("MAE train: %f" % mean_absolute_error(Y_train, Y_pred_train))
print("MAE test: %f" % mean_absolute_error(Y_test, Y_pred_test))

#Calcolo MSLE
print("MSLE train: %f" % mean_squared_log_error(Y_train, Y_pred_train))
print("MSLE test: %f" % mean_squared_log_error(Y_test, Y_pred_test))
#Calcolo MedAE
print("MedAE train: %f" % median_absolute_error(Y_train, Y_pred_train))
print("MedAE test: %f" % median_absolute_error(Y_test, Y_pred_test))
#Calcolo Explained Variance Score
print("EVS train: %f" % explained_variance_score(Y_train, Y_pred_train))
print("EVS test: %f" % explained_variance_score(Y_test, Y_pred_test))
#Adjusted R2
R2adj_train = 1 - (1-ada.score(X_train, Y_train))*(len(Y_train)-1)/(len(Y_train)-X_train.shape[1]-1)
R2adj_test = 1 - (1-ada.score(X_test, Y_test))*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)
print("R2adj train: ",R2adj_train)
print("R2adj test: ",R2adj_test)
'''

# Estrai le date dalle colonne
dates_train = data['Date'][:split_index].values
dates_train_shifted=add_next_dates(dates_train[1:])
dates_test = (data['Date'][split_index:])[:split_new].values
dates_test_shifted=add_next_dates(dates_test[1:])

#Numero di periodi di divisione della griglia
num_ticks = 5

# Grafico per i dati di addestramento
x_ticks_train_indices = np.linspace(0, len(dates_train_shifted) - 1, num_ticks, dtype=int)
x_labels_train = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in dates_train_shifted[x_ticks_train_indices]]
titolo = 'Andamento del modello con parametri impostati: '
titolo += ', '.join(train_test_nextpred_features)
plt.figure(figsize=(12, 6))
plt.suptitle(titolo, fontsize='large')
plt.subplot(1, 2, 1)
plt.scatter(dates_train_shifted, Y_train, color='blue', label='Training Data', s=10)
plt.plot(dates_train_shifted, Y_pred_train, color='red', linewidth=2, label='Modello')
plt.title('Training Data')
plt.xlabel('Data')
plt.ylabel('Valore di Chiusura del Giorno Successivo (€)')
plt.legend(loc='upper left', fontsize='small')
plt.xticks(dates_train_shifted[x_ticks_train_indices], x_labels_train)
plt.grid(True)

# Grafico per i dati di test
x_ticks_test_indices = np.linspace(0, len(dates_test_shifted) - 1, num_ticks, dtype=int)
x_labels_test = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in dates_test_shifted[x_ticks_test_indices]]
plt.subplot(1, 2, 2)
plt.scatter(dates_test_shifted, Y_test, color='blue', label='Dati di test', s=10)
plt.plot(dates_test_shifted, Y_pred_test, color='red', linewidth=2, label='Modello')
plt.title('Validation Data')
plt.xlabel('Data')
plt.ylabel('Valore di Chiusura del Giorno Successivo (€)')
plt.legend(fontsize='small')
plt.xticks(dates_test_shifted[x_ticks_test_indices], x_labels_test)
plt.grid(True)
plt.tight_layout()
plt.show()

# Previsione del prezzo di chiusura del giorno successivo
next_day_prediction = ada.predict(latest_data)
print("Previsione del prezzo di chiusura del giorno successivo (€):", next_day_prediction[0])

# Creazione di un grafico separato per gli ultimi 5 giorni e la previsione finale
plt.figure(figsize=(12, 6))
dates_last_five = dates_test_shifted[-5:] 
values_last_five_pred = Y_pred_test[-5:]
predicted_next_day = next_day_prediction[0]
values_last_five_actual = normalized_data['Next_Close'].tail(5).values

# Aggiungi la data successiva all'ultimo giorno
dates_last_five_plus_1 = add_next_dates(dates_last_five)

# Next Day prevision Plot
# Impostazione di valori per ticks e label
num_ticks = 5
x_ticks_plot = np.linspace(0, len(dates_last_five) - 1, num_ticks, dtype=int)
x_ticks_scatter = np.append(x_ticks_plot, len(x_ticks_plot))
x_labels = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in dates_last_five_plus_1[x_ticks_scatter]]
plt.plot(x_ticks_plot, values_last_five_pred, color='red', marker='o', label='Prediction ultimi 5 giorni')

#Grafico
plt.plot(x_ticks_plot, values_last_five_actual, color='blue', marker='o', label='Dati attuali ultimi 5 giorni')
plt.scatter(len(x_ticks_scatter) - 1, predicted_next_day, color='red', marker='o', label='Prev. Giorno Successivo')
plt.title('Ingrandimento sugli ultimi 5 giorni di Validation Data con previsione per il giorno successivo')
plt.xlabel('Data')
plt.ylabel('Valore di Chiusura del Giorno Successivo (€)')
plt.legend()
plt.xticks(x_ticks_scatter, x_labels)
plt.grid(True)  
plt.tight_layout()
plt.show()


#previsione su dati mai visti
Y_pred_new = ada.predict(X_new)

#Calcolo MSE
print("MSE new test: %f" % mean_squared_error(Y_new, Y_pred_new))
#Calcolo MAPE
print("MAPE new test: %f" % (mean_absolute_percentage_error(Y_new, Y_pred_new)*100))

'''
#Predizione sugli ultimi tot giorni di validazione
tot_giorni_start=60
tot_giorni_end=6
last_tot_real_data = Y_test[-tot_giorni_start:-tot_giorni_end]
last_tot_pred_data = Y_pred_test[-tot_giorni_start:-tot_giorni_end]


#Calcolo MSE per tot giorni
print("MSE tot test: %f" % mean_squared_error(last_tot_real_data, last_tot_pred_data))

#Calcolo MAPE per tot giorni
print("MAPE tot test: %f" % (mean_absolute_percentage_error(last_tot_real_data, last_tot_pred_data)*100))
'''