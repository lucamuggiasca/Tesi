import yfinance as yf
import matplotlib.pyplot as plt
import Sentiment
import numpy as np

# Carica i dati tramite api yfinance
TITMI = yf.Ticker("TIT.MI")
data= TITMI.history(period="1Y",actions=False) 
#data.reset_index(inplace=True)

#Sentiment
print(data.index[0])
print(data.index[-1])
SA = Sentiment.Sentiment_Analysis(query="Telecom", language='it' ,country='IT', start=data.index[0], end=data.index[-1])
polarity_array = np.array('')
for idx in data.index:
    print(idx)
    polarity_array = np.append(polarity_array, SA.do_Analysis(idx, debug=True))
    
    
data['polarity'] = polarity_array[:-1] # probabilmente va anche fatto un flip
plt.plot(polarity_array)
plt.show()

