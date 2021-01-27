import pandas as pd
veri = pd.read_csv('lungcancer.csv')
giris = veri.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]].values
sinif = veri.iloc[:,56].values
from sklearn.model_selection import train_test_split as tts
egitim_giris,test_giris,egitim_sinif,test_sinif=tts(giris,sinif, test_size=0.4)

from sklearn.linear_model import Perceptron
model=Perceptron()
model.fit(egitim_giris,egitim_sinif)
sonuc=model.score (test_giris,test_sinif)
print(sonuc)