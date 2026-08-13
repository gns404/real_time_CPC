import numpy as np
import xarray as xr
import pickle
from sklearn.mixture import GaussianMixture

##load data
# Z500: CORe: [6hourly > daily] > [regird -140 ~ -60 / 0 - 60 / 5 * 5] > [anomaly (climatology:1981-2010)]
filepath_z500 = "/work05/home/jihun/seasonal_forecast/reanalysis/CORe.daily.z500.ano.1981-2010.nc"
ds = xr.open_dataset(filepath_z500)
z500_da = ds.squeeze().z500
z500_train = z500_da.values
nt,ny,nx = z500_train.shape
z500_train = np.reshape(z500_train, [nt, ny*nx], order='F')

## GMM train and save
# number of weather types
n = 5

gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0).fit(z500_train)

model = gmm
fileName = "gmm_CORe.sav"
pickle.dump(model, open(fileName, 'wb'))

print(f"trained samples: {z500_train.shape[0]}, features: {z500_train.shape[1]}")    
