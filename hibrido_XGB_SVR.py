from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings("ignore") 

data_pm10=pd.read_csv("C:/Users/fonta/OneDrive/Documentos/CDD/6. Sexto semestre/Machine learning/Proyecto final/seriepm10.csv")

def makeXy(ts, nb_timesteps):
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        #if i-nb_timesteps <= 4:
            #print(i-nb_timesteps, i-1, i)
        X.append(list(ts.loc[i-nb_timesteps:i-1])) #Regressors
        y.append(ts.loc[i]) #Target
    X, y = np.array(X), np.array(y)
    return X, y

X, y = makeXy(data_pm10['pm10'], 28*24)

train_size = int(len(X) * 0.8)  

train_X, train_y = X[:train_size], y[:train_size]
test_X, test_y = X[train_size:], y[train_size:]


#---------------------------- Paso 1: Ajustar el modelo XGBoost
print(f'Ajustando el modelo XGB')
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42,tree_method='gpu_hist', predictor='gpu_predictor')
xgb_model.fit(train_X, train_y)

xgb_train_pred = xgb_model.predict(train_X)
xgb_test_pred = xgb_model.predict(test_X)

# ----------------------------Paso 2: Calcular residuos
train_residuals = train_y - xgb_train_pred

# ----------------------------Paso 3: Ajustar el modelo SVR en los residuos
print(f'Ajustando el modelo SVR')
svr_model = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=0.01)
svr_model.fit(train_X, train_residuals)

print(f'prediccion de svr en el conjunto de entrenamiento')
svr_train_residuals_pred = svr_model.predict(train_X)

print(f'prediccion de svr en el conjunto de test')
svr_test_residuals_pred = svr_model.predict(test_X)

# ----------------------------Paso 4: Combinar predicciones
hybrid_train_pred = xgb_train_pred + svr_train_residuals_pred
hybrid_test_pred = xgb_test_pred + svr_test_residuals_pred

# Métricas del modelo híbrido

train_r2 = r2_score(train_y, hybrid_train_pred)
test_r2 = r2_score(test_y, hybrid_test_pred)

print(f"R² de entrenamiento (híbrido): {train_r2:.4f}")
print(f"R² de prueba (híbrido): {test_r2:.4f}")

# Paso 5: Prueba de autocorrelación (Ljung-Box)
residuals_hybrid = test_y - hybrid_test_pred
lb_test = acorr_ljungbox(residuals_hybrid, lags=[10], return_df=True)
lb_pvalue = lb_test['lb_pvalue'].iloc[0]

print(f"Prueba de Ljung-Box (p-valor para residuos híbridos): {lb_pvalue:.4f}")
if lb_pvalue > 0.05:
    print("No se detecta autocorrelación significativa en los residuos (p > 0.05).")
else:
    print("Se detecta autocorrelación significativa en los residuos (p ≤ 0.05).")

resultados_hibrido={}
resultados_hibrido['result'] = {
    'score test': test_r2,
    'score train':train_r2,
    'residuos': residuals_hybrid,
    'predicc': hybrid_test_pred,
    'entrenamiento y': train_y,
    'observado': test_y
}

with open('resultados_hibrido_xgb_svr.pkl', 'wb') as file:
    pickle.dump(resultados_hibrido, file)