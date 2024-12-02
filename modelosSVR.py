import pandas as pd
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import  r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        #if i-nb_timesteps <= 4:
            #print(i-nb_timesteps, i-1, i)
        X.append(list(ts.loc[i-nb_timesteps:i-1])) #Regressors
        y.append(ts.loc[i]) #Target
    X, y = np.array(X), np.array(y)
    return X, y

data_pm10=pd.read_csv("C:/Users/fonta/OneDrive/Documentos/CDD/6. Sexto semestre/Machine learning/Proyecto final/seriepm10.csv")

resultados_svr = {}

for i in [7 * 24, 14 * 24, 21 * 24, 28 * 24]:
    nb_timesteps = i
    X, y = makeXy(data_pm10['pm10'], nb_timesteps)

    train_size = int(len(X) * 0.6)  # 60% entrenamiento
    val_size = int(len(X) * 0.2)    # 20% validación

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Definir los hiperparámetros para probar (solo C y gamma)
    param_grid = {
        'C': [0.1, 1, 10],          
        'gamma': [ 0.01, 0.1, 1],   
    }

    best_score = -np.inf
    best_params = None
    best_model_svr = None
    print(f"ventana {int(i/24)}")

    # Iterar sobre los hiperparámetros
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:

            print(f'C: {C} gamma: {gamma}')
            
            # Crear el pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),        # Escalar los datos
                ('svr', SVR(kernel='rbf', C=C, gamma=gamma))  # Usar el kernel radial (RBF)
            ])

            # Ajustar el modelo con los datos de entrenamiento
            pipeline.fit(X_train, y_train)

            # Evaluar en el conjunto de validación
            y_val_pred = pipeline.predict(X_val)
            score = r2_score(y_val, y_val_pred)

            # Guardar el mejor modelo
            if score > best_score:
                best_score = score
                best_params = {
                    'C': C,
                    'gamma': gamma
                }
                best_model_svr = pipeline

    # Evaluar en el conjunto de prueba
    y_test_pred = best_model_svr.predict(X_test)

    # Almacenar los resultados
    resultados_svr[i] = {
        'parametros': best_params,
        'score val': best_score,
        'score test': r2_score(y_test, y_test_pred),
        'residuos': y_test - y_test_pred,
        'predicc': y_test_pred,
        'modelo': best_model_svr,
        'entrenamiento y': y_train,
        'observado': y_test
    }

with open('resultados_SVR.pkl', 'wb') as file:
    pickle.dump(resultados_svr, file)
