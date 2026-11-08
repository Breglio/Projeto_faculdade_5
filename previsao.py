import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv(
    r'C:\Projeto integrador data Science eleições\AC.csv',
    sep=';',
    encoding='latin1'
)

df.replace('#NULO#', np.nan, inplace=True)
df.dropna(inplace=True)

print("registros", len(df))

X = df[['NM_MUNICIPIO',
        'NR_ZONA',
        'NR_SECAO',
        'DS_CARGO',
        'NR_VOTAVEL']]

y = df['QT_VOTOS']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.75,
    random_state=20
)

encoder = ColumnTransformer(
    [
        (
            "categorias",
            OneHotEncoder(handle_unknown="ignore"),
            ['NM_MUNICIPIO', 'DS_CARGO', 'NR_VOTAVEL']
        )
    ],
    remainder="passthrough"
)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

modelo = Pipeline([
    ("encoder", encoder),
    ("rf", rf)
])

print("carregando")

modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)

print("\nmodelo")

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))

with open("modelo_eleicao_2022.pkl", "wb") as arq:
    pickle.dump(modelo, arq)

print("\nfim")