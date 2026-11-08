import pandas as pd
import pickle

with open('modelo_eleicao_2022.pkl', 'rb') as arquivo:
    pipeline = pickle.load(arquivo)

novo_registro = pd.DataFrame({
    'NM_MUNICIPIO': ['PRAIA GRANDE'],
    'NR_ZONA': [317],
    'NR_SECAO': [363],
    'DS_CARGO': ['Deputado Estadual'],
    'NR_VOTAVEL': [44171]
})

previsao = pipeline.predict(novo_registro)

print('Quantidade prevista de votos:')
print(round(previsao[0]))