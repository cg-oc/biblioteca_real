from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlxtend.evaluate import bias_variance_decomp
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAÇÃO ---
SEMENTE = 42
TAMANHO_TESTE = 0.3
N_CPUS = 4  # Alinhado com o cpus-per-task do SLURM

modelos = ["DummyRegressor", "LinearRegression", "RidgeCV", "LassoCV"]
chaves = ["mse", "vies", "variancia"]

# Importa DF e amostra
PASTA_DATA = Path(__file__).resolve().parent / "data"
df = pd.read_pickle(PASTA_DATA / "refractive_index.pkl")
y = df["Indice de Refração"].to_numpy()
X = df.drop(columns=["Indice de Refração"]).to_numpy()

# DIVISAO TREINO-TESTE
TAMANHO_TESTE = 0.3
tamanhos_de_amostra = np.unique(np.logspace(2, 4.5, num=100).astype("int"))
lambdas = np.logspace(-4, 4, num=100)

X_treino_total, X_teste, y_treino_total, y_teste = train_test_split(
    X, y, test_size=TAMANHO_TESTE, random_state=SEMENTE
)

def processar_tamanho(tamanho_amostra):
    X_treino = X_treino_total[:tamanho_amostra]
    y_treino = y_treino_total[:tamanho_amostra]


    # DEFININDO MODELOS PARA BIAS_VARIANCE_DECOMP
    modelos_bench = [
        make_pipeline(StandardScaler(), DummyRegressor()),
        make_pipeline(StandardScaler(), LinearRegression()),
        make_pipeline(StandardScaler(), RidgeCV(alphas=lambdas, cv=5)),
        make_pipeline(
            StandardScaler(), LassoCV(alphas=lambdas, cv=5, max_iter=5_000)
        ),
    ]

    resultado_tamanho = {}

    for modelo in modelos_bench:
        nome_classe = modelo.steps[-1][1].__class__.__name__

        mse, vies, variancia = bias_variance_decomp(
            modelo,
            X_treino, y_treino,
            X_teste, y_teste,
            loss='mse',
            num_rounds=100,
            random_seed=SEMENTE
        )

        resultado_tamanho[nome_classe] = {
            "mse": mse,
            "vies": vies,
            "variancia": variancia
        }

    return resultado_tamanho

# --- EXECUÇÃO PARALELA (UTILIZA AS 4 CPUS) ---
resultados = Parallel(n_jobs=N_CPUS, verbose=5)(
    delayed(processar_tamanho)(tamanho)
    for tamanho in tamanhos_de_amostra
)

# --- CONSOLIDAÇÃO E SALVAMENTO ---
modelos = ["DummyRegressor", "LinearRegression", "RidgeCV", "LassoCV"]
chaves = ["mse", "vies", "variancia"]

for modelo in modelos:
    dados_modelo = {chave: [res[modelo][chave] for res in resultados] for chave in chaves}
    df_res = pd.DataFrame(dados_modelo, index=tamanhos_de_amostra)
    df_res.to_csv(PASTA_DATA / f"resultado_42_{modelo.lower()}.csv")