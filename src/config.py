"""Constantes centrais do projeto.

Centralizar aqui garante que todos os notebooks/experimentos usem os mesmos
valores — condição necessária para uma comparação justa entre os 3 modelos.
"""

from pathlib import Path

# --- Reprodutibilidade ---
SEED = 42

# --- Hiperparâmetros de treino (idênticos nos 3 experimentos) ---
BATCH_SIZE = 64
IMAGE_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 50          # teto; o EarlyStopping normalmente para antes
LEARNING_RATE = 1e-3

# --- Split dos dados ---
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15     # implícito: 1 - train - val

# --- Dataset ---
# Versão all-bands do EuroSAT no tfds: tensor (64, 64, 13), chave 'sentinel2'.
DATASET_NAME = "eurosat/all"
IMAGE_KEY = "sentinel2"
LABEL_KEY = "label"
NUM_BANDS = 13

# Ordem oficial das classes no tfds (índice = label inteiro).
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# --- Ordem das 13 bandas Sentinel-2 no tensor (índice posicional) ---
# 0:B1  1:B2  2:B3  3:B4  4:B5  5:B6  6:B7  7:B8  8:B8A  9:B9  10:B10  11:B11  12:B12
BAND_NAMES = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B10", "B11", "B12",
]

# Índices de canais por experimento.
#   B4=Vermelho(3), B3=Verde(2), B2=Azul(1), B8=NIR(7)
RGB_INDICES = [3, 2, 1]            # Modelo A — bandas visíveis
RGB_NIR_INDICES = [3, 2, 1, 7]    # Modelo B — visível + infravermelho próximo
ALL_INDICES = list(range(NUM_BANDS))  # Modelo C — todas as 13 bandas

# Índices usados para o cálculo de NDVI = (B8 - B4) / (B8 + B4)
NIR_INDEX = 7   # B8
RED_INDEX = 3   # B4

# --- Paths (construídos de forma portátil a partir da raiz do projeto) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
HISTORIES_DIR = RESULTS_DIR / "histories"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

# Arquivo com média/desvio por banda (calculados no treino, reusados nos modelos).
NORM_STATS_PATH = RESULTS_DIR / "norm_stats.json"


def ensure_dirs() -> None:
    """Cria a árvore de pastas de resultados se ainda não existir."""
    for d in (MODELS_DIR, HISTORIES_DIR, METRICS_DIR, FIGURES_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
