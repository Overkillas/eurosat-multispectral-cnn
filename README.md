# Classificação Multi-Espectral com EuroSAT

Estudo sobre o impacto de bandas espectrais não-visíveis (NIR e SWIR) na classificação de uso/cobertura do solo, usando imagens Sentinel-2 (dataset **EuroSAT**) e uma CNN em TensorFlow/Keras.

A mesma CNN é treinada três vezes, mudando **apenas** os canais de entrada — isolando o efeito das bandas espectrais:

| Modelo | Bandas | Canais |
|--------|--------|--------|
| **A — RGB** | B4, B3, B2 | 3 |
| **B — RGB + NIR** | B4, B3, B2, B8 | 4 |
| **C — Multi-espectral** | todas as 13 | 13 |

Mesma seed, mesmo split (70/15/15), mesma normalização → comparação justa do efeito das bandas extras.

## Resultados

| Modelo | Acurácia (teste) | F1 macro |
|--------|------------------|----------|
| A — RGB | 98,20% | 0,9813 |
| B — RGB+NIR | 98,05% | 0,9795 |
| C — Multi-espectral | **98,62%** | **0,9852** |

Usar todas as 13 bandas deu o melhor resultado, confirmando a hipótese — mas o ganho sobre o RGB é pequeno (+0,4 pp), e o NIR isolado (B) não ajudou. As classes que mais se beneficiaram foram de vegetação e água (PermanentCrop, HerbaceousVegetation, SeaLake). Análise completa em `notebooks/06_presentation.ipynb`.

---

## Arquitetura da CNN

CNN convolucional clássica, **idêntica nos três experimentos** — só muda o número de canais de entrada (3, 4 ou 13). Definida em [`src/models.py`](src/models.py) (`build_cnn`).

São **3 blocos convolucionais** com filtros crescentes (32 → 64 → 128). Cada bloco tem **dois** `Conv2D` 3×3 (`padding='same'`, sem bias) seguidos de `BatchNormalization` + `ReLU`, e termina com `MaxPooling2D` 2×2 que reduz a resolução pela metade. No topo, `GlobalAveragePooling2D` (em vez de `Flatten`), `Dropout(0.3)` e uma `Dense` softmax de 10 classes.

```
Input (64 × 64 × C)                              C = 3 (A) · 4 (B) · 13 (C)
│
├─ Bloco 1   Conv2D(32, 3×3) → BN → ReLU
│            Conv2D(32, 3×3) → BN → ReLU → MaxPool 2×2   →  32 × 32 × 32
│
├─ Bloco 2   Conv2D(64, 3×3) → BN → ReLU
│            Conv2D(64, 3×3) → BN → ReLU → MaxPool 2×2   →  16 × 16 × 64
│
├─ Bloco 3   Conv2D(128, 3×3) → BN → ReLU
│            Conv2D(128, 3×3) → BN → ReLU → MaxPool 2×2  →   8 ×  8 × 128
│
├─ GlobalAveragePooling2D                                →  128
├─ Dropout(0.3)
└─ Dense(10, softmax, float32)                           →  10
```

| Estágio | Saída (modelo C) | Parâmetros |
|---------|------------------|-----------:|
| Bloco 1 (2× Conv32 + BN) | 32 × 32 × 32 | 13.216 |
| Bloco 2 (2× Conv64 + BN) | 16 × 16 × 64 | 55.808 |
| Bloco 3 (2× Conv128 + BN) | 8 × 8 × 128 | 222.208 |
| GAP + Dropout | 128 | 0 |
| Dense (softmax) | 10 | 1.290 |
| **Total** | | **~292 mil** |

O nº de parâmetros é quase idêntico entre os modelos — só a **1ª convolução** muda com os canais de entrada: **A 289.642 · B 289.930 · C 292.522**. Isso reforça a comparação justa: a diferença de acurácia vem das bandas, não do tamanho do modelo.

**Decisões de projeto (o porquê):**
- **Dois Conv por bloco antes do pooling** — amplia o campo receptivo e captura texturas antes de reduzir a resolução.
- **BatchNormalization** — estabiliza e acelera o treino, permitindo taxa de aprendizado maior.
- **Conv sem bias** — o bias é redundante quando seguido de BN.
- **GlobalAveragePooling em vez de Flatten + Dense grande** — corta drasticamente os parâmetros (menos overfitting) e independe da resolução espacial.
- **Dropout(0.3)** antes da saída — regularização.
- **Dense final em `float32`** — estabilidade numérica do softmax sob *mixed precision* (`float16`) na GPU.

**Treino:** otimizador **Adam** (lr `1e-3`), perda **sparse categorical crossentropy**, **mixed precision** (`mixed_float16`) na GPU. Callbacks: `EarlyStopping` (paciência 10, restaura melhores pesos), `ReduceLROnPlateau` (fator 0.5, paciência 5) e `ModelCheckpoint` (melhor `val_accuracy`). Teto de 50 épocas — o early stopping costuma parar antes.

**Data augmentation** (só no treino): flips horizontal/vertical e rotações de 90° — válidos para imagens aéreas. Sem rotação contínua nem *color jitter*, que alterariam as assinaturas espectrais.

---

## Pré-requisitos

- **GPU NVIDIA** + drivers atualizados (o treino usa GPU; roda em CPU, porém lento).
- **Python 3.12** (o TensorFlow ainda não suporta versões mais novas).
- TensorFlow não oferece mais GPU no Windows nativo — em Windows, use **WSL2**. Passo a passo em [`docs/gpu_setup_guide.md`](docs/gpu_setup_guide.md).

Ambiente de referência: TensorFlow 2.21, tensorflow-datasets 4.9.10, Python 3.12.

---

## Como rodar do zero

### 1. Ambiente e dependências

A partir da raiz do projeto, crie e ative um ambiente virtual com Python 3.12 e instale as dependências:

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Linux/WSL  (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# O TensorFlow com GPU já vem pinado (tensorflow[and-cuda]); a flag traz CUDA/cuDNN.
```

### 2. Baixar o dataset (uma única vez)

O host oficial do EuroSAT (`madm.dfki.de`) costuma retornar **HTTP 403**. O script abaixo aponta para um mirror (torchgeo no Hugging Face) que serve exatamente o mesmo arquivo:

```bash
python scripts/prepare_dataset.py
```

Baixa ~2 GB e gera os tfrecords no cache padrão do tfds (`~/tensorflow_datasets/`).

### 3. Rodar os notebooks na ordem

Abra no Jupyter ou VSCode, selecionando o kernel do ambiente virtual criado:

| Ordem | Notebook | O que faz |
|-------|----------|-----------|
| 1 | `00_setup_and_verification.ipynb` | Valida GPU, TF e carga de uma amostra |
| 2 | `01_data_exploration.ipynb` | EDA: classes, bandas, NDVI, histogramas |
| 3 | `02_data_pipeline.ipynb` | Splits + normalização (salva `results/norm_stats.json`) |
| 4 | `03_train_model_a_rgb.ipynb` | Treina Modelo A (3 canais) |
| 5 | `04_train_model_b_rgb_nir.ipynb` | Treina Modelo B (4 canais) |
| 6 | `05_train_model_c_multispectral.ipynb` | Treina Modelo C (13 canais) |
| 7 | `06_presentation.ipynb` | Consolida resultados (não retreina) |

> O notebook **02 precisa rodar antes** dos de treino — ele gera o arquivo de normalização que os modelos consomem.

---

## Testar um modelo treinado

Scripts utilitários (rodar da raiz, com o venv ativo):

Gere localmente amostras de teste a partir do dataset (não versionadas) e classifique-as:

```bash
python scripts/show_results.py                          # resumo das métricas dos 3 modelos
python scripts/export_sample.py                         # gera amostras (.npy + .png) em sample_images/
python scripts/predict_file.py sample_images/ model_c_multispectral
python scripts/predict_demo.py model_c_multispectral    # sanity check visual no test set
```

`predict_file.py` aceita:
- `.npy`/`.tif` com as 13 bandas Sentinel-2 → funciona com qualquer modelo (uso correto);
- `.png`/`.jpg` (foto/recorte de satélite RGB) → só com o modelo A, e apenas para cenas de satélite vistas de cima.

Para testar com **imagens próprias** (fora do dataset, sem viés), coloque seus recortes de satélite em `test_images/` e rode `python scripts/predict_file.py test_images/ model_a_rgb` (ver `test_images/README.md`).

---

## Estrutura

```
.
├── docs/            # plano de implementação + guia de setup de GPU (WSL2)
├── notebooks/       # 00 → 06, na ordem de execução
├── src/             # módulos reutilizáveis (config, dados, modelo, treino, avaliação, plots)
├── scripts/         # download do dataset + utilitários de teste/inferência
├── test_images/     # coloque aqui suas próprias imagens de satélite para testar
├── results/         # modelos (.keras), históricos, métricas, figuras
├── requirements.txt
└── README.md
```

Módulos em `src/`:

- `config.py` — constantes (seed, batch, paths, índices de bandas, classes).
- `data_loader.py` — splits, normalização por banda, augmentation, pipeline `tf.data`.
- `models.py` — `build_cnn(input_shape)`, arquitetura única dos 3 experimentos.
- `training.py` — callbacks, compilação e laço de treino.
- `evaluation.py` — métricas no test set, matriz de confusão, relatórios.
- `visualization.py` — todas as funções de plotagem.

---

## Reprodutibilidade

- Seed fixa (`config.SEED = 42`) em NumPy e TensorFlow; splits determinísticos.
- Normalização calculada **só no treino** e reusada em val/test.
- Versões pinadas em `requirements.txt`.
- Métricas reportadas **sempre no conjunto de teste**.
