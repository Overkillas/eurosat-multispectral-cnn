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
