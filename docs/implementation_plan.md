# Plano de Implementação — Classificação Multi-Espectral com EuroSAT

> Projeto de classificação de imagens de satélite usando CNN, com foco em avaliar o impacto de bandas espectrais não-visíveis na acurácia do modelo.

---

## 1. Visão geral do projeto

### Problema
Classificar imagens de satélite (Sentinel-2) em **10 categorias de uso do solo**: floresta, pasto, vegetação rasteira, plantação anual/permanente, área residencial, área industrial, rodovia, rio, e mar/lago.

### Hipótese a investigar
**Bandas espectrais não-visíveis (infravermelho próximo e SWIR) melhoram significativamente a classificação de uso do solo em relação a apenas RGB?**

### Estratégia: 3 experimentos com mesma arquitetura
| Modelo | Canais de entrada | Bandas usadas | Total de canais |
|--------|-------------------|---------------|-----------------|
| **A — RGB** | Vermelho, Verde, Azul | B4, B3, B2 | 3 |
| **B — RGB + NIR** | RGB + Infravermelho próximo | B4, B3, B2, B8 | 4 |
| **C — Multi-espectral** | Todas as bandas Sentinel-2 | B1-B12 + B8A | 13 |

Mesma arquitetura, mesmo split de dados, mesma seed — a **única diferença** é o número de canais de entrada. Isso garante comparação justa.

### Diferencial científico
- Análise quantitativa de quanto cada conjunto de bandas contribui
- Comparação por classe (NIR provavelmente ajuda mais em vegetação; SWIR em água/sombra)
- Cálculo de NDVI como análise complementar

---

## 2. Stack tecnológica

| Componente | Escolha | Motivo |
|-----------|---------|--------|
| Linguagem | Python 3.12 (via deadsnakes) | TF não suporta Python 3.14 ainda — instalamos 3.12 ao lado. Ver `gpu_setup_guide.md` |
| Framework DL | TensorFlow 2.15+ / Keras | Pedido pelo usuário; melhor integração com tfds |
| Carregamento de dados | `tensorflow_datasets` (tfds) | EuroSAT já está incluso (RGB e all-bands) |
| GPU | WSL2 + TF GPU | Ver `gpu_setup_guide.md` |
| Análise/visualização | matplotlib, seaborn, scikit-learn | Padrão de mercado |
| Notebooks | Jupyter | Pedido pelo usuário |

---

## 3. Estrutura de pastas

```
av3-caio/
│
├── docs/                              # Documentação do projeto
│   ├── implementation_plan.md         # Este arquivo
│   └── gpu_setup_guide.md             # Guia de setup WSL2 + TF GPU
│
├── notebooks/                         # Notebooks Jupyter (ordem de execução numerada)
│   ├── 00_setup_and_verification.ipynb     # Validar ambiente e GPU
│   ├── 01_data_exploration.ipynb           # EDA do dataset
│   ├── 02_data_pipeline.ipynb              # Construir tf.data pipelines
│   ├── 03_train_model_a_rgb.ipynb          # Treino modelo A (3 canais)
│   ├── 04_train_model_b_rgb_nir.ipynb      # Treino modelo B (4 canais)
│   ├── 05_train_model_c_multispectral.ipynb # Treino modelo C (13 canais)
│   └── 06_presentation.ipynb               # Resultados consolidados (apresentação)
│
├── src/                               # Módulos Python reutilizáveis
│   ├── __init__.py
│   ├── config.py                      # Constantes (SEED, BATCH_SIZE, paths, etc.)
│   ├── data_loader.py                 # Funções de carregamento e pipeline tf.data
│   ├── models.py                      # Arquitetura da CNN
│   ├── training.py                    # Callbacks, função de treino
│   ├── evaluation.py                  # Métricas, matriz de confusão, relatórios
│   └── visualization.py               # Funções de plotagem reutilizáveis
│
├── results/                           # Saídas dos experimentos (gerado, não versionado)
│   ├── models/                        # Pesos salvos (.keras)
│   │   ├── model_a_rgb.keras
│   │   ├── model_b_rgb_nir.keras
│   │   └── model_c_multispectral.keras
│   ├── histories/                     # Históricos de treino (.json)
│   ├── metrics/                       # Métricas finais por modelo (.json)
│   ├── figures/                       # Gráficos salvos (.png)
│   └── logs/                          # Logs do TensorBoard
│
├── requirements.txt                   # Dependências Python
├── .gitignore                         # Ignorar results/, cache, etc.
└── README.md                          # Visão geral + como rodar
```

**Convenções:**
- Nomes de arquivos, pastas e variáveis em **inglês**
- Comentários no código e markdown nos notebooks em **português**
- Notebooks numerados na ordem de execução

---

## 4. Detalhamento dos notebooks

### `00_setup_and_verification.ipynb` — Setup e verificação
**Objetivo:** garantir que o ambiente está funcional antes de tudo.

Conteúdo:
- Markdown explicativo: o que o notebook faz e por que é o primeiro passo
- Verificar versão do TensorFlow
- Detectar GPU (`tf.config.list_physical_devices('GPU')`)
- Validar que CUDA está acessível e listar memória disponível
- Importar `tensorflow_datasets` e listar versões
- Teste pequeno: carregar 1 amostra do EuroSAT e exibir
- **Se algo falhar aqui, parar e consertar antes dos próximos**

### `01_data_exploration.ipynb` — Análise exploratória
**Objetivo:** entender o dataset visualmente e numericamente.

Conteúdo:
- Markdown introduzindo o EuroSAT (origem, Sentinel-2, 10 classes)
- Carregar via `tfds.load('eurosat/all', ...)` (versão all-bands tem as 13)
- Estatísticas: total de imagens, distribuição por classe (gráfico de barras)
- Visualização: grade 10×3 — uma amostra de cada classe
- **Comparação visual de bandas:** mostrar a mesma imagem nas 13 bandas (grade 3×5)
- Cálculo e visualização do **NDVI** = (B8 − B4) / (B8 + B4) em uma imagem de floresta vs uma de área residencial
- Histograma de valores por banda (importante pra escolher normalização)

### `02_data_pipeline.ipynb` — Pipeline de dados
**Objetivo:** construir e validar o `tf.data` pipeline antes de jogar nos modelos.

Conteúdo:
- Markdown explicando o split (70% train / 15% val / 15% test) e a importância de fixar seeds
- Função `make_splits(seed)` que retorna 3 datasets disjuntos
- Aplicar normalização **por banda** (z-score com média/std calculados do treino)
- Data augmentation apropriada pra satélite:
  - Flip horizontal e vertical (válido: imagens aéreas não têm "em pé")
  - Rotações de 90° (válido pelo mesmo motivo)
  - **Não usar:** flip + rotação contínua, color jitter (alteraria assinaturas espectrais!)
- Pipeline final: `cache → shuffle → batch → augment → prefetch(AUTOTUNE)`
- Visualizar batch após augmentation pra confirmar que está OK
- Salvar valores de normalização em arquivo (pra reusar nos notebooks de treino)

### `03_train_model_a_rgb.ipynb` — Modelo A (3 canais)
**Objetivo:** treinar e avaliar a CNN só com bandas visíveis (baseline).

Conteúdo:
- Markdown: descrição do experimento e o que esperar
- Importar funções de `src/`
- Selecionar canais RGB (índices 3, 2, 1 do tensor de 13 bandas → B4, B3, B2)
- Construir modelo via `build_cnn(input_shape=(64, 64, 3))`
- Mostrar `model.summary()` em formato legível
- Treinar com callbacks (ver seção 5)
- Avaliar no test set: acurácia, F1 macro, matriz de confusão, relatório por classe
- Plotar curvas de treino (loss e accuracy)
- Salvar:
  - Modelo treinado em `results/models/model_a_rgb.keras`
  - History em `results/histories/model_a_rgb.json`
  - Métricas em `results/metrics/model_a_rgb.json`
  - Figuras em `results/figures/model_a_*.png`

### `04_train_model_b_rgb_nir.ipynb` — Modelo B (4 canais)
**Objetivo:** testar se adicionar infravermelho próximo melhora.

Diferenças do notebook 03:
- Selecionar canais B4, B3, B2, B8 (índices 3, 2, 1, 7)
- `input_shape=(64, 64, 4)`
- Salvar como `model_b_rgb_nir.*`
- **Tudo mais idêntico** (mesma seed, mesma arquitetura, mesmo schedule)

### `05_train_model_c_multispectral.ipynb` — Modelo C (13 canais)
**Objetivo:** testar com todas as bandas disponíveis.

Diferenças:
- Usar todos os 13 canais
- `input_shape=(64, 64, 13)`
- Salvar como `model_c_multispectral.*`

### `06_presentation.ipynb` — Apresentação consolidada
**Objetivo:** um único notebook auto-contido pra apresentar resultados (sem código de treino).

Conteúdo:
- **Markdown extenso** explicando o trabalho (objetivo, hipótese, metodologia, resultados)
- Carrega resultados salvos de `results/` (sem retreinar nada)
- **Tabela comparativa** dos 3 modelos (accuracy, F1, tempo de treino, nº de parâmetros)
- **Curvas de treino lado a lado** (3 modelos no mesmo eixo)
- **3 matrizes de confusão lado a lado**
- **Gráfico de acurácia por classe** mostrando quais classes mais se beneficiaram das bandas extras
- **Análise NDVI**: mostrar visualmente como NIR ajuda a separar vegetação
- **Conclusão**: o que os resultados mostram, limitações, trabalhos futuros
- Foco em **legibilidade** e **gráficos prontos pra slides**

---

## 5. Detalhamento dos módulos em `src/`

### `config.py`
Constantes do projeto:
```python
SEED = 42
BATCH_SIZE = 64
IMAGE_SIZE = 64
NUM_CLASSES = 10
CLASS_NAMES = ['AnnualCrop', 'Forest', ...]  # ordem oficial do tfds
EPOCHS = 50  # com early stopping isso é teto
LEARNING_RATE = 1e-3
RESULTS_DIR = Path(...)  # paths construídos de forma portátil
```

### `data_loader.py`
- `load_eurosat()` — wrapper do `tfds.load('eurosat/all')`
- `make_splits(ds, train_frac, val_frac, seed)` — split reprodutível
- `select_channels(image, channel_indices)` — seleciona subconjunto de bandas
- `normalize_per_band(image, mean, std)` — normalização z-score por banda
- `build_pipeline(ds, channel_indices, batch_size, augment)` — pipeline tf.data completo
- `get_augmentation_layer()` — camada Keras de augmentation (flips + 90° rotation)

### `models.py`
- `build_cnn(input_shape, num_classes=10)` — arquitetura única usada nos 3 experimentos
- **Arquitetura proposta** (CNN clássica, ~500k parâmetros):
  ```
  Input(input_shape)
    → Conv2D(32, 3, padding='same') → BN → ReLU
    → Conv2D(32, 3, padding='same') → BN → ReLU → MaxPool(2)
    → Conv2D(64, 3, padding='same') → BN → ReLU
    → Conv2D(64, 3, padding='same') → BN → ReLU → MaxPool(2)
    → Conv2D(128, 3, padding='same') → BN → ReLU
    → Conv2D(128, 3, padding='same') → BN → ReLU → MaxPool(2)
    → GlobalAveragePooling2D
    → Dropout(0.3)
    → Dense(num_classes, softmax)
  ```
- Justificativa: pequena o suficiente pra treinar rápido, profunda o suficiente pra capturar texturas, BN estabiliza o treino com batch médios.

### `training.py`
- `get_callbacks(model_name, results_dir)` — retorna lista de callbacks:
  - `ModelCheckpoint` salvando o melhor modelo por val_accuracy
  - `EarlyStopping` com paciência 10 e `restore_best_weights=True`
  - `ReduceLROnPlateau` com paciência 5, fator 0.5
  - `TensorBoard` logando em `results/logs/{model_name}/`
- `train_model(model, train_ds, val_ds, epochs, callbacks)` — executa treino, retorna history

### `evaluation.py`
- `evaluate_on_test(model, test_ds, class_names)` — retorna dict com accuracy, F1, predições, true labels
- `compute_confusion_matrix(y_true, y_pred, class_names)` — retorna matriz e DataFrame
- `get_classification_report(y_true, y_pred, class_names)` — usa sklearn
- `save_metrics(metrics_dict, path)` — serializa em JSON

### `visualization.py`
- `plot_learning_curves(history, save_path=None)` — loss + accuracy lado a lado
- `plot_confusion_matrix(cm, class_names, save_path=None)` — heatmap seaborn
- `plot_band_comparison(image_13bands, save_path=None)` — grade 3×5 das bandas
- `plot_class_distribution(ds, class_names, save_path=None)` — barras
- `plot_comparison_table(results_a, results_b, results_c)` — pra notebook de apresentação
- `plot_per_class_comparison(results_a, results_b, results_c)` — barras agrupadas

---

## 6. Boas práticas aplicadas

### Reprodutibilidade
- Seeds fixas em **numpy, tensorflow, python random** (`config.SEED = 42`)
- Splits feitos com seed
- Mesma ordem de iteração entre experimentos
- `requirements.txt` com versões pinadas

### Performance
- `tf.data` com `.cache()`, `.prefetch(AUTOTUNE)`, `.batch()` antes de augmentation
- **Mixed precision** (`mixed_float16`) na GPU — quase 2× mais rápido
- Augmentation feita como camada Keras (roda na GPU, não na CPU)
- Cache do tfds dentro do filesystem WSL (ver guia de setup)

### Qualidade do código
- Funções em `src/`, notebooks importam — **sem duplicação**
- Type hints onde fizer sentido
- Docstrings curtas explicando o "porquê" (não o "o quê")
- Constantes em `config.py`, não espalhadas

### Qualidade dos notebooks (importante pra apresentação)
- Cada notebook começa com markdown explicativo
- Células de código curtas, intercaladas com markdown
- Outputs limpos (rodar "Restart & Run All" antes de versionar)
- Gráficos com títulos, labels, legendas
- **Não deixar prints de debug**

### Honestidade científica
- Sempre reportar no **test set** (nunca no val)
- Salvar e mostrar matriz de confusão completa (não esconder erros)
- Comparar com seed única primeiro; se sobrar tempo, rodar com 3 seeds e reportar média ± desvio

---

## 7. Ordem de implementação recomendada

Implementar nessa ordem garante que cada etapa valida a anterior:

1. **`gpu_setup_guide.md` primeiro** → configurar WSL2 + TF GPU
2. `requirements.txt` e `.gitignore`
3. `src/config.py` → constantes
4. `notebooks/00_setup_and_verification.ipynb` → **se isso não rodar, parar tudo**
5. `src/data_loader.py` (parte de carregamento e splits)
6. `notebooks/01_data_exploration.ipynb` → entender o dataset
7. `src/data_loader.py` (resto: normalização, augmentation, pipeline)
8. `notebooks/02_data_pipeline.ipynb` → validar pipeline antes de treinar
9. `src/models.py`, `src/training.py`, `src/evaluation.py`, `src/visualization.py`
10. `notebooks/03_train_model_a_rgb.ipynb` → treinar primeiro modelo (o mais simples)
11. **Validar que tudo funciona end-to-end antes de seguir** — se o modelo A der < 80% algo está errado
12. `notebooks/04_train_model_b_rgb_nir.ipynb`
13. `notebooks/05_train_model_c_multispectral.ipynb`
14. `notebooks/06_presentation.ipynb` → consolidar resultados
15. `README.md` → instruções finais de como rodar
16. **Revisar todos os notebooks** com "Restart & Run All" pra garantir reprodutibilidade

---

## 8. Critérios de sucesso

O projeto está concluído quando:

- [ ] GPU é detectada e usada no treino (confirmar via `nvidia-smi` durante treino)
- [ ] Os 3 modelos treinam sem erro até o fim
- [ ] Acurácia no test set > 85% nos 3 modelos
- [ ] Tendência esperada: Modelo C > Modelo B > Modelo A (mesmo que pouco)
- [ ] Todos os resultados salvos em `results/`
- [ ] `06_presentation.ipynb` roda end-to-end **sem retreinar nada**
- [ ] Gráficos legíveis e prontos pra slides
- [ ] README explica como rodar do zero

---

## 9. Riscos e mitigações

| Risco | Mitigação |
|-------|-----------|
| GPU não detectada no WSL | Seguir `gpu_setup_guide.md` passo a passo; validar no notebook 00 antes de seguir |
| Dataset multi-banda grande demais pra RAM | Usar `.cache()` em disco em vez de memória se necessário |
| Modelos não convergem | Reduzir LR; aumentar batch size; verificar normalização |
| Acurácias muito próximas entre os 3 modelos | Reportar mesmo assim — é um resultado válido (talvez RGB já carregue 90% da informação) |
| WSL com I/O lento no `/mnt/c` | Manter tfds cache em `~/tensorflow_datasets` (filesystem Linux) |

---

## 10. Próximos passos após o plano aprovado

Quando você aprovar este plano, a ordem prática será:

1. Configurar WSL/GPU (`gpu_setup_guide.md`)
2. Eu crio a estrutura de pastas e os arquivos esqueleto (`src/`, `requirements.txt`, etc.)
3. Eu implemento os módulos em `src/`
4. Eu crio os notebooks na ordem 00 → 06
5. Você executa cada notebook e me avisa de qualquer problema
6. Refinamos a apresentação juntos
