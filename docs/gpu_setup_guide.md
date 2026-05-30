# Guia de Setup — TensorFlow com GPU (Linux / WSL2)

> Como preparar um ambiente com TensorFlow + GPU NVIDIA para rodar este projeto.
> Em Linux, pule a parte de WSL e vá direto para a instalação do Python/venv.

---

## Por que WSL2 no Windows (e não TF nativo)?

A partir da versão **2.11**, o TensorFlow **não oferece mais suporte oficial a GPU no Windows nativo**. A forma recomendada de usar TF com GPU no Windows é via **WSL2 (Windows Subsystem for Linux)**, que roda um Linux real com acesso à GPU NVIDIA através do driver do Windows.

Em **Linux nativo**, basta ter o driver NVIDIA instalado e seguir a partir da Parte 3.

---

## Visão geral do ambiente

| Componente | Onde fica | Observação |
|-----------|-----------|------------|
| Código do projeto | onde você clonar o repositório | Em WSL, acessível em `/mnt/<letra>/...` se estiver num disco Windows |
| Cache do dataset (tfds) | `~/tensorflow_datasets` (padrão) | Mantenha no filesystem Linux nativo para I/O rápido |
| Ambiente Python | um venv com **Python 3.12** | TF ainda não suporta versões mais novas |
| Driver da GPU | no **Windows** (para WSL) ou no Linux | Em WSL, **não** instale driver NVIDIA dentro do Linux |

---

## Parte 1 — Instalar o WSL2 (somente Windows)

Em um PowerShell como administrador:

```powershell
wsl --install -d Ubuntu
```

Reinicie se solicitado, crie seu usuário Linux e atualize o WSL:

```powershell
wsl --update
wsl --version          # confirme VERSION = 2
```

> **Opcional — mover o WSL para outro disco.** Se o disco do sistema tiver pouco espaço, você pode exportar e reimportar a distro em outro drive:
> ```powershell
> wsl --shutdown
> wsl --export <DISTRO> D:\wsl-backup\ubuntu.tar
> wsl --unregister <DISTRO>
> wsl --import <DISTRO> D:\wsl\ubuntu D:\wsl-backup\ubuntu.tar --version 2
> ```
> Após reimportar, o usuário padrão pode voltar a ser `root`; defina o seu em `/etc/wsl.conf`:
> ```
> [user]
> default=SEU_USUARIO
> ```

---

## Parte 2 — Driver NVIDIA (somente Windows/WSL)

A GPU é exposta ao WSL pelo driver do **Windows** — **não instale driver NVIDIA dentro do WSL**.

1. Instale/atualize o driver NVIDIA pelo site oficial (Game Ready ou Studio).
2. Valide no PowerShell e dentro do WSL:

```bash
nvidia-smi   # deve mostrar a mesma GPU nos dois
```

Se `nvidia-smi` não for encontrado no WSL: `wsl --update`, depois `wsl --shutdown` e abra novamente.

---

## Parte 3 — Python 3.12

O TensorFlow ainda não suporta versões mais recentes do Python; instale o 3.12. No Ubuntu, via PPA deadsnakes:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
python3.12 --version   # Python 3.12.x
```

---

## Parte 4 — Ambiente virtual e TensorFlow

Na raiz do projeto:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

O `requirements.txt` já fixa `tensorflow[and-cuda]` — a flag `[and-cuda]` instala CUDA/cuDNN compatíveis automaticamente (não é preciso instalar o CUDA Toolkit à parte).

### Validar a GPU no TensorFlow

```bash
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Esperado:

```
TF: 2.x.x
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Se `GPU: []`, veja o troubleshooting no fim.

---

## Parte 5 — VSCode com WSL (opcional)

1. No VSCode (Windows), instale a extensão **WSL** (Microsoft).
2. Abra o projeto a partir do WSL: no terminal do WSL, dentro da pasta do projeto, rode `code .` — o VSCode abre conectado ao WSL (badge verde **WSL** no canto inferior esquerdo).
3. Instale as extensões **Python** e **Jupyter** no contexto WSL quando solicitado.
4. Ao abrir um `.ipynb`, em "Select Kernel" escolha o Python do seu venv (`.venv/bin/python`).

---

## Parte 6 — Validação end-to-end

```bash
python -c "
import tensorflow as tf
print('TF:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000]); b = tf.matmul(a, a)
    print('Multiplicacao na GPU OK, shape:', b.shape)
print('Tudo certo!')
"
```

Monitorar a GPU durante o treino (outro terminal): `watch -n 1 nvidia-smi`.

---

## Memória do WSL (importante)

Por padrão o WSL2 usa ~50% da RAM do host. Se o host tiver pouca memória, o treino pode estourar a RAM da VM e derrubar o kernel. Ajuste criando/editando `%USERPROFILE%\.wslconfig` no Windows:

```
[wsl2]
memory=10GB      # deixe folga para o Windows
swap=8GB
```

Aplique com `wsl --shutdown` (e reabra). Este projeto também mantém o uso de RAM baixo lendo os dados do disco em vez de cachear tudo em memória.

---

## Troubleshooting

**`tf.config.list_physical_devices('GPU')` retorna `[]`**
1. Confirme que `nvidia-smi` funciona (no WSL, no Windows).
2. Confirme que o venv está ativo e o Python é 3.12.
3. Reinstale: `pip uninstall -y tensorflow tensorflow-cpu` e `pip install "tensorflow[and-cuda]"`.

**TF detecta GPU mas roda na CPU**
- Veja os logs do TF (indicam o device escolhido). GPUs muito antigas têm suporte limitado a float16 (mixed precision).

**Erro ao baixar o dataset (HTTP 403)**
- O host oficial do EuroSAT bloqueia downloads; use `python scripts/prepare_dataset.py`, que aponta para um mirror.

**Falta de dependência ao preparar o dataset**
- O tfds para o EuroSAT all-bands exige `tifffile` e `importlib_resources` (já no `requirements.txt`).

**O kernel desconecta / disco a 100% durante o treino**
- Sintoma de falta de RAM na VM do WSL. Ajuste o `.wslconfig` (seção acima) e feche apps pesados no Windows.

---

## Referências

- TensorFlow install (pip/GPU): https://www.tensorflow.org/install/pip
- NVIDIA CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- WSL2: https://learn.microsoft.com/windows/wsl/
- Deadsnakes PPA: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
