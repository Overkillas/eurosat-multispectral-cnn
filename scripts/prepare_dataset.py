"""Prepara o EuroSAT all-bands no cache do tfds.

O host oficial (madm.dfki.de) está retornando HTTP 403. Este script aponta a
URL de download do builder para o mirror do torchgeo no Hugging Face, que serve
exatamente o mesmo `EuroSATallBands.zip`, e então roda download_and_prepare.

Rodar uma única vez: `python scripts/prepare_dataset.py`
"""

import tensorflow_datasets as tfds

HF_URL = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip"


def main():
    builder = tfds.builder("eurosat/all")
    # Substitui a URL oficial (403) pelo mirror do Hugging Face.
    builder.builder_config.download_url = HF_URL
    # Desativa verificação de checksum (a URL mudou; o conteúdo é o mesmo arquivo).
    dl_config = tfds.download.DownloadConfig(verify_ssl=True, register_checksums=False)
    builder.download_and_prepare(download_config=dl_config)
    print("DATASET PRONTO")
    print({k: v.num_examples for k, v in builder.info.splits.items()})


if __name__ == "__main__":
    main()
