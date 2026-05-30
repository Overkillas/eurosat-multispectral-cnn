# Imagens próprias para teste (fora do dataset)

Coloque aqui **suas próprias** imagens de satélite (`.png` / `.jpg`) — recortes do
Google Maps (modo satélite), por exemplo. A ideia é testar o modelo em imagens
**fora da distribuição do EuroSAT**, sem viés do dataset de treino.

Estas imagens só têm RGB, então use o **modelo A**:

```bash
python scripts/predict_file.py test_images/ model_a_rgb
```

## Dicas para um bom resultado
- Vista **de cima** (satélite), não foto inclinada nem de rua.
- Recorte aproximadamente **quadrado**, mostrando **um** tipo de uso do solo
  (floresta, plantação, quarteirão de cidade, rio, mar...).
- Sem texto/bordas/UI no recorte — só o terreno.
- O tamanho não importa (é redimensionado para 64×64 automaticamente).

## Limitação importante
O modelo foi treinado em imagens **Sentinel-2** (reflectância real, 13 bandas).
Uma foto/recorte RGB comum (0–255) está **fora do domínio** espectral de treino:
serve como teste de generalização, mas a confiança tende a ser menor e classes
parecidas se confundem. Para predição rigorosa seria preciso uma imagem Sentinel-2
de 13 bandas.
