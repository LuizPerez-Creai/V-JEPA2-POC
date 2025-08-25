# V-JEPA2 Webcam ES

Clasificación de acciones en **tiempo real** con webcam usando `facebook/vjepa2-vitl-fpc16-256-ssv2` ( Transformers). Etiquetas **traducidas al español**.

## Requisitos

* Python 3.9–3.11
* `torch`, `transformers`, `opencv-python`, `pillow`

```bash
pip install torch torchvision torchaudio
pip install transformers opencv-python pillow
```

## Uso

1. (Opcional) Coloca `labels_bilingual.csv` (columnas `en,es`) junto al script.
2. Ejecuta:

   ```bash
   python vjepa2_webcam_es.py
   ```
3. En consola verás:

   * `num_labels: 174`
   * `logits shape: (1, 174)`

## Controles

* **ESC**: salir
* **L**: panel de etiquetas
* **P**: pausar
* **S**: guardar clip (`captures/`)
* **+ / -**: ajustar umbral

## Archivos

* `vjepa2_webcam_es.py` — script principal
* `labels_bilingual.csv` — traducciones EN→ES (opcional)
* `model_labels_en.txt` / `model_labels_es.txt` — verificación de etiquetas
* `captures/` — frames guardados

## Notas

* `TOPK`, `CONF_THRESH`, `FRAME_SKIP` se ajustan en el script.
* Si ves `num_labels` distinto de 174, estás usando un checkpoint reducido.

Info:
https://ai.meta.com/vjepa/
https://huggingface.co/docs/transformers/main/model_doc/vjepa2#usage-example
https://huggingface.co/datasets/HuggingFaceM4/something_something_v2
