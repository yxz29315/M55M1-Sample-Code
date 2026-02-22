# Mouth Detection with SD Card Model Loading

This project supports loading a **YOLO-Fastest v1.1 mouth detection model** (224×224, 2 classes) from an SD card instead of using the embedded 80-class COCO model.

## Requirements

- **Model file**: `yolo-fastest-1.1-int8_vela.tflite` (your trained mouth detection model)
- **SD card**: FAT32 formatted, with the model file at root: `0:\yolo-fastest-1.1-int8_vela.tflite`
- **NuMaker-M55M1** board with SD card slot

## Build for Mouth Detection (SD Loading)

1. Open the Keil project: `KEIL/ObjectDetection_FreeRTOS.uvprojx`
2. Go to **Options for Target** → **C/C++** → **Preprocessor Symbols** → **Define**
3. Add to the existing defines:
   ```
   __LOAD_MODEL_FROM_SD__, ACTIVATION_BUF_SZ=0x200000
   ```
   (Replace `ACTIVATION_BUF_SZ=0x00130000` with `ACTIVATION_BUF_SZ=0x200000` for 2 MB tensor arena)
4. Build the project

## Default Build (Embedded 80-Class Model)

To use the original embedded YOLO-Fastest 320×320 model (no SD card):

- Do **not** define `__LOAD_MODEL_FROM_SD__`
- Keep `ACTIVATION_BUF_SZ=0x00130000`

## Model File Location

Place your mouth detection `.tflite` file on the SD card root as:

```
yolo-fastest-1.1-int8_vela.tflite
```

To use a different path, edit `MODEL_FILE` in `main.cpp` (inside the `#if defined(__LOAD_MODEL_FROM_SD__)` block).

## Anchors

The mouth detection config uses anchors scaled from the default 320×320 YOLO-Fastest v1.1 for 224×224 input. If your model was trained with different anchors, update `Model/MouthModelConfig.cpp`.
