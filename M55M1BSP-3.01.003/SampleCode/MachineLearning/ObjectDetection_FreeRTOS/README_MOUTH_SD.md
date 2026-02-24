# Mouth Detection with SD Card Model Loading

This project supports loading a **YOLO-Fastest v1.1 mouth detection model** (192×192, 2 classes: mouth closed, mouth open) from an SD card instead of using the embedded 80-class COCO model.

## Requirements

- **Model file**: `yolo-fastest-1.1-int8_vela.tflite` (Vela-optimized TFLite int8 model)
- **SD card**: FAT32 formatted, with the model file at root: `0:\yolo-fastest-1.1-int8_vela.tflite`
- **NuMaker-M55M1** board with SD card slot

## Build for Mouth Detection (SD Loading)

1. Open the Keil project: `KEIL/ObjectDetection_FreeRTOS.uvprojx`
2. Ensure **Options for Target** → **C/C++** → **Preprocessor Symbols** includes:
   ```
   __LOAD_MODEL_FROM_SD__, ACTIVATION_BUF_SZ=0x300000
   ```
3. Build the project (F7)

## Flash and Run

1. **Prepare SD card**: Copy `yolo-fastest-1.1-int8_vela.tflite` to the root of a FAT32 SD card
2. **Flash firmware**: Use Keil to download the built `.axf` to the M55M1 (NuEdgeWise) board
3. **Insert SD card** into the board's SD slot
4. **Power on** – the firmware loads the model from SD to HyperRAM and runs mouth detection

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

## Model Specification (192×192 Mouth Detection)

- **Input**: 192×192 RGB, 3 channels
- **Classes**: mouth closed (0), mouth open (1)
- **Architecture**: YOLO-Fastest v1.1, Ethos-U55-256
- **Deployment**: Copy `yolo-fastest-1.1-int8_vela.tflite` from Colab `workspace/ColabTrain/vela/` to SD root

## Anchors

The mouth detection config uses anchors from k-means on the mouth dataset (22,13 20,24 27,19 | 27,33 75,110 112,113). If your model was trained with different anchors, update `Model/MouthModelConfig.cpp`.
