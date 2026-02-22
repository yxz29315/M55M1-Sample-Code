# Tensor Allocation Failure – Root Cause Research

## Problem
`model.Init()` fails with **"tensor allocation failed!"** when loading the YOLO-Fastest mouth model from SD card.

## Research Summary

### 1. Ethos-U NPU Memory Requirements (Critical)

**Finding**: The Ethos-U55 NPU uses **SRAM only** for its working buffer (activation tensors / IFMs/OFMs).

- ARM Ethos-U TRM: "The Ethos-U NPU uses a working buffer located in **SRAM** for Input Feature Maps (IFMs) and Output Feature Maps (OFMs)."
- M55M1 scatter file: "Ethos-U55 can access **Flash, internal SRAM 0,1** => activation buffers and the model should only be placed in those regions."
- NPU_REGIONCFG_1=0 in `ethosu_config.h` → Region 1 (scratch/activation) = SRAM (AXI0).

**Implication**: The tensor arena must reside entirely in SRAM. If it overflows into HyperRAM (SPIM0, 0x82000000+), the NPU cannot access that portion and allocation/inference will fail.

### 2. Current Memory Layout

| Region | Address | Size | Content |
|--------|---------|------|---------|
| SRAM01_ALIASE | 0x81F00000 | 1 MB | NonCacheable + activation_buf (first part) |
| SPIM0 (HyperRAM) | 0x82000000 | 32 MB | activation_buf overflow + model at 0x82400000 |

- `SRAM01_HYPERRAM` in the scatter file allows `.bss.NoInit.activation_buf_sram` to overflow from SRAM into SPIM0.
- With `ACTIVATION_BUF_SZ=0x200000` (2 MB), the arena extends well into HyperRAM.
- The NPU cannot use the HyperRAM portion of the arena.

### 3. FreeRTOS Task Stack Size

- `main_task` stack: **2 KB** (`2 * 1024`).
- `main()` stack (PoseLandmark): **40 KB** (`STACK_SIZE = 0xa000`).
- `model.Init()` and `AllocateTensors()` use C++ (e.g. `std::vector`, `std::make_unique`) and graph traversal.
- 2 KB is very small for this path and can cause stack overflow and memory corruption.

### 4. Model Placement

- Model is loaded to **0x82400000** (HyperRAM).
- NPU_REGIONCFG_0=3 → Region 0 (command stream/weights) uses external memory (AXI1).
- On M55M1, AXI1 may be mapped to SPIM0, so the model in HyperRAM can be acceptable.
- Activation buffer placement is the main constraint.

### 5. TFLite Micro Allocation Failure Points

`AllocateTensors()` can fail at:

- `StartModelAllocation` → "Failed starting model allocation"
- `FinishModelAllocation` → "Failed to allocate tail/temp memory", "missing: X bytes"
- `PrepareSubgraphs` → Op Prepare (e.g. Ethos-U) failures
- Input/output tensor allocation

The generic "tensor allocation failed!" hides the exact TFLite Micro error. Enabling `MicroPrintf` (or equivalent) would show the real message.

## Recommended Fixes

### Fix 1: Restrict Activation Buffer to SRAM (Primary)

Use an activation buffer size that fits entirely in SRAM:

- SRAM01_ALIASE: 1 MB.
- Reserve space for NonCacheable (e.g. frame buffers) and `.bss.sram.data`.
- Use **512 KB (0x80000)** for the tensor arena as a safe default.

```c
// In project defines, change:
ACTIVATION_BUF_SZ=0x80000   // 512 KB - fits in SRAM
```

If 512 KB is insufficient, measure NonCacheable + sram.data and set the arena to the remaining SRAM.

### Fix 2: Increase main_task Stack Size

Reduce risk of stack overflow during model init:

```c
// In main.cpp, change:
ret = xTaskCreate(main_task, "main task", 8 * 1024, ...);  // 8 KB instead of 2 KB
```

### Fix 3: Enable MPU Region for SRAM + HyperRAM (If Needed)

If the default MPU does not cover 0x81F00000–0x83FFFFFF, enable and configure the template region:

- In `mpu_config_M55M1.h`: set `MPU_INIT_REGION1 = 1`.
- Use base `0x81F00000`, size `0x02100000` (33 MB), as in the template.

### Fix 4: Add Diagnostic Logging

To see the exact TFLite Micro error:

1. Ensure `TF_LITE_STRIP_ERROR_STRINGS` is **not** defined for debug builds.
2. Ensure `MicroPrintf` is wired to your debug UART.
3. Rebuild and run; the allocator’s `MicroPrintf` messages will show the real failure.

## Verification

1. Build with `ACTIVATION_BUF_SZ=0x80000` and increased `main_task` stack.
2. Confirm the arena base address is in SRAM (e.g. 0x81F0xxxx).
3. Run and check that `model.Init()` succeeds.
4. If it still fails, enable MicroPrintf and capture the detailed error.

## References

- ARM Ethos-U55 NPU Technical Reference Manual
- `M55M1BSP-3.01.003/Library/StdDriver/src/npu/ethosu_config.h`
- `M55M1BSP-3.01.003/SampleCode/MachineLearning/ObjectDetection_FreeRTOS/KEIL/M55M1.scf`
- TFLite Micro: `micro_allocator.cc`, `single_arena_buffer_allocator.cc`
