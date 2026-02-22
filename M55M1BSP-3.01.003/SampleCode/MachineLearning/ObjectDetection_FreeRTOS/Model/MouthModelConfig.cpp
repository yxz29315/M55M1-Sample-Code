/**************************************************************************//**
 * @file     MouthModelConfig.cpp
 * @brief    Config for YOLO-Fastest v1.1 mouth detection (224x224, 2 classes)
 *           Used when loading model from SD card (__LOAD_MODEL_FROM_SD__)
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#if defined(__LOAD_MODEL_FROM_SD__)

#include <cstddef>
#include <cstdint>

/* Model loaded from SD to HyperRAM - address and size set at load time */
#define MODEL_AT_HYPERRAM_ADDR  (0x82400000)

int32_t g_loadedModelSize = 0;

/* YOLO-Fastest v1.1 224x224 - anchors scaled from 320x320 (scale = 224/320 = 0.7) */
/* extern for external linkage - referenced by main.cpp and DetectorPostProcessing.cpp */
extern const int originalImageSize = 224;
extern const int channelsImageDisplayed = 3;
extern const float anchor1[] = {8, 13, 26, 34, 36, 92};
extern const float anchor2[] = {80, 51, 83, 139, 169, 167};
extern const int numClasses = 2;

namespace arm
{
namespace app
{
namespace yolofastest
{

uint8_t *GetModelPointer()
{
    return (uint8_t *)MODEL_AT_HYPERRAM_ADDR;
}

size_t GetModelLen()
{
    return (size_t)g_loadedModelSize;
}

} /* namespace yolofastest */
} /* namespace app */
} /* namespace arm */

#endif /* __LOAD_MODEL_FROM_SD__ */
