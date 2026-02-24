/**************************************************************************//**
 * @file     MouthModelConfig.cpp
 * @brief    Config for YOLO-Fastest v1.1 mouth detection (192x192, 2 classes)
 *           Used when loading model from SD card (__LOAD_MODEL_FROM_SD__)
 *           Anchors from k-means on mouth dataset (6 clusters).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#if defined(__LOAD_MODEL_FROM_SD__)

#include <cstddef>
#include <cstdint>

/* Model loaded from SD to HyperRAM - address and size set at load time */
#define MODEL_AT_HYPERRAM_ADDR  (0x82400000)

int32_t g_loadedModelSize = 0;

/* YOLO-Fastest v1.1 192x192 mouth detection - anchors from k-means on dataset */
/* Anchors: 22,13 20,24 27,19 | 27,33 75,110 112,113 (6 clusters, 3 per branch) */
extern const int originalImageSize = 192;
extern const int channelsImageDisplayed = 3;
extern const float anchor1[] = {22, 13, 20, 24, 27, 19};
extern const float anchor2[] = {27, 33, 75, 110, 112, 113};
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
