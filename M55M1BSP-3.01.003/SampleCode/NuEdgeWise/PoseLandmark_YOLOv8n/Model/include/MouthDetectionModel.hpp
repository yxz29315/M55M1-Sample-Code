/**************************************************************************//**
 * @file     MouthDetectionModel.hpp
 * @version  V1.00
 * @brief    Mouth detection model (YOLO-Fastest v1.1) header file.
 *           Input: 192x192 RGB, int8 = uint8 - 128
 *           Output: 2 tensors (stride 32: 6x6, stride 16: 12x12)
 *           Classes: 0 = mouth closed, 1 = mouth open
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef MOUTH_DETECTION_MODEL_HPP
#define MOUTH_DETECTION_MODEL_HPP

#include "Model.hpp"

/* YOLO-Fastest v1.1 mouth anchors */
extern const float mouth_anchor1[];
extern const float mouth_anchor2[];

namespace arm
{
namespace app
{

class MouthDetectionModel : public Model
{

public:
    static constexpr uint32_t ms_inputRowsIdx     = 1;
    static constexpr uint32_t ms_inputColsIdx     = 2;
    static constexpr uint32_t ms_inputChannelsIdx = 3;

protected:
    const tflite::MicroOpResolver &GetOpResolver() override;
    bool EnlistOperations() override;

private:
    static constexpr int ms_maxOpCnt = 2;
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* MOUTH_DETECTION_MODEL_HPP */
