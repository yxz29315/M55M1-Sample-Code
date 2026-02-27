/**************************************************************************//**
 * @file     MouthDetectionModel.cpp
 * @version  V1.00
 * @brief    Mouth detection model (YOLO-Fastest v1.1) source file.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "MouthDetectionModel.hpp"
#include "log_macros.h"

/* YOLO-Fastest v1.1 192x192 mouth anchors - stride 32 (6x6), k-means on dataset */
const float mouth_anchor1[] = {22.0f, 13.0f, 20.0f, 24.0f, 27.0f, 19.0f};

/* YOLO-Fastest v1.1 192x192 mouth anchors - stride 16 (12x12) */
const float mouth_anchor2[] = {27.0f, 33.0f, 75.0f, 110.0f, 112.0f, 113.0f};

const tflite::MicroOpResolver &arm::app::MouthDetectionModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::MouthDetectionModel::EnlistOperations()
{
    /* Same ops as YoloFastestModel - mouth model is YOLO-Fastest architecture */
    this->m_opResolver.AddDepthwiseConv2D();
    this->m_opResolver.AddConv2D();
    this->m_opResolver.AddAdd();
    this->m_opResolver.AddResizeNearestNeighbor();
    this->m_opResolver.AddPad();
    this->m_opResolver.AddMaxPool2D();
    this->m_opResolver.AddConcatenation();
    this->m_opResolver.AddTranspose();

#if defined(ARM_NPU)
    if (kTfLiteOk == this->m_opResolver.AddEthosU())
    {
        info("Added %s support to op resolver\n",
             tflite::GetString_ETHOSU());
    }
    else
    {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }
#endif /* ARM_NPU */
    return true;
}
