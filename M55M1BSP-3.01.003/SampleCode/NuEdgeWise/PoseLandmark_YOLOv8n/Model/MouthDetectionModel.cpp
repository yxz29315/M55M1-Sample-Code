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

/* YOLO-Fastest v1.1 mouth anchors - stride 32 (7x7) */
const float mouth_anchor1[] = {26.0f, 15.0f, 22.0f, 26.0f, 31.0f, 21.0f};

/* YOLO-Fastest v1.1 mouth anchors - stride 16 (14x14) */
const float mouth_anchor2[] = {29.0f, 34.0f, 35.0f, 47.0f, 111.0f, 130.0f};

const tflite::MicroOpResolver &arm::app::MouthDetectionModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::MouthDetectionModel::EnlistOperations()
{
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
