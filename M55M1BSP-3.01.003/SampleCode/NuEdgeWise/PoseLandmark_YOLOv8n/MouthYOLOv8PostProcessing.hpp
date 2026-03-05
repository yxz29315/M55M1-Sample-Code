/**************************************************************************//**
 * @file     MouthYOLOv8PostProcessing.hpp
 * @version  V1.00
 * @brief    YOLOv8n mouth detection post-processing (DFL, reg_max=16).
 *           Output order: box P3(0), box P4(1), box P5(2), cls P3(3), cls P4(4), cls P5(5).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef MOUTH_YOLOV8_POST_PROCESSING_HPP
#define MOUTH_YOLOV8_POST_PROCESSING_HPP

#include "FaceDetectionResult.hpp"
#include "MouthDetectionModel.hpp"

#include <forward_list>

#define MOUTH_INPUT_SIZE        192
#define MOUTH_NUM_CLASSES       2
#define MOUTH_REG_MAX           16
#define MOUTH_STRIDE_8          8
#define MOUTH_STRIDE_16         16
#define MOUTH_STRIDE_32         32

/* Output tensor indices - from actual model (log shows order: 0=cls P4, 1=box P4, 2=cls P5, 3=cls P3, 4=box P3, 5=box P5) */
#define MOUTH_BOX_P3_INDEX      4   /* [1, 576, 64] stride 8 */
#define MOUTH_BOX_P4_INDEX      1   /* [1, 144, 64] stride 16 */
#define MOUTH_BOX_P5_INDEX      5   /* [1, 36, 64] stride 32 */
#define MOUTH_CLS_P3_INDEX      3   /* [1, 576, 2] */
#define MOUTH_CLS_P4_INDEX      0   /* [1, 144, 2] */
#define MOUTH_CLS_P5_INDEX      2   /* [1, 36, 2] */

namespace arm
{
namespace app
{
namespace mouth_detection
{

struct Box {
    float x;
    float y;
    float w;
    float h;
};

struct MouthDetection {
    Box bbox;
    int strideIndex;
    int anchorIndex;
    int cls;
    std::vector<float> prob;
};

struct AnchorBox {
    float w;
    float h;
};

/**
 * @brief Post-processing for YOLOv8n mouth detection (2 classes, DFL reg_max=16).
 */
class MouthYOLOv8PostProcessing
{
public:
    explicit MouthYOLOv8PostProcessing(arm::app::MouthDetectionModel *model,
                                      float threshold = 0.25f,
                                      float iouThreshold = 0.45f);

    void RunPostProcessing(uint32_t imgNetRows,
                           uint32_t imgNetCols,
                           uint32_t imgSrcRows,
                           uint32_t imgSrcCols,
                           std::vector<face_detection::DetectionResult> &resultsOut);

private:
    arm::app::MouthDetectionModel *m_model;
    float m_threshold;
    float m_iouThreshold;
    int m_stride8_total_anchors;   /* 24*24 = 576 */
    int m_stride16_total_anchors;  /* 12*12 = 144 */
    int m_stride32_total_anchors;  /* 6*6 = 36 */

    std::vector<AnchorBox> m_stride8_anchors;
    std::vector<AnchorBox> m_stride16_anchors;
    std::vector<AnchorBox> m_stride32_anchors;

    void GetNetworkBoxes(std::forward_list<MouthDetection> &detections);
};

} /* namespace mouth_detection */
} /* namespace app */
} /* namespace arm */

#endif /* MOUTH_YOLOV8_POST_PROCESSING_HPP */
