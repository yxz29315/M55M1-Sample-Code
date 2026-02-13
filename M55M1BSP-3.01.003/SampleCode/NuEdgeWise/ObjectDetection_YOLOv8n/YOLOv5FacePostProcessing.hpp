#ifndef YOLOV5FACE_POST_PROCESSING_HPP
#define YOLOV5FACE_POST_PROCESSING_HPP

#include "DetectionResult.hpp"
#include "YOLOv8nODModel.hpp"

namespace arm {
namespace app {

class YOLOv5FacePostProcessing
{
public:
    explicit YOLOv5FacePostProcessing(arm::app::YOLOv8nODModel *model, float threshold = 0.5f);

    void RunPostProcessing(uint32_t imgNetCols, uint32_t imgNetRows,
                           uint32_t imgSrcCols, uint32_t imgSrcRows,
                           std::vector<yolov8n_od::DetectionResult> &resultsOut);

private:
    arm::app::YOLOv8nODModel *m_model;
    float m_threshold;
};

} /* namespace app */
} /* namespace arm */

#endif
