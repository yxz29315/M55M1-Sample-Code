#include "YOLOv5FacePostProcessing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace arm {
namespace app {

struct YOLOv5Det {
    float cx, cy, w, h, conf;
    int x, y, box_w, box_h;
};

static float calcIoU(const YOLOv5Det &a, const YOLOv5Det &b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.box_w, b.x + b.box_w);
    int y2 = std::min(a.y + a.box_h, b.y + b.box_h);
    if (x2 <= x1 || y2 <= y1) return 0.f;
    float inter = (x2 - x1) * (y2 - y1);
    float areaA = a.box_w * a.box_h;
    float areaB = b.box_w * b.box_h;
    return inter / (areaA + areaB - inter);
}

static void nms(std::vector<YOLOv5Det> &dets, float iouThresh) {
    std::sort(dets.begin(), dets.end(), [](const YOLOv5Det &a, const YOLOv5Det &b) {
        return a.conf > b.conf;
    });
    for (size_t i = 0; i < dets.size(); i++) {
        if (dets[i].conf == 0) continue;
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (dets[j].conf == 0) continue;
            if (calcIoU(dets[i], dets[j]) > iouThresh)
                dets[j].conf = 0;
        }
    }
}

YOLOv5FacePostProcessing::YOLOv5FacePostProcessing(
    arm::app::YOLOv8nODModel *model, float threshold)
    : m_model(model), m_threshold(threshold) {}

void YOLOv5FacePostProcessing::RunPostProcessing(
    uint32_t imgNetCols, uint32_t imgNetRows,
    uint32_t imgSrcCols, uint32_t imgSrcRows,
    std::vector<yolov8n_od::DetectionResult> &resultsOut)
{
    resultsOut.clear();
    TfLiteTensor *out = m_model->GetOutputTensor(0);
    if (!out || !out->data.data) return;

    const int numDet = 6300;
    const int dim = 16;
    float scale = 1.f, offset = 0.f;
    if (out->quantization.params) {
        auto *q = (TfLiteAffineQuantization *)out->quantization.params;
        if (q->scale && q->scale->size > 0) scale = q->scale->data[0];
        if (q->zero_point && q->zero_point->size > 0) offset = (float)q->zero_point->data[0];
    }

    std::vector<YOLOv5Det> dets;
    float *data = out->type == kTfLiteFloat32
        ? (float *)out->data.data
        : nullptr;

    int8_t *data8 = out->type == kTfLiteInt8
        ? (int8_t *)out->data.data
        : nullptr;

    int16_t *data16 = out->type == kTfLiteInt16
        ? (int16_t *)out->data.data
        : nullptr;

    for (int i = 0; i < numDet; i++) {
        float conf;
        float cx, cy, w, h;
        if (data) {
            cx = data[i * dim + 0];
            cy = data[i * dim + 1];
            w  = data[i * dim + 2];
            h  = data[i * dim + 3];
            conf = data[i * dim + 4];
        } else if (data8) {
            cx = scale * (data8[i * dim + 0] - offset);
            cy = scale * (data8[i * dim + 1] - offset);
            w  = scale * (data8[i * dim + 2] - offset);
            h  = scale * (data8[i * dim + 3] - offset);
            conf = scale * (data8[i * dim + 4] - offset);
        } else if (data16) {
            cx = scale * (data16[i * dim + 0] - offset);
            cy = scale * (data16[i * dim + 1] - offset);
            w  = scale * (data16[i * dim + 2] - offset);
            h  = scale * (data16[i * dim + 3] - offset);
            conf = scale * (data16[i * dim + 4] - offset);
        } else continue;

        if (conf < m_threshold) continue;

        float fXScale = (float)imgSrcCols / (float)imgNetCols;
        float fYScale = (float)imgSrcRows / (float)imgNetRows;

        YOLOv5Det d;
        d.conf = conf;
        d.cx = cx; d.cy = cy; d.w = w; d.h = h;
        int box_w = (int)(w * imgNetCols * fXScale);
        int box_h = (int)(h * imgNetRows * fYScale);
        int cx_px = (int)(cx * imgNetCols * fXScale);
        int cy_px = (int)(cy * imgNetRows * fYScale);
        d.x = std::max(0, cx_px - box_w / 2);
        d.y = std::max(0, cy_px - box_h / 2);
        d.box_w = std::min(box_w, (int)imgSrcCols - d.x);
        d.box_h = std::min(box_h, (int)imgSrcRows - d.y);
        if (d.box_w <= 0 || d.box_h <= 0) continue;
        dets.push_back(d);
    }

    nms(dets, 0.45f);

    for (const auto &d : dets) {
        if (d.conf == 0) continue;
        struct S_DETECTION_BOX box;
        box.x = d.x;
        box.y = d.y;
        box.w = d.box_w;
        box.h = d.box_h;
        box.cls = 0;
        box.normalisedVal = d.conf;
        resultsOut.push_back(yolov8n_od::DetectionResult(box));
    }
}

} /* namespace app */
} /* namespace arm */
