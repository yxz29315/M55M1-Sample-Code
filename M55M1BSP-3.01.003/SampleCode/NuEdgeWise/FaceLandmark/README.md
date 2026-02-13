# FaceLandmark
A demonstration sample for face landmark with optional **speaking detection**.
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Run
## Speaking Detection (optional)
Uses lip keypoint velocity and Mouth Aspect Ratio (MAR) to detect who is speaking:
- **Lip landmarks**: 6 key points (indices 61, 291, 78, 308, 87, 14)
- **Detection**: Speaking when lip velocity > threshold OR mouth open (MAR > threshold)
- **Visual feedback**: Bounding box turns **green** when speaking; "Speaking" label appears
- **Hysteresis**: Speaking state persists for a few frames after movement stops (reduces flicker)
- **Tuning in main.cpp**: `SPEAKING_VELOCITY_THRESHOLD` (1.0), `MAR_SPEAKING_THRESHOLD` (0.12), `SPEAKING_RELEASE_FRAMES` (5)
## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|Yolo fastest|192x192x1|441|443|131.1|
|FaceLandmark|192x192x3|679|460|38.6|

Total frame rate: 13 fps


