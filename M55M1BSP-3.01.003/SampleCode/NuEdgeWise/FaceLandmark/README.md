# FaceLandmark
A demonstration sample for face landmark with optional **speaking detection**.
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Run
## Speaking Detection (optional)
When enabled, the sample uses lip keypoint velocity to detect who is speaking:
- **Lip landmarks**: 6 key points around the mouth (MediaPipe Face Mesh indices 61, 291, 78, 308, 13, 14)
- **Detection**: When lip movement velocity exceeds threshold → "Speaking" state
- **Visual feedback**: Bounding box turns **green** when speaking; "Speaking" label appears above the face
- **Tuning**: Adjust `SPEAKING_VELOCITY_THRESHOLD` (default 2.0) and `SPEAKING_SMOOTHING_FRAMES` (default 3) in main.cpp
## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|Yolo fastest|192x192x1|441|443|131.1|
|FaceLandmark|192x192x3|679|460|38.6|

Total frame rate: 13 fps


