/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    Mouth detection sample (YOLOv8n ReLU6). Detects mouth open/closed.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "BoardInit.hpp"      /* Board initialisation */
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "MouthDetectionModel.hpp"
#include "MouthYOLOv8PostProcessing.hpp"
#include "FaceDetectionModel.hpp"
#include "FaceDetectorPostProcessing.hpp"
#include "FaceDetectionResult.hpp"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"
#include "ModelFileReader.h"
#include "ff.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

//#define __PROFILE__
#define __USE_DISPLAY__
//#define __USE_UVC__

#include "Profiler.hpp"

#include "ImageSensor.h"

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

#define NUM_FRAMEBUF 2  //1 or 2

/* Same as working project: 0x82400000 (exercise model works at this addr) */
#define MODEL_AT_HYPERRAM_ADDR (0x82400000)

#define MOUTH_DETECTION_THRESHOLD  				(0.05f)
#define MOUTH_NMS_THRESHOLD  					(0.45f)
#define FACE_PRESENCE_THRESHOLD  				(0.4f)

/* Temporal smoothing: require N consecutive agreeing frames before switching state */
#define SPEAKING_HYSTERESIS_ON   3   /* frames of "mouth open" to switch to Speaking */
#define SPEAKING_HYSTERESIS_OFF  4   /* frames of "mouth closed" to switch to Not Speaking */

typedef enum
{
    eFRAMEBUF_EMPTY,
    eFRAMEBUF_FULL,
    eFRAMEBUF_INF
} E_FRAMEBUF_STATE;

typedef struct
{
    E_FRAMEBUF_STATE eState;
    image_t frameImage;
    std::vector<arm::app::face_detection::DetectionResult> results_FD;  /* face boxes */
    std::vector<arm::app::face_detection::DetectionResult> results;       /* mouth detections */
} S_FRAMEBUF;


S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

namespace arm
{
namespace app
{
#define FACE_DETECTION_ACTIVATION_BUF_SZ (460000)
#define MOUTH_ACTIVATION_BUF_SZ          (512 * 1024)
static uint8_t tensorArena_FaceDetection[FACE_DETECTION_ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
static uint8_t tensorArena_Mouth[MOUTH_ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
} /* namespace app */
} /* namespace arm */

//frame buffer managemnet function
static S_FRAMEBUF *get_empty_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_EMPTY)
            return &s_asFramebuf[i];
    }

    return NULL;
}

static S_FRAMEBUF *get_full_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_FULL)
            return &s_asFramebuf[i];
    }

    return NULL;
}

static S_FRAMEBUF *get_inf_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_INF)
            return &s_asFramebuf[i];
    }

    return NULL;
}

#define IMAGE_DISP_UPSCALE_FACTOR 2
#if defined(LT7381_LCD_PANEL)
#define FONT_DISP_UPSCALE_FACTOR 2
#else
#define FONT_DISP_UPSCALE_FACTOR 1
#endif

/* Image processing initiate function */
//Used by omv library
#if defined(__USE_UVC__)
//UVC only support QVGA, QQVGA
#define GLCD_WIDTH	320
#define GLCD_HEIGHT	240
#else
#define GLCD_WIDTH	320 //256
#define GLCD_HEIGHT	240 //256
#endif

//RGB565
#define IMAGE_FB_SIZE	(GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE (IMAGE_FB_SIZE + 1024)

#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE	(1*1024)

__attribute__((section(".bss.vram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF == 2)
    __attribute__((section(".bss.vram.data"), aligned(32))) static char frame_buf1[OMV_FB_SIZE];
#endif

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

static void omv_init()
{
    image_t frameBuffer;
    int i;

    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end =  fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();

    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);

    for (i = 0 ; i < NUM_FRAMEBUF; i++)
    {
        s_asFramebuf[i].eState = eFRAMEBUF_EMPTY;
    }

    framebuffer_init_image(&s_asFramebuf[0].frameImage);

#if (NUM_FRAMEBUF == 2)
    s_asFramebuf[1].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[1].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[1].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    s_asFramebuf[1].frameImage.pixfmt = PIXFORMAT_RGB565;
    s_asFramebuf[1].frameImage.data = (uint8_t *)frame_buf1;
#endif
}

static bool g_isSpeaking = false;
static int  g_speakingCounter = 0;

static bool SmoothSpeakingState(bool rawMouthOpen)
{
    if (rawMouthOpen) {
        if (!g_isSpeaking) {
            g_speakingCounter++;
            if (g_speakingCounter >= SPEAKING_HYSTERESIS_ON) {
                g_isSpeaking = true;
                g_speakingCounter = 0;
            }
        } else {
            g_speakingCounter = 0;
        }
    } else {
        if (g_isSpeaking) {
            g_speakingCounter++;
            if (g_speakingCounter >= SPEAKING_HYSTERESIS_OFF) {
                g_isSpeaking = false;
                g_speakingCounter = 0;
            }
        } else {
            g_speakingCounter = 0;
        }
    }
    return g_isSpeaking;
}

static void DrawFaceWithState(
    const std::vector<arm::app::face_detection::DetectionResult> &faceResults,
    bool isSpeaking,
    image_t *drawImg
)
{
    int greenBox = COLOR_R5_G6_B5_TO_RGB565(0, COLOR_G6_MAX, 0);
    int redBox   = COLOR_R5_G6_B5_TO_RGB565(COLOR_R5_MAX, 0, 0);
    int boxColor = isSpeaking ? greenBox : redBox;
    const char *label = isSpeaking ? "Speaking" : "Not Speaking";
    int labelColor = isSpeaking ? greenBox : redBox;

    for (size_t i = 0; i < faceResults.size(); i++)
    {
        const auto &r = faceResults[i];
        imlib_draw_rectangle(drawImg, r.m_x0, r.m_y0, r.m_w, r.m_h, boxColor, 2, false);

        int labelY = (r.m_y0 - 14 > 0) ? (r.m_y0 - 14) : r.m_y0;
        imlib_draw_string(drawImg, r.m_x0, labelY, label, labelColor, 2, 0, 0, false,
                          false, false, false, 0, false, false);
    }
}

static int32_t PrepareModelToHyperRAM(void)
{
#define MODEL_FILE "0:\\best_full_integer_quant_vela.tflite"
#define EACH_READ_SIZE 512
	
    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */	
    f_chdrive(sd_path);          /* set default path */

	int32_t i32FileSize;
	int32_t i32FileReadIndex = 0;
	int32_t i32Read;
	
	if(!ModelFileReader_Initialize(MODEL_FILE))
	{
        printf_err("Unable open model %s\n", MODEL_FILE);		
		return -1;
	}
	
	i32FileSize = ModelFileReader_FileSize();
    info("Model file size %i \n", i32FileSize);

	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		i32FileReadIndex += i32Read;
	}
	
	if(i32FileReadIndex < i32FileSize)
	{
        printf_err("Read Model file size is not enough\n");		
		return -2;
	}
	
#if 0
	/* verify */
	i32FileReadIndex = 0;
	ModelFileReader_Rewind();
	BYTE au8TempBuf[EACH_READ_SIZE];
	
	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)au8TempBuf, EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		
		if(std::memcmp(au8TempBuf, (void *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), i32Read)!= 0)
		{
			printf_err("verify the model file content is incorrect at %i \n", i32FileReadIndex);		
			return -3;
		}
		i32FileReadIndex += i32Read;
	}
	
#endif	
	ModelFileReader_Finish();
	
	return i32FileSize;
}	


int main()
{
    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();
    info("main: BoardInit done, loading model...\n");

	/* Copy model file from SD to HyperRAM*/
	int32_t i32ModelSize;
		
	i32ModelSize = PrepareModelToHyperRAM();

	if(i32ModelSize <= 0 )
	{
        printf_err("Failed to prepare model\n");
        return 1;
	}

    /* Ensure SD writes to HyperRAM are visible before CPU reads model */
    __DSB();
    __DMB();

    /* Sanity check: verify TFLite magic at model start (TFL3 at offset 4) */
    const uint8_t *pModel = (const uint8_t *)MODEL_AT_HYPERRAM_ADDR;
    if (i32ModelSize < 12 || pModel[4] != 0x54 || pModel[5] != 0x46 || pModel[6] != 0x4c || pModel[7] != 0x33)
    {
        printf_err("Invalid TFLite model at 0x%08x: bad magic or size (len=%d)\n",
                   (unsigned)MODEL_AT_HYPERRAM_ADDR, (int)i32ModelSize);
        return 1;
    }
    info("Model magic TFL3 OK, readable from HyperRAM\n");

    /* Face detection model (embedded) */
    arm::app::FaceDetectionModel faceDetectionModel;
    if (!faceDetectionModel.Init(arm::app::tensorArena_FaceDetection,
                               sizeof(arm::app::tensorArena_FaceDetection),
                               (unsigned char *)arm::app::face_detection::GetModelPointer(),
                               arm::app::face_detection::GetModelLen()))
    {
        printf_err("Failed to initialise face detection model\n");
        return 1;
    }
    info("Face detection model init OK\n");

    /* Mouth model (from SD card) */
    arm::app::MouthDetectionModel model;
    if (!model.Init(arm::app::tensorArena_Mouth,
                    sizeof(arm::app::tensorArena_Mouth),
                    (unsigned char *)MODEL_AT_HYPERRAM_ADDR,
                    i32ModelSize))
    {
        printf_err("Failed to initialise mouth model\n");
        return 1;
    }
    info("Mouth model init OK\n");

    /* Setup MPU for tensor arenas */
    info("Set tensor arena cache policy to WTRA\n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // SRAM for face detection tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena_FaceDetection),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena_FaceDetection) + FACE_DETECTION_ACTIVATION_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA)
        },
        {
            // SRAM for mouth tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena_Mouth),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena_Mouth) + MOUTH_ACTIVATION_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA)
        },
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)fb_array),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)fb_array) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
#if (NUM_FRAMEBUF == 2)
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)frame_buf1),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)frame_buf1) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
#endif
    };

    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    TfLiteTensor *inputTensor   = model.GetInputTensor(0);

    if (!inputTensor->dims)
    {
        printf_err("Invalid input tensor dims\n");
        return 2;
    }
    else if (inputTensor->dims->size < 3)
    {
        printf_err("Input tensor dimension should be >= 3\n");
        return 3;
    }

    TfLiteIntArray *inputShape = model.GetInputShape(0);

    const int inputImgCols = inputShape->data[arm::app::MouthDetectionModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::MouthDetectionModel::ms_inputRowsIdx];
    const uint32_t nChannels = inputShape->data[arm::app::MouthDetectionModel::ms_inputChannelsIdx];

    if (inputImgRows != inputImgCols)
    {
        printf_err("Mouth model input must be square (got %dx%d)\n", inputImgRows, inputImgCols);
        return 4;
    }

    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);

    /* Face detection: input 192x192 grayscale */
    TfLiteIntArray *inputShape_FD = faceDetectionModel.GetInputShape(0);
    const int inputImgCols_FD = inputShape_FD->data[arm::app::FaceDetectionModel::ms_inputColsIdx];
    const int inputImgRows_FD = inputShape_FD->data[arm::app::FaceDetectionModel::ms_inputRowsIdx];
    TfLiteTensor *outputTensor0_FD = faceDetectionModel.GetOutputTensor(0);
    TfLiteTensor *outputTensor1_FD = faceDetectionModel.GetOutputTensor(1);
    const arm::app::face_detection::PostProcessParams postProcessParams_FD{
        inputImgRows_FD,
        inputImgCols_FD,
        GLCD_HEIGHT,
        GLCD_WIDTH,
        anchor1,
        anchor2,
        FACE_PRESENCE_THRESHOLD,
        0.45f,
        1,  /* numClasses */
        0   /* topN */
    };
    arm::app::FaceDetectorPostProcess postProcess_FD(outputTensor0_FD, outputTensor1_FD,
        s_asFramebuf[0].results_FD, postProcessParams_FD);

    /* Mouth PP anchor grids must match trained input size (e.g. 128 vs 192) */
    arm::app::mouth_detection::MouthYOLOv8PostProcessing postProcess(&model,
        MOUTH_DETECTION_THRESHOLD,
        MOUTH_NMS_THRESHOLD,
        inputImgRows);
	
    //display framebuffer
    image_t frameBuffer;
    rectangle_t roi;

    //omv library init
    omv_init();
    framebuffer_init_image(&frameBuffer);

#if defined(__PROFILE__)

    arm::app::Profiler profiler;
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;
    uint64_t u64CCAPStartCycle;
    uint64_t u64CCAPEndCycle;
#else
    pmu_reset_counters();
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);

    S_FRAMEBUF *infFramebuf;
    S_FRAMEBUF *fullFramebuf;
    S_FRAMEBUF *emptyFramebuf;

    //Setup image senosr
    ImageSensor_Init();
    ImageSensor_Config(eIMAGE_FMT_RGB565, frameBuffer.w, frameBuffer.h, true);

#if defined (__USE_DISPLAY__)
    char szDisplayText[100];
    S_DISP_RECT sDispRect;

    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

#if defined (__USE_UVC__)
	UVC_Init();
    HSUSBD_Start();
#endif

    while (1)
    {
        emptyFramebuf = get_empty_framebuf();

        if (emptyFramebuf)
        {
            //capture frame from CCAP
#if defined(__PROFILE__)
            u64CCAPStartCycle = pmu_get_systick_Count();
#endif

            ImageSensor_TriggerCapture((uint32_t)(emptyFramebuf->frameImage.data));
		}
		
        fullFramebuf = get_full_framebuf();

        if (fullFramebuf)
        {
            fullFramebuf->results_FD.clear();
            fullFramebuf->results.clear();

            /* --- Step 1: Face detection on full frame (192x192 grayscale) --- */
            TfLiteTensor *inputTensor_FD = faceDetectionModel.GetInputTensor(0);
            roi.x = 0;
            roi.y = 0;
            roi.w = fullFramebuf->frameImage.w;
            roi.h = fullFramebuf->frameImage.h;
            image_t resizeImg_FD;
            resizeImg_FD.w = inputImgCols_FD;
            resizeImg_FD.h = inputImgRows_FD;
            resizeImg_FD.data = (uint8_t *)inputTensor_FD->data.data;
            resizeImg_FD.pixfmt = PIXFORMAT_GRAYSCALE;
            imlib_nvt_scale(&fullFramebuf->frameImage, &resizeImg_FD, &roi);

            auto *req_FD = static_cast<uint8_t *>(inputTensor_FD->data.data);
            auto *signed_FD = static_cast<int8_t *>(inputTensor_FD->data.data);
            for (size_t i = 0; i < inputTensor_FD->bytes; i++)
            {
                int32_t v = static_cast<int32_t>(req_FD[i]) - 128;
                signed_FD[i] = static_cast<int8_t>(v);
            }
            faceDetectionModel.RunInference();
            postProcess_FD.RunPostProcess(fullFramebuf->results_FD);

            /* Expand face boxes 1.4x (like FaceLandmark) */
            float scaleFactorH = 1.4f;
            for (size_t i = 0; i < fullFramebuf->results_FD.size(); i++)
            {
                auto *fb = &fullFramebuf->results_FD[i];
                float scaleW = scaleFactorH * fb->m_h;
                float scaleH = scaleFactorH * fb->m_h;
                int newX = fb->m_x0 - (int)((scaleW - fb->m_w) / 2);
                int newY = fb->m_y0 - (int)((scaleH - fb->m_h) / 2);
                if (newX < 0) newX = 0;
                if (newY < 0) newY = 0;
                int newW = (int)scaleW;
                int newH = (int)scaleH;
                if (newX + newW >= (int)fullFramebuf->frameImage.w)
                    newW = fullFramebuf->frameImage.w - newX;
                if (newY + newH >= (int)fullFramebuf->frameImage.h)
                    newH = fullFramebuf->frameImage.h - newY;
                fb->m_x0 = newX;
                fb->m_y0 = newY;
                fb->m_w = newW;
                fb->m_h = newH;
            }

            /* --- Step 2: For each face, crop and run mouth model --- */
            static std::vector<arm::app::face_detection::DetectionResult> s_mouthTemp;
            for (size_t f = 0; f < fullFramebuf->results_FD.size(); f++)
            {
                const auto &faceBox = fullFramebuf->results_FD[f];
                roi.x = faceBox.m_x0;
                roi.y = faceBox.m_y0;
                roi.w = faceBox.m_w;
                roi.h = faceBox.m_h;

                image_t resizeImg;
                resizeImg.w = inputImgCols;
                resizeImg.h = inputImgRows;
                resizeImg.data = (uint8_t *)inputTensor->data.data;
                resizeImg.pixfmt = PIXFORMAT_RGB888;
                imlib_nvt_scale(&fullFramebuf->frameImage, &resizeImg, &roi);

                auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
                auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);
                for (size_t i = 0; i < inputTensor->bytes; i++)
                {
                    int32_t v = static_cast<int32_t>(req_data[i]) - 128;
                    signed_req_data[i] = static_cast<int8_t>(v);
                }
                model.RunInference();

                s_mouthTemp.clear();
                postProcess.RunPostProcessing(inputImgRows, inputImgCols, (uint32_t)roi.h, (uint32_t)roi.w, s_mouthTemp);

                /* Offset mouth results from face-crop to full-frame coords */
                for (size_t m = 0; m < s_mouthTemp.size(); m++)
                {
                    auto r = s_mouthTemp[m];
                    r.m_x0 += faceBox.m_x0;
                    r.m_y0 += faceBox.m_y0;
                    fullFramebuf->results.push_back(r);
                }
            }

            fullFramebuf->eState = eFRAMEBUF_INF;
        }
        infFramebuf = get_inf_framebuf();

        if (infFramebuf)
        {
            /* Determine raw mouth-open state: any detection with classId==1 means mouth open */
            bool rawMouthOpen = false;
            {
                size_t mi;
                for (mi = 0; mi < infFramebuf->results.size(); mi++) {
                    if (infFramebuf->results[mi].m_classId == 1) {
                        rawMouthOpen = true;
                        break;
                    }
                }
            }

            /* If no face detected this frame, keep previous smoothed state (don't reset) */
            bool speaking = g_isSpeaking;
            if (infFramebuf->results_FD.size() > 0)
                speaking = SmoothSpeakingState(rawMouthOpen);

#if defined(__PROFILE__)
			u64StartCycle = pmu_get_systick_Count();
#endif
            DrawFaceWithState(infFramebuf->results_FD, speaking, &infFramebuf->frameImage);
#if defined(__PROFILE__)
			u64EndCycle = pmu_get_systick_Count();
			info("draw cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

            //display result image
#if defined (__USE_DISPLAY__)
            //Display image on LCD
            sDispRect.u32TopLeftX = 0;
            sDispRect.u32TopLeftY = 0;
			sDispRect.u32BottonRightX = ((frameBuffer.w * IMAGE_DISP_UPSCALE_FACTOR) - 1);
			sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) - 1);

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif

            Display_FillRect((uint16_t *)infFramebuf->frameImage.data, &sDispRect, IMAGE_DISP_UPSCALE_FACTOR);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("display image cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#endif

#if defined (__USE_UVC__)
			if(UVC_IsConnect())
			{
#if (UVC_Color_Format == UVC_Format_YUY2)
				image_t RGB565Img;
				image_t YUV422Img;

				RGB565Img.w = infFramebuf->frameImage.w;
				RGB565Img.h = infFramebuf->frameImage.h;
				RGB565Img.data = (uint8_t *)infFramebuf->frameImage.data;
				RGB565Img.pixfmt = PIXFORMAT_RGB565;

				YUV422Img.w = RGB565Img.w;
				YUV422Img.h = RGB565Img.h;
				YUV422Img.data = (uint8_t *)infFramebuf->frameImage.data;
				YUV422Img.pixfmt = PIXFORMAT_YUV422;
				
				roi.x = 0;
				roi.y = 0;
				roi.w = RGB565Img.w;
				roi.h = RGB565Img.h;
				imlib_nvt_scale(&RGB565Img, &YUV422Img, &roi);
				
#else
				image_t origImg;
				image_t vflipImg;

				origImg.w = infFramebuf->frameImage.w;
				origImg.h = infFramebuf->frameImage.h;
				origImg.data = (uint8_t *)infFramebuf->frameImage.data;
				origImg.pixfmt = PIXFORMAT_RGB565;

				vflipImg.w = origImg.w;
				vflipImg.h = origImg.h;
				vflipImg.data = (uint8_t *)infFramebuf->frameImage.data;
				vflipImg.pixfmt = PIXFORMAT_RGB565;

				imlib_nvt_vflip(&origImg, &vflipImg);
#endif
				UVC_SendImage((uint32_t)infFramebuf->frameImage.data, IMAGE_FB_SIZE, uvcStatus.StillImage);				

			}

#endif

            u64PerfFrames ++;
			if ((uint64_t) pmu_get_systick_Count() > u64PerfCycle)
            {
                info("Total inference rate: %llu\n", u64PerfFrames / EACH_PERF_SEC);
#if defined (__USE_DISPLAY__)
                sprintf(szDisplayText, "Frame Rate %llu", u64PerfFrames / EACH_PERF_SEC);
                //sprintf(szDisplayText,"Time %llu",(uint64_t) pmu_get_systick_Count() / (uint64_t)SystemCoreClock);
                //info("Running %s sec \n", szDisplayText);

                sDispRect.u32TopLeftX = 0;
				sDispRect.u32TopLeftY = frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR;
				sDispRect.u32BottonRightX = (frameBuffer.w);
				sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) + (FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT) - 1);

                Display_ClearRect(C_WHITE, &sDispRect);
                Display_PutText(
                    szDisplayText,
                    strlen(szDisplayText),
                    0,
					frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR,
                    C_BLUE,
                    C_WHITE,
                    false,
					FONT_DISP_UPSCALE_FACTOR
                );
#endif
                u64PerfCycle = (uint64_t)pmu_get_systick_Count() + (uint64_t)(SystemCoreClock * EACH_PERF_SEC);
                u64PerfFrames = 0;
			}

            infFramebuf->eState = eFRAMEBUF_EMPTY;
		}

		//Wait CCAP ready
		if (emptyFramebuf)
		{
			//Capture new image

			ImageSensor_WaitCaptureDone();
#if defined(__PROFILE__)
			u64CCAPEndCycle = pmu_get_systick_Count();
			info("ccap capture cycles %llu \n", (u64CCAPEndCycle - u64CCAPStartCycle));
#endif
            emptyFramebuf->eState = eFRAMEBUF_FULL;		
		}
    }

    return 0;
}
