/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    Pose landmark network sample. Demonstrate pose landmark detect.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "BoardInit.hpp"      /* Board initialisation */
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "YOLOv8nPoseModel.hpp"       /* Model API */
#include "YOLOv8nPosePostProcessing.hpp"

#include "ExerciseClassifierModel.hpp"

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

#define YOLO_MODEL_AT_HYPERRAM_ADDR     (0x82400000)
#define YOLO_MODEL_FILE                "0:\\YOLOv8n-pose.tflite"

#define EXERCISE_MODEL_FILE            "0:\\EXERCISE_int8_vela.tflite"

#define POSE_PRESENCE_THRESHOLD  				(0.7)
#define POSE_KEYPOINT_VISIBLE_THRESHOLD  		(0.5)

// Text Layout Configuration (Right Sidebar)
#define TEXT_X_OFFSET  (320 * IMAGE_DISP_UPSCALE_FACTOR + 10) 
#define TEXT_Y_START   20  // The vertical starting position for the first line
#define LINE_HEIGHT    (FONT_HTIGHT * FONT_DISP_UPSCALE_FACTOR) // Height of one line of text

// Calculate subsequent Y positions automatically
#define TEXT_Y_PRED    (TEXT_Y_START)
#define TEXT_Y_REPS    (TEXT_Y_START + LINE_HEIGHT)
#define TEXT_Y_FPS     (TEXT_Y_START + (2 * LINE_HEIGHT))
#define TEXT_Y_AX    (TEXT_Y_START + (3 * LINE_HEIGHT))
#define TEXT_Y_AY    (TEXT_Y_START + (4 * LINE_HEIGHT))
#define TEXT_Y_AZ    (TEXT_Y_START + (5 * LINE_HEIGHT))
#define TEXT_Y_HR    (TEXT_Y_START + (6 * LINE_HEIGHT))

//int timepassed = 0;

// UART1 Bluetooth Buffers
#define RX_BUF_SIZE 128
extern "C" {
    volatile uint8_t g_u8RecData[RX_BUF_SIZE];
    volatile uint32_t g_u32DataIdx = 0;
    volatile bool g_bMsgReceived = false;

    // The IRQ Handler must be extern "C"
    void UART1_IRQHandler(void) {
        uint32_t u32IntSts = UART1->INTSTS;
        if (u32IntSts & UART_INTSTS_RDAINT_Msk) {
            while (!UART_GET_RX_EMPTY(UART1)) {
                uint8_t u8Char = UART_READ(UART1);
                if (g_u32DataIdx < (RX_BUF_SIZE - 1)) {
                    g_u8RecData[g_u32DataIdx++] = u8Char;
                    if (u8Char == '\n' || u8Char == '\r') {
                        g_u8RecData[g_u32DataIdx] = '\0';
                        g_bMsgReceived = true;
                    }
                } else {
                    g_u32DataIdx = 0; // Overflow protection
                }
            }
        }
        if (u32IntSts & (UART_INTSTS_RLSINT_Msk | UART_INTSTS_BUFERRINT_Msk)) {
            UART_ClearIntFlag(UART1, (UART_INTSTS_RLSINT_Msk | UART_INTSTS_BUFERRINT_Msk));
        }
    }
}

static constexpr int KP_WIN = 16;
static constexpr int KP_DIM = 51;

static int8_t g_kpWindow[KP_WIN][KP_DIM];
static int g_kpCount = 0;
static int g_kpHead = 0;

static inline int8_t QuantizeToInt8(float x, float scale, int32_t zeroPoint)
{
    int32_t v = (int32_t)lrintf(x / scale) + zeroPoint;
    if (v < -128) v = -128;
    if (v > 127)  v = 127;
    return (int8_t)v;
}

static const char* kExerciseLabels[] =
{
    "hip thrusts",
    "jumping jacks",
    "lunges",
    "pullups",
    "pushups",
    "squats"
};
static constexpr int kNumExerciseLabels = sizeof(kExerciseLabels)/sizeof(kExerciseLabels[0]);

enum{
	ePOSE_KP_INDEX_NOSE,				//0
	ePOSE_KP_INDEX_LEFT_EYE,			//1
	ePOSE_KP_INDEX_RIGHT_EYE,			//2
	ePOSE_KP_INDEX_LEFT_EAR,			//3
	ePOSE_KP_INDEX_RIGHT_EAR,			//4
	ePOSE_KP_INDEX_LEFT_SHOULDER,		//5
	ePOSE_KP_INDEX_RIGHT_SHOULDER,		//6
	ePOSE_KP_INDEX_LEFT_ELBOW,			//7
	ePOSE_KP_INDEX_RIGHT_ELBOW,			//8
	ePOSE_KP_INDEX_LEFT_WRIST,			//9
	ePOSE_KP_INDEX_RIGHT_WRIST,			//10
	ePOSE_KP_INDEX_LEFT_HIP,			//11
	ePOSE_KP_INDEX_RIGHT_HIP,			//12
	ePOSE_KP_INDEX_LEFT_KNEE,			//13
	ePOSE_KP_INDEX_RIGHT_KNEE,			//14
	ePOSE_KP_INDEX_LEFT_ANKLE,			//15
	ePOSE_KP_INDEX_RIGHT_ANKLE,			//16
	ePOSE_KP_NUMS,						//17
}E_POSE_KP_INDEX;


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
    std::vector<arm::app::yolov8n_pose::PoseResult> results;
} S_FRAMEBUF;


S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

namespace arm
{
namespace app
{
/* Tensor arena buffer */
static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
static uint8_t exerciseTensorArena[512 * 1024] ACTIVATION_BUF_ATTRIBUTE;
	
/* Optional getter function for the model pointer and its size. */
namespace pose_landmark
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace hand_landmark */
	
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

void Display_PutText_Wrapped(const char* text, int x, int y, uint16_t color, uint16_t bg, int scale) {
    int screenWidth = 800; 
    int charWidth = 8 * scale;   // Standard font is 8 pixels wide
    int charHeight = 16 * scale; // Standard font is 16 pixels tall
    
    int currentX = x;
    int currentY = y;
    
    // Loop through every character in the string
    for (int i = 0; i < strlen(text); i++) {
        // 1. Check if this character fits on the current line
        if (currentX + charWidth > screenWidth) {
            // It doesn't fit! Move to the next line (Wrap)
            currentX = x;             // Reset to Left
            currentY += charHeight;   // Move Down
        }

        // 2. Draw just this ONE character
        Display_PutText(&text[i], 1, currentX, currentY, color, bg, false, scale);

        // 3. Move X forward for the next character
        currentX += charWidth;
    }
}
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

static void DrawPoseLandmark(
    const std::vector<arm::app::yolov8n_pose::PoseResult> &results,
    image_t *drawImg
)
{
	arm::app::yolov8n_pose::PoseResult pose;
	int lineColor = COLOR_R5_G6_B5_TO_RGB565(0,COLOR_G6_MAX, 0);	
	int poseSize = results.size();
	
	for(int p = 0; p < poseSize; p ++)
	{
		pose = results[p];
		imlib_draw_rectangle(drawImg, pose.m_poseBox.x, pose.m_poseBox.y, pose.m_poseBox.w, pose.m_poseBox.h, COLOR_B5_MAX, 2, false);

		struct S_KEY_POINT keyPoint;
		struct S_KEY_POINT keyPointTemp;
		std::vector<struct S_KEY_POINT> keyPoints;

		keyPoints = pose.m_keyPoints;

		for(int k = 0; k < keyPoints.size(); k ++)
		{
			keyPoint = keyPoints[k];
			
			if(keyPoint.visible < POSE_KEYPOINT_VISIBLE_THRESHOLD)
				continue;

			//draw points
			imlib_draw_circle(drawImg, keyPoint.x, keyPoint.y, 1, COLOR_B5_MAX, 1, true);

			//draw lines
			if( k == ePOSE_KP_INDEX_NOSE || k == ePOSE_KP_INDEX_LEFT_SHOULDER)
			{
				//Don't draw line
			}
			else if(k == ePOSE_KP_INDEX_LEFT_EYE || k == ePOSE_KP_INDEX_RIGHT_EYE)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_NOSE];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_LEFT_EAR)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_EYE];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_RIGHT_EAR)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_EYE];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_RIGHT_SHOULDER)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_SHOULDER];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_LEFT_ELBOW)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_SHOULDER];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_RIGHT_ELBOW)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_SHOULDER];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_LEFT_WRIST)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_ELBOW];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_RIGHT_WRIST)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_ELBOW];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_LEFT_HIP)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_SHOULDER];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}			
			else if(k == ePOSE_KP_INDEX_RIGHT_HIP)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_SHOULDER];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);

				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_HIP];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);

			}
			else if(k == ePOSE_KP_INDEX_LEFT_KNEE)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_HIP];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}						
			else if(k == ePOSE_KP_INDEX_RIGHT_KNEE)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_HIP];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}			
			else if(k == ePOSE_KP_INDEX_LEFT_ANKLE)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_KNEE];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}						
			else if(k == ePOSE_KP_INDEX_RIGHT_ANKLE)
			{
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_KNEE];
				
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}						
		}
	}
	
}

static int32_t PrepareModelToHyperRAM(const char* modelFile, uint32_t dstAddr)
{
#define EACH_READ_SIZE 512

    TCHAR sd_path[] = { '0', ':', 0 };
    f_chdrive(sd_path);

    int32_t i32FileSize;
    int32_t i32FileReadIndex = 0;
    int32_t i32Read;

    if (!ModelFileReader_Initialize(modelFile))
    {
        printf_err("Unable open model %s\n", modelFile);
        return -1;
    }

    i32FileSize = ModelFileReader_FileSize();
    info("Model file size %i \n", i32FileSize);

    while (i32FileReadIndex < i32FileSize)
    {
        i32Read = ModelFileReader_ReadData((BYTE *)(dstAddr + i32FileReadIndex), EACH_READ_SIZE);
        if (i32Read < 0)
            break;
        i32FileReadIndex += i32Read;
    }

    ModelFileReader_Finish();

    if (i32FileReadIndex < i32FileSize)
    {
        printf_err("Read model file size is not enough\n");
        return -2;
    }

    return i32FileSize;
}




int main()
{
    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();
			SYS_UnlockReg();
			/* UART1 (Bluetooth) Clock & Pins */
			CLK_SetModuleClock(UART1_MODULE, CLK_UARTSEL0_UART1SEL_HIRC, CLK_UARTDIV0_UART1DIV(1));
			CLK_EnableModuleClock(UART1_MODULE);
			SET_UART1_RXD_PA2();
			SET_UART1_TXD_PA3();

			/* UART1 Hardware Init */
			SYS_ResetModule(SYS_UART1RST);
			UART_Open(UART1, 9600);
			UART_EnableInt(UART1, UART_INTEN_RDAIEN_Msk);
			NVIC_EnableIRQ(UART1_IRQn);
			SYS_LockReg();

			
	/* Copy model file from SD to HyperRAM*/		
		int32_t yoloModelSize = PrepareModelToHyperRAM(YOLO_MODEL_FILE, YOLO_MODEL_AT_HYPERRAM_ADDR);
	if (yoloModelSize <= 0)
	{
			printf_err("Failed to prepare YOLO model\n");
			return 1;
	}

	// Compute a safe address for the exercise model right after YOLO (align to 1MB boundary)
	uint32_t exerciseModelAddr =
			YOLO_MODEL_AT_HYPERRAM_ADDR + ((yoloModelSize + 0xFFFFF) & ~0xFFFFF);

	// --- Load Exercise model to HyperRAM ---
	int32_t exerciseModelSize = PrepareModelToHyperRAM(EXERCISE_MODEL_FILE, exerciseModelAddr);
	if (exerciseModelSize <= 0)
	{
			printf_err("Failed to prepare EXERCISE model\n");
			return 1;
	}

	// --- Init YOLO model interpreter ---
	arm::app::YOLOv8nPoseModel poseModel;
	if (!poseModel.Init(arm::app::tensorArena,
											sizeof(arm::app::tensorArena),
											(unsigned char*)YOLO_MODEL_AT_HYPERRAM_ADDR,
											yoloModelSize))
	{
			printf_err("Failed to initialise YOLO model\n");
			return 1;
	}

	// --- Init Exercise model interpreter ---
	arm::app::ExerciseClassifierModel exerciseModel;
	if (!exerciseModel.Init(arm::app::exerciseTensorArena,
													sizeof(arm::app::exerciseTensorArena),
													(unsigned char*)exerciseModelAddr,
													exerciseModelSize))
	{
			printf_err("Failed to initialise EXERCISE model\n");
			return 1;
	}

    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena) + ACTIVATION_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
				{
					// SRAM for exercise tensor arena
						ARM_MPU_RBAR(((unsigned int)arm::app::exerciseTensorArena),
												 ARM_MPU_SH_NON,
												 0,
												 1,
												 1),
						ARM_MPU_RLAR((((unsigned int)arm::app::exerciseTensorArena) + sizeof(arm::app::exerciseTensorArena) - 1),
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

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    TfLiteTensor *inputTensor   = poseModel.GetInputTensor(0);


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

    TfLiteIntArray *inputShape = poseModel.GetInputShape(0);

    const int inputImgCols = inputShape->data[arm::app::YOLOv8nPoseModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::YOLOv8nPoseModel::ms_inputRowsIdx];
    const uint32_t nChannels = inputShape->data[arm::app::YOLOv8nPoseModel::ms_inputChannelsIdx];

    /* Hand landmark model preprocessing is image conversion from uint8 to [0,1] float values,
     * then quantize them with input quantization info. */
    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);
		
		TfLiteTensor* exInTensor  = exerciseModel.GetInputTensor(0);
		TfLiteTensor* exOutTensor = exerciseModel.GetOutputTensor(0);

		arm::app::QuantParams exInQ  = arm::app::GetTensorQuantParams(exInTensor);
		arm::app::QuantParams exOutQ = arm::app::GetTensorQuantParams(exOutTensor);

		// Optional sanity prints:
		info("EX input bytes=%d\n", exInTensor->bytes);
		info("EX output bytes=%d\n", exOutTensor->bytes);

    // postProcess
		arm::app::yolov8n_pose::YOLOv8nPosePostProcessing postProcess(&poseModel, POSE_PRESENCE_THRESHOLD);
	
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

#define EACH_PERF_SEC 1
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
            //resize full image to input tensor
            image_t resizeImg;

            roi.x = 0;
            roi.y = 0;
            roi.w = fullFramebuf->frameImage.w;
            roi.h = fullFramebuf->frameImage.h;

            resizeImg.w = inputImgCols;
            resizeImg.h = inputImgRows;
            resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
            resizeImg.pixfmt = PIXFORMAT_RGB888;

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
            imlib_nvt_scale(&fullFramebuf->frameImage, &resizeImg, &roi);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("resize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
			//Quantize input tensor data
			auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
			auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

			for (size_t i = 0; i < inputTensor->bytes; i++)
			{
//				auto i_data_int8 = static_cast<int8_t>(((static_cast<float>(req_data[i]) / 255.0f) / inQuantParams.scale) + inQuantParams.offset);
//				signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
				signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
			}

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#if defined(__PROFILE__)
			profiler.StartProfiling("Inference");
#endif

			poseModel.RunInference();

#if defined(__PROFILE__)
			profiler.StopProfiling();
			profiler.PrintProfilingResult();
#endif

            fullFramebuf->eState = eFRAMEBUF_INF;
        }
        infFramebuf = get_inf_framebuf();

        if (infFramebuf)
        {
			//post process

#if defined(__PROFILE__)
			u64StartCycle = pmu_get_systick_Count();
#endif
			postProcess.RunPostProcessing(
				inputImgCols,
				inputImgRows,
				infFramebuf->frameImage.w,
				infFramebuf->frameImage.h,
				infFramebuf->results);
					
			// ---- EXERCISE CLASSIFIER (runs after pose postprocess) ----
		static int g_lastExercise = -1;
		static float g_lastScore = 0.0f;

		if (infFramebuf->results.size() > 0)
		{
				// Use first detected person
				const auto& pose = infFramebuf->results[0];

				// Build 51 float features: normalized x,y plus visible
				float feat[KP_DIM];
				int idx = 0;

				for (int k = 0; k < ePOSE_KP_NUMS; k++)
				{
						const auto& kp = pose.m_keyPoints[k];
						float xn = (float)kp.x / (float)infFramebuf->frameImage.w;
						float yn = (float)kp.y / (float)infFramebuf->frameImage.h;
						float cn = kp.visible;

						feat[idx++] = xn;
						feat[idx++] = yn;
						feat[idx++] = cn;
				}

				// Quantize and push into ring buffer
				for (int i = 0; i < KP_DIM; i++)
				{
						g_kpWindow[g_kpHead][i] = QuantizeToInt8(feat[i], exInQ.scale, exInQ.offset);
				}

				g_kpHead = (g_kpHead + 1) % KP_WIN;
				g_kpCount++;

				// Run classifier when window full
				if (g_kpCount >= KP_WIN)
				{
						// Input is int8 [1,16,51]
						int8_t* exIn = (int8_t*)exInTensor->data.data;

						// Flatten ring buffer oldest->newest
						int start = g_kpHead; // after wrapping, head points to oldest
						for (int t = 0; t < KP_WIN; t++)
						{
								int src = (start + t) % KP_WIN;
								memcpy(exIn + t * KP_DIM, g_kpWindow[src], KP_DIM);
						}

						exerciseModel.RunInference();

						int8_t* exOut = (int8_t*)exOutTensor->data.data;
						int outLen = exOutTensor->dims->data[exOutTensor->dims->size - 1];

						// Argmax
						int best = 0;
						for (int i = 1; i < outLen; i++)
						{
								if (exOut[i] > exOut[best]) best = i;
						}

						// Dequantize (meaningful if softmax output)
						float score = ((int)exOut[best] - exOutQ.offset) * exOutQ.scale;

						g_lastExercise = best;
						g_lastScore = score;

						const char* name = (best < kNumExerciseLabels) ? kExerciseLabels[best] : "unknown";
						//info("EXERCISE: %s (id=%d) score=%f\n", name, best, score);
				}
		}

#if defined(__PROFILE__)
			u64EndCycle = pmu_get_systick_Count();
			info("post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

            //draw bbox and render
            /* Draw boxes. */
			if(infFramebuf->results.size())
			{
#if defined(__PROFILE__)
				u64StartCycle = pmu_get_systick_Count();
#endif
				DrawPoseLandmark(infFramebuf->results, &infFramebuf->frameImage);
#if defined(__PROFILE__)
				u64EndCycle = pmu_get_systick_Count();
				info("draw hand landmark cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif
			}

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
			
									// After FPS text, add exercise text (same frame)
						if (g_lastExercise >= 0)
						{
								char exText[64];
								const char* name =
										(g_lastExercise < kNumExerciseLabels) ? kExerciseLabels[g_lastExercise] : "unknown";

								// If your output is softmax-ish, g_lastScore might be 0..1
								snprintf(exText, sizeof(exText), "Exercise: %s (%.2f)              ", name, g_lastScore);

								// Put it one text line below your FPS line
								Display_PutText_Wrapped(
										exText,
										650,
										FONT_HTIGHT*2 + 300,
										//(frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) + (2 * FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT),
										C_BLUE,
										C_WHITE,
										FONT_DISP_UPSCALE_FACTOR
								);
								//Display_PutText(
										//exText,
										//strlen(exText),
										//500,
										//(frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) + (2 * FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT),
										//C_BLUE,
										//C_WHITE,
										//false,
										//FONT_DISP_UPSCALE_FACTOR
								//);
						}
						// Display accelerometer and heart rate data
		if (g_bMsgReceived) {
				char localBuf[RX_BUF_SIZE];
				// Copy and reset immediately to minimize ISR lockout
				NVIC_DisableIRQ(UART1_IRQn);
				memcpy(localBuf, (const void*)g_u8RecData, g_u32DataIdx);
				g_u32DataIdx = 0;
				g_bMsgReceived = false;
				NVIC_EnableIRQ(UART1_IRQn);

				// Expected format: "A:ax,ay,az|HR:heartrate"
				float bAx = 0, bAy = 0, bAz = 0;
				char bHr[20] = {0};
				
				// Parse the string
				if (sscanf(localBuf, "A:%f,%f,%f|HR:%s", &bAx, &bAy, &bAz, bHr) >= 3) {
						S_DISP_RECT sRect;
						sRect.u32TopLeftX = TEXT_X_OFFSET;
						sRect.u32BottonRightX = Disaplay_GetLCDWidth() - 1;
						
						// Display AX
						//info("Time Passed: %d\n", timepassed);
						info("AX: %.2f\n", bAx);
						sprintf(szDisplayText, "AX: %.2f", bAx);
						sRect.u32TopLeftY = TEXT_Y_AX; sRect.u32BottonRightY = TEXT_Y_AX + LINE_HEIGHT - 1;
						Display_ClearRect(C_WHITE, &sRect);
						Display_PutText(szDisplayText, strlen(szDisplayText), TEXT_X_OFFSET, TEXT_Y_AX, C_BLACK, C_WHITE, false, FONT_DISP_UPSCALE_FACTOR);

						// Display AY
						info("AY: %.2f\n", bAy);
						sprintf(szDisplayText, "AY: %.2f", bAy);
						sRect.u32TopLeftY = TEXT_Y_AY; sRect.u32BottonRightY = TEXT_Y_AY + LINE_HEIGHT - 1;
						Display_ClearRect(C_WHITE, &sRect);
						Display_PutText(szDisplayText, strlen(szDisplayText), TEXT_X_OFFSET, TEXT_Y_AY, C_BLACK, C_WHITE, false, FONT_DISP_UPSCALE_FACTOR);

						// Display AZ
						info("AZ: %.2f\n", bAz);
						sprintf(szDisplayText, "AZ: %.2f", bAz);
						sRect.u32TopLeftY = TEXT_Y_AZ; sRect.u32BottonRightY = TEXT_Y_AZ + LINE_HEIGHT - 1;
						Display_ClearRect(C_WHITE, &sRect);
						Display_PutText(szDisplayText, strlen(szDisplayText), TEXT_X_OFFSET, TEXT_Y_AZ, C_BLACK, C_WHITE, false, FONT_DISP_UPSCALE_FACTOR);
						
						timepassed +=1;
						// Display HR
						//info("HR: %s\n", bHr);
						sprintf(szDisplayText, "HR: %s", bHr);
						sRect.u32TopLeftY = TEXT_Y_HR; sRect.u32BottonRightY = TEXT_Y_HR + LINE_HEIGHT - 1;
						Display_ClearRect(C_WHITE, &sRect);
						Display_PutText(szDisplayText, strlen(szDisplayText), TEXT_X_OFFSET, TEXT_Y_HR, (strcmp(bHr, "No") == 0 ? C_RED : C_MAGENTA), C_WHITE, false, FONT_DISP_UPSCALE_FACTOR);
				}
		}

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
                //info("Total inference rate: %llu\n", u64PerfFrames / EACH_PERF_SEC);
#if defined (__USE_DISPLAY__)
		
		
                sprintf(szDisplayText, "Frame Rate %llu", u64PerfFrames / EACH_PERF_SEC);
                //sprintf(szDisplayText,"Time %llu",(uint64_t) pmu_get_systick_Count() / (uint64_t)SystemCoreClock);
                //info("Running %s sec \n", szDisplayText);
				#define SCREEN_HEIGHT 240			
				uint32_t yOffset = (SCREEN_HEIGHT - (frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR)) / 2;
				sDispRect.u32TopLeftX = 0;																																																																																																													
				sDispRect.u32TopLeftY = frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR;
				sDispRect.u32BottonRightX = (frameBuffer.w);
				sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) + (FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT) - 1);

                Display_ClearRect(C_WHITE, &sDispRect);
                //Display_PutText(
                    //szDisplayText,
                    //strlen(szDisplayText),
                    //650,
										//0,
                    //C_BLUE,
                    //C_WHITE,
                    //false,
					//FONT_DISP_UPSCALE_FACTOR
                //);
								Display_PutText_Wrapped(
										szDisplayText,
										650,
										0,
										//(frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) + (2 * FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT),
										C_BLUE,
										C_WHITE,
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
