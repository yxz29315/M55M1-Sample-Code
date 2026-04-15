/**************************************************************************//**
 * @file     main.cpp
 * @version  V2.00
 * @brief    Thin shell: camera capture → SpeakingDetector → display.
 *           All ML logic lives in SpeakingDetector.cpp.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "BoardInit.hpp"
#include "SpeakingDetector.hpp"
#include "log_macros.h"
#include "imlib.h"
#include "framebuffer.h"

#undef PI
#include "NuMicro.h"

#include "Profiler.hpp"
#include "ImageSensor.h"

#include <vector>
#include <cstring>

//#define __PROFILE__
#define __USE_DISPLAY__
//#define __USE_UVC__

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

/* ------------------------------------------------------------------ */
/*  Frame buffer management                                            */
/* ------------------------------------------------------------------ */
#define NUM_FRAMEBUF 2

typedef enum {
    eFRAMEBUF_EMPTY,
    eFRAMEBUF_FULL,
    eFRAMEBUF_INF
} E_FRAMEBUF_STATE;

typedef struct {
    E_FRAMEBUF_STATE eState;
    image_t frameImage;
} S_FRAMEBUF;

static S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

static S_FRAMEBUF *get_empty_framebuf()
{
    int i;
    for (i = 0; i < NUM_FRAMEBUF; i++)
        if (s_asFramebuf[i].eState == eFRAMEBUF_EMPTY) return &s_asFramebuf[i];
    return NULL;
}

static S_FRAMEBUF *get_full_framebuf()
{
    int i;
    for (i = 0; i < NUM_FRAMEBUF; i++)
        if (s_asFramebuf[i].eState == eFRAMEBUF_FULL) return &s_asFramebuf[i];
    return NULL;
}

static S_FRAMEBUF *get_inf_framebuf()
{
    int i;
    for (i = 0; i < NUM_FRAMEBUF; i++)
        if (s_asFramebuf[i].eState == eFRAMEBUF_INF) return &s_asFramebuf[i];
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Display / image constants                                          */
/* ------------------------------------------------------------------ */
#define IMAGE_DISP_UPSCALE_FACTOR 2
#if defined(LT7381_LCD_PANEL)
#define FONT_DISP_UPSCALE_FACTOR 2
#else
#define FONT_DISP_UPSCALE_FACTOR 1
#endif

#if defined(__USE_UVC__)
#define GLCD_WIDTH   320
#define GLCD_HEIGHT  240
#else
#define GLCD_WIDTH   320
#define GLCD_HEIGHT  240
#endif

#define IMAGE_FB_SIZE  (GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef  OMV_FB_SIZE
#define OMV_FB_SIZE    (IMAGE_FB_SIZE + 1024)

#undef  OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE (1 * 1024)

__attribute__((section(".bss.vram.data"), aligned(32)))
static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];

__attribute__((section(".bss.vram.data"), aligned(32)))
static char jpeg_array[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF == 2)
__attribute__((section(".bss.vram.data"), aligned(32)))
static char frame_buf1[OMV_FB_SIZE];
#endif

char *_fb_base  = NULL;
char *_fb_end   = NULL;
char *_jpeg_buf = NULL;
char *_fballoc  = NULL;

static void omv_init()
{
    image_t frameBuffer;
    int i;

    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end  = fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();
    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);

    for (i = 0; i < NUM_FRAMEBUF; i++)
        s_asFramebuf[i].eState = eFRAMEBUF_EMPTY;

    framebuffer_init_image(&s_asFramebuf[0].frameImage);

#if (NUM_FRAMEBUF == 2)
    s_asFramebuf[1].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[1].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[1].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    s_asFramebuf[1].frameImage.pixfmt = PIXFORMAT_RGB565;
    s_asFramebuf[1].frameImage.data = (uint8_t *)frame_buf1;
#endif
}

/* ================================================================== */
/*  main — capture → detect → draw → display loop                     */
/* ================================================================== */
int main()
{
    BoardInit();
    info("main: BoardInit done\n");

    omv_init();

    image_t frameBuffer;
    framebuffer_init_image(&frameBuffer);

    /* --- MPU setup (tensor arenas + frame buffers, one call) --- */
    {
        void *faceArena, *mouthArena;
        uint32_t faceSize, mouthSize;
        SpeakingDetector_GetTensorArenas(&faceArena, &faceSize,
                                         &mouthArena, &mouthSize);

        const std::vector<ARM_MPU_Region_t> mpuConfig = {
            {
                ARM_MPU_RBAR((unsigned int)faceArena,
                             ARM_MPU_SH_NON, 0, 1, 1),
                ARM_MPU_RLAR((unsigned int)faceArena + faceSize - 1,
                             eMPU_ATTR_CACHEABLE_WTRA)
            },
            {
                ARM_MPU_RBAR((unsigned int)mouthArena,
                             ARM_MPU_SH_NON, 0, 1, 1),
                ARM_MPU_RLAR((unsigned int)mouthArena + mouthSize - 1,
                             eMPU_ATTR_CACHEABLE_WTRA)
            },
            {
                ARM_MPU_RBAR((unsigned int)fb_array,
                             ARM_MPU_SH_NON, 0, 1, 1),
                ARM_MPU_RLAR((unsigned int)fb_array + OMV_FB_SIZE - 1,
                             eMPU_ATTR_NON_CACHEABLE)
            },
#if (NUM_FRAMEBUF == 2)
            {
                ARM_MPU_RBAR((unsigned int)frame_buf1,
                             ARM_MPU_SH_NON, 0, 1, 1),
                ARM_MPU_RLAR((unsigned int)frame_buf1 + OMV_FB_SIZE - 1,
                             eMPU_ATTR_NON_CACHEABLE)
            },
#endif
        };
        InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());
    }

    /* --- Initialise the speaking detector module --- */
    int rc = SpeakingDetector_Init(NULL);  /* NULL = use defaults */
    if (rc != 0) {
        printf_err("SpeakingDetector_Init failed (%d)\n", rc);
        return 1;
    }

    /* --- Camera --- */
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

#if defined(__PROFILE__)
    arm::app::Profiler profiler;
    uint64_t u64StartCycle, u64EndCycle;
    uint64_t u64CCAPStartCycle, u64CCAPEndCycle;
#else
    pmu_reset_counters();
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle = pmu_get_systick_Count()
                          + (SystemCoreClock * EACH_PERF_SEC);
    uint64_t u64PerfFrames = 0;

    S_FRAMEBUF *infFramebuf;
    S_FRAMEBUF *fullFramebuf;
    S_FRAMEBUF *emptyFramebuf;

    SpeakingFaceResult faceResults[MAX_TRACKED_FACES];
    int numFaces = 0;

    /* ---- Main loop ---- */
    while (1)
    {
        /* 1. Trigger capture into an empty buffer */
        emptyFramebuf = get_empty_framebuf();
        if (emptyFramebuf) {
#if defined(__PROFILE__)
            u64CCAPStartCycle = pmu_get_systick_Count();
#endif
            ImageSensor_TriggerCapture((uint32_t)(emptyFramebuf->frameImage.data));
        }

        /* 2. Run ML on a full buffer */
        fullFramebuf = get_full_framebuf();
        if (fullFramebuf) {
            numFaces = SpeakingDetector_RunFrame(
                fullFramebuf->frameImage.data,
                fullFramebuf->frameImage.w,
                fullFramebuf->frameImage.h,
                faceResults, MAX_TRACKED_FACES);

            fullFramebuf->eState = eFRAMEBUF_INF;
        }

        /* 3. Draw + display an inference-done buffer */
        infFramebuf = get_inf_framebuf();
        if (infFramebuf) {
#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
            SpeakingDetector_Draw(
                infFramebuf->frameImage.data,
                infFramebuf->frameImage.w,
                infFramebuf->frameImage.h,
                faceResults, numFaces);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("draw cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#if defined (__USE_DISPLAY__)
            sDispRect.u32TopLeftX = 0;
            sDispRect.u32TopLeftY = 0;
            sDispRect.u32BottonRightX = ((frameBuffer.w * IMAGE_DISP_UPSCALE_FACTOR) - 1);
            sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) - 1);

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
            Display_FillRect((uint16_t *)infFramebuf->frameImage.data, &sDispRect,
                             IMAGE_DISP_UPSCALE_FACTOR);
#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("display image cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif
#endif /* __USE_DISPLAY__ */

#if defined (__USE_UVC__)
            if (UVC_IsConnect()) {
#if (UVC_Color_Format == UVC_Format_YUY2)
                image_t RGB565Img, YUV422Img;
                rectangle_t uvcRoi;

                RGB565Img.w = infFramebuf->frameImage.w;
                RGB565Img.h = infFramebuf->frameImage.h;
                RGB565Img.data = (uint8_t *)infFramebuf->frameImage.data;
                RGB565Img.pixfmt = PIXFORMAT_RGB565;

                YUV422Img.w = RGB565Img.w;
                YUV422Img.h = RGB565Img.h;
                YUV422Img.data = (uint8_t *)infFramebuf->frameImage.data;
                YUV422Img.pixfmt = PIXFORMAT_YUV422;

                uvcRoi.x = 0;  uvcRoi.y = 0;
                uvcRoi.w = RGB565Img.w;  uvcRoi.h = RGB565Img.h;
                imlib_nvt_scale(&RGB565Img, &YUV422Img, &uvcRoi);
#else
                image_t origImg, vflipImg;

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
                UVC_SendImage((uint32_t)infFramebuf->frameImage.data, IMAGE_FB_SIZE,
                              uvcStatus.StillImage);
            }
#endif /* __USE_UVC__ */

            /* Frame-rate counter */
            u64PerfFrames++;
            if ((uint64_t)pmu_get_systick_Count() > u64PerfCycle) {
                info("Total inference rate: %llu\n", u64PerfFrames / EACH_PERF_SEC);
#if defined (__USE_DISPLAY__)
                sprintf(szDisplayText, "Frame Rate %llu", u64PerfFrames / EACH_PERF_SEC);

                sDispRect.u32TopLeftX = 0;
                sDispRect.u32TopLeftY = frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR;
                sDispRect.u32BottonRightX = (frameBuffer.w);
                sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR)
                                             + (FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT) - 1);

                Display_ClearRect(C_WHITE, &sDispRect);
                Display_PutText(szDisplayText, strlen(szDisplayText),
                                0, frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR,
                                C_BLUE, C_WHITE, false, FONT_DISP_UPSCALE_FACTOR);
#endif
                u64PerfCycle = (uint64_t)pmu_get_systick_Count()
                             + (uint64_t)(SystemCoreClock * EACH_PERF_SEC);
                u64PerfFrames = 0;
            }

            infFramebuf->eState = eFRAMEBUF_EMPTY;
        }

        /* 4. Wait for camera capture to finish */
        if (emptyFramebuf) {
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
