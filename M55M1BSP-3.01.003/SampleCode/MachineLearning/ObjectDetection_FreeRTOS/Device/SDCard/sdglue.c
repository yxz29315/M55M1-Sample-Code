/******************************************************************************
 * @file     sdglue.c
 * @version  V1.00
 * @brief    SD glue functions for FATFS.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "NuMicro.h"
#include "diskio.h"
#include "ff.h"

#define DEF_CARD_DETECT_SOURCE  CardDetect_From_GPIO

static FATFS  _FatfsVolSd0;
static FATFS  _FatfsVolSd1;

static TCHAR  _Path[3];

void SDH0_IRQHandler(void)
{
    unsigned int volatile isr;
    unsigned int volatile ier;

    if (SDH0->GINTSTS & SDH_GINTSTS_DTAIF_Msk)
    {
        SDH0->GCTL |= SDH_GCTL_GCTLRST_Msk;
    }

    isr = SDH0->INTSTS;
    ier = SDH0->INTEN;

    if (isr & SDH_INTSTS_BLKDIF_Msk)
    {
        SD0.DataReadyFlag = 1;
        SDH0->INTSTS = SDH_INTSTS_BLKDIF_Msk;
    }

    if ((ier & SDH_INTEN_CDIEN_Msk) && (isr & SDH_INTSTS_CDIF_Msk))
    {
        {
            int volatile i;
            for (i = 0; i < 0x500; i++);
            isr = SDH0->INTSTS;
        }

#if (DEF_CARD_DETECT_SOURCE == CardDetect_From_DAT3)
        if (!(isr & SDH_INTSTS_CDSTS_Msk))
#else
        if (isr & SDH_INTSTS_CDSTS_Msk)
#endif
        {
            printf("\n***** card remove !\n");
            SD0.IsCardInsert = 0;
        }
        else
        {
            printf("***** card insert !\n");
        }

        SDH0->INTSTS = SDH_INTSTS_CDIF_Msk;
    }

    if (isr & SDH_INTSTS_CRCIF_Msk)
    {
        SDH0->INTSTS = SDH_INTSTS_CRCIF_Msk;
    }

    if (isr & SDH_INTSTS_DITOIF_Msk)
    {
        SDH0->INTSTS |= SDH_INTSTS_DITOIF_Msk;
    }

    if (isr & SDH_INTSTS_RTOIF_Msk)
    {
        SDH0->INTSTS |= SDH_INTSTS_RTOIF_Msk;
    }

    __DSB();
    __ISB();
}

int32_t SDH_Open_Disk(SDH_T *sdh, uint32_t u32CardDetSrc)
{
    SDH_Open(sdh, u32CardDetSrc);

    if (SDH_Probe(sdh))
    {
        printf("SD initial fail!!\n");
        return SDH_ERR_FAIL;
    }

    _Path[1] = ':';
    _Path[2] = 0;

    if (sdh == SDH0)
    {
        _Path[0] = '0';
        f_mount(&_FatfsVolSd0, _Path, 1);
    }
    else
    {
        _Path[0] = '1';
        f_mount(&_FatfsVolSd1, _Path, 1);
    }

    return SDH_OK;
}

void SDH_Close_Disk(SDH_T *sdh)
{
    if (sdh == SDH0)
    {
        memset(&SD0, 0, sizeof(SDH_INFO_T));
        f_mount(NULL, _Path, 1);
        memset(&_FatfsVolSd0, 0, sizeof(FATFS));
    }
    else
    {
        memset(&SD1, 0, sizeof(SDH_INFO_T));
        f_mount(NULL, _Path, 1);
        memset(&_FatfsVolSd1, 0, sizeof(FATFS));
    }
}

DWORD get_fattime(void)
{
    return 0x00000;
}
