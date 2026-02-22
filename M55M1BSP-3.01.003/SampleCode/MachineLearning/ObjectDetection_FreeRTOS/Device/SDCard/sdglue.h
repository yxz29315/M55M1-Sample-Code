/******************************************************************************
 * @file     sdglue.h
 * @brief    SD glue - SDH_Open_Disk declaration
 *****************************************************************************/
#ifndef __SDGLUE_H__
#define __SDGLUE_H__

#include "NuMicro.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t SDH_Open_Disk(SDH_T *sdh, uint32_t u32CardDetSrc);
void SDH_Close_Disk(SDH_T *sdh);

#ifdef __cplusplus
}
#endif

#endif
