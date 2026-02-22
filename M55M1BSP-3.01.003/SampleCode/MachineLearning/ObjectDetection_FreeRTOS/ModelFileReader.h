/**************************************************************************//**
 * @file     ModelFileReader.h
 * @version  V1.00
 * @brief    Read model file from SD card
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef __MODEL_FILE_READER_H__
#define __MODEL_FILE_READER_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef signed int     BOOL;
typedef unsigned char BYTE;
typedef signed int    INT32;
typedef const char   *PCSTR;
typedef void          VOID;

BOOL ModelFileReader_Initialize(PCSTR pszOutFileName);
BOOL ModelFileReader_Finish(VOID);
INT32 ModelFileReader_ReadData(BYTE *pbyData, INT32 i32DataSize);
INT32 ModelFileReader_FileSize(VOID);
VOID ModelFileReader_Rewind(VOID);

#ifdef __cplusplus
}
#endif

#endif /* __MODEL_FILE_READER_H__ */
