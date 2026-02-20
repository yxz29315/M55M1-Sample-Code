/**************************************************************************//**
 * @file     Reset_Handler_PreInit.c
 * @brief    Early HyperRAM init before BSS zeroing - required for 3MB activation buffer.
 *           The activation buffer extends into HyperRAM; it must be accessible when
 *           the C runtime zeros BSS at startup.
 ******************************************************************************/
#include "NuMicro.h"
#include "hyperram_code.h"

#define HYPERRAM_SPIM_PORT SPIM0

extern void SetDebugUartCLK(void);
extern void SetDebugUartMFP(void);
extern void InitDebugUart(void);

/**
 * Initialize HyperRAM before __PROGRAM_START zeros BSS.
 * Overrides weak Reset_Handler_PreInit in startup_M55M1.c.
 * UART must be init first because HyperRAM_Init->TrimDLLDelayNumber uses printf.
 */
void Reset_Handler_PreInit(void)
{
    SYS_UnlockReg();

    /* Use HIRC + APLL for clock (same as BoardInit with USE_HIRC_CLOCK) */
    CLK_SetBusClock(CLK_SCLKSEL_SCLKSEL_APLL0, CLK_APLLCTL_APLLSRC_HIRC, __HSI);
    SystemCoreClockUpdate();

    /* UART init BEFORE HyperRAM - HyperRAM_Init calls printf during DLL trim */
    SetDebugUartCLK();
    SetDebugUartMFP();
    InitDebugUart();

    /* Enable GPIO for HyperRAM pins */
    CLK_EnableModuleClock(GPIOG_MODULE);
    CLK_EnableModuleClock(GPIOH_MODULE);
    CLK_EnableModuleClock(GPIOJ_MODULE);

    HyperRAM_PinConfig(HYPERRAM_SPIM_PORT);
    HyperRAM_Init(HYPERRAM_SPIM_PORT);
    SPIM_HYPER_EnterDirectMapMode(HYPERRAM_SPIM_PORT);
}
