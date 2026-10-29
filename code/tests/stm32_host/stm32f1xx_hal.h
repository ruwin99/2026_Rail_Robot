#ifndef HOST_STM32F1XX_HAL_H
#define HOST_STM32F1XX_HAL_H
#include <stdint.h>
typedef struct { uint32_t CCR1, CCR2, CCR3, CCR4, CNT, ARR, DIER; } TIM_TypeDef;
typedef struct { TIM_TypeDef *Instance; } TIM_HandleTypeDef;
typedef struct { void *Instance; uint32_t RxState, ErrorCode; } UART_HandleTypeDef;
typedef struct { uint32_t ODR, BSRR; } GPIO_TypeDef;
typedef enum { GPIO_PIN_RESET, GPIO_PIN_SET } GPIO_PinState;
typedef enum { HAL_OK, HAL_ERROR, HAL_BUSY, HAL_TIMEOUT } HAL_StatusTypeDef;
extern TIM_TypeDef host_tim1, host_tim2, host_tim3, host_tim4;
#define TIM1 (&host_tim1)
#define TIM2 (&host_tim2)
#define TIM3 (&host_tim3)
#define TIM4 (&host_tim4)
#define TIM_CHANNEL_1 0U
#define TIM_CHANNEL_2 4U
#define TIM_CHANNEL_3 8U
#define TIM_CHANNEL_4 12U
#define TIM_CHANNEL_ALL 60U
#define TIM_IT_UPDATE 1U
#define TIM_FLAG_UPDATE 1U
#define USART2 ((void *)2)
#define USART3 ((void *)3)
#define HAL_UART_STATE_READY 0U
#define HAL_UART_ERROR_NONE 0U
#define TIM3_IRQn 3
#define USART2_IRQn 2
extern GPIO_TypeDef host_gpioa, host_gpiob, host_gpioc;
#define GPIOA (&host_gpioa)
#define GPIOB (&host_gpiob)
#define GPIOC (&host_gpioc)
#define GPIO_PIN_5 (1U << 5)
#define GPIO_PIN_6 (1U << 6)
#define GPIO_PIN_8 (1U << 8)
#define GPIO_PIN_9 (1U << 9)
#define GPIO_PIN_13 (1U << 13)
#define GPIO_PIN_14 (1U << 14)
typedef struct { uint32_t KR, PR, RLR, SR; } IWDG_TypeDef;
typedef struct { uint32_t CR; } DBGMCU_TypeDef;
extern IWDG_TypeDef host_iwdg;
extern DBGMCU_TypeDef host_dbgmcu;
#define IWDG (&host_iwdg)
#define DBGMCU (&host_dbgmcu)
#define IWDG_SR_PVU 1U
#define IWDG_SR_RVU 2U
#define DBGMCU_CR_DBG_IWDG_STOP (1U << 8)
typedef struct { uint32_t CTRL, CYCCNT; } DWT_Type;
typedef struct { uint32_t DEMCR; } CoreDebug_Type;
extern DWT_Type host_dwt;
extern CoreDebug_Type host_coredebug;
extern uint32_t SystemCoreClock;
#define DWT (&host_dwt)
#define CoreDebug (&host_coredebug)
#define CoreDebug_DEMCR_TRCENA_Msk (1UL << 24)
#define DWT_CTRL_CYCCNTENA_Msk 1U
uint32_t HAL_GetTick(void);
uint32_t __get_PRIMASK(void);
void __disable_irq(void);
void __enable_irq(void);
void host_set_compare(TIM_HandleTypeDef *, uint32_t, uint32_t);
uint32_t host_get_compare(TIM_HandleTypeDef *, uint32_t);
uint32_t host_get_counter(TIM_HandleTypeDef *);
#define __HAL_TIM_SET_COMPARE(t,c,v) host_set_compare((t),(c),(v))
#define __HAL_TIM_GET_COMPARE(t,c) host_get_compare((t),(c))
#define __HAL_TIM_GET_COUNTER(t) host_get_counter(t)
#define __HAL_TIM_SET_COUNTER(t,v) ((t)->Instance->CNT = (v))
#define __HAL_TIM_GET_AUTORELOAD(t) ((t)->Instance->ARR)
#define __HAL_TIM_DISABLE_IT(t,b) ((t)->Instance->DIER &= ~(b))
#define __HAL_TIM_ENABLE_IT(t,b) ((t)->Instance->DIER |= (b))
#define __HAL_TIM_CLEAR_FLAG(t,b) ((void)(t), (void)(b))
HAL_StatusTypeDef HAL_TIM_PWM_Start(TIM_HandleTypeDef *, uint32_t);
HAL_StatusTypeDef HAL_TIM_Encoder_Start(TIM_HandleTypeDef *, uint32_t);
HAL_StatusTypeDef HAL_TIM_Encoder_Start_IT(TIM_HandleTypeDef *, uint32_t);
HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *, const uint8_t *, uint16_t, uint32_t);
HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *, uint8_t *, uint16_t);
HAL_StatusTypeDef HAL_UART_AbortReceive(UART_HandleTypeDef *);
void HAL_GPIO_WritePin(GPIO_TypeDef *, uint16_t, GPIO_PinState);
void HAL_NVIC_ClearPendingIRQ(int);
#endif
