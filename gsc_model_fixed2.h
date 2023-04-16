#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>

#define FIXED_POINT	9	// Fixed point scaling factor, set to 0 when using floating point
#define NUMBER_MIN	-32768	// Max value for this numeric type
#define NUMBER_MAX	32767	// Min value for this numeric type
typedef int16_t number_t;		// Standard size numeric type used for weights and activations
typedef int32_t long_number_t;	// Long numeric type used for intermediate results

#ifndef min
static inline long_number_t min(long_number_t a, long_number_t b) {
	if (a <= b)
		return a;
	return b;
}
#endif

#ifndef max
static inline long_number_t max(long_number_t a, long_number_t b) {
	if (a >= b)
		return a;
	return b;
}
#endif

#if FIXED_POINT > 0 // Scaling/clamping for fixed-point representation
static inline long_number_t scale_number_t(long_number_t number) {
	return number >> FIXED_POINT;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) max(NUMBER_MIN, min(NUMBER_MAX, number));
}
#else // No scaling/clamping required for floating-point
static inline long_number_t scale_number_t(long_number_t number) {
	return number;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) number;
}
#endif


#endif //__NUMBER_H__
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  1
#define INPUT_SAMPLES   16000
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_5_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_5(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       8000
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    10
#define CONV_STRIDE         10

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_4_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_4(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    1
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  10


const int16_t conv1d_4_bias[CONV_FILTERS] = {-8, 14, -11, 9, 6, -15, -8, -1, -8, -4, 10, 5, -10, 1, 11, 5}
;

const int16_t conv1d_4_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-72, 70, -40, 84, 71, 55, 6, 72, 62, 92}
}
, {{12, -70, -24, -36, -39, -25, -68, 74, -25, -10}
}
, {{87, 97, 3, 5, 34, 33, -65, 37, 51, -98}
}
, {{10, -82, 0, 87, 11, -22, 79, -54, -84, 41}
}
, {{-38, 32, 61, 81, -51, 17, -72, -6, 41, -64}
}
, {{49, 55, -13, 22, 17, 14, 56, -57, 85, 61}
}
, {{85, 54, -88, -63, 13, -68, -48, 9, 57, 31}
}
, {{51, 81, 53, -35, -47, 68, 71, 43, 69, -82}
}
, {{19, 73, -84, -10, -72, 37, 10, -53, -62, -99}
}
, {{-11, 85, 21, -22, -90, -45, 51, 8, 69, -20}
}
, {{68, 41, -62, 58, 82, 12, -30, 61, -22, 16}
}
, {{35, -65, 50, -2, 85, -69, 0, 79, -87, 59}
}
, {{31, -94, 95, 82, -49, 12, 49, 12, -30, -24}
}
, {{-84, 35, 59, -59, -58, -45, -33, 40, -18, -12}
}
, {{84, -84, -48, -91, -69, -9, 42, 23, -40, -85}
}
, {{76, -20, 92, 98, 90, -43, -27, 14, 44, -62}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   800
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_6_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_6(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       400
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_5_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_5(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_5_bias[CONV_FILTERS] = {1, -10, -2, 1, 10, -11, -2, 2, 16, 0, -14, 13, 3, -4, 9, 6, 16, -3, 7, 15, 16, 10, -6, 8, 8, -12, 8, -5, -14, 0, -4, 16}
;

const int16_t conv1d_5_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{92, 80, -66}
, {21, -3, 42}
, {99, -18, 41}
, {-47, -29, 82}
, {26, 41, -69}
, {-76, -93, -105}
, {-18, 62, -92}
, {3, 47, 37}
, {74, -67, -51}
, {-38, 0, -5}
, {26, -56, 29}
, {-87, -12, 62}
, {-102, -95, 71}
, {-51, 5, -11}
, {13, 29, -56}
, {52, 93, 98}
}
, {{64, 4, 72}
, {-73, 0, 66}
, {42, 84, 86}
, {-51, -68, -52}
, {42, 18, 18}
, {-11, 83, -10}
, {-39, -88, 78}
, {-90, 98, 40}
, {93, 7, 65}
, {-99, -74, 40}
, {-81, 37, -29}
, {40, 30, -14}
, {-65, -69, -55}
, {35, -7, -6}
, {17, -27, -80}
, {56, 23, 40}
}
, {{-1, 68, 61}
, {-106, -8, -11}
, {-28, 85, -4}
, {-47, 70, -56}
, {-45, 7, -90}
, {35, 64, 32}
, {99, -73, 32}
, {42, -31, -78}
, {29, 78, -9}
, {-41, -7, -46}
, {-61, -59, -93}
, {74, 11, -80}
, {-8, -76, -48}
, {-70, 17, 56}
, {5, -65, 6}
, {49, 79, 1}
}
, {{-45, -65, 66}
, {105, 74, 81}
, {-69, 100, -88}
, {30, -16, 12}
, {33, 96, -32}
, {59, 95, 70}
, {-18, -90, 88}
, {-57, 3, -67}
, {2, -82, -72}
, {-13, -17, -12}
, {85, 31, -54}
, {46, 9, 47}
, {-3, 19, 5}
, {-11, -33, -100}
, {-18, -82, -100}
, {87, -67, -4}
}
, {{21, 57, -79}
, {-38, 0, -60}
, {-65, 84, -81}
, {61, 78, 33}
, {27, 46, -29}
, {42, 102, -70}
, {-72, 5, -83}
, {-9, 3, -37}
, {-13, 54, 3}
, {-33, 26, -10}
, {-74, 101, -78}
, {45, 75, 41}
, {51, 75, -61}
, {11, 36, -41}
, {13, 108, 34}
, {-81, 1, -47}
}
, {{2, 68, 64}
, {90, -14, 64}
, {22, 9, -92}
, {-30, -1, 47}
, {30, -63, -42}
, {55, 60, -60}
, {63, 6, 53}
, {-5, -59, -69}
, {-24, 108, -12}
, {-105, -63, -66}
, {-15, 34, -21}
, {-72, 44, 25}
, {37, -95, -83}
, {-84, -96, -80}
, {-90, 19, 78}
, {-30, -4, -34}
}
, {{95, 95, 69}
, {81, 83, 56}
, {-36, -45, -101}
, {105, 52, -39}
, {75, -70, -24}
, {7, 36, -40}
, {-45, 36, 83}
, {12, 11, 59}
, {77, 35, -14}
, {94, 6, 12}
, {-108, 59, 44}
, {-73, -28, 90}
, {-67, 23, -38}
, {-101, 99, -68}
, {-76, 0, 20}
, {-37, -38, -87}
}
, {{-21, 80, 68}
, {51, -107, -6}
, {-62, -23, -12}
, {57, -75, 2}
, {86, 109, 19}
, {2, 25, 22}
, {105, -88, -1}
, {82, -76, 97}
, {-28, 35, -1}
, {-59, 79, -89}
, {106, -42, -65}
, {-6, 80, -3}
, {28, 18, -5}
, {22, -59, -31}
, {103, -75, 87}
, {-72, -95, 21}
}
, {{-60, -72, -70}
, {97, -91, 32}
, {-12, 12, -40}
, {-27, 5, 65}
, {105, -57, 50}
, {-63, -23, 72}
, {-21, 70, 96}
, {-62, -41, -9}
, {76, 49, -83}
, {-53, -25, 47}
, {28, -21, 6}
, {-90, 33, -27}
, {-34, -36, 95}
, {73, -48, 26}
, {42, 37, -29}
, {50, 73, -36}
}
, {{27, 68, 90}
, {-5, 68, 17}
, {-98, -39, 29}
, {37, -32, -106}
, {-13, -101, -45}
, {8, -37, -72}
, {97, 13, -95}
, {9, -64, 80}
, {-53, 57, -103}
, {56, 96, -52}
, {-19, 46, 55}
, {-82, 75, -94}
, {54, -58, -44}
, {50, 76, -10}
, {-90, -90, 24}
, {85, -59, 2}
}
, {{-42, 45, 52}
, {-51, -88, 78}
, {100, 5, 98}
, {14, 36, 7}
, {3, 93, 51}
, {89, -45, -76}
, {-80, 98, 31}
, {46, 104, -45}
, {51, 93, -88}
, {54, -84, -77}
, {-29, -78, -88}
, {9, 28, 107}
, {89, 26, 49}
, {-16, -43, 57}
, {50, -46, -72}
, {62, -85, 48}
}
, {{-74, -31, 44}
, {62, 54, 62}
, {-78, 12, 20}
, {-61, 6, -83}
, {-16, -34, -98}
, {-14, 11, 71}
, {37, -99, -32}
, {41, -42, -7}
, {-64, -23, 94}
, {76, -9, -83}
, {4, 47, 82}
, {27, -10, -5}
, {-100, 25, -15}
, {-61, 80, 14}
, {-17, -100, -26}
, {66, -81, 102}
}
, {{-17, 56, -100}
, {30, -87, -88}
, {20, -15, -21}
, {-46, 98, -73}
, {9, 7, 99}
, {-61, -63, 22}
, {-82, 106, 120}
, {18, -3, -96}
, {72, 93, 72}
, {-60, 50, 122}
, {-59, 97, -86}
, {104, -20, -8}
, {-89, 55, 29}
, {-78, -29, -18}
, {45, -68, 16}
, {5, -51, 67}
}
, {{-105, 81, -24}
, {-20, -96, 73}
, {36, 79, 46}
, {-91, -18, 30}
, {51, 86, -99}
, {40, 9, -18}
, {-57, 15, 37}
, {-28, 96, 42}
, {-11, -14, -64}
, {-6, 119, -23}
, {14, -84, -79}
, {89, -44, 19}
, {113, 23, 8}
, {-66, -45, -98}
, {23, 72, -106}
, {48, 97, -91}
}
, {{-60, 60, 7}
, {50, -41, -43}
, {-22, -17, 49}
, {-104, -40, 67}
, {56, 42, -19}
, {15, -51, -78}
, {-23, 21, -81}
, {-13, -98, 43}
, {-52, 66, 68}
, {-45, -83, -56}
, {60, -85, 27}
, {23, 3, -6}
, {53, -4, 61}
, {85, -94, 1}
, {76, -32, 106}
, {-27, -28, 57}
}
, {{21, 92, 11}
, {82, 50, 17}
, {-94, -85, -91}
, {77, 85, 51}
, {29, -22, 49}
, {-56, -95, 76}
, {60, 80, 22}
, {-77, 81, 21}
, {-96, 21, -39}
, {31, 75, 0}
, {-62, 86, 18}
, {21, -19, -1}
, {2, 47, -2}
, {-69, -61, 38}
, {-39, -59, -19}
, {71, 105, -47}
}
, {{-98, 73, 99}
, {-14, 4, 3}
, {-73, -99, 46}
, {-31, 23, 27}
, {31, -39, 76}
, {-32, 17, 0}
, {57, 14, 11}
, {67, 32, 75}
, {-6, -17, -31}
, {3, -61, -56}
, {17, 15, -77}
, {-18, -49, 94}
, {-45, -102, -18}
, {-18, 29, 86}
, {-8, -13, 109}
, {26, -42, -26}
}
, {{-13, -11, -33}
, {28, 50, -45}
, {64, -40, -72}
, {60, 38, -73}
, {-35, -54, 38}
, {-42, 66, 24}
, {10, 87, 41}
, {-101, -13, 51}
, {38, 13, -6}
, {98, -74, -68}
, {-100, 104, 77}
, {-101, -20, -94}
, {-56, 6, 65}
, {-77, 28, -30}
, {-16, 44, -5}
, {-46, 29, 32}
}
, {{12, -94, 85}
, {-87, -42, -45}
, {-25, -74, 77}
, {-22, -38, 68}
, {16, 31, 72}
, {17, -61, -74}
, {23, -52, -26}
, {36, -6, -5}
, {-26, 25, -83}
, {-33, -28, 98}
, {-45, 7, -69}
, {91, -57, 52}
, {-67, -85, -67}
, {6, -40, -4}
, {46, 66, 69}
, {-31, 72, -38}
}
, {{-31, -76, 99}
, {-69, -52, 115}
, {-71, -40, 102}
, {62, 64, -20}
, {96, 73, -87}
, {-71, -47, 48}
, {-49, -29, 57}
, {-34, -65, 22}
, {15, 27, 5}
, {-87, -30, -7}
, {-93, -22, 13}
, {96, 35, 30}
, {-82, -3, 24}
, {-75, -26, 83}
, {-57, -67, 12}
, {84, 94, 57}
}
, {{-55, 14, -66}
, {108, 101, 40}
, {8, -61, 57}
, {72, -30, -53}
, {-6, -18, -39}
, {31, -94, 32}
, {69, -21, 43}
, {-35, -90, -13}
, {-1, 20, -65}
, {4, 38, -23}
, {-18, 87, 5}
, {-32, 101, -30}
, {-76, 99, 95}
, {94, 35, 62}
, {-34, -51, -87}
, {-2, -45, -82}
}
, {{-35, -1, 42}
, {-80, 46, 100}
, {-3, -60, 107}
, {71, -37, -80}
, {14, -71, 111}
, {-48, 11, -46}
, {45, 25, 62}
, {-41, -40, 84}
, {32, -34, 37}
, {-28, 40, -83}
, {-36, -32, 78}
, {110, 15, -92}
, {-63, 21, -53}
, {-85, -96, -17}
, {0, 55, -65}
, {72, -50, 24}
}
, {{1, -21, -94}
, {-47, 80, 27}
, {79, -31, -48}
, {23, -6, -20}
, {76, -92, 78}
, {26, 97, 76}
, {-65, 96, 102}
, {83, -55, 15}
, {-63, 86, 37}
, {101, -15, -84}
, {-18, -70, 12}
, {71, -56, 93}
, {66, 94, 100}
, {26, 81, 92}
, {-58, -100, -14}
, {12, -98, 45}
}
, {{26, -62, -98}
, {39, 3, 98}
, {6, 25, -80}
, {-96, -85, -30}
, {22, -11, -58}
, {22, 6, -9}
, {103, 27, -85}
, {104, 34, 27}
, {-77, 56, 33}
, {-22, 29, 100}
, {-97, -63, -39}
, {-48, 95, 61}
, {45, -64, -14}
, {37, -33, -81}
, {35, 71, -83}
, {1, 56, -87}
}
, {{-28, 107, 81}
, {49, -102, 110}
, {99, 29, 60}
, {59, -5, -3}
, {83, -102, 41}
, {-64, 44, -57}
, {-24, -38, 81}
, {5, 101, -13}
, {100, 10, 96}
, {62, -22, -54}
, {81, -47, 74}
, {80, -23, 4}
, {-78, -71, -2}
, {-88, 29, -83}
, {-7, -96, -71}
, {-97, 48, 50}
}
, {{86, -94, 50}
, {2, 97, 65}
, {-36, 45, 80}
, {-72, 36, 52}
, {-87, -56, -76}
, {-12, 73, 58}
, {35, 54, -11}
, {-85, -76, 67}
, {-16, -11, 92}
, {92, 48, -20}
, {-4, -18, 101}
, {-22, -50, -97}
, {-76, 12, 98}
, {-89, -37, 16}
, {50, 93, 35}
, {47, 37, -44}
}
, {{-83, 95, 93}
, {-87, 83, 17}
, {38, -52, 36}
, {94, 105, 26}
, {2, -44, -7}
, {-70, -30, -52}
, {-30, 64, -44}
, {-26, -58, 0}
, {-63, -37, -17}
, {-68, 112, 63}
, {62, -29, -57}
, {65, 76, 14}
, {22, 101, -71}
, {90, -87, 73}
, {-32, 68, 83}
, {-32, -1, 81}
}
, {{-75, 5, 3}
, {66, -89, 49}
, {71, 21, -30}
, {27, -38, -31}
, {67, 61, -29}
, {-1, 33, -14}
, {-8, -10, 35}
, {67, 12, 78}
, {86, -45, 14}
, {-89, 95, 19}
, {-100, -99, -102}
, {35, -68, 65}
, {-101, 98, 26}
, {-56, -24, 66}
, {-62, 72, 45}
, {50, 41, 27}
}
, {{9, 8, 48}
, {-95, -41, -41}
, {37, 23, -49}
, {83, -86, 87}
, {11, -82, -75}
, {46, -61, 85}
, {-102, 9, 90}
, {-52, 77, -52}
, {57, -46, -13}
, {47, 34, -62}
, {15, 68, -64}
, {-99, -82, 10}
, {4, 5, 0}
, {-11, -6, 47}
, {-66, 81, -75}
, {97, 77, -44}
}
, {{-28, 37, 8}
, {42, 82, 73}
, {92, 69, 44}
, {-100, -13, -99}
, {15, 93, -36}
, {-61, -7, 16}
, {-38, -40, -49}
, {-64, 17, 43}
, {103, 90, 54}
, {19, -63, -90}
, {-11, -95, 44}
, {-51, 10, -83}
, {-95, 30, -60}
, {-82, 101, -24}
, {-80, 106, 58}
, {95, 32, -108}
}
, {{-64, 50, -85}
, {5, 44, 25}
, {109, -54, 71}
, {100, 12, 27}
, {-48, 10, 0}
, {-65, -51, -100}
, {-25, -83, -32}
, {-73, -27, 32}
, {-45, -106, 5}
, {73, 88, 0}
, {9, 42, 31}
, {101, -9, -71}
, {67, -8, -86}
, {109, 90, -70}
, {27, 55, 5}
, {-71, -28, 56}
}
, {{30, -85, -47}
, {28, 41, 50}
, {-30, -14, -46}
, {91, -31, 46}
, {69, -52, 58}
, {59, -103, 54}
, {82, -72, -82}
, {-3, 18, 12}
, {-29, -84, 108}
, {-23, 27, 98}
, {-50, 53, -63}
, {9, -45, 78}
, {1, 93, -45}
, {-3, 0, 100}
, {47, -85, -42}
, {43, -101, 74}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   199
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_7_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_7(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       99
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_6_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_6(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_6_bias[CONV_FILTERS] = {18, -5, 18, 10, 21, 0, 5, -2, -1, 10, 13, 12, -12, -1, 0, 12, -3, -6, 11, -5, 10, 6, -3, 11, -6, 0, -3, -9, 0, -6, 1, -6, 0, -4, 9, -14, 5, 0, 4, 13, -7, 8, -4, 0, -12, 5, 13, 12, 1, 3, -12, -2, 0, 1, 3, 15, 16, 4, -5, 17, 0, 16, 2, 7}
;

const int16_t conv1d_6_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{36, 78, 56}
, {14, -5, -22}
, {-28, -19, 58}
, {19, -16, -65}
, {21, -23, -17}
, {10, 39, 67}
, {-77, -60, -35}
, {19, -17, 53}
, {-46, 51, -14}
, {-2, 49, 68}
, {60, -53, -9}
, {29, -58, 0}
, {24, -37, 28}
, {-27, 71, 27}
, {57, 81, -48}
, {2, 31, -71}
, {40, 59, 6}
, {-15, 8, -8}
, {-52, 44, 46}
, {43, -52, -73}
, {79, 76, 50}
, {-47, -22, -54}
, {-47, 6, -29}
, {15, 63, 54}
, {-59, 58, -42}
, {-45, 38, 13}
, {37, 31, 85}
, {51, -52, -4}
, {-16, -10, -77}
, {-34, -23, 18}
, {15, -23, -9}
, {35, 53, 28}
}
, {{37, 29, -45}
, {53, 42, -9}
, {-42, 12, 31}
, {-13, 55, -70}
, {19, -43, -31}
, {19, 57, -28}
, {-33, -62, 62}
, {9, -55, 30}
, {15, -11, 23}
, {53, 12, 58}
, {31, 53, 8}
, {23, 59, 63}
, {-34, 52, -28}
, {27, 56, -57}
, {2, 56, -1}
, {17, -31, -11}
, {-50, -2, 12}
, {22, -37, 33}
, {9, -15, -29}
, {30, 30, -26}
, {1, 4, -40}
, {-46, 10, 26}
, {15, -15, 37}
, {-71, -16, 43}
, {-16, -46, 8}
, {47, -47, 18}
, {-22, 48, 3}
, {-13, 9, 73}
, {24, -21, 73}
, {-48, -72, -35}
, {5, 49, -7}
, {66, -68, -25}
}
, {{28, -21, 86}
, {-54, 77, 47}
, {71, -35, 64}
, {-23, 7, -58}
, {20, 6, 17}
, {-43, 59, 78}
, {5, 20, 25}
, {-54, 53, -26}
, {-3, -43, 56}
, {-48, 49, -59}
, {34, 2, -34}
, {6, 86, 43}
, {-72, -20, -53}
, {75, -65, -49}
, {32, 37, -44}
, {-15, -46, -17}
, {-52, 63, 9}
, {75, -13, -13}
, {2, 18, -74}
, {-57, -27, 2}
, {79, -30, 53}
, {-72, 24, -60}
, {-46, -16, -21}
, {-22, -47, 69}
, {43, 5, -62}
, {-33, 55, -26}
, {-6, -21, -35}
, {-9, -57, -10}
, {-5, 13, 5}
, {44, -58, -45}
, {-41, 55, 46}
, {-52, 23, -44}
}
, {{63, 28, -47}
, {-15, 72, -23}
, {38, 67, -9}
, {-32, -50, 43}
, {72, 53, 20}
, {-25, 14, 30}
, {-42, -14, 87}
, {-55, -50, 39}
, {68, -6, 15}
, {15, 77, 13}
, {-11, 62, 21}
, {43, -63, 23}
, {1, 89, 7}
, {-64, 26, 25}
, {-28, -56, -54}
, {65, 61, 20}
, {-5, 34, 30}
, {6, 65, 44}
, {-25, 53, 10}
, {-57, 71, -26}
, {51, -78, 69}
, {9, -34, 21}
, {45, -60, 30}
, {-29, 20, 16}
, {-29, -15, -56}
, {20, -38, -53}
, {-32, -2, 49}
, {-35, 10, 68}
, {25, -50, -33}
, {26, 15, -63}
, {14, 11, -6}
, {55, 39, 0}
}
, {{-6, 68, -9}
, {36, 21, 13}
, {79, -35, -55}
, {-59, 23, -56}
, {42, 18, -25}
, {2, -42, -61}
, {61, -26, 22}
, {16, -62, 28}
, {35, -38, 81}
, {58, 44, -15}
, {42, -7, -36}
, {3, 11, 60}
, {34, -7, -16}
, {72, -32, -6}
, {18, -23, 89}
, {31, 5, -66}
, {33, 46, 5}
, {44, 22, -3}
, {5, 48, -40}
, {22, -2, -31}
, {-25, 51, -41}
, {-66, -58, -16}
, {31, -53, -17}
, {65, -6, -20}
, {-34, -11, -16}
, {-51, -57, 39}
, {-35, 28, -3}
, {0, 55, 52}
, {-32, 0, 24}
, {-39, -58, -24}
, {-22, 0, -11}
, {-9, -5, 56}
}
, {{37, 50, 3}
, {-50, 58, 71}
, {-22, 72, -45}
, {29, 57, -30}
, {65, 41, -21}
, {6, -5, 45}
, {33, 10, 73}
, {-66, -1, 45}
, {72, 34, -23}
, {-17, 8, 63}
, {14, -25, 30}
, {-14, 61, -27}
, {-25, -17, 25}
, {51, -4, -15}
, {55, 53, -3}
, {55, 20, -27}
, {42, -32, 20}
, {-77, -49, 47}
, {31, -18, -16}
, {0, -19, 3}
, {-28, -36, 30}
, {69, 37, -44}
, {-67, 44, 50}
, {-6, 20, 70}
, {-75, 28, 18}
, {-67, 68, 20}
, {13, 34, 38}
, {0, -62, -70}
, {-72, -17, -37}
, {-27, -60, 45}
, {32, -39, -9}
, {1, -51, 38}
}
, {{-16, -12, 53}
, {1, 29, 41}
, {-10, 56, 57}
, {57, -61, -66}
, {73, 57, 77}
, {69, -9, 37}
, {56, 24, -36}
, {-49, -6, 35}
, {-4, -37, 29}
, {-4, -60, -27}
, {48, 0, 3}
, {74, 22, -59}
, {72, -47, -52}
, {63, -61, 27}
, {54, -37, 18}
, {-47, 71, 40}
, {68, -63, -3}
, {35, 35, 54}
, {20, -35, 27}
, {-15, 31, 54}
, {-10, 57, -60}
, {63, -56, 28}
, {25, 63, 26}
, {-12, -39, 75}
, {20, -32, 39}
, {26, 13, 30}
, {-1, 6, 20}
, {23, 35, -41}
, {26, -58, 8}
, {-6, -70, 68}
, {-40, 37, 3}
, {-42, -55, 4}
}
, {{45, -45, 57}
, {60, 8, 64}
, {-36, -39, -28}
, {24, 37, -18}
, {-28, -27, -70}
, {27, -73, -54}
, {23, -18, -24}
, {-59, 28, -47}
, {-51, -32, -27}
, {41, -22, 8}
, {-16, -4, -22}
, {61, -48, 42}
, {-34, -27, 32}
, {-31, 43, -7}
, {-10, 35, 67}
, {-20, -21, -22}
, {49, 6, -36}
, {0, -50, -40}
, {-42, -19, -21}
, {44, 72, 61}
, {-53, 43, 51}
, {25, -70, 47}
, {12, -8, -4}
, {16, 41, -14}
, {44, 6, 62}
, {41, 12, -19}
, {67, -65, -62}
, {-50, 22, 27}
, {47, -71, 26}
, {7, -64, 5}
, {23, 54, 6}
, {-53, -28, 72}
}
, {{58, 16, 58}
, {48, 52, -67}
, {-53, 24, 8}
, {-71, -32, -15}
, {-7, 38, 35}
, {7, 58, 17}
, {70, 44, -70}
, {-21, 64, 27}
, {-42, 59, -52}
, {34, -67, 48}
, {-75, -54, -75}
, {46, -48, 48}
, {20, -33, 27}
, {13, 28, -51}
, {-21, 65, 59}
, {14, -22, -8}
, {46, 56, -72}
, {-7, -39, -14}
, {-46, -12, 56}
, {87, 24, 77}
, {28, -4, 29}
, {50, -16, -8}
, {21, -5, 49}
, {-41, 79, -37}
, {-63, 43, 58}
, {43, -34, -17}
, {-12, 74, -39}
, {65, -37, 4}
, {44, -55, -46}
, {61, -23, 43}
, {-54, 30, -60}
, {-15, -29, 48}
}
, {{45, -5, -17}
, {8, -55, 71}
, {-61, 11, 37}
, {-75, -49, 5}
, {28, -60, -48}
, {60, -31, 86}
, {1, -46, 52}
, {-25, -70, -53}
, {81, 83, 30}
, {60, -5, 26}
, {20, -4, -51}
, {-51, -22, 52}
, {50, -35, 58}
, {1, -54, -33}
, {-63, -55, 14}
, {-36, -64, -33}
, {-7, -53, -65}
, {46, 11, -32}
, {-1, 28, 36}
, {-33, 6, 36}
, {-63, 65, -11}
, {43, 46, -56}
, {-30, -62, 57}
, {-56, -70, -1}
, {7, 39, -19}
, {64, -13, 37}
, {66, 23, -32}
, {11, -49, -33}
, {-22, 17, -15}
, {54, -7, 30}
, {83, -4, 2}
, {-42, 48, 6}
}
, {{-61, -65, -54}
, {-29, 16, 26}
, {16, 25, 26}
, {-36, -51, -65}
, {8, 64, -42}
, {-53, -36, -39}
, {-13, 38, 17}
, {-65, -32, -48}
, {-48, 70, 44}
, {86, -35, -26}
, {-1, -46, 44}
, {-58, 12, -24}
, {-45, -49, -45}
, {-43, 16, 22}
, {50, 52, 4}
, {-29, -40, 63}
, {2, -57, 66}
, {7, 2, 82}
, {-38, 22, -2}
, {13, -59, -63}
, {57, 89, -50}
, {68, 15, -51}
, {2, 22, 17}
, {54, 12, -55}
, {-6, 23, 27}
, {39, -40, -42}
, {44, 29, 80}
, {60, 65, 51}
, {-57, 8, -20}
, {41, 60, -61}
, {9, 67, -6}
, {73, 50, -54}
}
, {{6, 64, 52}
, {-52, 23, 5}
, {-41, 54, -30}
, {4, -42, 59}
, {17, 24, -40}
, {-53, -36, -15}
, {3, -30, 15}
, {-66, -64, 45}
, {-43, 8, -29}
, {25, 26, -11}
, {49, 43, -9}
, {-66, 53, 22}
, {25, 11, -57}
, {32, -71, -42}
, {12, -40, -59}
, {50, 20, -66}
, {12, 32, 68}
, {48, -38, 69}
, {-75, 33, -64}
, {-68, -65, 86}
, {-8, -44, 47}
, {-76, -18, 44}
, {-35, -50, -42}
, {-14, 69, -38}
, {-47, 17, -27}
, {-16, 1, 60}
, {-51, 44, -59}
, {7, -10, 64}
, {-29, -49, 40}
, {56, -6, 21}
, {16, -35, 27}
, {70, -47, 77}
}
, {{-63, -55, -32}
, {65, 44, -16}
, {45, 54, -47}
, {55, -21, -41}
, {42, -31, -41}
, {1, 37, 62}
, {-20, 28, 21}
, {-45, 66, 81}
, {-66, -16, -5}
, {64, 20, -21}
, {-8, -62, -67}
, {-83, -57, -54}
, {48, -27, -16}
, {-55, 61, -55}
, {-23, 71, 7}
, {19, 53, -64}
, {44, 27, -37}
, {9, 55, 13}
, {26, 33, -38}
, {-3, -63, -3}
, {-5, 70, -36}
, {-80, 7, -57}
, {67, 11, 67}
, {-56, -19, -29}
, {-36, 42, 45}
, {7, 63, -18}
, {27, -14, -66}
, {25, -11, 14}
, {-31, 35, -22}
, {14, 11, -55}
, {54, 3, -15}
, {51, -52, 44}
}
, {{-32, 16, 67}
, {-56, 23, 36}
, {-38, 26, 20}
, {-61, 36, 55}
, {12, -25, -65}
, {17, -29, -41}
, {-46, 50, 73}
, {66, -46, 77}
, {-48, 38, -79}
, {-40, -47, -52}
, {18, -49, 1}
, {-13, -41, 63}
, {57, -50, -71}
, {86, 29, 32}
, {72, -19, 33}
, {0, 17, -39}
, {5, -56, -41}
, {-24, 28, -50}
, {-52, 41, -44}
, {13, -44, 0}
, {-68, -54, 55}
, {1, 19, -62}
, {59, -59, 27}
, {22, -66, -67}
, {-6, 64, 24}
, {-48, 40, -9}
, {67, -56, 29}
, {18, -56, 53}
, {68, 64, 4}
, {-49, 4, -41}
, {70, -59, 66}
, {-46, -16, 88}
}
, {{55, -46, -76}
, {55, 53, 34}
, {11, -39, 19}
, {-35, 36, -12}
, {28, -30, -45}
, {-30, -62, 0}
, {38, -19, 39}
, {-62, -68, 14}
, {4, -59, -43}
, {11, 61, -62}
, {62, -16, -29}
, {-14, -52, 29}
, {45, -21, 18}
, {55, 22, -40}
, {-37, -12, 54}
, {-74, 51, -40}
, {-36, -30, 69}
, {0, 6, -31}
, {-15, -55, -40}
, {-70, -60, 47}
, {-55, 42, -42}
, {7, 43, -62}
, {-44, -8, -72}
, {-60, -50, -29}
, {66, -26, -24}
, {-55, 60, 32}
, {-51, -11, -24}
, {-38, 16, -30}
, {57, -57, 18}
, {-28, 64, 72}
, {14, 44, -45}
, {-66, 54, 41}
}
, {{20, -9, 56}
, {-39, -57, -52}
, {19, 45, 10}
, {-11, 10, 34}
, {-57, -43, 26}
, {12, -47, -54}
, {27, 71, 26}
, {-39, 37, 8}
, {22, 44, 26}
, {-38, -58, 58}
, {-44, 15, 20}
, {47, 7, 60}
, {-30, -48, -62}
, {34, -12, -59}
, {2, -28, -27}
, {-59, 65, -12}
, {53, 9, 65}
, {-60, 39, -43}
, {67, 53, -5}
, {-5, -15, 74}
, {-32, 7, -4}
, {78, 4, -19}
, {-33, -27, -44}
, {52, -17, 24}
, {30, 9, -18}
, {62, -18, -31}
, {-31, 79, 0}
, {-3, -20, 20}
, {-32, -60, 71}
, {63, 57, -59}
, {-32, -49, 27}
, {-40, -17, -38}
}
, {{40, -19, -12}
, {-60, -36, -13}
, {-18, -31, 42}
, {44, -10, 33}
, {-71, 8, -2}
, {39, 35, -25}
, {-74, 17, -1}
, {28, -5, -25}
, {-2, 36, -53}
, {-36, -24, -54}
, {18, -34, 0}
, {46, -7, -2}
, {-64, 44, -26}
, {-29, -12, 15}
, {-20, 45, 21}
, {-21, -68, 44}
, {10, -13, -52}
, {-71, 42, -67}
, {40, 12, -50}
, {-74, 37, -54}
, {-69, 56, 52}
, {43, 7, 28}
, {-23, -44, -19}
, {-44, 68, 40}
, {-37, -2, -2}
, {-55, -72, -18}
, {43, 18, 54}
, {-43, 57, -21}
, {51, -40, 62}
, {-15, -44, 15}
, {-48, 61, -45}
, {-35, -60, -30}
}
, {{-64, 68, -62}
, {-18, 57, 62}
, {-1, 51, -13}
, {-23, 57, 36}
, {62, -30, -40}
, {44, -43, -39}
, {-29, 11, -67}
, {58, -18, -1}
, {-47, -39, -35}
, {59, -55, -33}
, {4, 69, 39}
, {-72, 55, -45}
, {31, 83, 57}
, {-49, -42, 9}
, {-72, -73, 33}
, {49, 0, 29}
, {-36, 48, 55}
, {47, 59, -41}
, {68, 20, -5}
, {-68, 31, 42}
, {25, -24, -57}
, {-17, -73, -4}
, {5, 45, 5}
, {-34, -4, 30}
, {46, -30, 55}
, {60, 61, 1}
, {-17, -50, -66}
, {65, -23, -57}
, {22, 31, 52}
, {63, -26, 19}
, {8, 25, -52}
, {60, -31, 43}
}
, {{-50, -41, -16}
, {15, -1, -32}
, {-24, 49, -37}
, {-53, -7, 68}
, {15, 15, 12}
, {0, -57, -27}
, {-40, 61, 47}
, {-66, 32, 5}
, {17, 52, -33}
, {47, 17, 57}
, {10, -42, -60}
, {-28, -53, -30}
, {64, 58, 8}
, {53, 35, -62}
, {-58, -30, 16}
, {-34, 6, 55}
, {66, 18, 76}
, {-1, -34, -22}
, {51, -23, 56}
, {13, -10, 72}
, {46, 16, 42}
, {41, -26, -67}
, {-61, -9, -67}
, {34, 22, 65}
, {31, 5, 51}
, {58, -50, 7}
, {-11, -52, 21}
, {-32, -38, 24}
, {60, -47, 56}
, {50, 78, 50}
, {-34, 61, 68}
, {25, -1, 25}
}
, {{32, 42, -63}
, {29, 18, -40}
, {4, 52, 42}
, {-40, 63, -38}
, {-5, -22, 51}
, {-20, 0, -55}
, {50, 16, -44}
, {0, -3, 26}
, {43, 48, 73}
, {-34, 63, 38}
, {-2, 32, -6}
, {63, 6, -15}
, {0, -57, -25}
, {-6, -62, 46}
, {-13, 13, -51}
, {57, 66, 2}
, {-45, 48, -3}
, {3, -9, 69}
, {10, -70, -38}
, {36, -9, 10}
, {7, 38, 16}
, {18, 2, 68}
, {41, 48, -48}
, {63, 34, -72}
, {32, -47, 65}
, {7, 18, 56}
, {34, -25, 27}
, {7, -9, -37}
, {-39, -24, -63}
, {66, 58, -52}
, {-2, 76, -22}
, {11, 74, 7}
}
, {{-1, -40, 56}
, {-9, -65, 0}
, {-8, -16, 47}
, {-61, -82, -84}
, {-80, 33, 20}
, {-54, -22, -59}
, {-32, 28, -1}
, {0, -32, -30}
, {-18, -7, 40}
, {-71, 1, 44}
, {-38, -34, -35}
, {-54, -23, -24}
, {50, -81, -67}
, {24, -48, 22}
, {67, -50, 23}
, {-1, 32, -35}
, {25, 55, 0}
, {-37, -66, -10}
, {-46, 21, 7}
, {3, -28, -11}
, {-64, -52, 57}
, {0, 62, 61}
, {-60, -39, 50}
, {22, -32, -61}
, {65, 69, 3}
, {-57, 40, -40}
, {56, 30, 17}
, {65, -25, 25}
, {77, 23, -37}
, {69, 52, -39}
, {-4, -26, -68}
, {43, -68, 34}
}
, {{27, -71, -7}
, {18, 55, -21}
, {8, -29, -6}
, {24, -12, -44}
, {10, -25, 17}
, {-6, 29, 32}
, {-72, -38, 3}
, {42, 5, 14}
, {56, 11, 12}
, {-41, -1, 18}
, {-46, -55, 64}
, {2, 0, 6}
, {-36, 51, -45}
, {-61, -51, -38}
, {49, -20, -7}
, {29, 68, -55}
, {-60, 27, 41}
, {-46, 54, 30}
, {36, 39, -43}
, {12, -1, 45}
, {-54, -31, 57}
, {-37, -4, 45}
, {0, 62, 41}
, {-28, 59, -7}
, {58, -73, 27}
, {44, 19, 4}
, {-23, -68, 19}
, {45, -43, -24}
, {-25, 37, -41}
, {-52, -13, -30}
, {-24, 19, 33}
, {-40, -51, -39}
}
, {{47, 5, 53}
, {-67, -42, -35}
, {55, -70, 50}
, {-56, -28, -29}
, {-13, 62, 62}
, {9, -33, -17}
, {-55, 41, -30}
, {0, -67, -73}
, {-2, -39, 4}
, {66, -52, -27}
, {-9, -59, -24}
, {-68, 62, 44}
, {25, -14, 2}
, {36, -42, -24}
, {52, -11, 47}
, {72, 65, -50}
, {21, -30, -26}
, {-4, 13, 9}
, {55, 48, 34}
, {8, -57, -13}
, {60, -25, -63}
, {57, 5, -2}
, {16, 67, 0}
, {28, -66, 4}
, {-46, -54, -14}
, {-2, -25, 67}
, {-21, 50, -42}
, {68, 56, -55}
, {-38, -18, 61}
, {48, 15, -49}
, {62, 39, -62}
, {49, -69, 6}
}
, {{0, -76, -5}
, {25, -65, 10}
, {63, 10, -1}
, {0, 42, 0}
, {-43, 41, 80}
, {-40, -13, -18}
, {-8, -62, -56}
, {8, -52, 64}
, {-32, 60, 0}
, {-34, 32, 65}
, {59, 4, -36}
, {50, -75, 16}
, {-32, -49, -5}
, {49, -71, 56}
, {-24, 55, 21}
, {-39, -42, -46}
, {-19, -19, 57}
, {44, -1, 27}
, {34, -69, 2}
, {-37, -17, 5}
, {-71, -54, -33}
, {-56, 27, 56}
, {18, 48, 49}
, {-58, 0, 13}
, {-60, -58, 71}
, {47, -28, -29}
, {36, 11, 72}
, {-44, -49, 36}
, {39, -22, -20}
, {-10, -41, 71}
, {73, -55, -68}
, {29, -17, -67}
}
, {{7, -10, -22}
, {35, 25, 58}
, {7, 60, 47}
, {-64, -5, -9}
, {-33, 69, 74}
, {-18, 14, -12}
, {63, 27, 36}
, {-19, -19, 30}
, {-74, -44, -69}
, {36, -31, -62}
, {5, 40, 4}
, {-36, 67, 29}
, {8, 47, 86}
, {-33, 65, 33}
, {-83, 38, -17}
, {-25, 55, 24}
, {-23, -53, 8}
, {-64, -16, 25}
, {59, 9, -54}
, {5, -78, 30}
, {-24, -77, -69}
, {-62, -42, 54}
, {-30, -16, -61}
, {-43, -56, -62}
, {31, -49, -49}
, {36, 67, -14}
, {-59, -39, -31}
, {-13, -24, -15}
, {16, -31, -27}
, {-3, 10, -49}
, {-67, -48, -38}
, {-41, -34, -69}
}
, {{2, 0, -12}
, {-52, 25, -10}
, {3, 19, -37}
, {-32, 11, -55}
, {-59, 64, -71}
, {-63, 54, -39}
, {-59, 51, -28}
, {1, -42, -54}
, {-5, 38, 13}
, {-64, 26, 45}
, {-17, -5, -14}
, {-28, 35, -31}
, {46, -73, 54}
, {-69, 3, 39}
, {-29, -23, 28}
, {4, -23, -8}
, {-39, -58, -26}
, {20, -71, -71}
, {68, 3, 24}
, {23, -68, 48}
, {50, -11, 4}
, {34, 2, -13}
, {5, -64, -54}
, {19, -56, -68}
, {-68, -2, -41}
, {71, -45, 41}
, {23, -31, 27}
, {-65, -54, -13}
, {-41, 13, -29}
, {50, -9, -27}
, {33, 20, 60}
, {-38, -41, 37}
}
, {{65, 48, -21}
, {52, -2, 52}
, {-25, -13, 49}
, {-45, -41, 4}
, {-25, -58, 33}
, {-31, 45, 43}
, {7, 71, 7}
, {56, -64, 32}
, {10, -10, -28}
, {7, -70, 46}
, {-61, -4, 70}
, {-13, -62, 58}
, {54, -47, 19}
, {19, 4, 24}
, {-9, -6, -19}
, {58, 65, 59}
, {-12, 50, 32}
, {48, 49, 34}
, {23, 55, 19}
, {25, -27, 50}
, {-26, 33, 12}
, {-72, -80, 13}
, {2, 38, -68}
, {5, -39, 17}
, {-42, -37, -53}
, {-33, 1, 40}
, {-13, 45, -27}
, {58, -32, 47}
, {65, 47, -2}
, {-20, 70, -48}
, {-81, 10, -31}
, {-79, 25, -11}
}
, {{8, 14, 25}
, {74, 31, 69}
, {79, 46, 50}
, {16, -71, -5}
, {-50, 56, -3}
, {-48, 60, 7}
, {-34, -7, 21}
, {-2, -9, -58}
, {-74, -16, 5}
, {8, 9, -72}
, {-20, 77, -59}
, {34, 36, 9}
, {7, -84, 5}
, {-27, -60, -84}
, {83, 2, 2}
, {12, -12, 52}
, {-26, -35, 12}
, {-78, 53, -36}
, {-42, -27, -46}
, {32, -42, 7}
, {-51, 41, 67}
, {51, -27, -43}
, {25, 56, -58}
, {-33, 19, 32}
, {-22, -27, 24}
, {-14, -67, 9}
, {-44, 66, -4}
, {28, 58, 4}
, {12, -47, 19}
, {-61, 52, 58}
, {32, 27, 0}
, {-4, -27, -33}
}
, {{31, 38, -68}
, {-69, 46, 29}
, {-16, -4, 4}
, {-14, 45, -19}
, {82, -54, -8}
, {-14, 21, -12}
, {-35, -8, 9}
, {-21, 12, -48}
, {8, 37, 23}
, {-48, -10, -72}
, {-28, 15, -46}
, {-11, 53, -28}
, {75, -55, 45}
, {37, -59, -21}
, {-46, -53, 35}
, {-47, 22, 57}
, {32, -25, 46}
, {51, 72, -53}
, {53, -24, 0}
, {-55, 42, 66}
, {12, 17, -12}
, {35, -82, 44}
, {37, 71, -35}
, {61, -30, 11}
, {47, 51, -73}
, {-19, 43, 54}
, {47, 35, 47}
, {-63, 34, 15}
, {44, 28, 10}
, {45, -48, -49}
, {40, -6, 14}
, {2, -39, 23}
}
, {{-37, -60, 67}
, {-20, 27, -37}
, {-47, -22, -46}
, {-3, 30, 50}
, {-33, -28, 15}
, {-64, 36, -5}
, {22, 53, 23}
, {11, 20, -46}
, {-55, -82, -78}
, {-29, -12, -50}
, {64, 19, 33}
, {63, -31, -2}
, {-54, -48, -69}
, {77, 40, 16}
, {67, 44, 14}
, {-52, -77, 5}
, {-41, -24, 16}
, {-69, -49, -13}
, {17, 31, -22}
, {48, -44, 35}
, {-72, -39, 14}
, {50, -20, -65}
, {46, -7, 26}
, {-70, 35, 39}
, {-5, 64, 0}
, {51, -23, 69}
, {-27, -18, -31}
, {9, -10, 47}
, {-7, -70, 73}
, {-61, -38, 12}
, {-63, 10, 14}
, {46, -31, 58}
}
, {{18, 57, -30}
, {37, 44, -45}
, {-33, 42, 48}
, {31, -16, 42}
, {-64, -7, 79}
, {17, -13, 17}
, {-61, -15, -54}
, {67, 18, -26}
, {-23, 2, -9}
, {63, 23, -7}
, {-33, -26, -38}
, {39, -56, 55}
, {-10, -10, 6}
, {24, 53, 41}
, {63, 67, 41}
, {-8, -2, -62}
, {-1, 16, 7}
, {62, 5, 70}
, {-38, 66, 24}
, {6, 58, -53}
, {25, -31, -52}
, {-47, -58, 64}
, {3, 77, -59}
, {-40, -29, 24}
, {-6, 58, -46}
, {-22, -27, -50}
, {48, 35, -10}
, {18, -2, 55}
, {-3, -25, -7}
, {-21, -56, -2}
, {40, 3, 41}
, {-68, 12, -63}
}
, {{42, -21, 22}
, {56, -69, 59}
, {5, 40, 48}
, {72, 22, -8}
, {16, 5, 0}
, {-18, 3, 73}
, {-8, 3, -17}
, {38, -20, -34}
, {2, -35, 36}
, {44, 70, 58}
, {43, -24, 29}
, {-69, -56, -7}
, {54, 41, -53}
, {33, 60, -73}
, {-61, 77, -76}
, {-24, -63, 23}
, {1, 51, 0}
, {58, -58, 14}
, {26, -80, 35}
, {-70, 22, -54}
, {-72, 60, 4}
, {-66, -75, -13}
, {14, 73, -67}
, {44, -67, -67}
, {42, 51, -8}
, {-12, -14, -38}
, {-4, 62, -23}
, {-57, -67, 5}
, {-23, 44, -32}
, {62, 65, -46}
, {75, 25, -70}
, {45, -1, 40}
}
, {{64, -31, 9}
, {16, 17, -20}
, {-1, 58, -29}
, {23, -25, 21}
, {48, -5, 33}
, {-50, 48, 37}
, {40, -33, -69}
, {3, -22, 47}
, {-17, 72, 1}
, {-13, -29, -42}
, {-67, 46, 67}
, {-40, -32, -49}
, {56, 30, 70}
, {-35, -4, 14}
, {-48, -76, 50}
, {-74, -62, 44}
, {68, -4, 14}
, {17, 27, 38}
, {56, -57, 45}
, {2, 4, 58}
, {-26, -60, -24}
, {63, 13, 61}
, {-56, -2, 33}
, {-16, 53, -20}
, {-54, 59, 53}
, {68, -25, 16}
, {-31, 1, 34}
, {-15, -54, -14}
, {-49, -7, 24}
, {-70, -60, 23}
, {35, -14, 70}
, {-70, -60, -29}
}
, {{36, -61, 23}
, {-11, -12, 35}
, {62, 62, -9}
, {-62, -46, 72}
, {-19, 29, -52}
, {-65, -67, 46}
, {0, -76, -72}
, {21, 67, 7}
, {15, -62, 63}
, {-40, 33, 28}
, {-65, -40, 7}
, {67, 41, -57}
, {-74, 28, -24}
, {23, -42, 47}
, {-16, 53, -47}
, {20, -33, -34}
, {66, 39, 25}
, {-26, 10, -12}
, {-45, 42, 68}
, {15, 43, 17}
, {55, -63, 65}
, {-36, 31, 17}
, {14, -74, -32}
, {56, 22, 2}
, {60, -23, -57}
, {-3, -24, -71}
, {22, 7, -30}
, {8, 69, 28}
, {-18, -10, -71}
, {-61, 10, -9}
, {-42, -43, 46}
, {-10, 24, -50}
}
, {{-14, -20, -47}
, {33, -67, 3}
, {53, -55, 4}
, {28, 16, 59}
, {-20, 11, 27}
, {-43, -25, -9}
, {35, 16, -33}
, {64, 7, 25}
, {-19, 11, 18}
, {-17, -40, -50}
, {19, 48, 62}
, {7, 28, 1}
, {34, -64, 34}
, {-66, 19, 1}
, {-72, 19, -30}
, {22, 52, -42}
, {-50, 26, -30}
, {-53, 4, 1}
, {8, 58, -69}
, {-38, 20, 40}
, {15, 42, 52}
, {51, 55, 25}
, {-11, 34, 68}
, {48, 16, -55}
, {-5, -27, -4}
, {-52, 25, 28}
, {-46, 32, 22}
, {-67, 5, -10}
, {-35, -27, 29}
, {-11, -38, -24}
, {-40, -21, 63}
, {-59, 68, 0}
}
, {{18, -16, 18}
, {-19, 22, 59}
, {71, 3, -69}
, {37, -52, -58}
, {55, 70, -60}
, {-36, -24, 22}
, {32, -41, 12}
, {17, 14, 0}
, {-10, -78, 17}
, {-58, 14, 69}
, {-42, 0, 47}
, {-67, -53, -21}
, {30, 22, 69}
, {64, -4, 8}
, {7, -17, 44}
, {34, -14, 43}
, {44, 21, -22}
, {71, -76, 79}
, {31, -20, 84}
, {51, -35, -81}
, {50, -67, 40}
, {-83, 58, 4}
, {-32, -12, -75}
, {-44, -52, -80}
, {-61, -34, -20}
, {35, 38, -13}
, {-32, -32, -68}
, {66, 53, 2}
, {74, 41, 10}
, {-25, 7, 58}
, {-6, 35, 56}
, {56, 38, -42}
}
, {{-17, -12, 51}
, {68, 37, 29}
, {52, 55, 58}
, {16, -27, -46}
, {8, 89, 60}
, {-10, 36, -10}
, {42, 57, -2}
, {-47, 7, 57}
, {7, 54, -22}
, {70, 69, 59}
, {-38, -42, -39}
, {64, -57, 60}
, {-9, -38, 45}
, {-13, 72, -22}
, {49, 23, 42}
, {40, 9, -4}
, {49, -56, -31}
, {32, -72, 3}
, {11, 42, 21}
, {85, -55, -1}
, {-31, -49, -58}
, {14, -24, -63}
, {-26, -23, 11}
, {-25, -42, 47}
, {34, -75, 33}
, {67, -10, 15}
, {5, -55, -40}
, {-67, -45, 61}
, {-65, -29, 79}
, {-7, -65, -68}
, {47, 62, -71}
, {60, 31, 47}
}
, {{-3, 66, 62}
, {-41, 47, 66}
, {-42, -48, -30}
, {15, -53, -55}
, {14, 39, 32}
, {15, 41, 22}
, {0, 21, 52}
, {17, 12, -63}
, {54, -54, -16}
, {62, -57, 60}
, {14, -35, 23}
, {15, 20, 39}
, {73, -58, 19}
, {-39, 74, -54}
, {1, -33, 2}
, {-22, -17, 73}
, {1, 14, -12}
, {50, 66, 47}
, {-40, -77, -67}
, {-3, 66, 66}
, {-77, -40, -63}
, {-2, 50, 6}
, {-1, -32, -71}
, {15, 31, -59}
, {15, -2, -32}
, {-69, -34, 37}
, {22, -65, -64}
, {50, -7, 64}
, {37, -70, -63}
, {69, 12, -58}
, {38, -25, 14}
, {-61, -34, 36}
}
, {{28, -61, -36}
, {-76, 39, -45}
, {-48, 65, 32}
, {0, -8, 30}
, {-2, -62, 49}
, {-9, 71, -4}
, {20, 22, 54}
, {-56, 20, 57}
, {13, 53, 67}
, {73, 63, -35}
, {-58, -43, 48}
, {-56, 55, 62}
, {-4, 75, 57}
, {0, 70, 23}
, {26, -80, -28}
, {-14, 73, -21}
, {57, 48, 62}
, {-16, -45, -46}
, {-30, -37, 71}
, {-44, 41, 49}
, {-7, -7, -13}
, {62, 21, 29}
, {-47, 55, -39}
, {-4, 28, -11}
, {-28, -17, -8}
, {-18, -9, -61}
, {-40, -39, 31}
, {-27, 48, 53}
, {20, -20, 30}
, {-7, 28, 69}
, {-32, 81, -2}
, {48, 26, 78}
}
, {{5, -29, -4}
, {75, -27, 23}
, {33, 27, -31}
, {-75, -63, -20}
, {-16, 58, -27}
, {-2, -57, -11}
, {5, -11, 11}
, {48, -38, -37}
, {53, 35, 80}
, {-47, 3, 6}
, {-24, -58, -35}
, {-45, 78, -22}
, {37, -58, 54}
, {65, -39, -62}
, {82, -7, -53}
, {48, 54, 67}
, {-34, -59, 20}
, {-6, -55, 20}
, {-81, -27, 39}
, {41, 51, 57}
, {70, 50, -1}
, {0, 17, 25}
, {29, 18, 62}
, {53, 84, 72}
, {47, 40, -40}
, {-41, 7, -44}
, {53, 76, 16}
, {5, -53, 6}
, {-9, -54, -63}
, {5, -21, 38}
, {60, 17, 47}
, {69, 41, 68}
}
, {{23, -30, 53}
, {-24, -13, 60}
, {29, 43, -48}
, {11, -33, 30}
, {-33, -36, 24}
, {58, -11, -26}
, {8, 3, -11}
, {20, -79, -52}
, {-4, -64, 42}
, {53, -5, 61}
, {-17, -45, -58}
, {-2, 63, 53}
, {64, -23, 25}
, {-11, 74, -39}
, {-17, -6, 42}
, {-54, 0, -41}
, {2, -15, -25}
, {-25, -40, -48}
, {39, -62, 58}
, {46, 17, 45}
, {14, 33, -64}
, {51, -49, 0}
, {34, -27, 60}
, {-28, 41, -50}
, {-27, 34, -51}
, {-25, 28, -18}
, {-18, -31, 60}
, {28, -6, 26}
, {-15, 31, -16}
, {41, 36, 32}
, {4, -10, 53}
, {-49, 14, -77}
}
, {{-33, -43, 21}
, {23, 42, -46}
, {3, -35, -50}
, {-68, 35, -66}
, {0, 68, 38}
, {34, 1, 50}
, {0, -26, -66}
, {-65, 58, 60}
, {30, 43, 26}
, {65, 40, 31}
, {13, -51, -40}
, {58, -65, -15}
, {90, 43, 44}
, {-9, 27, 40}
, {-54, -71, 39}
, {44, -5, 12}
, {2, -69, -13}
, {-53, -12, -47}
, {-26, -50, 40}
, {-5, 33, 39}
, {79, -38, 58}
, {62, 47, 82}
, {30, 10, 23}
, {76, 34, -52}
, {49, 70, -14}
, {-63, -76, 22}
, {34, -41, 25}
, {-51, 36, 9}
, {37, -17, -16}
, {-32, 74, 34}
, {71, 25, 22}
, {-20, 55, 43}
}
, {{-33, -3, 63}
, {-26, -29, 18}
, {16, 70, 6}
, {52, 61, 6}
, {21, 40, 38}
, {-62, 46, 22}
, {59, -29, 65}
, {50, -60, 43}
, {10, -53, -52}
, {-63, -27, -9}
, {63, 71, 43}
, {-35, 41, -65}
, {58, 0, -12}
, {67, 61, -49}
, {51, -61, -67}
, {-9, 37, 64}
, {52, -74, -8}
, {44, -63, -36}
, {3, 45, -27}
, {29, -35, 36}
, {-3, -8, 24}
, {-7, -9, 30}
, {0, -34, 68}
, {-12, 43, -41}
, {-59, 40, 25}
, {0, -51, 33}
, {58, 18, 9}
, {0, 53, 68}
, {51, 27, -69}
, {-31, 41, -14}
, {65, 56, -37}
, {5, 58, 48}
}
, {{66, 52, -64}
, {70, 43, -25}
, {46, -61, -32}
, {52, -25, -70}
, {-10, -25, -27}
, {70, 11, 15}
, {42, -35, 1}
, {-43, -32, -30}
, {-66, 45, -64}
, {-54, -25, 50}
, {27, 49, 12}
, {-70, 69, 52}
, {71, -54, -39}
, {-37, 60, -43}
, {-35, 13, 25}
, {-63, 66, 52}
, {40, 57, -12}
, {-54, -24, 0}
, {-47, -58, -37}
, {-11, 78, 39}
, {50, 29, -82}
, {-59, 43, -15}
, {-62, 22, -5}
, {-61, 45, -20}
, {-24, -1, -60}
, {71, -45, -55}
, {-40, -34, -51}
, {5, 72, 5}
, {-3, 76, 16}
, {46, -54, -8}
, {-36, 8, -16}
, {72, -69, 42}
}
, {{-24, 66, -62}
, {-8, -11, 1}
, {-28, 17, -17}
, {-27, -72, -35}
, {28, -29, 63}
, {45, 65, 36}
, {64, -60, 26}
, {48, 15, -4}
, {-9, -33, 20}
, {55, 60, 16}
, {3, 0, 24}
, {16, -6, -37}
, {-10, -40, 35}
, {60, -64, 62}
, {-17, -11, -44}
, {-34, 12, -64}
, {19, 57, 27}
, {-83, -5, 40}
, {55, -7, -63}
, {41, -83, -65}
, {-88, -53, -79}
, {-49, 2, -9}
, {-55, 0, -4}
, {17, 22, -39}
, {-38, -21, 22}
, {51, 17, 60}
, {-44, 46, -50}
, {9, -72, 30}
, {59, -24, 77}
, {-5, 15, 32}
, {62, 37, 4}
, {-57, -67, 22}
}
, {{-36, 36, 63}
, {-49, -38, 35}
, {-63, -13, -22}
, {37, 7, 38}
, {59, -23, -38}
, {-57, 35, 70}
, {-27, 65, 28}
, {-18, -52, -44}
, {33, -6, -27}
, {-54, -20, 28}
, {6, -22, 52}
, {-3, -25, -19}
, {16, 18, 35}
, {-75, 43, 71}
, {20, 73, 10}
, {-24, 28, 38}
, {78, -45, 49}
, {3, 56, -12}
, {17, -25, 52}
, {32, 63, 31}
, {-83, -60, -74}
, {-2, 13, 66}
, {-7, -59, 10}
, {1, 18, 74}
, {66, -64, -26}
, {-27, -25, 8}
, {12, 36, 85}
, {38, 67, -47}
, {0, 44, 70}
, {-3, -24, 19}
, {-16, -52, 65}
, {-50, 32, 5}
}
, {{-31, 27, 1}
, {37, -30, -50}
, {-24, -27, -62}
, {39, -66, 46}
, {0, 55, -55}
, {49, 57, -10}
, {54, 73, 69}
, {59, -77, 0}
, {50, 68, 52}
, {-32, -17, 62}
, {14, 4, -40}
, {41, -27, -26}
, {-28, -25, -74}
, {-66, 4, -33}
, {-5, 35, 9}
, {0, 48, -53}
, {-12, 32, 41}
, {81, -57, -25}
, {-57, 37, 25}
, {55, 7, 71}
, {23, -33, -25}
, {11, 8, 1}
, {-66, 7, -66}
, {-29, -22, -23}
, {-63, 52, -58}
, {29, -45, 65}
, {-25, -49, -2}
, {66, 59, -1}
, {0, -57, -66}
, {18, 68, 1}
, {-45, 68, 34}
, {-51, 30, -3}
}
, {{-29, -41, -45}
, {53, -28, 17}
, {58, -40, -26}
, {31, -62, -36}
, {55, -40, -74}
, {13, 18, -58}
, {-15, -12, 34}
, {73, 3, -5}
, {6, 53, 0}
, {-63, 53, -37}
, {31, -35, 14}
, {76, -2, 34}
, {23, 52, -29}
, {31, -79, 11}
, {-59, 14, 6}
, {43, 33, -55}
, {55, 48, 39}
, {66, -32, 43}
, {-17, 53, 59}
, {35, 68, 53}
, {53, -15, 0}
, {-23, -36, 3}
, {-37, 43, 33}
, {-17, 4, -46}
, {-11, -12, -40}
, {25, -25, 45}
, {47, 37, -64}
, {17, 59, 17}
, {54, 28, -36}
, {-13, -24, 61}
, {-3, -31, -18}
, {1, 27, 76}
}
, {{-28, -52, -65}
, {-16, -21, 63}
, {42, 15, -40}
, {35, 5, -51}
, {8, -79, -13}
, {-71, -17, 71}
, {-34, 67, 53}
, {-31, 41, -55}
, {38, -75, -67}
, {-52, 33, 26}
, {39, 66, 65}
, {5, 27, 22}
, {-88, -19, -35}
, {-28, 43, -42}
, {-56, 24, -26}
, {-37, 7, 10}
, {-36, 47, 8}
, {73, -38, 11}
, {33, -22, -57}
, {-60, 5, 48}
, {33, -20, 24}
, {21, 70, 30}
, {-67, -56, 37}
, {-55, -38, 65}
, {-38, 45, -71}
, {52, 63, -43}
, {57, 54, -75}
, {-9, -47, 46}
, {29, -8, 39}
, {-55, -38, 54}
, {-32, -50, 58}
, {-19, 31, 50}
}
, {{-4, 29, 29}
, {52, -27, 51}
, {69, 43, 6}
, {58, -70, 27}
, {-84, -37, 32}
, {42, 38, -63}
, {-13, -44, 24}
, {37, 49, 20}
, {37, 53, 40}
, {21, -60, 13}
, {29, 24, 13}
, {55, 60, 29}
, {-57, -87, 42}
, {34, 15, -25}
, {3, 65, -60}
, {0, 8, 1}
, {-33, -51, 47}
, {34, -29, 9}
, {-4, -11, -13}
, {-43, 32, -47}
, {-63, -40, -53}
, {-18, -24, -74}
, {-30, -65, -57}
, {-35, -12, 51}
, {64, -4, 2}
, {69, -3, 63}
, {-55, -41, -61}
, {-52, 18, -66}
, {-20, 28, 11}
, {69, 14, -24}
, {35, -31, -60}
, {74, -22, -61}
}
, {{-46, -20, -4}
, {56, -42, 39}
, {-45, 44, -56}
, {-3, -37, 1}
, {-62, -82, -41}
, {56, 42, -43}
, {-20, -6, 5}
, {-75, 32, -12}
, {-70, 29, 43}
, {38, 15, 7}
, {41, 33, 62}
, {-44, -35, -46}
, {-51, -73, -10}
, {-7, 57, -36}
, {-56, 30, -22}
, {-37, -46, -48}
, {-70, -21, 1}
, {44, -7, 49}
, {-48, 33, -25}
, {-10, -31, 5}
, {68, 29, 48}
, {-5, 0, -33}
, {7, 28, 30}
, {15, 22, 37}
, {22, 4, -10}
, {37, 40, -31}
, {-21, -59, 2}
, {-69, 65, 69}
, {7, 26, 26}
, {6, 0, 43}
, {81, 0, 25}
, {-4, -39, -22}
}
, {{18, -29, -42}
, {21, 1, 66}
, {-6, -14, 48}
, {15, 35, -34}
, {-35, -5, -38}
, {15, -22, -64}
, {-60, 13, -29}
, {-63, -64, -9}
, {-42, 42, -64}
, {-59, 54, 41}
, {-58, -18, 40}
, {11, -6, -10}
, {-54, 19, 7}
, {-18, 20, -21}
, {64, 62, -14}
, {-4, -43, -25}
, {5, -48, 10}
, {20, -44, 57}
, {57, 7, 34}
, {-72, -58, -20}
, {30, 31, 31}
, {-72, -33, -27}
, {-63, -52, 39}
, {17, 48, 8}
, {65, -63, 50}
, {-49, -35, -22}
, {27, 35, 39}
, {-72, 32, -5}
, {-8, -57, -16}
, {21, 54, -45}
, {39, 25, 1}
, {-44, 6, 33}
}
, {{-13, -6, -26}
, {-21, -46, 9}
, {26, 35, -21}
, {-37, -1, 14}
, {35, -42, -57}
, {-31, 70, 19}
, {-57, 44, 68}
, {22, 38, 60}
, {-8, 18, -33}
, {68, -10, -63}
, {-17, 47, -13}
, {-59, -34, -3}
, {60, -23, 34}
, {-52, 54, 37}
, {-51, -43, 2}
, {-49, -2, 9}
, {-32, -34, 44}
, {-50, 27, -61}
, {54, -61, 0}
, {44, 58, -57}
, {-66, 10, -55}
, {20, 14, 15}
, {-38, 28, -58}
, {-56, 41, 41}
, {61, -68, -21}
, {-26, 37, -2}
, {6, -44, -26}
, {-3, 5, -45}
, {-30, 38, -23}
, {65, -35, 45}
, {65, 6, 32}
, {3, -28, 57}
}
, {{-10, -42, 19}
, {60, 54, -51}
, {63, -50, -6}
, {-20, -35, -47}
, {-21, 19, -54}
, {63, -34, 1}
, {-55, 39, -35}
, {-25, 38, -8}
, {-83, -44, 8}
, {35, 0, 18}
, {-28, -34, -35}
, {-67, 11, 13}
, {-57, -27, 2}
, {-45, -4, 72}
, {-62, -64, 23}
, {10, -40, 21}
, {-65, 31, -16}
, {-46, -56, -35}
, {-6, -1, 12}
, {-50, 19, -18}
, {-72, -9, 5}
, {85, -13, 71}
, {79, -37, -3}
, {11, -62, -52}
, {-50, 25, 57}
, {35, -6, -52}
, {-9, -59, 24}
, {40, 26, -26}
, {-61, 22, 84}
, {53, 68, 10}
, {28, 56, -58}
, {24, -9, -15}
}
, {{0, 12, 28}
, {2, -6, 73}
, {15, 40, -18}
, {12, 35, -51}
, {54, -63, -46}
, {-22, -29, -69}
, {61, 48, -61}
, {-44, 48, -12}
, {-45, 49, -50}
, {51, 64, -17}
, {-24, -18, 34}
, {-58, 52, -22}
, {18, -73, -57}
, {-22, 23, -33}
, {-19, -20, 26}
, {-35, -62, 20}
, {41, -64, -71}
, {37, -66, -2}
, {-31, -28, 57}
, {45, 15, 12}
, {24, 36, 69}
, {-57, -77, 50}
, {28, -38, -55}
, {10, 11, -33}
, {-74, 25, -73}
, {-21, 66, 52}
, {65, -47, 14}
, {-12, -14, 48}
, {45, 45, 37}
, {48, -53, 2}
, {26, 65, -62}
, {61, 10, 76}
}
, {{-46, 64, 60}
, {71, -6, 50}
, {40, 65, -52}
, {57, 14, -9}
, {40, 82, 50}
, {5, -37, -22}
, {64, -65, -36}
, {24, -78, -9}
, {56, -36, -69}
, {32, 23, -80}
, {-33, -80, -33}
, {76, -68, -28}
, {1, 51, 13}
, {27, 35, 21}
, {-36, 58, 87}
, {71, -6, 54}
, {60, -34, 51}
, {-50, -44, -68}
, {-66, -68, -50}
, {-49, 29, 79}
, {-25, 93, 5}
, {57, -29, -16}
, {-6, -75, -34}
, {-2, 11, -53}
, {-35, -15, -20}
, {15, -67, 56}
, {-21, -37, 1}
, {-4, -40, 40}
, {3, -10, -45}
, {-49, 7, 27}
, {0, 29, -62}
, {28, -32, 32}
}
, {{37, 78, 29}
, {-23, 39, -5}
, {-4, -10, -76}
, {64, -75, 59}
, {31, 45, 42}
, {8, 36, -37}
, {-21, 2, -68}
, {-37, 45, 28}
, {10, 20, 68}
, {-26, 28, 44}
, {-11, 55, -54}
, {38, -9, 52}
, {-8, -64, -63}
, {67, -40, -60}
, {34, 47, 30}
, {-35, 6, -22}
, {-13, -41, 12}
, {-40, 64, 69}
, {-65, -36, -17}
, {-1, 25, 55}
, {-65, 50, 74}
, {14, -55, -23}
, {9, -38, 52}
, {-30, 17, 0}
, {21, 66, -2}
, {-21, -35, -69}
, {9, -43, -42}
, {65, -54, 40}
, {-49, -26, -4}
, {-5, 10, -54}
, {47, -68, -15}
, {6, -35, 4}
}
, {{23, -36, 34}
, {4, -29, -4}
, {-52, 28, 60}
, {44, -1, 13}
, {-27, 41, 71}
, {47, -38, -23}
, {14, -39, 31}
, {34, 1, -46}
, {38, -53, 63}
, {63, -29, -25}
, {-38, 10, -29}
, {25, 14, 4}
, {2, -35, 32}
, {21, -60, -49}
, {-34, 7, -2}
, {35, 62, 25}
, {-17, 42, 32}
, {-19, -14, 77}
, {-24, 70, -40}
, {37, 77, 9}
, {22, -4, -44}
, {32, 49, -65}
, {27, 48, 13}
, {46, 36, 55}
, {-52, -65, -76}
, {-77, -78, -25}
, {49, -19, -16}
, {-70, -15, -70}
, {27, -42, 64}
, {-1, -77, -39}
, {-59, -12, 65}
, {16, 0, 49}
}
, {{13, -55, 69}
, {-49, 43, -40}
, {-30, 4, -33}
, {53, -37, -64}
, {24, -50, 6}
, {-42, -65, 38}
, {-54, 33, -75}
, {-14, -19, -72}
, {18, 23, -56}
, {68, 0, 37}
, {5, 25, 60}
, {51, -11, 26}
, {-63, -27, -37}
, {-66, -66, 67}
, {-27, -1, -39}
, {-52, 19, 38}
, {24, -45, 32}
, {36, -58, -55}
, {-29, -65, 43}
, {0, -50, 38}
, {-45, -32, -8}
, {-44, 23, -40}
, {-54, -48, -50}
, {-28, -48, -1}
, {-56, 45, -60}
, {-58, -50, -48}
, {13, -12, -48}
, {-21, -40, -8}
, {-52, 72, -44}
, {38, 49, -72}
, {14, 26, -18}
, {42, 33, 39}
}
, {{34, -51, -36}
, {74, -44, -34}
, {-54, -64, -37}
, {-53, 51, -55}
, {9, 7, 65}
, {-54, 38, 42}
, {41, -49, 44}
, {-60, -3, 41}
, {15, 15, 75}
, {36, 68, 63}
, {-56, -13, -46}
, {-5, -18, 38}
, {-43, -29, 34}
, {26, 6, 66}
, {-42, -28, 30}
, {-22, 35, 23}
, {48, -63, 39}
, {-39, -11, 78}
, {-78, 52, 50}
, {-18, -1, 71}
, {40, 42, 81}
, {33, -25, -74}
, {-58, -20, -12}
, {-58, -55, -59}
, {67, -14, -64}
, {59, 7, -31}
, {3, 4, -33}
, {-32, 64, 62}
, {-48, 16, -16}
, {67, -23, 34}
, {-18, -16, 51}
, {-22, 45, 34}
}
, {{-32, -73, -47}
, {36, 29, -37}
, {-13, 28, 66}
, {47, -37, 61}
, {-16, -29, -6}
, {-25, -14, 6}
, {-45, 35, 46}
, {26, 12, 79}
, {19, -55, -22}
, {-41, 14, -1}
, {37, -52, -6}
, {-18, 17, 60}
, {-27, -12, -58}
, {44, -70, -18}
, {-29, -22, 56}
, {8, 27, -6}
, {64, 20, 21}
, {4, -61, 55}
, {29, 57, -43}
, {-10, -28, -30}
, {-62, 86, 57}
, {-51, -55, -54}
, {-44, 28, -1}
, {32, 27, -68}
, {26, 21, 72}
, {0, 66, 48}
, {44, -66, 24}
, {-19, 17, 21}
, {51, 39, -46}
, {8, -52, 28}
, {-11, 8, -72}
, {58, -25, 79}
}
, {{57, -70, 55}
, {-56, -75, -21}
, {-27, -22, -29}
, {-1, -40, 17}
, {33, -62, 75}
, {-33, -55, -18}
, {-69, -22, -46}
, {3, 53, 61}
, {37, -17, -55}
, {23, 40, -11}
, {12, -17, 48}
, {19, 79, 72}
, {-68, -20, -61}
, {62, -37, 41}
, {-55, 72, -56}
, {23, -50, 54}
, {29, 8, 5}
, {-66, -66, 23}
, {46, 27, -47}
, {37, 22, -30}
, {-38, -31, -33}
, {-26, -14, 5}
, {-52, 52, 69}
, {-19, 73, 44}
, {46, 25, 71}
, {2, 41, -66}
, {-39, -51, 28}
, {-11, 16, 42}
, {-43, -44, -36}
, {-56, 70, 6}
, {-88, 79, 40}
, {19, 49, 30}
}
, {{43, -58, -3}
, {-24, -57, 70}
, {-44, -44, -67}
, {-44, -60, 7}
, {-48, 56, -72}
, {67, 13, -31}
, {35, -49, -32}
, {39, -40, 20}
, {59, -37, 0}
, {38, 10, 63}
, {-60, -10, 16}
, {-51, -4, -16}
, {3, 40, 56}
, {17, 0, -9}
, {-38, 6, -35}
, {-7, -50, -31}
, {0, -35, -39}
, {-41, -1, 36}
, {52, 43, -11}
, {21, 31, 25}
, {49, -56, -61}
, {53, -68, 69}
, {-22, -50, 2}
, {18, 46, -51}
, {-50, 69, -37}
, {-21, 71, 54}
, {1, 24, 65}
, {28, -8, -1}
, {38, -6, -73}
, {-17, -23, -3}
, {68, 48, -32}
, {36, -49, 27}
}
, {{55, 53, 15}
, {-21, -1, 28}
, {-47, 57, 65}
, {-6, -32, 73}
, {-61, -14, -48}
, {-56, 61, -60}
, {-18, -24, 30}
, {68, 16, -72}
, {51, -46, -62}
, {4, -46, -7}
, {-17, 32, 32}
, {-30, -53, 75}
, {14, -50, 9}
, {29, 40, 61}
, {-10, -48, 14}
, {-26, -37, 72}
, {-20, 4, 15}
, {-32, 14, -60}
, {-18, 0, 17}
, {-9, 70, -5}
, {32, -86, 22}
, {-60, -42, -41}
, {-1, -76, -46}
, {-18, 32, 53}
, {-56, -27, 0}
, {80, -3, -55}
, {19, -22, -31}
, {-33, 47, -52}
, {-24, -14, -14}
, {35, 78, 64}
, {-32, -39, -46}
, {30, -48, 64}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   49
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_8_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_8(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       24
#define CONV_FILTERS        128
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_7_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_7(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    64
#define CONV_FILTERS      128
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_7_bias[CONV_FILTERS] = {10, 8, -2, 10, 16, -5, 7, 4, 6, -1, 3, -1, 21, 11, -2, -9, 7, 5, 1, 9, 7, -5, 2, -10, 3, 9, -4, -3, -3, 0, 21, -1, 8, 5, 0, 13, 7, 6, -1, -5, -4, -5, -4, 14, -5, 14, 2, -1, -2, 10, 0, -8, -5, -4, 0, -10, -10, 9, 0, -8, 13, 13, 10, -5, -4, 8, -11, 9, 0, 14, 14, -7, 5, -6, -9, 14, 18, 14, -1, -3, -2, 10, 2, 14, 16, -12, -3, -2, 5, -5, -3, 3, 11, 0, 2, -8, -11, -3, 2, -2, 18, -5, 8, -5, 20, 10, 6, -1, 5, 1, -1, -2, -8, -2, 5, -2, -9, -8, -4, -13, 7, 0, 1, -6, 16, 0, 0, -10}
;

const int16_t conv1d_7_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-7, 22, 43}
, {24, -33, -27}
, {50, -42, 12}
, {-12, -32, 50}
, {-20, 40, 12}
, {23, 44, -39}
, {35, 20, 38}
, {23, -12, 41}
, {-26, -36, 14}
, {29, -2, 19}
, {51, -2, 27}
, {-7, -13, 57}
, {-38, -18, -8}
, {-15, 1, 41}
, {2, 9, -26}
, {26, -25, 18}
, {21, -16, -10}
, {37, -47, 44}
, {55, -56, -32}
, {-35, -4, -9}
, {-20, -22, -18}
, {6, 44, -50}
, {-10, 17, 34}
, {22, -3, 15}
, {20, -52, 6}
, {-14, -18, 31}
, {54, -49, -40}
, {29, -36, 60}
, {35, 0, 52}
, {28, -30, -25}
, {24, 1, -21}
, {-36, 35, -17}
, {-43, 0, 9}
, {-50, 37, -2}
, {-46, -4, 13}
, {6, -52, 29}
, {-40, 10, 15}
, {6, -54, -27}
, {-3, 2, -44}
, {52, -3, -17}
, {-24, 5, 11}
, {-22, -16, -25}
, {-5, -2, -4}
, {-27, 0, 37}
, {-35, -21, -31}
, {-39, -11, -3}
, {49, -3, 14}
, {35, 48, -23}
, {35, 0, -36}
, {32, 6, -30}
, {46, 3, -21}
, {-38, -29, 1}
, {-11, 35, -36}
, {-18, -19, 28}
, {54, 31, 24}
, {28, 4, 8}
, {36, -37, -42}
, {-2, -24, 43}
, {-26, -1, -45}
, {4, -16, -20}
, {33, -32, -13}
, {-28, 9, -30}
, {28, 48, -37}
, {-1, 18, 12}
}
, {{35, -20, 42}
, {-8, -29, -7}
, {25, 66, -3}
, {-9, 10, -28}
, {31, 48, 27}
, {-25, 3, -12}
, {19, -27, 0}
, {54, -26, -30}
, {-31, 38, -29}
, {47, 66, -36}
, {69, -21, -30}
, {-26, 26, 58}
, {30, 20, -39}
, {-15, 21, 47}
, {-36, -23, 13}
, {37, -51, 20}
, {8, 21, -8}
, {40, -15, -50}
, {-43, 27, 32}
, {46, -46, 46}
, {16, 4, 41}
, {2, 42, 54}
, {12, 22, -4}
, {32, -35, 16}
, {-29, -29, -23}
, {14, 13, 32}
, {41, -28, 37}
, {36, -19, -26}
, {0, 40, -13}
, {29, -2, 19}
, {0, 31, 28}
, {23, 9, 6}
, {33, 40, 48}
, {-22, 0, 50}
, {55, -28, 13}
, {-16, 6, -50}
, {14, 18, -9}
, {-51, 33, -51}
, {47, -31, -27}
, {-56, -14, 0}
, {-19, -50, -22}
, {21, 38, -45}
, {-47, 24, 26}
, {11, -36, -29}
, {-21, -48, -33}
, {37, -43, 14}
, {40, -51, 49}
, {47, -11, 33}
, {23, -5, 32}
, {-21, 29, -28}
, {9, -42, -40}
, {11, 42, -40}
, {16, -6, 46}
, {-42, 31, 20}
, {39, -35, -21}
, {2, -24, 11}
, {28, -30, 4}
, {-21, 43, -6}
, {36, 25, 39}
, {9, 5, -35}
, {44, -48, -26}
, {46, -3, 45}
, {1, -58, -37}
, {-1, -57, 1}
}
, {{34, -39, -49}
, {23, 28, 0}
, {-21, 31, 10}
, {-26, 45, -9}
, {-27, -37, -54}
, {-7, 48, 8}
, {22, -53, -5}
, {-33, 41, -20}
, {-22, 5, -42}
, {8, 10, 11}
, {-16, -40, -39}
, {-4, -19, -12}
, {-20, 27, -29}
, {-34, 45, -15}
, {-2, 44, 15}
, {-15, -7, 9}
, {-7, -19, -14}
, {3, 7, 3}
, {-43, -39, 29}
, {20, -2, -53}
, {34, -20, -26}
, {17, -51, 43}
, {42, 4, 8}
, {41, 10, 7}
, {0, 45, 36}
, {9, -27, 44}
, {22, 13, -29}
, {-2, -15, -31}
, {28, -19, 14}
, {29, 15, 25}
, {4, -31, 6}
, {-44, -49, 1}
, {-14, -53, -28}
, {-44, 27, -3}
, {-2, 5, 29}
, {-45, -39, 6}
, {13, 28, 10}
, {-44, 23, -51}
, {19, 37, -29}
, {23, -23, -1}
, {-24, -14, 15}
, {12, -13, 48}
, {-13, -48, 22}
, {33, 11, -3}
, {49, -10, -4}
, {-16, 27, 38}
, {-31, 7, 4}
, {-21, -25, -48}
, {-40, 49, 48}
, {3, -24, -52}
, {-27, 38, -27}
, {-46, -10, 52}
, {45, -24, 49}
, {21, 42, 7}
, {-36, -12, -9}
, {-54, -11, 9}
, {-50, 12, -19}
, {-15, 28, -39}
, {-9, -26, -8}
, {30, 35, 46}
, {19, -32, 46}
, {17, 46, -12}
, {-11, -10, 9}
, {-40, -9, -29}
}
, {{-14, 32, 54}
, {-31, 8, 15}
, {0, 62, 42}
, {12, -8, -9}
, {19, -46, 15}
, {1, -12, -36}
, {-4, -3, 5}
, {37, -37, 31}
, {-18, -37, -22}
, {-5, -14, 63}
, {0, 16, -31}
, {2, -52, -11}
, {-36, 30, -46}
, {-29, 31, -37}
, {43, 22, 28}
, {25, 39, 0}
, {47, 24, -20}
, {50, 43, 40}
, {34, -40, -24}
, {-21, 26, -18}
, {39, 8, -18}
, {7, 3, -13}
, {35, -37, -28}
, {2, -12, 11}
, {-9, -15, 0}
, {-26, -33, -36}
, {-15, -2, 19}
, {26, 21, -53}
, {-25, -33, 51}
, {13, -49, -46}
, {-44, -1, -3}
, {-4, 12, 50}
, {4, -15, 27}
, {51, -40, -24}
, {-9, 26, 2}
, {20, -45, -35}
, {29, 20, 9}
, {54, 18, 15}
, {-6, -16, -8}
, {-34, 20, 35}
, {40, -7, 16}
, {27, -36, 24}
, {-41, -37, 49}
, {-19, 34, 30}
, {-32, -35, 26}
, {39, 30, -41}
, {39, -44, 34}
, {-6, -15, 3}
, {-32, 50, 15}
, {-20, -20, -8}
, {-35, -2, -32}
, {-14, -40, 55}
, {-31, 40, -7}
, {47, 17, 7}
, {-25, 15, -21}
, {18, 37, -24}
, {-11, 22, 1}
, {44, 48, 32}
, {-25, 24, 15}
, {72, -18, 41}
, {0, 3, 30}
, {-43, 53, -23}
, {9, -39, 41}
, {25, -40, -51}
}
, {{6, 12, 1}
, {-2, 7, -14}
, {-12, -41, 24}
, {-46, 36, -7}
, {61, -14, 5}
, {2, -34, -9}
, {49, -35, -25}
, {12, 14, -31}
, {41, -21, -41}
, {-47, -12, -15}
, {-12, -2, -55}
, {19, 40, -31}
, {17, -5, -32}
, {42, 6, 36}
, {-54, 34, 45}
, {-37, -26, 9}
, {33, 6, 9}
, {0, 8, -49}
, {8, 20, -3}
, {-37, 22, 47}
, {-57, -45, -24}
, {10, -9, -13}
, {70, 34, 12}
, {60, 33, 24}
, {4, -19, -10}
, {-42, 24, 49}
, {41, -13, 0}
, {1, -10, 35}
, {25, -18, -40}
, {-11, -33, 3}
, {-26, 51, -11}
, {1, -48, 46}
, {27, -17, 19}
, {39, 7, -2}
, {3, 21, -30}
, {-22, -52, -23}
, {-50, 44, -5}
, {0, -26, -28}
, {-41, 37, 32}
, {-27, -10, 39}
, {13, -41, 35}
, {6, -16, -20}
, {58, 17, 7}
, {-30, -39, -19}
, {34, 19, 8}
, {-16, -31, -7}
, {20, 38, -18}
, {-23, -42, 51}
, {17, 39, -23}
, {3, -37, -47}
, {33, 0, -35}
, {50, -48, 51}
, {-5, -54, 17}
, {30, -36, -1}
, {48, -41, 46}
, {-24, 0, 0}
, {-32, -12, -25}
, {8, -39, 1}
, {-52, -24, 33}
, {47, 22, 7}
, {13, 39, -46}
, {-18, -51, -29}
, {0, -22, 15}
, {43, -17, 29}
}
, {{34, -50, -43}
, {-12, 5, -54}
, {-19, 24, 32}
, {-12, -20, 33}
, {-52, 32, -15}
, {-34, -46, -15}
, {4, -23, -16}
, {-25, 4, -17}
, {1, 22, -20}
, {-17, 43, 18}
, {16, 2, 12}
, {-45, -23, -18}
, {-27, 36, -31}
, {-16, 26, -45}
, {24, -1, -38}
, {-40, 31, -46}
, {-38, 34, 50}
, {-2, -40, -39}
, {40, -42, 44}
, {-42, -5, 2}
, {23, 14, -4}
, {-44, -7, -13}
, {-18, -33, 37}
, {21, -23, 2}
, {23, -24, -45}
, {27, -29, 22}
, {17, -22, -44}
, {-33, -17, -6}
, {41, 16, 44}
, {-35, 38, -17}
, {-51, -51, 11}
, {-16, -38, 26}
, {27, 23, 31}
, {-8, 45, -48}
, {-48, -8, -11}
, {-32, 48, -47}
, {9, 8, -6}
, {-41, 23, -10}
, {-43, 20, -14}
, {-48, -44, -43}
, {-52, -29, 33}
, {20, -50, 38}
, {11, 5, -6}
, {-23, 23, -16}
, {-5, 5, 28}
, {44, -21, -46}
, {1, -14, 6}
, {-30, 11, -9}
, {-40, 32, 14}
, {39, 41, 44}
, {-39, 34, -40}
, {-33, 23, -39}
, {-13, -14, -19}
, {-3, 19, -25}
, {38, -44, 25}
, {-26, 44, 32}
, {11, -3, -18}
, {-18, -13, 39}
, {19, -40, 49}
, {16, 11, 5}
, {8, -24, -9}
, {-9, 0, 39}
, {-26, 33, 14}
, {4, -32, -13}
}
, {{17, 54, 14}
, {-29, 48, -53}
, {35, -12, 6}
, {10, 10, 19}
, {50, 43, -9}
, {-33, -46, 25}
, {48, -36, 31}
, {-32, 45, -30}
, {-21, 9, 7}
, {43, 54, 66}
, {55, 16, 60}
, {-21, -7, 33}
, {32, 25, -47}
, {-22, -38, -43}
, {-45, -3, -37}
, {34, -34, -7}
, {-20, -7, 36}
, {-4, 40, 42}
, {-12, 50, 47}
, {48, -23, 21}
, {-17, -10, 34}
, {35, 0, -28}
, {30, 11, -50}
, {26, 53, 1}
, {34, 50, 30}
, {39, 11, 43}
, {-20, 47, 43}
, {-38, -28, -25}
, {-36, -47, 35}
, {-55, -12, -9}
, {-8, 19, -13}
, {4, -8, 9}
, {-51, 22, -21}
, {-22, 27, -23}
, {-23, 26, 26}
, {40, -25, -29}
, {16, 35, -23}
, {39, -2, 1}
, {15, 7, 3}
, {56, -23, 15}
, {2, 10, -29}
, {-2, -33, -6}
, {7, -16, 45}
, {-50, 9, -23}
, {-11, 38, -39}
, {10, -4, 57}
, {-40, 55, 5}
, {20, -8, -11}
, {36, 33, -2}
, {-47, 14, 2}
, {-7, -24, -46}
, {53, 30, -14}
, {30, -6, -5}
, {-10, -24, 5}
, {6, -4, 11}
, {-36, 39, 16}
, {-36, 49, -26}
, {-6, -25, 6}
, {36, -5, -47}
, {-36, -26, -23}
, {-46, -22, -31}
, {12, -32, 1}
, {-58, -45, 44}
, {5, 35, 18}
}
, {{-9, 60, 28}
, {-30, -9, -9}
, {2, -44, 7}
, {-3, 0, -27}
, {-39, 35, -14}
, {-24, 49, 12}
, {18, 14, -28}
, {12, 13, 49}
, {-37, 0, -40}
, {-15, -38, 7}
, {-53, 48, 21}
, {41, 30, 8}
, {-3, 11, -5}
, {35, 21, 18}
, {-43, -33, -22}
, {-45, 2, 13}
, {0, -6, -4}
, {-29, 5, 2}
, {-12, 17, -25}
, {8, -3, 13}
, {17, 10, 53}
, {11, -17, 31}
, {15, -14, 27}
, {-37, 14, -32}
, {29, -13, -28}
, {-5, -17, 16}
, {6, 21, 13}
, {-8, 26, 36}
, {-26, -32, 41}
, {-34, 50, -46}
, {25, 0, -6}
, {22, -23, -20}
, {-25, -27, 2}
, {4, -27, 21}
, {-43, -8, -46}
, {20, 37, 41}
, {38, -8, -15}
, {-37, 54, 47}
, {-11, -53, 15}
, {-42, 11, -14}
, {19, 32, 11}
, {60, -47, -40}
, {46, 13, -22}
, {-19, -44, -15}
, {10, -4, -40}
, {-34, -14, 31}
, {20, 15, 2}
, {13, 45, -10}
, {27, 26, -52}
, {-31, -10, -25}
, {-13, -33, 36}
, {-24, -4, 14}
, {-5, 12, -2}
, {8, -30, 51}
, {-41, 19, -34}
, {24, -6, -5}
, {-5, -40, -19}
, {23, 22, -10}
, {48, -24, 21}
, {6, -58, 13}
, {40, 16, -43}
, {33, -52, 3}
, {-46, -15, -33}
, {53, 42, -45}
}
, {{29, 36, -18}
, {-4, -6, -20}
, {42, -20, 40}
, {8, -29, -13}
, {-12, 22, 34}
, {23, 43, 31}
, {23, -17, 42}
, {2, 11, 28}
, {27, 7, 13}
, {35, 11, -36}
, {0, -24, -27}
, {-4, 3, 3}
, {-41, -20, 34}
, {-18, 26, -33}
, {-1, -15, -31}
, {10, -17, 51}
, {8, 29, -21}
, {-28, -38, 12}
, {-19, -9, 46}
, {51, -35, 27}
, {25, 5, 0}
, {30, 36, 0}
, {14, -30, 40}
, {20, 19, 36}
, {38, 42, 7}
, {18, -45, -48}
, {-8, -22, -47}
, {24, -32, -56}
, {-36, 11, -35}
, {47, 48, 21}
, {28, 0, -55}
, {33, 18, 37}
, {-27, 22, -52}
, {1, 5, -16}
, {11, -42, 43}
, {15, -13, -50}
, {-48, 43, 23}
, {-24, 39, -41}
, {11, 17, -33}
, {18, 39, -2}
, {35, -4, -36}
, {-8, 54, 24}
, {-5, -15, 1}
, {-44, -21, -15}
, {12, -28, 32}
, {-10, 29, 8}
, {30, 30, 14}
, {-27, 0, 38}
, {2, 0, -1}
, {-49, 31, 49}
, {0, -42, 16}
, {55, 10, 2}
, {54, -21, 57}
, {-28, -56, 42}
, {-3, 17, 42}
, {-17, 41, -24}
, {23, 16, 40}
, {-25, 13, 56}
, {-18, 29, -51}
, {-7, -47, -36}
, {29, -6, 43}
, {18, -17, -38}
, {-16, -14, -37}
, {-27, 48, -8}
}
, {{-20, 10, -55}
, {-52, -14, -22}
, {-46, 44, -3}
, {62, -23, 34}
, {-15, 3, -19}
, {-9, -10, 19}
, {-11, 44, -38}
, {11, -11, 20}
, {35, -39, -12}
, {0, 23, 34}
, {4, -23, 5}
, {-8, 30, -38}
, {-40, 26, 28}
, {-21, 0, 14}
, {7, 16, -18}
, {-10, 39, -17}
, {4, 43, -15}
, {35, -11, 42}
, {47, -14, -25}
, {-42, 10, -33}
, {-1, -27, -22}
, {22, -45, -57}
, {11, -39, 43}
, {-3, 13, -52}
, {38, 48, -37}
, {-18, -32, -29}
, {28, 35, 11}
, {12, -13, -6}
, {20, -25, -37}
, {-8, -48, 0}
, {-11, -4, 36}
, {-39, -11, -27}
, {40, -20, 30}
, {-27, 17, 40}
, {9, 27, -50}
, {42, 0, -2}
, {-27, -34, 37}
, {-37, -3, -33}
, {51, -47, 14}
, {-7, -2, -28}
, {-33, 24, -28}
, {-4, -38, 42}
, {25, 27, 22}
, {2, 13, 24}
, {40, 47, -11}
, {14, -19, 33}
, {-22, 33, 0}
, {16, 34, -21}
, {17, 15, -26}
, {-34, 33, -37}
, {-44, -41, 42}
, {16, 7, -41}
, {31, 46, -24}
, {21, -12, 25}
, {8, 23, 18}
, {23, 3, 15}
, {16, 6, -57}
, {42, -10, 17}
, {11, -14, 26}
, {-56, -38, 21}
, {-30, 33, 46}
, {-58, -48, -44}
, {-9, -36, 34}
, {57, 23, -19}
}
, {{2, 1, 38}
, {0, -47, -40}
, {52, -4, 46}
, {43, -48, 33}
, {56, -28, -17}
, {-58, -19, -20}
, {10, 41, -10}
, {6, 9, -13}
, {-23, -43, -30}
, {51, 27, -14}
, {29, -50, -33}
, {-1, 6, -29}
, {-19, -41, -6}
, {-46, -26, -49}
, {38, -43, -46}
, {-40, -5, 14}
, {28, 23, 5}
, {-23, -46, 37}
, {62, -13, 24}
, {-7, 21, -58}
, {28, 24, -7}
, {1, 28, 45}
, {6, -51, 42}
, {45, 20, 0}
, {34, -52, 4}
, {-1, -13, 42}
, {44, -46, -11}
, {-5, -22, -9}
, {-29, -28, -10}
, {44, -6, -50}
, {40, -21, 18}
, {17, -3, -21}
, {-37, 0, 34}
, {1, -35, 26}
, {6, -32, -7}
, {59, 7, -35}
, {2, 38, -2}
, {-44, -31, -63}
, {-21, -25, -52}
, {32, -28, 28}
, {-52, 26, -50}
, {14, -35, -27}
, {67, 27, -16}
, {25, -18, -43}
, {-7, -41, -44}
, {44, -4, -36}
, {21, -32, -22}
, {-17, -39, 2}
, {-25, -25, -61}
, {58, 0, -39}
, {52, 7, 39}
, {49, 20, -20}
, {33, 33, -31}
, {-33, -9, -24}
, {-13, -51, 11}
, {-18, 37, -9}
, {-4, 62, -58}
, {-14, -35, -35}
, {-27, 9, -51}
, {-47, -12, -19}
, {-5, -45, 0}
, {-39, 62, -3}
, {-51, -41, 5}
, {-47, -34, 39}
}
, {{17, 45, -17}
, {-13, 11, -14}
, {5, 20, 24}
, {-2, -1, -14}
, {-38, 1, -29}
, {5, -34, -12}
, {32, -36, -6}
, {32, -44, 40}
, {-22, 30, 36}
, {-21, 2, 16}
, {-67, 2, 27}
, {32, 58, 19}
, {-16, -3, 10}
, {41, -34, 0}
, {-25, 19, 9}
, {-31, 45, 43}
, {13, 25, -3}
, {28, -10, 15}
, {-36, 9, 37}
, {2, -49, -19}
, {11, -11, 19}
, {18, -30, -3}
, {-53, 11, -25}
, {-8, -26, 0}
, {-10, -28, 19}
, {36, 26, -43}
, {-30, 27, 5}
, {-48, 18, 5}
, {-47, 18, -21}
, {8, -7, -37}
, {42, -15, -8}
, {6, 4, -12}
, {0, 51, -15}
, {31, 19, -33}
, {52, 4, 42}
, {-35, 47, -19}
, {23, 34, 11}
, {15, 41, 40}
, {-32, 0, -15}
, {45, -27, -35}
, {1, 23, -32}
, {-55, -26, 12}
, {-30, -28, -5}
, {34, -4, 19}
, {-50, -14, -10}
, {12, 40, -27}
, {13, -4, -45}
, {32, -10, -13}
, {45, 23, 10}
, {31, 32, -3}
, {-36, 46, 2}
, {-20, 22, 28}
, {38, 25, -33}
, {3, 22, -39}
, {18, -32, -25}
, {-52, 10, -23}
, {-48, -9, 36}
, {-21, -12, 49}
, {-35, 35, 51}
, {-58, -51, 28}
, {-43, 8, -41}
, {-50, 24, 40}
, {7, 12, -1}
, {-26, 12, 13}
}
, {{-7, 35, -3}
, {23, 25, 42}
, {64, 55, 21}
, {-8, 46, 21}
, {-26, -27, 16}
, {37, 36, 45}
, {-5, -53, 2}
, {45, 10, -11}
, {31, -1, 0}
, {-14, 16, -18}
, {-31, -15, -8}
, {-25, 2, 16}
, {37, 42, -16}
, {-33, 51, 14}
, {42, -9, -2}
, {41, -19, 26}
, {-42, 5, -3}
, {-50, -50, 7}
, {-15, -19, -29}
, {8, 3, -36}
, {41, 34, 45}
, {-3, -37, 45}
, {68, 51, 66}
, {-22, 19, 3}
, {-18, -28, 27}
, {-11, -47, 16}
, {32, -16, -7}
, {9, -53, 33}
, {27, 18, 25}
, {24, -39, -54}
, {17, -44, 33}
, {10, 32, 8}
, {40, -10, -45}
, {-21, 45, -45}
, {-27, -48, -8}
, {-30, -35, -59}
, {14, -38, -5}
, {-17, -21, -32}
, {33, 18, -6}
, {1, 56, 16}
, {-22, -60, -61}
, {25, -9, 27}
, {-33, 18, -25}
, {68, -30, 45}
, {4, 20, -45}
, {45, -16, -8}
, {26, 23, -41}
, {4, 23, 15}
, {-27, -1, -17}
, {-19, -24, -26}
, {-32, -26, 33}
, {12, -36, -1}
, {37, -30, -58}
, {9, -41, 19}
, {17, 4, -59}
, {19, -59, -2}
, {14, -40, 44}
, {27, -28, 38}
, {13, 18, -41}
, {71, 60, 41}
, {-29, 15, -52}
, {-37, -30, -30}
, {47, 1, -41}
, {5, 72, -12}
}
, {{-15, -14, 44}
, {-23, 27, 45}
, {20, 27, 61}
, {49, 19, -14}
, {-17, 7, -7}
, {2, 23, -2}
, {-40, -18, -22}
, {-1, -19, 51}
, {29, 32, 4}
, {-27, -34, -3}
, {-43, -28, -9}
, {49, 41, 37}
, {-42, 24, -55}
, {-55, -54, 48}
, {38, 35, 8}
, {29, 7, -17}
, {-10, -7, -10}
, {-40, 9, 33}
, {-42, -3, -20}
, {15, 36, 51}
, {22, -14, -29}
, {16, -44, -36}
, {-60, -19, -24}
, {1, 23, 31}
, {-53, 39, -49}
, {-2, 20, 25}
, {-44, -28, 35}
, {-42, -50, -43}
, {-33, 32, 26}
, {-5, -28, 55}
, {35, -1, 12}
, {-38, -29, 50}
, {-33, -24, -27}
, {-31, 29, -21}
, {-20, -27, 22}
, {-16, -5, -37}
, {-28, 37, 25}
, {20, -38, 5}
, {42, -33, -5}
, {-19, 10, -25}
, {-1, 39, -14}
, {-19, 37, -16}
, {1, -21, 17}
, {-18, 47, -34}
, {42, -50, 25}
, {-3, 4, 60}
, {32, 36, 25}
, {0, -25, 25}
, {22, -34, 11}
, {-15, 25, 20}
, {-57, 30, 36}
, {56, -20, 28}
, {-66, 5, 52}
, {-34, -48, -45}
, {-6, -52, 42}
, {12, 31, 19}
, {-24, 9, -7}
, {-11, -20, 30}
, {-13, -52, -38}
, {14, -37, 12}
, {34, -10, 50}
, {-18, 32, -39}
, {-20, -12, 24}
, {4, 36, -20}
}
, {{23, -17, 19}
, {-14, -42, -24}
, {-23, 18, 39}
, {-1, -59, -41}
, {29, 3, -17}
, {-53, -49, 13}
, {49, 47, -27}
, {48, 14, -42}
, {0, -3, 5}
, {-34, 2, 58}
, {28, 43, -51}
, {3, 50, -46}
, {23, 14, -38}
, {23, 3, 1}
, {33, -20, -35}
, {-40, -40, -27}
, {21, 20, -3}
, {-27, -45, -31}
, {-37, -40, -32}
, {36, 47, 25}
, {3, 41, 37}
, {40, -49, 9}
, {21, -26, 8}
, {-15, 31, 46}
, {-28, 45, -6}
, {7, 33, -4}
, {31, -31, -17}
, {-2, 35, -44}
, {0, 25, -38}
, {2, 28, 28}
, {-46, -17, 9}
, {45, 21, -20}
, {-5, -45, -3}
, {1, -31, -41}
, {29, 1, 14}
, {23, -38, 31}
, {43, -43, -6}
, {-41, -30, -38}
, {27, 5, -45}
, {11, 53, 23}
, {42, 9, -50}
, {-21, -29, 36}
, {-21, 38, -4}
, {28, 27, 14}
, {-43, -41, -12}
, {27, -23, 22}
, {-4, -31, -53}
, {13, -52, 12}
, {41, 42, 14}
, {22, -47, -19}
, {44, -12, 22}
, {31, -48, -15}
, {-28, -4, 46}
, {-13, 10, -34}
, {30, 49, -2}
, {59, 24, 36}
, {18, -31, -6}
, {18, -13, -20}
, {-35, -46, -46}
, {24, -6, 46}
, {24, -21, -45}
, {38, -3, 30}
, {9, 23, 4}
, {-56, 21, -16}
}
, {{11, 45, -2}
, {-8, -45, 5}
, {-16, -56, 46}
, {-48, 23, -18}
, {-34, -9, 19}
, {-42, -58, 7}
, {-28, 26, -29}
, {-51, -50, -51}
, {-18, -3, 38}
, {-10, -51, -22}
, {-22, 4, -62}
, {24, 22, 17}
, {-50, -11, -9}
, {-16, 4, 47}
, {-30, -9, -7}
, {-38, -50, -20}
, {-23, -36, -48}
, {-2, -12, 5}
, {-19, -44, -11}
, {31, -51, -56}
, {26, -42, 13}
, {-17, -8, -47}
, {-21, 11, -33}
, {-8, -2, 9}
, {-1, 50, -50}
, {-3, 43, 35}
, {-34, 32, -29}
, {51, 50, -8}
, {31, -29, 31}
, {-32, -16, 5}
, {-37, -47, 1}
, {42, 12, 58}
, {28, -3, -41}
, {-44, -42, 7}
, {-49, -34, -37}
, {7, -44, -17}
, {-52, 26, -28}
, {-27, 1, 8}
, {22, 21, 14}
, {-7, -26, 8}
, {17, 30, 41}
, {43, 3, 14}
, {38, -42, 16}
, {-9, -3, 7}
, {-12, -39, 19}
, {11, -4, 6}
, {-21, 44, -11}
, {-55, 37, 50}
, {-22, -14, -29}
, {22, 38, 38}
, {16, -26, 1}
, {24, 53, -23}
, {12, -40, 21}
, {8, -15, 40}
, {0, -50, -34}
, {30, 0, -23}
, {-42, -22, 7}
, {-47, -48, 16}
, {50, 26, -27}
, {-24, -48, -26}
, {-29, -37, -1}
, {-46, 17, 4}
, {34, -32, -46}
, {-8, 26, 30}
}
, {{10, -1, 47}
, {-33, 3, -37}
, {57, 12, 49}
, {-20, -12, -48}
, {12, 50, 2}
, {31, -22, 36}
, {-34, 18, 53}
, {-18, 31, -49}
, {-15, 45, 4}
, {47, 39, 28}
, {11, 18, 41}
, {23, 43, 38}
, {47, 44, -1}
, {9, -23, 30}
, {38, 25, -21}
, {14, 7, 39}
, {-3, -32, -17}
, {-6, 38, -16}
, {-52, -27, 17}
, {-9, -31, -48}
, {-48, -1, -18}
, {4, -6, 7}
, {16, 28, 5}
, {63, 24, -16}
, {-39, -29, 7}
, {-12, 35, 6}
, {29, 51, -21}
, {-38, -5, -24}
, {-4, 45, -24}
, {-34, 7, 50}
, {6, 44, 68}
, {2, -49, -34}
, {36, -47, 44}
, {-44, -5, -17}
, {18, -34, 18}
, {-22, 2, -30}
, {-12, 3, -2}
, {2, 15, -11}
, {13, 8, 7}
, {41, -4, -1}
, {7, -30, -31}
, {17, 25, -15}
, {-49, 45, 46}
, {-10, -15, -38}
, {51, -35, -1}
, {-1, -39, -4}
, {-16, -17, 39}
, {30, -44, -14}
, {43, -37, -7}
, {33, -39, -34}
, {-14, -1, -32}
, {-42, -7, 5}
, {43, -44, 18}
, {-36, -7, -55}
, {39, 55, 9}
, {-31, 19, -30}
, {20, 39, 17}
, {30, 37, -18}
, {35, 18, 0}
, {25, 11, 48}
, {-29, -1, -17}
, {10, -6, 38}
, {-20, -27, 34}
, {-9, 36, 43}
}
, {{8, -25, 16}
, {-44, -48, 23}
, {41, 56, 2}
, {-19, 24, 16}
, {-33, -33, 28}
, {22, 34, 4}
, {28, -10, -25}
, {-28, 23, 4}
, {-10, 44, 26}
, {42, -36, 11}
, {-23, -29, 9}
, {5, -39, 32}
, {24, -4, 23}
, {-1, -42, 10}
, {23, -49, 1}
, {4, -19, 27}
, {-49, -34, 0}
, {19, 8, -42}
, {21, -47, 42}
, {15, -12, 16}
, {0, 50, 24}
, {35, 17, 27}
, {3, -1, -3}
, {-18, -27, 20}
, {-49, -33, -8}
, {-37, 10, -44}
, {21, 8, 15}
, {-4, -17, 39}
, {9, -49, -20}
, {39, 0, -4}
, {-54, 28, 63}
, {-4, -13, -43}
, {13, -50, 21}
, {26, 9, -36}
, {32, 57, 26}
, {9, -52, -9}
, {5, -28, -36}
, {-38, -28, -14}
, {-1, 7, 46}
, {-40, 36, 35}
, {4, -33, -28}
, {42, 32, 49}
, {-1, 5, -35}
, {15, -56, 36}
, {44, 23, -27}
, {35, 25, 9}
, {0, -17, 33}
, {-15, -9, -49}
, {-21, 15, 37}
, {-9, -40, 45}
, {43, -42, 15}
, {-11, -14, 1}
, {-27, 8, 2}
, {15, -31, 26}
, {40, 44, 44}
, {61, -30, -17}
, {56, 47, 29}
, {-37, 52, 53}
, {-15, 49, 17}
, {-35, -21, 49}
, {-50, 25, -3}
, {-40, 25, -39}
, {-22, -9, 0}
, {-27, 35, -26}
}
, {{12, 40, -35}
, {-22, -37, -41}
, {12, -27, -11}
, {4, 28, 0}
, {48, -31, -18}
, {-28, 30, 26}
, {-6, -23, -16}
, {-40, -11, -46}
, {-10, -34, -7}
, {59, -21, 9}
, {0, 42, 65}
, {-53, 27, -31}
, {-14, 8, 26}
, {-13, -48, -21}
, {52, -20, 26}
, {49, -41, 40}
, {47, 6, 43}
, {20, 27, 32}
, {51, -49, 41}
, {25, -34, -8}
, {-29, 30, -19}
, {35, -18, 7}
, {16, 32, 31}
, {8, 3, 41}
, {27, 17, 10}
, {-23, -8, -10}
, {5, 2, -37}
, {-23, -25, 37}
, {-34, -25, -31}
, {-4, -48, -39}
, {-27, 13, 5}
, {-10, 42, 51}
, {-9, 50, 42}
, {-20, 24, 18}
, {-44, -24, -17}
, {39, 24, -42}
, {-5, 18, 41}
, {6, 36, 26}
, {14, 26, -2}
, {-25, 44, -29}
, {-31, -5, 36}
, {-26, -6, 17}
, {0, -17, 44}
, {30, -40, 13}
, {-1, -36, -18}
, {45, -19, -20}
, {21, -30, 0}
, {16, 26, 37}
, {-20, 26, -27}
, {-2, 47, 16}
, {-9, -53, -14}
, {-15, 31, -42}
, {32, 28, -6}
, {-9, 22, -31}
, {-12, -1, -35}
, {4, -40, 31}
, {48, -21, 25}
, {-33, 19, 54}
, {-23, -47, -20}
, {35, -4, -3}
, {10, -27, 5}
, {44, -25, 23}
, {-29, 3, 38}
, {-12, 21, 52}
}
, {{-11, -19, 28}
, {52, 26, -5}
, {-21, 53, 47}
, {38, 24, 33}
, {24, -6, 56}
, {3, -6, -35}
, {-24, 10, 11}
, {22, 18, -39}
, {33, 42, 20}
, {-22, -39, -19}
, {-3, -24, 27}
, {7, -7, 20}
, {-42, 37, -11}
, {-29, 34, -38}
, {-45, 47, 34}
, {38, 40, 24}
, {-26, 15, 0}
, {-5, -30, 47}
, {42, 47, -31}
, {-7, -4, -52}
, {-41, 27, 26}
, {-12, 43, -42}
, {-21, 32, 48}
, {63, 4, 0}
, {49, -2, -44}
, {-21, 19, 49}
, {40, 3, 7}
, {23, -34, 18}
, {49, -51, 36}
, {-15, -31, -17}
, {4, 61, 38}
, {11, 20, 14}
, {52, -7, -12}
, {-12, -19, -18}
, {-37, 6, -12}
, {-31, 10, 21}
, {15, 27, 31}
, {5, 44, -19}
, {-32, -22, -17}
, {27, 0, 47}
, {13, 36, -39}
, {48, 31, 0}
, {-29, -43, -33}
, {53, -43, 13}
, {-43, -34, 33}
, {-34, -16, 8}
, {6, 32, -30}
, {28, 30, -33}
, {33, -25, 48}
, {4, 0, 23}
, {-42, -30, -14}
, {-28, 18, 14}
, {-8, -35, -7}
, {-15, -7, 10}
, {-28, -47, -36}
, {36, 0, 32}
, {-7, -28, 20}
, {-8, -25, 23}
, {5, 44, 28}
, {59, 43, 6}
, {36, -21, -19}
, {42, -14, -27}
, {24, 36, -12}
, {11, 23, 12}
}
, {{2, 41, 27}
, {-50, -40, -16}
, {62, 56, -15}
, {-13, -14, -37}
, {24, 44, 24}
, {-49, 4, -27}
, {-12, -35, -14}
, {-16, -46, 28}
, {-28, 35, 5}
, {41, 37, 72}
, {27, 7, 52}
, {1, 61, 37}
, {-14, 34, 44}
, {0, -41, -32}
, {-26, 1, 40}
, {27, -3, -6}
, {15, 5, -20}
, {-48, 6, 50}
, {-52, 39, -30}
, {31, -33, 1}
, {41, -22, 50}
, {3, 47, 35}
, {-23, -14, 31}
, {35, 23, 18}
, {13, 30, 36}
, {5, 14, 8}
, {34, 35, -24}
, {20, 4, 19}
, {43, 14, 20}
, {19, 13, 0}
, {-34, 0, -7}
, {-34, -35, 22}
, {20, -9, -26}
, {18, -50, 13}
, {40, -28, 52}
, {-37, -6, -31}
, {-9, -4, 50}
, {-39, -21, 47}
, {-22, -35, 26}
, {6, 42, -46}
, {42, -8, -1}
, {44, -24, -17}
, {-45, 9, 33}
, {45, 25, 24}
, {-5, -2, 23}
, {30, -7, -34}
, {41, 41, 33}
, {-32, 8, -3}
, {41, -43, 45}
, {21, 28, -33}
, {-43, -8, 46}
, {-42, -42, 41}
, {-50, 32, -28}
, {34, -54, 2}
, {6, 1, 7}
, {23, 27, 56}
, {48, 51, 35}
, {-25, 33, -33}
, {-14, -47, 6}
, {16, 35, 22}
, {27, 29, 34}
, {-22, 19, 33}
, {-35, -13, -13}
, {-33, 0, -52}
}
, {{3, -37, 37}
, {-39, 20, -5}
, {-48, 1, -36}
, {-49, -6, 14}
, {9, 3, 23}
, {39, -28, -27}
, {-14, -37, 25}
, {32, 41, -21}
, {16, -20, 46}
, {7, 5, -39}
, {38, 12, 19}
, {15, -45, 17}
, {-12, -45, 27}
, {-34, 18, 36}
, {-20, 8, 1}
, {-46, -50, -9}
, {-5, 29, 27}
, {20, 13, -1}
, {36, 4, 10}
, {-20, 13, -44}
, {9, -43, 37}
, {45, -10, -38}
, {6, -19, -13}
, {16, -13, -20}
, {-40, -49, 15}
, {9, 51, 20}
, {42, 4, 41}
, {45, 46, -9}
, {-48, 29, 12}
, {14, -11, -13}
, {32, 30, 13}
, {24, -36, -39}
, {-38, -18, -11}
, {-29, 18, -47}
, {-14, 13, 8}
, {-42, -13, -53}
, {-18, 8, 44}
, {-8, -45, -39}
, {-4, -16, -48}
, {-13, 23, 21}
, {-2, -27, -9}
, {14, 25, -6}
, {-33, 14, -43}
, {18, -33, -44}
, {36, -8, -55}
, {-15, 3, 9}
, {36, 4, 28}
, {-38, 11, -3}
, {-12, 30, 15}
, {-49, 2, -13}
, {-37, -54, 10}
, {23, -19, 10}
, {19, -4, 21}
, {-44, -12, -31}
, {-41, -23, 40}
, {46, -7, -21}
, {6, -13, -14}
, {-36, -52, -40}
, {-45, 35, -30}
, {-20, 36, -19}
, {-23, -20, -50}
, {35, -3, 2}
, {-24, -47, -15}
, {6, -11, 3}
}
, {{-20, 32, -1}
, {23, -19, 47}
, {-29, -1, 46}
, {39, 21, 3}
, {-48, -2, 29}
, {41, 0, 39}
, {31, 48, -15}
, {-33, 28, 38}
, {-51, -23, 18}
, {15, 58, 49}
, {11, 46, 12}
, {-23, -48, 1}
, {0, -44, -13}
, {-9, 32, 0}
, {-43, -21, 4}
, {-39, -5, 15}
, {17, 27, 38}
, {-32, -19, 14}
, {-24, 8, -8}
, {41, 38, -47}
, {-22, -31, 4}
, {48, 38, 24}
, {45, 8, -49}
, {48, -25, -19}
, {15, -32, 44}
, {8, -43, 35}
, {-56, -9, 42}
, {-17, -55, -26}
, {-25, 24, -21}
, {36, 13, 0}
, {36, -31, 50}
, {-38, 3, -47}
, {30, -16, -45}
, {53, -23, 11}
, {-13, -44, -6}
, {45, 20, -20}
, {-12, 15, 46}
, {-12, -19, 14}
, {4, 55, -17}
, {29, 8, 38}
, {45, 38, 2}
, {-50, -27, 10}
, {30, 7, 22}
, {20, 21, 33}
, {11, 24, -27}
, {-24, -24, -40}
, {1, -41, 35}
, {8, -27, -36}
, {13, -43, -53}
, {9, 20, 23}
, {10, 37, -25}
, {-25, -30, 2}
, {-4, 24, 22}
, {-37, 56, -6}
, {18, 38, -43}
, {15, -11, 4}
, {-23, 31, -22}
, {-42, -9, -20}
, {-11, 19, 51}
, {24, 10, 10}
, {-38, -7, -11}
, {5, -47, -21}
, {32, -35, -21}
, {43, -44, -40}
}
, {{-43, 5, 4}
, {-26, 4, -11}
, {-52, 7, -63}
, {1, -40, 0}
, {24, -9, -8}
, {20, -4, 7}
, {-37, 7, -10}
, {-18, 18, 39}
, {53, -1, 47}
, {-43, 32, -34}
, {19, 34, 16}
, {14, -30, -32}
, {23, 48, 54}
, {0, -13, -24}
, {-50, 12, 30}
, {2, 28, 2}
, {26, 2, 17}
, {50, 35, -51}
, {23, -44, 30}
, {-13, 20, 42}
, {-45, -21, 11}
, {2, -3, -15}
, {33, 30, 5}
, {-4, 15, 10}
, {35, 51, 44}
, {39, -25, 48}
, {3, -20, -25}
, {39, 4, 50}
, {-41, 7, -16}
, {46, -4, -42}
, {-23, 11, 4}
, {-1, 27, -1}
, {-8, 9, -3}
, {-30, -56, -6}
, {10, 37, 32}
, {47, 42, -3}
, {-41, -43, -18}
, {-18, 0, 2}
, {13, -31, -35}
, {-50, 15, 16}
, {0, 6, 6}
, {-41, 8, 4}
, {27, -21, 16}
, {-2, -19, 48}
, {9, -28, 41}
, {-30, 28, 6}
, {18, -18, 37}
, {-12, -4, 38}
, {49, -49, 2}
, {31, -36, -23}
, {52, 12, -33}
, {4, -46, 42}
, {-26, -3, 25}
, {-15, 63, 24}
, {9, 10, -38}
, {-2, 13, -22}
, {19, -9, 23}
, {11, -43, 0}
, {41, 34, -6}
, {24, 25, -13}
, {-29, 0, 1}
, {-38, -6, 12}
, {1, -48, 40}
, {17, 12, -60}
}
, {{-34, -19, -47}
, {-37, -27, 4}
, {52, 6, -5}
, {0, 33, -24}
, {-36, -4, -39}
, {32, -28, -9}
, {22, -10, 47}
, {-19, -40, -1}
, {51, 28, -11}
, {15, -70, -37}
, {17, -31, 38}
, {-4, 41, 35}
, {-50, -50, -33}
, {-21, 8, -49}
, {24, -37, -26}
, {10, 8, -27}
, {-27, 45, 2}
, {20, 15, -44}
, {52, -9, 8}
, {24, -34, 16}
, {52, 38, 18}
, {42, -19, 33}
, {-15, 40, 20}
, {-25, -66, 2}
, {-31, 4, 28}
, {-22, 19, 6}
, {28, 38, 41}
, {-12, -46, -56}
, {15, 37, -36}
, {-37, -9, -3}
, {15, 22, 10}
, {27, -30, 14}
, {20, 7, 25}
, {27, -33, 0}
, {-26, -42, 1}
, {-12, 10, 42}
, {-11, 3, -22}
, {-14, 15, 27}
, {17, 18, -29}
, {-19, 39, 29}
, {-10, -14, 40}
, {-26, -8, -19}
, {-13, 29, 19}
, {1, -6, 36}
, {-8, 20, 29}
, {22, -12, -15}
, {-15, -35, 53}
, {14, 40, 44}
, {-30, -40, -50}
, {2, 5, -53}
, {-44, -33, -56}
, {25, 33, 4}
, {-29, -1, 23}
, {-8, -44, -44}
, {-4, -9, -24}
, {-23, -15, 49}
, {31, 47, 28}
, {41, -32, 34}
, {-39, -48, -17}
, {-41, -18, 1}
, {5, 44, -6}
, {34, -16, 33}
, {-22, -36, -37}
, {7, -10, 9}
}
, {{-18, 43, -18}
, {-4, -33, 7}
, {-43, 48, 70}
, {-48, 18, -31}
, {59, -33, 36}
, {-16, -29, 42}
, {47, -48, 21}
, {53, 22, -4}
, {24, 20, 22}
, {38, 1, 48}
, {18, 14, 35}
, {15, 21, 21}
, {-12, -42, 24}
, {30, -49, -28}
, {34, -27, 58}
, {-21, 47, 0}
, {-31, -38, -13}
, {-39, -7, 43}
, {12, 23, 27}
, {50, 2, -36}
, {27, -60, 45}
, {53, 21, 59}
, {-35, 20, -35}
, {69, 56, 2}
, {12, 36, -23}
, {-6, -45, -29}
, {-42, 28, 30}
, {-9, -12, -57}
, {-26, -32, -9}
, {-9, -12, -16}
, {55, 21, 11}
, {-47, 16, 50}
, {45, 7, -22}
, {-42, 10, -41}
, {41, -20, 32}
, {-45, 38, 40}
, {-9, 23, -19}
, {-26, -31, 36}
, {37, -17, 45}
, {-43, 35, -44}
, {-45, 21, 12}
, {-19, 1, 19}
, {-3, -34, 33}
, {24, -54, 44}
, {-27, 22, -29}
, {-23, -15, -55}
, {-34, 27, -45}
, {-26, -6, 26}
, {-22, 39, 29}
, {-43, -42, 18}
, {29, -40, 4}
, {33, -24, 14}
, {-20, 19, -24}
, {-22, 34, -43}
, {-2, -46, 42}
, {59, 53, 35}
, {30, 43, -11}
, {-12, -25, -50}
, {21, -15, 26}
, {20, 26, 49}
, {0, -42, 54}
, {47, 18, -4}
, {13, 5, 59}
, {21, -50, -35}
}
, {{-2, -48, -29}
, {-16, 3, -47}
, {-35, -29, -13}
, {30, -32, 23}
, {24, 24, -15}
, {-17, -19, -31}
, {-30, -44, -54}
, {16, -53, -16}
, {-23, -15, -26}
, {17, -49, -47}
, {-56, -35, -30}
, {-26, -11, -30}
, {-8, -48, 14}
, {-29, 1, 2}
, {-6, -30, 19}
, {24, 35, 36}
, {21, -23, -33}
, {-37, 36, -39}
, {11, 16, 2}
, {33, 45, -24}
, {29, -11, -12}
, {29, 29, -34}
, {-41, -7, 30}
, {27, -48, 28}
, {-40, 45, 35}
, {41, 33, -46}
, {-11, -53, -49}
, {-33, 38, 35}
, {5, 33, 40}
, {3, -31, 18}
, {46, -43, -39}
, {19, -52, 28}
, {-47, -56, 39}
, {-32, 32, -45}
, {-18, -3, -51}
, {46, 15, 19}
, {38, 40, 43}
, {36, -31, -39}
, {3, 38, 37}
, {-32, -22, 12}
, {43, -39, 22}
, {-14, 43, -35}
, {15, -30, 10}
, {10, -47, 25}
, {-43, -45, 30}
, {-4, 15, 6}
, {-30, 46, -50}
, {-31, -18, 2}
, {34, -31, -4}
, {-21, 37, -26}
, {-26, 17, 46}
, {22, 25, -39}
, {-9, -15, 11}
, {40, 6, -24}
, {29, -5, -14}
, {-49, 16, 4}
, {-39, 44, -16}
, {-18, -24, -35}
, {-21, 8, 39}
, {-31, -44, 12}
, {27, 5, -29}
, {-51, 26, -48}
, {-34, -10, 14}
, {-29, 34, -11}
}
, {{-43, -18, 5}
, {45, 43, -34}
, {-55, -2, -31}
, {-13, -45, -59}
, {35, -29, -9}
, {49, 1, 28}
, {9, -44, 6}
, {-28, 0, 16}
, {-2, -49, -9}
, {44, 28, 46}
, {31, -56, 5}
, {-30, 33, 48}
, {-20, -11, -25}
, {-12, 24, 21}
, {9, 22, -19}
, {37, -49, 0}
, {-13, -5, 47}
, {-43, -4, -23}
, {-45, -33, 50}
, {-3, -53, 6}
, {-8, 19, 45}
, {39, 35, 40}
, {-16, 49, -28}
, {-16, 41, 14}
, {0, 27, -1}
, {50, -21, -16}
, {-14, 33, 34}
, {-17, -45, 19}
, {-36, -38, 13}
, {46, 19, 24}
, {9, -55, -24}
, {-28, -10, -36}
, {7, -25, 9}
, {-36, 3, 27}
, {-4, 40, 0}
, {7, 32, -17}
, {-8, -35, -28}
, {22, -32, 47}
, {20, -51, -32}
, {-38, -17, 9}
, {44, 29, 1}
, {0, -47, -48}
, {7, 33, 21}
, {3, -22, -24}
, {-28, -30, 38}
, {-23, -10, -18}
, {41, 36, 23}
, {21, -19, -43}
, {-36, 26, -35}
, {-6, 6, -17}
, {-13, 48, -34}
, {-33, 35, -27}
, {29, 6, -41}
, {25, -21, 12}
, {-1, 31, 39}
, {-37, 21, -18}
, {29, 11, 28}
, {-25, 45, -30}
, {0, 46, 34}
, {-55, 21, -24}
, {35, -8, 49}
, {21, -6, 46}
, {38, 40, 21}
, {21, -4, -10}
}
, {{-27, -46, -54}
, {9, -42, 37}
, {-65, -10, 15}
, {-14, 34, 24}
, {-6, 53, -38}
, {45, -19, 49}
, {-52, 21, -25}
, {15, 11, -25}
, {12, 31, -6}
, {2, 0, -54}
, {-17, 13, 6}
, {-38, 42, 17}
, {11, 43, -19}
, {-23, -5, 37}
, {-34, 2, 32}
, {32, -14, 14}
, {5, -1, -42}
, {20, 37, -34}
, {-49, -51, -30}
, {6, -23, 14}
, {42, -13, -37}
, {39, 29, -54}
, {16, -55, 42}
, {-60, 34, 7}
, {1, 34, 50}
, {-16, -3, -35}
, {-18, 24, 35}
, {2, -25, 45}
, {47, 25, -25}
, {-9, 7, 38}
, {-26, -56, -37}
, {-47, 43, 36}
, {-48, -12, 4}
, {-27, 38, -15}
, {-45, -1, 30}
, {-41, 40, -9}
, {30, 23, 3}
, {6, 42, 2}
, {42, 33, -52}
, {48, -8, -19}
, {13, 13, 39}
, {11, -49, -36}
, {0, -48, 24}
, {-3, 3, -24}
, {-10, 49, 44}
, {17, 58, -30}
, {23, -50, 49}
, {12, -23, -37}
, {-7, -3, 9}
, {-23, 37, -43}
, {-26, -30, 3}
, {-2, -49, -1}
, {23, 10, 1}
, {-25, 31, 65}
, {4, 26, 33}
, {-47, -16, 9}
, {-26, 22, -55}
, {-11, -39, 12}
, {-24, 49, -22}
, {-9, 39, -15}
, {-10, 27, -17}
, {2, 1, -4}
, {40, -2, -46}
, {54, 49, -27}
}
, {{18, 12, -50}
, {-36, -35, 2}
, {42, 31, 20}
, {29, -1, -26}
, {-16, 13, -32}
, {-5, -31, -33}
, {45, -49, 3}
, {-11, -3, 41}
, {21, -2, -43}
, {-28, -16, -9}
, {10, 23, -39}
, {-50, 27, -27}
, {6, 7, -17}
, {2, -28, -45}
, {20, -26, -13}
, {14, -21, 5}
, {-21, 51, -24}
, {-7, -50, -30}
, {-41, -46, 49}
, {-34, -36, -40}
, {-12, 6, -37}
, {-13, -47, 27}
, {-36, -16, -2}
, {5, 28, -12}
, {3, 37, 25}
, {-21, -6, 14}
, {25, -22, -39}
, {14, -47, 29}
, {-2, -14, -47}
, {34, -1, -43}
, {-22, -22, -38}
, {-41, 28, 27}
, {-5, 14, -14}
, {-38, -13, 31}
, {-16, 36, 11}
, {-42, 49, -29}
, {-35, -24, -48}
, {-11, 7, 34}
, {-47, 43, 32}
, {-3, -23, -22}
, {34, -43, -41}
, {10, 26, -28}
, {-30, -23, 13}
, {-41, 4, 32}
, {25, 35, 31}
, {-44, -2, 38}
, {-8, 24, -42}
, {-24, -39, -39}
, {-41, 3, -43}
, {4, 10, -32}
, {-45, 17, -28}
, {-35, -47, -44}
, {-36, 41, -38}
, {29, 46, 26}
, {-3, 7, 12}
, {33, -6, 36}
, {31, 37, -4}
, {-9, -44, 41}
, {-50, 12, -32}
, {-38, -52, 21}
, {-6, -40, -20}
, {-24, 9, 41}
, {38, 30, -47}
, {-6, 36, -32}
}
, {{3, 31, 0}
, {57, 23, 28}
, {-16, 2, -3}
, {-22, -3, 21}
, {64, 15, 35}
, {-4, -26, 0}
, {19, -16, -35}
, {-24, -20, 10}
, {23, -5, -31}
, {26, -2, -10}
, {32, 20, -1}
, {41, 45, 58}
, {29, -5, -52}
, {29, -6, -20}
, {50, 31, 3}
, {40, 31, 39}
, {-18, -17, 44}
, {-31, -6, -23}
, {39, 31, -57}
, {-6, -26, -29}
, {9, 31, -29}
, {-14, 6, 3}
, {51, 58, 4}
, {3, 3, 53}
, {27, 30, -39}
, {45, -40, 31}
, {-36, 13, -46}
, {60, 35, 23}
, {41, 10, -24}
, {0, -18, 34}
, {-3, 4, 36}
, {18, 16, 49}
, {16, -2, 26}
, {37, 43, 45}
, {-29, 8, 5}
, {53, 49, 6}
, {-29, -17, 10}
, {-6, 18, -17}
, {12, -38, 34}
, {-29, 23, -14}
, {26, -54, -14}
, {70, 3, -15}
, {-20, -53, 8}
, {11, -32, -37}
, {9, 26, -6}
, {2, -30, -45}
, {26, 10, -12}
, {15, 48, 6}
, {-35, -4, -18}
, {-32, 34, 19}
, {52, -23, 26}
, {13, 19, -36}
, {-4, -39, -4}
, {33, 40, -62}
, {37, 1, -62}
, {61, 41, 48}
, {-33, -6, 10}
, {65, -34, -13}
, {38, -43, -27}
, {9, 8, 1}
, {11, 26, -39}
, {51, 29, -19}
, {-35, -33, 46}
, {-16, 27, 30}
}
, {{-40, -59, -62}
, {-27, -23, 32}
, {-32, -7, -2}
, {45, -31, 45}
, {28, -1, -8}
, {46, -5, 32}
, {47, -30, 10}
, {-46, -29, -14}
, {-47, 3, 27}
, {24, 5, 33}
, {37, -4, 8}
, {-45, -18, -42}
, {-34, 29, -50}
, {-14, -40, 11}
, {7, 26, -34}
, {14, 28, -2}
, {33, -16, 2}
, {-13, -21, -13}
, {-13, -3, 47}
, {55, 0, 0}
, {52, 9, -37}
, {-23, 30, 15}
, {-23, -46, 47}
, {10, -24, -5}
, {2, -32, 25}
, {22, -50, -32}
, {-26, 9, 43}
, {39, -33, -35}
, {15, 10, 31}
, {-52, -29, -33}
, {-1, 33, -47}
, {9, 54, -1}
, {-39, 49, 39}
, {0, -11, 58}
, {-43, -39, 2}
, {47, 48, 5}
, {11, 20, -2}
, {-33, 33, -32}
, {21, -34, 17}
, {52, 52, 0}
, {-20, -10, -46}
, {-10, 48, -5}
, {21, 29, -17}
, {7, -4, -8}
, {1, -39, 18}
, {1, 14, 20}
, {-35, 36, -30}
, {41, 7, -2}
, {-14, -2, 46}
, {7, 27, -9}
, {16, -29, -43}
, {-17, 22, 5}
, {-27, -23, 30}
, {-13, -27, 25}
, {-28, -20, 16}
, {46, 17, 34}
, {4, 23, -19}
, {-42, -21, 10}
, {-10, 12, 28}
, {-8, 21, -5}
, {-1, -25, -24}
, {-38, 0, 31}
, {38, -32, -22}
, {-9, 0, -41}
}
, {{13, 23, 18}
, {37, -37, 53}
, {24, -24, 7}
, {-18, 52, 61}
, {-31, -17, -13}
, {-23, -1, 28}
, {-44, 45, 43}
, {10, 45, -40}
, {9, -20, 33}
, {33, -9, 15}
, {12, -40, 1}
, {-47, 42, -40}
, {-26, -47, 38}
, {21, -64, 20}
, {-18, -25, -29}
, {36, -19, 24}
, {27, -11, -50}
, {-47, -6, 15}
, {-6, -11, -22}
, {12, 28, 34}
, {34, 47, 21}
, {-11, 55, 29}
, {30, 35, -36}
, {29, -32, 8}
, {-8, -50, -8}
, {-47, -42, -47}
, {-43, 12, -9}
, {40, 43, 12}
, {6, 36, -37}
, {17, -10, -49}
, {43, 24, 3}
, {1, 27, 3}
, {19, 22, 47}
, {-46, 32, -29}
, {47, 0, 46}
, {-45, -23, -47}
, {-27, 26, 24}
, {9, -17, -40}
, {-44, -17, 50}
, {28, -16, -28}
, {16, 51, 47}
, {41, 8, 20}
, {46, -13, 2}
, {40, 18, 6}
, {-30, 13, 14}
, {22, 64, 34}
, {7, -23, 23}
, {-40, -17, -25}
, {1, 7, -37}
, {-6, 12, -13}
, {-41, 30, -38}
, {-20, 41, -10}
, {-53, 61, -19}
, {-17, -24, 50}
, {-36, -47, 4}
, {4, -25, 54}
, {53, 30, -17}
, {4, 29, 5}
, {6, 48, 44}
, {67, 50, 0}
, {-29, 42, -15}
, {-28, 43, 0}
, {0, 3, -6}
, {-32, 55, -31}
}
, {{-26, -32, -10}
, {-45, 2, -50}
, {56, -35, -8}
, {39, -15, 60}
, {5, 7, -40}
, {42, 13, -33}
, {25, 9, -9}
, {-51, 29, -50}
, {44, -43, 12}
, {31, -4, 0}
, {-4, 33, 40}
, {22, 11, -26}
, {-33, -23, 42}
, {-30, 27, -13}
, {-42, -7, -13}
, {33, 21, -25}
, {29, -14, -47}
, {30, -37, 33}
, {17, 14, -48}
, {22, -13, 47}
, {11, -25, -3}
, {34, 21, -3}
, {-54, 33, -35}
, {-9, 31, 14}
, {45, 15, -42}
, {-46, 14, -39}
, {7, 25, 17}
, {-26, -50, 37}
, {-49, -24, 29}
, {7, -37, -49}
, {-13, -48, -40}
, {23, 42, 31}
, {-45, -14, 29}
, {0, 33, -22}
, {-16, -32, -5}
, {-41, 4, 19}
, {41, 35, 35}
, {13, 28, -6}
, {-49, -42, 49}
, {60, 19, 52}
, {-23, 28, -13}
, {-19, -11, 7}
, {30, 39, -20}
, {-31, 10, 13}
, {29, 40, -21}
, {-18, 1, -9}
, {-11, 33, -17}
, {2, -39, 20}
, {47, -14, 46}
, {-20, -50, 28}
, {32, 21, -36}
, {1, 0, 28}
, {0, 13, 25}
, {-12, 7, -3}
, {22, -20, 28}
, {26, 17, 27}
, {12, -11, -21}
, {50, 48, 7}
, {35, -24, 16}
, {4, -31, -33}
, {-3, 43, 19}
, {-28, 5, -5}
, {9, -17, -23}
, {-33, -27, -12}
}
, {{17, -12, -12}
, {30, 1, 43}
, {-1, -5, -16}
, {-21, 45, 16}
, {47, -43, 31}
, {-25, 48, -11}
, {-9, 50, -38}
, {-18, 10, -41}
, {15, 40, 43}
, {43, 57, 31}
, {32, 48, 0}
, {17, -2, 24}
, {54, 16, -25}
, {-21, 51, 26}
, {56, 0, 4}
, {-27, -43, -17}
, {-38, -33, -23}
, {5, 25, -20}
, {-35, 31, -31}
, {-31, -13, -36}
, {-22, -35, 3}
, {-49, -13, 8}
, {1, 13, -35}
, {-33, -21, 25}
, {36, 46, -25}
, {-49, -9, -48}
, {-41, -10, -5}
, {22, 37, -23}
, {-36, 25, -39}
, {-48, 2, -5}
, {33, 37, 24}
, {-42, 12, -12}
, {-44, 2, -29}
, {50, -63, 25}
, {49, 27, -3}
, {4, 47, 43}
, {8, 14, 6}
, {32, 16, -35}
, {43, 12, 29}
, {-3, -20, -29}
, {8, -26, -44}
, {21, 5, 3}
, {-1, 4, 31}
, {-16, 6, 13}
, {-11, -15, 22}
, {48, 0, 54}
, {27, -11, -45}
, {-7, 32, -44}
, {-48, -43, -41}
, {27, -33, -26}
, {-32, -43, -25}
, {-24, -9, -40}
, {-8, -24, -28}
, {51, 14, -10}
, {-42, 12, -25}
, {-17, 45, -46}
, {42, 38, -30}
, {-15, -11, 46}
, {12, -43, -21}
, {54, 2, -25}
, {-33, -24, -47}
, {26, -41, 6}
, {-40, -25, 23}
, {45, 25, 34}
}
, {{62, -19, -4}
, {23, 4, 0}
, {66, 54, -11}
, {17, -44, 20}
, {57, 59, -46}
, {-41, 46, -49}
, {-49, -41, 23}
, {-17, 44, 45}
, {9, -21, 15}
, {-2, 17, 60}
, {-38, 32, 22}
, {-3, -23, -17}
, {33, 41, -37}
, {19, -63, -37}
, {-37, -35, -32}
, {35, 0, -37}
, {-38, 6, 20}
, {36, -14, -39}
, {29, -10, 21}
, {51, -9, 33}
, {25, -30, 0}
, {16, -12, 45}
, {25, -6, 39}
, {-26, 38, -36}
, {-44, 36, 42}
, {24, -46, 49}
, {39, 21, 0}
, {-14, 33, 29}
, {-6, -54, -1}
, {29, -50, -55}
, {-2, -31, -11}
, {-51, -27, 22}
, {-22, -39, -53}
, {-28, 34, 15}
, {53, 4, 41}
, {-51, -6, 9}
, {35, 28, 28}
, {10, 51, 38}
, {49, 44, -31}
, {-33, -27, 67}
, {47, 27, 27}
, {55, 48, 10}
, {50, -16, 35}
, {-22, -50, 9}
, {-59, -46, -42}
, {-32, 18, -12}
, {3, -19, 26}
, {-21, 33, -44}
, {-3, 4, 12}
, {-15, 33, 46}
, {-23, -24, -53}
, {25, 57, 39}
, {37, -18, 50}
, {-9, -37, -37}
, {4, -26, 47}
, {23, -26, 37}
, {9, 2, -4}
, {-36, 1, -34}
, {27, 5, 32}
, {35, 13, -23}
, {33, 3, -2}
, {16, 27, -39}
, {54, 4, 7}
, {18, 5, 26}
}
, {{33, -15, 57}
, {-1, 18, -34}
, {-37, -22, 22}
, {-31, 28, -5}
, {-31, 5, 50}
, {-19, -14, 38}
, {43, 43, 16}
, {6, -11, -3}
, {40, -46, -30}
, {9, 30, 27}
, {46, 9, -31}
, {16, -2, -36}
, {14, 5, 47}
, {-11, 41, -15}
, {-23, 7, -7}
, {-7, -2, 14}
, {-29, -4, 10}
, {-35, 13, 6}
, {-53, 39, -13}
, {40, -46, 13}
, {-4, -55, -15}
, {13, -14, -32}
, {48, -29, -42}
, {62, 34, -21}
, {24, -3, -42}
, {28, 47, -50}
, {10, 49, 9}
, {44, -33, -1}
, {-17, -8, -34}
, {35, -50, 29}
, {-16, 65, 19}
, {-21, -49, -39}
, {3, -26, 20}
, {-22, -11, -41}
, {-37, -6, 58}
, {-52, -20, 12}
, {4, -35, 30}
, {6, -4, -20}
, {22, 20, 25}
, {15, -31, -47}
, {-11, -39, 3}
, {44, -46, -36}
, {-20, 46, -29}
, {-5, 48, -33}
, {-16, -28, -51}
, {-19, 1, -32}
, {-42, 11, 36}
, {-12, -30, 44}
, {-43, 43, 38}
, {9, 0, 41}
, {3, -3, 33}
, {17, 8, -30}
, {34, 17, 31}
, {-29, 14, -61}
, {-5, 43, -32}
, {46, 29, 26}
, {52, -24, 33}
, {28, 26, -31}
, {6, -50, -3}
, {32, -5, 4}
, {-22, 25, 31}
, {-10, -16, 25}
, {55, -1, -9}
, {9, -48, -31}
}
, {{-53, -13, 64}
, {-10, -3, -15}
, {33, -11, 25}
, {-15, -38, 0}
, {11, -18, -28}
, {9, -15, 49}
, {-45, 15, -46}
, {26, 12, 6}
, {19, 46, -40}
, {-21, 15, 39}
, {52, 35, 18}
, {-17, -42, -42}
, {-49, -2, 22}
, {11, -14, 6}
, {-37, 50, 49}
, {46, 19, -23}
, {16, 39, -39}
, {-17, -19, 5}
, {-24, -9, -42}
, {-50, 15, -11}
, {15, -36, -12}
, {-4, 37, -7}
, {-38, 1, -1}
, {-28, 36, 27}
, {-9, -5, 12}
, {31, 12, 26}
, {8, 34, -11}
, {-13, 37, -32}
, {-26, -50, 36}
, {-13, 8, 4}
, {8, 45, -48}
, {-40, 36, 45}
, {-5, 7, 6}
, {-39, -17, 56}
, {-31, 40, -6}
, {36, 7, 5}
, {-27, -30, -5}
, {48, -32, 44}
, {50, -1, 10}
, {-42, -22, 1}
, {43, 44, -50}
, {-40, 2, 41}
, {-5, -20, 29}
, {10, -16, 30}
, {17, 12, 15}
, {-16, -11, 29}
, {-47, 27, -3}
, {-55, -15, 37}
, {-56, 36, 39}
, {8, -44, 0}
, {10, 26, 10}
, {-31, 13, 40}
, {-28, 0, -35}
, {-7, -37, -16}
, {-47, 20, 4}
, {-33, 5, -30}
, {-21, 3, 34}
, {-17, 3, 36}
, {39, 45, -47}
, {58, -31, 64}
, {23, -39, -5}
, {3, -8, 31}
, {43, -20, -48}
, {-45, -14, -57}
}
, {{-29, -7, -15}
, {8, -18, 23}
, {-1, 40, 64}
, {23, 49, 42}
, {26, -40, 14}
, {-36, -20, -50}
, {-26, -29, -48}
, {15, 16, 1}
, {-24, -30, -32}
, {0, -41, 21}
, {-9, -41, -33}
, {40, -12, -34}
, {31, 48, -5}
, {50, -1, 7}
, {-16, 7, 17}
, {39, -38, 0}
, {22, 43, 12}
, {-22, -1, 28}
, {30, -8, 17}
, {-9, -28, 10}
, {21, -33, -25}
, {-27, -47, -5}
, {43, -32, -45}
, {-17, 30, -23}
, {34, 50, -2}
, {-49, 3, -7}
, {-35, -3, -30}
, {-34, -14, -49}
, {-33, 38, 4}
, {47, -3, 32}
, {-15, -24, -15}
, {-18, 24, 25}
, {15, 38, -24}
, {-10, -14, 15}
, {-24, -42, -28}
, {31, -46, 29}
, {37, 37, 47}
, {-39, 24, 18}
, {46, 32, 34}
, {-43, 31, 3}
, {-6, -30, 16}
, {-23, -3, 19}
, {37, -20, -43}
, {-3, 28, 0}
, {32, 3, 17}
, {14, 2, 45}
, {37, 16, -45}
, {-8, -21, 14}
, {37, -37, 12}
, {0, -33, 23}
, {23, -52, -5}
, {25, 32, -12}
, {-36, -20, 5}
, {34, -41, 7}
, {-14, -18, 40}
, {-35, 4, -39}
, {42, 30, 36}
, {15, -17, 35}
, {-50, 33, -37}
, {3, -11, 35}
, {-38, -19, -28}
, {-52, 24, 50}
, {-40, 1, 20}
, {-26, 35, 35}
}
, {{-39, -6, 13}
, {26, 28, 5}
, {-4, -31, -29}
, {-11, 22, -49}
, {-5, 47, 14}
, {13, 8, 11}
, {-39, -6, -23}
, {46, 15, -22}
, {-18, -12, -16}
, {10, -58, 42}
, {56, -52, 13}
, {41, 21, 37}
, {5, -23, -35}
, {-7, -15, 69}
, {-4, 21, -7}
, {-56, -4, 9}
, {43, -50, 20}
, {36, -34, 5}
, {-46, 42, 24}
, {41, 32, -1}
, {-55, 19, 31}
, {27, -5, -47}
, {-26, 6, 2}
, {-5, -28, -35}
, {3, 13, 31}
, {-30, 36, 17}
, {-6, -18, 11}
, {56, 68, 61}
, {6, 0, 43}
, {10, 23, -2}
, {-7, 4, -43}
, {-32, 18, -15}
, {-31, 47, 5}
, {-59, 0, -11}
, {-23, -21, -42}
, {15, 44, 31}
, {9, -11, 33}
, {-17, 34, 30}
, {-4, -34, 16}
, {-15, -46, 21}
, {-23, -25, 14}
, {-47, -41, 16}
, {-45, -40, -41}
, {29, 4, -17}
, {8, -25, 19}
, {28, -34, 24}
, {-48, -37, -38}
, {12, 48, 38}
, {-17, -6, -37}
, {50, 39, 49}
, {8, 55, 45}
, {32, 21, -45}
, {-2, 42, -13}
, {-18, 16, -14}
, {-17, 18, 37}
, {-48, 30, -35}
, {33, -11, 19}
, {-30, 6, -49}
, {6, 35, -7}
, {50, 14, 42}
, {15, 6, 5}
, {27, 32, 6}
, {-35, -50, 47}
, {-24, -54, -23}
}
, {{-32, 31, -6}
, {-40, 49, -31}
, {41, 17, 7}
, {48, -8, 35}
, {-44, -26, -7}
, {14, 20, 47}
, {52, -19, 41}
, {1, 5, 22}
, {-12, 19, 5}
, {29, 18, 0}
, {-30, 7, -32}
, {42, 36, 37}
, {23, -40, 20}
, {-30, 1, 42}
, {32, 8, 49}
, {-9, 37, -15}
, {27, 35, -40}
, {33, -21, 30}
, {-23, -7, -3}
, {-42, -4, 42}
, {1, 26, -54}
, {-13, -30, -23}
, {1, -39, 32}
, {-18, 32, 41}
, {32, -10, 21}
, {40, 33, 29}
, {-24, -42, -12}
, {49, 26, -40}
, {-46, -50, 7}
, {-14, 26, -5}
, {57, -38, -49}
, {32, 30, -7}
, {10, -25, -52}
, {30, 13, -15}
, {35, -52, -47}
, {15, -26, -33}
, {23, 40, -19}
, {4, -38, -47}
, {11, 28, 47}
, {42, -30, 8}
, {28, 39, -12}
, {33, 32, -22}
, {-41, 22, -13}
, {-8, -21, 22}
, {-31, -1, -48}
, {46, -41, 28}
, {8, 36, -28}
, {12, 21, -40}
, {42, -29, 5}
, {-36, 12, -34}
, {-40, -16, 14}
, {-6, 0, 37}
, {-34, 0, -41}
, {62, -39, 28}
, {-41, 22, -44}
, {-19, -35, -50}
, {37, -22, 60}
, {11, 8, -33}
, {22, -21, -49}
, {-51, 11, 46}
, {-34, 1, -37}
, {-3, 26, 24}
, {18, -10, 29}
, {-14, -11, -14}
}
, {{-22, 30, 31}
, {50, -40, -20}
, {-1, -27, -16}
, {7, -5, -19}
, {-4, -20, 32}
, {-16, 14, 45}
, {-42, 0, 8}
, {-13, -24, 13}
, {44, 44, -30}
, {-31, 36, 3}
, {-26, -19, 26}
, {15, 20, -45}
, {35, 33, 9}
, {7, -1, 18}
, {-49, -43, 9}
, {-4, 28, 30}
, {42, -25, -24}
, {-40, 32, 0}
, {-16, 47, -47}
, {-6, 51, -24}
, {7, 0, -8}
, {20, 40, -15}
, {-47, -40, 4}
, {-24, 1, -66}
, {51, -2, 46}
, {-30, -13, -12}
, {-19, 33, 21}
, {43, -35, 36}
, {-45, 0, 16}
, {49, 16, -52}
, {-41, -65, 37}
, {42, -15, -31}
, {8, -18, 28}
, {6, -42, 4}
, {-49, -52, 36}
, {-28, 29, 50}
, {53, 14, -40}
, {-29, -13, 10}
, {-24, 6, -28}
, {-31, 40, 17}
, {27, -6, 34}
, {17, -43, -20}
, {-9, 24, 45}
, {48, 41, -13}
, {-29, -20, -5}
, {-18, -50, 43}
, {-23, -28, 4}
, {-41, 1, -30}
, {-16, -35, -48}
, {20, -18, -32}
, {-15, 17, 37}
, {-50, 4, 13}
, {52, -1, 26}
, {16, -30, -33}
, {5, -18, 4}
, {-43, 35, -23}
, {-56, 25, 36}
, {-46, 25, -29}
, {-5, -15, 30}
, {18, 13, -10}
, {-13, 10, 31}
, {15, -16, -41}
, {5, -35, 25}
, {37, 50, 14}
}
, {{33, -4, -35}
, {-40, 7, -48}
, {-52, 0, -2}
, {65, -25, 2}
, {-21, -51, 2}
, {-14, 51, -27}
, {-34, -22, 28}
, {-7, -52, 9}
, {32, -22, -9}
, {42, 3, -48}
, {62, -24, -2}
, {-66, -20, 33}
, {41, 26, 54}
, {-16, -30, 56}
, {-32, -26, -30}
, {18, 33, 49}
, {-22, -19, 51}
, {53, -11, 15}
, {5, 43, -34}
, {-33, 3, 57}
, {-42, -19, -59}
, {-19, -9, 39}
, {10, 39, 36}
, {22, 24, 1}
, {10, -45, 13}
, {-47, 42, -9}
, {7, 32, 26}
, {13, -36, -5}
, {47, -5, 52}
, {-4, -33, 10}
, {14, 17, 1}
, {4, 9, -31}
, {2, -17, -38}
, {-37, 29, 3}
, {-12, -9, 43}
, {17, 36, -43}
, {-34, -35, 9}
, {31, 23, -11}
, {-5, 25, 7}
, {58, 32, -45}
, {-6, -34, 16}
, {27, -17, -6}
, {49, 26, -3}
, {-44, -35, 10}
, {49, -42, 9}
, {-5, -16, -41}
, {0, 47, -47}
, {-1, -4, 48}
, {-25, 3, -45}
, {-39, -42, -25}
, {39, 45, -30}
, {-36, 7, -19}
, {-2, 62, 60}
, {-12, 59, -3}
, {-50, -43, 18}
, {-31, 45, -8}
, {-27, -53, -10}
, {18, -33, -37}
, {1, 32, -10}
, {8, -26, -12}
, {14, 57, -43}
, {15, 17, 3}
, {35, -16, -15}
, {-18, -53, -22}
}
, {{9, -39, 18}
, {-20, 53, 55}
, {-13, 28, 22}
, {-7, 55, 0}
, {-46, -28, 2}
, {12, -41, 56}
, {-8, -7, 42}
, {27, 15, 10}
, {0, 31, 18}
, {33, 4, 20}
, {-5, -28, -30}
, {-41, 59, 42}
, {-5, -51, 17}
, {16, -39, 28}
, {-34, -50, -50}
, {-35, -13, -1}
, {23, 21, -51}
, {-2, -42, -15}
, {-10, 35, 40}
, {-24, 13, 10}
, {56, 1, 48}
, {16, -13, 23}
, {-20, -26, -45}
, {14, 29, 47}
, {-44, 22, -26}
, {-10, -28, 24}
, {13, -42, -20}
, {-12, 11, -20}
, {0, -7, -16}
, {-45, 51, 15}
, {17, -46, -8}
, {-34, -47, -6}
, {-31, 27, -31}
, {15, -9, 1}
, {-34, 53, -23}
, {-17, 0, -43}
, {24, -50, 3}
, {34, 16, 33}
, {3, 5, 40}
, {-38, 54, 52}
, {-13, -37, -4}
, {2, 41, -28}
, {-34, -18, 37}
, {27, 29, 24}
, {-49, -53, -19}
, {24, 29, 61}
, {29, 60, 7}
, {-4, 37, 55}
, {-52, 38, 30}
, {45, -39, 45}
, {-8, -25, -9}
, {26, -19, 54}
, {44, -3, 58}
, {24, -14, -36}
, {-16, -5, 11}
, {23, -18, 6}
, {18, 50, -42}
, {2, -5, -26}
, {-38, -30, -45}
, {-17, 16, -11}
, {39, 53, 50}
, {-23, -44, 8}
, {37, 0, -14}
, {13, -42, 67}
}
, {{-29, 1, 8}
, {17, 18, 41}
, {-62, -50, 13}
, {-32, -47, 8}
, {2, -18, -12}
, {18, -26, -2}
, {-43, -39, 14}
, {-20, -45, 1}
, {5, -17, 4}
, {-50, 22, -5}
, {28, -54, -67}
, {-32, 4, -34}
, {-23, 9, 28}
, {26, 52, 50}
, {10, -27, -50}
, {48, 25, -29}
, {14, -20, 16}
, {-5, 36, 45}
, {13, -4, -3}
, {44, 24, -9}
, {-26, 36, 24}
, {-33, -26, 16}
, {28, 37, -13}
, {21, 25, -12}
, {46, -42, 9}
, {34, 47, -46}
, {39, -26, 0}
, {44, 2, 39}
, {-6, 21, 0}
, {7, -27, 53}
, {-37, -49, 2}
, {-4, 11, 51}
, {-26, 8, -42}
, {-49, 35, 33}
, {45, -40, 43}
, {-26, -26, -18}
, {-14, 35, -26}
, {11, 6, -35}
, {-13, 44, 16}
, {26, -52, 18}
, {36, 14, 44}
, {-21, -13, -25}
, {43, 36, 48}
, {10, 48, -38}
, {-42, 36, 56}
, {0, -5, 11}
, {-18, 4, 33}
, {25, 28, -46}
, {44, 26, 9}
, {36, -48, -19}
, {8, 16, 13}
, {0, -11, 30}
, {2, 35, 43}
, {-3, 37, 45}
, {43, -47, 27}
, {29, -49, 10}
, {-52, 32, -43}
, {-7, -26, 9}
, {45, 15, 40}
, {23, -1, -51}
, {-18, -3, -13}
, {2, -34, -1}
, {-30, 23, 30}
, {47, 22, -14}
}
, {{35, 43, 40}
, {-3, 16, -8}
, {-16, -32, -10}
, {43, -52, 9}
, {-26, 27, -11}
, {-31, 36, -38}
, {53, -44, -8}
, {0, 44, 23}
, {8, 1, 41}
, {-9, 8, -14}
, {62, 53, 42}
, {-23, 45, 56}
, {45, 58, -17}
, {-21, 26, -38}
, {61, -9, 46}
, {33, -32, -44}
, {-6, 18, -8}
, {36, 51, 0}
, {-6, -11, 53}
, {13, 43, 24}
, {-36, 27, 20}
, {-19, 0, 26}
, {15, 59, 31}
, {0, 29, -4}
, {-20, 10, -4}
, {43, 10, 36}
, {-35, -25, 22}
, {52, -5, -42}
, {-34, -44, 46}
, {27, -33, -24}
, {14, 55, 8}
, {-4, -35, 44}
, {-34, 29, -56}
, {10, -39, -24}
, {-26, 38, 9}
, {-24, -34, -47}
, {-34, -2, 21}
, {4, -18, -31}
, {-41, -42, -11}
, {9, -27, -14}
, {-11, 15, 5}
, {6, -44, -31}
, {32, -26, 11}
, {15, -23, -56}
, {-8, -35, -23}
, {-37, -50, 1}
, {22, -37, -6}
, {-20, -38, 8}
, {14, -33, 21}
, {15, 38, -18}
, {-34, -5, 42}
, {36, -17, -31}
, {-7, -3, 35}
, {33, -45, 23}
, {58, 37, -26}
, {60, 18, 15}
, {-27, 58, -12}
, {26, -20, -37}
, {-36, 15, -22}
, {-6, -26, -6}
, {18, -8, -22}
, {63, 17, -18}
, {5, 12, -27}
, {-61, 3, -3}
}
, {{48, -33, 32}
, {-31, -5, -32}
, {20, 8, 2}
, {27, 35, 12}
, {39, -27, -18}
, {51, -31, -26}
, {-38, 24, 3}
, {8, 28, 32}
, {-33, -42, -36}
, {-23, -36, 45}
, {45, 16, 48}
, {42, -35, -37}
, {-27, 4, -3}
, {-18, 59, -42}
, {33, -11, 21}
, {-34, -40, 0}
, {-25, -37, 21}
, {39, -49, -16}
, {-1, -37, 10}
, {46, 17, -28}
, {14, 3, -2}
, {-38, 47, -37}
, {-14, -28, -13}
, {47, 14, 28}
, {-26, 45, -51}
, {-34, -36, 34}
, {-22, 50, -26}
, {-9, 3, 46}
, {3, -43, 25}
, {-22, 0, -22}
, {5, -8, 7}
, {-27, 27, 47}
, {4, -20, 24}
, {41, 6, -25}
, {1, 35, 56}
, {28, 29, 12}
, {6, -19, 3}
, {15, 31, -48}
, {-49, -31, -39}
, {-20, -13, -30}
, {-48, -27, 8}
, {1, -7, 39}
, {-9, 27, 1}
, {16, 42, -52}
, {1, -36, -53}
, {67, -39, 30}
, {9, 11, 14}
, {38, 8, -25}
, {-6, -24, 23}
, {-11, 33, -31}
, {-3, -41, 44}
, {48, 23, 45}
, {-41, 2, -47}
, {20, -4, -43}
, {20, 20, 33}
, {-24, 3, -32}
, {44, 37, -33}
, {5, -37, -20}
, {31, -24, 51}
, {1, 34, 56}
, {47, 26, -22}
, {24, -33, 30}
, {21, -50, -35}
, {17, -21, 20}
}
, {{-20, 49, 30}
, {12, -20, 24}
, {46, 35, 6}
, {-20, -58, -15}
, {-4, 42, 14}
, {8, -51, -35}
, {-44, 1, 35}
, {-11, 50, -41}
, {-49, -17, 47}
, {67, -38, -6}
, {-16, -46, 3}
, {22, -51, -23}
, {11, -40, 29}
, {50, -30, 21}
, {-18, 34, 3}
, {10, 22, 47}
, {-9, -1, 31}
, {-29, 48, 20}
, {35, 10, -34}
, {-11, 26, 48}
, {12, -24, 42}
, {-29, 9, -47}
, {-4, -31, 19}
, {12, -8, 41}
, {36, -1, 7}
, {44, -22, 27}
, {-12, 14, 43}
, {13, 29, 21}
, {42, -4, -14}
, {-47, -28, -41}
, {-5, 16, -5}
, {50, 39, 0}
, {26, 11, -47}
, {38, 35, -42}
, {-34, 7, -23}
, {-12, -33, -21}
, {45, 19, 6}
, {23, 26, -46}
, {-6, -49, -11}
, {28, 23, 38}
, {42, -18, -8}
, {14, -6, 28}
, {-48, 3, -3}
, {-1, 13, -29}
, {46, 6, -8}
, {-60, -55, -64}
, {-44, -37, 5}
, {-39, -3, 48}
, {37, 31, -8}
, {42, 35, 14}
, {18, -4, -44}
, {38, 27, 0}
, {-37, -6, 24}
, {13, -55, -57}
, {47, -15, 52}
, {4, -35, 0}
, {-44, -38, 22}
, {13, 41, 37}
, {-20, 16, -45}
, {-30, 35, 38}
, {25, 47, 26}
, {-8, -46, -11}
, {-8, 44, 50}
, {-42, -29, -19}
}
, {{14, -57, 42}
, {-4, 37, -6}
, {51, -32, 29}
, {6, 57, -36}
, {18, 39, 13}
, {53, 25, 51}
, {-11, 27, -6}
, {-49, 26, -15}
, {7, 19, -31}
, {-1, 26, 45}
, {16, 11, -49}
, {1, -27, -27}
, {-20, 42, -33}
, {-32, 55, -9}
, {-12, 62, -26}
, {-41, -23, 8}
, {44, 3, -35}
, {26, -1, 46}
, {-31, 45, 49}
, {7, -42, -5}
, {0, 33, 31}
, {-37, -15, -1}
, {18, -4, -19}
, {22, -58, -20}
, {1, 2, -17}
, {23, -34, 3}
, {-13, 23, 4}
, {-8, -53, -38}
, {23, -30, -5}
, {-5, -25, -21}
, {-18, -11, 3}
, {-11, 52, -14}
, {-16, -1, 7}
, {-11, -4, -2}
, {31, -33, 8}
, {8, -2, -3}
, {-26, 53, 16}
, {44, 8, -57}
, {-37, 30, -20}
, {0, -35, -27}
, {33, 16, -11}
, {-31, -33, 18}
, {-32, 8, -7}
, {-19, 28, 7}
, {0, -33, -34}
, {46, 44, -32}
, {39, -17, -5}
, {-45, -25, 43}
, {-1, 19, -11}
, {-21, -5, -22}
, {58, -24, -38}
, {-50, -17, -42}
, {11, 37, -40}
, {41, 41, 50}
, {-14, 44, -47}
, {30, -20, 53}
, {-38, -42, -54}
, {-4, -21, 3}
, {12, 10, 2}
, {53, -19, -31}
, {46, -27, 52}
, {21, 28, 4}
, {-15, 17, 40}
, {-33, -47, -18}
}
, {{-40, -27, -47}
, {-40, 25, 29}
, {42, 6, 0}
, {4, -11, 18}
, {8, 7, 12}
, {-40, 14, 24}
, {-46, -3, 40}
, {39, -32, 21}
, {-2, 35, 30}
, {34, 7, -61}
, {50, -9, 19}
, {50, 54, 43}
, {-24, 35, 37}
, {36, 3, -1}
, {22, -39, -10}
, {8, -21, 38}
, {7, 13, -48}
, {23, 17, -49}
, {32, 1, -26}
, {-29, 24, 33}
, {-34, -1, 44}
, {0, -28, 2}
, {-8, -50, -10}
, {14, -29, -30}
, {-45, 10, -37}
, {51, 25, 19}
, {14, 10, -18}
, {29, -5, 38}
, {19, -10, -35}
, {38, 0, -49}
, {-44, 33, -53}
, {27, 12, -47}
, {-31, -27, 43}
, {-23, -25, 4}
, {4, 52, 43}
, {36, 9, 28}
, {-49, -40, 0}
, {28, -13, 43}
, {-18, 2, 1}
, {-36, -18, 28}
, {-18, 16, 17}
, {-38, -41, 12}
, {-24, 12, 7}
, {10, 9, -20}
, {-26, -28, 43}
, {3, -43, -25}
, {41, 40, 17}
, {24, -12, 39}
, {-29, 0, 7}
, {37, -14, 40}
, {27, 26, -7}
, {-37, -47, 26}
, {-54, -12, 39}
, {-14, 35, 4}
, {30, 44, -37}
, {-2, 39, 10}
, {6, 49, 1}
, {6, -33, -62}
, {-28, -2, 41}
, {-18, -7, -12}
, {-32, 3, 7}
, {-58, -25, -9}
, {11, -23, -19}
, {-43, 58, 53}
}
, {{-18, 42, -39}
, {-31, 37, -13}
, {-7, 22, -30}
, {58, 5, 1}
, {32, 29, -30}
, {14, 31, -46}
, {-31, -48, 0}
, {-6, -2, -40}
, {50, 5, -19}
, {31, 47, 52}
, {0, 28, 38}
, {3, 41, -17}
, {21, 8, 23}
, {0, -14, -4}
, {-21, 53, -47}
, {39, -34, 24}
, {0, 47, 42}
, {0, 28, 51}
, {8, -47, -31}
, {12, 53, -47}
, {8, 19, -39}
, {11, -30, -40}
, {-1, 6, 18}
, {-32, 24, 14}
, {30, -8, 14}
, {15, -46, -40}
, {-21, 4, 33}
, {-23, -4, -49}
, {-32, 39, 43}
, {0, 33, -31}
, {8, -22, -22}
, {0, -10, -43}
, {52, -30, 55}
, {38, 12, -10}
, {46, -29, -20}
, {39, -34, 0}
, {-26, -34, -28}
, {-36, 3, -14}
, {37, -21, -16}
, {0, 17, -2}
, {9, 47, 53}
, {52, 7, 32}
, {33, 0, 7}
, {-24, -31, -14}
, {-16, 13, 41}
, {-37, -22, -26}
, {18, -23, 0}
, {3, -13, -33}
, {23, 3, -45}
, {4, 4, 37}
, {52, -30, 46}
, {-3, -8, -26}
, {-36, 36, -35}
, {13, 51, 66}
, {-27, -40, 16}
, {-4, -20, -7}
, {-48, 14, 32}
, {-54, -31, 43}
, {-16, 36, 17}
, {-31, 53, 7}
, {38, 42, 5}
, {-39, 47, 10}
, {36, 26, 25}
, {12, -41, 33}
}
, {{-28, -47, -5}
, {10, -17, -23}
, {-51, -21, 19}
, {-2, -29, -15}
, {-41, -5, -55}
, {17, -46, -1}
, {-43, 44, 30}
, {17, 32, -50}
, {25, -9, 47}
, {51, 42, 65}
, {53, 55, 10}
, {-44, 37, 15}
, {24, 30, -19}
, {65, 53, 60}
, {-7, 11, 8}
, {33, 24, 0}
, {27, -24, 8}
, {-17, -5, 23}
, {29, 8, -31}
, {-9, -32, -16}
, {13, -43, -53}
, {-19, 15, -32}
, {1, -2, -30}
, {-33, -24, 46}
, {26, 42, 45}
, {28, -11, -34}
, {49, -13, 45}
, {48, 5, -3}
, {30, 11, 48}
, {-24, 16, 41}
, {24, 26, 37}
, {-19, 36, 50}
, {12, 19, -38}
, {-7, -16, -52}
, {35, 23, -62}
, {38, -27, 23}
, {0, -41, 40}
, {31, 37, -37}
, {2, -9, 29}
, {28, -65, -39}
, {-13, 8, -18}
, {43, -55, -31}
, {24, 20, 44}
, {3, 27, 18}
, {15, 48, -27}
, {-14, 15, -17}
, {-45, 29, 25}
, {24, -15, -27}
, {-47, 8, -30}
, {-36, 26, -44}
, {22, -34, -42}
, {-37, 20, -12}
, {-14, 34, 32}
, {21, 35, 29}
, {0, -3, 24}
, {-55, -66, 17}
, {11, -56, -35}
, {-47, 17, 16}
, {-1, -38, 34}
, {9, -26, -51}
, {-9, 19, 30}
, {17, 16, 42}
, {-51, 36, -15}
, {-53, -43, -56}
}
, {{-25, 43, -25}
, {-12, -2, 24}
, {24, -42, 13}
, {31, -56, -15}
, {-38, -32, -22}
, {-17, 35, -32}
, {26, -34, -9}
, {30, 10, 15}
, {38, -43, 0}
, {22, 1, -17}
, {-34, -23, -34}
, {40, -10, -46}
, {15, -48, -12}
, {2, -2, -17}
, {8, 9, 45}
, {-26, 16, 21}
, {23, 13, 50}
, {30, 12, -40}
, {-2, -49, 43}
, {15, 36, 29}
, {-51, 7, 24}
, {-9, 0, 6}
, {45, -5, 5}
, {27, -22, 35}
, {-49, -46, -28}
, {42, -46, 37}
, {-47, -35, -45}
, {5, 41, -31}
, {24, -23, -9}
, {-30, -4, -29}
, {-20, -17, -39}
, {-17, -11, -56}
, {40, 4, -37}
, {34, -29, 33}
, {-12, -14, 42}
, {19, -13, -12}
, {23, -49, -53}
, {8, -45, 28}
, {41, 16, -42}
, {-18, 40, 33}
, {-25, -23, -45}
, {9, 25, 21}
, {-35, 6, 11}
, {-21, 17, 27}
, {29, 28, 42}
, {-36, -38, -18}
, {-48, -29, -36}
, {-23, 38, -54}
, {-53, -3, -10}
, {19, -45, -5}
, {0, -19, -17}
, {-35, 34, -11}
, {4, 15, -23}
, {29, -50, 19}
, {37, 30, -51}
, {-44, -47, -58}
, {-20, 9, 0}
, {17, 21, 47}
, {18, 43, 35}
, {-43, 2, -42}
, {-12, -28, -51}
, {36, -10, 26}
, {-7, 22, -46}
, {11, -15, -34}
}
, {{-10, -2, -30}
, {-42, 3, 6}
, {-39, -40, 8}
, {-26, -39, -4}
, {0, -49, 64}
, {23, 22, -26}
, {36, 5, 15}
, {54, -15, -27}
, {-25, 16, 49}
, {21, 33, -14}
, {-27, -39, 4}
, {-36, 26, -53}
, {-30, 51, -41}
, {41, 64, -17}
, {0, 19, 36}
, {-54, 1, -4}
, {-34, -41, -49}
, {43, -2, -30}
, {-20, -40, 40}
, {31, 0, 11}
, {-3, -45, 44}
, {-28, -34, 22}
, {35, -19, -8}
, {-49, 17, 0}
, {-34, 55, 43}
, {2, 40, 30}
, {25, 15, -3}
, {61, 13, 53}
, {-21, 16, -18}
, {9, 55, 53}
, {18, 5, 16}
, {13, 11, -45}
, {13, 31, 22}
, {33, -38, -22}
, {-29, -49, -52}
, {39, -9, 26}
, {-34, -26, 26}
, {19, -27, -6}
, {26, -42, -38}
, {-47, -59, -25}
, {47, 32, -50}
, {-33, -22, 24}
, {-31, -7, -6}
, {52, 22, -26}
, {42, 0, -18}
, {-27, -23, -27}
, {19, 9, 15}
, {7, -9, -41}
, {45, -25, 22}
, {36, -31, 0}
, {-19, -10, -20}
, {-23, 24, -11}
, {7, -52, -36}
, {-44, -19, -28}
, {15, 8, -30}
, {-55, 32, -52}
, {9, -20, 22}
, {-31, 35, 32}
, {41, 46, 39}
, {13, -27, -7}
, {0, 13, -4}
, {0, 22, -3}
, {33, -20, 32}
, {-23, -28, -34}
}
, {{22, 10, -11}
, {-30, 32, 39}
, {-14, -52, 12}
, {38, 3, -36}
, {47, 17, 48}
, {-16, 12, 6}
, {14, 32, -43}
, {40, -4, 32}
, {-35, 47, 18}
, {6, -27, -3}
, {-36, -27, 4}
, {56, -2, 48}
, {-50, -38, -20}
, {30, -22, -8}
, {-24, 6, 45}
, {18, -40, -54}
, {4, -27, -34}
, {-41, 16, 26}
, {1, -9, 38}
, {-25, 15, -27}
, {-18, -32, 39}
, {12, 4, -15}
, {3, -8, -26}
, {9, 55, 14}
, {19, 30, -44}
, {6, 20, 45}
, {-51, -6, -4}
, {-34, 42, -7}
, {-46, 39, -24}
, {-33, 37, 49}
, {15, 38, 17}
, {-20, -13, 45}
, {-55, -22, 4}
, {-29, -6, 25}
, {0, 36, 33}
, {-1, 31, 25}
, {25, -2, 19}
, {47, -49, -16}
, {2, -16, -9}
, {5, 29, 26}
, {-31, 6, 11}
, {10, -3, -19}
, {-4, -51, 40}
, {-35, 38, -23}
, {-17, -13, 39}
, {-20, 9, 10}
, {-21, -40, -20}
, {-24, 37, 9}
, {-28, 13, 36}
, {15, 2, -1}
, {-49, 22, 26}
, {-31, 38, 4}
, {-34, -30, -20}
, {-36, -5, 16}
, {-26, -51, -45}
, {44, 42, 48}
, {0, -42, 54}
, {21, -34, -11}
, {10, -2, 28}
, {5, -36, -34}
, {23, -25, -35}
, {-17, 33, 27}
, {57, 64, 44}
, {14, 32, -17}
}
, {{-38, -56, 32}
, {-46, 27, -39}
, {-54, -41, -26}
, {49, 11, 22}
, {0, -49, 9}
, {28, -39, -44}
, {-35, 3, 49}
, {39, 19, -13}
, {-10, 40, -20}
, {-10, -28, -59}
, {0, 11, 34}
, {-44, -32, -31}
, {0, -29, 38}
, {-4, 57, 18}
, {29, -16, -39}
, {39, 23, -21}
, {-9, -14, 11}
, {-23, 23, -8}
, {-2, 47, 38}
, {29, -32, 21}
, {38, -44, 33}
, {-19, 13, 27}
, {25, -36, -25}
, {-62, -40, -55}
, {-41, -48, -6}
, {5, 13, -40}
, {39, 44, -6}
, {-24, 5, 1}
, {11, -40, 43}
, {43, -31, -31}
, {-50, 23, 33}
, {-41, 45, -41}
, {2, 16, 4}
, {-53, -34, -36}
, {20, -58, -14}
, {-44, 13, -3}
, {-34, 38, 48}
, {7, 0, -13}
, {4, -3, -33}
, {19, 16, 30}
, {46, 26, 33}
, {-37, -43, -1}
, {38, 4, -14}
, {1, -44, -43}
, {31, -26, 29}
, {17, 32, -10}
, {-35, 42, 0}
, {7, 7, 19}
, {27, 36, 51}
, {23, -5, -11}
, {-36, -32, -33}
, {-57, -35, -33}
, {11, -1, 0}
, {11, 62, 12}
, {-26, -51, -21}
, {37, -2, 20}
, {-36, 0, 24}
, {17, -36, -27}
, {-13, 43, 48}
, {19, -20, -33}
, {-5, -26, 22}
, {13, -13, 12}
, {-26, 33, -51}
, {-36, -3, 14}
}
, {{17, -37, -15}
, {-42, -25, -8}
, {29, -5, 6}
, {33, 33, -19}
, {-51, 34, 31}
, {-46, -9, -32}
, {-26, 48, -45}
, {43, -5, 14}
, {-40, -47, -13}
, {-2, -21, -24}
, {-40, -10, -40}
, {-10, -43, 39}
, {36, 45, -9}
, {18, 31, -31}
, {42, 13, -47}
, {-7, 38, 20}
, {13, -47, 28}
, {16, 33, 40}
, {39, 34, -25}
, {44, 35, 18}
, {-25, -34, 8}
, {43, 33, -28}
, {38, -30, -44}
, {15, -20, -17}
, {50, -38, 24}
, {13, -30, 26}
, {41, -36, 14}
, {-37, 43, 55}
, {-34, 31, 42}
, {49, -7, 1}
, {28, 9, -29}
, {4, 0, 13}
, {26, 13, -9}
, {-49, -26, -38}
, {17, 32, -15}
, {40, 41, -32}
, {-14, 7, 51}
, {-16, -22, 0}
, {0, 17, 0}
, {-18, -61, 41}
, {-23, -16, -9}
, {-38, -42, -2}
, {-42, 28, 9}
, {-13, 3, -12}
, {41, 43, -26}
, {-35, -10, 33}
, {-4, -26, -48}
, {-9, -28, -13}
, {-18, -15, 0}
, {20, -10, 16}
, {-15, 21, 45}
, {-1, -38, -22}
, {-53, 38, -31}
, {-26, 53, -7}
, {-27, 26, -36}
, {29, -17, -57}
, {-52, 0, -63}
, {32, -30, -52}
, {38, 7, -36}
, {33, 25, -64}
, {19, 38, 24}
, {25, -47, -54}
, {26, -44, -20}
, {-12, -3, -3}
}
, {{4, 38, -27}
, {20, -14, 28}
, {-29, 30, 30}
, {-23, -14, -21}
, {42, -28, -36}
, {-29, -45, 6}
, {0, -26, 24}
, {24, 40, 30}
, {-35, 35, -8}
, {-3, 59, 51}
, {-36, 45, -14}
, {-13, 13, 35}
, {-37, -15, 30}
, {-29, 12, 23}
, {-40, 3, 4}
, {-6, 48, -11}
, {-6, 49, 10}
, {-36, 15, -12}
, {11, 46, -25}
, {-41, -21, 28}
, {2, -51, 34}
, {43, 21, 40}
, {15, -50, -16}
, {28, 45, -19}
, {39, -29, -20}
, {23, -3, -3}
, {-12, 45, 23}
, {17, -41, -24}
, {-18, -21, 45}
, {-37, 26, -9}
, {37, -48, -23}
, {10, -36, -45}
, {38, 36, -33}
, {-27, 21, -23}
, {48, -42, -2}
, {-41, -31, 20}
, {38, -17, -41}
, {40, 51, 19}
, {-19, -39, 7}
, {49, -29, 34}
, {-15, -35, -10}
, {-11, -25, 21}
, {19, -39, 33}
, {-14, -5, -15}
, {42, -20, 37}
, {-36, 8, -34}
, {-51, -4, -41}
, {-23, 4, 23}
, {-4, 33, 44}
, {16, 32, -30}
, {31, 32, 5}
, {6, -34, 38}
, {45, 47, -60}
, {11, 28, 54}
, {-7, 37, -37}
, {24, 40, -14}
, {49, -30, 57}
, {-37, 45, -42}
, {9, -30, -41}
, {54, 21, 25}
, {21, -33, -7}
, {-5, 55, -5}
, {16, 41, 36}
, {1, 1, 19}
}
, {{28, -41, 5}
, {-36, 49, 18}
, {-42, -25, -45}
, {-16, -33, 33}
, {-45, 24, 6}
, {-20, -30, 8}
, {-42, -38, -5}
, {2, -34, 30}
, {-41, -37, -16}
, {16, -11, 31}
, {-28, -21, -16}
, {16, -35, 16}
, {44, -34, 9}
, {47, 19, 27}
, {-51, 44, 47}
, {-28, -32, -8}
, {-24, -15, -47}
, {11, -42, -10}
, {-35, 13, 1}
, {16, -6, 25}
, {12, 22, -6}
, {35, -10, 46}
, {-15, -43, 28}
, {16, -51, 51}
, {-25, -38, 43}
, {25, -29, 38}
, {8, -30, -22}
, {21, -36, -34}
, {15, 5, -10}
, {-1, -11, -39}
, {-26, -45, 8}
, {-29, 9, -44}
, {-40, 37, -32}
, {-9, -35, 29}
, {-14, -31, 14}
, {37, -11, -6}
, {-21, -4, -29}
, {-21, -36, 36}
, {7, 8, -34}
, {-2, -34, -47}
, {-17, 19, 15}
, {32, -6, -40}
, {44, 43, 12}
, {0, -8, 44}
, {24, -13, -2}
, {-16, 45, -9}
, {-46, 25, 19}
, {7, -17, -2}
, {7, -19, -47}
, {-5, -50, 21}
, {39, 2, 49}
, {-41, -46, 38}
, {-41, -37, 48}
, {-8, -49, 25}
, {-41, 39, 16}
, {-34, 8, -14}
, {4, 13, -35}
, {40, -4, -29}
, {39, -44, 41}
, {14, -27, 7}
, {4, 14, -26}
, {44, -17, -41}
, {-24, -32, 40}
, {-44, 28, -28}
}
, {{-17, 21, 11}
, {31, 35, 9}
, {-43, -58, -16}
, {6, -11, -5}
, {-46, -39, -58}
, {12, 49, -21}
, {-18, -7, 2}
, {28, 41, -4}
, {3, 38, 9}
, {3, 28, -18}
, {-37, 10, 33}
, {-17, 3, -56}
, {53, 52, 2}
, {-4, 10, 18}
, {6, -51, -50}
, {-27, -46, -31}
, {6, -18, -42}
, {-20, 46, -17}
, {-30, 28, -3}
, {15, -22, -14}
, {-28, -26, 11}
, {15, -11, -44}
, {29, -41, -9}
, {-10, -20, -53}
, {10, 0, -20}
, {-10, -17, 25}
, {-42, 31, -28}
, {26, 32, 0}
, {-14, -4, 41}
, {-3, 19, 1}
, {-47, -22, -4}
, {-10, -4, -8}
, {16, 49, 10}
, {11, -44, 36}
, {0, 11, -45}
, {-24, -6, -31}
, {-29, -43, -3}
, {5, -11, 42}
, {-41, 10, 39}
, {3, 39, -17}
, {-4, 28, -35}
, {41, 47, -22}
, {-32, 37, -29}
, {34, -20, -18}
, {30, 22, 18}
, {-12, -12, 1}
, {42, -31, 38}
, {-25, -6, -25}
, {-27, -12, 14}
, {-27, 34, 38}
, {49, -7, 41}
, {48, 18, -13}
, {22, -3, -34}
, {49, 50, 50}
, {17, 36, -46}
, {-6, -36, 21}
, {21, 26, -34}
, {-1, 49, -15}
, {34, -3, 33}
, {-25, -31, -50}
, {37, -50, 40}
, {-35, -15, 0}
, {-59, -21, -2}
, {2, 38, -48}
}
, {{9, 46, 36}
, {28, -47, -31}
, {30, -26, -26}
, {28, 36, 49}
, {-17, -22, 6}
, {42, 22, -18}
, {10, -33, -17}
, {27, -44, -40}
, {47, -44, -4}
, {-17, 50, 69}
, {13, 22, -26}
, {54, 6, 39}
, {-36, -27, 33}
, {22, -52, -47}
, {56, 18, 56}
, {61, -2, -23}
, {-28, -22, 32}
, {-25, -7, -40}
, {52, 57, 60}
, {16, 25, 8}
, {26, 11, -30}
, {27, 37, 31}
, {-33, 0, -18}
, {0, 38, 65}
, {5, 15, 30}
, {48, 15, -18}
, {-6, 17, 34}
, {-45, -5, -29}
, {-24, 6, 41}
, {31, 2, 20}
, {-36, 20, 2}
, {-26, -9, -42}
, {40, -14, 32}
, {35, -40, 17}
, {-21, 41, -31}
, {19, -28, -20}
, {-6, -11, 57}
, {-23, -24, 25}
, {-38, 39, -12}
, {-27, 30, 49}
, {1, -7, 29}
, {33, -28, 40}
, {12, 0, 57}
, {37, 10, -29}
, {-33, 52, -8}
, {0, 38, -27}
, {55, 56, 40}
, {41, -9, -3}
, {-48, -35, -5}
, {-1, -18, 6}
, {-19, 14, -39}
, {-1, -34, 46}
, {30, -16, 14}
, {-2, -16, -43}
, {-26, -4, -37}
, {-4, 2, 42}
, {4, -14, 36}
, {58, -19, -22}
, {-41, 2, 13}
, {29, 4, -26}
, {5, 1, 3}
, {24, 26, 1}
, {7, 3, 8}
, {34, -20, 33}
}
, {{39, 31, 49}
, {25, -38, 25}
, {30, 25, 35}
, {-17, -20, -19}
, {62, 33, -2}
, {-24, 36, -38}
, {33, 0, 3}
, {-40, 22, -19}
, {17, -43, 28}
, {19, -45, 13}
, {60, 16, 15}
, {7, 38, 31}
, {17, 40, 41}
, {-14, -4, -26}
, {-41, -17, 30}
, {-30, 18, 63}
, {22, -37, 45}
, {54, -4, -22}
, {-5, 4, 3}
, {-20, 0, -24}
, {23, 15, 49}
, {-20, 64, 38}
, {42, 26, -45}
, {-22, -27, -2}
, {7, 3, 37}
, {25, 29, -5}
, {-22, -9, -12}
, {-53, -52, 0}
, {6, 14, 9}
, {19, 33, -34}
, {52, -9, 40}
, {-23, -37, -47}
, {54, -34, -4}
, {-37, 59, 36}
, {11, 7, 43}
, {-16, 30, 2}
, {-8, -38, -2}
, {-51, 8, 28}
, {37, 59, 4}
, {16, -12, -34}
, {-26, 31, 54}
, {47, -20, 55}
, {41, 28, 39}
, {20, 16, -30}
, {35, -19, 48}
, {-19, -6, -7}
, {48, 5, 39}
, {-16, -28, -15}
, {49, 29, 21}
, {-15, 31, 8}
, {-54, 20, -45}
, {13, 0, -10}
, {55, -23, 11}
, {15, 36, 22}
, {-2, -42, -46}
, {-12, 17, 50}
, {5, -43, -46}
, {49, 35, 28}
, {-2, -14, 43}
, {44, 34, 57}
, {-39, 19, -28}
, {32, -26, -42}
, {36, -52, -48}
, {28, 19, 42}
}
, {{-3, -2, -13}
, {-2, 49, 21}
, {42, 6, 4}
, {31, 15, -10}
, {42, -31, -30}
, {38, -11, 3}
, {-44, 32, -11}
, {20, -31, 12}
, {22, 39, 14}
, {32, -26, -20}
, {51, 45, 69}
, {4, -24, 0}
, {-5, -50, -44}
, {-22, -15, -65}
, {38, -24, -43}
, {-35, -8, 8}
, {39, -13, -10}
, {40, 18, -10}
, {-10, 43, 4}
, {-22, -41, 52}
, {64, -14, -12}
, {32, -43, 27}
, {6, -44, -10}
, {3, 43, 32}
, {-9, 40, -9}
, {10, 30, -48}
, {-11, -29, 48}
, {-6, 11, -42}
, {-53, 7, 32}
, {-14, 14, -10}
, {-37, -36, 65}
, {9, -48, 40}
, {0, -15, -31}
, {-25, 19, -37}
, {-15, 0, 43}
, {9, -28, 20}
, {7, -47, 47}
, {21, 12, 9}
, {53, 55, 2}
, {63, 26, -28}
, {5, 44, -38}
, {-35, -41, -38}
, {48, -28, -18}
, {-45, 13, -16}
, {45, -18, -48}
, {22, 16, 29}
, {43, 41, 15}
, {-15, 44, -8}
, {37, -51, -55}
, {34, 15, -41}
, {27, -44, -24}
, {-17, -44, 41}
, {-13, 17, -45}
, {-13, 21, -8}
, {23, -23, -18}
, {31, -8, -18}
, {-16, 69, 62}
, {45, -44, 4}
, {-36, -13, 19}
, {56, 7, -23}
, {-24, -10, -37}
, {-14, 46, 2}
, {-42, 22, 9}
, {-6, 56, 60}
}
, {{-23, 0, 37}
, {-43, -32, 43}
, {-47, 25, 5}
, {38, 6, -4}
, {16, 30, -7}
, {33, -16, -29}
, {-53, 8, -31}
, {-26, 36, 50}
, {4, -41, -47}
, {14, 29, -59}
, {-43, 26, 8}
, {-10, 12, -8}
, {-22, -42, 23}
, {-13, 44, 41}
, {-39, 40, -14}
, {-24, 23, -2}
, {-7, -24, -29}
, {-17, 23, 4}
, {-20, 3, 54}
, {-49, 7, -53}
, {-35, 13, 28}
, {18, -21, -42}
, {39, -14, -1}
, {-13, 35, -11}
, {-13, 27, 3}
, {-42, -12, 40}
, {-40, -33, -34}
, {-28, -31, 49}
, {-37, -22, 27}
, {-5, -19, -39}
, {-56, 42, 3}
, {1, -2, -6}
, {-7, 6, -6}
, {23, 36, -23}
, {-3, -31, -14}
, {-53, 20, -3}
, {-54, 24, 9}
, {6, -7, 13}
, {-13, 32, 22}
, {37, 2, 60}
, {21, 25, 9}
, {14, -26, -33}
, {-24, 7, 30}
, {-37, -4, -50}
, {-11, 40, 39}
, {23, 49, 51}
, {38, 56, 5}
, {26, -44, 9}
, {43, 45, 15}
, {-21, 15, -2}
, {19, 27, -1}
, {23, 49, -40}
, {-42, 34, 11}
, {4, 34, -62}
, {-49, 7, -52}
, {16, -26, 44}
, {33, 51, 21}
, {-39, 33, -57}
, {-47, -49, 4}
, {-37, -52, -11}
, {-8, 14, 8}
, {-12, 2, 33}
, {26, 19, -33}
, {-49, 11, -7}
}
, {{-23, 33, -31}
, {30, 7, 35}
, {-52, -30, 4}
, {3, 27, -22}
, {23, 5, -46}
, {11, 2, 28}
, {-20, -5, 10}
, {0, -15, -42}
, {50, 41, -29}
, {16, 6, 3}
, {-31, 18, -4}
, {49, -13, -23}
, {27, -39, -29}
, {-43, -24, 39}
, {-24, 18, 28}
, {-29, -37, -30}
, {26, 4, 10}
, {34, -24, -49}
, {24, 4, 32}
, {-30, 11, 45}
, {31, 51, 25}
, {-25, -38, -46}
, {-21, -39, -31}
, {-60, 18, -7}
, {47, 0, 36}
, {51, 13, -4}
, {-37, 5, 29}
, {23, 48, -6}
, {-7, -42, -46}
, {-8, -18, -8}
, {-45, 2, -42}
, {-34, 51, 46}
, {-16, -48, 8}
, {13, 7, 46}
, {3, 1, 31}
, {52, -24, 10}
, {20, 31, 28}
, {9, 22, 4}
, {0, 8, -14}
, {-11, -38, 38}
, {25, -25, -24}
, {7, 39, -36}
, {19, 24, -34}
, {-26, -30, -8}
, {28, -29, 39}
, {50, 24, 2}
, {-22, -27, -22}
, {-8, 45, 50}
, {54, 25, 22}
, {-47, 22, -21}
, {46, 7, 39}
, {50, -51, -55}
, {-15, -12, -46}
, {18, -15, 1}
, {-29, 41, -49}
, {-19, -39, 12}
, {-26, -32, 51}
, {27, -31, 26}
, {35, -43, 1}
, {-1, -45, -19}
, {36, 38, 19}
, {-47, -47, -16}
, {-35, 36, -24}
, {1, -24, -27}
}
, {{-22, -59, 12}
, {-19, -13, 53}
, {37, -28, -44}
, {2, -10, -13}
, {-16, 26, -10}
, {49, 30, 30}
, {-29, -3, -40}
, {-11, 48, -34}
, {-7, 57, 32}
, {-35, -28, -15}
, {-17, 2, -43}
, {48, 21, -18}
, {-48, -32, -27}
, {-58, -36, 1}
, {-46, -26, -22}
, {11, -21, -9}
, {20, 10, 27}
, {54, -35, 4}
, {13, 9, 41}
, {-2, 18, 46}
, {10, -20, 56}
, {-29, -50, -29}
, {15, -67, -29}
, {20, -10, 11}
, {32, 15, 33}
, {-21, -3, 37}
, {-26, 18, 33}
, {36, -40, -30}
, {3, -11, -18}
, {-30, 44, -16}
, {16, -27, -54}
, {-36, -32, -5}
, {-28, -7, -30}
, {4, 32, 47}
, {67, -31, 51}
, {10, 31, -18}
, {-6, 28, 19}
, {43, -10, -15}
, {16, -14, -11}
, {64, -23, 46}
, {-46, 0, 22}
, {-24, 59, 10}
, {-40, 57, 56}
, {51, 51, 22}
, {-6, 33, 39}
, {65, -27, 59}
, {5, 57, 0}
, {-21, 4, -34}
, {20, 7, 0}
, {36, -6, -6}
, {-11, 11, 27}
, {-9, -45, 52}
, {-15, 58, 5}
, {-52, 35, 0}
, {-17, -15, 26}
, {26, 39, -18}
, {33, 24, -19}
, {-40, -60, 33}
, {33, 19, -12}
, {-3, 13, -30}
, {15, -42, -27}
, {18, 4, -23}
, {28, -2, -20}
, {18, 40, -11}
}
, {{5, 39, -28}
, {23, -39, 25}
, {-60, -37, 29}
, {16, -17, 48}
, {-41, -1, 31}
, {-15, -22, 30}
, {-14, 51, 8}
, {32, 15, -46}
, {-14, -23, 10}
, {-13, -31, -57}
, {33, -17, -44}
, {-22, -63, -11}
, {-27, -5, -17}
, {69, 57, 14}
, {32, 21, 50}
, {30, -5, -49}
, {49, -5, -52}
, {40, -15, -12}
, {-37, -12, -37}
, {35, 32, 27}
, {15, 41, -32}
, {-26, 1, 3}
, {-6, -28, -25}
, {-48, -34, 19}
, {-41, 19, 44}
, {45, -27, 46}
, {53, 31, 46}
, {-38, 15, 39}
, {-14, -1, 49}
, {39, 42, 9}
, {11, -55, -33}
, {-16, 12, -7}
, {54, 42, 21}
, {-42, -24, 0}
, {-13, 9, 8}
, {46, -35, 9}
, {-37, -9, -18}
, {-7, -6, 52}
, {27, 6, -25}
, {38, 36, 15}
, {-49, 34, -37}
, {10, 31, -13}
, {19, 15, -5}
, {13, 50, -14}
, {48, -2, 2}
, {31, 54, -36}
, {-36, 1, -36}
, {-42, -10, 2}
, {-11, 49, 52}
, {25, 0, -34}
, {-6, 22, -4}
, {-47, 20, -13}
, {15, -26, -8}
, {-27, 10, 61}
, {-34, 20, -3}
, {-42, 27, -36}
, {29, -20, -60}
, {-36, -55, 26}
, {50, -1, -31}
, {16, -10, -50}
, {-37, 38, -16}
, {-46, -61, -24}
, {-5, -19, 40}
, {49, -1, -20}
}
, {{49, -37, 61}
, {7, 36, 18}
, {-4, -9, 3}
, {-40, 43, 25}
, {7, 46, 29}
, {-24, -22, 45}
, {-26, 41, -19}
, {38, -40, 13}
, {-47, -19, 23}
, {15, -24, 44}
, {48, -12, 17}
, {-39, 57, -30}
, {-50, -4, 15}
, {-62, -13, -24}
, {-2, 54, 18}
, {2, 25, 10}
, {-27, -25, 52}
, {-21, 41, -30}
, {-9, 11, -43}
, {-37, 32, 40}
, {0, -11, -9}
, {51, 54, -23}
, {-41, -2, 20}
, {16, 71, -20}
, {1, 38, -11}
, {-51, -40, -20}
, {14, -18, 14}
, {27, 21, -22}
, {-42, 49, 44}
, {22, 20, 4}
, {-7, -48, -1}
, {33, -9, -46}
, {-27, -41, 19}
, {-36, -1, -16}
, {-44, -38, 29}
, {3, 38, -35}
, {-10, 33, 10}
, {37, -50, -22}
, {0, -45, -16}
, {-20, 13, 14}
, {5, 45, 43}
, {-9, 52, -41}
, {-26, 48, -31}
, {-31, 6, 20}
, {-49, 23, 36}
, {48, -36, 10}
, {5, -2, 51}
, {-21, 46, -15}
, {-11, -5, 18}
, {-24, 32, -18}
, {33, -39, 9}
, {20, 0, -18}
, {19, 4, -50}
, {-35, -45, -53}
, {-5, -38, -35}
, {16, 39, -13}
, {20, 1, 11}
, {39, -10, 23}
, {-17, 27, 42}
, {14, 55, 3}
, {49, -8, -40}
, {32, -39, -19}
, {-23, 23, 17}
, {-31, -2, -38}
}
, {{20, 43, 39}
, {-47, 40, 46}
, {66, -5, -32}
, {-10, -33, 5}
, {12, -55, 35}
, {19, 33, -35}
, {24, -6, 19}
, {-14, 3, -13}
, {21, -7, -9}
, {60, -3, 57}
, {-30, 66, -27}
, {-3, -52, 41}
, {12, -31, -7}
, {-25, -24, -30}
, {27, -16, -24}
, {-21, -33, 11}
, {16, -49, 6}
, {0, -52, -52}
, {18, -46, 2}
, {40, 35, 34}
, {-50, -50, -16}
, {-15, 31, -4}
, {8, -17, 24}
, {41, -20, -13}
, {57, 51, 5}
, {44, -10, 10}
, {2, 4, 22}
, {-46, 5, -22}
, {-11, 42, 20}
, {-39, -23, -23}
, {49, 44, 38}
, {29, -34, 27}
, {-44, -35, -49}
, {-26, -11, -15}
, {17, -23, 42}
, {41, -23, 29}
, {26, -31, 23}
, {10, -21, -15}
, {-9, 40, -44}
, {41, 35, 22}
, {25, -25, 34}
, {6, 28, 48}
, {19, 35, 33}
, {-21, -49, -3}
, {15, -46, 32}
, {-22, -41, 0}
, {-34, 50, 33}
, {-6, -32, -33}
, {2, -8, -1}
, {45, 48, 9}
, {-46, -5, 34}
, {7, -17, -22}
, {-35, 33, 0}
, {-22, 13, 12}
, {50, -13, -1}
, {-36, -41, -33}
, {-44, -43, -38}
, {27, -45, 51}
, {-13, 41, -40}
, {27, 20, -25}
, {-40, -2, 33}
, {-17, -27, -26}
, {38, 10, 45}
, {-49, -23, -41}
}
, {{30, 66, 31}
, {14, -17, -45}
, {-23, -32, 7}
, {38, -11, -15}
, {45, 37, 61}
, {-6, 59, -16}
, {-19, 1, 18}
, {32, -44, -17}
, {2, -14, -31}
, {13, 30, -17}
, {46, 12, 59}
, {4, 1, 11}
, {-39, 13, -30}
, {-21, -44, 3}
, {36, -35, 35}
, {-22, 24, 29}
, {50, 32, -44}
, {42, 5, 7}
, {-28, -25, 54}
, {16, -17, 47}
, {-45, 30, 38}
, {-15, 39, 1}
, {-21, -24, 12}
, {17, 47, 27}
, {-22, -7, -8}
, {-24, 29, -49}
, {-27, -32, 35}
, {-50, -46, 7}
, {0, -23, 42}
, {23, 25, -4}
, {46, 0, -23}
, {41, 57, -40}
, {31, -7, 26}
, {-5, 33, 37}
, {-4, 38, -35}
, {52, -34, -17}
, {-9, -30, -40}
, {-43, 39, -38}
, {-18, 4, -1}
, {9, 9, 6}
, {11, 23, -13}
, {37, 33, 40}
, {-9, -40, -27}
, {45, 9, 36}
, {0, -37, -54}
, {-8, -26, -53}
, {2, -11, 16}
, {57, -22, 6}
, {-38, 6, 44}
, {-33, -17, 23}
, {17, -8, 13}
, {-21, -13, 33}
, {-17, -54, -56}
, {-62, 3, 45}
, {25, 42, -14}
, {27, 2, 53}
, {-44, 28, 1}
, {-56, 9, -8}
, {15, 3, -18}
, {21, -19, 47}
, {43, 16, -6}
, {42, 19, -53}
, {15, 34, 61}
, {33, -9, 12}
}
, {{57, 15, -24}
, {12, -15, 13}
, {3, 8, -33}
, {26, -59, -3}
, {63, -16, -1}
, {21, 1, -4}
, {38, -41, 41}
, {65, -42, 23}
, {-9, -2, -6}
, {-19, 39, 22}
, {-1, 35, -7}
, {56, -29, -8}
, {-26, 32, -22}
, {35, -29, -40}
, {19, -24, 19}
, {26, -12, 52}
, {-5, -19, 3}
, {-33, 39, 3}
, {-4, 42, 1}
, {43, -20, 29}
, {-21, 27, -32}
, {-3, 5, 34}
, {-14, -2, -42}
, {67, 21, -4}
, {-9, 37, -41}
, {22, 41, -44}
, {-44, -7, -23}
, {32, -40, 15}
, {-21, -41, -26}
, {55, -48, -9}
, {63, 0, -14}
, {-47, -19, -5}
, {-42, 8, 12}
, {18, -54, 6}
, {-27, 33, -28}
, {-2, -22, 8}
, {-31, -10, -50}
, {47, 26, -51}
, {7, -29, -25}
, {29, -51, 48}
, {-7, -1, -42}
, {48, -58, -6}
, {52, 38, -8}
, {23, -40, -13}
, {-23, -9, -31}
, {-12, -27, -18}
, {-32, 6, 8}
, {29, -26, 26}
, {-18, 5, 34}
, {-14, 16, 10}
, {-34, -11, -2}
, {-19, -17, 49}
, {22, -45, -37}
, {-24, -29, 8}
, {9, 25, -5}
, {59, -30, 14}
, {-18, -38, -2}
, {-35, 7, 28}
, {35, 5, 30}
, {32, 32, -8}
, {-3, -23, -23}
, {-32, -45, 38}
, {31, 4, 58}
, {38, 13, -43}
}
, {{-19, 8, -19}
, {3, 22, -17}
, {16, -17, -48}
, {30, 36, -62}
, {-40, 23, -2}
, {-3, -25, -10}
, {-15, -12, -14}
, {12, -32, -17}
, {0, -50, 3}
, {-42, 34, 29}
, {-1, -34, -19}
, {-9, -2, -7}
, {36, -53, -21}
, {-20, -40, 8}
, {-33, -15, 44}
, {41, 3, 33}
, {-10, -34, -6}
, {-56, 13, -13}
, {-6, -25, 23}
, {-53, 37, 23}
, {-48, -10, -11}
, {-29, -21, -53}
, {-19, 3, 37}
, {-40, -12, -30}
, {-42, -22, 19}
, {-38, -51, -39}
, {-27, -16, -48}
, {-51, -51, 35}
, {20, 38, 21}
, {-19, 4, -46}
, {13, -42, -3}
, {31, -10, -8}
, {-25, -23, 1}
, {11, 25, -11}
, {23, -35, 1}
, {-24, 19, 21}
, {-40, -5, -47}
, {-17, -20, 18}
, {-12, -25, 32}
, {-56, -2, -49}
, {7, 25, -25}
, {-30, 20, 5}
, {-6, 18, -14}
, {12, -37, -45}
, {-22, 35, 33}
, {-6, -40, -18}
, {-41, 40, 0}
, {-15, 39, 37}
, {-42, -21, -32}
, {-50, -10, -58}
, {13, -15, 26}
, {-14, 30, -49}
, {3, -4, 37}
, {27, -9, 27}
, {43, 40, 7}
, {-46, -50, 35}
, {-2, -32, 6}
, {-16, -44, -7}
, {41, 48, -48}
, {40, 37, -14}
, {-52, -20, 24}
, {16, -9, 19}
, {-18, 21, 51}
, {-22, -10, -41}
}
, {{20, 47, 10}
, {-36, 37, -9}
, {33, 17, 0}
, {35, 36, -3}
, {-24, 63, -10}
, {35, 49, -25}
, {21, 19, 23}
, {-42, -4, -10}
, {-32, -17, -47}
, {-14, -23, 15}
, {0, -7, -35}
, {24, 37, -22}
, {-5, -45, 40}
, {-30, 9, 7}
, {3, 26, -32}
, {-7, -51, -27}
, {1, -6, -47}
, {-33, -3, 29}
, {9, -46, -51}
, {-10, -24, -11}
, {-50, -40, -22}
, {61, 56, -33}
, {-22, -5, 25}
, {-9, 55, 56}
, {6, 49, 37}
, {32, 10, 12}
, {-34, -45, -50}
, {22, -47, -10}
, {0, 16, 26}
, {17, 49, -39}
, {31, 58, 5}
, {34, 15, -29}
, {-10, -28, 33}
, {54, -3, 30}
, {5, -22, -40}
, {45, -32, 7}
, {52, -38, 40}
, {-14, -34, 30}
, {40, 36, 31}
, {-13, -6, 32}
, {-38, 32, -6}
, {-20, 37, 13}
, {10, -5, -42}
, {31, 3, -43}
, {-1, 48, -31}
, {-7, -1, -17}
, {16, 35, 34}
, {39, -4, 14}
, {43, -48, 31}
, {-47, 44, 23}
, {16, 43, -36}
, {34, -40, 52}
, {24, -20, -31}
, {-42, -46, -22}
, {-13, 44, -21}
, {36, 38, 27}
, {3, 55, 15}
, {0, -49, 37}
, {22, -25, -44}
, {68, 42, -4}
, {45, 30, 3}
, {-13, 16, 25}
, {3, -28, -36}
, {-7, -50, -6}
}
, {{-52, 7, 19}
, {37, -27, -39}
, {16, -24, -61}
, {23, -41, 29}
, {-33, -15, 16}
, {38, 45, 50}
, {-47, -12, -29}
, {-18, 43, 34}
, {38, -29, 42}
, {-28, -57, 25}
, {-38, -34, -33}
, {-32, -46, 38}
, {11, 7, 38}
, {-24, 58, 53}
, {32, -48, -31}
, {36, 45, -20}
, {-21, 30, 39}
, {-15, -4, 43}
, {27, -18, -1}
, {-48, 36, 45}
, {13, 16, 35}
, {25, -36, 26}
, {-24, 5, -24}
, {8, -32, 33}
, {19, -48, 4}
, {45, -6, 22}
, {-10, 49, 54}
, {10, -1, 40}
, {-9, -45, -20}
, {31, 54, -20}
, {-46, 28, 1}
, {33, 8, -33}
, {-40, 41, -17}
, {-24, 40, 13}
, {17, -37, -44}
, {-39, 16, 16}
, {-8, 5, -49}
, {47, 21, 7}
, {45, -11, 39}
, {-41, -50, -21}
, {30, -35, -18}
, {-30, 1, 31}
, {1, -27, -21}
, {43, -13, 9}
, {-2, -9, -12}
, {-14, 30, -36}
, {-44, 28, -17}
, {27, -30, 51}
, {23, 8, -31}
, {50, 40, 29}
, {27, -39, -15}
, {-48, -23, -51}
, {46, 38, 24}
, {-21, 52, 39}
, {-38, 43, 38}
, {-12, -11, -45}
, {17, 16, -23}
, {-9, -43, 39}
, {-51, 39, 0}
, {21, 26, 1}
, {-48, -16, 30}
, {-40, 0, 17}
, {9, 8, -47}
, {7, 23, -34}
}
, {{-39, -15, 37}
, {6, -54, 34}
, {8, -58, 26}
, {-4, -62, 30}
, {-42, 22, 13}
, {-27, 14, 39}
, {13, 16, 0}
, {13, 18, -23}
, {-45, 9, -52}
, {-35, -6, 34}
, {-10, -14, 11}
, {29, 11, 12}
, {-23, 20, 14}
, {26, 0, 55}
, {-35, 53, 29}
, {28, -41, -30}
, {18, -49, -22}
, {-5, 18, -32}
, {29, -36, -32}
, {27, -34, -1}
, {7, 31, -17}
, {32, -1, 43}
, {19, 48, 23}
, {-46, -16, -29}
, {27, -39, 2}
, {-26, -49, -27}
, {24, 20, -27}
, {57, 49, 51}
, {-50, 11, 26}
, {37, -37, 45}
, {48, 16, 6}
, {0, 7, 51}
, {4, 38, -32}
, {11, -11, -37}
, {30, 14, 30}
, {-19, 35, 48}
, {35, 8, -44}
, {47, -15, 14}
, {11, -10, 33}
, {-30, -10, 31}
, {24, 13, -46}
, {13, 24, 11}
, {-12, 24, 33}
, {45, 40, -6}
, {-44, 33, 47}
, {-54, -41, -45}
, {-56, 29, 6}
, {52, 8, 2}
, {13, -44, 27}
, {-24, -46, 49}
, {-37, 54, 7}
, {2, -4, 10}
, {-42, 19, -3}
, {-23, 33, 34}
, {-28, 24, 3}
, {-22, -46, 24}
, {-4, -27, -2}
, {34, 0, 27}
, {40, 49, 31}
, {46, -49, -47}
, {38, 19, -36}
, {35, 24, -8}
, {-5, 6, -26}
, {-16, -55, -18}
}
, {{33, 33, 10}
, {-31, 19, -27}
, {-21, -38, -7}
, {-2, 22, -7}
, {-31, 24, 43}
, {20, -11, 35}
, {32, 5, 54}
, {12, 1, -18}
, {51, -43, -13}
, {12, 36, 7}
, {34, 54, 38}
, {0, 0, 56}
, {-14, -37, -44}
, {-29, -15, -43}
, {-28, -35, 16}
, {57, 43, -1}
, {46, -22, -48}
, {-38, 32, -47}
, {9, 3, 0}
, {48, 33, 58}
, {15, 26, 34}
, {29, 22, -20}
, {-45, 13, 10}
, {42, -31, 51}
, {-51, 9, -10}
, {11, 15, 15}
, {-15, -3, 46}
, {-32, 27, 8}
, {-12, 31, -26}
, {-6, -45, -18}
, {-38, -44, -37}
, {-16, -5, -35}
, {5, -45, 24}
, {-29, 28, -20}
, {22, 2, -5}
, {-30, -36, -3}
, {-30, 22, 41}
, {-34, -20, 12}
, {2, 1, 7}
, {-12, 24, 62}
, {10, -9, 47}
, {-1, -9, 5}
, {-22, -40, -42}
, {-56, 54, 37}
, {-2, -18, 19}
, {19, 13, 11}
, {5, 5, 28}
, {38, 12, 2}
, {-57, 0, 31}
, {-27, 27, 10}
, {42, 37, 34}
, {55, -7, 32}
, {-42, -27, 0}
, {4, -27, -15}
, {-39, -19, 14}
, {70, 53, -8}
, {50, -29, 60}
, {-13, -5, 34}
, {-26, -22, 50}
, {-7, -32, 43}
, {35, 21, 29}
, {-45, -30, 1}
, {-13, 24, -29}
, {0, -18, 18}
}
, {{48, 18, 5}
, {3, -30, 12}
, {3, 24, -22}
, {28, -50, 39}
, {50, -9, 25}
, {23, 0, -47}
, {-21, -21, -2}
, {-31, -21, 63}
, {-11, 3, 25}
, {-16, 54, -19}
, {-58, 38, 61}
, {-7, 19, 43}
, {-17, 47, -27}
, {7, -10, -51}
, {-26, 19, 32}
, {42, -25, 35}
, {-51, -36, -41}
, {-35, 27, 19}
, {9, -20, 45}
, {-47, -4, -2}
, {23, 24, 44}
, {51, -31, 13}
, {-13, 5, -10}
, {0, -61, 37}
, {-8, 14, -39}
, {-44, -4, 1}
, {15, -8, 12}
, {6, -16, 37}
, {-1, 27, 9}
, {-28, -43, 32}
, {17, 35, 29}
, {-22, -19, 46}
, {-37, -39, 31}
, {-58, -15, -60}
, {-28, 20, 36}
, {-12, -51, -1}
, {-56, -46, 41}
, {-45, 41, 12}
, {16, -21, -33}
, {0, 18, 23}
, {26, 5, -18}
, {-8, -4, 27}
, {-24, 26, 18}
, {64, 3, -42}
, {15, 3, -49}
, {-38, 13, 0}
, {-58, 47, -7}
, {35, 0, 33}
, {38, -45, -20}
, {31, 0, 2}
, {-32, -22, 49}
, {-18, 38, 16}
, {-10, 20, -64}
, {-6, 52, 17}
, {-41, 22, 44}
, {-30, 25, -12}
, {47, 22, 28}
, {27, -38, 47}
, {-16, -28, 30}
, {-21, 56, 30}
, {-43, 18, 39}
, {55, 45, 27}
, {51, 28, 0}
, {32, 49, -22}
}
, {{23, 25, -18}
, {-6, -31, -21}
, {23, -4, -1}
, {-34, -36, -19}
, {-50, 0, 34}
, {39, -37, 13}
, {2, -17, 47}
, {14, -2, 16}
, {-14, -47, -8}
, {-48, -42, -28}
, {46, 17, 45}
, {9, -26, 26}
, {-37, 31, 12}
, {-29, -18, -6}
, {-40, -45, -40}
, {3, -46, -54}
, {-8, -30, 21}
, {-44, -4, 25}
, {-31, 25, -44}
, {43, -54, 0}
, {-37, 12, -19}
, {-39, 46, -32}
, {9, -4, 19}
, {-44, 3, 15}
, {-18, -48, -49}
, {-35, -2, 10}
, {25, 42, -40}
, {46, -27, 6}
, {37, 42, 5}
, {32, 23, 38}
, {59, -5, 17}
, {10, -19, 22}
, {-55, 4, -8}
, {-47, 29, -26}
, {-4, -6, 6}
, {-16, -22, -1}
, {-29, 15, 11}
, {-40, -41, 13}
, {-44, 13, 16}
, {-7, -43, 50}
, {-47, -4, -15}
, {-38, 39, 42}
, {27, 17, -15}
, {27, -31, 12}
, {13, 34, 41}
, {-48, -33, 5}
, {43, 50, -35}
, {-9, -31, 43}
, {-37, -15, -21}
, {-38, -38, -51}
, {-48, 28, -35}
, {10, 20, -40}
, {0, 28, 45}
, {25, 53, 32}
, {34, -1, -35}
, {-23, -31, 6}
, {9, -1, 30}
, {44, 39, 5}
, {-51, -51, 4}
, {35, 52, 41}
, {20, 32, 44}
, {16, -2, 51}
, {11, -7, -34}
, {-28, -25, 70}
}
, {{-6, 21, 15}
, {6, 13, -50}
, {-1, 24, -23}
, {28, 26, 49}
, {-17, -11, 21}
, {40, -26, 29}
, {-14, 11, 34}
, {43, -34, 44}
, {17, -48, -34}
, {-51, -10, 15}
, {-52, 5, 63}
, {-27, -19, -5}
, {-23, -26, -15}
, {37, 46, 10}
, {18, 21, -27}
, {55, -36, -6}
, {22, 19, -51}
, {-5, 32, 44}
, {44, -12, -24}
, {-8, 24, -43}
, {19, -1, -6}
, {-29, -48, -41}
, {38, 57, -43}
, {-51, -47, -53}
, {22, 34, 6}
, {-44, -24, -51}
, {40, -32, -31}
, {-10, -11, -21}
, {-4, 45, 14}
, {44, -43, -21}
, {27, 37, 27}
, {-46, 29, 7}
, {32, -8, -47}
, {-37, -41, 28}
, {-30, -30, 28}
, {-40, 39, -24}
, {45, 35, -43}
, {31, 50, 35}
, {-2, -34, -32}
, {-26, 18, 4}
, {37, -9, 26}
, {-39, 54, 28}
, {-31, 46, 54}
, {-47, -15, 23}
, {-42, 7, -1}
, {7, 9, 22}
, {-7, 4, 5}
, {2, -7, -12}
, {-34, 2, -22}
, {0, -35, -28}
, {-21, -41, 32}
, {-30, -16, 8}
, {-6, 20, 39}
, {40, -36, 53}
, {-9, -21, -34}
, {6, -1, -7}
, {2, 41, 20}
, {37, 14, -49}
, {-3, -30, 16}
, {26, -34, -32}
, {-30, 2, -8}
, {-52, 19, -2}
, {-27, -49, -13}
, {39, 45, 39}
}
, {{2, 0, -16}
, {-51, 23, -54}
, {37, 20, 8}
, {-15, 6, 17}
, {-40, 31, 24}
, {33, -46, 34}
, {24, -4, -54}
, {-40, 40, -31}
, {-46, -24, -10}
, {-7, 14, 22}
, {-42, 5, 33}
, {-27, -46, 30}
, {-27, -55, -10}
, {-47, 10, 46}
, {31, 15, 29}
, {32, -31, 13}
, {47, -44, -50}
, {25, -13, 10}
, {-42, 15, 39}
, {-45, -35, -9}
, {-36, -13, 7}
, {-39, -30, 36}
, {-31, -6, -21}
, {12, 45, -5}
, {-20, 45, 18}
, {-22, 9, 20}
, {-10, -8, -42}
, {-16, -30, 27}
, {-38, -50, 0}
, {-47, -48, 45}
, {28, 6, 35}
, {15, -28, 20}
, {-19, 16, -3}
, {1, 10, 6}
, {-42, -30, -12}
, {34, 5, 10}
, {-12, -23, -53}
, {-22, 44, 5}
, {10, -52, -37}
, {49, 32, 36}
, {39, -6, 47}
, {-23, 36, -31}
, {-30, 10, 2}
, {11, -12, -30}
, {-12, -37, 28}
, {51, -12, 8}
, {-46, -48, -13}
, {-3, 19, 40}
, {45, 4, 42}
, {-12, -56, -3}
, {-34, -43, -26}
, {30, -24, -18}
, {22, 44, -46}
, {24, -1, -26}
, {45, 13, -22}
, {-4, -2, -3}
, {-10, -6, 46}
, {40, -36, 39}
, {28, -8, -25}
, {46, 30, 3}
, {-13, 20, -46}
, {5, 30, -27}
, {-49, -51, 15}
, {-17, -17, 16}
}
, {{-4, 22, 4}
, {-46, -35, 41}
, {30, 18, 12}
, {28, -30, -13}
, {-52, 20, -32}
, {-20, -51, -43}
, {-38, -42, -20}
, {1, 47, -50}
, {-6, 20, 16}
, {-44, 17, 30}
, {0, 46, -27}
, {20, -9, 15}
, {-1, -29, 21}
, {-17, 15, -21}
, {-20, -39, -45}
, {22, 2, -25}
, {38, -25, 17}
, {-40, -7, -39}
, {-1, -24, -9}
, {-33, -8, -44}
, {-35, -3, -3}
, {-7, -35, 12}
, {15, -25, -46}
, {-41, -50, -38}
, {-43, -30, -18}
, {-8, 0, 30}
, {33, -4, -6}
, {-10, 9, 50}
, {5, 24, -33}
, {32, -34, -19}
, {37, 25, 0}
, {-4, 16, 21}
, {10, 10, 39}
, {34, 49, 43}
, {30, 10, 0}
, {-49, 2, 13}
, {-46, 43, 9}
, {9, 10, -44}
, {29, 15, -42}
, {-15, -42, -28}
, {-33, 9, 32}
, {-13, 0, -47}
, {16, 23, -25}
, {46, -8, 47}
, {34, -26, 31}
, {-27, 17, 45}
, {25, 32, -25}
, {22, 23, 50}
, {36, -44, -8}
, {21, 11, 6}
, {-46, -39, -5}
, {12, 42, -49}
, {-2, 51, -2}
, {0, 35, -46}
, {-49, 4, -43}
, {22, -31, -2}
, {23, -35, -45}
, {38, -50, -14}
, {4, 31, 17}
, {48, -1, -31}
, {-14, 40, -20}
, {6, 34, 1}
, {-39, -34, -27}
, {-12, 11, 11}
}
, {{-23, 10, -11}
, {1, 13, 9}
, {55, -15, 32}
, {15, 5, -30}
, {-18, 25, 15}
, {19, 5, 25}
, {42, 41, -34}
, {-56, 18, -18}
, {-39, 39, 34}
, {14, -25, 59}
, {21, 5, -7}
, {-13, 45, -28}
, {6, -4, -27}
, {-31, 13, 34}
, {-20, 36, 44}
, {3, 51, -36}
, {49, 15, -46}
, {36, -42, -43}
, {-56, 25, 27}
, {-31, -7, 17}
, {-20, 11, 46}
, {42, -26, 62}
, {-49, -44, -26}
, {-8, 9, 54}
, {-42, -11, 26}
, {49, 2, 30}
, {-3, 18, 43}
, {-24, 13, 4}
, {-21, 16, -15}
, {11, 11, 31}
, {19, 56, 35}
, {-4, -39, 33}
, {-34, 31, -32}
, {1, -20, 21}
, {56, -27, -26}
, {-57, 21, -35}
, {-26, -2, -8}
, {-57, 42, 45}
, {22, 42, -17}
, {-20, 28, 27}
, {6, 40, -27}
, {17, 40, 9}
, {11, -40, -42}
, {-16, 10, -25}
, {-13, 7, 43}
, {41, -3, 6}
, {-24, 16, 34}
, {-16, -27, -56}
, {-55, 46, 45}
, {35, 0, 20}
, {17, -40, -5}
, {-16, -17, -4}
, {-9, -46, -40}
, {-1, 2, -31}
, {-37, -17, -42}
, {-12, 21, -13}
, {12, -27, -31}
, {-19, 43, -20}
, {-24, 24, 43}
, {27, 50, -7}
, {-14, -27, -17}
, {-5, -27, 27}
, {-10, 44, 9}
, {1, -46, 20}
}
, {{33, 22, -35}
, {3, -7, -14}
, {17, -3, 21}
, {-26, 65, 57}
, {64, 13, 8}
, {37, 33, -50}
, {-31, -26, 42}
, {-22, 29, 40}
, {-50, -40, -6}
, {-23, -29, -4}
, {48, 20, 33}
, {-51, 26, 9}
, {48, -21, 44}
, {12, -6, -30}
, {22, 25, 40}
, {29, 29, 19}
, {-43, 26, -2}
, {-29, 46, 34}
, {-47, 2, -41}
, {28, -35, 21}
, {26, -16, 42}
, {43, 29, -10}
, {39, -47, -29}
, {23, 0, -43}
, {-54, -12, -11}
, {-49, 34, -19}
, {-9, 20, 4}
, {24, -30, 37}
, {0, 11, 40}
, {-34, 17, 26}
, {-56, -34, -4}
, {4, 45, 8}
, {-6, 47, -24}
, {-6, 30, 48}
, {-30, 8, 53}
, {-10, 45, -48}
, {4, 31, -40}
, {-46, 32, 3}
, {-14, -13, -39}
, {15, 21, 3}
, {11, 24, 30}
, {3, -22, 25}
, {30, 8, 9}
, {45, 8, 44}
, {-8, -7, -39}
, {11, -15, 42}
, {8, 0, 6}
, {-20, 25, -6}
, {-38, 48, -42}
, {-12, 34, -41}
, {-35, 17, -47}
, {28, 54, -44}
, {40, -32, 9}
, {22, 29, 17}
, {-9, 44, -1}
, {-31, -15, 55}
, {-22, -11, 39}
, {26, -18, 43}
, {-37, -9, 22}
, {19, -54, -41}
, {-8, -12, 31}
, {-10, 42, 22}
, {4, -54, 41}
, {-23, 34, -18}
}
, {{19, -37, -28}
, {-30, -36, -11}
, {12, 60, -15}
, {29, -5, 46}
, {53, -19, 38}
, {-38, 39, 38}
, {47, 15, 35}
, {8, 9, -25}
, {-47, 22, -40}
, {0, 8, 40}
, {-44, 63, -33}
, {46, -2, 8}
, {-12, -46, -27}
, {-42, -41, -41}
, {-26, 33, -20}
, {-18, 19, 35}
, {-29, -51, 13}
, {38, 4, -54}
, {39, 6, 44}
, {28, -26, 0}
, {51, 68, 15}
, {-23, -36, 15}
, {-4, -24, 2}
, {52, -16, -39}
, {-39, -35, 9}
, {25, 32, 22}
, {-35, -2, -24}
, {-26, 36, -18}
, {27, 36, 27}
, {-15, 20, 48}
, {18, -29, -19}
, {19, 35, -53}
, {-29, -24, 41}
, {49, 21, -35}
, {59, 56, 4}
, {-32, 20, 8}
, {28, -47, -52}
, {-15, 20, -15}
, {-28, -1, 11}
, {23, 31, -22}
, {33, 20, -15}
, {30, 31, -24}
, {40, 10, 30}
, {6, -29, 13}
, {-24, -6, -21}
, {-8, 52, 3}
, {-14, -42, -10}
, {-23, 7, -25}
, {40, -2, -11}
, {41, 44, -46}
, {33, -45, 34}
, {-30, 2, -39}
, {-30, -20, -18}
, {-40, -28, 8}
, {38, -5, -35}
, {-18, 22, -28}
, {29, -12, 33}
, {15, -34, 38}
, {19, 33, 0}
, {26, -21, 45}
, {-50, 4, -9}
, {45, 31, 27}
, {10, -8, -19}
, {-7, 63, -23}
}
, {{33, -24, -7}
, {-31, -11, 40}
, {31, 4, 20}
, {25, 5, 0}
, {0, 33, -11}
, {22, -35, -23}
, {36, 26, -8}
, {-27, 31, -3}
, {23, 0, -11}
, {48, 19, 59}
, {46, -23, 48}
, {67, -11, 44}
, {17, 3, 30}
, {-31, -10, 26}
, {-39, 8, 44}
, {57, 29, 9}
, {-23, 21, 28}
, {-9, -26, -14}
, {-2, 46, 29}
, {-14, -46, 19}
, {58, 64, 4}
, {-1, 1, 43}
, {11, -58, -37}
, {65, 28, 36}
, {24, -40, -39}
, {17, -31, 20}
, {-20, 21, -39}
, {30, 31, -54}
, {26, -18, 48}
, {7, 31, -30}
, {1, 37, 3}
, {-18, -22, 11}
, {-52, 46, 26}
, {44, 23, -21}
, {22, 0, 19}
, {-43, 18, 21}
, {-36, -33, 2}
, {31, -43, -21}
, {-13, 24, -12}
, {34, 24, 2}
, {-22, -31, -36}
, {42, 29, 42}
, {-24, -30, -26}
, {-30, -29, -20}
, {-58, 0, 39}
, {-15, -1, 39}
, {24, 54, 39}
, {11, 54, 2}
, {-17, 19, -1}
, {0, 17, 0}
, {35, -10, 24}
, {37, 29, 57}
, {-7, -59, -46}
, {-39, 2, -48}
, {20, -4, 46}
, {35, 45, 44}
, {28, 57, 0}
, {22, -22, -1}
, {-37, 30, -44}
, {-1, 44, 40}
, {-45, -20, -51}
, {-21, -7, 2}
, {44, -21, -40}
, {8, -12, 23}
}
, {{-59, 33, 38}
, {-34, 11, 34}
, {-10, -48, 34}
, {11, 18, -8}
, {-44, -7, 23}
, {3, 1, -29}
, {1, 22, 45}
, {14, 24, 10}
, {12, 31, 47}
, {18, -5, 5}
, {-52, 39, 6}
, {-41, -38, -52}
, {30, 25, -30}
, {-23, -3, 60}
, {21, -40, 31}
, {38, -14, 33}
, {33, 13, -25}
, {-16, 7, 5}
, {-47, 20, -51}
, {15, 47, 7}
, {6, -19, -3}
, {-49, 18, -22}
, {-49, -15, 42}
, {-50, -68, -49}
, {0, 19, 10}
, {10, 6, -42}
, {33, 3, 7}
, {7, -17, -12}
, {42, -4, -33}
, {-43, 4, 4}
, {18, 2, -28}
, {56, -27, 31}
, {-8, -2, -41}
, {31, -23, 21}
, {-47, -26, -27}
, {2, 10, -33}
, {54, 10, 25}
, {10, -38, -16}
, {-25, 22, -26}
, {10, -39, -22}
, {21, -28, -21}
, {5, 31, 13}
, {23, 38, 21}
, {-29, 31, 55}
, {50, 9, 41}
, {-21, -50, -37}
, {-5, 41, -54}
, {-19, -45, -46}
, {30, -37, -19}
, {-33, 9, 41}
, {49, 19, -23}
, {12, 29, 3}
, {15, 19, -33}
, {12, 30, 1}
, {30, -26, 24}
, {-44, -41, -42}
, {2, -51, -20}
, {-17, -53, -22}
, {45, 11, 38}
, {-52, -15, -51}
, {33, -21, 51}
, {30, -48, -35}
, {27, -54, 17}
, {-24, 55, 35}
}
, {{32, -2, -39}
, {18, -42, 15}
, {-39, 40, 19}
, {-16, 54, 24}
, {-23, -27, -56}
, {4, -15, 38}
, {-33, 31, 4}
, {23, -8, 31}
, {47, 35, -13}
, {-14, -40, -45}
, {1, 60, 15}
, {-39, 38, -53}
, {7, 29, -18}
, {35, -9, -45}
, {21, 25, 19}
, {-41, -16, -6}
, {11, 24, 35}
, {3, -46, -26}
, {-8, 15, -22}
, {-49, 28, 27}
, {-26, -21, 27}
, {-39, -31, 39}
, {-32, 17, -53}
, {-53, -21, -11}
, {21, -15, -55}
, {19, -21, -6}
, {32, -35, -17}
, {24, 3, -27}
, {4, -35, 8}
, {-42, 23, -33}
, {-35, -51, 32}
, {6, 27, -27}
, {-25, -2, -10}
, {46, -2, 27}
, {-36, -20, 15}
, {45, -40, 0}
, {15, 33, -12}
, {35, -39, 22}
, {-32, -26, -23}
, {55, -43, -11}
, {4, 45, -50}
, {14, 13, 33}
, {5, -24, 0}
, {-17, 3, 21}
, {-22, -40, 36}
, {-35, 27, -44}
, {15, 30, -22}
, {27, 13, 0}
, {27, -51, 21}
, {41, 15, 39}
, {-28, -42, -32}
, {-21, 8, -15}
, {40, 42, -1}
, {18, 63, 39}
, {44, -9, -12}
, {-55, 29, -11}
, {-52, 39, 42}
, {3, 13, -14}
, {32, -33, -16}
, {-55, -27, -55}
, {-32, -51, -20}
, {15, 0, -17}
, {2, -60, -40}
, {-26, -34, 37}
}
, {{-21, 15, 34}
, {-12, -12, 51}
, {20, 29, -49}
, {27, -34, -37}
, {-51, 46, -9}
, {40, 13, 27}
, {33, 21, 51}
, {47, -30, 52}
, {-9, 27, -10}
, {-56, -68, -6}
, {-38, 38, -25}
, {2, 14, 2}
, {36, -8, -8}
, {0, 53, -24}
, {-8, 22, -27}
, {-3, -18, -51}
, {-22, 49, 46}
, {-11, 34, -48}
, {-19, 18, 40}
, {46, -20, 20}
, {14, 49, 21}
, {17, -25, 7}
, {10, -45, 20}
, {36, 14, -19}
, {-19, 40, 10}
, {-12, -29, 6}
, {11, 34, 45}
, {42, -3, 45}
, {-23, -35, 4}
, {-27, 29, 43}
, {37, 31, -37}
, {37, 23, -32}
, {-24, -50, -48}
, {-8, -14, -14}
, {11, -2, -21}
, {-46, 29, 4}
, {36, -11, 47}
, {-24, 0, 43}
, {-20, -22, -29}
, {4, 46, 23}
, {-34, -45, -42}
, {-23, -50, 20}
, {-7, 48, -36}
, {-14, 37, -39}
, {39, 21, 40}
, {-19, 16, -5}
, {5, 21, 21}
, {6, -8, -5}
, {-4, 42, -9}
, {53, -16, 52}
, {36, -30, -5}
, {-54, 4, 41}
, {-39, -16, -34}
, {-53, -46, -34}
, {9, 1, -42}
, {-23, -8, 6}
, {-6, -28, 46}
, {-11, 2, -19}
, {-31, 23, 42}
, {14, 30, -36}
, {-3, 37, -26}
, {-10, -6, -41}
, {-45, 40, 29}
, {-29, 3, -24}
}
, {{21, 31, 33}
, {39, 13, 18}
, {-23, -7, -54}
, {-2, 15, 42}
, {27, -1, -29}
, {19, 45, -9}
, {45, 17, 34}
, {-49, 47, 24}
, {-30, 6, -17}
, {40, -6, 23}
, {-37, -20, -43}
, {-35, 57, 46}
, {-29, -43, -31}
, {10, 23, -36}
, {-2, -14, 9}
, {14, -28, 23}
, {9, 48, 18}
, {37, -13, -15}
, {-45, -30, 44}
, {47, 13, 14}
, {34, 35, -6}
, {35, -17, -4}
, {24, 46, -46}
, {11, 33, -35}
, {-4, 27, -21}
, {-45, -15, -29}
, {34, -15, 33}
, {-39, -15, 32}
, {-44, 26, -47}
, {7, -35, 25}
, {-14, -30, -25}
, {30, -51, -30}
, {36, 45, -14}
, {1, -49, 26}
, {-17, -10, 7}
, {27, 11, -3}
, {-1, -42, -9}
, {-4, -17, -41}
, {-29, 2, -15}
, {-31, 48, 6}
, {-46, -43, 38}
, {-45, 39, 37}
, {-50, 30, 10}
, {-46, 27, 20}
, {4, 2, -10}
, {-32, 55, -15}
, {24, 1, 3}
, {44, -15, -1}
, {30, -34, 15}
, {5, 5, -50}
, {3, 24, -8}
, {45, -37, 16}
, {36, 36, 11}
, {4, 1, 38}
, {10, -44, -12}
, {-1, 52, -20}
, {43, -14, -26}
, {6, -43, 5}
, {29, -47, 11}
, {5, -13, -2}
, {-45, -30, 41}
, {-21, 26, -50}
, {-5, 48, 30}
, {28, 58, 29}
}
, {{5, 25, -39}
, {26, -35, -54}
, {34, -25, -1}
, {16, -17, -26}
, {44, 20, -38}
, {-19, -16, 0}
, {35, 17, -23}
, {-53, 32, -34}
, {-43, -16, -9}
, {-19, -35, -30}
, {1, -44, -21}
, {11, 1, -36}
, {-43, -28, 39}
, {7, 27, 37}
, {-50, 0, 45}
, {0, -18, 46}
, {21, -24, -23}
, {-38, -28, 29}
, {-20, -54, -52}
, {-24, 37, -29}
, {9, -23, -5}
, {44, 3, -40}
, {-23, -51, 2}
, {-41, -3, -35}
, {-29, -31, 34}
, {-3, 35, -18}
, {-19, -32, 14}
, {-12, -48, -16}
, {-14, -38, -50}
, {-44, -26, -18}
, {-5, -40, -8}
, {-4, -51, -31}
, {23, 17, 0}
, {-7, -24, -44}
, {10, -15, -42}
, {17, -18, 20}
, {37, -37, 12}
, {-30, -18, 4}
, {28, 42, -57}
, {36, 21, 7}
, {-54, -1, 41}
, {-49, 4, 25}
, {44, 44, -6}
, {5, -17, -25}
, {-17, -56, -32}
, {30, -33, -34}
, {-4, 29, -20}
, {-38, -54, -42}
, {-2, 36, 13}
, {26, -4, -56}
, {33, -38, 47}
, {28, 22, -7}
, {37, 22, 20}
, {-44, 4, 6}
, {-24, -52, -56}
, {-53, -6, -45}
, {-15, 11, -20}
, {8, 46, 17}
, {-9, 11, -19}
, {32, -42, -12}
, {12, 0, 17}
, {-14, 46, 34}
, {-44, -13, 41}
, {4, -51, -5}
}
, {{20, 24, -29}
, {-48, 46, 3}
, {-35, 3, -18}
, {45, 53, -19}
, {27, -2, 0}
, {16, -52, -47}
, {19, 26, -35}
, {-26, -54, 27}
, {37, 31, 3}
, {3, -17, 13}
, {-34, 47, 67}
, {-28, -47, 12}
, {11, -28, -34}
, {-9, -2, 26}
, {9, -47, 36}
, {-43, -20, 3}
, {-36, -47, -29}
, {33, 8, 46}
, {11, 25, -23}
, {-19, 4, -20}
, {-14, 27, -23}
, {-6, 11, 31}
, {-6, 30, -37}
, {25, 0, -44}
, {17, 15, -45}
, {-13, -10, -38}
, {-27, 0, 20}
, {16, 46, 4}
, {2, 10, 46}
, {42, -51, -29}
, {10, 6, 48}
, {-52, 51, 14}
, {22, -7, -40}
, {-45, 13, -21}
, {-37, -46, -7}
, {-46, 35, -30}
, {41, -29, -28}
, {-19, -26, -24}
, {-10, -2, -28}
, {8, -8, 46}
, {-31, 5, -35}
, {-14, -34, 21}
, {2, -38, 50}
, {-34, 45, -10}
, {-22, 46, 18}
, {2, 46, 23}
, {-25, -41, -8}
, {-23, 5, -38}
, {-25, 46, -28}
, {-43, -25, -36}
, {25, -43, 42}
, {-17, 47, 20}
, {56, -19, 22}
, {0, 0, 26}
, {40, 18, 24}
, {-2, 34, -47}
, {-5, -17, 0}
, {22, 46, -45}
, {9, -5, -23}
, {6, -30, 13}
, {-10, -43, 47}
, {-38, 2, 46}
, {27, 22, -51}
, {13, -10, 40}
}
, {{-19, -38, 25}
, {0, 5, 30}
, {37, -4, 54}
, {36, 15, -30}
, {-28, 15, -12}
, {-5, 22, -25}
, {22, 52, 22}
, {2, -23, -29}
, {-45, -1, -11}
, {24, -29, -22}
, {-5, -29, 8}
, {-30, 32, -24}
, {24, -51, 46}
, {-18, -57, -24}
, {18, 20, -25}
, {-3, 37, 31}
, {-34, -16, 39}
, {28, -16, -41}
, {-25, 37, -23}
, {41, 39, 45}
, {61, 17, 39}
, {-17, -50, 23}
, {2, 16, -37}
, {-41, 7, 9}
, {-25, 28, 44}
, {19, 15, -39}
, {-37, 32, -46}
, {-50, -18, -2}
, {50, -17, 28}
, {14, 24, -40}
, {-31, 25, 22}
, {-33, 34, -42}
, {46, -37, -16}
, {47, 56, 52}
, {-23, 51, 18}
, {30, -12, -15}
, {-43, -31, 16}
, {46, -1, -17}
, {42, 45, 52}
, {57, 28, 66}
, {-5, 8, -49}
, {7, 58, -28}
, {-6, -42, 53}
, {42, -44, 3}
, {19, -17, 8}
, {5, -28, 11}
, {-16, 18, 56}
, {-18, 6, 5}
, {-27, -53, -8}
, {19, 6, -49}
, {-19, 40, -13}
, {42, 12, 50}
, {-25, 38, -45}
, {-14, 25, 45}
, {-33, -24, 9}
, {16, -29, 33}
, {11, 16, 44}
, {5, 29, 52}
, {-26, -3, 28}
, {-28, -32, 6}
, {24, 7, -21}
, {-50, 3, 41}
, {-13, -36, 5}
, {-41, 45, -12}
}
, {{-16, 27, 15}
, {13, -16, -31}
, {45, -21, 56}
, {47, 9, -6}
, {45, 15, 19}
, {-30, -37, -23}
, {4, 45, -16}
, {-17, -33, -10}
, {-49, 21, 8}
, {-30, -4, 42}
, {-33, 61, 23}
, {-23, -3, 59}
, {-2, 3, 47}
, {-31, 24, 3}
, {44, 38, 23}
, {-12, 13, 49}
, {26, -24, 24}
, {14, 14, 13}
, {-8, 0, 2}
, {7, 41, 37}
, {5, -13, -48}
, {58, 53, -27}
, {-49, -19, 55}
, {34, -26, 35}
, {28, 1, 8}
, {-7, -8, 36}
, {19, 14, -47}
, {-46, -59, -11}
, {36, 18, -20}
, {21, -56, -22}
, {-7, 36, 50}
, {46, -12, -49}
, {-37, 30, 32}
, {-39, -7, 49}
, {5, 0, 1}
, {29, 42, -53}
, {23, 42, -23}
, {-3, 13, -45}
, {-37, 9, 2}
, {-27, -13, 15}
, {-21, 42, 21}
, {-12, 49, -50}
, {8, -8, -28}
, {31, 4, -39}
, {-19, 21, -41}
, {-42, -33, 40}
, {36, 40, -27}
, {30, -17, -19}
, {-19, -47, -54}
, {49, 36, -23}
, {-29, -46, -48}
, {39, -20, -23}
, {2, -25, -6}
, {-49, -31, 3}
, {45, 40, 34}
, {31, 37, 21}
, {47, -22, 37}
, {-30, 52, -34}
, {-45, 1, -36}
, {-16, 21, -13}
, {7, -34, -27}
, {-6, 48, 43}
, {2, 4, -11}
, {-26, 6, -7}
}
, {{12, -16, -40}
, {53, 52, -40}
, {34, -37, 30}
, {-58, -34, -7}
, {48, 52, 28}
, {-44, -45, -15}
, {-31, 0, 18}
, {-43, -31, 30}
, {-21, 0, 37}
, {0, -12, -56}
, {30, 6, 43}
, {-38, -37, 1}
, {25, -27, -46}
, {64, 12, -11}
, {17, 39, -33}
, {-9, -30, 41}
, {50, -46, -13}
, {-3, 46, -13}
, {-7, 4, 26}
, {13, 23, 13}
, {7, 19, -20}
, {13, -27, 8}
, {-28, 14, 1}
, {-9, 4, -17}
, {-20, 9, 7}
, {-15, -28, -12}
, {15, -37, 32}
, {54, 54, 44}
, {11, -19, 14}
, {-38, 2, 53}
, {-37, 17, -31}
, {-24, 50, -9}
, {7, 13, -40}
, {-5, -14, -39}
, {6, 35, 7}
, {28, 0, 25}
, {-8, 40, 37}
, {-10, -25, -31}
, {37, -38, -33}
, {29, 28, -30}
, {-37, -15, -29}
, {24, 39, 8}
, {19, -20, 7}
, {24, 2, 55}
, {35, 7, -21}
, {-5, 18, 3}
, {-2, 20, -29}
, {-7, -30, -26}
, {53, -27, 48}
, {50, 35, 50}
, {0, 24, 52}
, {-45, 7, -10}
, {45, -41, 40}
, {30, 11, 43}
, {-32, 48, -27}
, {-18, -32, 0}
, {-3, 2, 0}
, {15, -46, -32}
, {21, 40, -40}
, {30, -25, 19}
, {-30, -26, 49}
, {6, -43, -22}
, {22, 42, 47}
, {25, -5, 50}
}
, {{33, 37, -2}
, {-15, 39, 16}
, {-20, 5, 18}
, {2, 15, -25}
, {11, -35, 20}
, {44, 8, -41}
, {37, -48, -25}
, {21, 41, 3}
, {35, 27, 4}
, {-16, 63, -28}
, {1, 25, -4}
, {-27, 26, -26}
, {33, -23, 26}
, {-47, 26, 11}
, {-27, -26, 30}
, {29, 33, -41}
, {-41, 34, -31}
, {44, -24, 34}
, {50, -30, -46}
, {-34, 31, 8}
, {-50, -47, 16}
, {-20, 40, 8}
, {42, -50, 41}
, {46, -11, 23}
, {-18, -39, 6}
, {-35, -24, -41}
, {-50, 9, -54}
, {28, -17, 34}
, {1, 14, 26}
, {-22, -42, -14}
, {37, -27, 49}
, {-21, -30, 46}
, {26, -3, -39}
, {-48, 0, 57}
, {-42, 8, 38}
, {3, 18, -58}
, {-2, 22, -48}
, {-16, -10, 3}
, {47, 18, 40}
, {55, 28, 54}
, {-36, 53, -12}
, {0, -25, -6}
, {-29, 47, -42}
, {-53, -3, -12}
, {-23, -7, -22}
, {1, -43, -38}
, {-11, -21, -5}
, {-51, -14, 47}
, {27, 2, 17}
, {14, -43, -48}
, {2, 18, 12}
, {26, -41, 19}
, {48, 7, 30}
, {-42, 40, 11}
, {37, 23, 31}
, {-46, 41, -17}
, {-49, 55, 55}
, {-65, -9, -4}
, {-33, -23, 16}
, {64, 25, 6}
, {-2, 16, -49}
, {-19, 19, 51}
, {-2, 37, -41}
, {8, 19, -45}
}
, {{34, 10, 16}
, {19, -5, 19}
, {-57, -65, -29}
, {58, 68, 37}
, {27, 5, -9}
, {-19, 14, -1}
, {24, 25, -35}
, {28, -27, 6}
, {-13, 34, 20}
, {-51, -28, 0}
, {-1, 18, 5}
, {-19, -16, -55}
, {2, 3, 26}
, {42, -25, 0}
, {17, 30, -40}
, {51, 20, 29}
, {-24, 19, -2}
, {28, 32, 27}
, {28, -22, 48}
, {19, 16, 0}
, {27, 19, 8}
, {-27, -20, 2}
, {39, 25, -56}
, {-42, -62, -62}
, {34, 42, 5}
, {45, 19, -29}
, {26, -47, -12}
, {-2, 2, -5}
, {34, 33, 49}
, {40, -49, 38}
, {-34, 26, 30}
, {-22, 45, 7}
, {-10, -46, -10}
, {37, 44, 39}
, {34, 33, -20}
, {-45, -2, 25}
, {-28, 7, -24}
, {27, 35, -11}
, {-19, 24, 6}
, {-12, -7, -6}
, {2, -19, 14}
, {21, 51, 0}
, {33, -18, 40}
, {-3, 10, 45}
, {-22, 19, 26}
, {38, 35, 31}
, {-26, 33, 29}
, {-7, -22, -9}
, {-3, -4, -39}
, {-21, 16, 29}
, {34, -30, -6}
, {-28, -24, -49}
, {42, 60, 36}
, {15, 48, 2}
, {6, 6, -17}
, {20, -16, -45}
, {27, 32, -31}
, {-9, -7, 46}
, {-18, 14, -27}
, {-6, -8, 0}
, {19, 36, -6}
, {-23, 43, 13}
, {-44, 27, -38}
, {-23, -56, -5}
}
, {{-26, -65, -53}
, {-16, -33, -47}
, {-27, -15, -4}
, {-33, -12, -12}
, {-49, 34, -24}
, {3, -31, 0}
, {47, 34, -7}
, {25, 1, 36}
, {0, -29, 8}
, {7, -7, 32}
, {-9, -40, 17}
, {-48, -14, 1}
, {50, 32, 51}
, {57, -29, -28}
, {42, -36, -24}
, {41, 3, 47}
, {-41, 31, -38}
, {17, 30, 25}
, {-49, 19, -34}
, {-31, -12, 24}
, {-17, -39, 36}
, {30, -37, -37}
, {-16, 19, 20}
, {28, -8, -44}
, {58, 53, -40}
, {-37, -25, -5}
, {-8, 12, 22}
, {50, -24, 46}
, {-48, -29, 35}
, {31, 14, 59}
, {1, -33, -62}
, {19, 31, 12}
, {17, 31, -30}
, {-20, 0, -9}
, {2, -37, -17}
, {6, 35, -38}
, {2, 48, 33}
, {17, 7, 53}
, {-30, -50, -3}
, {-67, -13, -39}
, {-31, -49, -50}
, {-22, 11, 47}
, {-21, 8, 3}
, {-32, 3, 16}
, {-18, -36, 55}
, {-14, -18, 22}
, {41, 26, 8}
, {50, 2, -26}
, {-32, -37, 5}
, {1, -38, -44}
, {-14, 36, 6}
, {-17, -16, 3}
, {3, 50, -39}
, {-16, -34, -11}
, {-16, 50, -34}
, {-39, 32, 5}
, {-36, -57, -50}
, {39, 44, -43}
, {27, -18, 24}
, {-4, -24, 3}
, {-21, 54, -41}
, {6, 37, 9}
, {-38, -43, 37}
, {30, 2, -28}
}
, {{37, 24, 27}
, {-15, 43, 48}
, {-26, -41, -41}
, {36, 44, 25}
, {35, -5, -12}
, {-45, 11, -15}
, {40, 3, 41}
, {-8, -36, -8}
, {-18, 8, 32}
, {12, -23, 18}
, {-27, -43, -46}
, {-20, -2, 24}
, {-25, -20, 26}
, {-9, 26, -23}
, {-24, -6, -10}
, {50, 52, -30}
, {-32, -6, -21}
, {44, 10, 5}
, {-32, 0, 12}
, {-24, 47, -42}
, {-8, -2, 18}
, {-12, 31, -31}
, {41, 17, 17}
, {-25, -25, 21}
, {-45, 3, -54}
, {43, -36, -21}
, {18, 11, 41}
, {6, -12, 36}
, {21, -49, 14}
, {-35, -44, 22}
, {27, -12, 26}
, {9, 18, 43}
, {15, 30, 33}
, {-41, -50, 26}
, {16, -38, -34}
, {-21, -49, 44}
, {2, -7, -41}
, {45, 24, -3}
, {-46, -50, -11}
, {18, -36, 29}
, {-9, 12, 41}
, {-11, 16, 14}
, {-9, 37, 12}
, {0, 17, -7}
, {-13, 28, 4}
, {53, 61, 21}
, {23, -1, -20}
, {-8, 8, -33}
, {-26, -20, -49}
, {32, -51, 9}
, {44, 25, 28}
, {-2, -17, -13}
, {55, 42, -23}
, {-37, 34, -17}
, {32, 1, 44}
, {26, 19, -46}
, {13, 5, 37}
, {50, -16, -3}
, {-20, 22, -29}
, {-12, -17, -8}
, {-15, -20, 9}
, {34, -43, -31}
, {8, -40, 24}
, {14, 21, 31}
}
, {{3, 10, -27}
, {-36, 1, -32}
, {47, 55, 28}
, {36, 21, -24}
, {-29, -27, -52}
, {-23, -36, -22}
, {0, 14, -24}
, {-46, -45, 47}
, {-51, 14, -47}
, {3, 60, 0}
, {55, 2, 43}
, {53, 44, -36}
, {7, 59, -22}
, {25, 21, 10}
, {-14, 53, 2}
, {13, -50, -54}
, {17, 52, 45}
, {-42, 45, -25}
, {-15, 30, 48}
, {26, -10, 51}
, {-42, -30, -21}
, {-1, 53, 48}
, {14, 18, -11}
, {28, 7, 7}
, {34, 31, 42}
, {2, 31, 6}
, {50, -10, -46}
, {24, 56, -15}
, {40, 48, -19}
, {-44, -9, 22}
, {1, -35, 41}
, {-23, 0, 48}
, {-15, -21, 42}
, {7, 32, 11}
, {-6, -50, 13}
, {26, -4, -13}
, {-3, 61, -42}
, {7, 22, -9}
, {42, -42, 16}
, {-25, -11, -22}
, {43, 12, 40}
, {13, 19, -3}
, {42, 44, 6}
, {-8, -1, 12}
, {10, -13, -32}
, {32, -38, 19}
, {35, -33, -37}
, {-15, 32, -46}
, {12, 28, 11}
, {-44, -40, 12}
, {45, -12, 47}
, {51, -6, 10}
, {14, -44, -5}
, {56, 22, -36}
, {-20, 17, -40}
, {-2, 3, -53}
, {23, -24, -6}
, {-6, 2, 21}
, {34, 34, -14}
, {66, 11, 32}
, {19, -29, 15}
, {20, 18, 32}
, {-45, 41, -13}
, {8, -7, -45}
}
, {{-38, 15, 18}
, {-37, -3, -33}
, {-23, -23, -5}
, {37, -43, -21}
, {26, -47, 25}
, {40, -44, -51}
, {-42, -25, 0}
, {-9, -37, -29}
, {-40, -46, -40}
, {-5, 32, -8}
, {-3, -19, 24}
, {-1, -50, 22}
, {42, -35, -43}
, {-45, -35, 30}
, {-27, -26, 20}
, {40, -26, -39}
, {-27, -14, -20}
, {-28, 42, -21}
, {41, 22, -22}
, {-45, 22, -8}
, {-3, 21, 2}
, {-30, -35, -26}
, {-39, -31, -3}
, {26, -16, -37}
, {17, 43, 24}
, {0, -33, 35}
, {-24, 0, 46}
, {-33, -44, -22}
, {-1, 52, -1}
, {0, -17, -35}
, {25, -36, 25}
, {-37, -9, -10}
, {-18, 9, -43}
, {-5, 30, 37}
, {-37, 42, -3}
, {-4, -21, 21}
, {-33, 53, 33}
, {38, -12, 15}
, {-17, 48, 31}
, {30, 9, -45}
, {-5, 40, -38}
, {23, -7, -52}
, {16, -15, 13}
, {29, 6, -30}
, {25, 0, 18}
, {-35, -23, -11}
, {18, 41, -19}
, {41, 20, 15}
, {-42, -47, 13}
, {-15, 52, 31}
, {-9, -38, -20}
, {-21, -31, 2}
, {33, 0, 25}
, {-35, -11, 7}
, {9, -44, 4}
, {-51, -17, -42}
, {-29, 15, 16}
, {6, -34, 35}
, {-8, -15, -36}
, {0, -15, -39}
, {-24, -38, -17}
, {31, 25, 12}
, {8, -36, 43}
, {4, -10, 34}
}
, {{66, 6, 0}
, {43, 0, 50}
, {57, -18, 34}
, {36, -19, -24}
, {-9, 49, -6}
, {-45, 2, 0}
, {-15, 1, -35}
, {-9, -16, 44}
, {-41, 33, 23}
, {63, -29, 12}
, {39, -6, 11}
, {-15, -11, -4}
, {-45, 38, -32}
, {-55, -23, -2}
, {48, 67, -1}
, {-13, 22, -23}
, {-36, 14, -1}
, {1, -26, 38}
, {53, 28, 33}
, {18, -26, -44}
, {1, 39, -2}
, {54, -6, 2}
, {-18, 13, -51}
, {44, -2, 1}
, {-46, 55, 0}
, {-7, -4, 38}
, {-17, -2, -5}
, {12, 31, 2}
, {0, -14, -8}
, {6, 32, 42}
, {7, 23, -13}
, {24, -10, -37}
, {-43, -8, 0}
, {28, -22, 39}
, {40, 14, -23}
, {-26, -3, 0}
, {-38, 41, -20}
, {-4, 34, 0}
, {-10, 6, -3}
, {34, 16, -15}
, {20, -18, 27}
, {4, -38, 45}
, {16, 37, -27}
, {58, 64, 9}
, {-28, -41, -35}
, {-41, 49, -28}
, {-2, 31, -44}
, {29, -13, 58}
, {33, -3, -47}
, {28, 41, -43}
, {21, -31, 10}
, {61, -45, -27}
, {8, -2, 30}
, {-42, -49, -45}
, {-23, 10, -20}
, {26, -2, -26}
, {-33, 4, 46}
, {44, -35, -3}
, {-35, -43, -28}
, {23, -19, -5}
, {-32, 48, 34}
, {63, -21, -26}
, {31, 4, 13}
, {-18, -37, -20}
}
, {{-51, 42, 51}
, {-25, -32, 41}
, {-14, -2, -52}
, {21, 34, 47}
, {29, 5, -8}
, {23, -46, -37}
, {-5, 17, -52}
, {4, -55, 42}
, {-39, 43, -44}
, {9, -4, -18}
, {-3, 34, 1}
, {37, 50, 42}
, {7, 32, 6}
, {0, -36, -11}
, {26, 3, -50}
, {8, 40, -9}
, {-34, 29, -50}
, {-32, 40, -47}
, {45, -10, -35}
, {-32, -29, -26}
, {-34, 37, -32}
, {-35, 28, -25}
, {-26, -47, -42}
, {-42, -14, -48}
, {41, 45, 27}
, {4, 33, 39}
, {28, -17, -15}
, {-21, 33, -18}
, {-3, 4, -17}
, {-23, -14, -2}
, {10, 33, -11}
, {13, 4, -43}
, {-19, -15, 44}
, {-44, 45, -14}
, {12, 43, -4}
, {4, -33, -18}
, {-28, 0, -37}
, {-28, 17, 16}
, {-23, 0, -10}
, {37, -45, 3}
, {-48, -11, 20}
, {28, -26, 7}
, {-13, 45, -55}
, {-31, 13, -26}
, {40, -51, 24}
, {-37, -37, -31}
, {-36, -7, -13}
, {-38, 0, 6}
, {0, -37, 29}
, {-14, -8, 28}
, {-58, 25, -53}
, {36, 29, -8}
, {-42, -5, 11}
, {-40, -40, 40}
, {-24, -43, -39}
, {32, -29, -34}
, {-30, -23, 46}
, {-37, 4, 49}
, {-42, 43, -20}
, {10, 0, 50}
, {36, -36, 5}
, {5, -37, -11}
, {46, 22, 50}
, {-38, -39, -3}
}
, {{29, -39, -11}
, {24, -34, 0}
, {27, 61, 35}
, {4, 31, -21}
, {-12, 34, -31}
, {22, -11, 38}
, {-13, 7, -20}
, {42, -16, 55}
, {0, 5, 46}
, {-15, 61, -58}
, {-30, -13, 8}
, {17, 1, -23}
, {-9, 36, 18}
, {0, 37, 53}
, {-51, 5, -43}
, {-11, 13, 12}
, {-48, 47, 48}
, {-51, -26, -11}
, {38, -19, -43}
, {19, 33, 49}
, {4, -12, -6}
, {55, -33, 43}
, {41, 53, 51}
, {61, 19, 59}
, {-37, 39, 0}
, {-43, -38, -33}
, {22, 26, 45}
, {41, -21, -26}
, {-21, -29, 8}
, {-26, -33, 16}
, {-30, 4, -41}
, {-1, 43, 4}
, {49, 45, 32}
, {20, -42, -25}
, {40, 12, -27}
, {-45, -23, 20}
, {-43, -16, -48}
, {-24, 25, -39}
, {-5, -37, -50}
, {12, 25, 16}
, {37, -24, -23}
, {-33, -18, -27}
, {-2, -20, 6}
, {-46, 29, 26}
, {49, -14, 44}
, {41, 4, -13}
, {7, 36, 1}
, {-19, -39, -27}
, {-18, 3, -23}
, {40, -29, -32}
, {-1, -25, -8}
, {46, 40, 3}
, {-16, -42, 30}
, {-36, -18, 6}
, {24, -27, 42}
, {-8, 46, 6}
, {42, -23, 53}
, {15, -16, -47}
, {10, -47, -39}
, {-14, 27, -22}
, {21, 3, 22}
, {33, -31, 15}
, {-41, 12, 0}
, {-31, 51, -30}
}
, {{-4, -19, -20}
, {-8, 11, -9}
, {-25, 63, -40}
, {44, 24, 19}
, {-21, 28, 16}
, {-28, 20, -18}
, {31, 28, 49}
, {-41, -31, 20}
, {7, -52, 45}
, {38, 6, 29}
, {-12, 45, 36}
, {4, -38, 10}
, {42, -34, 33}
, {33, 45, -46}
, {27, -34, 30}
, {24, -52, 32}
, {23, 12, 37}
, {29, 32, -46}
, {-4, 37, -30}
, {48, -36, -39}
, {36, -17, -59}
, {22, -29, -57}
, {22, -18, 41}
, {-19, -25, -16}
, {-17, -30, 34}
, {42, 34, 18}
, {-35, -25, 17}
, {-27, -25, 36}
, {-2, -33, -36}
, {19, -50, 34}
, {32, 9, -10}
, {-44, 39, 13}
, {-43, -32, -40}
, {-31, -7, -39}
, {28, 32, -7}
, {22, -26, -24}
, {-1, 11, 50}
, {-37, 48, -55}
, {8, 41, 13}
, {-32, 29, 40}
, {-26, 2, -24}
, {-39, 5, 47}
, {28, -5, 37}
, {-44, -30, -32}
, {-10, -32, 11}
, {-25, -15, 0}
, {-15, -34, -21}
, {43, 20, -23}
, {12, -25, -50}
, {-14, 32, -33}
, {12, -18, 36}
, {-21, -53, 8}
, {-42, 23, -48}
, {47, -24, 19}
, {30, 5, -9}
, {16, 34, -10}
, {19, -25, -51}
, {18, 48, 9}
, {32, -40, -44}
, {25, -21, -19}
, {-12, -44, 42}
, {-44, 14, 22}
, {34, -44, 6}
, {36, 2, -34}
}
, {{47, 55, -7}
, {48, -23, -8}
, {-33, 33, 44}
, {38, 19, 22}
, {0, 40, 35}
, {-37, -3, -24}
, {16, -36, 33}
, {-32, -27, 65}
, {-29, -10, 0}
, {-48, 37, 32}
, {-20, 45, -39}
, {0, -15, 36}
, {5, 36, 15}
, {-29, 44, -29}
, {-10, -17, -35}
, {53, 0, -17}
, {-33, 51, -39}
, {-32, 12, -12}
, {-37, 5, 4}
, {40, 1, -41}
, {-50, -32, -46}
, {62, -7, 35}
, {-44, 49, 10}
, {17, -22, 51}
, {53, -52, 43}
, {-2, 10, 26}
, {-9, -16, 38}
, {-6, 53, 17}
, {-39, 59, 21}
, {-8, 12, 10}
, {-20, 18, 29}
, {-9, 34, -33}
, {-25, -30, 36}
, {1, -52, -23}
, {32, 27, 25}
, {19, -53, 15}
, {54, 19, 20}
, {24, 3, -49}
, {42, -42, -51}
, {50, -8, -37}
, {-41, 32, -38}
, {49, -13, 18}
, {-32, -7, 19}
, {51, 27, 26}
, {-37, -25, -4}
, {-44, 40, -19}
, {-6, -37, -13}
, {9, -2, 11}
, {-41, -60, -34}
, {28, -11, -31}
, {44, -27, -7}
, {-22, 35, 67}
, {21, -6, -29}
, {-28, 13, -28}
, {54, 15, -30}
, {64, 5, 29}
, {41, -4, -6}
, {54, 57, 29}
, {5, -24, -34}
, {-21, 57, 30}
, {2, 35, -1}
, {3, -32, -2}
, {41, -37, 14}
, {27, -58, -19}
}
, {{-32, 44, 27}
, {31, 28, -39}
, {11, 1, 41}
, {20, -22, -15}
, {60, 23, 53}
, {-48, -6, -17}
, {-38, 33, -9}
, {25, -36, 7}
, {16, 37, -17}
, {19, -45, 14}
, {-33, -22, 31}
, {5, 56, -20}
, {-25, 28, -57}
, {-19, 31, 7}
, {-24, 6, -49}
, {49, -11, 20}
, {-44, 8, -38}
, {-43, 40, 22}
, {-9, -14, -33}
, {34, 42, -8}
, {63, 66, 55}
, {-22, 7, 10}
, {-2, 17, -4}
, {18, -28, 0}
, {1, -9, -1}
, {-22, -37, 28}
, {37, 29, 44}
, {-53, 0, 17}
, {1, 14, 7}
, {-41, 8, -41}
, {-9, -35, -9}
, {-20, -52, -32}
, {-49, 42, 12}
, {24, -11, -40}
, {1, -38, 3}
, {42, -24, -22}
, {41, 1, -23}
, {5, 5, -32}
, {49, 4, 0}
, {28, -6, -4}
, {-31, -3, 14}
, {7, 52, -5}
, {-9, 19, 1}
, {0, -15, 0}
, {-10, 35, -25}
, {57, -28, 57}
, {17, -17, -35}
, {41, -38, 16}
, {-34, -10, -54}
, {11, -17, -29}
, {-43, -40, 11}
, {-45, -37, -30}
, {22, -2, -39}
, {-30, 37, -61}
, {45, 22, 26}
, {3, 52, 36}
, {41, 26, 45}
, {-31, 47, 31}
, {-5, -43, 30}
, {34, -32, 8}
, {10, -22, 40}
, {-9, 0, 38}
, {-34, 31, 57}
, {-21, 67, -21}
}
, {{-39, -16, -32}
, {26, 37, -40}
, {6, 12, 0}
, {4, -27, -2}
, {34, 3, 12}
, {26, 11, -13}
, {-8, 21, 21}
, {-36, -33, -35}
, {-25, -35, 8}
, {-43, -20, -66}
, {47, -19, 43}
, {41, 21, -27}
, {1, -35, -20}
, {-9, 2, 21}
, {-31, -7, -58}
, {8, 8, 43}
, {39, -27, -3}
, {10, 40, -34}
, {-28, 42, 43}
, {-13, 21, 15}
, {-6, 28, 59}
, {42, -12, -42}
, {-4, 26, -28}
, {0, 21, 11}
, {-39, 44, -32}
, {20, -51, 29}
, {-45, 50, -12}
, {-34, -55, -7}
, {-11, 24, 36}
, {-50, 41, -34}
, {-22, -49, 8}
, {-6, -17, 39}
, {32, 23, 29}
, {-26, 23, -34}
, {23, -31, 10}
, {-17, 9, -20}
, {12, 32, 46}
, {50, 49, 33}
, {49, -21, 13}
, {37, 29, -30}
, {33, 47, -36}
, {25, -14, -4}
, {-18, 47, 15}
, {-27, 42, 37}
, {6, 9, 2}
, {-33, 55, 11}
, {5, 35, 9}
, {12, -25, -34}
, {32, 12, 47}
, {3, 42, 27}
, {-24, -24, -51}
, {-41, 2, 52}
, {42, 23, 39}
, {-14, 0, 11}
, {22, -9, -31}
, {-5, 35, 32}
, {-29, 53, 55}
, {-30, -7, 10}
, {38, -33, 51}
, {23, -42, -6}
, {-47, 45, 7}
, {4, 37, 27}
, {14, 27, 29}
, {-4, -30, 38}
}
, {{-33, 26, 27}
, {1, -37, -54}
, {8, 37, -5}
, {-19, 10, -3}
, {68, -3, 16}
, {6, -5, 0}
, {45, 25, 3}
, {-54, -5, 46}
, {39, -44, 13}
, {-53, -6, -11}
, {35, -16, 20}
, {4, 23, -24}
, {-42, -38, 23}
, {47, -6, 54}
, {12, -17, 12}
, {15, -8, -32}
, {19, 34, -28}
, {-17, 12, -18}
, {21, -6, 29}
, {-49, -34, -43}
, {-20, -66, 1}
, {-20, -8, 7}
, {38, -32, -35}
, {-20, -35, -4}
, {43, 0, 43}
, {16, -13, 0}
, {-9, 1, -8}
, {-7, -19, 22}
, {15, -21, -22}
, {34, -41, 38}
, {-38, 30, 14}
, {32, 33, -7}
, {-27, 14, 3}
, {5, -29, 7}
, {29, -48, 44}
, {40, 48, 4}
, {-42, 44, 45}
, {3, -50, -29}
, {-24, 4, -3}
, {-34, -47, 16}
, {-31, 20, 25}
, {-4, 39, -31}
, {-46, 46, -5}
, {-36, 6, 9}
, {-54, -32, -16}
, {0, -23, 60}
, {22, 36, -8}
, {36, -55, -44}
, {-32, -14, -18}
, {22, -36, 15}
, {44, 12, 48}
, {-24, -26, 42}
, {-43, -50, 34}
, {20, -14, -25}
, {0, 11, -48}
, {-45, 20, -28}
, {-22, -28, -57}
, {59, 20, 26}
, {-51, -47, -44}
, {-26, 19, 39}
, {-37, -36, -13}
, {-16, 46, -17}
, {-32, 5, 4}
, {-8, 24, -30}
}
, {{52, -10, 52}
, {-23, -3, -22}
, {6, -34, 33}
, {5, 38, 55}
, {38, 8, -29}
, {-18, 3, 20}
, {26, -24, 30}
, {23, 16, -25}
, {49, -27, -40}
, {-21, -14, 67}
, {-9, -25, -14}
, {-50, -26, -22}
, {37, 42, 9}
, {-52, -43, -35}
, {-20, -32, -35}
, {-38, -47, -15}
, {-22, -6, -8}
, {50, -26, 36}
, {-38, 49, 45}
, {-44, 8, -33}
, {-65, -3, -42}
, {35, -44, 8}
, {-6, -6, -35}
, {68, 59, 1}
, {3, -39, 38}
, {-42, 3, 23}
, {-19, 8, 7}
, {-16, 17, -7}
, {-7, 41, 3}
, {22, -32, 20}
, {46, -6, 38}
, {-41, 34, -49}
, {-43, 1, -42}
, {16, 55, 51}
, {37, 44, 49}
, {-27, -47, -12}
, {8, 42, 45}
, {29, 24, 0}
, {26, -31, -39}
, {-4, -27, 14}
, {34, -3, -32}
, {-17, 47, 8}
, {30, -44, 15}
, {-25, 45, -30}
, {16, -41, -44}
, {18, 31, -33}
, {49, 15, 47}
, {48, 34, -9}
, {5, 0, 50}
, {-24, -21, -31}
, {25, -9, -41}
, {-42, 37, -22}
, {21, 45, -16}
, {61, 34, 19}
, {11, -18, 28}
, {64, 14, 18}
, {32, -14, -30}
, {-39, 18, -9}
, {12, -2, -21}
, {60, -36, 30}
, {-6, -50, 31}
, {32, 51, -13}
, {0, -4, 50}
, {8, -2, -29}
}
, {{-37, 7, -15}
, {8, 25, 49}
, {-46, -34, 28}
, {-3, 5, -7}
, {-3, 16, -46}
, {-26, -30, 0}
, {42, 21, -15}
, {34, -13, -35}
, {13, -14, 17}
, {-15, 37, -7}
, {-37, 20, -42}
, {7, 13, 9}
, {-46, -17, -10}
, {21, -11, -8}
, {27, 6, 41}
, {8, 8, 6}
, {-13, -5, 30}
, {5, -5, 50}
, {-1, -30, 33}
, {9, 46, -49}
, {-36, -30, 13}
, {3, -6, 36}
, {-30, -36, 27}
, {49, 17, -38}
, {14, 14, -2}
, {-23, -10, 40}
, {9, -41, 22}
, {49, 18, -27}
, {31, -25, 18}
, {50, -24, -32}
, {53, 2, -13}
, {-37, 42, -31}
, {-45, 8, -21}
, {25, -21, -16}
, {-39, -1, 0}
, {-32, 11, -3}
, {-22, 45, 41}
, {45, 6, 3}
, {-12, 38, 16}
, {33, -32, 39}
, {-39, 38, 8}
, {48, -35, -29}
, {45, -24, -43}
, {48, 30, 40}
, {-44, 49, -6}
, {0, -19, 10}
, {-21, -10, 43}
, {-32, 7, 37}
, {-36, 33, -20}
, {31, -9, -13}
, {26, -49, -23}
, {44, 10, -41}
, {20, 30, -46}
, {10, -49, -2}
, {0, 10, -32}
, {22, -47, 24}
, {-27, -31, 2}
, {42, 18, -27}
, {40, -23, -4}
, {17, -8, 0}
, {33, 50, 40}
, {45, -2, 5}
, {13, 51, 22}
, {-18, 28, 31}
}
, {{7, 28, -13}
, {45, -59, -25}
, {-32, -56, 27}
, {44, -23, 22}
, {-29, 22, -27}
, {-39, -4, 21}
, {-20, 22, -24}
, {29, 28, -34}
, {-55, -29, -27}
, {-51, 28, 51}
, {33, 29, -49}
, {13, -51, -50}
, {-54, 5, 45}
, {-6, 42, 5}
, {26, 0, -10}
, {17, 38, 11}
, {40, -2, -24}
, {-26, -1, -6}
, {5, -21, -12}
, {-43, 14, -41}
, {39, 52, -13}
, {-9, 6, -41}
, {-16, -54, -1}
, {25, 32, -11}
, {43, -46, -10}
, {-21, -21, 48}
, {-37, -7, 24}
, {-58, 5, 7}
, {17, -36, -20}
, {-56, 39, 2}
, {-22, 33, 4}
, {-3, 28, -10}
, {17, -52, 29}
, {-22, 19, -5}
, {9, -29, -3}
, {-26, 29, -44}
, {20, 38, -29}
, {0, -4, 24}
, {-19, -10, 1}
, {6, 24, -31}
, {11, -39, 31}
, {12, -48, 1}
, {-20, 22, -35}
, {-39, -45, -36}
, {-12, 32, -32}
, {26, -18, 48}
, {-36, 19, 44}
, {-53, 26, -39}
, {-41, -27, 27}
, {-43, -11, -16}
, {20, -2, 19}
, {-11, -32, -19}
, {45, 9, -19}
, {3, 37, -44}
, {-12, -20, -16}
, {-35, -35, -48}
, {39, 35, -41}
, {-14, 43, -41}
, {-20, 35, 3}
, {28, 40, -52}
, {36, 28, -32}
, {21, 16, -21}
, {42, -23, -30}
, {41, 17, -38}
}
, {{-26, -40, -8}
, {34, 34, -4}
, {28, 1, 38}
, {-37, -30, -9}
, {-39, -37, -13}
, {-23, -3, -29}
, {-49, -6, 49}
, {36, -20, -19}
, {25, -54, -4}
, {-28, 49, 49}
, {21, 30, -39}
, {9, 42, -54}
, {0, -27, -40}
, {2, 0, -45}
, {33, 16, -44}
, {-17, 3, -2}
, {50, -23, -19}
, {39, 34, -22}
, {-11, -27, -54}
, {45, -5, 42}
, {1, 22, -8}
, {11, -52, 38}
, {-25, -36, -7}
, {-44, -32, -10}
, {11, 39, 41}
, {14, -37, -43}
, {-35, -15, -53}
, {-20, 4, -17}
, {-35, -10, -41}
, {40, 34, -11}
, {2, -34, 26}
, {26, -49, -16}
, {21, -2, -36}
, {-25, 18, -14}
, {14, 38, -38}
, {4, 0, -21}
, {-41, -44, 38}
, {27, -48, 7}
, {-7, -39, -35}
, {1, 0, 8}
, {29, -41, 2}
, {12, -26, 0}
, {36, 43, 27}
, {-42, 33, 43}
, {-43, 30, 8}
, {20, 1, -10}
, {-33, 52, -3}
, {38, 35, 19}
, {16, 49, 5}
, {-5, -1, 4}
, {39, 41, -43}
, {-27, -16, -41}
, {23, 35, -34}
, {31, 19, -14}
, {-30, -31, 37}
, {-20, -7, -14}
, {38, 26, -29}
, {23, -41, -1}
, {-6, -37, -33}
, {-24, 49, 33}
, {18, -47, -26}
, {12, 20, 14}
, {16, -19, 28}
, {-29, 55, -1}
}
, {{6, -17, 44}
, {48, 40, -36}
, {0, -39, 40}
, {-21, 7, -31}
, {40, 0, -24}
, {20, 47, 9}
, {-13, 27, 7}
, {36, 1, -1}
, {-28, 33, -54}
, {-30, 72, 52}
, {-40, 40, 20}
, {-56, -14, -22}
, {-25, 19, 25}
, {-37, 52, -32}
, {-5, 57, 52}
, {-27, 26, -11}
, {48, -1, -25}
, {51, -4, 0}
, {42, 34, 36}
, {48, -17, 28}
, {-19, 8, -46}
, {52, -39, 9}
, {37, -8, 29}
, {-13, 4, -6}
, {-4, 12, 40}
, {8, 0, 24}
, {-35, 9, 38}
, {57, 7, 52}
, {-22, 20, -22}
, {-27, 26, 37}
, {28, -27, 9}
, {48, -43, -5}
, {5, 24, -15}
, {3, 34, -15}
, {-34, -2, 41}
, {25, -12, 0}
, {-16, -2, -38}
, {-4, -44, -51}
, {-1, 47, 42}
, {-44, 8, -52}
, {1, -41, 3}
, {39, 20, -41}
, {-12, -27, -35}
, {9, 45, 43}
, {6, -34, 21}
, {-37, -57, 0}
, {-14, 30, -41}
, {-14, 21, -28}
, {-24, 44, -44}
, {42, -37, 2}
, {50, 9, 54}
, {-12, -25, 25}
, {-31, 38, -56}
, {-39, 28, 29}
, {-27, -31, -10}
, {52, -53, -12}
, {-45, 42, -16}
, {-43, -39, 25}
, {-4, -11, -40}
, {-30, 27, 0}
, {18, -43, 5}
, {14, -27, 10}
, {0, -17, 11}
, {39, 0, -65}
}
, {{-18, -11, 21}
, {-45, 4, 27}
, {52, 26, -36}
, {-2, 12, -23}
, {-31, 10, -18}
, {43, 26, 37}
, {-49, 18, -18}
, {21, 22, -24}
, {-8, 7, -36}
, {0, 4, -47}
, {-41, 37, 53}
, {-32, -26, -34}
, {-45, 17, -21}
, {5, -15, -39}
, {-27, -24, -25}
, {-13, -10, 23}
, {-30, 28, -50}
, {11, -19, 3}
, {28, -47, 25}
, {50, 12, 36}
, {-44, -17, -10}
, {-16, 13, -44}
, {35, -13, 21}
, {-43, -10, -37}
, {0, 0, -4}
, {25, 8, 44}
, {-13, 31, -35}
, {39, 35, -21}
, {-15, 9, -25}
, {-18, 28, 3}
, {5, -17, -40}
, {-45, 36, 0}
, {-6, 20, -35}
, {15, 22, -29}
, {16, -42, 43}
, {-6, -12, -17}
, {39, -9, 51}
, {-39, 14, 26}
, {-30, 43, -46}
, {-24, -2, -26}
, {-34, 35, -18}
, {-17, 59, 1}
, {24, -30, -17}
, {33, 37, -15}
, {-20, 46, -41}
, {-9, 13, 53}
, {-1, -50, 34}
, {43, 33, 50}
, {-20, 28, 20}
, {36, -21, -54}
, {1, -4, 1}
, {-20, -33, -32}
, {15, -30, 52}
, {-31, 60, 22}
, {29, -13, -17}
, {19, 31, -32}
, {-12, 26, -42}
, {-15, -24, 43}
, {-3, 7, 50}
, {-26, -9, -1}
, {33, -45, -50}
, {-17, -42, -6}
, {-41, -19, 30}
, {26, 35, -37}
}
, {{35, 67, 61}
, {41, 0, 8}
, {-24, 9, -6}
, {-58, -52, -33}
, {-45, -3, -5}
, {44, 27, 23}
, {-11, 44, -26}
, {6, 1, -40}
, {24, 4, -6}
, {-16, 46, 11}
, {52, 9, 32}
, {39, -23, 47}
, {7, -15, 0}
, {31, -12, -34}
, {47, -23, 35}
, {7, -17, 28}
, {-46, -49, -46}
, {-13, 45, -9}
, {-25, 6, 18}
, {22, 38, 42}
, {-53, -54, 0}
, {2, 2, -23}
, {50, -21, 5}
, {25, 39, 0}
, {-3, 27, 8}
, {-42, -51, -12}
, {45, 3, -19}
, {-26, 33, 31}
, {-30, 42, -46}
, {34, -40, -4}
, {-23, 57, 6}
, {-31, -13, 42}
, {4, -36, -34}
, {-30, -38, -26}
, {35, 25, -48}
, {-23, -30, -16}
, {-40, -37, 9}
, {-21, 34, 50}
, {-8, 35, -18}
, {-33, -24, -31}
, {-3, -9, 0}
, {13, 35, -52}
, {10, -29, 21}
, {-34, -24, 31}
, {-36, -6, 45}
, {20, -1, -32}
, {23, -15, -39}
, {-25, 16, 47}
, {-7, 35, 30}
, {-21, 22, -8}
, {21, -24, 35}
, {48, -20, 48}
, {41, -57, -50}
, {15, 37, -6}
, {25, 18, -10}
, {34, 28, 20}
, {-6, -45, 18}
, {15, -12, 24}
, {-7, -48, -30}
, {14, -28, 65}
, {19, 21, 7}
, {-13, 38, -12}
, {7, -37, 12}
, {-4, -62, 8}
}
, {{49, 14, 4}
, {40, -11, 36}
, {-19, 17, -16}
, {-19, -14, -13}
, {43, 33, 28}
, {-14, -11, 40}
, {-45, -36, 5}
, {-48, -41, 48}
, {-48, -35, -38}
, {-9, 45, 66}
, {50, -26, -18}
, {40, -19, 36}
, {49, -19, 46}
, {38, 27, -8}
, {10, -13, 20}
, {0, -34, -43}
, {-15, -20, -13}
, {45, 47, 28}
, {31, 0, -35}
, {15, -7, 32}
, {-58, -10, -1}
, {5, -14, -35}
, {19, 16, -30}
, {-35, -16, -4}
, {-32, -45, 11}
, {-6, 13, 46}
, {14, 37, -29}
, {11, 37, -12}
, {29, 2, -49}
, {-47, 47, 54}
, {-6, 4, 42}
, {43, -24, 22}
, {50, -23, 55}
, {25, 49, 18}
, {-43, -8, 10}
, {23, -7, 33}
, {-11, 22, 25}
, {-6, 50, 51}
, {-13, -23, 17}
, {-11, -16, -61}
, {28, 43, 7}
, {-31, -54, -22}
, {-35, -2, -41}
, {-7, -37, 33}
, {28, -40, 0}
, {-14, -26, 8}
, {34, 27, -49}
, {47, 21, -14}
, {18, -44, 51}
, {43, 25, 13}
, {17, 2, 14}
, {-19, -33, 44}
, {0, -34, -10}
, {-41, -35, 4}
, {21, -40, 42}
, {29, 38, -17}
, {-40, -4, 26}
, {19, -8, 48}
, {26, -18, 10}
, {1, -47, -26}
, {-15, 25, 19}
, {8, -28, 20}
, {15, -18, -11}
, {27, -41, -53}
}
, {{-4, -52, 2}
, {33, 16, -15}
, {-60, 38, -66}
, {57, 3, 62}
, {-14, -34, 6}
, {5, -45, -41}
, {-21, -11, 19}
, {-24, -2, 25}
, {49, -35, 32}
, {-52, -52, 25}
, {50, 8, 17}
, {-56, -27, 27}
, {46, -14, 53}
, {31, -33, -31}
, {-46, -41, 12}
, {3, -11, -32}
, {-38, 12, 8}
, {32, 10, 17}
, {0, -47, -18}
, {-32, 49, -46}
, {-6, -39, 14}
, {18, 15, 32}
, {-8, 46, -33}
, {-16, -14, 21}
, {24, -31, -50}
, {-51, -24, -46}
, {-34, 32, 27}
, {-34, 31, -15}
, {9, 0, 29}
, {28, -3, 40}
, {12, 9, 15}
, {30, 49, -30}
, {28, 21, -10}
, {-7, 28, 35}
, {-57, 0, -38}
, {-34, -11, 46}
, {-13, 47, 49}
, {-2, 6, 48}
, {48, -16, -23}
, {43, -32, -31}
, {-43, 46, 17}
, {54, -35, 42}
, {-24, 33, 36}
, {-38, -20, 24}
, {-23, 49, -45}
, {56, 6, 48}
, {-35, -12, 24}
, {-49, -28, 16}
, {-32, 39, -26}
, {-11, 6, 6}
, {-32, -33, 25}
, {-50, 0, -17}
, {10, 22, -52}
, {46, 23, -18}
, {8, -13, -7}
, {2, 4, 32}
, {-42, -38, 29}
, {5, -30, 3}
, {-35, 49, -35}
, {-25, -37, -17}
, {42, -5, -24}
, {-23, -41, -38}
, {0, 32, 38}
, {42, -14, 3}
}
, {{-34, 20, -4}
, {-31, -33, 23}
, {18, -33, 2}
, {-42, -58, -48}
, {-29, 32, -26}
, {-35, -41, -49}
, {17, 7, 47}
, {12, 46, -1}
, {11, -25, 8}
, {-6, 48, 15}
, {44, -25, -40}
, {-17, -18, -25}
, {-16, -17, -4}
, {23, -3, -35}
, {16, -39, 21}
, {32, -1, 12}
, {0, 24, 16}
, {22, -10, -43}
, {41, 6, 28}
, {27, 18, 2}
, {23, -31, 10}
, {9, 40, 42}
, {-31, 13, 9}
, {-17, 11, -15}
, {-36, 5, 54}
, {29, 10, -17}
, {49, -19, 51}
, {-19, -39, 10}
, {-43, -19, -29}
, {13, -37, -36}
, {-51, 28, 3}
, {-26, -20, -38}
, {-18, 0, -36}
, {-28, 4, -42}
, {1, -54, -46}
, {-6, 11, 2}
, {-16, 6, -46}
, {39, -9, 0}
, {-8, 16, 13}
, {-38, -22, 18}
, {12, 36, -5}
, {-42, -50, 14}
, {-22, 23, 1}
, {26, -22, 43}
, {-36, -31, -1}
, {-15, -18, 8}
, {-46, -9, 8}
, {-1, 5, -21}
, {37, -14, 49}
, {29, 26, 43}
, {35, 21, 31}
, {31, 32, -24}
, {15, -4, 14}
, {12, -34, -33}
, {37, 5, 17}
, {6, -29, 11}
, {3, 20, -46}
, {3, -8, 14}
, {-31, 21, 12}
, {-48, -27, 22}
, {41, -27, 46}
, {40, -18, -8}
, {-40, 43, 16}
, {-3, 3, 32}
}
, {{-44, -16, -11}
, {-2, -18, -49}
, {-22, 29, -51}
, {2, -24, 1}
, {33, -26, -4}
, {-34, 29, -31}
, {-27, -44, 0}
, {7, 26, -30}
, {-19, -44, 37}
, {-19, 8, -68}
, {30, 44, 30}
, {3, 19, -9}
, {32, -7, 31}
, {11, -12, -10}
, {18, 5, -7}
, {39, -25, -20}
, {41, 18, 36}
, {-23, -13, -25}
, {-28, -40, 16}
, {21, 8, 47}
, {-45, -6, 16}
, {-49, -29, 36}
, {-2, -14, 11}
, {-45, -50, 33}
, {9, -17, -5}
, {5, 33, -17}
, {-40, 45, 9}
, {-50, -22, -6}
, {32, 42, 26}
, {-45, -9, 23}
, {1, 2, -20}
, {-35, 19, 20}
, {-41, -39, -11}
, {18, -28, 13}
, {-18, 37, -25}
, {50, 2, 49}
, {-5, 35, -16}
, {48, 14, -2}
, {24, -11, 41}
, {7, 25, 12}
, {31, -10, 20}
, {-27, -15, -19}
, {-52, 27, 29}
, {-44, -6, 11}
, {-27, 26, -27}
, {-29, 29, 13}
, {33, -18, -47}
, {-32, -1, 19}
, {-46, -23, 44}
, {35, -36, 6}
, {-14, -14, -1}
, {28, -44, -34}
, {46, -41, 39}
, {16, 10, -11}
, {-12, 23, 31}
, {4, -33, 0}
, {50, 14, -38}
, {-1, 30, 15}
, {-1, -30, -39}
, {-28, -8, -57}
, {-47, -44, -30}
, {-25, -31, 37}
, {-43, 2, -12}
, {40, -24, -48}
}
, {{15, 34, -43}
, {25, 29, 17}
, {-49, 19, -25}
, {15, 2, -41}
, {-53, -7, -50}
, {-13, 6, 19}
, {32, 4, 16}
, {-23, -49, -21}
, {0, -34, 39}
, {-62, 6, -13}
, {-15, 27, 29}
, {26, -15, -34}
, {59, -2, -35}
, {-36, 54, 52}
, {50, 52, 14}
, {-48, 37, -21}
, {4, -41, 1}
, {30, 13, 0}
, {-31, 49, -37}
, {-15, -9, 19}
, {-57, -43, -13}
, {-11, -26, -29}
, {-3, -41, -19}
, {1, 0, -36}
, {6, 32, -26}
, {-12, 34, 28}
, {46, 15, 48}
, {63, -26, 49}
, {1, -41, -30}
, {-18, -19, 30}
, {-62, -13, -50}
, {-8, -11, 38}
, {-36, -49, -7}
, {11, -26, 7}
, {-57, -7, -39}
, {-44, 40, 14}
, {14, -5, 45}
, {7, 27, 0}
, {51, -25, 49}
, {-60, -48, -32}
, {41, 1, -12}
, {-20, 12, 12}
, {24, 52, 15}
, {-33, 44, -42}
, {-27, -4, 44}
, {12, -23, -9}
, {12, -51, 1}
, {-6, 26, -18}
, {-18, 25, 41}
, {41, -23, -28}
, {52, 51, 51}
, {-24, 20, 1}
, {-21, 8, 27}
, {-2, -7, 24}
, {-37, 46, 0}
, {-44, 12, -58}
, {-22, -54, 11}
, {-4, -15, 12}
, {-4, 19, 19}
, {-56, 13, -6}
, {-8, 44, -30}
, {40, -39, 15}
, {15, 23, -56}
, {24, -11, -68}
}
, {{61, 18, 46}
, {10, -46, -1}
, {-29, 43, -11}
, {-49, -52, -25}
, {47, 46, 67}
, {35, 13, 1}
, {-15, -32, -48}
, {-33, 29, 24}
, {-44, -47, 4}
, {67, 17, 38}
, {12, -2, 47}
, {-22, 34, -46}
, {4, -21, -24}
, {-13, 17, -32}
, {39, 54, -19}
, {10, -44, 30}
, {26, 36, 34}
, {32, -3, 9}
, {16, 25, -8}
, {-52, -30, 44}
, {-2, -10, -19}
, {-4, 13, -14}
, {-31, 40, -15}
, {7, 33, 1}
, {-18, 40, 41}
, {-15, -48, 15}
, {-9, 36, -13}
, {-8, 45, 4}
, {-25, -37, -24}
, {32, -29, -27}
, {7, -19, 2}
, {30, -1, 3}
, {-4, 8, -30}
, {-36, -5, -30}
, {-23, 23, 4}
, {4, 35, 39}
, {2, 49, 45}
, {3, 51, 0}
, {-44, -39, 28}
, {10, 27, 33}
, {-44, 27, -5}
, {-34, -33, 35}
, {6, -16, 0}
, {-47, 35, 38}
, {-10, -19, 25}
, {25, 14, 12}
, {-39, 42, 10}
, {16, 0, 34}
, {-13, 1, 31}
, {16, 39, 41}
, {38, -5, 1}
, {4, 39, -5}
, {-8, -8, 36}
, {27, -37, -66}
, {35, 37, 31}
, {10, 6, -32}
, {-10, -4, -17}
, {-42, -18, -34}
, {-36, 37, 33}
, {-13, 48, -9}
, {9, -6, 31}
, {53, -13, 20}
, {-23, 11, -12}
, {2, -19, -49}
}
, {{4, 11, -30}
, {-24, 46, -11}
, {25, -22, 46}
, {-10, -53, 2}
, {50, 1, 23}
, {-26, 48, 46}
, {13, -15, 18}
, {-24, -21, 36}
, {-2, 48, -2}
, {-9, 41, 5}
, {28, -11, -62}
, {56, -15, -20}
, {36, 28, 38}
, {-35, 16, 35}
, {32, 31, -28}
, {-47, -49, 7}
, {-31, 5, 39}
, {49, -14, 36}
, {-27, 2, -13}
, {50, -24, -23}
, {8, 47, 4}
, {4, -46, 8}
, {-40, 0, -4}
, {12, 36, -2}
, {38, -15, 38}
, {-2, -44, -28}
, {16, -50, 51}
, {36, 32, -35}
, {50, -34, 18}
, {49, -9, -33}
, {-22, -22, 15}
, {45, -22, 50}
, {-48, -5, 23}
, {43, -13, 31}
, {-25, 1, -1}
, {20, 49, -17}
, {44, -49, -42}
, {16, -31, -41}
, {33, -20, 43}
, {-44, -18, -24}
, {-31, -38, 30}
, {21, 13, -12}
, {-3, -30, -6}
, {14, 52, -9}
, {14, 0, 28}
, {-38, -43, 33}
, {-53, 31, -16}
, {-19, -20, 6}
, {52, 22, -42}
, {-16, 25, -6}
, {-19, -12, 17}
, {4, -28, 26}
, {-48, 22, 0}
, {-18, -28, 31}
, {21, -7, -21}
, {-43, 9, 24}
, {-20, 47, 37}
, {-30, 38, -49}
, {-38, -51, -48}
, {28, 21, 7}
, {-28, -19, -50}
, {-1, 53, 1}
, {-37, -16, 19}
, {3, -5, -41}
}
, {{53, 38, -15}
, {12, -10, -41}
, {-35, 0, 28}
, {4, 30, 20}
, {16, 9, -39}
, {-35, 46, -10}
, {-4, 39, -41}
, {-19, -33, 47}
, {43, 42, 37}
, {22, -22, 8}
, {18, 57, -4}
, {-28, 11, 18}
, {-46, -19, -39}
, {-6, 36, -13}
, {-38, 29, 52}
, {18, 1, -9}
, {39, -45, 30}
, {37, 12, -19}
, {33, 50, -46}
, {-19, -49, 33}
, {28, -8, 11}
, {-26, 31, -25}
, {-12, -22, 3}
, {-10, 30, 56}
, {-14, -5, 46}
, {-28, 8, -25}
, {23, 22, -12}
, {-40, 12, 17}
, {-7, -20, 3}
, {-24, 42, 19}
, {-20, 22, 15}
, {-43, 47, 32}
, {32, -37, 15}
, {31, -31, -11}
, {48, -49, 0}
, {-21, -4, 29}
, {-9, -50, -42}
, {-39, 41, -17}
, {-19, -35, -52}
, {-36, 25, -48}
, {36, 42, 15}
, {8, -12, -14}
, {35, 43, -52}
, {41, -22, 1}
, {-26, -8, 29}
, {-52, -8, -35}
, {16, -23, -19}
, {8, 32, 10}
, {-4, -29, 44}
, {31, -4, -5}
, {30, -37, 41}
, {-13, 39, 6}
, {38, 13, -16}
, {30, -26, -5}
, {-24, -2, 52}
, {28, -40, -28}
, {-37, 41, -29}
, {-31, 1, 13}
, {29, -14, -27}
, {29, -7, -33}
, {32, 28, 4}
, {-29, -34, 48}
, {13, -10, 64}
, {-4, 25, -51}
}
, {{38, 23, 23}
, {33, -15, -13}
, {24, 3, 26}
, {-26, 34, -4}
, {-4, 24, -37}
, {24, -47, -49}
, {-38, -55, -52}
, {-11, -44, 15}
, {-34, 18, 7}
, {16, -40, 30}
, {17, -52, -57}
, {19, -48, -55}
, {0, -22, -20}
, {47, -11, 48}
, {45, -56, -47}
, {-48, 33, 2}
, {-48, 26, 9}
, {27, -54, -49}
, {5, 6, 43}
, {-43, -4, -16}
, {6, 8, -28}
, {13, 2, -23}
, {40, 46, 31}
, {5, 24, -55}
, {1, -22, -11}
, {34, -19, 25}
, {-24, 43, -42}
, {35, 26, 26}
, {-18, -53, -27}
, {5, 19, -8}
, {-41, 20, -1}
, {-39, -6, 4}
, {30, -20, -22}
, {-42, -25, -18}
, {49, 31, -16}
, {-18, -47, -38}
, {-35, 4, -8}
, {-42, -23, 38}
, {22, 26, 43}
, {37, -2, 30}
, {-40, 38, 39}
, {7, -37, 28}
, {-1, 18, 31}
, {-14, 30, -25}
, {31, 10, 11}
, {-43, -27, 10}
, {0, -6, -50}
, {-32, -49, 11}
, {-31, -17, -6}
, {-51, 34, -18}
, {-1, 24, 23}
, {45, -4, 14}
, {-28, -21, 37}
, {-27, -42, 39}
, {-38, 46, 36}
, {-7, -41, 27}
, {-40, -9, 2}
, {43, -51, 29}
, {0, -10, -21}
, {0, 16, -55}
, {45, 17, 6}
, {-3, 32, -4}
, {16, 43, 26}
, {34, 6, 29}
}
, {{32, 7, -25}
, {-22, 19, 58}
, {-32, 19, -16}
, {-35, 27, 44}
, {33, 33, 31}
, {-14, -24, -34}
, {-27, -8, -2}
, {28, -32, 0}
, {51, 4, 15}
, {-23, 33, 36}
, {2, 11, -25}
, {-15, 30, 40}
, {15, 20, 13}
, {-40, 49, 26}
, {67, 20, 30}
, {-18, 29, -42}
, {-37, 47, 25}
, {-6, -40, -27}
, {59, -39, 6}
, {33, 44, -42}
, {-41, -62, 42}
, {30, -28, 24}
, {-13, -16, -30}
, {2, -18, -12}
, {27, -26, 0}
, {23, 27, 46}
, {38, -28, -37}
, {37, 47, 18}
, {32, -37, 14}
, {-46, 55, 4}
, {-25, 27, -21}
, {3, -25, 29}
, {-33, -35, 24}
, {21, -5, 45}
, {0, 50, 10}
, {29, -5, -21}
, {25, 33, -45}
, {-18, -45, -20}
, {-15, -29, -40}
, {15, 55, 22}
, {1, -39, 32}
, {6, -25, -57}
, {30, -1, -40}
, {-44, 47, 36}
, {23, 45, 13}
, {-17, -42, 17}
, {36, -29, -24}
, {-6, 6, -39}
, {20, -33, 14}
, {61, 20, 16}
, {52, 43, -30}
, {2, 1, 42}
, {32, -4, 9}
, {30, -27, 19}
, {0, 19, -10}
, {5, 39, 12}
, {-30, 10, -13}
, {-5, 42, -30}
, {-20, -28, -24}
, {53, 5, -10}
, {-7, 57, -12}
, {-14, -21, 44}
, {-31, 19, 4}
, {-58, -11, -44}
}
, {{12, -56, 11}
, {-51, -8, 37}
, {-39, -9, 18}
, {-33, 10, 6}
, {0, -45, 41}
, {-5, -14, 40}
, {-14, 41, -25}
, {-12, 44, 14}
, {40, 11, -17}
, {-48, -33, -21}
, {47, -34, 55}
, {12, -15, -25}
, {42, 30, -38}
, {-24, 22, -9}
, {-33, 47, -13}
, {0, -47, -40}
, {32, -46, -35}
, {45, 25, 25}
, {19, 23, -46}
, {-51, 22, 0}
, {15, 38, -4}
, {-14, 47, -21}
, {-55, -29, -44}
, {40, 30, 31}
, {-22, 14, 18}
, {-30, 0, 50}
, {-50, -32, -47}
, {24, -19, -12}
, {50, 19, -21}
, {-22, 46, -14}
, {-42, 8, 0}
, {-7, 18, 0}
, {-2, 42, 17}
, {0, 24, 31}
, {-16, 21, 46}
, {-48, 37, 10}
, {-33, 30, 17}
, {23, -10, 35}
, {2, 24, 51}
, {-11, 37, -32}
, {34, -15, -7}
, {37, 5, 53}
, {-48, 41, -11}
, {44, -5, 21}
, {1, 4, 2}
, {-39, -26, 63}
, {-41, -22, -9}
, {43, 9, 34}
, {49, -9, -9}
, {-26, 25, -42}
, {-26, -35, -45}
, {-46, -45, 38}
, {-33, -11, -21}
, {-23, 51, -19}
, {-41, 32, 26}
, {-32, 22, -1}
, {-40, 38, 21}
, {42, -44, 48}
, {-32, -50, 15}
, {-41, 23, -20}
, {-26, 0, -8}
, {36, 19, 29}
, {-36, 13, 18}
, {-40, 21, 47}
}
, {{-37, -35, 12}
, {15, 45, -29}
, {-36, 24, -17}
, {34, 50, 23}
, {32, -34, -23}
, {-12, 28, -16}
, {-19, 4, 33}
, {-27, -9, -9}
, {46, 25, -39}
, {-17, -18, -38}
, {-41, -50, 41}
, {39, -17, 21}
, {35, -3, -52}
, {-51, 5, 19}
, {-49, -19, 29}
, {-39, 38, -41}
, {-32, -26, -18}
, {21, -29, -40}
, {12, -5, 21}
, {41, -29, -25}
, {-2, -16, 15}
, {37, -41, 0}
, {37, 14, -32}
, {-15, -52, -3}
, {25, -5, -8}
, {38, 17, -42}
, {46, 48, 37}
, {-21, 26, 9}
, {-3, -27, -11}
, {-20, -39, 3}
, {1, -3, 30}
, {-27, -30, 26}
, {6, -45, 1}
, {-50, -9, -24}
, {-41, -7, 0}
, {25, 15, 23}
, {22, 28, -3}
, {-36, 18, -1}
, {0, 32, -22}
, {15, 20, 9}
, {4, 0, -1}
, {52, 29, 44}
, {28, -4, 33}
, {32, -23, 8}
, {23, -2, 18}
, {58, 13, 42}
, {-29, -30, 47}
, {23, 22, -10}
, {41, 38, 12}
, {-9, 23, -51}
, {-7, 0, -12}
, {-35, 48, 31}
, {25, -17, 7}
, {24, 48, 54}
, {-39, 14, -15}
, {7, -15, -57}
, {-54, -18, 46}
, {44, 52, 1}
, {-36, -16, 48}
, {14, 23, -7}
, {-47, -39, 36}
, {29, -55, -42}
, {31, -13, -56}
, {62, 62, 3}
}
, {{0, -19, -33}
, {-8, 0, 51}
, {-16, -69, 18}
, {-33, 1, 13}
, {37, -30, -10}
, {-33, 2, -33}
, {-31, -23, 12}
, {39, 3, -33}
, {45, 28, -39}
, {11, -1, -9}
, {-19, 7, -24}
, {0, 23, -14}
, {34, -36, -43}
, {-5, 19, 55}
, {-19, 50, -38}
, {14, -53, -38}
, {-29, 45, 16}
, {23, 42, 5}
, {26, 46, 26}
, {-10, -25, -43}
, {56, 46, -21}
, {39, -22, -13}
, {16, -34, 35}
, {30, -62, -45}
, {6, -14, -6}
, {15, 3, 40}
, {-33, -27, -21}
, {32, 44, -12}
, {38, 36, 30}
, {19, 33, -22}
, {-38, 14, -41}
, {-18, 34, -8}
, {-5, 0, 5}
, {21, -32, -40}
, {-48, -38, 31}
, {-16, 7, 51}
, {13, 1, -1}
, {12, 2, -8}
, {-43, -53, 4}
, {-19, -64, -29}
, {10, 38, 41}
, {-2, 16, 14}
, {-15, 39, -14}
, {-45, 10, 8}
, {45, 49, -6}
, {-21, 28, 27}
, {-6, 33, -33}
, {-12, -31, -36}
, {33, 42, -47}
, {4, 34, -2}
, {3, 39, 56}
, {-21, -17, -53}
, {-29, 11, -45}
, {39, 27, 12}
, {39, 19, -25}
, {19, -60, -56}
, {-34, -16, -22}
, {-53, -34, 56}
, {-38, -43, -49}
, {-63, 25, -42}
, {15, 35, -15}
, {7, 33, -38}
, {5, -37, 39}
, {19, 41, 35}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  128
#define INPUT_SAMPLES   22
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_9_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_9(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    averagepool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  128
#define INPUT_SAMPLES   11
#define POOL_SIZE       11
#define POOL_STRIDE     11
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t average_pooling1d_1_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d_1(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned short x;
  long_number_t avg, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
      tmp = 0;
      for (x = 0; x < POOL_SIZE; x++) {
        tmp += input[k][(pos_x*POOL_STRIDE)+x];
      }
#ifdef ACTIVATION_RELU
      if (tmp < 0) {
        tmp = 0;
      }
#endif
      avg = tmp / POOL_SIZE;
      output[k][pos_x] = clamp_to_number_t(avg);
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_DIM [1][128]
#define OUTPUT_DIM 128

//typedef number_t *flatten_1_output_type;
typedef number_t flatten_1_output_type[OUTPUT_DIM];

#define flatten_1 //noop (IN, OUT)  OUT = (number_t*)IN

#undef INPUT_DIM
#undef OUTPUT_DIM

/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES 128
#define FC_UNITS 40
#define ACTIVATION_LINEAR

typedef number_t dense_2_output_type[FC_UNITS];

static inline void dense_2(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]) {			                // OUT

  unsigned short k, z; 
  long_number_t output_acc; 

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0; 
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ( kernel[k][z] * input[z] ); 

    output_acc = scale_number_t(output_acc);

    output_acc = output_acc + bias[k]; 


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = clamp_to_number_t(output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0)
      output[k] = 0;
    else
      output[k] = clamp_to_number_t(output_acc);
#endif
  }
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 128
#define FC_UNITS 40


const int16_t dense_2_bias[FC_UNITS] = {-1, 0, 10, -12, 11, 11, 0, 8, -1, -5, 0, 5, -4, 1, 13, 0, 9, -2, -12, 13, 7, 7, 0, 9, 0, 4, 1, 0, -7, 8, 2, -13, -4, -3, 11, -12, 2, 8, -10, 4}
;

const int16_t dense_2_kernel[FC_UNITS][INPUT_SAMPLES] = {{6, 34, -84, 6, 17, -69, -89, 70, 23, -2, -38, -79, 64, 11, 35, 44, -12, 26, -49, -6, -76, 20, -71, -35, 45, -77, 24, -2, 77, -32, -84, -16, 41, -18, -53, -86, 13, -32, -20, -76, 47, 39, 33, 44, -59, -41, -40, 96, -81, 74, 22, 63, 18, 79, 20, -52, 80, 4, 10, -75, 31, -81, 34, 12, -22, -35, -20, -21, -93, -12, 3, 75, 64, 87, -2, 38, -30, 0, -36, 2, -47, -48, -50, 71, -24, 21, -2, 72, 41, -38, 57, -43, -90, -3, -34, 73, 81, 2, 20, -101, 15, 46, -51, 42, -64, 15, -29, 44, -62, -11, 3, -15, 56, 23, -8, 85, -8, 33, -58, -27, 23, -2, 1, -78, -74, -62, 85, 25}
, {-73, 53, -80, 33, 40, 71, 58, -36, -67, -25, -1, -72, 92, 70, 62, 34, 60, 36, -33, -6, -57, -28, -42, -64, 16, 17, 92, -71, -81, -23, 76, -76, -21, -68, -42, -96, -79, 46, 36, 81, 77, 94, 83, -87, 68, -76, 56, 69, 53, -61, 94, -79, -76, -70, -19, -45, 50, 75, 53, 48, -9, 46, -65, 33, 41, 6, 58, -13, 17, -62, -60, 65, 63, -33, -64, -39, 90, -35, 4, 15, -33, -11, 85, -79, -15, -10, -13, 5, -84, 27, 57, -49, -48, -58, -86, -90, 79, -74, -52, -49, 111, -94, 10, 83, 90, -77, -41, -93, 85, 0, -47, -58, -11, -13, -17, 53, -17, -54, 31, -70, -51, 2, 90, -79, 116, 22, 58, 95}
, {41, 103, 38, -15, 110, 97, 79, 47, 8, -56, -60, 5, 59, 66, 13, -12, 48, -64, 34, 16, 55, 4, 52, -18, 31, -69, -90, 86, -15, 91, 76, 16, 65, -59, -32, 74, -39, 0, -17, -84, -62, -92, -90, 48, 10, -38, -64, -89, -74, -3, -60, -66, -77, -61, -66, -89, -13, 19, 9, 46, -67, 92, 44, 53, 42, 92, 47, -9, -88, 1, 75, -52, -96, -34, -73, 84, 90, 66, 67, -90, -77, 6, -37, 89, 73, -35, 55, -10, 0, -16, -97, 19, 76, 82, 0, -34, 70, -79, -82, -11, 42, -57, 93, -73, 38, 36, -24, 108, -59, 76, 89, -69, -53, 83, 65, 44, -47, -80, 32, -66, 4, -70, 67, -34, 85, -68, 6, -84}
, {-85, 51, -91, 46, 13, -74, -56, -17, -21, -68, 76, -84, -58, -19, 77, 70, -7, 28, 70, -64, -72, -21, 71, -45, 46, -60, -67, -24, -33, -44, -57, -25, 44, -52, 58, -36, -13, 21, 97, 56, 78, 90, -50, -63, -47, -25, -101, 98, -75, 32, -19, -91, -73, 73, -108, -98, -16, -50, 35, 70, -24, 0, -23, -59, 77, -78, -3, -34, 52, 57, -109, 44, 68, -73, 99, -36, -71, 48, 62, -62, -43, -18, 4, 62, 12, 62, 1, -90, -46, 7, 85, 88, 35, -47, -35, 70, -57, -6, 34, 25, 36, 42, 23, -21, 43, 47, -40, 26, -1, 12, -23, -10, 19, 32, -20, 51, 54, 88, -60, 99, 31, -71, 46, -32, 59, -5, 81, 80}
, {-97, 1, 42, 36, 46, -44, 10, 38, -82, 35, -101, -2, -76, -24, 103, -94, 38, 82, 0, -21, 61, 0, 70, -3, 96, 43, 36, 15, 58, 31, -64, 47, -80, 17, 42, -54, -31, 71, -24, 43, 30, -91, -79, 48, 65, -79, 63, -51, -37, 71, 11, 38, 64, -13, -69, -64, -34, -54, 16, 19, 17, 75, -28, -39, 74, -65, -63, 42, 84, -36, 59, -76, 0, 5, -20, 32, 62, -67, 72, -14, -81, -17, -18, 37, -35, -3, -76, -48, -40, -54, -32, 34, 93, -63, -89, 63, 57, 53, 54, 20, 76, 81, -42, 66, 0, 2, -30, 19, -40, -25, 83, 22, -43, 54, -90, 16, 31, -33, -30, 47, 36, -9, 55, -50, 19, 92, 87, -49}
, {10, -49, -10, 36, 30, 3, 41, 73, 52, 76, -9, 78, 1, 45, 59, -77, 6, 60, 0, 96, 86, -84, 94, -75, 40, 50, -47, 71, 11, 80, -43, 90, 64, 26, -9, -2, 70, 44, -8, -70, -59, -25, -96, 21, -40, 95, -70, 29, 101, 77, -39, 0, 19, -56, 19, 17, -50, -11, -35, -49, 87, -29, 18, 76, 6, 47, 78, -45, -25, 86, 48, -48, 96, 11, -52, -60, 15, 100, 38, 32, -63, 98, 5, 75, 28, -76, -25, 67, 69, 14, 24, 2, -32, -103, -19, -36, -69, -25, 35, -33, 21, 54, -70, 9, -70, 24, -15, -34, 93, 18, 94, 1, -67, -39, -47, -75, 80, -47, -45, 10, 19, 21, 69, 70, 73, -20, -27, -70}
, {-84, 27, 49, -5, -17, -71, -8, 15, 84, 87, -68, -21, 14, 59, -67, -51, -69, 38, 64, -91, -3, 83, -18, -43, 8, 11, -60, -75, -30, 61, 53, 82, 20, -23, -80, -50, 80, -10, -3, -11, 88, -37, 14, 89, -79, -14, 21, 15, 83, 11, 59, -22, -51, -81, -49, 66, -63, -76, 93, 43, 58, -64, -3, 38, -85, -31, 45, 20, 10, -38, 30, 28, -26, -36, -89, -70, 78, 72, 87, -43, 43, -85, 20, -30, -57, 15, 33, 3, 94, -53, -86, 41, -34, -76, 49, 56, 27, 18, -1, -62, 0, -90, -37, -49, -56, -45, 83, -10, -77, -85, -22, -56, -42, 46, 94, -83, -91, 39, 0, 22, 12, -62, -51, -72, -5, -57, 15, 91}
, {63, 36, 44, -8, 23, -20, -35, -35, 16, -61, -40, 28, 14, 44, 26, -15, 5, 59, -63, -47, -44, 5, 18, -87, -63, 56, 11, -94, -72, 73, 113, 5, -10, 33, -35, 23, 91, -65, 33, -6, 30, -66, -39, -39, 22, -49, 26, 95, 81, 41, -44, 39, 27, -19, 18, -81, -85, -39, 81, 30, 68, 24, -60, -11, -87, -54, -62, -53, 85, 62, 101, -38, 73, -76, -61, 39, 81, 100, -23, 84, -80, -33, 12, -4, -2, 0, 16, -63, -90, 22, -85, 23, 51, -12, 47, 90, -12, -46, 46, 53, 11, 14, 12, -92, 24, -8, 0, 19, -4, -14, -88, 87, 93, 65, -8, -78, -27, -69, 44, -47, -15, 48, -62, 72, -67, 4, -50, 25}
, {103, -41, 46, 48, -66, -65, 31, 80, 42, -52, -40, 85, -73, -63, -33, -6, 20, 62, 94, -25, 35, -55, -65, -47, -49, 50, 34, 37, 47, 90, 43, 47, -11, 86, -53, -33, 52, 77, -20, -25, -21, -50, -82, -30, -17, 16, 14, 57, 45, -54, 5, 0, 46, -58, 28, -68, 56, 37, 8, -49, 72, 5, 64, -21, 10, -99, -43, 30, -38, 83, 70, 78, -3, 23, -30, -11, -73, 72, -5, -9, 92, 67, -76, -24, -30, -43, -35, -25, 20, 59, -65, 28, 89, -31, 1, 8, 71, -77, 54, -26, 93, 80, -49, 100, -6, -74, -89, 69, 95, 62, -47, -26, -54, 69, -83, 77, 79, 9, -59, 33, 95, 55, -26, 34, -5, -60, -96, -88}
, {68, -95, 25, -81, -82, 88, 26, 30, 48, -25, -26, 68, -46, 8, -12, 29, -15, -84, -90, -52, 90, 53, -41, -68, -93, -90, -1, 2, -23, -19, 20, -41, -103, -65, -76, -17, -83, -65, -28, -12, 76, 63, -88, -72, 64, -14, -12, 2, 64, 0, -42, 0, 31, 65, 94, 41, 88, 40, -42, -20, -89, 48, -88, 5, 79, -51, -1, -64, -98, -71, -2, -4, -54, 51, -47, -27, -92, -65, 87, 36, -68, -9, 15, -10, 19, -10, -20, 41, -88, 53, 25, -30, -96, 83, -7, -78, 83, 62, -49, -22, -88, -6, 74, 29, 36, 14, 42, -63, -15, 44, -12, -96, -86, 56, -81, -54, 56, 56, 4, -28, -31, -21, 10, 9, 51, -83, 9, 77}
, {-10, -51, -59, 41, 99, -50, -18, 70, -52, 37, 95, 84, -24, -33, 81, 28, 25, -40, -70, 33, 48, -83, -49, -80, 9, -12, 11, -24, -7, 57, -29, -42, -103, 65, -64, -30, 94, -87, 20, 8, 46, -78, -34, 1, 42, -5, 60, 97, 3, 78, 6, -11, -66, 29, 74, -53, -62, 18, 72, -48, 39, -22, 19, 45, -91, -78, -51, -67, 49, 76, 58, -94, -56, 63, -49, -52, -20, 14, 85, 0, -26, 2, 70, -67, -66, -17, 47, 81, 84, -2, -91, 84, 13, 88, 6, 45, -29, -50, 71, 84, 28, 4, -45, 78, 77, 15, -22, 95, -88, 37, 34, -41, -47, -41, 93, 3, -11, 80, 55, 12, 35, 80, -9, 46, -68, 55, -25, 62}
, {97, -87, -54, 95, -4, 4, -67, 58, 7, -82, 58, -37, 33, 2, -7, -18, -90, -12, -44, -25, 88, -82, -64, -69, 17, -24, -46, 41, -69, -60, 96, -70, 43, 74, -75, 34, 56, 71, -29, 49, -63, 24, -89, -81, 25, 78, 75, 31, -83, -86, -72, 32, -10, -41, -31, -34, -45, -43, 72, 91, -73, 13, 80, -68, 66, 0, -66, 63, 29, 98, 8, 21, -66, -40, 91, -85, 72, 93, -103, -73, -7, -53, 76, -6, -27, 62, -15, 20, -18, 9, -73, 37, 15, -8, -11, -57, -80, -91, 84, 21, -39, 25, 77, -33, 99, -37, 90, 10, -4, -85, 63, -55, 70, -57, -66, 55, -34, 74, 90, -32, 69, 23, 85, -52, 0, -20, -88, -60}
, {70, -57, -47, 15, -30, -63, -32, 10, 53, 21, -6, 23, -22, -33, -50, -73, -95, 27, -28, -92, 15, -58, -14, -50, 55, -66, -38, -4, 43, 89, -48, -12, -58, -23, 19, -93, -53, 47, -62, 72, 77, 82, 102, 13, 79, 11, -35, -36, 37, -71, -53, -34, 88, -78, -57, 73, -34, -84, 5, -7, -50, 96, 32, 67, -20, 85, 37, 10, 29, 40, -87, 8, 54, -37, 79, -52, -61, 51, 44, 8, -37, -89, 5, 12, 18, -54, 74, -95, -85, -59, -49, 74, -91, 93, -21, 21, 70, 68, 20, -60, -68, -13, -7, 45, -93, -86, -72, 49, 63, -75, 63, 35, 94, -42, -7, -78, -16, 89, -14, -66, -28, 12, -32, 20, -104, -40, 0, -96}
, {-13, 83, -61, -50, -8, -36, -10, -9, -72, 87, 35, 76, 34, 16, -10, -82, 58, -37, 86, 21, 40, 6, 22, -8, -25, 71, 67, 50, -31, -69, -59, -108, 13, -91, 74, -82, -91, -56, -82, -27, -88, -90, -29, 21, -88, 106, 65, 54, 64, -87, 86, 93, 31, 89, 28, -72, -6, -19, -12, -46, 36, -102, -11, 26, 56, 2, -66, -91, 51, 5, 18, -25, 14, -19, 83, -99, 59, -65, 76, -46, -26, 38, -15, 89, 80, -44, 54, 7, 19, -18, -24, -39, -35, -37, 18, 4, 97, 23, 1, 43, 22, -43, -51, 27, 63, 89, -85, 6, 92, 32, -58, -17, 23, 10, 85, 96, -97, 81, -8, 75, 75, -63, -26, -19, -19, 38, -76, -30}
, {-37, 27, -1, 8, -66, -13, -49, 83, -10, -75, 2, 107, 85, 63, 21, -13, 58, 11, 68, 33, 90, 81, -25, -36, -28, 26, 6, -28, -51, -70, 9, -43, -35, 57, -19, -4, 94, -6, -27, 36, 7, -2, -80, 76, 11, -28, -39, -49, 45, 92, -78, -5, -28, 21, 69, -10, -66, -1, -24, -36, 56, -53, 65, 102, -46, -67, -75, 100, 21, 52, 74, -70, -44, -81, -25, 35, -3, -83, 72, 98, -83, -23, 54, 94, 41, -77, -12, -89, -68, 27, 5, 36, 34, 53, 21, -70, 10, 14, -69, -76, -50, 78, 42, -66, 83, 38, 41, 90, 27, -91, 13, -78, -64, -88, 78, -19, -5, 70, -36, -28, 46, -46, 7, -90, 8, 29, 18, 64}
, {-46, 11, -46, -75, 87, -34, -73, -10, 85, 83, 45, -88, 13, -34, -31, -41, 5, 29, -43, 61, -79, -92, -72, 88, 87, 50, 22, 51, -68, -91, 66, -68, -84, -8, 6, -57, 78, 17, -82, -61, 61, -30, -57, 46, 4, 29, 14, -12, 22, -79, -40, -35, -5, 67, 10, 16, 83, -37, -81, 29, 7, -63, -35, 74, -35, -93, -68, -9, -68, -64, 70, 25, 13, 72, -60, -22, -4, -49, -104, -77, 42, 64, -94, -21, 4, -31, -70, 82, -25, 4, -39, -41, -54, -74, -63, 14, -75, 79, -71, -76, 57, -55, 33, -37, 16, -88, -52, -51, 17, -5, 79, -17, 46, -49, 33, 6, 46, 83, -34, -35, -49, -22, -7, -88, 94, 63, -72, 2}
, {-69, 60, -78, 103, 86, -82, -17, -86, 9, -1, -4, 9, 52, 16, 0, -99, 28, 0, -30, -53, -23, 13, -88, -85, -86, 1, -79, -69, -76, 60, 35, 58, 6, 3, 89, 63, -53, 79, -23, 41, -71, 4, -35, 86, -20, 97, 24, 38, 85, -94, 79, -6, -70, 1, 18, 4, -56, 42, -29, -50, -15, 79, -63, -60, -80, 33, -7, -27, -10, 71, 66, 10, 78, -74, 59, 98, 87, -30, -14, -29, -49, 45, 56, -89, 83, -4, -2, -60, -35, 18, -56, -69, -45, 44, 77, -8, -20, 74, -42, -44, -39, -51, -48, -79, 90, 70, -43, -30, 96, 0, -69, 12, -57, -52, -29, 38, 17, 65, -52, -79, -80, -70, 2, -45, 41, 69, 10, -10}
, {-56, -85, -14, -7, -58, -38, -54, -86, -62, -20, -53, -18, 47, -51, -14, -12, 83, 78, -42, 22, -32, 64, -97, -77, 42, -13, -73, 43, -3, 14, 28, 74, -51, -38, 53, -69, 15, -94, -72, -14, 80, -49, 66, 38, 33, -103, 86, -34, -78, 53, -49, -35, -81, 17, -38, -51, -89, -7, -40, -49, -46, 78, -25, 32, 47, -39, -36, 47, -60, -5, 25, 26, 23, 0, -94, -1, -63, 3, -71, 32, 18, 47, -83, -74, -39, -24, -62, 66, 22, -89, 37, -63, 93, -30, -3, 11, 76, -96, -81, -39, -81, 76, -8, -22, 32, 6, -8, 14, 53, -55, -32, 85, 0, -11, 20, -93, -33, 33, 82, -65, 56, -40, 82, -27, 26, 28, 58, -10}
, {0, 23, 41, 52, 39, 72, 30, -53, 30, 86, 54, -36, 2, -49, 37, 25, 5, 65, 48, 87, 14, 11, -1, -11, 49, 39, -7, -44, -84, 55, -47, -26, -97, 52, 60, -34, -75, -96, -54, -59, 18, -51, 2, 0, 28, 16, -80, 90, -30, -80, -27, 81, -27, -71, -23, 73, 3, -53, 27, 53, 35, -5, 38, -2, -44, 18, 10, 25, -30, -67, -75, 47, 67, 86, 76, 62, -110, 85, -86, -97, -86, 11, 66, -98, -30, -32, 60, 35, -80, -81, -35, -29, 64, -29, 25, 6, 8, -30, -82, -66, -102, 22, -78, -40, -53, -44, 17, -3, 0, -32, 15, 93, 96, -21, -58, -24, 79, -81, 81, 5, -51, -12, 34, 44, 5, -24, -50, -4}
, {-73, 93, 63, -37, 12, 9, 70, 21, 38, 15, 15, 80, -59, 71, -81, -10, 1, 29, 71, -27, -27, 2, 59, -75, -43, -14, 11, -7, -26, -10, 91, 61, -48, -76, 84, 79, -12, 70, -90, 0, 32, -45, 26, 98, -31, 30, 75, -100, -28, 4, 48, 0, -85, 14, 0, 74, -12, -59, 94, 72, 84, 90, 19, 21, 61, 57, -9, 8, 63, -84, 14, 77, -85, 29, -12, 69, 78, -6, -65, 49, -89, 98, -56, 81, 29, -27, 57, 75, -80, 23, 76, 10, 91, -100, -68, -88, -19, -55, 0, 9, -6, 7, 46, 61, 42, -2, -48, -51, 83, 15, 4, 31, 65, -72, 30, -98, 1, 45, -69, 68, 64, -23, 86, -3, 46, 45, -10, -109}
, {9, 74, -70, -40, 19, -7, 0, -53, 44, -72, 77, 21, 16, -61, -78, -10, 67, 37, 96, -19, 65, -65, -87, -75, 10, -12, -5, 46, 8, 26, -51, -103, -74, 73, -72, -50, -65, 4, -75, 30, 13, 1, -45, 44, 85, -21, 29, 62, -13, 6, -22, -54, -72, -84, 43, 38, 8, 69, 70, -15, -14, -94, -87, 87, 50, 12, -87, -45, -72, -41, 16, 28, -26, 19, -58, 0, 81, -33, -44, -71, 38, -49, -45, 24, 32, -43, 5, -22, 21, 86, -97, 81, 83, 7, -17, -90, 0, -20, -12, 30, 29, -6, 85, 81, -38, -40, -69, -85, 9, -51, 6, 29, -93, -43, 51, -5, -52, 57, -34, 63, -78, 56, -62, -27, 54, 10, -98, -57}
, {-21, 37, -91, 26, -22, -21, 64, -68, 80, 93, 34, -49, 101, 106, 33, 71, 92, 61, -77, 22, -43, 48, 52, -35, 97, -35, -83, -32, 13, -90, 52, -77, -41, -73, 91, -54, -92, -55, -84, 21, -38, 7, -70, 93, -39, 93, 7, 13, -67, 41, -97, -57, -59, 23, -27, 47, 79, 14, -38, -80, -42, 77, 88, -32, 79, -37, -23, 88, -95, 27, 73, -57, 83, 61, -14, 68, 106, 95, -86, -84, -8, 24, 79, -36, 59, -64, -59, -49, 36, -86, -57, -85, -68, -67, -47, 40, 54, 36, -73, 87, 77, 90, 34, -32, 16, 11, 49, -10, -40, 31, -52, -18, 9, -12, 53, -60, 25, -62, -22, -80, -67, 37, 45, -32, -72, -88, 24, -95}
, {-33, 16, -16, 87, 13, -33, 7, -77, 48, 80, 104, 9, -30, 47, -64, 33, -85, 38, 82, -31, 7, -49, -41, -91, -86, -70, 29, 8, -70, -91, 53, -45, -41, 42, 81, -47, -58, -75, -2, -59, 92, 92, 34, 6, -56, 0, 94, -97, 46, -58, 77, -61, 29, 44, 30, 34, 82, -54, -57, 72, -67, 82, 4, -85, 92, -66, -96, -51, -61, -51, -22, -71, -76, 29, 87, -25, -85, -6, 50, 91, 95, 51, 34, 96, 78, -30, 9, 70, 23, 90, -22, -1, 77, -86, 82, -7, -24, 2, -76, 98, -96, -18, 33, 7, 69, -37, -42, 83, -51, 40, 92, -83, 85, 48, 57, -23, -36, 35, 95, 7, -48, -93, -80, 91, -67, 9, -72, -24}
, {88, 32, 15, 72, 30, 67, 43, 90, 11, -2, 87, -36, 32, -43, 13, -100, -7, 6, 93, 16, 89, 98, 24, 61, 31, 99, 26, -35, -10, 22, -36, -89, 20, 65, 19, 2, 72, 81, 19, -58, 32, -46, -63, -29, -63, -23, 11, -94, -31, -23, 62, 28, -52, 11, 19, 81, 67, 63, 15, 36, 24, 84, 80, -34, 74, -64, -80, 19, -77, 80, -60, -69, -50, -78, -17, -64, 55, 27, 78, 47, 2, 36, 86, 6, -18, -36, 9, -27, -21, 46, -68, -65, 48, -4, -44, -26, -94, 87, -73, -37, 88, 79, -53, -16, 81, -64, 72, -82, -46, 93, 20, -16, 4, -64, -86, -74, -39, 3, 10, -86, 74, 50, 53, -5, 0, 5, -78, 61}
, {23, 19, 61, -65, 87, 76, 93, 26, 8, 64, 64, 59, -34, -59, -10, 35, -40, 55, -76, -25, -56, 61, -71, 21, -41, -18, -22, -40, 11, -86, 37, -69, 78, 61, -53, -90, -66, -66, -96, 21, -33, 52, -59, -78, 81, -35, -47, 41, -12, -40, 38, -50, -26, -32, -32, -15, -85, 75, -49, 60, -75, 61, 9, 5, -10, -60, -13, -21, 3, -92, -81, -33, -85, 93, -81, 32, 36, 0, 74, 16, 74, 2, 74, -29, 73, -55, -25, 45, 101, 65, 78, 37, -25, 2, -61, -66, -30, 89, -35, 60, -40, 37, 42, 23, 77, -10, 75, 36, 1, -89, 69, -84, -90, 77, -54, 31, 41, -58, -92, -36, 76, 53, 32, -62, 12, 65, 96, 86}
, {78, -51, 34, 65, 56, -87, 67, 38, 46, -81, 104, -96, 12, -3, 22, -41, 81, -1, 17, 51, -16, -75, 78, -61, 94, 18, 4, -63, 31, -5, 83, 27, 82, 9, -30, 82, -9, 63, -2, -92, -37, -40, 87, -95, -68, 41, 50, -80, 8, -33, 84, 56, -61, -75, -96, -12, 47, 78, -38, -81, 22, 20, 9, -11, -85, -46, -52, 80, 11, 61, -7, -11, 46, 50, 24, 63, -37, 32, -67, 70, -17, 38, 68, -61, 72, -15, 66, 57, -47, -82, 47, 37, 81, 67, 67, 22, -16, 9, -54, 91, -10, -79, 7, 13, 93, 16, -5, 62, 94, -8, 83, -12, 82, 74, 67, 24, 67, 44, 0, -37, 30, -90, 86, 85, 67, 63, 36, 77}
, {5, 29, 86, 32, -42, -30, -56, -83, 35, 84, 72, -93, -52, 71, 93, -61, 86, -40, 55, 89, 39, 91, 97, -1, 93, 63, -40, 27, -43, -74, 85, -1, -64, 49, -4, 11, -77, 21, 0, -82, 51, 40, 12, 96, 84, -54, -76, 1, 65, -29, 48, 62, 88, -76, 37, -48, 81, 78, -40, -90, 24, 65, 4, -29, 60, 66, -13, -13, 78, 86, -43, 89, -36, -25, -64, -58, -31, 53, -26, 11, -92, 57, -60, 19, -72, 41, 62, 6, -3, -16, -49, 45, -35, -24, -24, -80, 42, -59, 29, -4, 68, 42, -67, -90, 79, 85, 56, 80, 22, 2, -69, -38, 45, 49, -79, 66, -40, 15, 81, -1, -14, -51, -19, 62, -87, 43, -11, -25}
, {-75, 95, 91, -20, -95, -23, 78, 28, -6, 91, 49, -51, -78, 9, 98, 39, 13, -55, 73, -27, -67, 29, 36, 3, 72, -15, -18, -82, 60, -75, -69, 94, 29, -48, 42, -86, -88, -65, 43, -88, -22, -14, -39, 32, 26, -95, -40, -82, 11, -29, 77, 34, -81, -16, -48, -68, 60, 49, -44, 6, 76, 96, 45, -69, 62, -61, -91, -85, -8, -8, 22, -2, 51, -18, -86, 67, -94, -97, 2, -92, -18, 75, 92, -30, -55, 19, 23, -78, 15, 78, 26, 67, -79, -31, 17, 40, 7, -81, 81, 71, 41, 64, 26, 88, 31, -55, -76, -16, 88, 50, 89, 87, -4, 61, -9, -45, 95, 75, 57, -97, -32, 28, 13, 49, -42, 40, -1, -63}
, {-37, -14, -89, 61, 93, 55, 26, 65, -77, 20, 32, 79, 41, 33, 86, 40, -106, -3, 78, -43, -75, -39, -96, -9, -33, -35, 47, 49, 14, 7, -44, -33, 13, 44, -44, 89, -24, 97, -62, -78, -97, 12, -89, 88, 32, -91, -68, -23, -16, 66, -69, -8, 1, -40, -9, -54, 44, 77, -37, 100, -107, 57, 17, 62, 23, 70, 30, 34, -89, -45, 1, 55, -49, 38, 60, 0, -84, 10, -23, 66, -19, -29, 83, 59, -19, 64, -42, 2, -60, -55, -4, 24, 39, 47, 6, -4, -88, -15, 3, -12, 50, 0, -48, -104, 10, -14, -94, 60, -31, 17, 86, 54, 51, 83, -5, -4, 55, 16, -80, 81, -15, -69, 27, 49, -86, 46, 59, -29}
, {105, 19, -77, 95, -41, 90, 73, -67, -15, -8, 100, -70, 82, -26, 68, 45, -28, 58, 32, -93, -75, 29, 57, -95, -60, -43, 94, -74, -58, -57, 114, -28, 40, -92, 14, 46, 43, 71, -54, -92, 49, -44, -44, 73, -73, 42, -63, -53, 45, 13, 98, -34, 15, 6, 8, 30, 24, 20, -41, 75, 66, 84, 60, -29, -88, 91, -33, -10, 70, -18, -68, 6, 88, -57, 1, 42, -32, 9, -73, -7, 27, 62, -74, 8, 104, -47, -18, -35, -54, -76, -58, 60, 45, 90, 70, -35, -15, 11, 61, -64, 75, -34, 69, -50, 99, 37, 34, -71, 60, -31, 92, 95, 90, 91, -49, -30, -23, -86, -70, -27, 42, -19, 30, 33, 83, -86, -30, -101}
, {-78, -57, -15, 63, -68, 14, -21, -3, 11, 30, 4, 37, 37, -2, 66, 66, -63, 30, -58, -25, 38, -42, -81, -30, 45, 43, -88, -90, -87, 41, 62, 43, 12, 25, 23, 28, 78, 68, 23, 49, 77, -68, -13, 90, 25, 61, 80, 50, 52, 48, -9, 43, -69, 20, -8, -11, -84, 75, 48, -90, -15, 76, 20, -28, 59, 70, 51, 57, 18, 3, -41, 36, 74, -61, -26, -67, 8, -25, -51, 27, -37, -94, -2, 49, -73, -85, 13, -27, 52, -84, 35, -39, -91, -77, -72, -14, 75, 3, 6, -33, 0, -79, -51, 29, -26, -50, -58, -79, -65, -5, -68, 93, 25, -6, 10, 82, -27, -71, 64, -7, 38, 10, 87, -81, 102, 13, 75, -102}
, {52, 80, -83, 6, -99, 57, 66, -15, 62, -74, -33, -37, -115, 24, -51, -77, -80, -66, 22, -29, -16, -10, -30, 43, -66, -47, -20, -58, -62, -5, 3, 23, 65, -37, -27, -24, -42, -35, -30, -65, 24, 22, -72, 16, 100, -32, -47, 91, 85, -49, -52, 26, 39, 51, -105, 28, -48, 43, 9, 15, -62, 48, -83, -78, 51, -41, 68, 84, 4, -35, -107, 69, 78, 80, -20, 1, -66, 18, 46, -50, -85, 48, 76, -76, -49, 0, 0, -70, 34, -58, -40, -88, -71, 80, -44, 6, -37, -18, 43, 73, -61, 2, -29, 90, -53, -83, 36, -102, 21, -45, -20, -68, -56, -73, 70, 76, 13, -76, 11, 35, -11, -44, -12, 1, 19, 34, 65, 45}
, {-61, -24, -52, -24, 103, 48, -55, -43, -61, 85, -31, 61, -31, -37, 26, 65, -98, 47, -84, -62, -53, 74, 8, 55, 22, 81, 75, -17, 91, 8, -29, 75, 74, 29, -1, -31, -92, -5, 56, 4, 2, -89, 52, 78, 8, 10, -51, -5, -38, 93, 23, -38, -66, 43, 9, -1, 18, -69, 39, 23, -10, -70, -8, -7, 0, 54, 45, -74, -25, 0, 79, -65, 59, 23, 23, -25, -21, -73, -53, -55, 55, -50, 46, -2, 66, -62, 0, 35, 3, -90, -66, 54, 79, 56, -84, -11, 34, 85, -45, 53, -46, 41, -46, -60, 65, -22, -71, 45, 52, 13, -48, 3, -43, 86, -71, -73, 32, -66, -19, 41, 28, 33, 53, 44, 61, 3, -77, 101}
, {79, -28, 66, -15, -106, 10, 13, 49, -50, 45, -22, -20, -35, 15, -1, -69, -39, -12, -15, 32, 37, 88, 0, -92, 72, 87, 76, -53, 58, -38, -70, 54, 55, 67, -59, -52, 47, 34, -62, 64, 92, 9, -51, -88, -86, 78, -83, 33, 81, -73, 63, 17, 70, -19, 86, 70, 66, 67, 23, 11, 20, -94, -67, -53, -65, -111, 92, 63, 67, 8, 9, -35, -22, -53, 99, -25, 16, 73, -23, 61, 70, -41, -67, -100, -34, 31, -79, -73, -80, 36, -61, 5, -61, 97, -14, 85, 14, -12, 56, 82, -72, 20, -17, -3, 56, -31, -82, 72, 11, 41, -72, 60, 97, -35, -58, 41, -85, 45, 0, -94, 60, 95, 66, -75, 6, -41, -69, 82}
, {0, 23, 64, -43, -62, -90, 30, 35, 33, 57, 44, -28, 41, -39, 91, 27, 48, 20, 66, 90, 21, 5, -20, -23, 9, 61, -64, 79, -77, 63, -14, -16, 47, -20, 54, 33, -29, 8, 0, 57, -13, 18, 52, -27, -65, -47, 89, 5, 11, -39, 13, -60, -89, -7, 25, 67, -80, -68, 38, -4, 10, 59, -34, -48, -89, -11, 10, 17, -41, 15, -25, 63, 76, 38, -69, -26, 24, 66, 86, -45, -24, 69, 65, 27, 39, -79, -56, -40, -29, -36, 60, 13, 35, -13, 79, 79, -72, -1, -84, -60, -56, -42, 75, -34, -11, 17, -76, -80, 83, -94, 100, -66, 2, -96, -66, -71, -55, -29, 10, 48, 50, -14, 50, -8, -2, 66, -81, -79}
, {19, 7, -67, -75, 39, -78, -39, 76, 16, -62, -60, 80, 6, 55, -34, 23, 21, 21, -31, -92, 92, 55, -23, 70, 25, -87, -11, 10, -54, 68, 22, -14, -88, 57, 33, -43, -76, 64, 72, 54, 71, 5, 35, -25, -91, 63, -20, -61, 87, -42, 20, 62, -64, 90, -106, 19, 48, -53, -46, -7, -87, -45, -2, -80, 82, 52, 23, 51, -13, 88, -89, -3, -59, -9, 22, -67, 59, -9, -13, -78, 22, -74, 52, -31, 56, -39, 28, -81, -33, 45, 77, -64, -25, 5, -72, 72, 69, 9, 32, 10, -41, -68, -26, 76, 58, 47, 29, -67, 4, -66, -82, 90, 53, 43, 81, -35, 59, 6, -47, -13, -68, 65, 86, -92, 85, -56, -41, 66}
, {-21, -37, -49, 23, 90, 77, 17, 45, 29, 64, -109, 94, -93, 35, -92, -82, -39, 15, -47, 60, -82, 76, -82, -35, 37, 61, 98, -51, -49, 0, 13, 91, 101, 49, -35, 71, -11, 87, -74, 20, 3, 37, 88, -8, -24, 31, -13, -56, -9, -49, -14, -73, -57, -41, -1, -86, 93, 28, -11, -15, 96, 55, -16, 79, 45, 113, -70, 57, -49, -26, -30, -59, -27, -25, -86, 13, -83, 41, 29, 22, 57, -10, 66, -37, 52, 49, 99, -68, 84, 60, 11, 6, -20, -13, -77, 42, -44, 0, -53, 61, -81, -15, -19, 70, 62, 84, 10, -82, -4, 46, -64, 59, -39, -14, -95, -2, -46, -21, 93, -28, -68, -21, -32, -26, -68, 89, 97, 33}
, {-55, 24, -74, -17, 25, -25, -48, 55, 44, 14, -70, -12, 38, -44, -22, -74, -43, -76, -97, 34, 81, -38, -32, -96, 47, 20, 59, -11, 82, 31, 64, 41, -1, -7, -62, 4, 43, -69, 13, -65, 26, -56, 49, -4, -45, 20, -59, 36, -74, -76, -68, 19, -59, 44, -52, 95, -44, 86, 10, 18, -55, 51, 34, 81, 22, 48, 10, 11, 90, 67, 3, 95, 11, 88, 5, 64, -61, 56, -80, -15, 12, 69, -54, 48, -57, -94, -69, -82, 56, 73, -13, 71, 11, 31, 61, 66, -72, 42, 84, -57, -18, 65, 30, -53, 36, 72, 59, 82, 61, 75, 10, 89, -16, -40, -85, 27, -54, 21, 48, -50, 56, 70, -77, 40, -28, -42, 104, 14}
, {-13, -88, 9, -80, 60, -25, 1, -20, 48, 96, 13, -12, -3, 60, -34, -40, -98, 5, -70, 49, -81, -42, 22, 76, 49, -45, -18, -17, 90, 85, -13, 53, 61, 76, -80, -83, -22, 75, 36, 3, -91, 18, 35, -3, 5, 59, -92, -67, -41, -4, 92, 12, 67, 95, 91, 66, -81, 48, -32, -70, -108, -90, -10, 70, 66, 48, 61, 9, -31, -36, -13, 73, 41, -26, 17, -81, 69, 69, 3, -42, 70, -39, -3, 15, 0, 20, 11, -8, -85, -45, -81, -2, -50, 69, 10, 27, 18, 44, -67, 24, -28, -58, -5, -79, -58, 38, 78, -57, 11, 0, 48, -73, -77, -84, -61, 73, 53, 20, 64, 15, -22, 81, 16, 5, 26, -23, 89, 45}
, {21, -75, 50, 59, 91, 33, -46, 76, 83, -77, -99, -83, 63, 60, -52, -76, -76, 51, -23, 5, -7, -4, -90, -59, 43, -92, 19, -37, 17, 74, -43, 52, 98, 14, -23, 94, -70, -32, -36, 91, -98, 64, -66, 1, 3, -31, 35, -75, -58, -56, -64, -46, 73, 14, -37, 42, 64, -71, 68, -75, 82, -42, 11, -76, -25, 96, 1, -30, -33, 23, -52, 83, 67, 37, 23, -82, -43, 49, -46, -95, 22, -45, 78, -76, 98, 0, 84, 87, 31, -74, 3, 70, -55, 88, -92, 47, -96, -9, -78, -1, 76, 35, -20, -41, 63, 52, 25, 91, 46, -3, 0, 50, -91, -15, 58, 25, 35, -41, 26, 0, 17, 15, 1, -57, 42, -59, -65, 60}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES 40
#define FC_UNITS 4
#define ACTIVATION_LINEAR

typedef number_t dense_3_output_type[FC_UNITS];

static inline void dense_3(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]) {			                // OUT

  unsigned short k, z; 
  long_number_t output_acc; 

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0; 
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ( kernel[k][z] * input[z] ); 

    output_acc = scale_number_t(output_acc);

    output_acc = output_acc + bias[k]; 


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = clamp_to_number_t(output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0)
      output[k] = 0;
    else
      output[k] = clamp_to_number_t(output_acc);
#endif
  }
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 40
#define FC_UNITS 4


const int16_t dense_3_bias[FC_UNITS] = {7, -7, -2, 10}
;

const int16_t dense_3_kernel[FC_UNITS][INPUT_SAMPLES] = {{-154, 136, -67, 5, 32, 151, -83, 1, 81, -113, 67, 189, -178, 75, 122, 32, 31, -134, -121, -140, 182, 164, -16, 26, 2, 72, 88, -104, -50, 185, 24, -189, -126, -54, 41, 5, -84, -21, -137, -125}
, {-20, 108, -113, 101, -4, -158, -90, -157, 12, 168, 84, 11, -63, 20, -99, 118, -123, -88, 93, -184, -28, -37, -117, -84, 33, -186, 50, -179, 166, -152, -67, -86, 48, 53, -17, 97, -168, -102, -5, -76}
, {-166, -45, -154, 23, 29, 60, 170, -181, -19, -76, -144, -158, 95, -130, -1, -112, -136, -34, 32, -156, -141, -154, 151, -99, 155, -4, 159, 138, 58, -63, -51, -141, 5, -189, -6, 69, 172, 11, -65, -20}
, {-5, -21, 178, -192, 90, 93, 87, -85, -115, 61, 11, 32, -67, -83, 118, 72, -30, -54, -77, -119, 85, 197, -115, -10, 162, -189, 118, -92, 168, -112, -128, -194, 94, -184, 27, -90, 149, 109, -44, 134}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define MODEL_OUTPUT_SAMPLES 4
#define MODEL_INPUT_SAMPLES 16000 // node 0 is InputLayer so use its output shape as input shape of the model
#define MODEL_INPUT_CHANNELS 1

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  //dense_3_output_type dense_3_output);
  number_t output[MODEL_OUTPUT_SAMPLES]);

#endif//__MODEL_H__
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"

 // InputLayer is excluded
#include "max_pooling1d_5.c" // InputLayer is excluded
#include "conv1d_4.c"
#include "weights/conv1d_4.c" // InputLayer is excluded
#include "max_pooling1d_6.c" // InputLayer is excluded
#include "conv1d_5.c"
#include "weights/conv1d_5.c" // InputLayer is excluded
#include "max_pooling1d_7.c" // InputLayer is excluded
#include "conv1d_6.c"
#include "weights/conv1d_6.c" // InputLayer is excluded
#include "max_pooling1d_8.c" // InputLayer is excluded
#include "conv1d_7.c"
#include "weights/conv1d_7.c" // InputLayer is excluded
#include "max_pooling1d_9.c" // InputLayer is excluded
#include "average_pooling1d_1.c" // InputLayer is excluded
#include "flatten_1.c" // InputLayer is excluded
#include "dense_2.c"
#include "weights/dense_2.c" // InputLayer is excluded
#include "dense_3.c"
#include "weights/dense_3.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_3_output_type dense_3_output) {

  // Output array allocation
  static union {
    max_pooling1d_5_output_type max_pooling1d_5_output;
    max_pooling1d_6_output_type max_pooling1d_6_output;
    max_pooling1d_7_output_type max_pooling1d_7_output;
    max_pooling1d_8_output_type max_pooling1d_8_output;
    max_pooling1d_9_output_type max_pooling1d_9_output;
    dense_2_output_type dense_2_output;
  } activations1;

  static union {
    conv1d_4_output_type conv1d_4_output;
    conv1d_5_output_type conv1d_5_output;
    conv1d_6_output_type conv1d_6_output;
    conv1d_7_output_type conv1d_7_output;
    average_pooling1d_1_output_type average_pooling1d_1_output;
    flatten_1_output_type flatten_1_output;
  } activations2;


  //static union {
//
//    static input_2_output_type input_2_output;
//
//    static max_pooling1d_5_output_type max_pooling1d_5_output;
//
//    static conv1d_4_output_type conv1d_4_output;
//
//    static max_pooling1d_6_output_type max_pooling1d_6_output;
//
//    static conv1d_5_output_type conv1d_5_output;
//
//    static max_pooling1d_7_output_type max_pooling1d_7_output;
//
//    static conv1d_6_output_type conv1d_6_output;
//
//    static max_pooling1d_8_output_type max_pooling1d_8_output;
//
//    static conv1d_7_output_type conv1d_7_output;
//
//    static max_pooling1d_9_output_type max_pooling1d_9_output;
//
//    static average_pooling1d_1_output_type average_pooling1d_1_output;
//
//    static flatten_1_output_type flatten_1_output;
//
//    static dense_2_output_type dense_2_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d_5(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_5_output
  );
 // InputLayer is excluded 
  conv1d_4(
    
    activations1.max_pooling1d_5_output,
    conv1d_4_kernel,
    conv1d_4_bias,
    activations2.conv1d_4_output
  );
 // InputLayer is excluded 
  max_pooling1d_6(
    
    activations2.conv1d_4_output,
    activations1.max_pooling1d_6_output
  );
 // InputLayer is excluded 
  conv1d_5(
    
    activations1.max_pooling1d_6_output,
    conv1d_5_kernel,
    conv1d_5_bias,
    activations2.conv1d_5_output
  );
 // InputLayer is excluded 
  max_pooling1d_7(
    
    activations2.conv1d_5_output,
    activations1.max_pooling1d_7_output
  );
 // InputLayer is excluded 
  conv1d_6(
    
    activations1.max_pooling1d_7_output,
    conv1d_6_kernel,
    conv1d_6_bias,
    activations2.conv1d_6_output
  );
 // InputLayer is excluded 
  max_pooling1d_8(
    
    activations2.conv1d_6_output,
    activations1.max_pooling1d_8_output
  );
 // InputLayer is excluded 
  conv1d_7(
    
    activations1.max_pooling1d_8_output,
    conv1d_7_kernel,
    conv1d_7_bias,
    activations2.conv1d_7_output
  );
 // InputLayer is excluded 
  max_pooling1d_9(
    
    activations2.conv1d_7_output,
    activations1.max_pooling1d_9_output
  );
 // InputLayer is excluded 
  average_pooling1d_1(
    
    activations1.max_pooling1d_9_output,
    activations2.average_pooling1d_1_output
  );
 // InputLayer is excluded 
  flatten_1(
    
    activations2.average_pooling1d_1_output,
    activations2.flatten_1_output
  );
 // InputLayer is excluded 
  dense_2(
    
    activations2.flatten_1_output,
    dense_2_kernel,
    dense_2_bias,
    activations1.dense_2_output
  );
 // InputLayer is excluded 
  dense_3(
    
    activations1.dense_2_output,
    dense_3_kernel,
    dense_3_bias, // Last layer uses output passed as model parameter
    dense_3_output
  );

}
