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

typedef number_t max_pooling1d_40_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_40(
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

typedef number_t conv1d_32_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_32(
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


const int16_t conv1d_32_bias[CONV_FILTERS] = {2, -5, -16, -15, 33, -25, -39, -7, 7, -23, -16, -40, 21, -29, -6, -12}
;

const int16_t conv1d_32_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{110, 0, 23, -52, 88, 64, -56, -83, 89, 78}
}
, {{86, -17, 15, -24, 0, -31, -51, -66, -41, -107}
}
, {{51, -72, -6, -27, -92, 49, -91, 26, 39, 56}
}
, {{-11, -51, 56, 28, -87, -17, -80, -82, 66, 99}
}
, {{-5, 0, -39, -2, 0, 50, -61, -23, -72, -99}
}
, {{-102, -30, 37, -17, -69, -49, 58, 46, 71, 92}
}
, {{27, 103, 53, -9, -1, -62, 4, -50, 22, -101}
}
, {{-40, 0, -45, 32, 13, 3, 7, -91, 20, -91}
}
, {{87, -50, -58, -24, 83, -70, -28, 51, -34, -27}
}
, {{74, -1, 65, 54, -46, -4, 10, 32, 95, 105}
}
, {{-66, -45, -65, -78, 10, -27, -55, -63, -29, 61}
}
, {{-90, -76, -21, -6, 70, -16, -35, -14, 14, 74}
}
, {{-17, 84, 4, 46, 96, -32, -6, 19, 82, 83}
}
, {{69, 74, -25, 48, -95, -96, 53, 37, 0, 9}
}
, {{-44, 91, 92, 16, 84, -59, -89, -95, -100, -27}
}
, {{-85, -55, -56, 0, -3, -46, 55, 80, -53, 54}
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

typedef number_t max_pooling1d_41_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_41(
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

typedef number_t conv1d_33_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_33(
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


const int16_t conv1d_33_bias[CONV_FILTERS] = {17, 10, -18, 4, -13, 0, 0, -18, -1, 15, 0, -13, 26, 12, -30, -10, -29, 40, 0, 5, 66, 51, -14, -7, 7, -3, -3, -10, -9, 24, 2, -42}
;

const int16_t conv1d_33_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{35, 24, -52}
, {94, -85, 50}
, {104, -69, -48}
, {-95, -24, -44}
, {3, -76, 33}
, {39, -51, -47}
, {47, -115, 67}
, {-7, 67, -43}
, {-67, 78, -68}
, {65, 71, 39}
, {-24, -41, 3}
, {70, 9, -65}
, {0, -33, -94}
, {-35, -126, 80}
, {-73, 33, 61}
, {-24, -52, -109}
}
, {{-7, 4, 71}
, {-69, -71, -59}
, {-84, 79, -93}
, {54, 46, -122}
, {82, 38, 24}
, {-46, 105, 0}
, {14, -39, -95}
, {-26, -37, -49}
, {33, 79, -37}
, {14, 44, 21}
, {-9, -77, 11}
, {-109, 29, 96}
, {-53, 31, 52}
, {99, -19, -19}
, {91, -2, 49}
, {-16, 41, -41}
}
, {{29, -49, 91}
, {-72, 76, 19}
, {-50, -43, 95}
, {51, -14, -16}
, {34, -28, -110}
, {-118, 65, -43}
, {-26, -3, 45}
, {53, 6, 62}
, {-52, 33, -36}
, {21, 106, -83}
, {92, -99, 13}
, {-3, -59, -58}
, {-38, -28, 26}
, {-19, -33, 16}
, {-40, 96, 4}
, {3, -41, 90}
}
, {{77, -87, -20}
, {-93, 72, -80}
, {-3, -27, -33}
, {57, -4, 13}
, {75, 76, 24}
, {-17, -22, -7}
, {25, -70, -59}
, {-8, 30, 28}
, {-68, 68, 8}
, {47, -25, -32}
, {47, -73, 7}
, {-76, 51, 77}
, {89, 72, 38}
, {-64, -23, 47}
, {-106, -19, 36}
, {-50, 85, 28}
}
, {{30, 43, 6}
, {-47, 57, 80}
, {57, 19, 75}
, {112, -3, -60}
, {12, -97, 18}
, {-90, 80, -25}
, {13, 17, -73}
, {54, -24, 96}
, {-28, -86, -29}
, {-42, -54, -9}
, {-44, -15, 20}
, {60, -47, 42}
, {-64, 22, -76}
, {54, 24, -60}
, {94, -105, 50}
, {-78, -8, 33}
}
, {{20, 90, 100}
, {-94, -52, 70}
, {94, -28, 55}
, {43, 86, -86}
, {10, -34, 37}
, {27, -16, -15}
, {-47, 105, 31}
, {-92, 3, 55}
, {-106, -52, -82}
, {-30, -46, -70}
, {-90, 74, -3}
, {-106, 49, -94}
, {-112, 54, -92}
, {99, 102, 16}
, {-17, -68, -53}
, {-29, 5, 12}
}
, {{19, -30, 49}
, {105, -121, -96}
, {14, -87, 17}
, {36, -11, -64}
, {2, -130, 20}
, {-91, 73, 14}
, {-38, -79, 106}
, {78, -22, 66}
, {-85, 84, 85}
, {70, 42, -112}
, {-61, 84, 45}
, {-66, 20, -73}
, {-46, -8, -29}
, {6, -49, -56}
, {15, 34, 87}
, {-105, 18, 58}
}
, {{-80, -3, -74}
, {-77, 35, -65}
, {-64, 45, -59}
, {55, 69, 13}
, {43, 89, 21}
, {-40, -2, -103}
, {-37, 5, -17}
, {-100, 42, -26}
, {-71, 78, -72}
, {46, -79, 26}
, {-14, -36, -79}
, {-62, 69, 55}
, {7, 45, 10}
, {63, -102, -11}
, {-63, 73, -82}
, {-4, 3, -89}
}
, {{-104, -15, -66}
, {-8, -53, -88}
, {-78, 84, 10}
, {-41, 28, -55}
, {-27, 0, -23}
, {-39, -35, -60}
, {-64, -30, 18}
, {-72, -74, 44}
, {-82, 33, -44}
, {73, 97, -39}
, {32, 95, 72}
, {-88, -20, 56}
, {-5, -65, -93}
, {-81, 95, -59}
, {-41, 95, -102}
, {19, 99, 73}
}
, {{-69, 0, 3}
, {-21, -51, 98}
, {-15, 92, -40}
, {17, -51, 46}
, {107, 87, -14}
, {-41, -50, -47}
, {-94, -85, 93}
, {12, 20, 95}
, {59, -52, 63}
, {-43, -43, -90}
, {-64, 41, -43}
, {-49, 47, -58}
, {66, 90, -16}
, {22, -55, -67}
, {99, 2, 10}
, {110, -67, -2}
}
, {{-78, 0, -117}
, {-39, 7, -60}
, {-9, 38, -17}
, {-45, 94, 17}
, {-78, 46, 42}
, {-43, -25, 57}
, {-91, -75, 105}
, {-49, 25, -15}
, {-22, -1, 27}
, {89, 91, -33}
, {92, -86, -30}
, {71, -46, 107}
, {44, 16, 70}
, {67, -78, 103}
, {-53, 24, 62}
, {-40, -88, -48}
}
, {{-66, -78, -102}
, {-35, -26, -9}
, {86, -36, 26}
, {29, 39, 52}
, {-103, -36, 99}
, {81, -96, 25}
, {-24, 33, 73}
, {14, -12, -74}
, {-74, 73, 34}
, {98, 98, 52}
, {6, 3, -47}
, {44, -68, 12}
, {-10, 46, -104}
, {-51, 84, 80}
, {-76, -21, 86}
, {-69, -28, -31}
}
, {{-102, -13, 97}
, {-6, -85, 18}
, {40, -15, -87}
, {68, -60, -23}
, {-10, -83, -27}
, {-25, 111, -39}
, {9, -13, 96}
, {-77, -63, -41}
, {96, 45, 54}
, {-110, -47, 65}
, {-35, -81, 101}
, {107, -43, -57}
, {73, 29, -30}
, {-73, 60, 84}
, {-32, 44, 64}
, {9, -30, -15}
}
, {{15, -77, 77}
, {98, 21, -78}
, {15, 89, 25}
, {-71, 95, -32}
, {23, -67, -67}
, {-69, 125, -52}
, {-38, -33, 72}
, {91, 79, 74}
, {12, -82, -72}
, {-6, -51, 115}
, {94, -77, 119}
, {-63, 17, -101}
, {113, 43, 89}
, {-27, -5, -47}
, {21, -81, -99}
, {9, -68, 3}
}
, {{-20, -47, 51}
, {-138, 53, -33}
, {22, -6, 74}
, {99, 2, -5}
, {21, -16, -47}
, {104, 17, -40}
, {-83, 38, 16}
, {-71, 82, 64}
, {60, -69, -40}
, {30, 60, -120}
, {36, 45, 3}
, {43, 49, 60}
, {89, 16, 15}
, {84, -126, 93}
, {-22, -57, 11}
, {96, -70, -88}
}
, {{34, 17, 37}
, {-50, -94, 14}
, {-2, 5, 46}
, {56, -86, -55}
, {-23, -3, 16}
, {39, -57, -22}
, {15, -40, -98}
, {-71, -104, 22}
, {-112, 5, 36}
, {-49, -73, -33}
, {64, -111, -15}
, {-53, 60, -37}
, {-102, -11, 43}
, {-36, -40, -56}
, {18, -3, -11}
, {-103, 51, 2}
}
, {{0, 109, 82}
, {60, 45, 84}
, {18, 85, -47}
, {-94, -67, 30}
, {-56, -44, -92}
, {67, -109, 61}
, {73, 48, 40}
, {43, 55, 9}
, {5, 28, -30}
, {-29, -76, 82}
, {94, 93, -100}
, {7, -79, 56}
, {-35, -23, -77}
, {-36, 103, 30}
, {8, -41, -107}
, {-20, -114, -32}
}
, {{11, 79, 42}
, {-91, 54, 17}
, {-2, 37, -57}
, {39, 106, -76}
, {24, 56, -40}
, {-94, -110, -84}
, {-72, 31, 32}
, {51, 40, -100}
, {38, 29, -34}
, {-29, -8, 31}
, {61, 51, -26}
, {-18, -77, -47}
, {102, 48, 83}
, {-3, -84, -41}
, {-29, -67, -10}
, {-10, 73, -32}
}
, {{82, -82, 24}
, {79, 2, -94}
, {-37, 13, 58}
, {8, 60, -10}
, {-8, -9, 18}
, {-66, -12, 1}
, {102, -16, -12}
, {-101, 0, -47}
, {29, -80, -55}
, {80, 93, -52}
, {44, 90, -77}
, {-58, 0, -85}
, {-73, 26, -53}
, {-50, 23, 18}
, {-23, -49, -23}
, {81, -20, -85}
}
, {{64, 87, 90}
, {-10, 41, -34}
, {68, -58, -60}
, {48, 75, -32}
, {-14, 100, 66}
, {83, -85, 22}
, {39, 72, 74}
, {-68, -83, 51}
, {38, 16, -35}
, {-69, -70, 96}
, {-72, -94, -105}
, {50, 53, -99}
, {-5, -59, -51}
, {-18, 59, -64}
, {-71, -46, -11}
, {-89, 56, 99}
}
, {{21, 110, -99}
, {-34, 77, 0}
, {103, -61, 84}
, {-46, -76, -90}
, {-57, 104, 78}
, {-69, -28, 80}
, {7, -56, -40}
, {-98, -5, 58}
, {60, 77, -75}
, {135, 67, -39}
, {-97, -32, -39}
, {-26, -105, 0}
, {49, -16, 89}
, {2, 126, -106}
, {-57, -83, -80}
, {-42, -11, -59}
}
, {{-60, 95, -76}
, {84, -5, -70}
, {-57, 12, -80}
, {36, -3, 62}
, {-16, 43, 51}
, {-82, -80, -94}
, {-61, -30, -21}
, {176, 11, 64}
, {39, 39, 26}
, {0, -56, -41}
, {125, -31, -31}
, {94, 22, -6}
, {-31, 107, -45}
, {-52, -32, 25}
, {-55, -99, 14}
, {-80, 54, -99}
}
, {{-36, 103, 79}
, {-75, 62, -109}
, {22, 21, -22}
, {-4, -11, -23}
, {-90, 4, -119}
, {35, 0, 31}
, {45, 66, -80}
, {-106, -13, -44}
, {-46, -111, -103}
, {80, 20, 71}
, {82, -90, -11}
, {100, 51, 106}
, {-56, 8, -39}
, {30, 76, 69}
, {-102, 73, 6}
, {60, -45, 40}
}
, {{15, 57, -71}
, {24, 98, 22}
, {-36, 103, -30}
, {-67, -5, 12}
, {-102, 27, -4}
, {14, -47, -93}
, {-77, -69, 36}
, {53, -76, 62}
, {-82, 42, 29}
, {-12, -90, 79}
, {102, -76, -102}
, {85, 4, 77}
, {-26, -49, -58}
, {110, -51, -23}
, {-107, 66, -65}
, {-19, 112, 44}
}
, {{6, 106, -48}
, {-45, -7, 55}
, {6, 56, -44}
, {86, 108, -1}
, {-93, -14, -56}
, {145, -80, -50}
, {79, -77, -90}
, {22, -48, -8}
, {16, 79, 127}
, {-91, 87, -34}
, {-62, 65, -42}
, {21, -50, -14}
, {-61, 63, -31}
, {50, 58, 78}
, {-27, -39, -14}
, {-6, 88, 8}
}
, {{-111, -74, 16}
, {-49, 8, -22}
, {-108, 72, -55}
, {-61, 10, 23}
, {-68, 21, 38}
, {-51, 31, 92}
, {-69, 89, 55}
, {-76, -35, 100}
, {-68, 88, -92}
, {-54, 63, -51}
, {-5, -16, 14}
, {-90, 24, -4}
, {-2, 94, 67}
, {21, 35, -56}
, {11, 76, -11}
, {20, -66, -4}
}
, {{-27, -83, -47}
, {-27, -52, -28}
, {9, 18, 66}
, {78, -46, -72}
, {-41, 37, -30}
, {45, -17, 53}
, {-5, -25, 64}
, {-54, -104, 62}
, {-99, -56, -68}
, {-18, 78, 12}
, {-7, -74, -67}
, {-86, 88, -39}
, {-80, -21, -89}
, {16, -3, -32}
, {-54, 73, -10}
, {-33, -100, -42}
}
, {{-6, -52, 94}
, {22, -8, 105}
, {76, 72, 20}
, {10, 41, 69}
, {54, -100, 79}
, {5, 71, -88}
, {83, -58, 42}
, {83, -32, 16}
, {29, 57, 38}
, {-84, 5, -15}
, {11, -54, -96}
, {-80, -5, -105}
, {91, -75, 79}
, {-77, 40, 60}
, {13, 31, -25}
, {-89, 31, 67}
}
, {{-48, 23, 81}
, {13, -63, 100}
, {12, 55, -98}
, {36, -62, 29}
, {-8, -55, 10}
, {13, -97, -56}
, {-15, 58, -51}
, {-35, -94, 31}
, {-12, 72, 54}
, {16, -16, 68}
, {22, 106, -1}
, {-70, -41, 16}
, {18, -29, -89}
, {-7, -83, 100}
, {97, -20, -67}
, {62, 97, -29}
}
, {{-22, 115, -58}
, {-65, -70, -13}
, {65, 32, -40}
, {48, 27, 32}
, {-92, -8, 96}
, {44, 71, -33}
, {3, -91, -97}
, {-43, -48, -109}
, {83, 79, 43}
, {-87, -35, 19}
, {90, -19, 63}
, {27, 112, 50}
, {-41, -42, 47}
, {-93, -44, -89}
, {101, 61, -79}
, {62, -50, -34}
}
, {{8, -94, 18}
, {11, 44, 61}
, {-29, -82, 37}
, {-20, -16, 46}
, {43, 33, 39}
, {40, -1, -52}
, {51, -64, -4}
, {-109, 2, -13}
, {25, -79, 43}
, {-28, 28, -72}
, {84, -101, 38}
, {-121, -33, 116}
, {100, -74, -10}
, {-54, 30, -95}
, {-29, 88, -64}
, {-6, 85, 65}
}
, {{-90, 74, -100}
, {89, 18, 67}
, {34, -88, -98}
, {36, 114, 20}
, {-27, 37, 101}
, {-22, 20, -16}
, {96, -6, 94}
, {8, 54, -96}
, {21, 54, -6}
, {-100, -25, 94}
, {-42, -16, 0}
, {68, -62, 77}
, {-46, 106, -70}
, {-78, -72, -6}
, {27, -110, -23}
, {85, -43, 52}
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

typedef number_t max_pooling1d_42_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_42(
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

typedef number_t conv1d_34_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_34(
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


const int16_t conv1d_34_bias[CONV_FILTERS] = {-8, 28, 15, -20, -26, 4, -10, 7, 7, -21, 36, 2, -7, 27, 11, -6, -21, -14, -4, 9, -8, -3, -5, 8, -1, -13, -2, 29, 2, 14, 32, -23, -10, -21, 6, 11, -21, 17, 42, 28, -13, 37, -7, -5, 7, 7, 13, -5, -19, -18, 5, 6, 2, -7, 2, 0, -1, -4, -9, 0, 34, 42, -5, -5}
;

const int16_t conv1d_34_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{28, -84, -47}
, {-55, -51, -14}
, {25, 52, -65}
, {-50, 39, 35}
, {70, 49, 30}
, {60, 55, 37}
, {29, -43, 61}
, {49, -66, 3}
, {-3, -28, 23}
, {42, -77, -58}
, {55, 46, -14}
, {49, 28, -17}
, {-10, 49, -19}
, {-34, 18, -34}
, {50, -58, 44}
, {36, -13, -53}
, {-26, 56, 15}
, {-9, -74, 26}
, {39, 51, -18}
, {-46, 25, -46}
, {-84, 56, -51}
, {-27, 25, 48}
, {53, 43, -70}
, {-7, 0, -64}
, {-37, -83, 9}
, {-63, -39, -59}
, {-66, -23, -47}
, {71, 56, 56}
, {-34, 35, 17}
, {-80, 77, 10}
, {-46, 22, 67}
, {-79, -23, -69}
}
, {{18, -92, 56}
, {-25, 54, -39}
, {-5, -24, 18}
, {-45, 51, 35}
, {42, -76, 37}
, {57, -42, -35}
, {-6, -65, 16}
, {-67, 9, 17}
, {-29, -30, -29}
, {-3, 22, -26}
, {41, -6, 46}
, {6, -38, 8}
, {79, -10, -1}
, {-30, 96, 54}
, {-62, -19, 17}
, {-40, 22, -55}
, {38, 31, -43}
, {-18, -40, 18}
, {23, -24, -48}
, {26, 4, -103}
, {41, 118, 132}
, {65, 108, 25}
, {-54, -48, 43}
, {-55, -12, 42}
, {-58, 1, 30}
, {-8, 21, 0}
, {10, 41, 40}
, {82, 43, -53}
, {56, -40, -43}
, {19, 43, -42}
, {20, -11, -23}
, {29, -61, -22}
}
, {{34, -76, -54}
, {57, -12, -75}
, {9, -12, -47}
, {48, 56, -48}
, {-5, 24, -43}
, {58, -44, -32}
, {35, -33, 30}
, {-78, -15, -2}
, {-16, 27, -16}
, {-15, -64, -60}
, {60, 19, -59}
, {36, -21, -12}
, {45, 18, -47}
, {5, 54, 54}
, {68, -39, -41}
, {-61, -77, 16}
, {-26, -84, -55}
, {-16, 31, 26}
, {9, 75, -53}
, {27, -40, 47}
, {-31, 54, 52}
, {11, 15, 138}
, {-27, 46, -50}
, {-2, 62, -26}
, {0, 31, -52}
, {82, -22, 44}
, {41, -15, -48}
, {48, -29, -28}
, {-44, -72, 31}
, {76, -41, 40}
, {-58, 84, 61}
, {-48, 53, -46}
}
, {{-20, -53, -56}
, {55, 78, -56}
, {-44, -60, 22}
, {-8, 75, -70}
, {-3, -49, -72}
, {54, -23, 10}
, {36, -82, 53}
, {-23, -2, 43}
, {-60, -47, 34}
, {-85, -65, 46}
, {15, 0, -13}
, {-67, -25, -52}
, {12, 56, 20}
, {69, -22, -96}
, {16, 70, 15}
, {1, 51, 12}
, {-87, 26, 4}
, {-63, -54, -84}
, {-32, -79, -46}
, {-43, 26, 30}
, {27, -28, 62}
, {-102, 92, 28}
, {-46, -8, -30}
, {30, 49, 59}
, {12, 3, -18}
, {-70, -56, -64}
, {-14, 7, 62}
, {-14, -6, -4}
, {9, 2, 58}
, {23, 43, -45}
, {34, 49, 0}
, {44, -16, 63}
}
, {{26, -30, 34}
, {20, 55, 13}
, {8, 50, -80}
, {-92, -26, -62}
, {57, 71, 14}
, {1, 38, 52}
, {-23, 4, 66}
, {-24, 72, -31}
, {49, 8, 8}
, {34, 33, -34}
, {7, 20, -76}
, {58, 12, 34}
, {-56, 47, -57}
, {54, -72, 50}
, {-46, -56, 1}
, {56, 49, 28}
, {-36, 49, -30}
, {-17, 20, -94}
, {-14, 12, -25}
, {31, 82, 0}
, {-11, -41, 5}
, {-30, 58, 124}
, {-12, 7, 40}
, {4, -20, -55}
, {21, 8, -48}
, {-32, 15, -49}
, {12, 18, 48}
, {69, -35, -28}
, {-54, 13, -48}
, {10, 23, -81}
, {-32, 41, 40}
, {-2, -25, 22}
}
, {{43, -54, 72}
, {-13, -83, -33}
, {74, -17, 12}
, {40, 22, -1}
, {-29, -9, -79}
, {0, -26, -3}
, {32, -71, -46}
, {34, -58, -44}
, {-50, -35, -67}
, {51, -18, 60}
, {93, -36, -83}
, {74, 2, 30}
, {-41, -48, 51}
, {-27, 25, -39}
, {6, -25, 55}
, {30, -68, 78}
, {-1, 78, -48}
, {41, 11, -64}
, {-51, -2, -34}
, {-74, 15, 41}
, {-55, 53, -116}
, {-38, 41, 38}
, {-53, 46, 40}
, {-61, -33, -87}
, {-42, 62, 85}
, {-30, -52, 24}
, {-62, 56, -49}
, {3, -16, 11}
, {-61, 43, 70}
, {15, 51, -30}
, {14, 57, -11}
, {62, -52, -57}
}
, {{30, 15, -81}
, {49, -32, 52}
, {-13, -36, 28}
, {-44, -10, -87}
, {61, 63, 17}
, {34, 23, -6}
, {64, 43, 5}
, {0, -37, -4}
, {20, -40, 56}
, {25, -50, 37}
, {30, 17, -35}
, {-52, -67, 26}
, {-53, 4, 13}
, {-51, 49, -77}
, {36, 78, 74}
, {3, -37, 17}
, {2, -57, 26}
, {-68, -67, -60}
, {39, -58, -76}
, {54, -68, 7}
, {37, -14, -95}
, {30, 91, 67}
, {47, -33, 59}
, {63, -51, 81}
, {-54, -29, 25}
, {-30, -34, 3}
, {57, -34, -26}
, {-40, 39, -46}
, {12, -39, -35}
, {55, 26, -6}
, {-39, 8, -19}
, {-5, 0, -7}
}
, {{48, 3, -6}
, {85, 21, 28}
, {-41, -32, 9}
, {-34, -15, -41}
, {-19, -34, 60}
, {76, 33, -21}
, {91, -88, 46}
, {7, 43, -73}
, {38, -19, 17}
, {26, -13, -18}
, {-58, -39, -8}
, {-7, -12, -51}
, {52, 25, 30}
, {-69, -66, -7}
, {75, 86, -48}
, {31, 73, -32}
, {-25, -12, 71}
, {-27, -73, -96}
, {47, -9, -80}
, {-28, 112, -7}
, {-1, -16, -43}
, {180, 21, -44}
, {-57, -13, 63}
, {1, 56, 53}
, {-49, 26, 21}
, {3, 28, -11}
, {-56, 16, 6}
, {64, -55, 69}
, {59, -5, -25}
, {-93, -49, -1}
, {-23, 35, -6}
, {-24, 20, 14}
}
, {{-19, -6, -17}
, {86, 35, 46}
, {-21, 25, 16}
, {-64, -23, -36}
, {81, 13, -16}
, {33, -25, -51}
, {71, 67, 63}
, {36, 71, -27}
, {-49, 39, -12}
, {-42, 45, 6}
, {18, 53, -36}
, {6, 30, 38}
, {-31, 46, -2}
, {-50, -8, -27}
, {-33, -67, 23}
, {22, -18, -48}
, {-2, 16, 40}
, {-23, -1, -28}
, {-6, -13, 14}
, {-69, -53, -19}
, {-40, -45, 10}
, {121, 61, -15}
, {13, -53, 61}
, {26, -41, 7}
, {-52, 94, -38}
, {-63, -31, 40}
, {-18, -38, 18}
, {-38, 69, -32}
, {30, -33, -9}
, {46, 2, 30}
, {52, 6, -48}
, {-56, -48, -50}
}
, {{16, -80, -14}
, {-6, -56, 63}
, {37, -8, -92}
, {42, -26, 31}
, {-6, -45, -73}
, {-26, 47, -35}
, {-44, 4, 21}
, {26, -42, 71}
, {40, -40, 2}
, {-45, 58, -45}
, {-36, 7, -43}
, {-43, 27, -50}
, {-49, 2, 9}
, {81, -4, -15}
, {42, 30, -67}
, {-32, 0, -40}
, {7, 42, -2}
, {52, -3, -34}
, {-60, 10, 22}
, {-17, -46, -51}
, {-104, 50, -3}
, {-30, -35, 21}
, {24, 19, -50}
, {37, -78, 25}
, {23, -30, 0}
, {8, 6, -14}
, {-45, -1, -71}
, {-64, 34, 68}
, {22, 56, 63}
, {17, -68, -38}
, {-32, 33, 27}
, {-60, -42, 17}
}
, {{-5, 52, 79}
, {22, -7, 31}
, {-58, 2, -91}
, {-64, -16, 36}
, {-46, 35, 22}
, {30, 72, -3}
, {-55, 1, -58}
, {37, -3, -44}
, {-8, 61, -25}
, {8, -49, 61}
, {-53, -39, -62}
, {38, -55, -12}
, {84, 56, -39}
, {-1, 36, -7}
, {-45, -26, 9}
, {-66, 32, 66}
, {-58, -42, -45}
, {-6, -37, 27}
, {-56, 24, 72}
, {7, -3, 70}
, {6, 61, 63}
, {52, 17, 77}
, {64, -9, 31}
, {49, -24, -66}
, {5, 37, 103}
, {28, -69, -40}
, {-8, 40, 43}
, {-6, -59, -49}
, {6, -36, 71}
, {19, 74, -20}
, {-47, 63, -66}
, {22, 63, 16}
}
, {{29, -13, 19}
, {-6, 87, -48}
, {-42, 9, 29}
, {-73, 54, 60}
, {-20, 36, -30}
, {16, -26, 11}
, {18, -59, 15}
, {66, -66, 29}
, {-27, -10, 0}
, {47, -61, 55}
, {-1, -60, -5}
, {0, 13, -20}
, {59, -70, -18}
, {50, 18, 13}
, {-14, 0, 58}
, {-3, 62, 35}
, {-64, 37, 13}
, {42, -25, -13}
, {-28, 53, -22}
, {-10, -55, -10}
, {-25, 19, 35}
, {3, -37, 118}
, {-34, 42, -9}
, {64, -66, -25}
, {-63, -85, -76}
, {-17, 14, -20}
, {38, -70, -11}
, {14, 55, -75}
, {67, 32, -53}
, {-60, -20, -28}
, {-9, 12, 46}
, {35, -64, -21}
}
, {{-16, 71, -18}
, {74, -37, -1}
, {7, -83, -25}
, {37, 41, -13}
, {50, -37, 45}
, {-2, 33, -6}
, {-10, -2, -25}
, {64, -30, -10}
, {-13, 16, -25}
, {18, -23, -30}
, {50, 46, -22}
, {-44, 12, -20}
, {-61, 5, -66}
, {-48, -20, -24}
, {54, -1, -68}
, {17, 53, 6}
, {-4, -18, 2}
, {1, -31, 43}
, {64, -56, -49}
, {-58, 8, -71}
, {5, -14, 37}
, {86, -57, 25}
, {35, 63, 59}
, {77, 5, -11}
, {-54, 4, -3}
, {-89, -56, 40}
, {25, -65, 64}
, {72, -55, 7}
, {56, 35, -8}
, {-52, 21, -5}
, {-68, -30, -76}
, {55, -39, -40}
}
, {{64, -33, -2}
, {-59, 43, 50}
, {-77, 6, 19}
, {-51, 11, 77}
, {23, 29, 32}
, {47, 9, 7}
, {14, 9, -13}
, {18, 35, 49}
, {28, -44, -12}
, {24, 44, 46}
, {-55, -62, -21}
, {39, -59, -33}
, {95, -16, 3}
, {-26, 33, -7}
, {-27, 24, -17}
, {49, -6, -7}
, {49, -46, -44}
, {-18, 37, -16}
, {83, -25, 0}
, {17, -8, -37}
, {115, 15, 27}
, {49, 20, 62}
, {-13, -23, 27}
, {-55, -26, -9}
, {-6, -33, -89}
, {-36, 30, -64}
, {-38, -44, -18}
, {25, 72, -63}
, {-20, 22, 38}
, {-7, 14, 43}
, {42, 46, 23}
, {-59, 43, -95}
}
, {{-5, -74, 6}
, {-24, 51, -34}
, {40, -71, -49}
, {49, 41, 61}
, {4, -60, 56}
, {-56, 16, -55}
, {-8, 55, -72}
, {-73, -42, 65}
, {-51, 15, 41}
, {9, 41, 27}
, {-59, -15, 29}
, {-57, 44, 8}
, {-29, 25, -41}
, {61, -26, -7}
, {17, -9, 15}
, {62, -2, 67}
, {3, -9, -9}
, {81, 75, 40}
, {-53, -34, -75}
, {67, 6, -62}
, {-16, 1, 22}
, {-47, 78, 78}
, {-67, -43, 62}
, {-43, -45, 39}
, {-63, 21, 50}
, {18, -61, -23}
, {40, -60, -68}
, {-21, -36, 72}
, {26, 10, -69}
, {-51, 64, -24}
, {-77, 52, -16}
, {-31, -45, -28}
}
, {{-23, 7, 52}
, {-3, -27, 62}
, {56, 26, 40}
, {58, -52, 69}
, {38, -56, 71}
, {29, 6, 45}
, {10, 13, -62}
, {25, 31, -69}
, {-65, -52, -3}
, {34, -4, 72}
, {-19, -87, -73}
, {79, 9, -15}
, {-27, -39, 47}
, {-36, -38, 18}
, {66, -50, -64}
, {50, -20, -61}
, {-48, 45, -37}
, {-55, 31, -20}
, {45, -30, 17}
, {-36, -58, -11}
, {-47, 30, 13}
, {-17, -40, 48}
, {38, 8, 57}
, {-33, -74, -22}
, {27, -2, 37}
, {-7, -29, 14}
, {-66, 50, 51}
, {-61, -49, 50}
, {-19, -42, 60}
, {-26, -103, -31}
, {19, -74, 79}
, {20, -39, -16}
}
, {{89, 67, -45}
, {-21, 88, 59}
, {57, -66, -25}
, {-84, 23, -54}
, {-1, -44, 20}
, {6, -63, 31}
, {-39, 9, 66}
, {54, 3, 31}
, {11, 50, -32}
, {32, 4, 64}
, {12, -80, -43}
, {43, -1, 29}
, {-46, 28, 14}
, {-31, 25, 53}
, {54, -36, 29}
, {-11, 0, -51}
, {-44, 34, -7}
, {27, -39, -83}
, {-74, -3, 0}
, {13, -18, 68}
, {26, -18, 34}
, {-145, 27, 20}
, {-25, 8, 14}
, {-47, 54, 4}
, {44, 8, -66}
, {-28, 22, -69}
, {19, -53, -21}
, {2, 63, 48}
, {-31, 9, 7}
, {-78, -39, -104}
, {21, -74, -13}
, {-3, 52, 47}
}
, {{43, -59, -51}
, {47, 40, -67}
, {58, -2, 17}
, {41, 41, 65}
, {-48, 9, 15}
, {31, 45, 30}
, {-9, -18, -27}
, {-58, 55, 37}
, {-45, -43, -52}
, {36, 28, -51}
, {-34, 52, 42}
, {-26, 59, 52}
, {-45, -22, -68}
, {-49, 22, -7}
, {-29, -9, 47}
, {57, -55, -74}
, {-72, 54, 72}
, {40, -70, 44}
, {-49, 59, -47}
, {65, -68, 39}
, {0, 60, -63}
, {-7, -63, 45}
, {7, -75, -64}
, {-26, -66, -55}
, {-2, 0, 8}
, {86, 8, 32}
, {-12, 71, -5}
, {12, -32, 24}
, {40, -48, -32}
, {-29, -33, 17}
, {-5, 4, 31}
, {-13, -8, 46}
}
, {{-10, -35, -63}
, {82, 0, -17}
, {-51, 51, 62}
, {-27, -8, -61}
, {-40, 90, 47}
, {-39, 16, 6}
, {77, 84, 43}
, {34, -46, 30}
, {31, 51, -41}
, {-38, -12, 37}
, {-84, 11, 56}
, {-37, 53, 8}
, {-10, 13, 68}
, {59, -61, 11}
, {-47, -25, 21}
, {52, 14, 30}
, {49, -7, 15}
, {-53, -71, -24}
, {13, -27, -78}
, {-19, 66, -26}
, {-25, -21, 23}
, {51, 69, 29}
, {43, 45, 40}
, {16, 70, -28}
, {-37, -55, -3}
, {28, -28, 41}
, {-64, 41, 49}
, {46, 2, 18}
, {-48, -66, -55}
, {-29, -44, 46}
, {-59, -78, -70}
, {57, -41, -64}
}
, {{-24, 15, 43}
, {70, 41, 16}
, {19, 4, -56}
, {-53, -64, 16}
, {54, -16, 32}
, {-39, -15, -55}
, {13, 48, -55}
, {9, -40, 43}
, {-24, -10, 0}
, {8, 25, 8}
, {67, 34, 84}
, {17, -9, -18}
, {18, -37, -47}
, {7, 29, -60}
, {-41, -45, -38}
, {-55, 40, 38}
, {55, -51, 49}
, {42, 67, 40}
, {53, 50, 14}
, {56, -25, 47}
, {49, 9, -26}
, {207, -31, 15}
, {-3, -57, -68}
, {70, -60, -18}
, {46, 13, 61}
, {90, 91, 27}
, {64, 8, -16}
, {25, 6, -13}
, {24, -2, 5}
, {-35, -27, 35}
, {67, -38, 40}
, {-20, 12, 55}
}
, {{-57, 19, -31}
, {-15, -48, -61}
, {-31, 54, 15}
, {-56, -74, -58}
, {-40, 0, 72}
, {-48, 79, -8}
, {25, -49, -80}
, {51, -25, -53}
, {54, 3, -46}
, {-69, -37, 49}
, {-73, 79, -18}
, {-19, 9, -44}
, {-31, 31, -39}
, {42, -46, 0}
, {-33, 16, 61}
, {76, -44, 63}
, {48, 68, 23}
, {-7, 49, -29}
, {-62, -48, 55}
, {24, 7, 59}
, {22, -62, 42}
, {57, 27, 3}
, {-84, -5, -26}
, {-57, -2, 11}
, {26, 17, -26}
, {-39, -18, 77}
, {25, 5, -13}
, {-60, 45, 16}
, {-61, -54, 38}
, {-52, 56, 81}
, {47, 29, -35}
, {-9, -29, 30}
}
, {{-46, -2, -22}
, {-73, 17, -11}
, {-72, 39, 64}
, {-27, 0, -65}
, {97, 62, -13}
, {26, -35, 38}
, {-36, 29, -43}
, {-54, -13, 74}
, {-28, -34, 52}
, {68, 55, -75}
, {5, -74, 11}
, {9, 9, -61}
, {-73, -35, -8}
, {30, -97, -19}
, {-81, -1, -21}
, {8, -20, 61}
, {19, -55, 68}
, {37, 23, -50}
, {-64, -11, 44}
, {-71, -44, -25}
, {35, -70, 70}
, {-9, 34, 8}
, {47, -34, -23}
, {-63, -12, -20}
, {-46, 57, 43}
, {-88, 41, -51}
, {29, -37, -47}
, {73, -26, 3}
, {28, -64, -35}
, {27, -68, -74}
, {3, -64, 10}
, {60, -17, 37}
}
, {{30, -28, 90}
, {-67, 77, -4}
, {7, -38, -1}
, {20, -2, -6}
, {9, -4, 49}
, {-12, 53, 8}
, {12, 65, 47}
, {-58, 9, 42}
, {-35, 49, -19}
, {64, 53, 47}
, {-67, -10, 53}
, {29, 25, -66}
, {37, -48, 44}
, {12, -29, -36}
, {-17, -44, -24}
, {13, 25, 1}
, {1, 49, -43}
, {-41, -78, -63}
, {-45, 0, 42}
, {-38, -28, -35}
, {72, -19, 9}
, {37, 147, 48}
, {-15, 72, 16}
, {38, -51, 19}
, {-26, -23, 0}
, {26, 8, -11}
, {-71, -15, 56}
, {72, 2, 39}
, {-2, 2, -33}
, {-14, -61, -62}
, {62, -38, -12}
, {30, -29, -4}
}
, {{-6, 54, 4}
, {89, 68, -14}
, {-33, 35, 38}
, {9, 89, 60}
, {-5, 3, -100}
, {-89, 4, 14}
, {29, 40, -20}
, {-15, -65, -54}
, {45, -26, -16}
, {-75, -77, 2}
, {-6, -56, 40}
, {-85, -46, 18}
, {-41, 46, 84}
, {-8, 49, -59}
, {-76, -22, -4}
, {-66, -25, 32}
, {8, -9, 42}
, {-20, -65, 30}
, {24, 24, -89}
, {-57, 3, 18}
, {42, -43, 27}
, {86, 21, 132}
, {-28, -29, 77}
, {20, 58, -17}
, {-26, -50, 35}
, {-6, -58, 56}
, {-18, -29, -26}
, {18, -33, 10}
, {-62, -47, 4}
, {14, -72, -15}
, {3, 79, 25}
, {-11, -54, 9}
}
, {{28, 74, 18}
, {-62, -30, 2}
, {-45, -15, 3}
, {-19, -52, 9}
, {-19, 25, -18}
, {-2, 26, -76}
, {72, 28, 10}
, {41, 16, -16}
, {-23, 39, 14}
, {35, 68, -48}
, {57, -24, -90}
, {-36, 11, 32}
, {23, -2, -17}
, {-2, -1, 16}
, {-3, -44, -55}
, {-41, 67, -64}
, {15, 68, 6}
, {-10, 22, 66}
, {0, -50, -71}
, {-41, 76, -51}
, {18, 64, 32}
, {158, 63, -38}
, {19, -41, -23}
, {-19, 30, 60}
, {53, 46, -13}
, {-23, -39, -58}
, {38, 50, -54}
, {1, -8, 82}
, {30, 62, 57}
, {33, -39, -73}
, {6, -38, 51}
, {-42, 7, 55}
}
, {{23, 24, 0}
, {35, -24, -27}
, {-47, 61, -8}
, {53, 14, -26}
, {41, -58, 8}
, {22, -42, 39}
, {40, -18, -65}
, {-26, 68, 43}
, {31, -4, -40}
, {-21, -82, -55}
, {-59, 14, -19}
, {-58, 83, -24}
, {3, 0, 58}
, {-32, 59, 18}
, {-59, 11, -14}
, {14, -31, -71}
, {-14, -38, 41}
, {-50, -41, -61}
, {-8, 56, -60}
, {-9, -6, -57}
, {-24, -32, -26}
, {-70, -48, 14}
, {-18, -29, -11}
, {-65, -57, 4}
, {-68, 65, 1}
, {-26, 65, 5}
, {-52, -35, -55}
, {53, -56, -53}
, {36, -27, -53}
, {9, -32, -53}
, {-44, -38, -52}
, {-38, -50, -51}
}
, {{-53, 67, 67}
, {70, 34, 58}
, {23, -65, -12}
, {27, -60, -86}
, {26, -67, -48}
, {27, -27, 31}
, {17, 37, 82}
, {-12, 60, 19}
, {-25, 14, 14}
, {30, -22, 30}
, {-2, -36, -39}
, {62, -44, -25}
, {46, 50, -61}
, {24, 53, 40}
, {0, 16, -47}
, {60, 34, -25}
, {37, -78, -14}
, {-56, 63, 72}
, {-39, 27, 74}
, {43, 24, 61}
, {52, 33, 63}
, {8, 30, 30}
, {-66, 18, 22}
, {48, -55, 74}
, {-20, 30, -25}
, {37, 21, 39}
, {-53, 49, 0}
, {-20, -8, 11}
, {38, 21, -50}
, {-26, -33, 32}
, {-20, 8, -5}
, {-46, 10, 28}
}
, {{-46, 10, 20}
, {69, 82, -50}
, {-4, 20, 45}
, {72, 105, 12}
, {-7, 32, 73}
, {-53, -70, 49}
, {-69, -2, 3}
, {-29, 9, -26}
, {-11, -32, 21}
, {-80, -52, 61}
, {-79, -50, 11}
, {-62, -60, -32}
, {14, -47, 82}
, {-20, 30, -46}
, {-81, 55, -7}
, {56, -77, -48}
, {-53, -92, -72}
, {71, -1, 20}
, {-12, -45, 2}
, {36, -56, -6}
, {16, 48, 129}
, {32, 103, 134}
, {-6, 22, 79}
, {10, -60, 46}
, {-44, -73, -28}
, {44, 32, 37}
, {3, 24, -53}
, {67, -46, -28}
, {-55, -48, -42}
, {59, 13, -25}
, {-54, 59, 16}
, {-81, 37, 2}
}
, {{-48, -29, 39}
, {-62, 0, 0}
, {15, 27, 29}
, {36, 27, -21}
, {38, -68, 32}
, {23, -84, 48}
, {5, 10, 26}
, {17, 52, -65}
, {24, -73, 11}
, {69, -9, -46}
, {69, 38, 30}
, {3, 31, 20}
, {40, -70, 0}
, {16, -5, 57}
, {-30, -37, 42}
, {2, -20, 13}
, {63, -60, -73}
, {-32, -28, 12}
, {-45, 45, -77}
, {-1, -54, -20}
, {-32, 50, -43}
, {-12, -48, 39}
, {-33, -33, -13}
, {-38, -88, 19}
, {-26, -36, -16}
, {0, -19, 34}
, {50, 39, 25}
, {31, 21, 6}
, {3, -21, -51}
, {48, -61, 55}
, {-72, 66, 23}
, {34, -85, -2}
}
, {{32, -9, 35}
, {8, 39, 47}
, {-40, -64, 45}
, {-18, -70, -9}
, {68, -83, -1}
, {-20, -63, -63}
, {-56, 48, -60}
, {-87, 32, 18}
, {11, 39, 24}
, {36, 2, -54}
, {-48, -16, -27}
, {96, -53, -2}
, {-66, -62, -16}
, {-44, 20, 93}
, {62, -85, -13}
, {-27, 47, 66}
, {60, 27, -44}
, {31, -9, 37}
, {-71, -6, -29}
, {2, -31, -64}
, {41, 83, 67}
, {144, 50, 54}
, {12, -20, -37}
, {-89, -26, -52}
, {37, 58, -103}
, {-11, -12, -69}
, {63, -2, 8}
, {-11, 51, 11}
, {-29, 91, 4}
, {-25, -77, 9}
, {-18, 26, 13}
, {-23, 71, -44}
}
, {{-96, -20, -53}
, {57, 88, 10}
, {-33, 52, 68}
, {20, -6, -4}
, {-44, -38, -16}
, {-14, -43, 30}
, {12, 71, 58}
, {6, 91, -41}
, {69, 100, 83}
, {40, 46, 20}
, {57, -48, 2}
, {-19, -72, -8}
, {10, -39, -41}
, {67, 44, 69}
, {-14, -31, -70}
, {-6, 19, 12}
, {-67, -28, -3}
, {46, 88, -14}
, {32, -47, 100}
, {-25, 9, -44}
, {32, -2, 49}
, {45, 88, -48}
, {-10, 9, 14}
, {57, 46, -53}
, {-61, -20, -29}
, {-43, 54, -39}
, {-52, -5, 70}
, {-32, 57, -11}
, {-32, -18, -70}
, {32, -13, 10}
, {13, -8, 72}
, {-74, -78, -60}
}
, {{61, -82, -70}
, {15, 55, -11}
, {38, -9, 47}
, {68, 14, 53}
, {-7, -7, -46}
, {-34, 27, 45}
, {-55, -59, -63}
, {-45, 66, -55}
, {82, -1, 74}
, {-53, -54, -75}
, {-18, -14, 15}
, {-63, 19, 15}
, {25, 13, 52}
, {-6, 14, 30}
, {42, -58, 44}
, {-57, 35, -20}
, {42, -58, -45}
, {30, -39, 37}
, {66, 63, 39}
, {-14, 34, 28}
, {-8, -1, -78}
, {-46, 16, -25}
, {69, -22, 24}
, {2, 28, -31}
, {9, -63, 39}
, {75, 66, -30}
, {27, -14, 73}
, {-61, -11, 45}
, {-40, -23, 29}
, {75, 67, 60}
, {-52, -34, -47}
, {-26, -3, 14}
}
, {{13, 26, -41}
, {-2, -61, 58}
, {-67, 50, -35}
, {44, -25, -11}
, {29, -67, -33}
, {-66, -10, 26}
, {-73, -8, 33}
, {10, -33, 43}
, {-39, 17, 40}
, {54, 15, -82}
, {37, -74, -83}
, {61, -35, -21}
, {-56, -58, 40}
, {48, 14, -71}
, {6, 64, 6}
, {17, 59, -41}
, {0, 11, -12}
, {-14, -68, 52}
, {-72, -10, 14}
, {-77, -53, -33}
, {-5, -46, 11}
, {79, 34, -46}
, {26, -43, -54}
, {-74, -51, -61}
, {-38, -11, 53}
, {24, -31, 24}
, {39, 69, 3}
, {27, -48, -55}
, {57, 23, -10}
, {-27, 37, -36}
, {-56, -41, 5}
, {-46, 47, 36}
}
, {{-7, -5, 31}
, {1, 61, 20}
, {38, 68, 73}
, {24, 30, 49}
, {91, -64, -28}
, {-43, -35, 37}
, {-28, 19, -20}
, {18, -30, -39}
, {6, 56, 48}
, {59, -3, 35}
, {-2, 49, 62}
, {27, -26, -27}
, {-81, -31, -29}
, {-63, -57, -18}
, {-30, 40, 46}
, {-55, -56, 13}
, {23, -53, -59}
, {5, -3, 15}
, {-3, 9, 41}
, {3, -69, 36}
, {-27, -28, -29}
, {9, 33, -91}
, {20, 26, 5}
, {30, 57, -33}
, {-27, -38, 46}
, {59, -17, 41}
, {-60, 50, -42}
, {10, -43, -7}
, {-4, 62, 38}
, {18, -72, 41}
, {-56, -11, 47}
, {8, -10, -59}
}
, {{-85, -16, -33}
, {30, 33, 38}
, {17, -6, 8}
, {15, 24, -42}
, {28, -62, -9}
, {-20, 90, -21}
, {-78, -20, -6}
, {19, -1, 38}
, {-2, -17, -102}
, {32, 84, -27}
, {-48, 39, -15}
, {6, 36, 70}
, {-74, 44, -97}
, {-4, 37, -9}
, {-59, -50, 14}
, {-57, -19, 72}
, {-11, 61, 21}
, {33, 46, 71}
, {-51, -31, -34}
, {-29, -45, 14}
, {-2, 71, -29}
, {-15, 0, 85}
, {-74, 66, -48}
, {-51, -75, 12}
, {-18, 92, 2}
, {-53, -50, 6}
, {-18, 1, 40}
, {-58, 23, -36}
, {-23, 57, -36}
, {44, 11, 78}
, {68, -60, 85}
, {-58, 40, -9}
}
, {{-30, -47, -46}
, {-17, 33, 41}
, {36, -45, 34}
, {-19, -65, -34}
, {-33, -43, 61}
, {2, 71, -53}
, {-32, -12, 52}
, {4, -2, -29}
, {71, -55, 76}
, {30, -15, 39}
, {26, -40, 9}
, {-78, 17, -35}
, {38, 35, 35}
, {-74, -52, 42}
, {-54, 54, 9}
, {-17, -36, -60}
, {-51, -25, 65}
, {-20, 48, 55}
, {57, 45, 75}
, {13, 56, 35}
, {-39, 44, 60}
, {79, -44, -53}
, {0, -68, 21}
, {43, -59, -38}
, {36, -65, -25}
, {-45, -15, 79}
, {54, 25, -11}
, {-12, 15, -37}
, {-46, -20, 22}
, {55, 48, 53}
, {38, -2, -64}
, {-40, 28, 14}
}
, {{-8, 29, 12}
, {82, 54, 104}
, {68, -65, 7}
, {61, 38, -59}
, {79, 8, 80}
, {-2, 9, 9}
, {71, -26, 42}
, {45, -47, -47}
, {10, 26, 72}
, {73, -62, -7}
, {2, -43, -89}
, {16, 51, -47}
, {7, -5, -1}
, {54, 12, -1}
, {35, 14, -66}
, {-21, -68, 53}
, {-81, 69, 25}
, {-23, 11, -61}
, {48, 47, 63}
, {26, 11, -65}
, {-51, -93, -73}
, {-17, 14, 68}
, {-33, 52, -56}
, {-20, -24, -83}
, {40, -8, -12}
, {44, -22, 27}
, {-8, 71, -23}
, {74, -34, 61}
, {-68, -63, -42}
, {-12, 17, -39}
, {68, -50, -7}
, {-45, -10, 25}
}
, {{-13, 34, 5}
, {-22, -49, 46}
, {84, 62, -63}
, {60, 64, 25}
, {29, -54, -18}
, {-51, -7, 71}
, {-71, 15, -23}
, {-48, -57, 37}
, {7, 20, -13}
, {-77, 31, 26}
, {67, 55, -61}
, {-67, 21, 93}
, {-87, -10, -17}
, {17, -56, -66}
, {1, 14, 78}
, {56, 12, 56}
, {-62, -13, -2}
, {-71, -20, -86}
, {50, -30, -5}
, {-61, -4, 29}
, {-9, -2, 43}
, {106, -38, 111}
, {-10, 61, -55}
, {-39, -5, 33}
, {18, -24, 58}
, {-42, -21, 23}
, {34, 13, 5}
, {4, 20, 39}
, {-65, -98, -35}
, {69, 46, -57}
, {78, -24, -49}
, {-61, 4, 31}
}
, {{-29, -72, 62}
, {-47, -79, 14}
, {-21, 28, 20}
, {-16, 65, 67}
, {32, -53, -79}
, {25, -24, 67}
, {19, 0, 37}
, {-84, 40, 17}
, {-26, -1, -19}
, {-33, -37, 29}
, {-57, 93, 17}
, {23, 36, 42}
, {10, -47, 53}
, {43, -37, -27}
, {-11, -7, 27}
, {-38, 36, -62}
, {79, -35, 55}
, {-40, 28, -16}
, {58, 10, -71}
, {62, 21, -33}
, {-25, 141, 162}
, {86, 123, 90}
, {-50, -39, 0}
, {21, 9, 38}
, {64, -33, -4}
, {-6, 102, 26}
, {-66, 23, 24}
, {-67, 12, 36}
, {7, -43, 81}
, {-6, 5, 17}
, {-43, -10, -72}
, {-19, -60, 29}
}
, {{-22, 2, -51}
, {-86, -29, 66}
, {53, -80, -50}
, {23, -18, -64}
, {-8, 23, -79}
, {0, -4, -50}
, {-15, -10, 7}
, {-75, 33, 31}
, {-41, 29, 8}
, {49, 55, 36}
, {12, 15, -63}
, {-34, 74, 41}
, {-6, 12, 17}
, {-73, -67, 49}
, {16, -51, -40}
, {-1, -68, 66}
, {-4, 60, -19}
, {-27, -23, 54}
, {-50, -16, 5}
, {4, -53, 65}
, {121, 72, 17}
, {61, 0, 45}
, {61, 3, -55}
, {24, 20, -71}
, {-15, 77, 29}
, {15, 41, -1}
, {-35, -34, -72}
, {-105, -75, -15}
, {-64, 72, 80}
, {-42, -24, -9}
, {72, -63, 13}
, {-17, 62, 24}
}
, {{5, 42, 45}
, {-49, -11, 60}
, {-43, 11, 42}
, {-6, -35, -4}
, {-25, -78, -34}
, {-25, -54, 53}
, {14, 27, -30}
, {-2, 33, -13}
, {-63, -24, -21}
, {33, 37, 29}
, {-26, -38, 22}
, {-35, -8, 23}
, {41, 52, -17}
, {60, -79, 48}
, {37, -68, -74}
, {-1, 62, -50}
, {-43, 21, -53}
, {-89, 67, 38}
, {-25, 57, -24}
, {-24, -23, 43}
, {6, 28, -20}
, {-72, -12, 27}
, {15, -13, -39}
, {-6, -74, -20}
, {-42, 73, -13}
, {-38, -53, 7}
, {-60, -8, 2}
, {8, 28, -16}
, {-50, -27, -14}
, {-74, -29, -20}
, {54, 22, 17}
, {35, -6, 51}
}
, {{10, 10, 25}
, {66, 35, 20}
, {-56, -25, -76}
, {-84, -69, 28}
, {29, 25, -54}
, {18, -57, 30}
, {80, 73, 16}
, {42, -86, -55}
, {81, -92, -24}
, {81, -36, 1}
, {-11, 28, -18}
, {-27, -58, -23}
, {34, -37, 82}
, {-44, 83, 0}
, {-31, -28, 28}
, {-56, 51, 60}
, {-30, 0, -24}
, {4, -5, 56}
, {38, -1, -16}
, {-23, -68, 62}
, {128, 38, 3}
, {41, 63, 41}
, {-89, -60, -4}
, {53, -34, -21}
, {15, -56, -33}
, {58, -60, 50}
, {44, 33, 28}
, {57, 58, 93}
, {-36, -15, 56}
, {26, 5, -11}
, {-28, -34, 40}
, {5, -11, -86}
}
, {{-96, -44, 36}
, {-25, 2, -59}
, {46, 68, 4}
, {77, -70, -62}
, {38, 69, -70}
, {62, -68, -35}
, {47, -47, -27}
, {25, -2, 33}
, {5, 37, -45}
, {48, 29, -5}
, {-24, 32, -68}
, {-38, 37, -45}
, {11, 53, -36}
, {-12, 26, -52}
, {-46, 63, -3}
, {-6, 59, 49}
, {11, 38, 42}
, {-23, -68, -37}
, {-68, -51, -1}
, {-35, -28, -46}
, {36, 40, -4}
, {56, 10, -32}
, {63, -32, -28}
, {-24, 44, -8}
, {-58, -38, 1}
, {4, 75, 9}
, {54, -49, -73}
, {38, -34, 1}
, {40, 34, 30}
, {-2, -76, -61}
, {49, 36, -25}
, {-23, 82, -19}
}
, {{1, -32, -45}
, {-52, -19, -21}
, {-44, 35, -21}
, {-3, 4, -12}
, {-53, -40, 14}
, {-25, 20, 20}
, {13, -79, 6}
, {-59, 35, -42}
, {88, 3, 34}
, {-31, -55, -18}
, {65, -65, 55}
, {74, -57, 26}
, {5, 107, 8}
, {84, 1, 10}
, {77, 39, 32}
, {-52, 62, -18}
, {54, 15, -72}
, {17, -59, -76}
, {83, 30, 20}
, {6, 71, -8}
, {-27, 50, -74}
, {-16, -64, 64}
, {-31, -36, -50}
, {-16, -34, 3}
, {43, -46, -31}
, {88, 30, -11}
, {-53, 45, 6}
, {8, -36, 63}
, {-9, 83, -11}
, {-33, 67, -46}
, {-54, 44, -70}
, {-68, -21, 24}
}
, {{-40, 51, 76}
, {41, -21, -12}
, {-5, 49, -64}
, {-58, -25, -63}
, {53, -25, -78}
, {-39, 13, 18}
, {-25, 56, -21}
, {-48, 86, 13}
, {3, 22, 39}
, {-72, -49, -24}
, {30, -52, -48}
, {11, 11, -70}
, {10, -3, 69}
, {56, -4, 36}
, {-63, 24, -51}
, {-1, 3, -32}
, {-31, -80, 40}
, {65, 25, 56}
, {35, -57, -1}
, {6, 64, -5}
, {5, 4, 80}
, {-19, 91, 46}
, {-28, -57, 23}
, {-56, 40, 74}
, {22, 55, -44}
, {104, -29, 40}
, {7, 11, 17}
, {-43, 61, 7}
, {-67, -42, 52}
, {81, 24, -30}
, {27, 48, -73}
, {51, -22, 7}
}
, {{21, 0, -48}
, {89, 40, 49}
, {-49, -29, 20}
, {1, 33, -50}
, {-44, 106, -12}
, {69, 88, -49}
, {12, 85, 72}
, {-42, 72, 2}
, {44, 14, -42}
, {-17, -14, -28}
, {-60, 65, 65}
, {80, -11, -35}
, {31, -19, -32}
, {-59, -53, -60}
, {35, -59, 36}
, {56, -23, -38}
, {8, 70, -52}
, {-3, -48, 24}
, {-14, -2, -28}
, {-36, -30, 85}
, {-84, 21, 39}
, {65, 30, -31}
, {33, 53, -34}
, {8, -65, 100}
, {-54, -34, 32}
, {52, 70, -33}
, {69, 69, -4}
, {-54, 62, -9}
, {12, -65, -59}
, {-29, 45, 60}
, {-11, 77, 38}
, {-35, -34, -45}
}
, {{-14, 26, 22}
, {-57, -67, -87}
, {65, -46, 11}
, {-36, -91, -84}
, {-28, 1, -30}
, {67, 8, -70}
, {-41, -31, 0}
, {49, 3, -23}
, {50, 10, 48}
, {-14, 31, -40}
, {45, 6, -15}
, {47, 30, -28}
, {59, 74, -7}
, {-100, 50, -32}
, {51, 41, -43}
, {-9, -9, -11}
, {50, 56, 30}
, {53, 53, 30}
, {55, 23, 47}
, {46, 64, 31}
, {47, -9, 0}
, {-36, 88, 150}
, {-23, 41, -55}
, {-38, 28, -83}
, {37, 72, 72}
, {59, 24, -22}
, {69, -4, -11}
, {0, -53, -41}
, {20, -4, 64}
, {6, 24, 60}
, {-57, 10, -14}
, {67, 10, 34}
}
, {{-45, -69, -21}
, {30, -47, 37}
, {16, 51, -18}
, {19, 21, 0}
, {27, -17, -51}
, {-14, -78, -1}
, {51, 45, 9}
, {53, -9, 56}
, {28, 55, 10}
, {-46, -74, 24}
, {14, 22, 15}
, {-44, -24, -19}
, {-25, 3, 63}
, {-50, -61, -28}
, {-8, -62, 40}
, {-15, -38, -27}
, {52, 30, -80}
, {27, -41, 57}
, {57, 32, 35}
, {-19, -25, 28}
, {41, 58, -75}
, {28, 62, 39}
, {-48, 19, 63}
, {-20, -78, -76}
, {4, -57, 12}
, {-74, -58, 39}
, {-67, 21, -31}
, {-55, 51, 46}
, {-24, -61, 69}
, {-23, -70, 41}
, {26, -9, -71}
, {15, 44, 29}
}
, {{-12, 81, -13}
, {34, -25, -69}
, {-59, 15, -63}
, {9, -19, 16}
, {-16, 19, -4}
, {-59, -68, -97}
, {-50, -26, 48}
, {23, 13, 57}
, {59, 5, 31}
, {63, 63, 42}
, {42, 37, 80}
, {-52, 70, -30}
, {-9, -89, -9}
, {86, 7, -55}
, {20, 69, -8}
, {-52, 16, -61}
, {-56, -8, -34}
, {29, 1, -23}
, {-29, -30, 32}
, {-2, -26, -58}
, {-61, 46, -52}
, {-53, 21, -9}
, {-68, 14, 47}
, {-57, 71, -85}
, {66, 2, 20}
, {42, 0, 44}
, {-59, 70, 73}
, {79, 62, 6}
, {-9, 4, 35}
, {-66, 43, -55}
, {1, -70, -35}
, {63, -34, 0}
}
, {{-74, -47, -15}
, {-28, 25, -5}
, {42, 23, 60}
, {39, 19, 73}
, {64, -36, -3}
, {22, 39, -72}
, {-14, 44, 38}
, {61, 32, -79}
, {-76, 9, -47}
, {-26, 45, -49}
, {82, 66, 6}
, {-15, -3, -55}
, {-13, -48, 10}
, {-55, -42, 63}
, {18, -19, -50}
, {15, 53, 75}
, {-51, -28, -12}
, {-35, 36, -34}
, {-70, -67, 17}
, {-66, 14, -75}
, {-62, 54, -23}
, {-21, 11, -71}
, {6, -38, 13}
, {39, -36, 11}
, {63, -18, -17}
, {54, 35, -52}
, {-45, 47, -32}
, {-43, -7, 58}
, {-18, 45, 0}
, {63, -47, 26}
, {2, -41, 76}
, {66, -20, 7}
}
, {{18, 96, 7}
, {-38, -3, -25}
, {-5, 51, 31}
, {-47, 31, 0}
, {48, 38, -49}
, {52, -29, -1}
, {-26, 20, 0}
, {-61, -20, 81}
, {-56, 2, -3}
, {-37, 18, -33}
, {30, -35, -46}
, {-37, 14, 27}
, {31, 36, 58}
, {-43, 24, -45}
, {16, 58, 52}
, {43, 60, -61}
, {82, 11, 52}
, {-44, 2, -34}
, {-19, 37, -47}
, {-12, 55, 71}
, {-22, -64, -14}
, {23, 52, -53}
, {-16, -73, -14}
, {-46, 50, -21}
, {-53, 19, -58}
, {69, 18, -15}
, {40, 20, 43}
, {39, 29, -16}
, {-16, -38, 7}
, {4, 23, -20}
, {-54, -20, 5}
, {-51, 34, 46}
}
, {{30, -90, 52}
, {54, -12, -48}
, {-42, 64, -44}
, {-29, -53, -55}
, {-33, -2, 5}
, {34, 29, -9}
, {51, 44, -45}
, {58, -59, 5}
, {-10, 37, -67}
, {-3, 11, -14}
, {48, -40, 59}
, {41, -39, -53}
, {-56, -19, -48}
, {-14, -60, 69}
, {19, -32, -19}
, {-30, -21, -2}
, {-43, 33, 1}
, {56, 14, -42}
, {52, 79, -60}
, {-28, -34, 6}
, {28, -81, -16}
, {134, 41, 24}
, {-59, 67, 34}
, {-31, -48, 8}
, {3, 85, 52}
, {-22, 27, 17}
, {0, 34, -53}
, {-71, -21, -27}
, {35, 63, 80}
, {17, 8, -57}
, {42, 76, -46}
, {72, -14, -28}
}
, {{63, -13, -20}
, {-91, -62, 26}
, {46, 1, 8}
, {95, -18, 39}
, {-7, -75, -41}
, {-8, -14, -61}
, {-21, -62, -84}
, {-69, 8, -67}
, {-43, 57, 64}
, {-72, -53, -45}
, {21, -49, -30}
, {-71, 34, 85}
, {65, -52, 84}
, {35, -27, 48}
, {-40, 0, 28}
, {46, -10, -9}
, {-53, -68, -18}
, {-48, -34, 49}
, {-23, -70, -45}
, {-25, 42, 49}
, {-51, -28, -17}
, {16, -1, 54}
, {70, 63, -39}
, {46, -9, 27}
, {-31, -5, 16}
, {70, -1, 27}
, {43, 53, -2}
, {-4, 78, 56}
, {6, -75, 87}
, {-13, -31, -28}
, {80, -33, 45}
, {62, -56, -41}
}
, {{-63, -70, -4}
, {0, -78, 50}
, {-31, -52, -4}
, {34, 47, 47}
, {53, -4, 62}
, {-24, -29, 26}
, {-51, 13, -60}
, {0, 1, -67}
, {9, -44, 44}
, {-11, 15, 1}
, {-64, 12, 71}
, {-54, -23, -6}
, {49, -25, 57}
, {56, -30, 56}
, {42, 71, 0}
, {-40, 55, -11}
, {-38, -6, 67}
, {4, -59, -21}
, {-52, -46, 73}
, {17, -46, 21}
, {-76, -11, 0}
, {32, 19, -16}
, {11, -28, -50}
, {57, 22, 50}
, {-3, 34, -37}
, {70, -45, 62}
, {57, 24, 64}
, {-42, 59, -35}
, {13, 30, 58}
, {67, 42, 4}
, {-69, 40, 65}
, {23, 39, -3}
}
, {{-49, -10, 33}
, {-19, 22, 63}
, {-29, 11, -96}
, {-21, -24, -66}
, {78, 5, -13}
, {-89, -59, 61}
, {-74, 41, 17}
, {-77, -29, 9}
, {-3, -36, -17}
, {-67, -78, -11}
, {63, 44, -66}
, {13, -37, -29}
, {-17, 33, -61}
, {1, -66, -29}
, {-42, -61, 5}
, {-33, 10, -42}
, {51, -52, 61}
, {47, 44, 9}
, {-28, -77, -63}
, {34, 55, -59}
, {-26, -74, -58}
, {96, 57, -70}
, {-63, 20, -48}
, {-34, 32, 20}
, {1, 18, 66}
, {35, -61, -4}
, {71, 57, -36}
, {-33, -29, -59}
, {24, 32, -18}
, {-17, 63, 48}
, {61, 69, 23}
, {-62, -78, -62}
}
, {{44, -8, 21}
, {-2, 18, 53}
, {-33, 32, 82}
, {-51, -32, -13}
, {12, -32, 33}
, {-72, 39, 53}
, {7, 42, -3}
, {24, 24, -13}
, {-71, -55, -80}
, {-27, 56, -54}
, {-27, 65, -55}
, {40, -26, -76}
, {-16, -21, -40}
, {18, 79, 62}
, {-59, -67, 44}
, {75, -9, 52}
, {59, -14, -28}
, {-1, -29, 29}
, {45, -67, -9}
, {2, -12, -7}
, {76, 29, -23}
, {132, 131, 66}
, {10, -35, 21}
, {-58, 37, 56}
, {40, -32, 47}
, {64, -72, 30}
, {34, -48, -4}
, {-40, -20, -65}
, {13, 71, 38}
, {-56, 1, 0}
, {-84, 37, -32}
, {-12, 29, -17}
}
, {{25, -37, 54}
, {14, -20, -72}
, {-57, 13, -58}
, {2, 57, 24}
, {57, -35, 26}
, {66, -63, 30}
, {0, -7, 51}
, {-9, 67, -27}
, {-28, -80, 38}
, {-2, -59, 32}
, {59, -48, 1}
, {-73, 42, -63}
, {-37, 29, 74}
, {63, -35, -69}
, {34, -18, 64}
, {20, 45, 41}
, {-55, 61, -33}
, {-65, -33, 39}
, {33, -71, -4}
, {28, -55, -38}
, {116, 0, 0}
, {-55, -82, -78}
, {10, 51, 29}
, {42, 58, 56}
, {-61, -74, 5}
, {-64, -49, 11}
, {67, -1, 55}
, {62, -13, 39}
, {43, 42, -61}
, {25, -72, 15}
, {13, 10, -78}
, {34, -27, 35}
}
, {{-41, -65, -31}
, {0, -33, 43}
, {-18, 39, 1}
, {31, -64, 15}
, {-59, -10, 21}
, {25, -68, -45}
, {-28, 72, 15}
, {50, -29, -8}
, {-48, -69, 7}
, {-55, 74, -67}
, {15, -71, -41}
, {74, 35, -73}
, {-6, -64, -25}
, {-59, 12, -28}
, {34, -2, 50}
, {-78, 6, 16}
, {-37, 63, -74}
, {-49, -58, 69}
, {12, -16, -61}
, {46, 60, 37}
, {-71, 23, -59}
, {-47, -19, -47}
, {-61, -71, -88}
, {-66, 46, 32}
, {-42, -17, -29}
, {-27, 37, -71}
, {67, 66, 46}
, {-16, 67, 57}
, {36, 21, 57}
, {-14, 13, -13}
, {-42, -88, 60}
, {54, 5, -12}
}
, {{-25, -57, -60}
, {17, -15, 22}
, {63, -92, 26}
, {-52, -27, -65}
, {56, 10, 30}
, {-12, 77, -72}
, {72, -51, 62}
, {-36, -29, -55}
, {-23, -50, 50}
, {-37, 36, -81}
, {-16, -57, -42}
, {13, -77, -54}
, {73, 49, -37}
, {-60, -51, 37}
, {47, 31, 64}
, {-9, -67, 23}
, {-20, -38, -60}
, {14, 29, 58}
, {75, 37, -41}
, {-9, -58, 10}
, {63, 17, -84}
, {-74, 16, 20}
, {-19, -29, 54}
, {-49, 33, 57}
, {0, -34, 34}
, {29, -34, 20}
, {-1, 1, 39}
, {6, -31, 27}
, {-13, -19, -49}
, {-1, -62, -72}
, {43, -81, 17}
, {67, 12, 21}
}
, {{-41, -14, 4}
, {36, -71, 10}
, {-59, -2, 68}
, {-35, 63, -2}
, {66, 21, -37}
, {10, 57, 28}
, {-7, -30, -15}
, {65, -5, -26}
, {42, 51, -80}
, {67, 21, -41}
, {68, 44, 59}
, {-42, 20, 0}
, {-94, 34, -52}
, {9, -56, -5}
, {40, -76, 42}
, {6, -60, -11}
, {-19, -18, -56}
, {-55, 7, -39}
, {71, 71, -45}
, {-12, -13, -42}
, {-76, -47, 3}
, {-18, -103, 45}
, {-25, -66, -73}
, {38, -29, -13}
, {-27, -32, 62}
, {13, 80, 35}
, {-62, -11, -46}
, {67, 1, -52}
, {-34, 88, 24}
, {-5, 76, 20}
, {66, 50, 18}
, {13, 50, 76}
}
, {{-6, -18, -17}
, {-3, 2, -42}
, {92, 95, -30}
, {51, 9, 30}
, {-74, -41, 9}
, {-63, 35, 54}
, {39, -57, -36}
, {1, 2, 46}
, {-61, 66, 45}
, {41, 31, -21}
, {-22, -1, 34}
, {78, 10, 0}
, {-39, -43, 48}
, {72, 23, -49}
, {34, 44, 68}
, {-44, 52, -60}
, {-15, -41, 24}
, {67, -9, 7}
, {11, 73, 37}
, {36, 27, -20}
, {-24, 34, 128}
, {177, -25, 104}
, {-9, -21, 4}
, {-50, 3, -53}
, {76, 95, 14}
, {104, 71, -25}
, {-16, 55, 48}
, {-19, -61, -61}
, {-30, -27, 28}
, {30, -19, 75}
, {70, 31, 67}
, {-65, -47, -49}
}
, {{52, 67, -77}
, {-82, -19, 11}
, {-48, -56, -36}
, {48, -40, 5}
, {-60, 65, 44}
, {24, 26, 57}
, {-19, 0, 54}
, {0, -62, 59}
, {12, -71, 45}
, {16, 31, 49}
, {81, -57, -58}
, {-43, 9, -10}
, {20, 32, -27}
, {-63, -23, -17}
, {34, 2, 38}
, {-21, 30, 38}
, {37, -15, 2}
, {1, 35, 33}
, {14, -43, 28}
, {-46, 46, 11}
, {91, 2, 110}
, {39, 95, 153}
, {32, -84, -49}
, {-108, -9, -9}
, {92, 16, 53}
, {-35, -44, -39}
, {19, 72, -5}
, {-86, 45, -49}
, {58, 57, 13}
, {85, 9, -68}
, {-68, -42, -63}
, {46, -25, 51}
}
, {{62, -43, -48}
, {24, -39, -14}
, {-4, -68, 35}
, {-11, -45, 36}
, {46, 41, 25}
, {-30, -35, 53}
, {20, 37, 49}
, {29, -35, 30}
, {60, 34, 59}
, {57, -3, 58}
, {-49, 38, -78}
, {30, -73, 9}
, {56, 47, -7}
, {-21, 10, -50}
, {-27, -38, 7}
, {31, 63, 15}
, {-41, -26, -2}
, {-58, -19, -82}
, {17, -69, 28}
, {-72, 71, -45}
, {63, 23, 21}
, {-14, 81, -2}
, {24, 27, 44}
, {-50, 5, -35}
, {36, -71, -42}
, {-50, -75, -32}
, {37, -26, 14}
, {-68, -27, -77}
, {38, 59, -48}
, {-23, 49, 4}
, {-16, 10, -41}
, {58, 22, 2}
}
, {{-2, 8, 27}
, {-16, 67, 30}
, {-22, -79, 63}
, {-54, -6, -25}
, {32, -44, -20}
, {29, -28, -35}
, {-34, -17, -18}
, {-24, 32, -46}
, {35, 23, -25}
, {52, 76, -5}
, {70, -30, -43}
, {0, -33, 26}
, {52, 52, -31}
, {-82, 59, 50}
, {14, -38, 46}
, {-4, -4, 32}
, {-1, 56, 19}
, {-30, -65, -54}
, {26, -20, -30}
, {72, 50, 18}
, {69, 45, 88}
, {117, -19, -75}
, {66, -66, 7}
, {0, -42, -67}
, {-12, 40, -8}
, {-80, -58, -33}
, {-51, -53, -18}
, {-12, 3, -11}
, {-27, 71, -12}
, {-78, -59, 55}
, {46, -58, -13}
, {77, 28, -84}
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

typedef number_t max_pooling1d_43_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_43(
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

typedef number_t conv1d_35_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_35(
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


const int16_t conv1d_35_bias[CONV_FILTERS] = {4, 28, 6, -12, -3, -3, -3, -10, 27, 18, 2, -7, -4, -4, 24, -8, 11, 8, 8, 24, -7, -6, -6, -11, -7, -6, 24, -22, 18, 15, -1, 22, -14, -8, -5, 3, 22, -5, 4, 21, -11, -7, -11, 34, 30, -1, 7, -5, 9, -14, -5, 20, 10, -9, 0, -2, 7, -11, -19, 4, -6, -7, -15, -1, 2, 19, 0, 26, 11, 29, 25, 3, 10, 34, -14, -3, -3, -3, -1, -4, -6, -2, -9, 3, 12, -9, 19, -2, 22, 6, 5, -11, 10, -7, -10, 10, -16, -3, -7, 28, -8, 5, -6, 4, -12, -20, -8, 2, -13, 37, -8, 9, -2, 20, 6, 7, 2, -7, -5, -16, -6, 11, 10, -5, -9, 4, 8, -2}
;

const int16_t conv1d_35_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{27, -6, 49}
, {-72, -60, 21}
, {-21, -20, 2}
, {11, -3, 47}
, {4, -22, 49}
, {-9, 46, 10}
, {6, -7, 57}
, {46, 3, 90}
, {-29, -3, -4}
, {-14, 48, -44}
, {18, -26, 36}
, {-3, 17, 7}
, {38, 25, -27}
, {-18, -20, -12}
, {-1, 52, 22}
, {44, -5, -35}
, {13, 32, 57}
, {-16, 9, -31}
, {-18, -22, 27}
, {-3, 46, -10}
, {-12, 7, -20}
, {-36, 4, 50}
, {28, 21, 55}
, {9, -47, -12}
, {-48, 13, 34}
, {-12, 9, 19}
, {58, 29, 50}
, {-1, -3, -66}
, {6, -57, -12}
, {-26, -40, -2}
, {48, 18, 36}
, {30, 37, -33}
, {-2, -11, -41}
, {24, -29, -38}
, {-43, -12, -44}
, {52, -1, 41}
, {-1, 43, 14}
, {-48, 4, -11}
, {-29, -2, -35}
, {-63, -41, -43}
, {-8, -37, -13}
, {-12, -25, -42}
, {-9, -47, -39}
, {-14, -17, -39}
, {5, 55, -33}
, {-9, 45, -14}
, {-40, -21, -23}
, {9, 6, 40}
, {-18, 38, 46}
, {1, -32, -7}
, {8, 42, 9}
, {25, -40, -12}
, {61, -19, -33}
, {53, 5, 30}
, {-11, 42, 5}
, {-47, -24, 23}
, {-5, -29, 12}
, {-47, 6, -23}
, {-14, 28, 46}
, {-28, 16, 30}
, {-4, 54, -10}
, {11, 0, 28}
, {-12, -44, 8}
, {-45, 27, 28}
}
, {{47, 28, -51}
, {47, -48, 10}
, {15, -24, 32}
, {-32, -26, -67}
, {-19, -6, -36}
, {27, 35, 3}
, {-19, -24, -15}
, {33, -27, 16}
, {-2, 21, -21}
, {37, 12, -27}
, {75, -22, 56}
, {-5, 1, -42}
, {15, -45, 40}
, {-10, 53, 41}
, {2, -40, 18}
, {-60, 29, 12}
, {-39, -40, -40}
, {47, -26, 47}
, {42, 36, 18}
, {12, 31, 7}
, {48, 22, 19}
, {-13, -38, 60}
, {9, 3, -12}
, {13, 9, -2}
, {41, -18, 37}
, {0, 51, -49}
, {59, 31, 41}
, {25, 78, 19}
, {6, 28, -28}
, {15, -65, -8}
, {24, 78, 8}
, {16, 56, 53}
, {-32, 18, -51}
, {12, 2, -23}
, {-5, 5, 18}
, {33, 23, 39}
, {5, 34, -54}
, {-29, 22, 28}
, {45, 56, 56}
, {-45, -60, 2}
, {-10, 16, -4}
, {23, 48, 40}
, {-55, -4, -15}
, {-16, 16, 61}
, {53, 0, -11}
, {71, 3, 50}
, {47, -37, 48}
, {18, 49, -28}
, {31, 12, -9}
, {15, -22, 20}
, {-31, 30, -31}
, {61, -22, 44}
, {28, 16, 37}
, {-35, 14, 24}
, {2, -9, -52}
, {-29, -15, 1}
, {-43, 16, -14}
, {-18, -52, 2}
, {-35, -29, -3}
, {-26, -9, 56}
, {-27, -13, -23}
, {24, 23, 82}
, {-28, 34, -3}
, {14, 34, 0}
}
, {{14, -39, 13}
, {0, 9, -13}
, {-16, 4, 36}
, {0, 51, 37}
, {-25, 21, 5}
, {67, 54, 36}
, {-21, -36, -61}
, {-41, -59, -36}
, {-9, -39, -18}
, {26, -43, 16}
, {-44, -5, 23}
, {28, -52, 21}
, {-35, -33, -21}
, {-22, -64, 53}
, {-27, 34, -15}
, {39, -37, -16}
, {-39, -58, -61}
, {64, 61, -29}
, {29, -1, 3}
, {57, 38, -23}
, {56, 30, 41}
, {-8, -32, -15}
, {-49, 9, -32}
, {-51, -40, 29}
, {68, -42, 22}
, {44, 25, -18}
, {33, 11, 18}
, {-24, 34, -27}
, {37, 42, 54}
, {-59, -4, -35}
, {1, -30, -38}
, {9, 14, 55}
, {11, -14, 37}
, {-6, -17, 59}
, {-5, -20, -11}
, {13, 8, -44}
, {-43, -21, -12}
, {-35, 22, 44}
, {43, 18, -15}
, {-37, 38, -58}
, {5, 46, 59}
, {-52, -87, -58}
, {42, 35, -44}
, {5, 21, -5}
, {13, 20, 6}
, {38, -35, -21}
, {43, 11, -38}
, {-11, 0, 52}
, {29, -30, -12}
, {60, 44, 10}
, {-22, -4, 17}
, {-19, -1, -2}
, {-19, 40, 40}
, {14, 15, -12}
, {-20, 43, 20}
, {51, 46, -42}
, {9, -21, 17}
, {39, -7, -45}
, {-29, -4, -68}
, {-15, 5, -4}
, {-33, 60, 17}
, {33, -38, -6}
, {15, 20, -21}
, {-36, 24, 38}
}
, {{13, 13, -3}
, {9, 34, 26}
, {6, 43, -54}
, {62, -19, 5}
, {1, 1, -42}
, {-27, 41, -31}
, {-58, -9, -62}
, {-25, -8, -71}
, {29, -10, 48}
, {18, -31, 23}
, {-17, -42, -20}
, {4, -10, 15}
, {-32, 12, 21}
, {-24, -45, 4}
, {27, 15, -1}
, {35, -41, -45}
, {-28, -56, -9}
, {-42, 34, 32}
, {-26, 5, -29}
, {32, 52, 41}
, {20, -23, -21}
, {40, 9, -6}
, {13, 9, -31}
, {17, -3, -21}
, {-35, 17, 44}
, {-36, 15, -32}
, {-1, 16, 0}
, {-61, 7, -35}
, {48, -29, -18}
, {-12, -14, -15}
, {40, 13, -36}
, {-43, -12, -4}
, {35, -38, 31}
, {-32, 8, 39}
, {-8, -16, 16}
, {34, -13, 16}
, {33, -3, -26}
, {-12, -30, 13}
, {-24, 6, 32}
, {-13, -18, -14}
, {42, -2, -54}
, {-64, 25, -63}
, {-27, -23, 40}
, {72, -31, 6}
, {-45, -53, -41}
, {36, 10, 6}
, {-28, -38, 23}
, {0, 25, -39}
, {13, 38, -11}
, {-13, -31, -27}
, {-15, -29, -38}
, {1, 4, -21}
, {0, -37, 2}
, {11, 29, 24}
, {-18, -16, -3}
, {-45, 48, 57}
, {-23, -36, -25}
, {34, -56, 46}
, {9, -4, -2}
, {-14, 15, 56}
, {48, 2, 50}
, {-52, 12, 22}
, {-27, 14, -32}
, {-39, 32, -40}
}
, {{-7, 11, -26}
, {67, 3, 72}
, {10, -50, 28}
, {25, 17, -25}
, {4, -33, 32}
, {-32, -21, -57}
, {-8, 47, 21}
, {-11, 0, 30}
, {29, -60, 12}
, {-25, 44, 27}
, {-18, 7, 27}
, {6, 1, 38}
, {-19, 22, -11}
, {10, -34, 21}
, {-17, -42, 60}
, {19, -19, -25}
, {2, 40, 18}
, {16, -5, -58}
, {-40, -6, -14}
, {-39, -20, 6}
, {26, 20, -23}
, {-20, -35, 6}
, {-11, 15, 14}
, {-8, 17, 1}
, {1, -12, -15}
, {-22, -3, 43}
, {42, -19, -41}
, {53, 65, 59}
, {7, 33, 65}
, {24, 54, 38}
, {53, 10, -30}
, {-45, -49, -39}
, {-39, 15, 18}
, {-7, 23, -16}
, {31, 16, -1}
, {34, -10, 13}
, {0, 33, 1}
, {-20, 82, -35}
, {-1, -2, 7}
, {-36, 47, -39}
, {-45, 32, -11}
, {45, 21, 76}
, {27, -6, 24}
, {16, -61, -18}
, {-48, -44, -23}
, {-52, 0, 15}
, {-35, -17, -9}
, {-47, 42, -43}
, {-44, 23, -11}
, {13, 42, -18}
, {-40, -28, -49}
, {-35, 38, -11}
, {37, -13, 33}
, {9, -2, 14}
, {12, 13, 45}
, {54, 25, 42}
, {7, -24, 46}
, {6, 51, -25}
, {-32, -27, 29}
, {-12, 13, 20}
, {-12, -11, -39}
, {-12, 2, -10}
, {-20, -16, -1}
, {-37, 6, -9}
}
, {{-24, 33, -41}
, {-48, 25, 7}
, {26, -35, 35}
, {-52, -54, -22}
, {-7, -41, 37}
, {-10, -25, 16}
, {12, -41, -39}
, {34, -28, -36}
, {49, 24, 42}
, {-46, 29, -39}
, {18, -10, 10}
, {-18, -17, 33}
, {40, -47, -13}
, {-47, 2, 42}
, {34, -9, -4}
, {-40, 7, 20}
, {-17, -26, -8}
, {24, 20, -27}
, {-20, 7, 7}
, {6, 38, 21}
, {-20, -10, 46}
, {-13, -44, -7}
, {26, -40, -55}
, {-45, -12, -55}
, {39, 12, 11}
, {-25, -8, -20}
, {32, -54, -49}
, {-35, 43, -26}
, {1, -41, 16}
, {-12, -48, 33}
, {0, 20, 1}
, {-28, -8, 17}
, {-29, 41, 17}
, {-40, -52, 19}
, {19, -44, 32}
, {-16, -53, -5}
, {9, -33, -29}
, {26, 23, -5}
, {-21, -25, -6}
, {-24, -35, -43}
, {19, -5, -26}
, {12, 24, -33}
, {-11, 31, -27}
, {-13, -30, 13}
, {7, -44, -49}
, {-41, 37, -30}
, {35, 18, -40}
, {-50, -46, -48}
, {-12, -10, 46}
, {-49, -26, 9}
, {5, -14, 16}
, {15, -30, -32}
, {3, 1, -21}
, {-12, -52, -27}
, {-26, -26, 44}
, {48, -32, 34}
, {-24, -44, -38}
, {-47, 32, -35}
, {1, 23, -15}
, {-22, 18, -20}
, {-40, 4, -45}
, {-44, 22, 9}
, {-11, 24, 43}
, {-3, 48, -26}
}
, {{-1, 35, -44}
, {54, 37, 35}
, {-41, -38, -14}
, {34, 29, 29}
, {4, -55, 18}
, {-39, 16, -27}
, {-54, 33, 16}
, {4, -30, -60}
, {17, -24, 0}
, {-2, -36, 30}
, {-6, 36, 35}
, {35, 46, -34}
, {-43, -23, -17}
, {-25, -46, 24}
, {-10, 24, -30}
, {3, 47, -42}
, {29, 50, -42}
, {-37, 24, -15}
, {0, -34, -4}
, {-20, -24, -47}
, {25, -6, 22}
, {9, -38, 22}
, {-31, -5, 15}
, {41, 19, 27}
, {-3, -36, -15}
, {71, 41, -12}
, {43, -20, 31}
, {10, -38, 39}
, {17, 24, 0}
, {-26, -12, -17}
, {24, 43, -39}
, {26, -20, -40}
, {-1, 33, 42}
, {-1, -6, 48}
, {-6, 32, -18}
, {-42, -19, 49}
, {16, -12, 20}
, {55, 21, -3}
, {0, -11, 8}
, {-8, 5, 34}
, {-44, -9, 1}
, {0, 15, -41}
, {43, -21, -29}
, {40, -25, 27}
, {-14, -42, -8}
, {-30, -59, -27}
, {-46, -26, 43}
, {28, -39, 0}
, {38, -31, 52}
, {46, -29, -14}
, {22, 20, 48}
, {26, -52, -43}
, {-12, 39, -47}
, {-44, -50, 20}
, {-48, -36, 9}
, {3, 41, 35}
, {-52, 23, 48}
, {29, -34, 41}
, {15, 45, 17}
, {-5, -1, -48}
, {-19, -48, -9}
, {40, -25, -29}
, {-25, -39, 39}
, {-18, 32, -22}
}
, {{-5, -40, 38}
, {5, -17, -8}
, {2, 45, 3}
, {-41, -5, -22}
, {-6, -57, -48}
, {53, 46, 11}
, {21, -37, 34}
, {44, -8, -18}
, {9, -1, -30}
, {-4, 33, 25}
, {30, 49, -45}
, {-28, -57, 16}
, {-7, 1, 29}
, {-28, -10, 17}
, {-18, -6, -41}
, {-15, -50, 25}
, {-34, -9, -41}
, {9, 29, 5}
, {24, -11, -21}
, {33, 35, 0}
, {26, 7, -11}
, {30, 50, 22}
, {55, -18, -24}
, {-29, -44, -9}
, {0, 14, 35}
, {-17, -34, 26}
, {16, -25, -11}
, {-5, 21, 2}
, {-32, -18, -13}
, {-41, 0, -5}
, {-36, -22, -21}
, {-55, -15, 59}
, {-53, -41, 35}
, {-36, 34, -10}
, {59, 19, -37}
, {-33, -5, -15}
, {15, -38, -33}
, {-16, 11, 23}
, {54, -14, -30}
, {-19, -29, 0}
, {1, -37, 43}
, {50, -22, -55}
, {30, -10, -35}
, {58, -29, 0}
, {-5, -6, -27}
, {25, 62, 36}
, {-8, 51, -40}
, {-10, 35, -45}
, {37, 33, 33}
, {-2, -8, 8}
, {40, 24, 31}
, {39, 18, 4}
, {10, 46, 36}
, {1, -30, 60}
, {-28, -34, -6}
, {42, 49, -8}
, {-35, -29, 39}
, {35, 31, 16}
, {-25, 31, -24}
, {19, 10, 55}
, {-44, -8, -3}
, {25, -1, -2}
, {-39, -10, -18}
, {-24, -49, -9}
}
, {{2, -29, -20}
, {-17, 26, -4}
, {2, -12, 40}
, {11, -5, -61}
, {-35, -4, 12}
, {-9, -39, -41}
, {35, -8, -25}
, {23, 9, 6}
, {-52, -40, -46}
, {-16, -67, 1}
, {76, 15, 77}
, {68, -64, 14}
, {-30, 26, -14}
, {36, 44, 20}
, {66, -48, 24}
, {-63, 30, -6}
, {-49, -23, -45}
, {31, -55, 28}
, {-8, -32, 3}
, {-26, -11, -28}
, {-66, -32, 12}
, {-16, -16, 33}
, {31, 21, 10}
, {72, -44, 50}
, {-67, 22, -41}
, {-29, 18, -12}
, {29, -45, 12}
, {36, 0, 105}
, {-12, -16, 39}
, {-40, 12, -4}
, {42, -13, 58}
, {48, -22, -35}
, {-14, 1, -55}
, {18, -66, 36}
, {39, 24, 23}
, {30, 53, -40}
, {-49, -12, -28}
, {31, -53, 54}
, {-44, 6, -10}
, {1, 63, -59}
, {-62, 4, -38}
, {-30, 75, 24}
, {19, -19, 7}
, {38, -21, 22}
, {35, 4, 0}
, {45, -40, -38}
, {28, -5, -27}
, {-54, 50, -49}
, {-7, 31, 28}
, {-38, -42, -43}
, {11, 3, -24}
, {-43, -16, -58}
, {18, -58, -17}
, {-37, 45, -1}
, {-41, 15, -27}
, {-7, 17, 52}
, {51, -28, -23}
, {-18, 75, -2}
, {-30, -5, 38}
, {13, -4, -38}
, {76, 18, 48}
, {-10, 60, 38}
, {-23, 46, -5}
, {-39, 46, -2}
}
, {{32, -3, 12}
, {-5, 59, 72}
, {-37, -17, -28}
, {21, -32, 57}
, {-17, 2, -37}
, {1, -19, 12}
, {-33, -27, -14}
, {-37, 0, -34}
, {72, -36, -35}
, {33, 21, -34}
, {38, 27, 34}
, {43, 37, 8}
, {-38, -7, -41}
, {-17, 50, 43}
, {51, -4, 26}
, {-29, -9, 65}
, {25, -55, 2}
, {-41, -35, -5}
, {10, -24, -55}
, {-17, 20, 38}
, {-24, -17, 14}
, {-40, -12, 32}
, {-38, -2, -38}
, {6, 81, -9}
, {27, 38, 65}
, {-29, 8, -21}
, {-21, -48, -35}
, {52, 31, -31}
, {-6, 43, 74}
, {83, 31, 27}
, {60, 30, 6}
, {2, -20, -22}
, {25, -46, 39}
, {24, 1, 4}
, {57, -42, 45}
, {43, -42, 12}
, {1, -3, -36}
, {-29, 53, 71}
, {69, 12, 78}
, {34, 81, 77}
, {27, 26, -8}
, {18, -18, 6}
, {1, 50, -35}
, {-22, -44, 5}
, {-4, 19, 50}
, {-46, 26, -15}
, {37, -8, -29}
, {-13, -29, -19}
, {39, -35, 19}
, {-15, 49, 27}
, {-49, -1, -20}
, {9, 35, -4}
, {-55, 40, -37}
, {18, -9, -37}
, {-32, -2, -53}
, {42, 20, 53}
, {14, 26, -29}
, {-13, -11, 59}
, {1, 19, 3}
, {-18, -19, 19}
, {-23, -15, 0}
, {14, 30, 49}
, {9, 2, 5}
, {25, 28, 43}
}
, {{42, 39, 42}
, {31, -31, 1}
, {19, 1, 38}
, {-23, 15, 6}
, {-25, -8, 44}
, {9, -3, 14}
, {-1, 1, 3}
, {49, 93, 68}
, {36, -30, -9}
, {18, -45, -6}
, {10, -19, -18}
, {-3, 41, 45}
, {-15, -11, 51}
, {-2, -3, -37}
, {-32, 25, -52}
, {-22, -24, 37}
, {13, 46, -22}
, {-33, 11, -42}
, {52, 39, 50}
, {49, -12, 30}
, {22, -32, -60}
, {-8, 55, -33}
, {29, 14, 58}
, {-14, 43, -38}
, {-48, 26, 10}
, {27, -24, 12}
, {-36, -22, -38}
, {-23, -42, -23}
, {-32, -71, 5}
, {-5, -47, -17}
, {-18, -12, -32}
, {-25, 26, 0}
, {-44, -27, -38}
, {2, 39, -14}
, {17, 14, -33}
, {-35, -17, 11}
, {33, -8, 37}
, {16, -45, 12}
, {-28, -20, 6}
, {20, 20, -34}
, {6, -24, 46}
, {-44, 54, 8}
, {-1, 21, 5}
, {-4, 47, 5}
, {69, -24, 23}
, {-8, 76, 0}
, {-1, -17, -34}
, {-5, 47, -48}
, {46, -21, -7}
, {4, 42, -36}
, {-2, -28, 3}
, {27, -24, -16}
, {-34, 20, -24}
, {17, -43, 15}
, {2, -11, 24}
, {-9, -42, 0}
, {41, -10, 47}
, {25, -21, -48}
, {3, -41, 1}
, {-36, 18, 26}
, {-8, -45, 44}
, {-20, 4, 20}
, {-29, 32, 49}
, {37, -47, -24}
}
, {{-46, -33, 42}
, {50, -16, -21}
, {-23, 90, -27}
, {-14, 34, 8}
, {4, 17, 27}
, {63, 2, 3}
, {6, 1, -16}
, {-39, -38, 65}
, {42, 37, -18}
, {2, 50, -26}
, {-8, 36, 19}
, {-28, 40, -3}
, {-35, -32, 9}
, {-48, -12, -28}
, {-21, -50, -35}
, {-39, 4, 25}
, {-31, 10, -17}
, {19, 66, 30}
, {-21, -6, 31}
, {-38, -37, 17}
, {-23, -27, -33}
, {-17, 46, 21}
, {-12, -12, -36}
, {-36, -14, -9}
, {-3, -18, -40}
, {33, -37, 31}
, {6, 24, -11}
, {52, 24, -44}
, {-15, -9, 26}
, {-25, -19, 11}
, {-18, -6, 26}
, {-10, 15, -1}
, {-40, 42, -33}
, {20, 37, 32}
, {80, 47, -39}
, {-12, -1, 40}
, {29, -44, -17}
, {87, 70, 0}
, {71, 76, 5}
, {-51, -25, 33}
, {8, 15, -47}
, {-65, 10, -24}
, {-30, 40, 0}
, {31, 47, 21}
, {-31, 39, -48}
, {15, -34, -9}
, {26, -46, -13}
, {47, -15, 32}
, {-23, -31, -45}
, {24, -22, 47}
, {48, 20, 5}
, {28, 51, -50}
, {62, 49, 25}
, {-17, 7, 53}
, {15, -5, 7}
, {32, -40, 12}
, {50, -49, 47}
, {-39, 12, -36}
, {-7, -43, -14}
, {29, 70, 0}
, {-16, 39, 8}
, {-20, -57, -57}
, {-7, 2, -32}
, {-1, -37, 8}
}
, {{-33, -7, 39}
, {-43, -15, 8}
, {-4, -51, 57}
, {14, -12, -22}
, {1, -38, -45}
, {38, 48, 13}
, {38, 2, 42}
, {29, 78, 45}
, {-19, 34, -50}
, {7, 20, -6}
, {30, -18, 45}
, {-22, -48, -30}
, {37, 15, -40}
, {-26, -50, 36}
, {-3, 36, -7}
, {22, 51, 35}
, {0, 46, -33}
, {-31, -19, 4}
, {22, -17, -38}
, {37, -34, 8}
, {24, -24, 40}
, {12, -3, 9}
, {9, 15, 2}
, {-38, 35, 12}
, {15, 48, 31}
, {0, 62, -58}
, {15, 18, 36}
, {-44, 29, -32}
, {-47, 25, 15}
, {-28, 0, 7}
, {5, 11, -54}
, {37, -32, 26}
, {-52, 34, -46}
, {31, -15, -34}
, {23, 38, -34}
, {-1, -52, -22}
, {-22, -21, -54}
, {-47, 38, -21}
, {-12, -21, -40}
, {42, 7, -7}
, {0, -10, 37}
, {60, -11, 15}
, {51, -18, -40}
, {12, 27, 50}
, {38, -33, -35}
, {29, -22, 30}
, {-20, -15, -41}
, {23, -6, -26}
, {-1, -33, 16}
, {-6, -11, -22}
, {51, -31, -43}
, {-11, -23, 23}
, {27, 18, -23}
, {40, -44, 16}
, {-13, 40, -47}
, {1, 15, -29}
, {19, -39, -15}
, {-6, -4, 5}
, {-31, 23, -49}
, {-17, -11, -28}
, {-61, -29, 38}
, {-42, 30, -19}
, {40, 23, 43}
, {11, 8, 31}
}
, {{-12, 32, 1}
, {1, 68, -17}
, {-18, -32, -22}
, {-27, 28, -31}
, {32, -9, 36}
, {-25, -8, -6}
, {-47, -5, -2}
, {7, -33, -20}
, {-17, -6, -6}
, {49, -26, 4}
, {6, -55, -2}
, {31, 53, 3}
, {29, 29, -48}
, {17, -28, 26}
, {49, 39, -23}
, {25, -38, -6}
, {-29, -33, 42}
, {43, 22, -8}
, {7, -20, -46}
, {28, -66, 35}
, {-36, 36, -12}
, {-29, -28, 34}
, {52, -16, -34}
, {-26, 17, 13}
, {48, -54, -18}
, {42, -47, -12}
, {-19, 47, -19}
, {46, 66, -16}
, {61, 20, -10}
, {87, 48, 64}
, {-12, 53, -35}
, {-57, 35, -5}
, {20, 50, 51}
, {-38, -29, -37}
, {-13, -60, -50}
, {-45, -23, -30}
, {36, 43, -27}
, {25, 58, 44}
, {-4, -40, 45}
, {46, -6, 19}
, {24, -41, 14}
, {66, 5, -10}
, {27, 12, 3}
, {-40, -65, -48}
, {-43, -30, -32}
, {-59, 18, -38}
, {5, 34, 0}
, {27, 14, -41}
, {-45, 2, -5}
, {5, -3, 37}
, {-41, -48, -35}
, {-26, -56, 38}
, {-69, 10, -57}
, {12, 37, 34}
, {-37, 33, -39}
, {-5, 0, 6}
, {-3, -13, -11}
, {1, -4, -7}
, {36, -31, 3}
, {34, -19, 36}
, {-71, 47, -68}
, {-6, -23, 16}
, {-40, -31, 14}
, {2, 44, -26}
}
, {{50, 37, 2}
, {67, 15, -12}
, {-48, -22, -27}
, {-12, 4, 1}
, {31, 15, -29}
, {54, -18, -39}
, {-44, -44, 4}
, {-32, -70, -19}
, {32, 63, 12}
, {-17, 1, 26}
, {43, 18, -42}
, {-30, 58, -58}
, {25, -40, 12}
, {-14, 22, 50}
, {25, 47, -43}
, {14, 26, 35}
, {-31, -23, 20}
, {11, -16, 4}
, {-38, 50, -20}
, {34, -32, 47}
, {24, -10, 35}
, {6, -28, -38}
, {35, -54, -22}
, {23, 31, 26}
, {11, -26, 4}
, {-9, -8, 27}
, {-46, 27, -35}
, {37, -1, -1}
, {11, 36, 20}
, {51, -30, 46}
, {19, -30, -7}
, {-30, -4, 50}
, {-11, -18, -2}
, {4, 34, -44}
, {13, 7, 63}
, {-30, -53, 8}
, {-56, -14, 26}
, {45, -13, -9}
, {29, 9, 111}
, {30, -40, 43}
, {34, -23, -54}
, {55, 59, -37}
, {45, -32, 52}
, {-21, 19, -34}
, {47, -33, 3}
, {-35, -14, -9}
, {38, 1, 0}
, {12, 37, -15}
, {15, -8, -5}
, {19, -25, 5}
, {30, 11, 32}
, {-22, 20, 41}
, {3, -38, 8}
, {-20, 33, 15}
, {36, -29, -38}
, {-2, 20, -16}
, {-7, 44, 30}
, {-31, 25, -26}
, {9, -37, -51}
, {-28, 16, 49}
, {-7, 73, -8}
, {56, -51, 62}
, {48, -34, 51}
, {-38, -28, 7}
}
, {{39, 12, 34}
, {36, 0, 52}
, {-43, -54, -11}
, {54, 9, -36}
, {-3, 42, -12}
, {-17, 30, -34}
, {-41, -22, 38}
, {60, 72, 26}
, {-41, 31, -24}
, {-32, -44, -3}
, {-51, 1, -3}
, {13, 48, -1}
, {0, -43, -20}
, {-27, 26, 46}
, {-49, -51, 3}
, {10, -36, 60}
, {27, 51, 53}
, {-37, -15, 42}
, {-24, -21, 8}
, {-20, -15, -53}
, {20, 0, 53}
, {-47, -19, 53}
, {44, -8, 32}
, {-35, 48, 64}
, {18, -6, -10}
, {-9, 10, 35}
, {42, -38, -45}
, {13, -30, -39}
, {13, -38, 52}
, {8, 36, 55}
, {-13, -21, 6}
, {-21, 13, -21}
, {-23, -28, -23}
, {-39, 41, 25}
, {-22, -23, -29}
, {37, 23, -36}
, {-20, 55, 28}
, {-19, 32, 33}
, {-17, -5, -35}
, {-33, 13, 52}
, {-26, -8, -6}
, {-23, 31, -31}
, {14, 30, -16}
, {56, -55, 0}
, {-9, 26, 19}
, {34, 46, -35}
, {-21, 15, -14}
, {32, -16, 0}
, {-7, -3, -1}
, {-40, 34, -29}
, {21, -20, -11}
, {32, -12, -42}
, {-25, 20, -11}
, {3, -32, 20}
, {-41, -31, 33}
, {-31, 45, -34}
, {0, 39, 37}
, {-25, 2, 55}
, {37, 5, -11}
, {-17, 43, 15}
, {-1, -15, -8}
, {-24, -28, 17}
, {-4, 22, 6}
, {9, 56, -45}
}
, {{-54, -14, 2}
, {-25, 67, 13}
, {-31, 0, 14}
, {2, -13, 31}
, {-19, 11, -20}
, {-26, 41, -11}
, {-6, 38, -48}
, {11, -42, 10}
, {2, 28, 49}
, {-42, 19, 50}
, {-39, -30, -1}
, {13, -38, 11}
, {-29, 1, 17}
, {-48, 34, 45}
, {23, -15, 43}
, {-16, 9, 2}
, {-34, 19, 5}
, {-28, 27, -26}
, {9, 2, 49}
, {-30, -44, -6}
, {10, 5, 6}
, {41, -23, 42}
, {-8, -43, 37}
, {8, -26, 4}
, {46, -5, -42}
, {-20, 67, -16}
, {-36, 11, 8}
, {-39, -7, 31}
, {44, 63, -6}
, {34, 51, 22}
, {30, 33, -27}
, {-5, 6, 19}
, {-5, 37, -34}
, {9, 47, 5}
, {-36, 26, 59}
, {31, 33, 19}
, {8, -43, -39}
, {53, 13, 28}
, {32, -4, 59}
, {64, 19, 61}
, {47, 31, 9}
, {-26, 29, -10}
, {21, -10, -14}
, {1, -39, -17}
, {-24, -18, -60}
, {-8, -40, -40}
, {41, 28, 32}
, {-11, -28, 16}
, {11, 20, 15}
, {21, -39, -1}
, {-1, 0, 40}
, {-9, -26, 18}
, {-13, 21, -12}
, {43, -37, -32}
, {-34, 16, 7}
, {4, 45, -11}
, {-12, 21, 34}
, {55, -38, 10}
, {-43, 25, -27}
, {-9, -41, 24}
, {47, 21, -10}
, {-6, 24, 37}
, {-21, 8, -8}
, {18, -39, -14}
}
, {{19, 54, -6}
, {9, -54, 23}
, {-26, -5, 46}
, {12, 40, -54}
, {-31, 11, -32}
, {11, 28, -48}
, {40, -11, 39}
, {107, 46, 31}
, {-6, -11, 15}
, {18, 37, -8}
, {3, -5, -47}
, {-5, -7, -31}
, {53, 40, -25}
, {-13, 28, 9}
, {-20, -24, -5}
, {43, -8, 21}
, {-16, 47, -16}
, {-29, 14, 50}
, {-23, 52, 43}
, {42, -9, 51}
, {11, -16, -21}
, {55, 16, 17}
, {26, -14, -6}
, {-18, -8, -20}
, {17, -6, 8}
, {22, 26, 23}
, {52, 1, -44}
, {-24, 23, 1}
, {-25, -56, 2}
, {29, 15, -64}
, {9, 31, 49}
, {50, 24, 48}
, {-28, 0, 19}
, {48, -5, -27}
, {34, 25, 48}
, {58, -21, 16}
, {-20, 13, -32}
, {5, -4, -65}
, {-22, -9, -12}
, {-67, -26, -23}
, {-19, 30, 33}
, {63, 34, 45}
, {23, 42, -5}
, {-24, -33, 22}
, {45, 69, 7}
, {41, 85, 0}
, {0, 23, 51}
, {35, -18, -13}
, {-16, -42, -4}
, {-46, 9, 49}
, {-11, 31, -35}
, {-44, 47, 15}
, {13, 26, 41}
, {-23, 48, 39}
, {29, 62, 33}
, {-52, 0, -50}
, {3, -32, -31}
, {43, 37, -27}
, {-7, -36, -40}
, {27, -42, 41}
, {32, 46, -10}
, {21, -30, 29}
, {28, -44, 40}
, {33, -31, -29}
}
, {{-42, 37, 11}
, {-7, -37, 38}
, {-48, -47, -23}
, {29, -64, -27}
, {-40, 10, -44}
, {-28, -32, -39}
, {24, 49, 9}
, {14, 72, 54}
, {44, -9, 10}
, {1, -4, -52}
, {51, 9, -1}
, {-36, -38, -32}
, {-2, 50, -40}
, {8, 50, -4}
, {53, -4, -37}
, {0, -6, -39}
, {-7, 43, 48}
, {4, -9, 31}
, {58, -46, 21}
, {42, -24, -6}
, {-2, -29, -42}
, {33, -15, -4}
, {22, -36, 42}
, {12, 1, 86}
, {-52, 57, 42}
, {22, 13, -4}
, {-50, 48, -4}
, {51, -8, 58}
, {-7, -64, -24}
, {-68, 49, -38}
, {36, 51, 59}
, {18, -51, -25}
, {33, 15, -46}
, {20, -11, -14}
, {26, -17, -28}
, {44, 6, -48}
, {1, 19, -10}
, {-18, 19, -47}
, {-30, 17, 34}
, {-34, 55, -31}
, {-24, -55, -67}
, {0, 49, 41}
, {35, 29, 5}
, {-56, -20, -36}
, {-1, 27, 2}
, {11, -3, 17}
, {-24, 53, 3}
, {-8, -8, -31}
, {17, 16, 14}
, {-8, 37, -33}
, {33, -43, 31}
, {30, -22, -19}
, {43, -59, -6}
, {4, -13, -43}
, {-52, 38, 26}
, {35, -25, -14}
, {40, 1, 3}
, {-32, 62, 29}
, {-28, -7, -41}
, {31, 38, -56}
, {26, 16, 35}
, {38, 94, 60}
, {13, 26, -31}
, {1, 3, -47}
}
, {{-47, 25, -66}
, {32, 46, 16}
, {-34, 15, -38}
, {-44, -52, 26}
, {-37, 10, -10}
, {-20, 39, 52}
, {-3, 47, -15}
, {-8, -4, -8}
, {-24, 27, -24}
, {-13, -63, -40}
, {-10, -30, -21}
, {-53, 72, 19}
, {-35, -12, 4}
, {18, 16, 15}
, {-30, -30, -63}
, {-15, -86, 4}
, {18, 3, 18}
, {-47, 29, 27}
, {-24, -37, 15}
, {13, 6, 38}
, {-2, 23, -55}
, {-28, -35, 3}
, {-34, -37, 61}
, {2, -10, 12}
, {25, -52, 36}
, {-12, -1, 3}
, {20, -31, -26}
, {6, 102, -33}
, {-45, 31, -38}
, {39, -18, 62}
, {44, 82, -23}
, {-58, -15, -22}
, {7, -45, 7}
, {-71, -28, 10}
, {48, -17, 54}
, {24, -48, 21}
, {-19, -36, -54}
, {-13, 37, 55}
, {2, -21, 19}
, {17, -46, 81}
, {-15, -25, 14}
, {70, 36, 1}
, {54, -15, -23}
, {-35, -47, 18}
, {63, 7, -24}
, {-22, 44, 16}
, {56, -17, -1}
, {38, -58, -24}
, {-42, -27, -28}
, {-24, -51, -15}
, {-9, -38, -28}
, {10, 25, 33}
, {-38, 2, -11}
, {-18, 12, 40}
, {-19, 32, -29}
, {8, -8, -10}
, {33, -19, 3}
, {25, 0, 1}
, {-45, -2, -33}
, {23, -47, 4}
, {22, 80, 20}
, {41, -32, 60}
, {34, -3, 37}
, {-6, -55, 19}
}
, {{6, 12, -51}
, {-41, 0, 32}
, {17, 7, 32}
, {-43, -18, -46}
, {-30, -30, 6}
, {43, 27, -47}
, {-41, 8, 46}
, {30, 57, 100}
, {38, -19, 41}
, {39, 46, -32}
, {32, 40, 13}
, {-33, -33, -52}
, {-15, -47, 39}
, {-11, 38, 5}
, {-3, -41, -1}
, {-4, 14, 23}
, {35, -20, 9}
, {38, 38, -29}
, {55, -5, 59}
, {59, 45, -52}
, {-54, 47, 8}
, {-35, -23, 26}
, {50, 44, 43}
, {66, 30, 6}
, {-27, -42, 16}
, {-3, 0, 43}
, {-32, 42, 0}
, {-33, -37, -20}
, {-70, -17, -42}
, {-34, -18, 8}
, {-42, -49, -42}
, {-46, -34, 18}
, {-4, -17, 10}
, {31, 15, 42}
, {12, -36, 6}
, {-50, -26, 32}
, {5, 46, 45}
, {-14, -45, 9}
, {23, 35, -20}
, {-5, 33, -57}
, {-41, 30, 34}
, {24, 4, -10}
, {21, -40, 38}
, {-23, 39, 41}
, {-30, -13, 33}
, {-17, 13, 14}
, {-53, 34, 9}
, {-23, -1, 7}
, {-15, -12, -11}
, {-15, 21, -48}
, {40, 11, 38}
, {20, -1, 11}
, {16, 17, -10}
, {-54, -28, 6}
, {-40, -40, 9}
, {1, -49, -30}
, {-33, 39, -25}
, {8, 16, -40}
, {10, -39, 25}
, {18, -27, -63}
, {-18, -8, -19}
, {52, 44, 27}
, {6, -7, 33}
, {44, 0, 5}
}
, {{42, 49, 46}
, {-26, -18, 4}
, {26, -49, 35}
, {37, 35, -17}
, {-21, 45, -18}
, {-33, 14, 15}
, {-41, 43, 11}
, {-23, 94, 33}
, {34, -20, 66}
, {-22, 7, -45}
, {29, 45, 8}
, {-36, -34, -5}
, {1, 37, -31}
, {7, -19, -15}
, {-27, -12, -26}
, {36, -16, 1}
, {16, 27, 1}
, {29, -50, -31}
, {2, 36, 8}
, {41, -11, -42}
, {-8, 2, 4}
, {-20, 14, -19}
, {34, -8, -16}
, {61, 56, -31}
, {42, 53, -39}
, {46, 29, -3}
, {-21, 38, -19}
, {15, 3, -16}
, {13, -47, 38}
, {60, 15, -32}
, {-36, -59, -3}
, {-30, -29, -23}
, {-39, -20, -35}
, {2, 47, -7}
, {-25, -2, -28}
, {-18, -56, -9}
, {-34, 52, 40}
, {55, -13, 58}
, {-56, 9, 29}
, {40, -47, 13}
, {37, -10, 0}
, {42, -46, -38}
, {9, -5, -15}
, {25, -46, -53}
, {-5, 10, -17}
, {-22, -39, 35}
, {-17, 34, -28}
, {-38, 8, -5}
, {3, 55, -22}
, {7, 46, -38}
, {60, -22, 37}
, {26, 4, -33}
, {-11, -14, -38}
, {25, -50, 35}
, {22, 38, 34}
, {50, 13, 7}
, {50, -29, 35}
, {-25, 27, 6}
, {-6, -11, -36}
, {-14, -54, 39}
, {15, -23, -66}
, {-42, -31, 20}
, {27, 40, -47}
, {-9, 8, 31}
}
, {{-16, -20, 17}
, {29, 13, -48}
, {49, -56, 52}
, {-22, 6, -4}
, {50, 52, 47}
, {25, 67, 29}
, {-42, -11, 26}
, {7, 34, 31}
, {56, -2, 50}
, {56, 11, 8}
, {27, -34, -49}
, {-43, -26, -25}
, {38, -28, 14}
, {-40, 14, 6}
, {15, 43, -37}
, {40, -6, 11}
, {-3, -3, 24}
, {7, 10, -16}
, {-18, 6, 20}
, {20, 67, 7}
, {-44, 54, 26}
, {52, 54, 58}
, {53, -18, -38}
, {-30, -22, -40}
, {61, 27, 1}
, {24, 21, -60}
, {-11, -35, -4}
, {11, -80, -11}
, {16, -25, -26}
, {-38, -19, -84}
, {6, -18, -19}
, {-31, -59, 8}
, {11, 7, 31}
, {-21, -29, 6}
, {-12, 66, 33}
, {28, -34, -35}
, {26, 14, 48}
, {-35, -31, -34}
, {-32, 0, -6}
, {3, -9, -12}
, {72, 27, 34}
, {20, -46, -34}
, {48, 20, -15}
, {86, 80, -29}
, {10, -4, -8}
, {-43, 17, 14}
, {4, 4, 8}
, {-41, 46, 0}
, {51, 58, 51}
, {48, -26, 35}
, {11, 6, -11}
, {-51, 44, 24}
, {33, -29, 59}
, {-23, -37, -6}
, {-40, 55, -46}
, {-52, 8, -13}
, {-14, 3, 6}
, {-46, -41, 22}
, {-38, 0, -9}
, {-1, 69, -25}
, {0, 8, 30}
, {3, -25, -9}
, {-39, 32, 10}
, {11, 28, 43}
}
, {{38, -27, 10}
, {-36, -8, -57}
, {-22, 2, 32}
, {13, 51, 42}
, {36, -37, -34}
, {-10, -31, -58}
, {-42, 44, 40}
, {70, 90, 47}
, {10, 53, 8}
, {-34, 34, -27}
, {-23, 16, 20}
, {43, -32, 6}
, {7, -31, 31}
, {29, 29, -12}
, {-13, 41, 2}
, {44, -51, 16}
, {70, 54, 6}
, {0, 9, -30}
, {36, 19, 40}
, {26, 10, 37}
, {-35, -3, 37}
, {-25, -38, 43}
, {19, 28, 10}
, {3, 68, 22}
, {47, -45, 37}
, {37, 12, 32}
, {-43, -15, -42}
, {-12, 29, 2}
, {-21, 27, -24}
, {8, 51, 11}
, {-46, -5, 54}
, {10, 46, 18}
, {-33, -22, 17}
, {-37, 6, 5}
, {-42, -40, -21}
, {-2, -39, -18}
, {41, 31, 52}
, {26, 57, -7}
, {-11, -48, -23}
, {34, 7, 57}
, {9, -49, -28}
, {15, 49, 37}
, {-34, 0, 42}
, {21, -29, 5}
, {-13, -14, 39}
, {-36, 6, 40}
, {18, -9, -19}
, {-4, -43, -20}
, {28, 7, 7}
, {-8, 9, -20}
, {-15, -9, 25}
, {-50, 0, -34}
, {-25, -1, 29}
, {-29, -55, 5}
, {-7, 56, -45}
, {36, -2, -26}
, {27, 5, 0}
, {25, -25, -30}
, {-39, 55, -33}
, {-19, 19, 18}
, {-50, 37, -38}
, {56, -18, -1}
, {-20, 41, 12}
, {-1, 54, -19}
}
, {{18, -47, -17}
, {-74, -23, 15}
, {-37, 47, 44}
, {-1, 45, -7}
, {13, 28, -48}
, {-6, 51, 7}
, {-1, -11, 7}
, {-3, -64, -2}
, {5, -31, 16}
, {-7, -46, -25}
, {8, 38, 36}
, {-56, -54, 44}
, {-15, -35, 16}
, {40, -37, -49}
, {28, -38, 5}
, {-5, -29, -23}
, {-34, 11, 45}
, {3, -47, 31}
, {-38, 37, -9}
, {35, -23, -15}
, {-29, -5, 3}
, {-4, -45, 40}
, {-33, -49, 16}
, {-18, 0, 26}
, {44, 28, -3}
, {-12, 3, 8}
, {-20, -55, 34}
, {4, 15, -35}
, {-54, -3, 34}
, {-50, -5, 16}
, {-23, 9, 33}
, {-25, -5, 46}
, {-49, -49, -50}
, {13, 35, -44}
, {40, 49, -55}
, {9, 5, 26}
, {19, -26, 32}
, {-8, 34, 45}
, {-2, -16, 2}
, {-11, 0, -17}
, {23, 43, 12}
, {-26, 28, 1}
, {-33, 23, 46}
, {18, -44, 36}
, {38, -45, -41}
, {-8, 34, 54}
, {-9, 45, -3}
, {-11, 10, -2}
, {-39, -44, -12}
, {-15, -29, -11}
, {57, 11, -12}
, {-4, -54, -24}
, {5, -13, 27}
, {-45, 18, -25}
, {-55, 14, 21}
, {0, 14, 11}
, {-35, 3, -31}
, {-2, -42, 44}
, {43, 19, -29}
, {-29, 47, -33}
, {22, 14, 0}
, {-16, -62, -13}
, {-46, 33, -49}
, {-37, 9, 20}
}
, {{8, 25, 26}
, {-32, -66, -56}
, {45, 25, 6}
, {-8, 17, -38}
, {-11, 4, 47}
, {-56, -33, -17}
, {9, 22, -31}
, {36, -32, 28}
, {-3, -10, 33}
, {14, 2, 27}
, {-61, -10, 16}
, {-49, -7, -15}
, {15, -42, 27}
, {-50, 35, -43}
, {38, -24, 32}
, {-2, -40, -41}
, {56, -1, 26}
, {-27, 21, -34}
, {63, -16, -11}
, {-48, 38, 54}
, {-34, 29, 11}
, {-30, -37, 27}
, {-42, -6, 40}
, {-20, -29, 38}
, {-24, 49, -26}
, {-8, -34, 5}
, {50, 19, -29}
, {30, 8, -3}
, {-44, -33, -26}
, {22, 4, -58}
, {17, 8, 49}
, {-26, 13, 50}
, {29, -29, -31}
, {47, -16, -17}
, {-47, -27, 25}
, {23, -35, -6}
, {59, 54, 53}
, {-21, -10, -13}
, {41, 1, -58}
, {-69, -65, 26}
, {10, 0, 28}
, {-34, -20, -33}
, {-14, -35, -6}
, {-18, 2, -4}
, {-41, 56, 42}
, {14, -23, 12}
, {-22, -23, -12}
, {47, 43, -11}
, {7, -13, 12}
, {47, -24, -6}
, {-17, 66, -28}
, {-28, -43, 33}
, {-8, 43, -14}
, {42, 1, 16}
, {-24, 27, -16}
, {38, 11, 32}
, {-13, -6, 6}
, {37, -22, -43}
, {21, 7, 6}
, {-17, -12, 12}
, {-16, 16, -14}
, {-39, 16, -47}
, {-39, 42, 14}
, {35, 18, 2}
}
, {{-6, 40, -9}
, {52, 43, 38}
, {34, -34, -26}
, {-14, -17, -27}
, {40, -38, 14}
, {22, -12, 56}
, {-55, -32, -50}
, {39, 5, -58}
, {29, -39, 12}
, {-34, -22, 58}
, {13, 88, 6}
, {26, -10, -31}
, {-47, 22, -8}
, {68, -37, -7}
, {38, -36, 57}
, {-44, 41, 0}
, {-36, -48, 20}
, {-15, -18, -32}
, {-56, 13, -38}
, {21, 2, 27}
, {7, 16, -24}
, {7, -38, -20}
, {-28, 23, -33}
, {30, -10, 12}
, {41, 16, 35}
, {-22, 1, -21}
, {-40, 4, -40}
, {8, 56, -26}
, {27, 12, 32}
, {11, -14, 48}
, {15, -6, 58}
, {-31, -34, -25}
, {-23, 26, -33}
, {30, -41, 38}
, {-14, 40, 11}
, {-14, 32, -28}
, {0, 17, -2}
, {64, 58, -1}
, {47, 8, 35}
, {-6, 62, -13}
, {-7, 13, -34}
, {78, 32, 73}
, {43, -44, 2}
, {38, 9, -46}
, {-19, -17, -27}
, {-37, 18, -19}
, {26, -44, -1}
, {45, -5, 35}
, {-5, -20, 31}
, {33, 31, 30}
, {-30, -44, -47}
, {43, -5, -37}
, {21, -14, 26}
, {-7, 8, 42}
, {-11, 46, -30}
, {-31, -21, 43}
, {-22, -28, -14}
, {48, -47, 45}
, {-39, 0, 47}
, {29, -18, 20}
, {-1, -2, 9}
, {-16, 21, -2}
, {-44, -10, 38}
, {-20, 10, 21}
}
, {{46, 37, -16}
, {13, 49, -39}
, {58, -13, 32}
, {62, 57, 52}
, {-31, 8, 35}
, {31, 25, 5}
, {18, 13, 35}
, {94, 45, 102}
, {100, 135, 17}
, {74, 23, 43}
, {-59, 24, -25}
, {12, -11, 0}
, {-13, -39, -10}
, {-60, -9, -25}
, {-65, -39, 1}
, {40, 46, 95}
, {79, 27, -1}
, {49, 59, 65}
, {70, 48, -17}
, {-12, -24, 7}
, {39, -18, -39}
, {0, 58, -19}
, {39, 3, 14}
, {21, 62, 40}
, {55, 1, 16}
, {-10, 13, 77}
, {-15, -31, -25}
, {33, -52, -18}
, {-35, -8, 11}
, {29, -49, -54}
, {-22, -24, 37}
, {51, 62, 54}
, {-20, 22, -20}
, {-13, 12, 10}
, {-29, 28, 5}
, {18, -35, 20}
, {62, 73, 100}
, {-40, -19, 10}
, {6, 101, -33}
, {48, 26, -17}
, {-6, 50, 43}
, {-33, -49, -11}
, {10, -13, 9}
, {96, 98, 96}
, {-24, 14, 35}
, {77, 58, 62}
, {-9, -20, -19}
, {-3, -46, 14}
, {65, 20, 53}
, {8, 43, 53}
, {-2, 13, 45}
, {-17, -16, -32}
, {-7, 23, -23}
, {26, 20, -6}
, {7, 57, -1}
, {-3, -38, 35}
, {-1, -36, 5}
, {42, 16, 0}
, {45, -2, -17}
, {5, -11, 29}
, {-25, -9, 57}
, {41, 16, -29}
, {-22, -10, -29}
, {-42, -16, -11}
}
, {{9, 20, -12}
, {16, 42, -11}
, {9, 70, -30}
, {-36, -33, -26}
, {-28, 60, 19}
, {14, -16, 64}
, {36, -58, -24}
, {-28, -15, 14}
, {25, -17, 33}
, {7, -40, -39}
, {21, -55, 18}
, {-42, 0, -55}
, {-24, -5, 20}
, {-31, -34, -46}
, {-7, 7, 42}
, {-36, -45, -62}
, {43, 51, -31}
, {8, 14, 27}
, {12, 39, 6}
, {-2, 1, 68}
, {-41, -5, 22}
, {37, 2, 34}
, {46, -23, 13}
, {43, 24, -48}
, {-30, 11, -23}
, {-43, 69, -18}
, {9, -24, 34}
, {-43, -48, -35}
, {-40, 49, -40}
, {-8, -34, -47}
, {3, -47, 49}
, {72, -2, -16}
, {-62, 43, 36}
, {4, -45, 28}
, {19, 62, -28}
, {4, 19, 13}
, {40, 0, -1}
, {-20, -38, -53}
, {36, 42, 4}
, {-38, 16, 17}
, {34, 4, -16}
, {13, 0, -48}
, {-36, 47, 48}
, {72, 3, -9}
, {-46, 16, -22}
, {-21, -14, 54}
, {32, 8, 20}
, {-35, -39, -16}
, {41, 8, 44}
, {-17, -29, 64}
, {16, 53, -29}
, {34, 26, 46}
, {67, 25, 6}
, {42, -28, 11}
, {53, -25, 19}
, {50, 50, -29}
, {-22, 37, 13}
, {-55, -46, -44}
, {51, -29, -47}
, {-12, 36, 50}
, {15, 22, 16}
, {-21, 28, -85}
, {-36, 13, 10}
, {31, -13, 21}
}
, {{22, -32, 5}
, {21, 9, 0}
, {44, 18, 59}
, {34, 8, 38}
, {-33, -19, -44}
, {25, 57, 51}
, {-55, 26, 17}
, {-62, -51, -69}
, {-29, 28, -5}
, {13, -12, 43}
, {-71, -38, -81}
, {-27, -15, 3}
, {11, -28, 11}
, {-4, -25, -44}
, {-41, 4, -21}
, {4, -34, 1}
, {-21, 0, -7}
, {7, -4, 65}
, {-55, -26, -39}
, {-5, 50, -47}
, {-28, 44, -21}
, {39, -18, -46}
, {-43, 40, -39}
, {38, -26, 14}
, {67, -20, -58}
, {16, -4, 13}
, {60, 18, -52}
, {-71, -36, -4}
, {6, 42, -1}
, {-37, -12, -20}
, {-6, -23, -47}
, {-13, -32, -20}
, {-27, -7, 29}
, {-37, 18, 51}
, {-19, -17, 31}
, {-19, 27, -31}
, {-6, 42, -28}
, {25, -2, 46}
, {-5, -23, 47}
, {1, 23, -36}
, {32, -41, -63}
, {15, -51, -81}
, {-11, 22, 21}
, {12, 36, 29}
, {-47, 32, -16}
, {14, 37, -27}
, {-20, 63, 55}
, {-43, -11, -3}
, {40, -15, -43}
, {-9, -25, -14}
, {38, 24, -5}
, {11, 62, 67}
, {46, -49, 65}
, {13, 41, -11}
, {-36, 44, -25}
, {12, 61, -31}
, {-15, -37, -31}
, {46, 17, -30}
, {56, -26, -4}
, {5, 59, 31}
, {20, 71, 51}
, {-21, -20, -15}
, {-25, -25, -5}
, {16, -27, -8}
}
, {{2, 22, -4}
, {-9, 19, 50}
, {17, -13, -21}
, {47, -25, 2}
, {-39, 26, 19}
, {26, -2, 49}
, {25, -1, -14}
, {-44, -73, -75}
, {-19, -18, 25}
, {10, 4, 26}
, {14, -45, -3}
, {39, 39, 46}
, {0, -7, -50}
, {-17, -15, 44}
, {-46, -21, -13}
, {54, -11, 29}
, {-30, 15, -34}
, {58, 54, 6}
, {4, -16, -45}
, {47, 5, 0}
, {40, 12, -4}
, {12, 11, -53}
, {-11, -9, 5}
, {17, -1, -38}
, {33, -39, -6}
, {-5, 20, -27}
, {10, -18, 1}
, {-43, 18, -33}
, {49, 10, 13}
, {35, 1, -42}
, {-10, -26, -55}
, {28, 8, -44}
, {-9, 12, 12}
, {45, -34, 42}
, {-38, -30, -41}
, {40, -22, -49}
, {18, 13, -11}
, {45, 46, 44}
, {-8, 15, -52}
, {-30, -38, 0}
, {-37, -27, -29}
, {14, -44, -23}
, {0, -39, -14}
, {-15, 25, 56}
, {13, -26, -25}
, {0, -11, -29}
, {39, -10, -7}
, {51, -54, -23}
, {14, -37, -18}
, {64, -38, 53}
, {-42, -17, 5}
, {-28, -24, -29}
, {-36, 47, 40}
, {-8, -20, 10}
, {34, -46, -35}
, {43, 13, 26}
, {11, 34, 46}
, {-48, -9, -12}
, {-28, 6, -46}
, {5, -19, 12}
, {59, 5, 33}
, {-59, -20, -13}
, {-24, 42, 43}
, {-28, 36, -41}
}
, {{-4, -20, -24}
, {22, -6, 95}
, {40, -23, -27}
, {-52, 43, -12}
, {-48, -29, -27}
, {-15, 23, 52}
, {-19, -22, 37}
, {4, -21, 0}
, {34, 20, 0}
, {24, 32, -61}
, {-48, -25, 4}
, {-21, -44, 53}
, {0, -12, 39}
, {13, -36, 62}
, {0, 22, -21}
, {-54, 39, -8}
, {31, -39, -21}
, {23, -14, -40}
, {-23, 7, -55}
, {3, 44, 18}
, {-18, 32, -8}
, {-29, 34, 2}
, {-5, -29, -13}
, {38, 24, -35}
, {-2, -2, 2}
, {34, 25, -36}
, {29, -3, 1}
, {34, -22, -8}
, {62, 14, -26}
, {9, -4, -22}
, {20, 46, 13}
, {33, -25, 32}
, {-48, -45, 40}
, {-40, -52, -42}
, {23, 0, 19}
, {27, 43, -57}
, {-60, -30, -17}
, {-35, 43, 42}
, {51, 20, 104}
, {7, -13, 8}
, {-31, -7, 4}
, {1, -17, 48}
, {-55, -11, 13}
, {4, 7, 48}
, {-24, 35, -14}
, {3, 25, 2}
, {26, -21, -46}
, {14, 50, -59}
, {-53, -61, -23}
, {15, 6, -33}
, {-40, -61, -38}
, {30, -5, 21}
, {44, -12, 47}
, {-15, -27, 0}
, {-39, 2, -30}
, {-4, -35, 3}
, {-4, 32, 18}
, {12, -19, -4}
, {18, -52, 34}
, {46, 62, -45}
, {53, -5, 62}
, {39, -8, -27}
, {-12, 34, -47}
, {18, -20, 25}
}
, {{-47, -21, 26}
, {10, -58, -64}
, {-35, -53, 6}
, {28, 32, -18}
, {14, -10, -2}
, {-27, 31, -54}
, {22, 52, 52}
, {-33, 90, 77}
, {22, 1, 37}
, {47, 41, 29}
, {-17, 21, -10}
, {-14, 6, -28}
, {15, 24, -36}
, {-23, -36, -12}
, {19, -3, 8}
, {27, 36, -49}
, {5, -39, -4}
, {10, -20, 18}
, {-31, -10, 24}
, {1, -1, -29}
, {25, 31, -18}
, {41, 38, -6}
, {32, 45, 58}
, {2, -37, 29}
, {30, 41, 0}
, {43, -28, -6}
, {-7, 50, -20}
, {6, 31, 17}
, {-51, -9, -29}
, {22, -32, -44}
, {-47, 38, 6}
, {54, 9, -34}
, {20, 47, 51}
, {8, -8, -11}
, {-70, -25, -27}
, {-39, -14, 27}
, {25, 66, -9}
, {-37, -23, 29}
, {12, -69, -66}
, {-33, -50, -31}
, {9, -33, -26}
, {-17, 27, 20}
, {49, 8, -22}
, {30, -32, 15}
, {49, 20, 12}
, {47, -45, 62}
, {-47, 16, 5}
, {-32, -27, -5}
, {-37, 29, -21}
, {-44, 12, 11}
, {30, 19, 55}
, {-18, 16, 15}
, {-14, 7, -2}
, {47, 33, 9}
, {-50, 29, 10}
, {33, -17, 49}
, {-43, -29, 27}
, {-3, 22, 33}
, {-31, 0, -33}
, {-15, 42, 6}
, {26, 17, 3}
, {-10, -28, -4}
, {-35, 22, 39}
, {42, -52, -7}
}
, {{-52, -57, -21}
, {39, 23, -33}
, {46, -25, -47}
, {-21, -23, 8}
, {-8, -5, -3}
, {25, -8, -37}
, {-8, 37, 6}
, {40, 41, 28}
, {19, 46, -19}
, {4, -24, -50}
, {3, 2, -42}
, {28, -38, 6}
, {-4, 16, -6}
, {0, -35, -7}
, {-50, -32, -37}
, {33, 42, -39}
, {8, 34, 25}
, {25, 34, -41}
, {-4, 34, -9}
, {46, 45, 26}
, {-3, 32, 34}
, {43, -15, 8}
, {-21, -35, -53}
, {13, 20, -7}
, {-10, 43, -7}
, {28, 3, -34}
, {0, -52, -46}
, {-58, -12, 18}
, {37, 19, 42}
, {54, 18, -24}
, {-1, 5, -9}
, {46, -43, -28}
, {1, -43, -54}
, {-32, -30, 24}
, {-48, 46, -9}
, {-44, -2, -31}
, {-7, -45, 37}
, {48, 13, 36}
, {21, 7, 36}
, {-7, -61, -44}
, {13, -39, -16}
, {-39, -14, 11}
, {-32, 49, -7}
, {18, -24, -39}
, {35, 10, 12}
, {24, -7, -46}
, {11, -25, 1}
, {24, 32, 39}
, {4, 13, -27}
, {-44, -54, 17}
, {-13, 8, -33}
, {8, 19, -30}
, {-46, -47, 35}
, {11, 40, -10}
, {32, 42, 38}
, {-19, 0, -49}
, {14, 24, -22}
, {48, -40, -5}
, {-10, 29, -51}
, {41, -47, 31}
, {0, -51, -60}
, {4, -40, 17}
, {-44, 32, -39}
, {-42, 33, -24}
}
, {{10, 12, 39}
, {27, -16, 42}
, {20, -60, -54}
, {13, 32, 2}
, {-1, -21, 5}
, {25, -30, 20}
, {9, 55, 21}
, {26, 78, 44}
, {32, -20, 5}
, {0, -25, 46}
, {-35, -28, 8}
, {39, 24, 11}
, {34, -23, 50}
, {59, 31, 34}
, {-14, -42, -22}
, {-5, 23, 59}
, {55, 48, 27}
, {12, 32, 10}
, {-39, 32, -28}
, {-17, -19, -58}
, {-44, -21, -49}
, {49, -7, 47}
, {-40, -13, 35}
, {36, 62, -30}
, {14, -12, 33}
, {48, -7, -4}
, {-10, -34, -24}
, {56, -19, 7}
, {45, 1, -3}
, {49, 26, 22}
, {56, -25, 20}
, {-35, 1, -15}
, {-48, 26, -32}
, {4, 45, -46}
, {-28, -36, -40}
, {19, 32, 8}
, {-47, -2, 0}
, {-15, 62, -7}
, {0, 38, -53}
, {49, 9, -6}
, {16, -29, -3}
, {57, 28, 54}
, {27, 44, 14}
, {-56, 27, -30}
, {55, -47, 9}
, {-40, 10, 33}
, {-42, 23, 37}
, {-21, 32, 31}
, {-19, -36, 31}
, {-35, 12, 24}
, {7, -17, 10}
, {-47, -33, 33}
, {37, 16, -48}
, {-21, 37, 23}
, {-43, -36, -26}
, {-30, -50, -38}
, {39, 25, 15}
, {51, -49, -21}
, {49, 26, 45}
, {-48, 44, -52}
, {21, -4, -49}
, {24, -16, 92}
, {45, -37, 38}
, {-46, -41, 51}
}
, {{7, 49, 32}
, {-22, -32, -10}
, {59, 34, -7}
, {-1, 19, 40}
, {-3, -29, -49}
, {-29, 14, -21}
, {-25, -2, 51}
, {17, 44, 36}
, {-50, 47, 35}
, {-28, -50, 30}
, {24, 30, -14}
, {40, 41, 22}
, {-21, -20, -5}
, {-54, -16, 0}
, {16, 15, 0}
, {49, -54, -1}
, {19, 43, 30}
, {-34, -30, -9}
, {15, 0, 5}
, {9, 11, 21}
, {2, 13, 50}
, {-6, -8, 35}
, {45, 20, 16}
, {30, -49, 11}
, {43, 31, 6}
, {34, 21, -50}
, {43, -6, -11}
, {-40, 28, -16}
, {3, -20, -62}
, {-35, -20, -3}
, {-17, -49, -4}
, {-24, -27, -36}
, {11, -46, 21}
, {31, 50, 49}
, {-17, -10, -42}
, {-5, 39, -44}
, {50, 11, 57}
, {34, -36, 4}
, {12, -17, -43}
, {14, -10, 10}
, {59, 40, 41}
, {-41, 9, -41}
, {-36, 36, -11}
, {44, 24, 67}
, {-26, 20, -43}
, {18, 1, 57}
, {-47, 33, -1}
, {10, 21, -15}
, {-7, 19, 4}
, {-10, -14, 0}
, {-31, 26, 53}
, {34, 22, -23}
, {24, 51, 44}
, {15, 5, 0}
, {-3, -31, -9}
, {44, 23, 8}
, {-3, 6, 5}
, {11, 25, 34}
, {-1, -14, 34}
, {47, -32, -39}
, {11, -16, -48}
, {-31, -51, -13}
, {4, -36, -20}
, {-32, 31, 43}
}
, {{9, -37, 29}
, {26, 29, 31}
, {20, 44, -2}
, {8, -47, 38}
, {-33, 46, 38}
, {59, -21, -9}
, {-52, -57, -20}
, {20, -21, 3}
, {9, 6, -12}
, {8, 7, -2}
, {-9, 42, 23}
, {41, -40, 22}
, {14, -16, -22}
, {-11, -32, -19}
, {-12, -17, -7}
, {54, 35, -48}
, {-1, -20, 16}
, {-6, 40, -26}
, {-47, -13, -43}
, {23, -2, -22}
, {-41, -12, -36}
, {38, -63, -37}
, {19, -39, 19}
, {-12, -31, 40}
, {7, 39, -8}
, {-14, 55, 42}
, {-15, 26, -7}
, {-53, 19, -10}
, {54, 70, -1}
, {-7, 55, -4}
, {-12, -2, 2}
, {-51, -45, -27}
, {35, 28, 1}
, {38, -34, -27}
, {56, -22, 46}
, {-18, 42, -28}
, {-29, 44, -44}
, {22, 59, 63}
, {66, 64, 75}
, {-3, 62, 35}
, {47, -44, 24}
, {-6, 20, -44}
, {27, -15, -7}
, {32, 25, 21}
, {-30, -3, -35}
, {-34, 24, -20}
, {47, -1, 4}
, {31, -44, -2}
, {-7, -3, 13}
, {37, 45, -26}
, {-60, -10, 21}
, {35, -31, -12}
, {-26, -20, -4}
, {-44, 52, 33}
, {20, 37, 33}
, {21, -21, 8}
, {43, -51, -36}
, {43, 60, -57}
, {44, 18, 7}
, {-34, 18, -8}
, {47, 21, 50}
, {75, -47, 39}
, {-38, -18, 16}
, {33, -45, -48}
}
, {{-44, -54, 51}
, {-36, -56, -47}
, {34, 15, -16}
, {29, 32, 51}
, {-46, -41, 35}
, {11, 40, 19}
, {-53, 41, 33}
, {-36, -43, -4}
, {-46, 29, -31}
, {-39, 1, -17}
, {-17, -30, -44}
, {35, 29, 43}
, {13, 20, -30}
, {-7, -34, 10}
, {-32, -49, -2}
, {38, -28, 13}
, {-53, -29, -18}
, {34, 12, 55}
, {2, 28, -48}
, {-15, -12, 5}
, {29, 0, -17}
, {-57, -41, 24}
, {-44, -27, 38}
, {0, -50, -39}
, {-23, -48, 19}
, {4, 37, -39}
, {-7, -49, -30}
, {-16, -22, -13}
, {8, 1, 2}
, {-30, -39, -41}
, {-6, -1, -15}
, {-43, -21, -39}
, {13, -23, -3}
, {30, -3, 36}
, {-22, -40, -3}
, {-49, -67, -28}
, {38, -19, 31}
, {0, 42, 47}
, {15, 38, 9}
, {-9, -6, -32}
, {49, -1, -8}
, {-15, 37, -54}
, {32, -27, 44}
, {-14, 5, 28}
, {-3, -26, 26}
, {15, -49, 29}
, {-54, 14, 28}
, {-16, 16, -16}
, {-7, -41, -34}
, {-39, 0, -2}
, {-57, 29, 13}
, {-46, -20, 14}
, {41, 27, -12}
, {-58, 21, 6}
, {23, -43, -12}
, {5, 37, -55}
, {-44, 50, 18}
, {-19, -49, 12}
, {46, 0, 34}
, {-9, -2, 17}
, {-12, -11, 48}
, {-68, 2, -16}
, {-45, -44, -18}
, {45, -51, -44}
}
, {{41, 7, -29}
, {21, 46, -20}
, {32, -48, -12}
, {10, 37, -40}
, {-25, 21, -9}
, {-54, -18, -45}
, {41, 23, 17}
, {-31, 66, 26}
, {30, 50, 50}
, {-35, -44, -43}
, {55, -28, 47}
, {-19, 18, -42}
, {48, 38, 41}
, {11, 44, 53}
, {17, -29, -48}
, {23, -23, 39}
, {49, 28, 12}
, {-18, -34, 10}
, {47, -3, 38}
, {-32, 39, -41}
, {-42, 41, 40}
, {3, -49, -21}
, {35, 39, -6}
, {-20, 20, 19}
, {56, 23, 3}
, {-17, -37, 29}
, {-19, 21, 2}
, {-1, 55, 11}
, {35, -4, -39}
, {52, -31, 23}
, {20, 19, -39}
, {45, -18, -42}
, {27, 7, -15}
, {29, -35, -31}
, {0, 14, 22}
, {-14, 13, 13}
, {-51, -27, -34}
, {-26, -40, -21}
, {98, 35, 13}
, {34, -22, 39}
, {25, 23, 0}
, {85, 56, 30}
, {-19, 25, -18}
, {3, 41, 24}
, {14, -34, 41}
, {26, -18, 36}
, {-6, -22, -1}
, {-9, 45, -25}
, {-37, -21, -14}
, {-36, 23, -40}
, {65, 22, -19}
, {17, -41, -18}
, {3, 0, 8}
, {-18, 9, 47}
, {-46, 8, -35}
, {-42, -18, -12}
, {-4, -15, -9}
, {3, -15, -38}
, {-6, -49, -9}
, {-37, -37, -19}
, {-28, 67, 39}
, {-3, 19, 79}
, {2, -35, 48}
, {26, -20, -15}
}
, {{-47, -21, 24}
, {0, 13, -25}
, {-27, 9, -9}
, {39, 18, 17}
, {40, -38, -34}
, {-3, -23, 33}
, {-32, 12, -8}
, {22, -18, 0}
, {-27, -29, 7}
, {-51, -29, -21}
, {-41, 10, 2}
, {19, 27, 22}
, {35, -27, -25}
, {42, 18, 40}
, {46, 8, 7}
, {-16, 13, -38}
, {0, 29, -6}
, {-24, -7, 39}
, {-4, 37, 44}
, {35, -39, 25}
, {-36, -51, 16}
, {-10, -42, -16}
, {6, -28, -26}
, {-14, 41, -28}
, {6, -21, 14}
, {-2, 3, -49}
, {-34, 6, -53}
, {-26, -5, -44}
, {-27, -21, -43}
, {47, -22, -32}
, {27, 33, -60}
, {34, -15, -18}
, {-32, 9, -22}
, {-5, -31, 8}
, {-11, 1, -9}
, {39, 35, -33}
, {-15, -7, -25}
, {30, -1, 17}
, {2, 0, 97}
, {-5, 0, 74}
, {16, -9, 26}
, {24, 30, 43}
, {16, -9, 12}
, {-40, 27, 11}
, {36, -37, -24}
, {-47, 23, -47}
, {51, -6, -15}
, {33, -43, 38}
, {-53, -43, -52}
, {47, 32, -54}
, {-53, 32, -37}
, {1, 40, -21}
, {4, -1, 0}
, {-7, 22, -38}
, {-4, 50, 3}
, {24, 2, -1}
, {-2, 29, 20}
, {30, -22, -26}
, {30, 35, 5}
, {29, -18, 24}
, {16, 93, -38}
, {61, 17, 78}
, {-46, -34, -24}
, {18, 26, -2}
}
, {{-27, 12, -43}
, {9, -3, 49}
, {-56, 42, 13}
, {55, 54, 59}
, {16, 36, -24}
, {9, -18, -27}
, {16, 46, 16}
, {103, 0, 72}
, {18, 4, 50}
, {9, 7, 32}
, {-29, 37, 44}
, {47, -17, -43}
, {51, 11, 13}
, {29, 9, 30}
, {6, -27, 17}
, {-24, -47, -14}
, {-13, 27, 56}
, {-45, 2, -43}
, {-7, 17, 17}
, {-54, -46, 18}
, {-13, 42, -21}
, {-39, 24, 18}
, {52, 26, -8}
, {48, -30, 13}
, {26, 25, 25}
, {26, 32, 52}
, {2, 45, 26}
, {-18, -18, -25}
, {-8, 45, -45}
, {35, -27, 19}
, {-56, -47, -31}
, {5, 37, 33}
, {-48, -33, 7}
, {7, -48, -39}
, {-33, 14, -20}
, {-42, -29, -36}
, {63, 2, -15}
, {6, 51, 47}
, {21, -1, -13}
, {-39, 9, 45}
, {52, -47, -47}
, {-22, 12, -42}
, {-4, 4, -19}
, {32, 6, 57}
, {-29, 24, -50}
, {5, -27, -47}
, {-21, -46, 4}
, {-22, -14, -44}
, {18, 32, 59}
, {12, 16, -25}
, {-20, 46, -11}
, {35, 17, -49}
, {25, -39, 23}
, {18, 39, 31}
, {-21, 3, 10}
, {47, -27, -8}
, {-5, 44, -8}
, {0, 54, 44}
, {-12, -27, 45}
, {-11, -52, -2}
, {-31, -60, 22}
, {34, 18, -8}
, {-15, -24, -35}
, {36, -17, -15}
}
, {{37, 39, 48}
, {25, 33, -56}
, {-54, 6, -20}
, {-56, -12, 53}
, {-58, -2, 5}
, {-29, 10, 40}
, {30, -8, -33}
, {-14, -24, 23}
, {0, -27, -42}
, {-16, -49, -27}
, {-40, 5, -17}
, {-22, 14, 0}
, {-47, 27, -19}
, {-58, 45, 13}
, {-42, -10, -57}
, {-31, -14, 14}
, {-1, 32, -46}
, {-36, -20, 33}
, {-25, -27, -31}
, {-52, 38, 31}
, {-33, 10, -33}
, {-10, 7, -1}
, {12, 13, 35}
, {0, -7, 8}
, {26, -17, 16}
, {-51, 55, -44}
, {-24, -9, -51}
, {19, 45, 34}
, {-32, 25, 26}
, {-25, -34, 12}
, {-31, 11, 23}
, {-30, -48, -55}
, {-26, -55, -20}
, {-12, -41, 32}
, {-60, -61, -53}
, {-64, -21, -5}
, {14, -49, 5}
, {47, 30, 40}
, {-51, -1, 3}
, {33, -37, -9}
, {3, 38, 29}
, {-55, -16, -42}
, {-12, 31, 12}
, {-42, -21, 3}
, {7, 55, -44}
, {-25, 23, -10}
, {-26, 15, 26}
, {-41, 6, 4}
, {16, -19, -23}
, {31, -60, 18}
, {16, 8, -16}
, {0, -42, -19}
, {5, -27, -23}
, {-35, -32, -49}
, {-37, -24, 49}
, {-45, 26, -49}
, {-30, -10, -38}
, {9, -13, 22}
, {16, 13, -7}
, {12, -42, -2}
, {9, 9, 27}
, {54, 2, -19}
, {4, 17, -19}
, {-24, -42, -38}
}
, {{40, 24, -30}
, {30, 32, 58}
, {21, -59, -23}
, {-38, -14, 18}
, {40, -20, 6}
, {-23, 10, -8}
, {-53, 48, -16}
, {-37, -34, -18}
, {-45, 20, -45}
, {30, 18, 2}
, {39, 16, 30}
, {14, 16, 52}
, {-54, 36, 34}
, {33, -9, -48}
, {26, -46, -35}
, {65, -8, 37}
, {-3, 53, 19}
, {37, -22, -24}
, {-28, -44, -22}
, {-26, -1, 32}
, {-32, -6, 23}
, {-28, -13, -51}
, {27, -29, -47}
, {48, -25, 42}
, {-53, 45, 49}
, {-25, 18, -28}
, {-27, -24, -9}
, {-15, 23, -20}
, {11, 11, 20}
, {-17, 5, 56}
, {-18, 14, -73}
, {8, -35, -34}
, {-42, -12, -27}
, {-38, -44, 2}
, {32, 0, -18}
, {-12, -10, -9}
, {18, 12, 8}
, {27, -2, 24}
, {-19, -28, 4}
, {-6, 21, 34}
, {-45, 0, -43}
, {-17, -27, -31}
, {16, -25, 7}
, {56, -2, 43}
, {-39, -50, 26}
, {-32, 19, -1}
, {37, -24, 24}
, {9, 11, -14}
, {39, 17, 38}
, {-45, 22, -27}
, {-15, -34, -41}
, {8, -29, -43}
, {20, 16, -27}
, {-20, -9, -13}
, {49, 46, -23}
, {4, -13, -38}
, {-25, -40, 14}
, {24, 3, 31}
, {-1, 5, 37}
, {6, -24, -14}
, {-14, -31, 40}
, {11, 36, 30}
, {-47, -37, 42}
, {9, -12, 32}
}
, {{-35, 39, 44}
, {-32, 0, 56}
, {69, 23, 51}
, {3, -28, -11}
, {28, -15, -16}
, {30, 6, 28}
, {-27, 55, -17}
, {37, -36, -11}
, {14, -27, -31}
, {-9, 40, -26}
, {56, 3, -4}
, {-18, -9, -21}
, {-35, 25, -22}
, {11, 33, 60}
, {54, 31, 12}
, {-71, -55, -58}
, {-47, -33, -44}
, {48, -13, -42}
, {26, 40, -2}
, {58, 67, 44}
, {-18, 1, -25}
, {-36, 0, 11}
, {-15, 41, -26}
, {10, -69, 53}
, {4, 21, 51}
, {20, -70, -29}
, {63, -3, -7}
, {86, 46, 2}
, {-26, 20, 47}
, {-49, 11, -38}
, {56, 34, 31}
, {37, -18, -30}
, {50, -3, -23}
, {58, -47, 36}
, {-31, -10, 15}
, {52, 68, -4}
, {-47, -49, -41}
, {-29, 35, -11}
, {-4, 29, 25}
, {-54, 42, -51}
, {-27, 40, -26}
, {47, 0, 32}
, {-7, -43, -38}
, {-18, -27, 34}
, {55, 61, 1}
, {4, 62, 23}
, {-46, 57, -30}
, {-28, 21, 21}
, {15, 3, 49}
, {-28, 1, -29}
, {16, -13, -37}
, {-32, -19, -40}
, {48, 16, 0}
, {16, 43, 14}
, {-37, -17, -3}
, {-45, 1, -29}
, {-22, 44, -8}
, {11, 18, 11}
, {8, 3, -20}
, {9, 0, 56}
, {58, -40, -13}
, {3, 51, -9}
, {-21, 22, -40}
, {38, 56, -41}
}
, {{-41, 20, 17}
, {42, 55, 43}
, {45, 20, 58}
, {1, 39, -38}
, {-34, 36, -31}
, {10, -8, 4}
, {-9, -35, 32}
, {16, -26, -26}
, {-30, 33, -36}
, {-20, 12, -4}
, {33, 68, 35}
, {19, -36, 48}
, {-10, -17, -5}
, {69, 38, -8}
, {-43, 34, 29}
, {-50, -43, -29}
, {-45, 26, -49}
, {-43, -41, -14}
, {-7, 9, 22}
, {1, 53, 18}
, {-28, -54, -48}
, {4, 29, -67}
, {25, -15, 39}
, {17, -48, 39}
, {-40, -33, 15}
, {-7, -20, -29}
, {-22, 20, 29}
, {-9, -7, 24}
, {44, -47, -18}
, {30, -15, 12}
, {44, 28, -8}
, {48, -4, 0}
, {36, -4, -35}
, {-53, -43, -33}
, {55, 2, -17}
, {-19, -20, -13}
, {-68, -6, 1}
, {42, 7, -8}
, {71, 48, -17}
, {-37, 18, -13}
, {-38, -56, -57}
, {45, 33, 61}
, {-23, -21, 31}
, {21, -55, -9}
, {11, 0, -14}
, {34, 22, 8}
, {23, -46, -19}
, {-54, 19, 0}
, {10, 3, 17}
, {27, -18, 18}
, {13, -34, 35}
, {-19, 33, 31}
, {-56, -46, 29}
, {-26, -46, -44}
, {-32, 44, 33}
, {20, -47, 4}
, {29, 37, -8}
, {0, -15, 21}
, {-22, -21, -11}
, {-46, 30, 52}
, {-32, 28, 72}
, {40, 86, 0}
, {25, 25, -39}
, {-23, -35, 38}
}
, {{-45, -42, -16}
, {-47, 30, -6}
, {-35, 16, 20}
, {40, -53, 30}
, {1, -54, -46}
, {-23, -39, -26}
, {-4, -45, 57}
, {1, 20, 40}
, {-40, -27, 30}
, {-53, -36, 8}
, {44, -37, 65}
, {30, 31, -23}
, {48, 46, 12}
, {-33, 16, 37}
, {-21, 27, -10}
, {17, 44, -9}
, {-12, 39, -30}
, {-25, 27, -6}
, {-38, 13, -12}
, {-25, 3, 27}
, {-50, 41, -42}
, {-36, -48, 32}
, {22, 19, -51}
, {42, 38, 62}
, {-45, 13, 16}
, {-27, 10, -35}
, {1, 23, 41}
, {-10, -11, 54}
, {-29, 19, -33}
, {22, 11, -58}
, {18, 43, -3}
, {50, 24, -4}
, {26, 51, -22}
, {-5, 1, 40}
, {-17, -59, -60}
, {29, -20, 33}
, {-15, -12, -11}
, {31, -21, 11}
, {2, 3, 40}
, {61, 22, 2}
, {5, 25, -49}
, {33, 88, 61}
, {-4, -27, 11}
, {-72, -12, -39}
, {-21, -5, -3}
, {-4, -38, 16}
, {9, 14, 40}
, {34, -54, -53}
, {30, 4, -34}
, {-51, 19, 24}
, {50, -27, -20}
, {42, 8, 6}
, {-61, -54, 2}
, {-32, -54, 6}
, {26, -17, -26}
, {0, 19, -9}
, {-39, -5, -7}
, {0, -33, -55}
, {-4, -11, 7}
, {-35, 23, -38}
, {37, 41, 31}
, {55, 11, 58}
, {-20, 7, 17}
, {2, 32, -11}
}
, {{-38, 36, 17}
, {5, 30, -56}
, {-42, -9, -18}
, {-55, -5, 4}
, {20, -26, -21}
, {6, 36, -51}
, {-31, -12, 37}
, {9, 92, 22}
, {-38, -43, 49}
, {29, 40, -39}
, {39, 53, 57}
, {-46, 22, 28}
, {37, -21, 12}
, {24, -32, 27}
, {-17, -9, -37}
, {-34, 44, -17}
, {59, 12, 31}
, {34, 9, -53}
, {16, 52, -28}
, {42, -45, 9}
, {-50, 20, 3}
, {14, -32, 27}
, {-38, -25, -32}
, {0, 33, 39}
, {37, 0, -42}
, {32, -13, -12}
, {29, 50, -26}
, {53, -38, 60}
, {-75, -9, -62}
, {-51, 32, -73}
, {31, 8, 0}
, {20, -5, 36}
, {38, -44, 29}
, {-52, 3, -25}
, {29, -53, -17}
, {14, 31, 15}
, {-27, -31, 41}
, {18, -19, -4}
, {-48, -20, 53}
, {-38, -28, -45}
, {23, -6, -5}
, {-33, 42, 43}
, {-15, -40, 24}
, {-55, 28, -5}
, {21, 33, 32}
, {46, 61, 70}
, {-48, 10, 16}
, {32, -33, 17}
, {-3, 42, 40}
, {-20, -52, -45}
, {31, 44, 2}
, {-23, 3, 40}
, {23, -13, -28}
, {47, -32, 46}
, {37, -42, -3}
, {-18, -25, -12}
, {33, -37, 33}
, {12, -13, 35}
, {-30, 44, -5}
, {17, -31, 44}
, {15, 39, -18}
, {0, 5, 0}
, {-6, 22, 5}
, {-39, 35, 17}
}
, {{36, -35, 19}
, {36, -39, 14}
, {-41, 26, -53}
, {-22, -51, 30}
, {13, 12, 34}
, {-52, 16, 23}
, {-1, -35, 39}
, {-46, 55, -31}
, {-21, -4, 18}
, {38, 29, -5}
, {-53, -27, 30}
, {-13, 22, 26}
, {-43, 5, 43}
, {34, 39, -1}
, {-39, -9, -6}
, {-38, -8, -46}
, {-34, -50, 22}
, {43, 31, -3}
, {-52, -14, 24}
, {34, -47, 18}
, {-53, 52, 8}
, {-43, -36, 5}
, {4, 0, -29}
, {-22, 0, -1}
, {-2, 22, -17}
, {-22, -4, 25}
, {47, -47, -43}
, {33, 13, 17}
, {-50, 32, -35}
, {-30, 39, -48}
, {26, -42, 6}
, {21, 20, 43}
, {-13, -21, 18}
, {12, -39, -10}
, {37, -58, -35}
, {-22, 4, -51}
, {3, -37, 1}
, {-12, 16, 12}
, {10, 29, 142}
, {26, 37, 26}
, {-47, 11, -26}
, {35, 0, -30}
, {36, -39, -51}
, {9, 36, -33}
, {-47, -41, -23}
, {-43, -9, -21}
, {3, -54, -22}
, {-8, 47, -15}
, {12, 21, -8}
, {0, -5, -48}
, {23, -16, -19}
, {-47, -37, 24}
, {41, -3, 46}
, {-45, -41, -9}
, {26, -11, -2}
, {45, 11, -8}
, {15, -44, -41}
, {6, -20, 36}
, {-27, -15, -53}
, {21, -32, 21}
, {-3, 3, 15}
, {6, 36, 0}
, {-45, -25, -23}
, {-38, 44, -23}
}
, {{26, -39, -15}
, {21, 35, 0}
, {6, 9, 1}
, {5, -6, 21}
, {-49, -15, -7}
, {-21, 9, 54}
, {14, 30, 43}
, {-62, -46, -39}
, {37, -36, 11}
, {48, 36, -14}
, {29, 28, 32}
, {-17, 7, -55}
, {-13, -32, -9}
, {47, 42, -2}
, {24, 10, 40}
, {39, 26, 53}
, {33, -29, -36}
, {-9, 46, 0}
, {11, 29, 1}
, {-16, -42, -6}
, {-6, 33, 19}
, {-7, -10, 1}
, {-49, -61, -35}
, {14, 28, 3}
, {2, -2, -14}
, {64, 2, 39}
, {-27, 27, -47}
, {-48, -29, 40}
, {-30, -11, 37}
, {-7, -12, 63}
, {28, 22, -38}
, {-27, 19, 0}
, {-28, -25, 15}
, {38, 0, -22}
, {33, 35, -19}
, {-33, -22, -18}
, {38, -3, 18}
, {28, -16, 58}
, {83, 60, 59}
, {-6, 2, 51}
, {-31, -38, 44}
, {43, 0, -25}
, {35, -24, 23}
, {-31, -4, -3}
, {-8, -1, -48}
, {-55, -1, -12}
, {-17, -47, 27}
, {-18, -9, -26}
, {-4, 31, 12}
, {17, 40, 10}
, {25, -30, 15}
, {30, 46, -48}
, {31, -51, 23}
, {49, 12, -6}
, {32, -36, -44}
, {-8, -27, 6}
, {-32, -12, 16}
, {45, 43, -8}
, {-40, -13, -17}
, {56, 55, 28}
, {-6, -20, 28}
, {48, -28, 4}
, {22, -52, 1}
, {42, -2, 32}
}
, {{-22, -20, -37}
, {31, -7, 9}
, {-14, 14, -39}
, {-47, -6, 6}
, {1, 30, 3}
, {-43, -31, -33}
, {-15, -50, -37}
, {29, 17, -23}
, {-3, 30, 0}
, {20, -30, -2}
, {-33, -12, -19}
, {34, -27, -19}
, {-6, -12, 11}
, {-63, 5, -18}
, {12, -30, -52}
, {26, -1, -29}
, {54, 51, 15}
, {-1, 14, -9}
, {-13, 35, 33}
, {5, 5, 42}
, {38, -24, -3}
, {-34, -65, -20}
, {-35, 35, -7}
, {61, 50, 23}
, {-19, 12, -3}
, {-49, 29, 9}
, {-32, 54, 8}
, {47, 38, 10}
, {28, -50, -57}
, {8, -10, 19}
, {19, 53, -14}
, {57, 34, 48}
, {-55, -58, -35}
, {1, 33, -19}
, {-10, -32, -11}
, {-10, 37, 23}
, {52, 45, 17}
, {-8, -44, -7}
, {-11, -9, -50}
, {-9, -25, 3}
, {-21, 9, -3}
, {0, 34, 23}
, {37, -48, -2}
, {-18, 19, 52}
, {41, 65, -26}
, {2, 23, 21}
, {-50, -23, 4}
, {-54, 25, -42}
, {-21, -28, -40}
, {15, -24, -48}
, {33, 14, 10}
, {19, -6, -46}
, {9, -14, 21}
, {46, -4, -5}
, {40, 3, -21}
, {-24, -14, -30}
, {-44, 19, -42}
, {-47, -38, 26}
, {8, -13, 18}
, {-13, -41, -36}
, {34, 16, -35}
, {-73, -52, -14}
, {-9, -46, -35}
, {-16, -11, -2}
}
, {{-35, 16, -41}
, {-18, -38, -28}
, {-47, -14, -46}
, {12, 51, 49}
, {8, 25, 40}
, {-19, 41, -2}
, {50, 39, -25}
, {23, 15, 30}
, {-2, -31, 56}
, {-24, 13, -37}
, {43, -24, -11}
, {-40, 57, -15}
, {-30, -8, 49}
, {3, 40, -29}
, {48, 26, 23}
, {21, 30, 30}
, {-23, -10, 46}
, {0, -21, 26}
, {38, 0, 45}
, {41, -2, -43}
, {-29, -8, -25}
, {10, 37, -18}
, {-16, 26, 53}
, {38, -14, 71}
, {-5, 42, -23}
, {-11, -54, 36}
, {45, 18, -13}
, {36, -24, -2}
, {0, 19, -26}
, {15, -20, 37}
, {8, -32, 53}
, {55, -10, -20}
, {0, 53, 47}
, {-39, 36, 34}
, {-47, -7, -52}
, {-27, 33, 13}
, {13, 23, 45}
, {9, 28, 51}
, {-59, 33, 1}
, {21, 28, 29}
, {-30, -35, -22}
, {28, 56, 29}
, {-9, -10, 31}
, {37, 17, 0}
, {4, -14, 42}
, {-23, 30, -21}
, {0, 24, -27}
, {-18, 18, 12}
, {36, 0, -34}
, {25, 44, -31}
, {46, 57, -3}
, {17, -44, 16}
, {-15, -36, -2}
, {-41, -40, 34}
, {-37, -22, -3}
, {-5, 25, 36}
, {54, 21, 51}
, {-8, 28, -12}
, {-14, 44, 54}
, {-46, -45, -27}
, {12, -39, -18}
, {24, 31, 36}
, {12, -29, 54}
, {46, -44, -34}
}
, {{1, -6, 36}
, {24, 41, -31}
, {19, 14, 51}
, {-45, -22, 26}
, {25, -13, -20}
, {39, -21, -54}
, {-13, 9, -49}
, {45, 29, -24}
, {23, 2, 24}
, {-43, -42, -47}
, {-20, 56, 63}
, {44, 55, -13}
, {47, 50, 21}
, {27, -11, 44}
, {-2, 52, 2}
, {-5, -36, -36}
, {16, 47, 51}
, {0, -49, 26}
, {-40, -8, -21}
, {-10, -11, 4}
, {-40, -9, -58}
, {5, 14, -7}
, {-20, -15, 44}
, {13, 26, -58}
, {30, 37, -12}
, {25, -18, 40}
, {-14, 52, 42}
, {23, 38, 45}
, {-23, -11, -10}
, {-13, -37, 32}
, {59, 46, -8}
, {-3, 54, 43}
, {-44, 38, -43}
, {-23, -16, -37}
, {-9, 35, -18}
, {2, 5, -6}
, {-38, -30, -9}
, {-22, 0, -13}
, {-34, 0, -6}
, {29, -27, 44}
, {9, -48, -52}
, {44, -8, 98}
, {-53, -28, -28}
, {13, -6, 12}
, {-6, 2, 12}
, {-31, 11, -26}
, {-19, 45, 13}
, {32, -14, -26}
, {21, 21, -48}
, {19, 24, -33}
, {53, 67, 38}
, {12, -44, 11}
, {-12, -16, 0}
, {45, -34, -16}
, {47, -41, -34}
, {60, -18, 21}
, {-28, -38, -2}
, {-11, -62, 25}
, {-27, -15, 51}
, {-1, -2, -14}
, {-46, -2, -44}
, {73, 53, 35}
, {-42, -27, -48}
, {14, -17, -6}
}
, {{-44, 7, 6}
, {-22, -3, 17}
, {61, 11, -5}
, {21, 35, -50}
, {40, 23, 34}
, {57, 15, -1}
, {-54, -53, 28}
, {-33, -25, -83}
, {-16, -16, -22}
, {-17, -25, 27}
, {28, 12, -35}
, {-53, 26, 3}
, {-27, 1, 25}
, {-50, 37, 52}
, {-30, 34, -32}
, {-32, -25, 17}
, {-28, -52, 28}
, {39, 44, 52}
, {0, -32, -38}
, {39, 46, -5}
, {-7, 0, -21}
, {2, 1, 39}
, {42, -43, -28}
, {33, 3, 24}
, {-48, -52, 8}
, {-7, 31, -20}
, {11, 16, 32}
, {15, 26, -30}
, {-26, -42, 33}
, {-41, 32, 14}
, {0, -55, 49}
, {49, 4, -17}
, {35, 18, 35}
, {-41, 32, -14}
, {24, 31, 23}
, {48, 1, -3}
, {31, -32, -36}
, {-45, 7, 22}
, {22, -8, -34}
, {-22, 37, 19}
, {12, 37, -5}
, {-18, -14, -26}
, {-6, -24, 20}
, {-20, 68, 7}
, {-10, 44, 50}
, {-6, 2, -22}
, {39, 12, -18}
, {48, 18, 26}
, {-20, -12, 28}
, {-19, -13, -35}
, {30, 44, -18}
, {-5, 2, 50}
, {-27, 60, 54}
, {-32, -13, 50}
, {-26, 4, 42}
, {34, 1, -28}
, {22, 35, 31}
, {-5, 14, -7}
, {28, -30, 27}
, {28, 57, -9}
, {-9, 24, 27}
, {-5, 10, -71}
, {9, 28, -20}
, {44, 15, -38}
}
, {{-12, -40, -15}
, {64, 27, -4}
, {36, -39, 22}
, {59, 37, 62}
, {48, 44, -35}
, {-27, 42, -35}
, {-18, 1, -38}
, {-13, -44, -22}
, {-14, 36, 23}
, {-33, 30, 37}
, {10, -67, 38}
, {32, 25, 15}
, {-46, -4, -41}
, {0, 14, 16}
, {25, 20, 27}
, {85, 63, 0}
, {47, -19, -47}
, {-7, 20, -1}
, {-3, 4, -31}
, {-26, -48, -30}
, {45, -21, -36}
, {-14, -39, -42}
, {-43, -50, -23}
, {-11, 9, 26}
, {11, -2, 20}
, {35, 15, -35}
, {-50, -46, -9}
, {18, 19, -53}
, {-3, 68, 58}
, {57, 52, -14}
, {-33, -70, -62}
, {27, -28, -25}
, {-14, -33, 32}
, {21, 11, 14}
, {35, -32, 6}
, {24, 14, -20}
, {-19, 2, 34}
, {29, 12, 20}
, {20, 50, -14}
, {-2, 9, 26}
, {19, -27, 0}
, {-37, -21, -67}
, {33, 0, 37}
, {18, -24, 44}
, {11, -20, 2}
, {22, 19, -45}
, {-1, 17, -12}
, {-38, 17, 27}
, {67, 33, 49}
, {30, 53, -20}
, {-29, 17, 28}
, {11, -30, 3}
, {9, 18, 11}
, {24, 21, -18}
, {-13, 16, 25}
, {46, -12, -27}
, {30, 35, -17}
, {-4, 50, 16}
, {-33, -16, 16}
, {28, 53, 30}
, {18, 4, 5}
, {-12, -33, -77}
, {-37, -58, 10}
, {52, 1, 4}
}
, {{18, 16, 14}
, {10, -47, -3}
, {-11, -33, 42}
, {-32, -28, 37}
, {44, 40, -37}
, {-25, -31, -69}
, {56, 46, -25}
, {7, 31, -5}
, {10, 3, 36}
, {14, 3, 31}
, {63, 12, -26}
, {33, -36, 24}
, {-15, -5, -22}
, {58, 23, 41}
, {50, -53, 46}
, {16, -29, -23}
, {-45, -22, -5}
, {3, -48, -42}
, {-30, -15, -29}
, {9, -2, 22}
, {35, -34, -25}
, {-20, -31, -37}
, {-29, 51, 52}
, {35, -30, -12}
, {35, 44, 3}
, {5, -34, -11}
, {54, 47, 53}
, {21, 14, 72}
, {47, -58, 13}
, {-13, 19, 28}
, {25, -35, 57}
, {3, -24, 34}
, {3, -39, -11}
, {21, 26, 24}
, {-12, -7, 0}
, {-40, -44, -24}
, {45, -17, -25}
, {-41, 64, 12}
, {5, -23, -18}
, {15, 10, 43}
, {34, 34, -34}
, {52, -15, 90}
, {-44, 29, -11}
, {17, -17, -11}
, {29, 38, -35}
, {17, 12, 42}
, {-7, 7, 21}
, {7, -43, 15}
, {19, 19, -19}
, {-27, 2, 5}
, {23, -29, 0}
, {10, -3, -34}
, {15, -44, -28}
, {15, -2, -45}
, {11, -57, 5}
, {-4, -37, 20}
, {48, -33, -27}
, {51, -11, -15}
, {49, -39, 40}
, {-50, 17, -58}
, {-12, 31, -18}
, {-23, 67, 65}
, {40, -39, 30}
, {-31, -38, 30}
}
, {{0, 10, -21}
, {9, -13, -44}
, {-23, 8, 14}
, {-49, -32, -20}
, {-29, -20, -20}
, {3, 39, 52}
, {35, 13, 0}
, {-39, -13, -9}
, {-4, 7, 49}
, {50, 27, -50}
, {-56, -6, -31}
, {15, 16, -8}
, {-43, 22, -53}
, {1, 11, -13}
, {-46, -25, 31}
, {24, 52, -6}
, {6, -47, -39}
, {-8, 33, 44}
, {11, -37, -30}
, {-20, 8, 14}
, {51, -26, -42}
, {33, -14, -47}
, {17, 20, -29}
, {-11, 15, 31}
, {-7, -32, -38}
, {51, 5, 32}
, {18, -9, 42}
, {-66, 24, 7}
, {41, 44, 36}
, {9, -26, -28}
, {-36, -14, 11}
, {44, 7, 38}
, {33, -4, -12}
, {55, 10, -11}
, {-10, 16, 5}
, {53, -20, -33}
, {51, 11, 49}
, {-19, 5, -53}
, {-20, 17, -26}
, {-56, -40, -1}
, {-40, -51, -41}
, {16, -40, -8}
, {-50, -7, -14}
, {37, 45, 57}
, {-49, 38, 23}
, {31, -27, 40}
, {45, 28, -21}
, {41, 15, -50}
, {44, 44, -26}
, {-37, 48, -24}
, {36, -12, 28}
, {31, -31, -6}
, {-50, -11, 36}
, {29, -1, 26}
, {52, -36, 33}
, {-21, -2, 15}
, {-42, -4, 10}
, {-7, 11, 4}
, {-11, 19, -38}
, {-6, 39, -12}
, {40, -22, 0}
, {26, -73, 5}
, {1, 22, 11}
, {-33, -53, 49}
}
, {{44, -31, -11}
, {53, 20, 12}
, {36, -43, -41}
, {1, 49, 21}
, {0, -26, -44}
, {-35, -12, 47}
, {-37, 1, -36}
, {-11, 25, -20}
, {29, -53, -8}
, {4, 24, -7}
, {-20, -22, -39}
, {0, 21, 50}
, {-48, 3, -38}
, {36, -16, 1}
, {24, -23, 47}
, {-6, 14, -20}
, {41, -29, -32}
, {17, 7, -6}
, {-29, -54, -39}
, {-3, -11, -40}
, {-18, 37, 5}
, {-14, -39, -22}
, {-42, -25, 10}
, {62, -15, -13}
, {36, -2, -15}
, {41, -31, -8}
, {32, 30, -4}
, {68, 94, 4}
, {70, 8, 54}
, {44, 21, 53}
, {43, -19, 10}
, {-11, -3, -39}
, {0, -35, -39}
, {3, 38, -30}
, {16, 24, -16}
, {-49, 21, 9}
, {25, 11, -50}
, {-11, 21, 59}
, {54, 55, 44}
, {54, 36, 31}
, {-14, -6, 29}
, {36, 0, 84}
, {14, 43, 18}
, {-48, -52, -72}
, {-3, 26, -14}
, {21, -10, 27}
, {20, 31, -50}
, {-39, 22, -37}
, {-44, -52, 19}
, {45, 31, 49}
, {-20, 46, 11}
, {0, -16, 27}
, {-41, 32, 21}
, {46, -44, -49}
, {-45, 12, 21}
, {57, -31, 58}
, {9, -36, -8}
, {25, 19, 73}
, {-9, 27, 43}
, {-31, -57, -15}
, {-24, 6, -49}
, {-30, -4, 30}
, {44, 15, 23}
, {-19, -9, 62}
}
, {{13, 1, 5}
, {8, -28, 2}
, {50, -20, 23}
, {-21, -22, -31}
, {46, -11, -8}
, {-47, 17, -28}
, {-10, -33, 38}
, {82, 17, 47}
, {-44, -9, 62}
, {45, -37, -1}
, {-48, 16, 27}
, {10, 35, 30}
, {40, 37, 38}
, {-16, -22, -28}
, {-9, -5, 41}
, {-17, 0, 20}
, {8, -37, 56}
, {40, -36, -2}
, {31, 10, 50}
, {19, 34, -36}
, {-42, 10, -7}
, {18, -7, 26}
, {47, 16, 34}
, {36, 19, -6}
, {-18, -14, 13}
, {-25, 12, 21}
, {-49, -6, 6}
, {-53, 5, 6}
, {10, 7, -50}
, {-49, -36, 41}
, {24, 1, -15}
, {-40, -26, -39}
, {-19, -2, 4}
, {-16, -24, 0}
, {-32, 16, 36}
, {45, 35, -21}
, {29, 16, 76}
, {-3, 31, 40}
, {-15, -54, -35}
, {-55, 22, 4}
, {-42, 25, 42}
, {-30, -18, -6}
, {4, 26, -28}
, {17, -44, -11}
, {23, 25, -28}
, {22, 14, -3}
, {28, -7, 16}
, {-30, 28, -54}
, {12, -2, 15}
, {-31, -17, 23}
, {-42, -17, -28}
, {-28, -43, 39}
, {9, -11, 17}
, {7, 4, -12}
, {-49, -25, -5}
, {1, 16, -28}
, {15, -48, 26}
, {32, -3, 38}
, {6, -54, -5}
, {-41, -27, -34}
, {56, 0, -34}
, {-49, -6, 30}
, {40, -26, 10}
, {9, 41, 40}
}
, {{15, -23, -41}
, {21, 5, 24}
, {-1, 5, -40}
, {16, -31, 8}
, {-8, 24, -10}
, {35, 65, 2}
, {-53, -23, 12}
, {-35, -68, -57}
, {95, -7, -8}
, {-44, -25, 13}
, {-15, -19, -41}
, {-25, -7, 35}
, {11, 29, -20}
, {24, -6, 25}
, {36, -55, -67}
, {46, -51, 9}
, {14, -19, -63}
, {54, 29, 40}
, {20, -14, -48}
, {-31, 57, 24}
, {23, 82, 24}
, {-24, -39, -34}
, {22, -66, -23}
, {-31, 28, 8}
, {17, -29, -38}
, {-8, -12, -2}
, {2, -52, 18}
, {12, -18, -31}
, {5, -18, -12}
, {11, -8, 25}
, {-23, -75, -38}
, {-18, -22, 56}
, {11, -46, -8}
, {-33, -21, 56}
, {27, -17, -18}
, {-17, 13, 13}
, {38, 14, 28}
, {-48, -8, 36}
, {49, 23, 43}
, {39, -47, -9}
, {55, 34, 24}
, {12, -11, -35}
, {31, -15, 26}
, {9, -29, 44}
, {-33, 37, 8}
, {-43, -12, 4}
, {-18, 52, -46}
, {23, -2, 47}
, {50, 7, -61}
, {12, 0, 60}
, {-6, 3, -33}
, {24, 25, 54}
, {-24, 11, -30}
, {42, 28, 32}
, {-5, 40, -47}
, {-33, 16, -12}
, {-47, -13, -12}
, {-39, -27, -29}
, {-32, -29, -35}
, {46, 9, 16}
, {-9, 44, 64}
, {34, -11, -13}
, {-1, -25, -39}
, {-46, 43, 38}
}
, {{-34, -21, -34}
, {47, 48, -20}
, {-37, -33, 49}
, {46, 38, -17}
, {-41, -18, -1}
, {33, 50, 44}
, {-28, -52, -24}
, {-27, -92, -80}
, {6, 39, 6}
, {29, 1, 0}
, {7, -64, 17}
, {-21, 5, -52}
, {-8, 26, 4}
, {39, -41, 3}
, {49, -38, 28}
, {20, 23, -16}
, {-12, -26, -24}
, {-35, 37, -14}
, {7, -21, -1}
, {1, 56, 48}
, {53, -38, 14}
, {51, -26, 24}
, {-33, 7, 22}
, {-1, -16, -20}
, {52, -40, -28}
, {-6, 20, -28}
, {-32, -24, -21}
, {40, -13, -41}
, {33, 3, 61}
, {0, -22, -41}
, {-27, 13, -21}
, {-50, 12, -3}
, {-2, -29, 22}
, {17, -37, 50}
, {28, -27, 18}
, {33, -7, 21}
, {0, -10, -16}
, {-18, 45, 16}
, {-11, 56, 55}
, {-31, -32, 31}
, {-25, 17, -25}
, {-8, -13, -22}
, {14, -24, -29}
, {58, 66, 8}
, {-9, -3, 1}
, {8, -25, -34}
, {17, 34, 2}
, {-5, 22, 10}
, {35, 9, 31}
, {31, 27, -14}
, {35, 0, 40}
, {23, -10, 58}
, {-6, -31, 3}
, {55, -2, 6}
, {38, -2, 44}
, {-41, 47, 34}
, {46, 41, 25}
, {-33, 11, 29}
, {-53, -22, 32}
, {16, 47, 24}
, {-28, 26, 28}
, {-51, -72, -79}
, {-53, -35, 16}
, {-8, 36, 1}
}
, {{26, -33, -9}
, {-2, 77, -24}
, {0, 55, -50}
, {-13, -10, -23}
, {1, -34, -9}
, {29, 53, 59}
, {-18, 3, -9}
, {-44, -27, -38}
, {76, -55, 73}
, {-40, 55, -33}
, {-14, -31, 14}
, {-2, -11, 22}
, {2, 23, -15}
, {-43, -9, -69}
, {39, -45, 30}
, {57, -24, -23}
, {-12, -28, -23}
, {11, 61, 55}
, {41, -7, -23}
, {-15, -30, -12}
, {31, 49, 20}
, {-16, 49, -28}
, {-5, -46, 37}
, {-4, 15, -14}
, {-18, -6, -40}
, {-44, 7, -16}
, {-24, 5, 14}
, {-22, -16, 20}
, {23, -1, 59}
, {-7, 2, 7}
, {19, -19, -2}
, {-37, -1, -43}
, {15, -24, 51}
, {-19, 4, 10}
, {-11, -17, 70}
, {-45, 10, 34}
, {-19, 44, 39}
, {-44, 4, -2}
, {4, 17, -19}
, {-22, 59, 60}
, {46, 17, 0}
, {9, 40, -18}
, {4, -42, 12}
, {72, 34, -57}
, {-27, -69, -47}
, {-43, 11, -4}
, {0, -37, 52}
, {-9, -20, 37}
, {26, 3, 56}
, {-41, 57, -8}
, {29, -53, -21}
, {32, -23, 10}
, {20, -15, 35}
, {37, 12, -26}
, {-57, -15, 36}
, {30, 51, -10}
, {19, -18, 15}
, {-35, -18, -40}
, {-21, -25, -6}
, {38, -23, 44}
, {13, 5, -16}
, {-17, -13, 34}
, {41, 43, 7}
, {-49, -23, -29}
}
, {{-12, -7, -19}
, {-14, 14, -15}
, {-28, 31, -50}
, {17, -52, 24}
, {19, 34, 0}
, {-23, -16, 1}
, {-4, 5, -52}
, {33, -4, 42}
, {15, -8, -50}
, {-34, -8, 29}
, {42, -13, -38}
, {-17, 0, -15}
, {19, 41, -20}
, {31, -34, -47}
, {-47, 7, -17}
, {-6, -26, -13}
, {-54, -51, 8}
, {-51, -8, 24}
, {-10, 27, 23}
, {-20, -29, -22}
, {-55, -44, 27}
, {-37, -23, -35}
, {-50, -46, -47}
, {16, -22, 31}
, {-51, -18, 24}
, {-31, -43, 10}
, {7, -31, 19}
, {-41, 22, 32}
, {-69, 7, -42}
, {33, 8, 22}
, {-24, 13, 0}
, {-37, -37, 33}
, {39, -52, 35}
, {3, -27, -49}
, {39, 1, -15}
, {2, 38, 16}
, {-31, -46, -16}
, {-39, 10, 1}
, {-17, -48, -9}
, {-15, -7, 16}
, {35, 20, 32}
, {-4, 14, 24}
, {26, -17, 30}
, {-18, -25, -45}
, {13, -46, -31}
, {-9, 11, 25}
, {34, 8, 43}
, {24, 15, 31}
, {-57, -41, 14}
, {-11, -47, -54}
, {3, 16, 33}
, {24, -18, -37}
, {-48, 15, 23}
, {31, -50, -11}
, {29, 35, -10}
, {-18, -5, -24}
, {-5, -22, 38}
, {15, 42, 7}
, {-51, -47, 3}
, {-17, -46, -48}
, {-43, -24, -49}
, {-41, -51, -46}
, {-46, -6, 14}
, {1, 46, -49}
}
, {{1, -50, 51}
, {-9, -34, 8}
, {-61, -18, 28}
, {54, -24, -12}
, {-12, 36, -8}
, {0, 18, 34}
, {-25, 33, 0}
, {34, 44, 17}
, {14, -41, 38}
, {40, -20, 52}
, {-46, 17, -47}
, {-25, -43, -19}
, {20, 6, -32}
, {-54, -63, -9}
, {31, -52, -51}
, {25, 7, -16}
, {27, -48, 4}
, {15, 1, 17}
, {19, -44, -12}
, {65, 17, -47}
, {5, -44, 11}
, {-13, -22, -51}
, {1, 37, -14}
, {-48, -22, 24}
, {21, 16, -41}
, {-22, 10, -32}
, {-9, -38, -50}
, {32, 30, -13}
, {3, -42, -34}
, {-6, 11, 22}
, {10, 27, -14}
, {-31, 53, 46}
, {58, 38, -5}
, {-1, 20, 42}
, {50, 51, 32}
, {-5, 9, 46}
, {53, 27, 36}
, {-17, 5, -41}
, {6, 20, -17}
, {-1, -21, -4}
, {42, 4, -50}
, {-59, -48, -45}
, {-33, -9, -30}
, {43, 72, 2}
, {46, 38, -48}
, {-29, -6, 6}
, {19, 60, -41}
, {31, 27, -37}
, {66, -34, -46}
, {-25, 10, 30}
, {70, -25, 31}
, {1, 18, 27}
, {-19, -13, 42}
, {-11, -2, 20}
, {-23, 7, 6}
, {9, -49, -39}
, {-41, 43, 11}
, {-13, -62, -29}
, {-4, 48, -2}
, {7, -16, 15}
, {28, 23, -2}
, {-2, 6, -32}
, {43, 15, -3}
, {-34, -15, -5}
}
, {{25, -1, 21}
, {-38, -22, -25}
, {6, -31, 4}
, {6, -27, 2}
, {-6, -42, -52}
, {42, 41, -57}
, {9, -1, 0}
, {26, 56, -17}
, {-9, 7, 17}
, {-32, -7, 18}
, {-49, -19, 8}
, {28, 0, 25}
, {-55, -45, 36}
, {-20, 48, 11}
, {-25, -11, -21}
, {-21, 39, 25}
, {-2, -6, -33}
, {-9, 32, 8}
, {-51, -32, 49}
, {66, -18, 26}
, {15, 31, -18}
, {15, -25, 1}
, {-33, -27, 34}
, {-24, -38, -46}
, {29, -32, 15}
, {52, 26, 15}
, {-52, -14, -5}
, {33, 28, -48}
, {-20, 21, 39}
, {-30, 27, -21}
, {-22, -36, 0}
, {-53, 27, -22}
, {27, -49, 41}
, {-48, -9, -9}
, {-20, 55, 8}
, {-2, 35, 43}
, {0, -37, -29}
, {-15, -47, -37}
, {-41, -29, -39}
, {7, -29, 27}
, {46, 42, -12}
, {-5, -38, -4}
, {46, 4, -18}
, {50, -7, 32}
, {-32, -39, 18}
, {-5, -33, -5}
, {34, -13, -47}
, {-47, -17, -41}
, {29, 2, -52}
, {-50, 11, -47}
, {41, 31, -25}
, {-19, 40, -35}
, {-36, -34, -19}
, {-35, 4, 45}
, {27, -21, -11}
, {-31, -52, 29}
, {47, -16, -47}
, {26, -7, 0}
, {25, 16, 34}
, {19, -9, -44}
, {-30, -47, 1}
, {5, 4, 30}
, {12, -52, -29}
, {-13, -24, 5}
}
, {{-12, 23, -2}
, {17, -49, 18}
, {-6, -19, -31}
, {-41, -53, -10}
, {37, 36, -12}
, {-53, 14, 24}
, {11, -25, 11}
, {12, 83, 107}
, {4, -26, 45}
, {-26, -38, 12}
, {27, -34, -19}
, {52, -44, 23}
, {-19, 17, -23}
, {-36, 14, 46}
, {-38, -8, -14}
, {-32, 26, -26}
, {51, -34, -16}
, {-1, 0, -18}
, {-31, -22, 58}
, {18, 57, -29}
, {43, 4, -51}
, {6, 55, 6}
, {-14, -10, 30}
, {-22, -19, -19}
, {38, -16, -31}
, {-10, -16, -35}
, {56, 50, -7}
, {64, 42, -14}
, {-48, -10, -9}
, {11, -18, 37}
, {25, 49, 13}
, {-23, 39, 7}
, {-17, 0, 33}
, {-10, -31, -15}
, {6, 13, -72}
, {-23, -26, -50}
, {3, 56, 8}
, {35, 27, -9}
, {-46, 8, -15}
, {-10, -10, 22}
, {45, 32, 49}
, {55, 26, 0}
, {-39, 25, 37}
, {-38, 8, 17}
, {9, 63, 33}
, {33, 36, 64}
, {-12, 5, 25}
, {28, -15, 0}
, {0, 29, 43}
, {-29, -49, -31}
, {67, 72, 37}
, {-4, 13, -38}
, {-6, 57, -37}
, {32, -43, -44}
, {2, -53, 24}
, {4, 41, 4}
, {-52, -38, 26}
, {16, -31, 46}
, {8, 41, 48}
, {-32, 4, -34}
, {52, 23, -58}
, {-26, -8, 1}
, {4, 15, 47}
, {-39, 15, -6}
}
, {{-33, -50, 45}
, {-18, 3, 62}
, {-22, 47, 15}
, {-44, -17, -33}
, {-28, -28, -47}
, {25, -10, -15}
, {-6, -12, 3}
, {3, 0, -55}
, {-59, 30, -50}
, {9, -48, -54}
, {32, 47, 47}
, {40, -10, 32}
, {-55, 18, 0}
, {-27, 50, 23}
, {-46, -12, 15}
, {37, -47, -45}
, {43, 31, 55}
, {-57, -20, 12}
, {-18, 41, -44}
, {31, 26, -19}
, {50, -5, -1}
, {43, -44, -13}
, {18, -20, 16}
, {-34, 8, -38}
, {0, -11, 13}
, {1, -39, 44}
, {-42, 12, -8}
, {12, 58, -36}
, {51, -39, 25}
, {2, 24, 59}
, {-30, 76, -40}
, {0, -46, 26}
, {7, -16, 30}
, {6, 0, -39}
, {46, -13, 39}
, {36, -9, 26}
, {-56, -38, -62}
, {-13, 91, 40}
, {95, 16, 101}
, {-8, -24, -11}
, {0, 39, -27}
, {89, 90, 12}
, {24, -64, -11}
, {39, -64, -63}
, {17, -6, 37}
, {-34, -18, -29}
, {11, -31, 8}
, {-18, -34, 21}
, {-44, 2, -60}
, {-29, -35, -19}
, {39, -15, 0}
, {-42, 0, -1}
, {-21, -10, 21}
, {28, -14, 6}
, {35, 16, 21}
, {22, 14, 3}
, {3, 50, -25}
, {20, 20, 20}
, {4, 15, 14}
, {-30, -3, 8}
, {-50, 22, 56}
, {36, 30, 67}
, {-49, 40, 33}
, {62, -2, -31}
}
, {{-19, -13, 48}
, {49, 56, 8}
, {-15, -13, -16}
, {-4, 15, 49}
, {11, 23, -27}
, {43, 23, -11}
, {-18, 4, -2}
, {6, 10, -16}
, {39, 47, -11}
, {39, -4, -12}
, {1, 4, 0}
, {37, -31, -56}
, {20, -25, -52}
, {-33, 45, 15}
, {-21, 49, 43}
, {-36, -32, -41}
, {27, -13, -32}
, {58, -39, 55}
, {45, -50, 14}
, {-31, -11, 29}
, {20, 20, 4}
, {12, -11, -8}
, {-59, 0, -22}
, {-38, 18, -29}
, {11, -16, -6}
, {3, -24, 36}
, {38, -1, 7}
, {-53, -15, -61}
, {-24, -40, -14}
, {7, 30, 15}
, {-24, -22, -47}
, {25, -6, 10}
, {-25, 12, 41}
, {3, 18, 41}
, {58, -2, -18}
, {12, 53, -39}
, {-39, 31, 50}
, {18, 53, 54}
, {9, 46, 60}
, {-60, 17, -19}
, {-23, -13, -16}
, {-81, -32, -38}
, {1, 13, 31}
, {-12, 6, -15}
, {-4, -67, 46}
, {46, -12, -32}
, {13, -17, -7}
, {47, 17, -19}
, {15, 37, 17}
, {35, -28, 54}
, {-27, -44, -56}
, {-6, -43, -21}
, {65, 35, -28}
, {9, 35, -18}
, {44, 40, -11}
, {-1, 42, -11}
, {-28, -15, -46}
, {-12, 13, -10}
, {-43, 0, -13}
, {-35, 36, 46}
, {15, -21, 44}
, {-19, -41, -56}
, {7, -33, -14}
, {6, 47, 44}
}
, {{-1, -19, -24}
, {20, -1, 77}
, {11, -29, 22}
, {-43, -25, -13}
, {14, 46, 34}
, {36, -41, -43}
, {-33, 5, -34}
, {6, -9, -51}
, {-18, -23, -3}
, {23, -27, 34}
, {49, 10, 4}
, {17, -21, 48}
, {-20, 25, 19}
, {-24, 5, 37}
, {27, 37, 10}
, {25, 20, -72}
, {-16, 32, -24}
, {6, 17, -48}
, {-39, 31, 9}
, {-26, -22, -17}
, {-24, -5, 1}
, {-28, -35, -3}
, {-6, -44, 2}
, {22, 8, -14}
, {-27, 37, -7}
, {15, -37, 3}
, {20, -2, -41}
, {50, 17, 5}
, {-57, 39, 11}
, {-64, 58, -56}
, {2, 9, 8}
, {-6, -51, -13}
, {-22, 45, -6}
, {-24, -12, -17}
, {16, -7, 48}
, {45, 5, 14}
, {-42, -35, -62}
, {21, -8, -11}
, {86, 59, -41}
, {15, 24, -27}
, {33, -37, -1}
, {-23, 69, 42}
, {-10, -1, 26}
, {-12, -6, -67}
, {17, 51, -35}
, {28, 28, 0}
, {-36, -27, -34}
, {-14, 42, -4}
, {14, 0, 33}
, {-15, 43, 29}
, {34, -21, -27}
, {7, 36, -47}
, {42, -23, 44}
, {40, 21, -56}
, {1, 30, -39}
, {-8, 29, -24}
, {-19, -15, -50}
, {-43, 16, -29}
, {-13, 39, 24}
, {5, 5, 1}
, {42, -3, 55}
, {26, 60, 48}
, {-41, 15, 1}
, {-45, 40, -11}
}
, {{3, 22, 36}
, {70, 30, -28}
, {21, 32, -43}
, {-38, -13, -7}
, {-46, 48, 16}
, {16, 32, 42}
, {-27, 29, 23}
, {-66, 3, -40}
, {27, 49, -63}
, {-32, -23, -40}
, {-14, 14, 8}
, {-32, 1, -48}
, {-24, 2, 16}
, {1, -12, 26}
, {-10, -34, 9}
, {50, 19, -44}
, {11, 32, -22}
, {-7, -23, 44}
, {37, -40, 7}
, {37, 14, -20}
, {-8, 32, -11}
, {38, -38, -12}
, {-31, -50, -17}
, {14, 37, -20}
, {58, -26, 17}
, {19, 55, -18}
, {-10, -35, 41}
, {39, 33, -3}
, {-11, -22, 17}
, {1, 66, 8}
, {-37, 28, -64}
, {-38, 2, -9}
, {-8, 51, -30}
, {-14, -49, -34}
, {-4, 5, -12}
, {16, -25, 20}
, {4, -39, 1}
, {-11, 0, 80}
, {42, -26, 85}
, {31, 0, 81}
, {15, -18, 44}
, {9, 6, 25}
, {48, -51, 33}
, {12, -37, -48}
, {-11, 26, 16}
, {5, 2, -19}
, {39, -55, 54}
, {-34, -50, 40}
, {-39, 13, -32}
, {33, -34, 10}
, {-33, -20, -22}
, {2, -52, 32}
, {23, -46, 9}
, {45, -11, 3}
, {33, -20, -9}
, {-36, 35, -3}
, {-26, 21, -35}
, {43, 43, 59}
, {36, -6, -47}
, {-38, -45, -31}
, {1, 68, 8}
, {25, -28, 28}
, {13, 3, -13}
, {50, 46, 23}
}
, {{-24, -34, -9}
, {24, 42, -21}
, {-36, 53, -3}
, {-44, -32, 7}
, {-11, -45, 41}
, {6, -24, 59}
, {17, -71, -40}
, {11, -36, -15}
, {-63, -13, -63}
, {-66, -39, -7}
, {62, 6, -3}
, {-51, 8, -37}
, {-49, -53, 33}
, {26, 47, 52}
, {-14, 39, 45}
, {-13, -36, 6}
, {-9, 4, -4}
, {-52, -86, -21}
, {-40, 36, 30}
, {59, 14, 25}
, {-33, -14, -59}
, {29, -45, 40}
, {21, 25, -3}
, {11, -7, -35}
, {43, 19, -26}
, {-39, 2, -21}
, {19, -29, -14}
, {40, 64, 67}
, {2, 61, -51}
, {45, 37, 9}
, {-34, 18, -4}
, {-12, -15, -53}
, {23, -5, -65}
, {13, -18, -19}
, {46, 12, 0}
, {62, -1, -3}
, {-38, -26, -13}
, {66, -12, 73}
, {57, -11, 93}
, {20, 24, 34}
, {14, -44, -60}
, {102, 37, 72}
, {37, -62, -4}
, {4, -36, -19}
, {-9, 54, 41}
, {37, -38, -42}
, {2, -57, 38}
, {-12, -42, -9}
, {-9, 31, -55}
, {32, -47, -25}
, {2, -52, 33}
, {31, -26, -21}
, {-40, -48, -19}
, {-35, -22, 20}
, {-18, -5, 81}
, {58, -5, 21}
, {-50, -49, 27}
, {81, 66, -37}
, {-19, -4, 13}
, {22, -52, -44}
, {42, 1, 36}
, {66, 37, 35}
, {-40, -32, 52}
, {11, -23, -15}
}
, {{36, 58, -8}
, {-4, 9, -26}
, {15, -15, 7}
, {8, 9, -11}
, {4, -20, -21}
, {-39, 46, -3}
, {-27, -38, -64}
, {-60, -98, -78}
, {49, 21, 35}
, {13, 0, 11}
, {-81, 30, 15}
, {-4, -48, -12}
, {5, 36, -43}
, {19, -37, -40}
, {12, 56, -13}
, {5, -29, -44}
, {34, -26, -65}
, {-16, -1, 44}
, {-34, 17, 20}
, {39, 35, 46}
, {1, 39, -27}
, {-54, 3, -2}
, {17, 22, 27}
, {-23, -26, -62}
, {0, 0, -2}
, {43, 8, 18}
, {-20, -25, -10}
, {-11, -72, -37}
, {-21, 51, 26}
, {-3, 22, -68}
, {-17, 54, -44}
, {-11, 10, -1}
, {-48, -14, -45}
, {-6, 0, 30}
, {30, -2, 0}
, {40, 20, 44}
, {-4, -15, -28}
, {-8, 18, -30}
, {63, 35, -14}
, {0, 11, 35}
, {57, -29, 22}
, {-18, -55, -32}
, {-38, -28, -34}
, {6, 21, 15}
, {-27, -15, 2}
, {27, 39, -5}
, {-9, 33, -4}
, {23, 18, -29}
, {56, 11, 22}
, {-19, 6, 30}
, {-53, 0, -53}
, {7, 52, -2}
, {64, -9, 14}
, {29, 53, 10}
, {51, 30, -9}
, {-38, 13, 52}
, {2, 51, 32}
, {1, -5, 15}
, {-42, 34, 9}
, {12, 43, 18}
, {45, 46, 50}
, {34, -8, -35}
, {21, 5, -37}
, {34, -32, 19}
}
, {{4, 9, -55}
, {46, -15, 43}
, {39, 7, 19}
, {-40, 49, 19}
, {15, 36, 41}
, {-4, 29, 34}
, {26, 5, 10}
, {-64, -14, -60}
, {8, -41, -13}
, {-41, -18, -11}
, {-49, -37, 34}
, {-51, -33, 7}
, {21, -50, -46}
, {-18, -8, 29}
, {-29, -31, -59}
, {45, -28, 55}
, {41, -57, -33}
, {-20, 43, 22}
, {0, -45, -20}
, {8, 47, -10}
, {61, 14, 40}
, {-37, 7, -18}
, {14, 11, 24}
, {6, -63, -55}
, {-40, -6, -39}
, {15, 50, 47}
, {39, -5, -39}
, {-45, -35, -42}
, {7, 25, 28}
, {0, 13, -11}
, {-32, 30, -69}
, {2, -37, 48}
, {30, -23, 38}
, {24, 16, 26}
, {0, 20, -10}
, {-2, -43, -47}
, {45, -2, 16}
, {60, 36, 42}
, {50, -4, 6}
, {-13, 67, 26}
, {-15, 34, -5}
, {35, 33, -13}
, {-36, 26, 27}
, {-7, 4, -41}
, {-47, 49, -33}
, {0, 8, 31}
, {43, -3, -23}
, {33, -9, 6}
, {-18, 30, -15}
, {3, -29, 13}
, {-28, -56, 13}
, {44, -7, 28}
, {-41, -59, 38}
, {13, -6, 44}
, {-8, 5, -16}
, {30, 3, 20}
, {-16, -42, 19}
, {-25, 9, 27}
, {44, -49, -25}
, {59, 54, -31}
, {0, 13, 59}
, {43, -58, 10}
, {39, -50, -37}
, {-46, -27, -50}
}
, {{27, -22, -13}
, {32, 36, 7}
, {16, 54, -4}
, {50, -35, -19}
, {20, -7, 26}
, {11, -7, -25}
, {2, 22, 42}
, {5, -14, 17}
, {28, -16, 0}
, {19, 22, -4}
, {-50, 10, 0}
, {37, 28, -64}
, {51, -32, -14}
, {-1, 25, 25}
, {-26, -13, -18}
, {41, -4, -28}
, {6, 24, 2}
, {-29, 26, -9}
, {-27, -55, -49}
, {-53, -4, 47}
, {34, 6, -8}
, {-7, 42, -6}
, {-44, 0, 1}
, {2, 48, -61}
, {-55, -8, 52}
, {6, -13, -17}
, {4, 15, -1}
, {-56, -20, -43}
, {9, -6, 38}
, {-27, -19, -12}
, {-56, -27, 10}
, {-6, -40, 17}
, {21, -28, 46}
, {50, 1, 29}
, {7, -25, 13}
, {-32, -15, 29}
, {-37, -61, 19}
, {-11, 15, 8}
, {-56, 43, 45}
, {-36, -66, 31}
, {35, 0, 37}
, {-47, -21, -4}
, {-4, 30, -3}
, {-3, 33, -67}
, {-53, -1, -26}
, {38, -38, 33}
, {-28, -17, -41}
, {-49, 34, -46}
, {25, -40, -45}
, {-43, -58, 4}
, {-30, -12, -47}
, {40, -3, -14}
, {19, 19, -55}
, {-46, 26, 26}
, {31, -12, 1}
, {-20, -9, 44}
, {-52, -12, -49}
, {-20, 14, -30}
, {-41, 34, -1}
, {-29, -29, 30}
, {55, 21, 30}
, {4, -15, 45}
, {-19, -30, -46}
, {-26, 44, -32}
}
, {{46, -25, -13}
, {32, 4, 82}
, {57, 32, 59}
, {-12, -18, -38}
, {-26, 15, 5}
, {-4, 24, -54}
, {49, -22, 7}
, {19, -64, 27}
, {-45, -22, -39}
, {-51, -15, -14}
, {73, 32, 29}
, {24, 12, 44}
, {12, -10, -32}
, {75, 59, 13}
, {25, 32, 65}
, {-35, -11, -97}
, {-25, -4, -68}
, {40, -29, -8}
, {-18, -38, 55}
, {39, 25, -41}
, {-18, -35, 17}
, {61, 5, -36}
, {-8, -6, -50}
, {-36, 11, -3}
, {-33, 74, -28}
, {29, -21, -33}
, {-25, -29, 26}
, {59, -32, 99}
, {-49, -29, 24}
, {-55, 35, 11}
, {17, -13, 2}
, {46, -41, -29}
, {-40, 54, -41}
, {14, -45, 8}
, {-41, -29, 23}
, {2, -22, 30}
, {-54, 34, -1}
, {1, 9, 60}
, {-9, 53, -27}
, {27, 46, 12}
, {-30, 12, -41}
, {12, 92, 94}
, {-43, -34, 26}
, {-32, 22, -9}
, {-3, 49, -2}
, {0, -13, -17}
, {-7, 20, -52}
, {27, -22, 33}
, {8, 30, -35}
, {-4, -27, -8}
, {-59, -32, -2}
, {-3, 35, -43}
, {43, -26, 15}
, {-46, -39, 17}
, {-3, -69, 6}
, {38, 31, -36}
, {34, -7, -28}
, {21, 67, 7}
, {3, -19, -49}
, {-27, 14, -32}
, {61, -70, 1}
, {34, 69, -19}
, {-51, 19, 25}
, {-36, 68, -53}
}
, {{42, 26, 16}
, {-39, 7, 21}
, {-58, -21, 21}
, {-4, -18, 1}
, {3, -24, -25}
, {12, -40, -8}
, {-22, 7, -34}
, {17, 84, 104}
, {19, 54, -9}
, {42, 12, 21}
, {-45, -37, 20}
, {32, -42, 33}
, {1, 24, 14}
, {-59, -1, -7}
, {3, -48, -1}
, {-28, 37, 52}
, {55, 16, 5}
, {-3, 48, 5}
, {15, -31, 1}
, {-28, 10, 24}
, {49, 9, -42}
, {4, -18, -33}
, {-38, 40, -1}
, {19, 17, 53}
, {2, -37, 17}
, {-20, -14, -44}
, {-12, -2, 1}
, {-29, -11, -39}
, {-21, -19, 24}
, {24, 46, 45}
, {-12, -58, 3}
, {-38, -40, -23}
, {-39, 39, 15}
, {33, 40, 44}
, {33, 5, -50}
, {45, -20, 7}
, {15, 46, 36}
, {2, -5, 0}
, {-59, 3, -53}
, {-22, 0, 43}
, {-35, 41, 31}
, {1, -36, -54}
, {-16, 51, 5}
, {1, -34, -20}
, {-46, -19, 13}
, {19, -26, 20}
, {42, 5, 31}
, {18, -20, -15}
, {53, -36, 32}
, {25, 25, 51}
, {-4, -18, -30}
, {-20, -8, -27}
, {-25, 42, 34}
, {-9, -52, -1}
, {-27, -1, -45}
, {-18, 38, 45}
, {25, 7, 40}
, {-16, 27, -30}
, {40, 0, 48}
, {43, -23, -39}
, {6, 0, -63}
, {40, -10, -8}
, {-43, 39, 15}
, {29, 49, 0}
}
, {{-47, -27, 29}
, {-42, 25, 38}
, {-5, 16, -55}
, {0, -50, -13}
, {2, -13, 40}
, {-35, -1, 10}
, {27, 9, 32}
, {-34, -17, -38}
, {31, 21, -7}
, {-44, 46, -50}
, {-39, -2, -32}
, {-49, 2, 6}
, {-20, 44, 43}
, {19, 8, 29}
, {44, -51, -5}
, {-24, -43, 46}
, {-22, 15, 11}
, {2, -46, -25}
, {-3, -52, 25}
, {-2, -20, -31}
, {-13, 9, -28}
, {24, 29, -27}
, {29, 14, 21}
, {-19, -24, -14}
, {11, -26, 20}
, {-43, 26, 16}
, {-1, 36, -47}
, {-44, -28, -34}
, {-29, 19, -19}
, {19, -28, -4}
, {-16, -38, -24}
, {-21, -30, -43}
, {-14, -56, 29}
, {38, 6, 13}
, {-35, -21, -12}
, {-31, 40, -9}
, {35, 5, 37}
, {11, -11, 12}
, {26, -31, 52}
, {-54, -31, 34}
, {-34, -57, 2}
, {43, -27, 11}
, {35, -17, 12}
, {-53, 12, -14}
, {9, 42, -9}
, {-13, 32, -5}
, {-14, -43, -7}
, {41, 2, -50}
, {-54, -41, -30}
, {-2, -12, -13}
, {40, -22, 23}
, {10, 35, 25}
, {16, -28, 22}
, {-48, 28, -44}
, {-27, -5, -13}
, {-43, -7, -23}
, {-50, -45, -47}
, {-10, -39, 37}
, {38, -16, 2}
, {37, -55, 36}
, {-51, 32, -51}
, {-10, -40, 25}
, {28, -32, 1}
, {44, 11, -37}
}
, {{-36, 43, -35}
, {25, 40, -18}
, {-34, -14, 6}
, {51, -15, 15}
, {-31, 40, -23}
, {5, 40, 65}
, {27, -26, -29}
, {-92, -77, -1}
, {-24, 6, -45}
, {2, 0, -33}
, {-58, -64, 6}
, {39, -40, -21}
, {43, -2, -19}
, {-48, 35, -6}
, {5, 44, -6}
, {-26, -4, -31}
, {43, 28, 43}
, {-20, 36, 44}
, {-3, 3, -28}
, {15, 40, 57}
, {9, -39, 29}
, {7, -37, -11}
, {5, 38, 8}
, {-27, -22, -26}
, {41, -17, 14}
, {36, 52, -20}
, {-49, -43, 27}
, {-52, -25, 23}
, {23, -9, -6}
, {-21, 15, 32}
, {6, -52, -77}
, {-21, 20, 36}
, {-17, 14, -23}
, {-39, 18, 38}
, {10, 38, 54}
, {-56, -26, -16}
, {22, 50, -3}
, {15, 3, 48}
, {47, 36, 26}
, {-17, 43, 14}
, {56, 55, -41}
, {16, -10, -57}
, {-18, 27, -28}
, {30, 2, -20}
, {17, 15, -3}
, {14, 30, -11}
, {-39, -5, 29}
, {15, -1, -7}
, {-3, -11, 22}
, {-3, 23, 17}
, {-3, -18, -26}
, {8, 28, 17}
, {16, -31, -25}
, {-33, -34, 31}
, {-13, -12, -13}
, {1, 32, -1}
, {-22, -31, -29}
, {-32, 22, -27}
, {-38, -48, -8}
, {-3, 0, 58}
, {5, -33, 50}
, {-74, -28, 21}
, {-10, 37, 17}
, {4, 42, 51}
}
, {{-34, 0, -50}
, {31, 53, 2}
, {-13, -25, -60}
, {-20, -40, 21}
, {-28, 35, 30}
, {27, 17, 10}
, {46, -31, -41}
, {20, -7, -43}
, {-13, -42, 18}
, {40, -15, 36}
, {50, -4, -26}
, {26, 56, -39}
, {28, 38, 29}
, {-9, -32, -27}
, {-26, 36, -19}
, {-42, -2, -26}
, {15, 12, 8}
, {27, 36, 27}
, {19, -42, -18}
, {-49, -10, -38}
, {-34, 23, 34}
, {-14, -48, -23}
, {11, 13, -1}
, {56, -23, 40}
, {6, 35, -17}
, {-16, -58, -13}
, {-29, -51, -40}
, {34, 42, 7}
, {-35, 37, -7}
, {23, 41, 75}
, {66, -42, -17}
, {-4, 1, 12}
, {-21, -45, 46}
, {41, -3, -6}
, {-52, -10, 23}
, {-24, 11, 7}
, {-7, 22, 38}
, {49, 0, -13}
, {-43, 36, 8}
, {-34, 14, -5}
, {-24, 17, -38}
, {77, -23, 18}
, {-41, 5, -17}
, {-63, -41, -37}
, {51, -54, 15}
, {-58, -53, -54}
, {0, -3, 37}
, {-11, -48, -5}
, {-3, 25, -3}
, {-40, 8, -45}
, {40, -17, -3}
, {-23, 31, -29}
, {-37, -10, -22}
, {-2, 47, 7}
, {-43, -30, 17}
, {-11, 38, 12}
, {-29, 7, 11}
, {56, -18, -12}
, {49, 46, -40}
, {27, 12, 34}
, {-60, -15, -25}
, {29, 2, 24}
, {48, -18, 41}
, {15, -27, 41}
}
, {{19, 17, -14}
, {62, 28, -18}
, {-38, -13, -32}
, {-31, -34, 31}
, {42, -13, 20}
, {-48, 44, -4}
, {-23, -29, 15}
, {-30, -13, 40}
, {-12, 13, -43}
, {-46, 23, -19}
, {60, -45, 16}
, {-2, 5, 20}
, {-43, -2, -16}
, {-36, 34, 8}
, {33, 47, -42}
, {30, 32, 27}
, {26, -3, 55}
, {-27, 29, -7}
, {30, -37, -30}
, {25, -49, -46}
, {-41, -3, -12}
, {45, 37, 56}
, {27, -52, 45}
, {15, 77, 2}
, {57, 35, 46}
, {-69, -34, 36}
, {6, -41, -33}
, {12, 90, 32}
, {-4, -41, 8}
, {42, 10, -8}
, {26, 58, -7}
, {39, 15, -4}
, {-23, 6, 52}
, {38, -43, 46}
, {-51, -11, 16}
, {-15, 34, -51}
, {-13, 25, -18}
, {46, -37, 69}
, {27, 37, 10}
, {-15, 36, 32}
, {52, 13, -3}
, {19, 23, 63}
, {-45, 2, 46}
, {-19, -19, -24}
, {9, -24, -53}
, {-14, -9, -23}
, {33, -41, 14}
, {-33, 3, -29}
, {-36, -37, -24}
, {8, 48, 3}
, {-12, -41, -30}
, {-8, 37, -22}
, {7, -3, 5}
, {19, -7, -29}
, {15, 35, 40}
, {39, -23, -2}
, {-30, 5, 42}
, {-16, 0, 51}
, {25, 6, 29}
, {-36, 0, -59}
, {13, 0, -25}
, {72, -4, 60}
, {2, 3, -43}
, {53, -13, 8}
}
, {{-22, 19, -29}
, {-16, -54, -47}
, {-8, 43, 3}
, {-27, -51, -49}
, {27, -38, -38}
, {-8, 30, -42}
, {41, 24, 23}
, {31, -2, 37}
, {49, 21, -19}
, {46, -50, 11}
, {-18, -32, 3}
, {-14, -10, 20}
, {-22, -51, 29}
, {26, -20, -47}
, {-13, 18, -28}
, {-31, -3, 27}
, {19, -1, -13}
, {-48, -19, -54}
, {21, -10, -45}
, {-13, 24, 3}
, {18, -1, 40}
, {-23, 41, 45}
, {-7, 8, 2}
, {-28, 30, -58}
, {-42, 19, 24}
, {-33, 9, 26}
, {28, -2, 41}
, {1, 29, 44}
, {-36, -36, 17}
, {-1, 12, 3}
, {26, -57, -18}
, {25, -41, -12}
, {-3, -51, -18}
, {47, 6, 43}
, {-44, 47, -31}
, {-23, 32, -47}
, {-58, -35, -51}
, {-27, 21, -42}
, {-4, 61, -33}
, {26, -49, 29}
, {4, 22, -47}
, {-18, 61, -53}
, {-49, -27, -24}
, {-9, -21, 7}
, {35, -19, -53}
, {-29, 22, 8}
, {-35, 30, -1}
, {26, -27, 38}
, {11, -25, -6}
, {19, -41, -51}
, {16, -27, 27}
, {-49, 10, -42}
, {-37, 42, 38}
, {-20, 40, 17}
, {31, 30, -3}
, {-50, -22, -38}
, {-33, 16, -41}
, {27, 42, 37}
, {-35, -23, -33}
, {-5, -36, 9}
, {-4, -13, -46}
, {-45, 12, 28}
, {-3, -46, 5}
, {3, -52, 37}
}
, {{-46, -48, 41}
, {2, -8, 32}
, {-51, -9, 34}
, {39, 9, 36}
, {5, -29, 41}
, {-19, 27, 33}
, {16, 24, -38}
, {82, 45, 12}
, {-39, 42, 19}
, {-19, -34, 25}
, {-46, -42, -34}
, {2, 10, -45}
, {39, -21, -23}
, {1, 35, -45}
, {-52, 13, 22}
, {24, 50, 20}
, {10, -37, -12}
, {-41, -15, 43}
, {5, 24, -24}
, {46, -41, -29}
, {39, 22, 47}
, {12, -2, -32}
, {17, -25, -27}
, {-16, 28, 31}
, {-23, -38, 41}
, {-15, 22, 43}
, {-7, 41, 32}
, {15, 5, -21}
, {-22, -20, -24}
, {48, 47, 9}
, {10, -44, -24}
, {-1, 3, -1}
, {54, -20, -40}
, {43, 2, -2}
, {-36, -23, 12}
, {-11, 19, -49}
, {9, 54, 12}
, {25, -17, 2}
, {0, 7, 17}
, {-23, -7, 26}
, {-32, 0, -21}
, {-43, 21, 3}
, {-24, 7, 30}
, {3, 12, -44}
, {29, -40, -5}
, {50, -19, 14}
, {-23, 23, 42}
, {-1, -39, 40}
, {-43, 17, 41}
, {8, 11, 25}
, {49, 12, -2}
, {-31, 11, 1}
, {0, -29, -1}
, {-48, -44, 8}
, {-39, -5, -35}
, {11, 1, -7}
, {3, 39, -12}
, {-35, 33, 7}
, {-2, -39, 0}
, {-33, -6, 3}
, {-8, 18, 35}
, {-8, 34, -42}
, {-25, -28, 9}
, {31, 36, -12}
}
, {{-28, 34, -11}
, {-29, 25, -15}
, {-1, -47, -29}
, {8, -42, 5}
, {-21, 51, -32}
, {-30, -37, 10}
, {28, 36, 58}
, {85, 59, 47}
, {12, -20, 35}
, {30, 12, -22}
, {-26, 11, 30}
, {-47, 0, -17}
, {-24, 32, 10}
, {14, -24, -29}
, {-14, -31, -27}
, {-12, 53, 38}
, {-13, 17, 40}
, {34, 21, 9}
, {-4, -17, 55}
, {55, 5, 5}
, {-6, -52, 51}
, {16, 28, 4}
, {-20, 15, 14}
, {52, 13, -19}
, {15, 7, -19}
, {-39, 21, 0}
, {12, -35, -6}
, {-36, -16, 34}
, {-29, -37, 27}
, {-28, 0, 4}
, {-35, 33, 9}
, {-7, -15, -23}
, {13, 28, 11}
, {-18, -9, 34}
, {-33, -5, -4}
, {28, 9, -26}
, {48, 48, -9}
, {-22, 42, 14}
, {31, -28, -20}
, {32, -48, -50}
, {5, 28, -33}
, {-36, -8, 0}
, {-46, -24, -10}
, {-27, 31, -12}
, {7, -1, 16}
, {46, 52, 23}
, {-17, -12, -1}
, {-5, 26, 15}
, {40, 51, -1}
, {-28, 39, -36}
, {-4, 52, -4}
, {3, 30, -29}
, {32, -50, 32}
, {-39, -30, 15}
, {-7, -33, -8}
, {51, 11, 33}
, {-11, 43, 32}
, {10, -36, 36}
, {-30, 33, -19}
, {-43, 21, -28}
, {-30, -16, -19}
, {-28, -22, -5}
, {17, 30, 14}
, {43, 22, -26}
}
, {{-40, 1, 18}
, {-39, 41, 34}
, {-65, -22, 29}
, {-3, -26, 40}
, {-3, 19, -54}
, {-49, -55, -56}
, {-15, 39, -55}
, {-39, -41, -25}
, {-23, -27, -42}
, {-29, -15, 30}
, {-4, -48, -48}
, {-18, 34, -38}
, {40, -20, 26}
, {34, -6, 33}
, {-38, -32, -15}
, {-50, 20, 43}
, {-38, -20, -37}
, {-11, -46, 17}
, {-51, -24, -32}
, {29, -56, -38}
, {-41, -23, -15}
, {-52, 3, -31}
, {-1, 19, 21}
, {-14, 31, 0}
, {-6, -9, -46}
, {-44, -8, -36}
, {34, -13, -26}
, {-31, -39, 33}
, {-34, -42, 12}
, {43, -21, -17}
, {-41, -40, 37}
, {-43, 3, 41}
, {30, -51, -50}
, {-56, 22, 27}
, {28, -42, -63}
, {31, -44, -9}
, {35, -57, -55}
, {43, -19, 14}
, {-51, -45, 26}
, {31, -44, -21}
, {-52, 2, 25}
, {-40, -6, 39}
, {29, -36, 12}
, {-31, -54, -61}
, {-30, 33, -41}
, {-30, 6, 20}
, {-33, -41, -56}
, {-46, 35, 10}
, {-8, -21, -48}
, {23, 34, -16}
, {-38, -17, -48}
, {-21, -35, -54}
, {35, 6, 11}
, {-57, -6, -20}
, {-54, 16, -31}
, {-50, -6, 19}
, {-53, -22, -8}
, {18, -34, -28}
, {-19, -7, 33}
, {-5, 30, 4}
, {-69, -17, 29}
, {12, -46, 1}
, {-5, -14, 12}
, {-20, -39, -23}
}
, {{-17, -13, -50}
, {-35, -13, -31}
, {10, 62, -5}
, {-43, 17, -54}
, {-18, -49, -36}
, {-44, 0, 17}
, {39, 45, -9}
, {24, 37, 9}
, {-2, -23, 0}
, {-11, 56, -47}
, {9, -50, -3}
, {33, -12, 24}
, {-5, 53, 2}
, {-2, -27, -25}
, {-4, 60, -50}
, {2, -20, 20}
, {-38, -15, 8}
, {-27, 33, -29}
, {43, -26, -12}
, {-4, 60, 2}
, {17, -1, -52}
, {7, -31, -25}
, {46, -20, 13}
, {23, 21, -11}
, {-7, 24, 38}
, {-10, 50, -33}
, {6, 0, 0}
, {15, 31, -61}
, {0, 29, -5}
, {21, 0, -79}
, {52, -24, 6}
, {23, -9, -10}
, {15, 39, 45}
, {-33, 9, -4}
, {17, -4, 59}
, {-14, 44, -19}
, {-42, 50, 53}
, {-24, -42, -47}
, {26, -2, -9}
, {-52, 5, -54}
, {-48, -42, 52}
, {-8, -17, -54}
, {-47, -28, -49}
, {-56, 63, 10}
, {69, 5, 15}
, {48, 8, -13}
, {6, -6, -38}
, {25, 9, 38}
, {28, 38, 45}
, {-28, 42, -40}
, {37, -4, 43}
, {56, 13, -26}
, {34, 58, 13}
, {33, -25, -31}
, {53, 51, -22}
, {-32, 19, 0}
, {32, -25, 30}
, {-6, -33, -41}
, {28, 21, 27}
, {5, 51, 0}
, {29, 3, -11}
, {-15, 9, 6}
, {36, -55, 28}
, {-21, -33, 19}
}
, {{-35, 19, -55}
, {43, 31, 46}
, {-21, -21, 43}
, {22, 13, -23}
, {35, -29, -17}
, {30, 46, -23}
, {-18, -14, -12}
, {-91, -68, -47}
, {-36, 38, 8}
, {29, 34, -38}
, {19, -27, -8}
, {6, -16, 20}
, {29, -17, 47}
, {-36, 48, 8}
, {-36, -17, -48}
, {-34, -42, -57}
, {-27, 31, -56}
, {25, 41, 45}
, {-5, 41, -32}
, {44, 21, 45}
, {31, -38, -22}
, {-24, 22, 17}
, {36, 30, 10}
, {-54, 12, 2}
, {38, 49, -41}
, {44, -12, 14}
, {4, 2, -35}
, {-45, 1, 46}
, {-4, -24, -23}
, {-28, 39, -35}
, {-35, 20, -34}
, {-50, -50, -34}
, {42, -36, 1}
, {26, -38, 15}
, {29, -25, 55}
, {-30, 33, -37}
, {16, -34, -1}
, {-35, 8, 26}
, {63, 66, 53}
, {50, 35, 5}
, {-55, -16, 32}
, {52, 49, -31}
, {-9, -10, -53}
, {-28, 40, -6}
, {16, 2, -1}
, {-1, -15, -35}
, {15, 12, -42}
, {17, -32, -20}
, {-47, 23, 40}
, {37, -10, -46}
, {-45, -30, -12}
, {47, -5, 33}
, {-7, 43, -31}
, {48, -42, 32}
, {-11, 46, -26}
, {42, 31, 14}
, {46, 42, 23}
, {24, -43, -45}
, {11, 11, -31}
, {-9, 55, -29}
, {14, 52, 61}
, {28, 38, -61}
, {-18, 18, -18}
, {22, -25, 17}
}
, {{-16, -24, -24}
, {-4, -74, -8}
, {-13, 44, -22}
, {-15, 33, 34}
, {18, 37, 7}
, {6, -25, -15}
, {-4, 23, 44}
, {52, -3, 40}
, {29, 32, -26}
, {-30, 6, 30}
, {-43, 16, -12}
, {-19, 19, 16}
, {28, -43, -48}
, {-35, 24, -2}
, {-30, -45, 34}
, {-18, 11, 12}
, {-22, -22, -4}
, {46, -22, 55}
, {48, 15, -13}
, {30, 13, 13}
, {-7, 17, -33}
, {51, 39, -15}
, {23, -28, -14}
, {23, 11, 0}
, {-8, -13, -39}
, {-35, -26, -1}
, {-7, 24, -19}
, {-48, 40, 35}
, {6, 30, -18}
, {-67, 25, 4}
, {12, -10, 41}
, {-52, 37, 21}
, {-7, 39, -3}
, {-23, 51, -45}
, {40, -14, -1}
, {-27, 33, 10}
, {45, -41, -17}
, {-18, -25, -33}
, {3, 18, -40}
, {-9, 24, -39}
, {10, -10, 22}
, {34, -77, -26}
, {46, -54, -44}
, {42, 2, 35}
, {37, -4, 56}
, {14, 21, -47}
, {-47, -26, -34}
, {7, 50, -56}
, {-7, 50, 39}
, {-4, -21, -24}
, {9, 55, -12}
, {51, 1, -14}
, {31, 20, 56}
, {33, 12, -14}
, {-12, -19, -16}
, {4, -12, -11}
, {-42, -10, 39}
, {-2, 30, -29}
, {45, 0, 33}
, {11, -47, -2}
, {-37, -19, 38}
, {24, 38, 25}
, {35, 5, -24}
, {-52, -45, -6}
}
, {{51, -20, 31}
, {-27, -2, -5}
, {-3, 26, -27}
, {-55, -32, -28}
, {12, 3, 34}
, {42, -34, -36}
, {33, 39, 14}
, {61, 30, 29}
, {76, -26, 14}
, {-15, -16, 24}
, {26, 31, 8}
, {42, -35, -24}
, {20, 38, 8}
, {0, -37, -19}
, {-28, -35, 22}
, {13, 6, -49}
, {64, -8, 9}
, {-9, 39, 17}
, {51, -11, 24}
, {-16, 65, 5}
, {17, 37, 8}
, {-27, 72, 24}
, {52, 58, 33}
, {-32, -25, -8}
, {-23, -21, -26}
, {43, -4, -1}
, {35, 63, 49}
, {-4, -13, -40}
, {-43, 15, 14}
, {18, -1, -41}
, {3, 13, -15}
, {15, -1, -27}
, {65, -36, -33}
, {35, -31, -29}
, {21, -18, -23}
, {25, 23, 12}
, {-18, -9, 41}
, {5, -30, -66}
, {-6, 7, -30}
, {-10, -56, 31}
, {-32, 3, -26}
, {-22, 12, -4}
, {-9, -31, -52}
, {6, 17, 42}
, {42, 57, 76}
, {68, 64, -6}
, {5, 43, -15}
, {-8, -4, 5}
, {-29, 27, -9}
, {28, -25, -46}
, {35, 41, 19}
, {3, -3, 71}
, {5, -15, 19}
, {-17, -7, 37}
, {55, -8, 9}
, {28, 41, 30}
, {33, -2, 27}
, {-42, -28, 5}
, {47, 30, 19}
, {-22, -13, 14}
, {58, 20, 51}
, {58, -34, 13}
, {-1, 18, 65}
, {-21, -1, 42}
}
, {{19, 11, -18}
, {13, 27, 13}
, {6, -42, -18}
, {61, -26, -3}
, {-10, 11, 31}
, {31, 58, -26}
, {37, 43, -21}
, {-22, -11, -66}
, {12, -36, 36}
, {59, -31, 51}
, {30, -28, -26}
, {-35, -27, 1}
, {28, -52, -40}
, {-38, -5, -18}
, {-35, -22, -3}
, {12, -12, 49}
, {31, -12, 30}
, {23, 51, -3}
, {-47, 16, 16}
, {19, 18, -22}
, {57, 59, -12}
, {-11, -58, -39}
, {-10, -64, 0}
, {18, -45, -14}
, {22, -5, 48}
, {-5, 8, 12}
, {0, -47, 22}
, {-13, -23, -22}
, {14, 18, 36}
, {47, -23, -16}
, {-54, -55, -20}
, {22, 50, -37}
, {4, -46, -5}
, {28, -14, -13}
, {38, 53, 32}
, {17, 31, 6}
, {-5, -15, 42}
, {28, 51, 3}
, {46, -21, 2}
, {58, -10, 3}
, {43, -37, -36}
, {-41, 7, -75}
, {17, 0, 37}
, {-43, -42, 22}
, {-49, -10, 0}
, {34, 9, 8}
, {-24, 49, -22}
, {38, 20, 44}
, {46, 39, 25}
, {-33, -10, 10}
, {-3, -40, -9}
, {-21, 28, -39}
, {-43, 44, 12}
, {38, 24, 37}
, {-48, -13, 36}
, {-18, -28, 52}
, {1, 9, 36}
, {-6, -1, 32}
, {-30, -9, -10}
, {-1, 2, 41}
, {-19, -2, -8}
, {-37, -26, -5}
, {-2, -10, 29}
, {-30, 8, -25}
}
, {{-16, -14, -43}
, {52, 66, 81}
, {37, -10, 61}
, {-20, 2, -49}
, {-2, 25, -32}
, {9, 52, 58}
, {-63, -51, -21}
, {-42, -63, 4}
, {21, 57, 27}
, {-31, -40, -43}
, {-18, 29, -18}
, {27, -3, 0}
, {-32, 18, 36}
, {22, -49, -4}
, {-23, -48, 6}
, {16, 4, 29}
, {-64, -49, 19}
, {22, 48, -11}
, {-14, -55, -15}
, {36, -30, 48}
, {-32, 0, 48}
, {-18, 19, 22}
, {-35, 47, -41}
, {8, 32, 32}
, {8, -49, 3}
, {50, 7, 30}
, {45, -11, 22}
, {51, 1, -16}
, {8, 7, 0}
, {12, -3, 31}
, {24, -28, 42}
, {-20, 14, -40}
, {-5, -39, 24}
, {1, 4, 29}
, {50, 9, 0}
, {16, 28, -19}
, {-61, -16, -24}
, {53, 31, 62}
, {1, 106, 18}
, {10, 54, -7}
, {23, -41, -36}
, {76, 38, 50}
, {-30, -19, -18}
, {33, 2, -23}
, {-27, -17, 33}
, {37, 1, -13}
, {-5, 15, 38}
, {17, -24, -46}
, {27, -33, -1}
, {-44, 5, -38}
, {27, -39, -40}
, {26, 55, -28}
, {-8, -12, -10}
, {34, 37, 18}
, {-48, 28, -20}
, {39, -44, -37}
, {8, 24, -17}
, {49, -23, -13}
, {-30, -23, 45}
, {-10, -9, 44}
, {33, 2, 32}
, {1, 2, 46}
, {39, 3, -46}
, {-14, -3, 50}
}
, {{54, -33, 20}
, {14, -45, -62}
, {61, 32, 29}
, {-38, -31, -8}
, {1, -3, 13}
, {47, -44, -4}
, {-31, -6, 37}
, {-18, 78, 62}
, {6, 15, 7}
, {29, 23, -2}
, {-24, 31, -18}
, {0, -30, 31}
, {19, 2, 26}
, {39, -35, 12}
, {31, 20, -50}
, {-36, 27, -55}
, {12, 7, -4}
, {39, 40, 57}
, {-14, 31, -1}
, {28, 25, 31}
, {30, 46, -56}
, {-51, 50, 14}
, {40, -5, 53}
, {34, 44, -19}
, {2, 14, -51}
, {46, 10, -27}
, {-23, -23, 27}
, {40, 32, -30}
, {-41, 38, -44}
, {-58, -12, -14}
, {14, 52, 11}
, {-32, -11, -23}
, {-47, 16, -46}
, {17, 40, 26}
, {65, 43, 49}
, {44, -9, 42}
, {53, 1, 8}
, {5, -63, -47}
, {11, 25, -8}
, {-20, -1, -21}
, {-33, -12, -9}
, {14, 6, -5}
, {22, 29, 33}
, {26, 31, 59}
, {72, -8, -17}
, {42, 56, 38}
, {26, -35, 52}
, {-20, 19, 30}
, {-10, 31, 45}
, {-11, -26, -25}
, {2, 37, -18}
, {46, 23, 15}
, {22, 4, -9}
, {35, 15, -9}
, {42, -22, 44}
, {-8, 48, 15}
, {34, 32, 26}
, {-42, -5, -17}
, {31, 35, -45}
, {-3, 11, 47}
, {33, 21, 35}
, {-25, 8, -60}
, {-9, 37, 5}
, {-42, 0, -49}
}
, {{54, -9, 16}
, {-3, -15, -59}
, {56, -34, 68}
, {-31, 16, -16}
, {12, 48, 18}
, {35, -55, -27}
, {4, 53, 49}
, {27, 111, 105}
, {61, 17, -8}
, {28, 29, 34}
, {-1, 42, 21}
, {41, 8, 45}
, {-21, -9, -41}
, {16, 42, 45}
, {-38, -34, 5}
, {-12, 37, -17}
, {-30, 9, 29}
, {-36, -55, -29}
, {20, -4, 6}
, {30, -30, 51}
, {-49, 16, 22}
, {-31, 17, 46}
, {7, 49, 4}
, {-45, -50, -54}
, {-22, 34, 41}
, {-35, -20, -6}
, {52, 54, 49}
, {47, 8, 1}
, {-30, -9, -68}
, {-32, -21, -40}
, {6, 30, 62}
, {7, -41, 26}
, {-6, -31, -48}
, {-16, -40, -37}
, {-7, 26, 42}
, {19, 52, 13}
, {-18, 10, -12}
, {-15, 31, -63}
, {13, -47, 15}
, {-34, -58, 31}
, {-30, 9, -39}
, {20, -40, 63}
, {8, -53, -23}
, {46, 36, 65}
, {39, 51, 38}
, {38, 43, 61}
, {-2, 4, 34}
, {-36, -9, 39}
, {3, 18, 26}
, {-56, -16, 13}
, {2, 0, 16}
, {-27, 0, 21}
, {55, -34, 6}
, {-25, 38, 26}
, {33, -40, 5}
, {42, 10, -40}
, {11, 25, 17}
, {40, -7, -11}
, {30, -26, 47}
, {4, -2, 31}
, {-32, 28, -44}
, {-3, -26, 37}
, {-24, -34, 13}
, {17, 9, 0}
}
, {{31, -15, -32}
, {-38, 23, -22}
, {16, -51, 43}
, {-21, -51, 8}
, {44, -54, 20}
, {48, 16, 17}
, {25, -33, 43}
, {21, -34, -23}
, {0, 3, -35}
, {-39, -9, 2}
, {-22, 6, -3}
, {-15, 26, -22}
, {-4, 6, 36}
, {-32, 30, -22}
, {-58, -12, -30}
, {18, -9, 33}
, {-58, -36, -1}
, {-46, 0, 22}
, {-53, 11, -36}
, {6, 27, -27}
, {-10, -34, -19}
, {-34, 28, 9}
, {-21, 30, 23}
, {36, -29, -41}
, {-31, 9, -51}
, {32, 26, -11}
, {-56, -32, 1}
, {-53, 44, 39}
, {-7, 34, 36}
, {5, 27, -45}
, {-16, 45, -6}
, {9, 8, -49}
, {-14, 30, -37}
, {20, 4, -55}
, {-15, 1, 31}
, {42, -25, 17}
, {-46, 47, 6}
, {-19, 14, -10}
, {-56, 7, -18}
, {-17, -15, -18}
, {20, -38, -29}
, {6, -52, 35}
, {-50, -60, -40}
, {5, -53, -33}
, {-52, -9, 34}
, {42, -18, 10}
, {16, 12, -10}
, {5, -42, 30}
, {-40, -24, -9}
, {-52, 33, -3}
, {-51, 19, 45}
, {-38, 24, 30}
, {-57, 7, -24}
, {-7, -56, -18}
, {-25, -38, 3}
, {39, 31, 25}
, {-22, 10, -47}
, {45, 13, 29}
, {24, -15, -3}
, {-49, 37, 22}
, {42, -29, -7}
, {40, 21, 31}
, {-7, -27, -40}
, {-56, -43, 35}
}
, {{20, 30, 47}
, {-1, 7, 7}
, {-7, 39, -20}
, {-48, -42, -34}
, {13, -18, -8}
, {53, -34, 60}
, {0, -35, -48}
, {-31, -47, -42}
, {-52, -20, -48}
, {-44, 4, 4}
, {-5, 46, 18}
, {46, -20, -10}
, {2, -33, 50}
, {-15, -47, -44}
, {44, -42, -12}
, {18, 31, -47}
, {-23, -20, 22}
, {7, 43, 29}
, {15, -30, 18}
, {34, 48, -26}
, {36, 0, 30}
, {-39, 14, -5}
, {9, 23, -34}
, {18, -57, -5}
, {38, -37, 3}
, {37, -16, -43}
, {38, 31, 46}
, {-17, -45, -14}
, {-2, 19, 45}
, {10, -39, -59}
, {45, -44, -19}
, {-5, -14, 56}
, {20, 16, 27}
, {-18, 9, 38}
, {-14, 8, 54}
, {30, 34, 33}
, {16, 3, 11}
, {-7, -6, 26}
, {39, 15, -17}
, {38, 7, -63}
, {31, -16, -36}
, {-36, 1, -20}
, {-11, 24, -27}
, {13, 46, 27}
, {-37, -17, 0}
, {-5, -20, 52}
, {13, 5, -32}
, {35, 46, -16}
, {42, 43, -22}
, {51, -39, -4}
, {35, 43, -34}
, {52, 0, 32}
, {28, -29, 65}
, {-2, 41, 0}
, {-13, -24, -15}
, {-45, -27, -27}
, {31, 46, -26}
, {0, -35, -33}
, {18, 7, -30}
, {54, 0, 10}
, {-31, -3, 41}
, {17, -47, 0}
, {36, 15, -40}
, {10, -17, -1}
}
, {{-43, 16, 60}
, {25, 19, -7}
, {-52, 4, 55}
, {35, -7, -8}
, {48, -16, -30}
, {4, 56, -4}
, {9, 37, 12}
, {-11, 3, 7}
, {6, 14, -3}
, {-51, -20, 6}
, {-24, -17, 7}
, {31, -9, -41}
, {32, 50, 52}
, {-80, 1, 2}
, {48, -2, 55}
, {28, 84, -26}
, {-13, -5, -55}
, {4, 53, 51}
, {3, -54, -34}
, {28, -33, 1}
, {13, -57, 32}
, {23, -26, -35}
, {34, -33, -9}
, {-21, 1, 62}
, {-39, -40, -3}
, {-18, -27, 43}
, {-4, 42, 16}
, {18, -49, -16}
, {20, 47, -8}
, {13, -56, -32}
, {5, 8, -25}
, {61, 36, -12}
, {-62, -2, 29}
, {50, 3, -2}
, {37, -5, 53}
, {13, -15, 19}
, {19, -39, -31}
, {45, 39, 12}
, {12, 37, 9}
, {0, -64, 22}
, {-10, 30, 4}
, {5, -63, -5}
, {15, -5, -24}
, {73, 12, 80}
, {19, -49, 12}
, {-24, -39, -42}
, {-6, -31, -61}
, {-12, -52, 6}
, {-14, 20, 8}
, {-38, 34, 50}
, {5, 9, 19}
, {84, -6, 35}
, {63, 63, 8}
, {-39, -13, 39}
, {-31, 25, 3}
, {20, 34, 7}
, {-38, -36, -7}
, {-48, -37, -4}
, {-2, 18, -29}
, {18, -41, 0}
, {3, 62, -19}
, {-61, -35, -60}
, {36, -23, -3}
, {-30, 26, 10}
}
, {{-9, -9, -10}
, {37, 20, -36}
, {-2, 7, 23}
, {26, 30, -2}
, {-56, -7, 44}
, {-39, -49, -21}
, {34, 5, 3}
, {-10, -15, 7}
, {3, 27, 3}
, {-11, -15, 20}
, {29, 22, -33}
, {-48, 17, -1}
, {-2, -37, -17}
, {-21, 0, -28}
, {42, -40, 44}
, {-20, 31, 6}
, {42, -58, 13}
, {-4, -49, 6}
, {-35, -20, -38}
, {-6, 12, -49}
, {-11, 40, 0}
, {31, 27, -48}
, {-18, -55, -16}
, {37, 40, -50}
, {-33, 1, -47}
, {-20, -36, 39}
, {-21, 28, -47}
, {-56, 32, 20}
, {-30, 32, -8}
, {-62, 32, -52}
, {-1, -31, 23}
, {-55, -36, 0}
, {22, -6, -31}
, {-46, -58, -48}
, {20, -28, -29}
, {-47, -57, -42}
, {-56, 24, -29}
, {-56, -13, -55}
, {0, 32, -19}
, {37, -2, 6}
, {-55, 32, -37}
, {20, 7, -24}
, {0, 40, 33}
, {9, -35, 34}
, {-32, -13, 8}
, {-37, -14, -24}
, {-25, 22, -5}
, {-48, -13, 31}
, {-4, 39, -7}
, {25, 17, 41}
, {-43, -2, -35}
, {13, 13, -24}
, {0, 25, 29}
, {-46, -41, 5}
, {-2, -21, 5}
, {-16, -18, -24}
, {6, -44, 8}
, {33, -56, -55}
, {32, 39, -9}
, {27, -46, -2}
, {-4, 40, 0}
, {-45, -20, 39}
, {20, -25, -58}
, {7, 22, -48}
}
, {{35, -21, 1}
, {18, -61, -53}
, {61, 19, 52}
, {47, -4, 6}
, {-11, -22, 37}
, {65, -17, 34}
, {37, -46, -39}
, {-10, 1, -5}
, {69, 37, 47}
, {-36, 51, -44}
, {40, -3, -31}
, {-18, -25, 46}
, {0, 1, 23}
, {-37, 0, -51}
, {37, 7, -27}
, {-6, -35, -10}
, {-4, -46, 31}
, {-26, 12, 40}
, {21, 41, -8}
, {70, 33, 41}
, {53, 32, 42}
, {-37, 46, 17}
, {-31, 31, -28}
, {-62, 29, 32}
, {0, 0, 28}
, {-17, 20, 43}
, {-37, 6, 19}
, {-17, 23, -46}
, {24, 10, -26}
, {6, -26, 9}
, {9, 57, 43}
, {17, 37, 7}
, {41, 39, 6}
, {34, -21, -32}
, {13, 67, 29}
, {34, 13, -30}
, {3, -18, 21}
, {2, -1, -60}
, {-42, -23, 10}
, {-17, -54, -37}
, {-43, 38, 28}
, {-39, -43, -34}
, {6, -28, 10}
, {50, 76, 84}
, {63, -37, 11}
, {70, 31, -2}
, {20, 24, -14}
, {-38, 33, 44}
, {-33, 20, 4}
, {19, 29, 11}
, {-19, 18, 28}
, {56, 25, -17}
, {65, 11, 6}
, {59, 14, 28}
, {8, 43, 44}
, {-46, 43, 18}
, {2, 7, 45}
, {-48, 11, -17}
, {-15, -48, -2}
, {-15, 36, 7}
, {-2, 8, 62}
, {-27, -17, -20}
, {9, 39, 30}
, {-47, 12, -45}
}
, {{-13, 12, -14}
, {63, 37, 42}
, {11, -35, 18}
, {-6, -44, -3}
, {-7, 44, -38}
, {29, 25, 47}
, {26, -35, 3}
, {-54, -46, 0}
, {82, 40, 76}
, {41, -33, -12}
, {-65, -24, -46}
, {-3, 29, 14}
, {38, 0, -24}
, {-50, -13, -7}
, {-39, -50, 4}
, {42, 34, -2}
, {-36, -64, 24}
, {-8, -20, 18}
, {-10, -19, -43}
, {54, 52, -55}
, {9, -1, 9}
, {-2, -24, 9}
, {-24, -59, 27}
, {-36, -20, -18}
, {-10, 32, -26}
, {61, 54, -25}
, {28, -31, -41}
, {-14, -5, -44}
, {-1, 48, 56}
, {-39, -20, 34}
, {31, -29, -65}
, {4, -11, -23}
, {-15, -10, -56}
, {-34, 7, -13}
, {63, -15, 36}
, {5, -34, 17}
, {29, -15, 34}
, {42, 26, 40}
, {32, 93, 54}
, {12, 50, -10}
, {-4, 0, -39}
, {44, 7, -44}
, {14, -28, 27}
, {59, -9, 26}
, {-12, 15, 11}
, {43, -8, -42}
, {-6, -17, 38}
, {-34, 53, -49}
, {50, -20, -26}
, {-2, 45, -7}
, {-66, 40, 26}
, {-11, -25, 15}
, {-26, 51, 50}
, {-20, 7, 29}
, {-11, 13, -51}
, {-49, -16, -16}
, {11, 34, -58}
, {-42, -56, 34}
, {-35, 37, 8}
, {13, 49, -45}
, {34, -1, 0}
, {-33, -61, -50}
, {16, -24, -50}
, {30, 28, -35}
}
, {{-46, -22, 8}
, {-32, -35, 8}
, {-52, 31, 5}
, {-10, -36, 38}
, {19, -51, 12}
, {-7, -16, 11}
, {19, 48, 18}
, {0, 92, 68}
, {-37, -9, -38}
, {-14, -5, 16}
, {44, 43, 36}
, {-43, -34, -33}
, {19, -47, -39}
, {6, 37, -6}
, {36, 29, -15}
, {40, 19, -17}
, {12, -13, 19}
, {3, -58, -11}
, {-2, 31, 21}
, {31, -22, -19}
, {4, 6, -12}
, {-24, 36, -34}
, {-37, 0, 42}
, {-27, 18, -36}
, {-19, 2, -46}
, {5, -7, 35}
, {-2, -27, 37}
, {31, 0, 14}
, {-68, -18, 0}
, {-30, 31, 5}
, {-29, -63, 19}
, {-45, -6, -23}
, {-16, 19, 24}
, {5, -8, -7}
, {5, 12, -28}
, {2, -36, -41}
, {20, -22, -21}
, {-34, 21, 37}
, {-60, 23, 20}
, {-29, -33, 34}
, {25, -21, -44}
, {20, -4, 19}
, {-15, 21, -41}
, {17, 26, 27}
, {-47, 24, 13}
, {-20, 10, 10}
, {33, -20, 4}
, {25, 7, 44}
, {22, 1, 39}
, {-42, 22, -18}
, {37, 56, -12}
, {33, 32, -37}
, {-50, -21, 52}
, {-49, 50, 13}
, {5, 4, 10}
, {11, -20, 7}
, {24, -29, 11}
, {-5, -40, -3}
, {28, 10, -50}
, {6, -20, -31}
, {-11, -21, 24}
, {2, 10, -13}
, {50, 23, 51}
, {-13, 45, 24}
}
, {{-11, 35, 14}
, {-36, 13, -58}
, {-64, 6, 30}
, {19, -1, -28}
, {-5, 25, -44}
, {30, -31, -20}
, {-34, 24, 25}
, {-20, -12, -16}
, {21, 22, 56}
, {3, 8, 41}
, {1, 0, 23}
, {-19, 4, -27}
, {-16, -66, -43}
, {-29, -11, 32}
, {-23, 31, -26}
, {23, 4, 2}
, {-36, -33, 11}
, {35, 7, -2}
, {-12, -19, -3}
, {-36, -20, -28}
, {-17, -61, -15}
, {8, 31, -44}
, {-52, 38, -51}
, {33, -46, -45}
, {-47, 22, -51}
, {-21, 7, -34}
, {-56, -41, -46}
, {-23, -23, 37}
, {-48, -34, -35}
, {-42, 14, 2}
, {-31, -10, -28}
, {-41, 5, 27}
, {-38, 0, 26}
, {9, -57, -32}
, {-18, -38, 24}
, {0, 7, -7}
, {-3, -65, 50}
, {13, -46, -26}
, {-2, 47, -29}
, {-1, 27, 24}
, {14, 41, -15}
, {7, -35, 5}
, {26, -2, -46}
, {20, -25, 17}
, {-2, 27, 26}
, {-2, 35, 48}
, {9, 11, -25}
, {-19, 48, -20}
, {-4, -54, 13}
, {3, 4, 17}
, {45, 38, 15}
, {-20, 34, 42}
, {-42, 47, 45}
, {-54, -12, -26}
, {53, 55, -14}
, {11, -47, -69}
, {-49, 1, -28}
, {-16, 53, -35}
, {-8, 29, 31}
, {33, 36, 14}
, {-41, -33, 46}
, {-53, 47, 1}
, {-66, -3, 34}
, {-48, -29, -7}
}
, {{48, 2, -6}
, {-15, 48, -43}
, {-23, 6, -31}
, {3, 9, -3}
, {-28, -41, -50}
, {33, -56, -35}
, {34, 34, 30}
, {-16, -73, -11}
, {27, 12, -52}
, {-49, -25, -49}
, {28, -19, 49}
, {-10, -31, -50}
, {40, -51, -5}
, {3, -5, -10}
, {-45, -9, 19}
, {36, -79, -56}
, {18, -1, 4}
, {21, -17, -2}
, {10, -55, -7}
, {44, -35, -11}
, {28, -49, -18}
, {51, -48, 41}
, {54, -38, -39}
, {-67, 66, 5}
, {54, 0, -17}
, {17, -17, -1}
, {23, 23, -14}
, {-31, 30, 0}
, {34, -32, -75}
, {-4, 15, -35}
, {-14, 76, -21}
, {-40, -11, -40}
, {17, -41, -31}
, {-5, 0, -47}
, {15, 12, 52}
, {47, 50, -30}
, {6, 19, -32}
, {0, 55, 39}
, {6, 42, 50}
, {-23, -31, 55}
, {-19, -46, -9}
, {58, 95, 70}
, {24, -24, -31}
, {-18, -36, 22}
, {19, -4, 33}
, {7, 3, 48}
, {10, 41, 43}
, {0, 1, -36}
, {-1, -30, -27}
, {-35, 23, -9}
, {35, -57, 7}
, {31, -14, 37}
, {-50, 42, 13}
, {41, 35, -51}
, {-4, 22, 45}
, {31, 16, -31}
, {-38, -24, -51}
, {39, -2, -41}
, {11, 23, 20}
, {-10, 31, -13}
, {-10, 11, 7}
, {71, 34, 94}
, {20, 35, 5}
, {23, 28, 40}
}
, {{18, 10, 25}
, {4, -43, 26}
, {-44, -34, 3}
, {7, 6, 15}
, {47, 16, -46}
, {-30, 8, 44}
, {-47, -40, 37}
, {37, 24, 41}
, {24, 20, 33}
, {-40, 12, 25}
, {-51, -1, 30}
, {-19, 18, 8}
, {-10, 32, -30}
, {-7, 5, 35}
, {9, 34, -33}
, {-50, -10, 29}
, {2, -37, -21}
, {22, -11, -17}
, {23, -40, -30}
, {-17, 28, -43}
, {0, 17, 5}
, {34, 27, 12}
, {-32, 9, 12}
, {-19, -29, 19}
, {-36, 23, 16}
, {-1, -48, 36}
, {10, -47, -23}
, {19, 7, -6}
, {-45, -25, 8}
, {6, -35, -18}
, {3, 42, -7}
, {-42, -26, -5}
, {-45, -31, 15}
, {-30, 20, 21}
, {35, -9, 37}
, {40, -18, 28}
, {5, 33, -12}
, {-24, -38, -25}
, {0, 2, -22}
, {-44, 24, -24}
, {-57, -28, -45}
, {-30, -48, -12}
, {-8, -13, -28}
, {-44, 3, -18}
, {9, 11, -35}
, {-21, 10, 23}
, {-49, -20, -2}
, {-9, -39, -41}
, {6, -36, 43}
, {4, -38, -15}
, {27, -7, -12}
, {-23, -28, 25}
, {20, -37, -18}
, {-35, 4, -8}
, {32, 6, -14}
, {-8, -16, -21}
, {39, 9, -2}
, {-38, 30, 44}
, {-11, -31, -39}
, {23, -18, -50}
, {-33, 25, -21}
, {-50, -26, -42}
, {26, -29, -50}
, {0, -7, 25}
}
, {{5, 39, 4}
, {71, 68, 0}
, {-9, 6, 3}
, {-30, -19, -18}
, {22, 10, 27}
, {-23, 32, 30}
, {-7, 51, -16}
, {50, -49, 11}
, {26, -5, 41}
, {28, -14, 9}
, {-22, 31, 44}
, {-9, 46, -42}
, {-30, -45, -50}
, {65, 25, 15}
, {51, -15, -29}
, {48, -6, 0}
, {49, -43, 36}
, {16, -49, 42}
, {39, 21, -12}
, {29, -8, 43}
, {40, -14, -55}
, {-28, 19, 5}
, {10, -15, 40}
, {42, -7, 30}
, {7, 27, -30}
, {-2, -14, 30}
, {-8, -3, 39}
, {34, 69, 39}
, {-11, 48, -7}
, {5, -29, 18}
, {40, -2, -35}
, {27, 16, 9}
, {35, -12, 46}
, {2, -46, -30}
, {-37, -41, -14}
, {-14, 37, -12}
, {4, -41, -36}
, {-21, 49, 32}
, {60, 5, 103}
, {76, 59, 68}
, {11, 22, 30}
, {76, -4, 84}
, {-42, -15, 17}
, {-49, -50, -8}
, {-9, -13, 22}
, {-54, -5, -31}
, {-8, -40, 0}
, {-45, 25, 19}
, {-41, -4, -50}
, {-20, -25, -32}
, {1, -8, -13}
, {3, -33, -31}
, {-47, 22, 23}
, {23, -44, -50}
, {11, 28, -1}
, {-7, 4, -1}
, {14, -41, -31}
, {-20, 43, 21}
, {26, 1, 10}
, {-44, -36, -26}
, {21, 51, -27}
, {44, -2, 0}
, {41, -11, 41}
, {-32, -45, -27}
}
, {{-18, 32, 27}
, {16, -44, -23}
, {-46, -22, -40}
, {-38, 29, -23}
, {-6, -14, 39}
, {-45, 10, 17}
, {-8, 32, -4}
, {69, 3, 52}
, {53, 7, -15}
, {-34, 25, -18}
, {7, 7, 0}
, {38, 28, -48}
, {-37, -3, -4}
, {-20, 2, -16}
, {-19, 54, -46}
, {65, 40, 21}
, {31, -20, 22}
, {-26, -9, -40}
, {41, -34, 3}
, {56, -6, -59}
, {-24, 32, -6}
, {43, 9, -3}
, {67, -8, -16}
, {30, 4, 32}
, {42, -53, -56}
, {24, -43, -9}
, {15, -12, -5}
, {10, 7, -35}
, {-72, -56, 10}
, {8, -66, -54}
, {17, 14, 31}
, {-35, -40, 22}
, {-24, 25, 9}
, {37, 36, 4}
, {6, 16, -29}
, {-44, -12, 36}
, {25, -23, 9}
, {24, -24, -37}
, {37, 22, -17}
, {-43, 15, 14}
, {-29, 20, -9}
, {-41, 10, 2}
, {46, -47, -52}
, {51, 44, 48}
, {19, 21, 1}
, {41, -27, -30}
, {-49, 28, 15}
, {52, 45, -5}
, {-7, -39, 30}
, {10, -50, 10}
, {52, 15, 41}
, {-12, 40, -10}
, {-10, 29, 12}
, {-4, 48, -38}
, {-16, 7, -14}
, {6, 46, -39}
, {24, -3, 11}
, {39, 21, 21}
, {48, -34, 57}
, {37, 21, 23}
, {13, -35, 10}
, {41, 19, -22}
, {-14, 56, -39}
, {-43, -43, -38}
}
, {{45, -41, 56}
, {4, 52, -38}
, {-27, 5, 19}
, {-49, 5, 50}
, {-44, -1, -31}
, {-42, -1, -65}
, {-21, 42, 2}
, {35, 17, -7}
, {11, 49, 55}
, {44, 22, 14}
, {52, -18, 13}
, {-26, 40, 51}
, {21, -31, 8}
, {12, -20, -3}
, {41, 9, -32}
, {-8, -17, 11}
, {21, 49, -12}
, {-35, -13, -57}
, {54, -6, 11}
, {57, -26, -46}
, {9, -30, -41}
, {-9, -8, 41}
, {16, 72, 10}
, {23, 59, -5}
, {-37, 35, -50}
, {29, 33, -35}
, {-35, 36, 8}
, {81, 46, 78}
, {-20, 25, 0}
, {-49, 9, -48}
, {56, -17, 17}
, {58, 11, -25}
, {42, -50, 8}
, {1, 30, 12}
, {-34, -42, -9}
, {-26, 42, 1}
, {29, -19, -7}
, {41, -24, 3}
, {26, 12, 32}
, {9, 27, 4}
, {-63, 23, 20}
, {73, 65, 37}
, {-33, -9, 26}
, {41, 42, -46}
, {29, 0, 11}
, {69, 45, 56}
, {31, -17, 16}
, {4, 52, -18}
, {-25, -9, -22}
, {-51, 44, -28}
, {7, 3, 68}
, {-45, -8, 30}
, {-2, 16, -32}
, {-8, -51, -26}
, {-24, -14, -32}
, {-36, -32, -21}
, {3, -31, -37}
, {-52, 54, 38}
, {16, 0, 29}
, {-14, 7, -49}
, {38, 8, 23}
, {45, 44, 23}
, {0, 6, -7}
, {-21, -12, 39}
}
, {{0, 10, -18}
, {51, 55, 49}
, {7, 26, 10}
, {2, -26, 49}
, {18, 24, -19}
, {30, 48, -9}
, {37, 32, 0}
, {-14, 9, 1}
, {38, 9, -5}
, {-32, -23, -33}
, {-17, -36, 11}
, {14, 23, 18}
, {-23, -21, -3}
, {-8, -42, -9}
, {53, 1, -14}
, {-42, -1, 39}
, {-14, -29, 3}
, {8, -37, -10}
, {3, 35, -30}
, {30, 6, -29}
, {3, 38, -13}
, {-53, -25, -7}
, {36, -52, 38}
, {23, 45, -13}
, {35, 22, -50}
, {32, 62, -21}
, {-28, -33, -26}
, {-21, 51, -25}
, {-16, -3, -43}
, {-12, 45, 53}
, {33, -54, 8}
, {2, -49, 41}
, {20, 15, 41}
, {44, 0, 15}
, {14, -19, 35}
, {-38, -13, -1}
, {-11, -23, 27}
, {20, 30, 15}
, {-48, 53, -5}
, {56, 0, -18}
, {51, 8, -22}
, {14, 11, -77}
, {43, -35, 9}
, {-5, 48, -28}
, {-28, -41, -42}
, {32, 36, -2}
, {-27, 0, -25}
, {4, -2, -20}
, {61, 50, -48}
, {0, 42, 0}
, {-13, 5, -5}
, {-24, 4, -32}
, {49, -11, -50}
, {-45, -13, -46}
, {30, -31, -10}
, {51, 25, -11}
, {-9, -36, -53}
, {-12, -20, 0}
, {22, 25, -45}
, {11, 55, -21}
, {-33, -16, 31}
, {-28, -58, -56}
, {-2, 39, 21}
, {33, 12, 16}
}
, {{15, -6, -30}
, {-2, 61, 55}
, {31, -29, -39}
, {-9, 69, 67}
, {-41, 4, 27}
, {-8, -35, 9}
, {-7, -41, 43}
, {-12, -78, -69}
, {-51, -25, -43}
, {-19, 63, 0}
, {-32, 0, -26}
, {13, -31, -10}
, {-38, 6, 7}
, {44, -37, -31}
, {0, -20, -44}
, {55, 49, 22}
, {17, -20, 31}
, {9, 30, 15}
, {32, -48, 6}
, {-30, 37, 34}
, {9, 25, -29}
, {-62, -16, -55}
, {-5, -27, -58}
, {43, -26, -25}
, {-13, 9, 44}
, {-28, -37, -8}
, {-20, -24, 13}
, {2, 0, -13}
, {23, 4, -31}
, {33, -12, 58}
, {-45, -49, -15}
, {34, -45, -32}
, {-19, -28, -27}
, {13, 17, 39}
, {48, 47, -1}
, {2, -19, -59}
, {45, 34, 28}
, {26, 21, -3}
, {-15, -35, 55}
, {21, 68, 38}
, {-18, -26, 27}
, {-9, 17, -50}
, {37, 32, 48}
, {-59, -15, -8}
, {-54, -53, -24}
, {-31, 14, -47}
, {-10, -32, -24}
, {-50, -44, 43}
, {31, 4, -32}
, {11, 12, 42}
, {16, 7, -20}
, {-35, -15, 10}
, {8, -14, 15}
, {-26, 24, 39}
, {-36, -9, 15}
, {-13, 42, -32}
, {-23, -27, -35}
, {54, 27, 47}
, {0, 60, 29}
, {32, 54, 7}
, {-14, 16, 43}
, {35, 16, -30}
, {-20, -28, -42}
, {49, 60, 26}
}
, {{51, -18, 34}
, {-66, 24, 14}
, {-27, -6, 9}
, {-14, -8, 37}
, {32, 46, 55}
, {22, 52, -19}
, {-24, 41, -29}
, {-22, 59, 101}
, {-30, 33, -39}
, {-22, 42, 49}
, {-27, -10, -9}
, {-31, 14, -39}
, {-33, -43, 45}
, {-67, -47, 34}
, {44, 23, -29}
, {-16, -42, -54}
, {26, 24, 0}
, {54, 7, -38}
, {-39, 4, 56}
, {42, 23, -15}
, {44, 16, 18}
, {-43, -26, 65}
, {-44, -20, -3}
, {-16, 18, -27}
, {-6, 7, 29}
, {-15, 40, 27}
, {61, 8, 30}
, {7, -42, -73}
, {19, 8, 18}
, {-25, 10, 10}
, {56, 28, 0}
, {48, 30, -51}
, {1, 20, 10}
, {-16, -13, -12}
, {56, -8, 41}
, {-36, -40, -37}
, {63, -24, 45}
, {18, 33, -56}
, {-2, -93, -4}
, {-51, -59, -45}
, {-14, -19, 3}
, {-52, -33, -31}
, {-8, -28, -1}
, {53, 49, 16}
, {-17, -18, -4}
, {-1, -35, 52}
, {50, -10, -28}
, {-26, -17, 42}
, {54, -40, 39}
, {42, -18, -53}
, {26, 32, 73}
, {-36, -47, 36}
, {36, 15, 50}
, {42, 14, 22}
, {-27, -33, 37}
, {17, -10, 18}
, {-35, 8, -41}
, {-30, 28, 25}
, {-42, 6, 28}
, {9, -24, 44}
, {56, -44, -40}
, {-70, -70, -40}
, {10, -38, -11}
, {-51, 3, -28}
}
, {{-2, 0, 24}
, {-50, -22, -11}
, {-13, 24, 24}
, {-47, 45, -22}
, {7, 15, -11}
, {65, -12, 33}
, {27, -32, 12}
, {50, -3, 59}
, {-3, -39, -21}
, {-35, 1, -42}
, {8, -14, -49}
, {-12, 48, -40}
, {38, -29, 27}
, {17, -40, 22}
, {25, -45, -36}
, {-35, 19, 31}
, {9, -9, -53}
, {8, -21, -37}
, {8, 39, -52}
, {16, -22, 61}
, {-5, -9, 22}
, {-13, -9, 10}
, {48, 39, -1}
, {-1, 18, 9}
, {-38, -37, 0}
, {-30, 66, 50}
, {38, 53, -47}
, {-7, 6, -34}
, {-34, -34, -2}
, {4, -31, -31}
, {-23, -20, -43}
, {-31, 44, -26}
, {3, 6, -20}
, {0, 53, 3}
, {41, 25, 63}
, {-29, 50, -12}
, {20, 36, 3}
, {-26, 4, -34}
, {-11, 15, 19}
, {-3, 18, -55}
, {-23, -21, 22}
, {-51, 16, 10}
, {35, 34, 24}
, {90, 60, 80}
, {32, 21, 22}
, {51, 29, -17}
, {-46, -17, -16}
, {6, 9, 32}
, {51, 57, 33}
, {25, 37, 49}
, {-46, -5, -32}
, {12, 6, -5}
, {39, 30, -13}
, {36, 54, 48}
, {63, -26, -7}
, {44, 20, 49}
, {9, -24, -14}
, {24, 38, -21}
, {-46, 0, 26}
, {69, 53, 10}
, {1, -23, 43}
, {-1, -14, -16}
, {27, 15, -28}
, {-53, -41, -43}
}
, {{-7, 51, -18}
, {-47, -37, -30}
, {-56, 78, 14}
, {4, 0, -24}
, {-24, 19, 27}
, {6, 64, 13}
, {50, -41, 4}
, {13, 79, -27}
, {52, 20, -36}
, {-52, -25, 23}
, {-53, -59, -34}
, {3, -33, -57}
, {-38, 52, 29}
, {-64, -45, -59}
, {34, -2, -25}
, {64, 32, 29}
, {18, -39, -47}
, {43, 22, 6}
, {28, 8, -32}
, {36, -19, -38}
, {-11, -1, 1}
, {23, -49, -24}
, {8, 8, -64}
, {9, -34, 25}
, {-42, 5, -25}
, {-4, -10, 19}
, {-28, 2, -1}
, {-22, -16, -21}
, {-17, -41, 41}
, {-50, 22, -2}
, {14, 24, 35}
, {33, 59, -11}
, {-4, 6, -39}
, {30, -16, 40}
, {9, 26, 28}
, {-31, -34, -6}
, {31, -26, 42}
, {-34, -31, -27}
, {2, 52, 0}
, {-18, -22, -37}
, {57, -14, -34}
, {-15, -18, -9}
, {40, -11, -33}
, {44, 70, -5}
, {-13, 41, 43}
, {16, 65, -28}
, {-12, 9, -30}
, {-2, 30, 23}
, {23, 19, 5}
, {29, 40, 55}
, {-42, 13, 36}
, {-21, -37, -9}
, {-15, 8, 51}
, {39, 12, 36}
, {-38, -7, -18}
, {-14, -56, -34}
, {-45, 19, 25}
, {-14, -39, 9}
, {-3, -13, 28}
, {-11, 11, -13}
, {50, 76, 18}
, {25, -76, -14}
, {28, 39, 37}
, {17, -7, -8}
}
, {{-16, -31, -34}
, {45, 77, 23}
, {-13, -2, 33}
, {7, -34, 26}
, {-29, -4, -28}
, {55, -36, 13}
, {58, -52, -4}
, {-11, -43, -7}
, {-30, -40, -38}
, {-25, -54, -40}
, {47, 52, -20}
, {-17, -30, 54}
, {-15, 19, -51}
, {35, 8, 84}
, {29, 10, -27}
, {-34, 47, -45}
, {-34, -34, -40}
, {42, -36, -8}
, {-16, 19, 32}
, {-6, 25, -5}
, {15, 26, -20}
, {8, 12, 24}
, {21, 10, -27}
, {36, -38, 18}
, {-44, -9, 26}
, {-1, -38, 13}
, {-26, 45, -8}
, {31, -25, 67}
, {-11, -10, 32}
, {-63, 74, -37}
, {12, 17, 1}
, {25, 3, 8}
, {-30, 10, -10}
, {29, -12, 40}
, {27, 14, -23}
, {-34, 2, 1}
, {-29, -48, -33}
, {91, -5, -8}
, {23, 38, 52}
, {55, 63, -44}
, {-5, 4, 0}
, {-36, 26, 68}
, {-27, 7, 22}
, {-46, 5, -39}
, {23, 24, 7}
, {28, -2, -6}
, {0, 40, 30}
, {-23, 6, 24}
, {20, -64, 23}
, {-43, 16, -64}
, {22, 22, -30}
, {-29, 46, -47}
, {32, -43, 16}
, {-43, -10, 8}
, {-13, -9, 52}
, {-36, 14, 34}
, {30, -41, 29}
, {0, 83, 0}
, {-53, 51, -15}
, {-20, 0, -17}
, {47, -52, 57}
, {-10, 39, -8}
, {40, 53, 20}
, {-53, 4, 19}
}
, {{1, 19, -48}
, {-42, 29, -37}
, {17, -38, 0}
, {-39, -25, 28}
, {-27, 42, -9}
, {-9, 14, -5}
, {4, 19, -5}
, {-25, -38, 34}
, {-13, -15, -27}
, {39, 61, -25}
, {10, -21, 42}
, {18, 14, 6}
, {-45, -56, -51}
, {43, -17, 29}
, {21, 17, -29}
, {-54, -38, 13}
, {33, -21, -11}
, {-15, -6, 11}
, {22, -23, 45}
, {-37, -37, 7}
, {-25, -27, -32}
, {11, 30, -2}
, {-48, -35, -45}
, {-21, 45, -13}
, {-28, -19, -45}
, {-21, 26, -24}
, {-9, -19, -49}
, {19, 0, 30}
, {-46, 0, 8}
, {34, -20, 19}
, {10, -33, 28}
, {26, 36, -54}
, {-27, -12, -33}
, {-5, -5, 33}
, {45, 37, 41}
, {16, -15, -21}
, {-19, -1, -33}
, {-48, -6, -58}
, {-42, 4, 43}
, {31, -37, 14}
, {43, -19, 48}
, {31, -2, 0}
, {-35, -1, 9}
, {-52, 1, -63}
, {1, 46, -30}
, {-48, -32, 17}
, {-38, -29, 18}
, {22, 31, -13}
, {35, -21, -14}
, {-44, -43, 38}
, {19, 4, 10}
, {-28, -8, 44}
, {29, 51, -50}
, {-36, 8, -32}
, {-23, -4, -25}
, {-10, 33, 27}
, {-59, -52, 27}
, {26, -18, -26}
, {-10, 47, -12}
, {-59, 22, 16}
, {44, -52, -8}
, {11, -5, 18}
, {17, -36, 29}
, {-11, -49, -40}
}
, {{-52, 27, -5}
, {35, 32, 32}
, {23, -13, 52}
, {8, -6, 35}
, {-36, -39, 9}
, {8, -23, -9}
, {6, 30, 51}
, {-37, -69, 13}
, {-26, 12, 27}
, {-14, -38, -29}
, {3, 13, -34}
, {-43, 27, 57}
, {13, -48, -53}
, {0, 38, -5}
, {24, -5, -37}
, {-54, 12, 24}
, {47, 40, -55}
, {9, -37, -13}
, {15, -4, 25}
, {13, 20, 12}
, {-36, -38, -5}
, {-2, -10, -17}
, {-15, 17, 41}
, {43, 28, 72}
, {-58, -20, -15}
, {-5, -46, 42}
, {-53, 21, -34}
, {44, -47, 82}
, {-14, -35, -1}
, {-39, 79, 4}
, {15, 3, 36}
, {45, -42, -24}
, {8, 0, -21}
, {-39, -60, -16}
, {40, -42, 21}
, {-1, -48, 18}
, {-48, -45, -43}
, {73, -37, 73}
, {30, 8, 11}
, {61, 8, 22}
, {38, -26, -30}
, {18, 44, 62}
, {35, 28, 39}
, {-46, -64, 5}
, {-35, -12, -40}
, {32, -67, -3}
, {-23, 44, -46}
, {-33, 24, 14}
, {-41, -49, 9}
, {26, -19, -7}
, {-16, 15, 37}
, {-29, -9, -49}
, {20, -9, -4}
, {26, -20, 27}
, {-49, 20, 28}
, {-4, 28, 8}
, {35, 0, 49}
, {-8, -12, 11}
, {-3, 7, -29}
, {16, 30, -3}
, {23, -53, 58}
, {68, 38, 28}
, {-23, 46, 29}
, {26, -34, -5}
}
, {{33, -28, -9}
, {25, 18, 14}
, {24, -31, 0}
, {39, -23, 34}
, {36, -28, -1}
, {29, 58, 42}
, {-20, 39, -54}
, {-16, -50, -62}
, {46, -67, 41}
, {-5, 48, -26}
, {-15, -62, -45}
, {-16, -29, 26}
, {-6, -5, -32}
, {-2, 40, -6}
, {-44, 36, -32}
, {4, 26, 23}
, {-34, 8, -5}
, {35, 31, 43}
, {11, 37, -2}
, {28, -28, 9}
, {28, 73, 51}
, {-43, -20, -42}
, {14, 8, -19}
, {-7, -52, -6}
, {-17, 29, 18}
, {47, -9, 38}
, {21, -50, -40}
, {1, -17, 23}
, {-22, 32, 56}
, {45, -43, 60}
, {-61, -67, 8}
, {-20, 32, 27}
, {-22, -20, -24}
, {43, 33, -32}
, {52, -28, -7}
, {-8, -42, 0}
, {36, 44, -15}
, {26, 27, 45}
, {6, 52, 35}
, {70, 31, 12}
, {37, -11, -14}
, {12, -37, -23}
, {47, 25, 9}
, {53, -28, 1}
, {-26, -48, -12}
, {-14, -38, -6}
, {43, 37, -52}
, {-49, -13, 38}
, {45, 58, -38}
, {32, 53, 41}
, {-41, -14, -32}
, {45, 35, -19}
, {-8, -20, -28}
, {-1, 42, 48}
, {41, 43, -4}
, {-3, -19, -12}
, {-9, 13, -46}
, {14, 32, -59}
, {12, 23, -9}
, {-15, 10, 10}
, {-27, 67, -35}
, {-28, 6, -77}
, {-42, -51, -19}
, {34, 23, 25}
}
, {{31, 15, 25}
, {-24, -8, -39}
, {5, 80, 40}
, {22, -40, -27}
, {19, 21, -14}
, {-21, 70, 3}
, {27, -31, -47}
, {22, -53, -53}
, {-21, 2, 10}
, {-46, -8, -9}
, {12, 4, -45}
, {-25, 13, 0}
, {13, -22, -34}
, {50, -8, 64}
, {-25, 14, 25}
, {14, 11, 15}
, {-30, -34, -13}
, {27, 37, 23}
, {-45, -47, 41}
, {7, 13, 15}
, {-14, 22, 41}
, {25, 39, -33}
, {2, 16, -51}
, {39, -60, 31}
, {16, -2, 28}
, {-27, 19, -21}
, {-2, -11, -22}
, {-13, -34, 39}
, {52, -35, 7}
, {-40, -34, -17}
, {2, -24, 57}
, {36, -13, 12}
, {24, 34, -9}
, {-22, 46, 47}
, {0, 64, 59}
, {-26, 64, 11}
, {-31, -11, -2}
, {47, 19, -3}
, {28, 81, 63}
, {20, 31, 15}
, {3, -7, -27}
, {-12, 28, -23}
, {9, 4, -15}
, {-7, 57, 80}
, {56, -18, 0}
, {6, -4, 50}
, {43, 1, -8}
, {-2, 27, -13}
, {30, -36, -38}
, {51, 50, -51}
, {34, -41, 22}
, {29, -5, 1}
, {6, 8, 1}
, {38, 17, -44}
, {23, 1, -3}
, {36, 12, 28}
, {11, -19, -35}
, {9, 30, 4}
, {40, 8, 15}
, {38, 54, -24}
, {-18, 49, 62}
, {12, -20, -22}
, {-28, -31, 39}
, {-36, -24, -7}
}
, {{-14, -25, 12}
, {50, -44, 0}
, {48, 22, -25}
, {12, 19, 20}
, {-28, 6, 21}
, {-41, 24, -36}
, {-30, 48, 0}
, {-20, -10, -6}
, {-41, -32, 17}
, {15, 46, 1}
, {55, 64, -28}
, {46, 44, 44}
, {47, -7, -11}
, {-19, -48, 31}
, {23, -51, -2}
, {35, 14, 38}
, {-25, -44, -18}
, {-36, 18, -33}
, {-45, 19, 19}
, {-14, -40, -23}
, {-32, -10, -7}
, {6, -25, 1}
, {18, -41, 30}
, {46, 20, 3}
, {-15, -8, 28}
, {34, 18, -40}
, {18, 32, 55}
, {0, 45, 0}
, {-37, 3, 32}
, {38, 62, 80}
, {70, -55, 14}
, {45, -52, -44}
, {-23, -33, -43}
, {44, 20, 25}
, {-48, -33, -30}
, {-41, -54, 7}
, {-50, 28, -45}
, {22, 51, 36}
, {-55, -6, 32}
, {-20, -18, 18}
, {29, -13, 15}
, {48, 5, 24}
, {46, 32, 9}
, {-25, 0, -16}
, {-1, -49, -45}
, {45, -19, -48}
, {-6, 54, -43}
, {-5, 21, 33}
, {27, 48, -30}
, {5, 4, 39}
, {7, 14, -18}
, {38, 8, -19}
, {-10, -73, -42}
, {-12, 38, 16}
, {29, 35, -23}
, {10, -22, -36}
, {-20, -23, 9}
, {-5, 21, 16}
, {16, 20, -1}
, {-48, -13, -51}
, {58, -42, 11}
, {-25, 48, 15}
, {6, -26, -41}
, {-3, 40, -21}
}
, {{-11, -1, 45}
, {-39, -67, -11}
, {-13, -5, 14}
, {-34, -15, 39}
, {-14, 44, -25}
, {52, 4, -20}
, {33, -52, -34}
, {-54, -39, -27}
, {-28, -27, -11}
, {-37, -10, 15}
, {-3, 23, 23}
, {-15, 33, -11}
, {-12, -35, 50}
, {-52, 51, -32}
, {-6, -22, 27}
, {-8, -4, 22}
, {13, 38, 36}
, {-16, -3, 0}
, {21, 46, 7}
, {-7, -20, 18}
, {2, 12, 7}
, {50, 12, 6}
, {-17, -3, -19}
, {9, -10, -8}
, {-35, -31, -38}
, {30, -23, -21}
, {-31, 37, 28}
, {1, 11, -8}
, {-46, 34, 4}
, {34, 3, -23}
, {0, -8, 16}
, {0, -34, -20}
, {34, -3, 14}
, {15, 35, -47}
, {-20, 0, -2}
, {-36, -40, -41}
, {35, -14, -6}
, {17, 11, 0}
, {62, -6, -18}
, {34, 5, -1}
, {-48, 8, -7}
, {-75, -40, 14}
, {2, 9, -50}
, {15, 13, 20}
, {1, 42, 20}
, {63, 70, -25}
, {29, 47, 12}
, {-34, -2, -20}
, {-36, -22, 43}
, {29, 32, 8}
, {-25, -27, 38}
, {23, -43, -10}
, {20, 65, -25}
, {38, 35, -44}
, {14, -3, 29}
, {9, -7, 0}
, {12, 1, -41}
, {33, -44, 39}
, {42, 20, 29}
, {-6, 23, 58}
, {-3, 61, -12}
, {-70, -57, -72}
, {45, 6, -26}
, {-27, 41, 43}
}
, {{-32, -31, 0}
, {33, -36, -21}
, {39, -44, 7}
, {-43, 29, 65}
, {29, 49, 31}
, {19, -28, 25}
, {-37, 20, -4}
, {98, 89, 106}
, {21, -1, 44}
, {-33, -5, 4}
, {44, 18, -19}
, {47, -27, -20}
, {14, -41, 60}
, {-16, 36, -44}
, {0, 0, -32}
, {30, -12, -18}
, {62, 27, 40}
, {19, -35, 18}
, {56, 29, -9}
, {6, 35, -57}
, {38, -32, -50}
, {7, 70, -7}
, {40, 6, 34}
, {-36, 50, -24}
, {38, -33, -2}
, {-53, -37, -50}
, {-12, -29, 18}
, {30, -10, -39}
, {29, 32, -45}
, {-3, -36, -37}
, {6, -37, -16}
, {-19, -21, -53}
, {20, 30, -39}
, {-43, -9, -13}
, {-32, -22, 22}
, {-29, 10, 57}
, {50, -30, 36}
, {39, -83, -42}
, {-17, -61, 18}
, {-22, -56, -20}
, {14, -3, -46}
, {-18, 50, -27}
, {15, 47, -23}
, {43, -11, -38}
, {12, -15, -37}
, {-8, 72, 31}
, {45, 53, -30}
, {-21, 35, 55}
, {37, 4, -10}
, {-4, -65, -21}
, {56, 68, 0}
, {19, -18, 5}
, {-34, 21, 2}
, {48, 38, -28}
, {26, -42, -52}
, {-55, -17, -47}
, {-14, -22, 37}
, {38, 23, 49}
, {0, 42, -4}
, {-34, 19, 1}
, {-25, -25, -58}
, {-15, 58, -32}
, {-28, -4, 64}
, {52, 41, -20}
}
, {{14, -35, 11}
, {-12, 4, 41}
, {18, 5, -15}
, {-21, -22, -25}
, {40, 20, -30}
, {2, -47, 25}
, {-29, -52, 34}
, {-38, 16, 24}
, {-57, -40, 4}
, {-35, 31, 42}
, {39, 16, -10}
, {45, -13, 21}
, {26, 34, 0}
, {37, -32, 39}
, {-4, -13, 28}
, {10, 18, -7}
, {-35, 21, -49}
, {1, 29, 21}
, {24, 0, 15}
, {-43, -46, -1}
, {22, 16, -40}
, {-10, 39, 32}
, {-54, 0, -6}
, {10, 3, 61}
, {-20, -17, 23}
, {25, -7, -11}
, {37, 6, 3}
, {40, -14, 65}
, {-33, -24, 26}
, {-20, -22, -4}
, {-38, -30, 51}
, {-51, -49, 32}
, {15, -19, 14}
, {49, 35, -48}
, {-27, 6, -33}
, {13, -56, 25}
, {10, 22, 9}
, {1, -24, -33}
, {-3, -7, 40}
, {-3, -6, 23}
, {-49, -30, -23}
, {-15, 81, 81}
, {-13, 19, 37}
, {5, -14, -81}
, {-69, -14, -50}
, {-17, -24, -22}
, {-17, 8, -21}
, {-28, 38, -37}
, {-30, 9, 44}
, {-4, -3, 42}
, {53, 17, -5}
, {28, 36, 0}
, {6, -3, 5}
, {8, -48, 21}
, {-51, -58, 17}
, {-8, -39, -19}
, {41, 12, -11}
, {-3, 59, 53}
, {26, -21, -13}
, {32, -48, -45}
, {14, -24, 46}
, {46, 30, 59}
, {0, -11, 24}
, {25, 57, 26}
}
, {{-7, 36, -39}
, {-3, -41, -23}
, {-18, 5, -61}
, {19, -18, -1}
, {46, -11, -12}
, {6, 5, 36}
, {24, 10, 21}
, {-18, 44, -37}
, {-48, -40, -44}
, {36, 11, -12}
, {24, 21, -25}
, {-47, 6, 8}
, {26, 2, 12}
, {-62, 14, 19}
, {0, 46, -17}
, {15, 20, -11}
, {-43, -15, -1}
, {5, -36, 24}
, {-54, -19, -31}
, {3, -30, -55}
, {28, 30, 9}
, {-2, 20, -38}
, {26, -8, -32}
, {-17, -48, -15}
, {-37, 35, 34}
, {-51, 44, -27}
, {34, -5, 12}
, {26, 56, 10}
, {30, 39, -54}
, {-20, -25, -19}
, {25, 36, -13}
, {-21, -31, 29}
, {-24, 41, 14}
, {-31, 28, 36}
, {-56, 14, -39}
, {24, 18, 31}
, {-57, 4, -36}
, {-32, 11, 49}
, {-24, 18, -45}
, {13, 27, -43}
, {13, -14, -44}
, {-53, 44, -42}
, {-32, 22, -47}
, {40, 4, -34}
, {-11, 10, -37}
, {34, -55, 24}
, {-35, -17, -14}
, {34, 31, -54}
, {-11, -17, -6}
, {-58, 25, -28}
, {-51, 37, -16}
, {7, 3, -39}
, {37, -6, 20}
, {-24, -53, -19}
, {28, 23, -44}
, {44, -24, -50}
, {15, -58, -14}
, {7, 0, 42}
, {-12, -37, 17}
, {29, -9, 38}
, {6, -4, 28}
, {-32, -51, 26}
, {-28, 27, -39}
, {13, -48, -44}
}
, {{-40, -59, -19}
, {9, 3, -40}
, {9, 34, 18}
, {25, -8, -20}
, {12, -56, 10}
, {-62, 35, -63}
, {11, -46, -27}
, {19, 6, -28}
, {-17, -16, 6}
, {-33, 1, -39}
, {-30, -22, 37}
, {34, -34, 5}
, {0, 27, -45}
, {-59, -21, -31}
, {4, -19, -46}
, {-17, -29, -31}
, {22, 26, -64}
, {18, -32, 29}
, {5, -57, 20}
, {-40, 13, -11}
, {-6, -41, -4}
, {28, 40, 0}
, {22, -40, 9}
, {-26, -69, -34}
, {28, -14, -24}
, {28, -28, 43}
, {-48, 29, -65}
, {7, -58, 1}
, {-28, 37, 1}
, {5, -28, -14}
, {-51, 14, -7}
, {2, -57, -51}
, {24, 54, 32}
, {-14, -48, -12}
, {42, -16, -43}
, {5, 31, -46}
, {-40, -62, -3}
, {-18, 54, -9}
, {33, 28, 7}
, {14, -20, 19}
, {-15, 7, -34}
, {10, 25, 34}
, {-10, -43, 33}
, {16, -55, 30}
, {38, 17, 25}
, {-2, 0, 39}
, {-16, -34, -45}
, {-28, -1, -50}
, {50, 9, 19}
, {-18, 21, 24}
, {-32, -44, -27}
, {8, -57, -38}
, {-6, 13, 44}
, {-38, -35, 24}
, {29, 44, -39}
, {-8, -62, -29}
, {-17, 42, -51}
, {-59, -22, -7}
, {0, -15, -4}
, {-36, 9, -56}
, {-20, 45, -43}
, {-60, 20, 34}
, {2, 36, 19}
, {34, -22, -35}
}
, {{-44, 20, -22}
, {52, 2, -9}
, {-10, 58, -6}
, {6, 10, -58}
, {-15, 11, 32}
, {41, -36, 7}
, {-38, -39, -35}
, {-43, -56, -52}
, {-50, 15, 49}
, {-9, 30, -18}
, {-46, -27, 2}
, {-49, 36, -10}
, {18, -28, -5}
, {20, -20, 4}
, {44, -1, 8}
, {45, -22, 49}
, {-48, -14, -33}
, {35, 40, 47}
, {16, 44, 33}
, {-23, 31, -13}
, {20, 0, 50}
, {42, 26, -42}
, {-6, 24, -14}
, {38, 8, -2}
, {-30, -34, -2}
, {10, -38, -13}
, {-44, 1, -12}
, {-4, -19, 37}
, {-1, 11, -37}
, {-18, 9, -24}
, {-27, -41, -9}
, {28, 50, 37}
, {6, 16, 20}
, {-23, -30, 25}
, {72, 13, -1}
, {-49, -47, -24}
, {37, -22, -10}
, {39, 40, 39}
, {57, 13, 86}
, {-41, -56, 0}
, {-39, 48, 10}
, {0, -73, 22}
, {-2, -24, 47}
, {58, 42, -6}
, {-16, -53, 29}
, {-10, 2, -52}
, {33, 39, 48}
, {44, -24, -40}
, {-35, -6, -18}
, {2, 12, -33}
, {8, 42, -13}
, {25, 39, -2}
, {45, -25, -31}
, {33, 40, -35}
, {-39, -20, 45}
, {-39, -9, 4}
, {-50, -36, -35}
, {-12, -33, 30}
, {-50, 13, -17}
, {3, 11, -31}
, {6, 0, -36}
, {36, 25, -51}
, {30, 36, -19}
, {-11, 29, -2}
}
, {{30, 18, 36}
, {22, 39, 40}
, {47, 3, -32}
, {-30, -21, 3}
, {-22, 27, -35}
, {-31, 78, -12}
, {-34, -44, -27}
, {0, 7, -54}
, {-12, -6, -3}
, {41, -22, 39}
, {3, -70, 9}
, {41, 28, -10}
, {26, -33, 10}
, {-39, -16, 10}
, {34, -31, -2}
, {34, -69, -3}
, {7, -30, -67}
, {63, 2, 11}
, {-38, -41, -29}
, {0, 55, 28}
, {-30, -32, -28}
, {51, -14, -17}
, {18, -39, -29}
, {-21, -37, 14}
, {-13, -1, 11}
, {-22, 20, 13}
, {-11, -37, 47}
, {-39, -61, -20}
, {-38, -25, -7}
, {11, -29, 27}
, {-21, -42, 15}
, {-26, -2, -15}
, {3, -16, 26}
, {45, 12, 19}
, {-15, 57, 7}
, {28, 26, -32}
, {45, -43, -58}
, {-2, 58, -16}
, {-19, 44, -8}
, {48, -28, -41}
, {12, 34, -32}
, {-13, -23, -68}
, {34, 46, 20}
, {-4, -6, 29}
, {-33, 20, -45}
, {24, 51, -25}
, {7, -21, 45}
, {-50, 0, 36}
, {15, 41, -37}
, {52, 4, 20}
, {4, 27, -30}
, {-5, 58, 81}
, {-24, -50, 37}
, {1, 52, -3}
, {7, 61, 64}
, {27, -11, -13}
, {4, -3, 4}
, {31, -12, -8}
, {35, -70, -53}
, {4, 57, 41}
, {60, -19, 37}
, {4, -70, -50}
, {22, -36, 10}
, {-5, -39, -29}
}
, {{17, 40, -49}
, {-3, 62, 2}
, {43, -46, 3}
, {-31, 31, -23}
, {7, 15, -43}
, {-43, -10, -19}
, {34, 21, -17}
, {-30, -47, 3}
, {-23, -26, -39}
, {-6, -4, 25}
, {-52, 21, 8}
, {50, 27, 48}
, {-49, -29, 41}
, {56, 32, 62}
, {17, -43, 12}
, {43, 43, -17}
, {-27, -17, -8}
, {18, 11, -37}
, {-49, -49, 15}
, {-42, -19, 28}
, {7, 23, 4}
, {0, -3, -18}
, {35, 0, -49}
, {56, -23, 37}
, {-34, -18, -31}
, {39, 5, 32}
, {23, 2, 32}
, {80, 47, 0}
, {19, 31, 58}
, {18, 14, 1}
, {47, -58, 51}
, {-8, -2, 33}
, {30, 52, -35}
, {-45, -47, -32}
, {-26, 61, 1}
, {1, 24, -20}
, {15, -6, 11}
, {57, 59, 60}
, {32, 75, -42}
, {45, 34, 49}
, {11, 14, 19}
, {55, 28, 16}
, {-17, 10, -34}
, {-20, 26, -52}
, {19, -24, 36}
, {-36, 5, -41}
, {-21, 12, -28}
, {40, -8, 25}
, {-4, 10, 40}
, {-8, 30, 30}
, {-5, 20, -20}
, {23, 0, 47}
, {-54, -5, -52}
, {-43, 0, -10}
, {31, 47, -19}
, {0, -27, 1}
, {-20, 39, -16}
, {-17, 54, 37}
, {15, 34, -8}
, {-42, -21, -51}
, {-6, 1, -40}
, {15, 12, -13}
, {-5, -20, -21}
, {38, 54, 39}
}
, {{-29, 28, 34}
, {5, -25, 14}
, {-57, -28, -25}
, {64, -24, 38}
, {-28, -17, -23}
, {21, 45, -40}
, {0, -26, 64}
, {22, 78, 0}
, {-33, 18, 14}
, {12, 38, 53}
, {-50, 50, -11}
, {-11, 66, 19}
, {18, 0, 9}
, {17, -18, 4}
, {-11, 47, -8}
, {66, 65, 54}
, {27, 69, 12}
, {2, -14, -8}
, {-6, 61, 56}
, {-19, 1, -60}
, {63, -33, -41}
, {-1, -17, -38}
, {34, -21, -30}
, {66, 43, -4}
, {43, 28, -21}
, {51, -16, -16}
, {-40, 0, 9}
, {37, -12, 3}
, {20, -8, 0}
, {61, -1, 50}
, {16, -33, -23}
, {-32, -31, 8}
, {47, 0, -12}
, {-38, -40, 38}
, {-24, -34, -16}
, {22, -23, -3}
, {-22, -18, 5}
, {-8, 67, 66}
, {-38, 0, 8}
, {2, 69, 26}
, {53, 2, -17}
, {-13, -3, -61}
, {53, 45, 11}
, {-36, 5, -61}
, {-23, 14, -34}
, {30, -32, -10}
, {-40, 41, -16}
, {52, 11, -16}
, {45, 33, 48}
, {30, 8, 7}
, {48, 65, 82}
, {-37, -44, 32}
, {10, 27, -52}
, {-33, -3, -38}
, {17, 21, 23}
, {42, 35, -23}
, {-15, -15, -19}
, {48, 1, 47}
, {-20, -6, -24}
, {0, -45, -5}
, {-23, -8, 17}
, {-8, -38, -20}
, {-20, -2, 51}
, {41, 23, 39}
}
, {{33, -43, 2}
, {4, -25, 63}
, {25, 1, -55}
, {13, -27, -28}
, {-44, -50, 25}
, {-42, 9, -9}
, {-30, 26, 23}
, {19, 1, 15}
, {-47, -52, 20}
, {20, 22, 34}
, {35, -19, -25}
, {-22, 28, 42}
, {-46, 43, 16}
, {-27, -27, 2}
, {15, -41, 5}
, {-18, 57, 14}
, {52, 8, 9}
, {-45, -42, -25}
, {48, 1, 42}
, {4, -22, -9}
, {23, -3, 31}
, {7, -4, 3}
, {22, -15, 40}
, {-14, 48, 12}
, {45, -8, 3}
, {38, 33, 24}
, {17, -12, 24}
, {23, 0, 6}
, {1, 41, 5}
, {65, -13, 14}
, {-24, 14, -57}
, {29, 38, 21}
, {26, 0, 16}
, {-39, -23, -51}
, {24, 6, 29}
, {-56, -16, -57}
, {-37, 25, -21}
, {67, -1, -16}
, {-15, -47, -15}
, {62, -4, -16}
, {-36, 2, -46}
, {11, -47, 0}
, {23, 0, 9}
, {-13, 23, 20}
, {-16, -23, -34}
, {2, 19, -68}
, {-51, 46, -47}
, {41, -46, -12}
, {6, -16, -14}
, {-43, 23, 13}
, {9, -20, 37}
, {46, -2, -31}
, {27, -49, -35}
, {-41, -10, -26}
, {0, 32, 30}
, {52, 6, 5}
, {0, -46, 41}
, {17, 5, 0}
, {47, 41, -2}
, {-25, -21, 37}
, {18, -33, 3}
, {20, 23, 19}
, {7, -26, -3}
, {35, 38, -19}
}
, {{18, 3, -18}
, {5, 16, 69}
, {18, 14, -32}
, {16, 50, 9}
, {31, -24, -4}
, {41, 34, 21}
, {-14, 53, -3}
, {-10, -13, -20}
, {1, 44, -25}
, {-5, -43, -21}
, {-20, 42, 27}
, {32, -45, -33}
, {-25, 48, 17}
, {50, -37, 34}
, {-29, -44, -30}
, {14, -8, 37}
, {-23, -23, 4}
, {-36, 31, 0}
, {22, -47, 36}
, {38, -55, 5}
, {-39, -20, 27}
, {12, -32, 9}
, {14, 8, 2}
, {22, 39, 44}
, {3, 34, -19}
, {34, 9, 13}
, {6, 14, 29}
, {5, 15, -39}
, {49, -35, 12}
, {45, -53, 27}
, {7, 6, -37}
, {-14, -23, -50}
, {-48, 5, 26}
, {18, 31, 18}
, {7, -13, 28}
, {48, -22, -47}
, {31, -9, -32}
, {-41, 72, 32}
, {29, -34, 40}
, {-9, -29, 21}
, {19, -2, 21}
, {64, -46, 57}
, {-44, 52, -4}
, {-61, 14, -40}
, {6, -54, 1}
, {9, 34, -51}
, {-41, 25, -42}
, {0, 21, 49}
, {-31, -21, 16}
, {-16, -9, -23}
, {29, -44, -5}
, {-42, 5, 45}
, {-33, 22, -46}
, {43, -11, 40}
, {-13, -33, -50}
, {42, 46, -23}
, {43, -10, -21}
, {-7, 26, 59}
, {-4, -9, -2}
, {-14, -59, -7}
, {-39, 37, -72}
, {39, 4, 24}
, {-40, -45, -13}
, {2, -6, 50}
}
, {{-38, 48, -28}
, {25, -14, 31}
, {42, 13, -25}
, {3, 9, -30}
, {42, -2, 45}
, {18, -49, -39}
, {18, 24, -19}
, {-9, 97, 2}
, {7, -5, -29}
, {-43, 14, 19}
, {10, -4, 11}
, {7, -36, -2}
, {-13, 20, 53}
, {19, -13, -2}
, {-7, 26, -19}
, {-48, -57, 5}
, {-4, 45, -33}
, {-29, -6, 37}
, {44, 27, -26}
, {17, -21, -23}
, {-6, -18, -20}
, {-15, 46, -30}
, {-25, -34, 11}
, {49, -46, 20}
, {0, -40, -13}
, {45, 36, -6}
, {-32, 38, 57}
, {51, 15, 78}
, {23, -4, -27}
, {41, -4, -2}
, {67, 30, 70}
, {-24, -44, -28}
, {-15, -4, -26}
, {24, -14, -12}
, {46, -47, -36}
, {-17, 25, -47}
, {-16, 44, -22}
, {37, -24, -54}
, {-1, 37, -10}
, {-12, 11, 17}
, {40, -45, -8}
, {-16, 1, 23}
, {-25, -43, -27}
, {14, -3, -45}
, {11, -1, 25}
, {43, -6, 12}
, {45, 47, 3}
, {26, -35, -28}
, {29, 10, -26}
, {29, -52, 17}
, {9, 59, 45}
, {-1, 28, 16}
, {-21, -43, -5}
, {-1, 32, -48}
, {3, -10, 46}
, {20, 28, 43}
, {52, 5, 20}
, {42, 1, 41}
, {-21, -20, -13}
, {16, 10, -40}
, {21, -51, -9}
, {0, 27, 60}
, {0, 51, 37}
, {-16, 16, -12}
}
, {{4, -32, 11}
, {15, -68, -1}
, {5, 26, 35}
, {10, -19, -20}
, {49, -31, 45}
, {23, -54, 33}
, {23, 9, 70}
, {60, 110, 75}
, {35, 125, 30}
, {-14, 16, -30}
, {2, -23, 1}
, {35, 20, 28}
, {-37, 57, -33}
, {22, -9, -53}
, {-2, -34, -53}
, {-6, 13, -8}
, {0, 18, -31}
, {22, 1, 19}
, {81, 85, 25}
, {45, 25, -14}
, {13, -54, -68}
, {21, 15, -36}
, {64, 31, -5}
, {48, 45, -15}
, {37, -7, -28}
, {-32, 9, 18}
, {30, 29, -19}
, {31, -21, -5}
, {-53, -33, -7}
, {28, 4, 25}
, {-3, -6, -9}
, {0, -47, -25}
, {-18, 31, 25}
, {-33, 19, -38}
, {-51, 9, 14}
, {-25, -16, -30}
, {-41, 59, 19}
, {-11, 1, 23}
, {27, -52, -40}
, {6, -16, 18}
, {41, -6, 23}
, {19, 49, 5}
, {-16, 29, -5}
, {-30, -34, -32}
, {-10, 4, -22}
, {98, 16, 47}
, {-51, -48, -19}
, {11, -2, 55}
, {28, -56, -30}
, {-49, -40, 3}
, {82, 28, 87}
, {-45, 26, -27}
, {-30, 53, 16}
, {-58, 30, 26}
, {-42, 33, -8}
, {-19, -19, -47}
, {50, 34, -38}
, {-31, -28, 30}
, {-42, -4, 3}
, {-13, -4, -21}
, {-59, 2, 0}
, {19, 47, -12}
, {-40, 0, 13}
, {36, -5, -47}
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

typedef number_t max_pooling1d_44_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_44(
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

typedef number_t average_pooling1d_8_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d_8(
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

//typedef number_t *flatten_8_output_type;
typedef number_t flatten_8_output_type[OUTPUT_DIM];

#define flatten_8 //noop (IN, OUT)  OUT = (number_t*)IN

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

typedef number_t dense_16_output_type[FC_UNITS];

static inline void dense_16(
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


const int16_t dense_16_bias[FC_UNITS] = {-7, -21, -7, -9, 23, 7, 18, -18, -19, 10, 5, -11, -5, -4, -10, -1, -10, -3, 6, 6, -16, 19, 1, -17, -6, 20, -17, -5, -14, -13, -16, -15, 4, 19, 11, -3, 14, 5, -17, 1}
;

const int16_t dense_16_kernel[FC_UNITS][INPUT_SAMPLES] = {{91, 42, 13, -8, -94, -79, -68, 26, -108, -105, 18, 68, -27, -113, -92, -66, -4, -11, 31, -118, -78, -34, -24, -33, 79, -4, -74, 7, -60, -19, -93, -33, 73, -24, -46, -29, -34, 45, -2, -5, 20, -59, 78, 55, 17, -68, 90, 1, 11, 42, -90, -27, -57, 21, -94, -68, -66, 26, 59, 37, 107, -42, 97, -59, 77, -35, 1, -48, -33, -41, -78, -70, 2, -2, -34, -81, -25, 25, -95, 86, -6, 64, -5, -38, -71, 42, 25, -43, -2, 73, -84, 29, 62, 102, -54, 96, 57, -1, -83, -73, 68, -30, -15, -76, -44, -51, 51, 9, -15, -91, 74, -90, -41, -66, -81, 71, 19, 82, 71, 0, 33, 27, -83, -93, -98, 47, -55, 26}
, {-83, -27, 50, 51, 2, 77, 17, -14, -54, 69, -74, 24, 52, -18, -45, 56, -93, -40, -96, -48, 88, 71, 104, 5, -26, 76, -104, 64, -59, -94, 86, -62, 27, 33, -75, -73, 67, -11, -94, -80, 57, 70, 99, -46, -43, -25, 92, -5, 65, 88, 28, -110, 4, 53, -15, -70, 37, 80, 1, -50, -11, -56, -3, -58, 59, -86, 79, -11, -47, -12, -87, 53, 79, -87, 51, -32, -16, 75, 86, 70, 47, -59, -98, -83, 11, 57, -37, 50, -56, -41, -23, 71, -70, 50, -36, -47, 20, 52, -67, 20, 64, -25, -37, 77, 93, 23, 59, 28, 48, -93, 92, -10, 70, -26, -93, -39, 76, 10, -23, 12, -3, 52, -82, 0, -15, -76, 0, 94}
, {68, 79, -26, 74, 47, 62, -90, 49, 113, -4, -60, 0, -27, 35, 62, 8, -86, 35, 25, 34, 50, -44, 48, 109, -76, -81, -47, 27, 89, -77, -37, 53, 101, -24, 49, 12, -43, -44, 70, 74, 34, 83, 66, -83, -29, 75, 90, 51, -65, -60, 92, 62, -84, -93, -56, -36, 96, -22, -128, -88, -119, -52, -29, -58, 104, -2, -42, 60, 104, 50, -89, -108, -86, -2, 64, -24, 0, 53, 23, 8, -2, 32, -90, -2, 77, 94, -46, 35, -51, 87, 15, 79, -61, -105, -6, 38, -63, 92, 64, 38, 63, 18, 12, 133, -60, -78, 48, -19, 35, 1, -57, 33, -71, -79, 77, -95, 38, 4, -30, 27, -91, -77, -29, 28, 44, 104, 79, 84}
, {-33, 77, -119, -3, 69, 81, 52, -2, 3, -40, -86, 30, -30, 77, -41, -12, 23, 50, 106, -86, 96, 35, -94, 64, -57, -1, -61, 61, -31, 52, -106, -23, -46, -3, 82, -84, -35, 18, 53, 3, 97, 21, -2, 5, 59, 96, -15, 102, -36, -11, 43, 56, 91, -44, -53, -40, 6, 30, -42, 14, -103, -12, 69, 78, 64, -58, -40, -33, -55, 66, 70, -79, 28, -41, 5, -18, 7, 43, -74, 40, 44, 83, 61, -14, 13, -55, -101, 71, 26, -86, -24, -19, -98, -69, -19, 11, 54, -43, 20, 46, -58, 110, 11, 129, -90, 27, -91, -91, -58, -50, 71, -50, 87, 51, 61, -79, 6, 74, -40, 55, 52, -93, 112, 51, 71, 37, 52, -1}
, {48, -20, -81, -79, 96, 25, 54, -83, 119, -39, -48, -59, -13, -9, 50, -56, 32, 60, 94, 31, -90, -100, -59, -28, 54, 9, 15, -122, -15, -22, -49, 27, 48, 4, 34, -51, -14, -16, 60, 86, 8, -69, 12, 51, 31, 80, -89, -24, 24, -77, -45, 112, 75, -87, 40, 48, 56, -28, -31, 76, -102, -44, 28, 95, 35, 18, 22, -4, 63, 140, 57, -11, 56, 144, 2, 80, -58, -16, -44, -29, -14, -17, -82, 58, -76, 83, 107, 29, 112, -6, -56, 19, 56, 106, 32, 69, -67, 26, -37, -16, -1, 18, -23, -84, -30, -88, -60, 21, -36, 78, 96, 110, 32, 52, -40, -50, -20, -51, -38, -3, 46, -8, -81, -98, -96, -60, 53, -31}
, {4, 60, 100, -11, 96, -62, -27, 37, 7, -19, -61, -64, -58, 68, 14, 84, -11, 27, 0, 91, 11, 3, 85, 29, 5, 59, -27, -122, 51, -49, -42, -30, -26, -94, -10, 32, -29, -16, -90, 92, 5, -54, -63, -52, 66, 6, 37, 8, 105, -94, 51, 35, -36, 83, -83, -79, 51, -79, 56, -55, -7, 33, 33, -45, 38, 60, 7, -80, -3, 72, 49, -49, -16, 77, -37, 31, -78, 85, -41, -31, -53, 45, 61, -62, 15, 3, -84, -7, 82, -38, -92, -16, -89, -22, 86, -50, -38, 3, -16, -3, 74, -56, 17, -98, -67, -38, -76, 16, -47, -36, -18, -23, 91, 78, 105, -68, -88, -84, -91, -10, -25, 106, -35, -5, -41, 76, -65, -64}
, {-43, 23, 94, -44, -72, 18, -56, -69, 38, 71, 82, 49, -36, 34, 4, -103, 28, 27, -85, 53, 22, 56, 65, -28, 30, -90, 41, -136, -82, -21, -48, -64, 9, -29, -39, -30, 52, -46, 59, 83, 14, 104, 24, -64, 72, -59, -71, 67, 28, -35, -83, -25, 19, -3, -18, -88, -9, 58, -41, -21, 85, -49, 80, 32, -4, -66, 30, 75, 95, 28, -15, 67, 45, 122, -51, -1, -66, -45, -1, 50, -57, -52, -42, -37, 60, -91, 46, 6, 119, -83, 21, 90, 35, -42, -55, -90, -30, -77, 88, 81, -30, -8, -80, 12, -86, 37, -97, 69, -32, 96, -68, 51, 58, -49, 37, 24, -76, 10, 72, 63, -13, -32, 3, 62, -75, -42, 68, -6}
, {73, -4, -58, 0, -54, -31, -77, -92, -137, -47, 82, 75, -61, 61, 42, 80, 13, 7, -92, -72, 61, 103, -29, -17, -54, 88, -2, 102, 58, -2, 19, 21, -42, 13, -79, -38, 43, 35, -96, 40, 16, -4, -61, 51, 4, 57, -40, 61, -75, 52, -10, -68, 9, -51, 28, 53, -75, -76, -33, 90, 20, 0, -20, -37, 51, -8, 52, 11, -100, -50, -77, 77, -4, 33, -3, 12, -73, -19, 78, 86, -80, 77, 37, 7, 2, 53, 24, -41, -102, 18, 50, 77, -96, -65, -4, 59, -5, 58, -40, 64, -5, -80, -51, 23, -3, -47, -9, -23, 14, -67, -72, -119, -2, -89, -102, 88, 69, -79, -86, 65, 39, -8, 44, 116, -54, -35, 32, -27}
, {20, -20, -51, -88, -67, -92, -38, -30, 36, 66, 34, -58, 86, 0, -49, 9, 26, -65, -9, 25, 38, -12, 26, -41, 86, -47, -31, 87, -15, 37, -71, 30, 73, -18, -7, 80, -87, -18, -25, -68, 48, -72, 35, -71, 38, 75, 19, -2, -33, 69, 81, -76, 69, -8, -61, 23, -33, -70, -15, -62, 89, -50, 94, 70, 34, -85, -62, -17, -102, -124, -42, -69, -11, 23, -11, 51, -87, -6, 24, 69, 33, 26, -30, 24, 59, 22, 87, 7, 52, -77, -35, -86, -74, -32, 72, -77, 7, -57, 69, -49, 57, 76, 17, 29, -52, -89, 35, 65, 44, -55, -63, 30, -87, 41, 80, 92, 33, -93, 12, -19, -12, 19, -89, 89, 95, -76, 43, 53}
, {90, 102, 77, 15, -24, -91, 3, 90, -8, -81, 46, 96, 70, -29, 59, -74, -37, 51, -59, -60, 22, 62, -34, -77, -24, -90, 44, 54, 114, 126, 104, 56, -1, 39, -91, 83, -73, 37, 61, -60, -97, -37, -3, -70, 66, 0, -19, 1, -64, 68, 33, 77, 83, 78, -66, 19, -89, -80, -24, 45, 37, 66, 59, -39, 15, -87, 80, -84, -106, -64, 67, 100, -42, -30, -61, 57, 65, 8, -9, 30, -77, 43, -7, 76, 41, -14, 84, 63, 1, 64, -11, 22, 10, -16, -4, 100, 46, 3, 2, -42, -66, -114, -87, -81, 95, -80, 111, 79, 48, 13, 50, -80, -20, 15, -7, 31, 13, -28, 10, -54, 39, 55, 1, -38, -70, -34, -83, 47}
, {50, -13, -3, -4, -18, 9, -33, 26, 35, -21, 92, 38, -3, -61, -68, 10, -53, -81, -7, 113, 104, 35, -84, -26, -72, -60, 69, -40, 58, -96, -84, 46, -59, -31, -52, -31, -40, 21, -48, 55, -86, -44, -58, 92, 1, -22, 40, 84, -9, -28, 89, 5, 26, 40, -48, 15, 36, 79, -112, -22, -77, -67, 17, 19, 114, -4, 35, 59, -80, 68, 43, -67, -45, 102, -4, -37, 31, 49, 67, 64, -17, 16, 25, -20, -26, -7, 90, -85, -32, 8, 91, 61, -20, -19, -23, 29, -69, -59, 84, 87, 46, 90, 77, 117, 15, -42, -36, 72, 59, 73, -11, -17, 59, -49, 58, 33, -19, 0, -75, 44, -46, 8, -48, -92, 57, 13, 20, -65}
, {51, -52, -49, -70, -67, -66, -24, -57, 15, -46, 96, 5, 82, -61, -44, 17, -20, 55, -69, -70, 97, 56, 64, 84, 90, 15, -40, 139, 9, -76, 3, 12, 77, 71, 56, 41, -45, 91, 69, 56, -29, -29, 0, 4, 43, -72, -68, 27, -82, 85, -67, 5, -61, 54, 94, -62, -96, 94, 53, -3, 94, 53, -21, 40, 98, 6, -94, -83, -69, -3, 64, -9, 15, 26, 80, -65, -22, -44, -36, 42, -27, -13, 54, 103, -82, 28, 84, -63, -48, 62, 78, -7, 62, 84, 12, -56, -92, 62, 16, 66, 46, -45, 88, -73, -60, -109, 85, -4, 34, -93, 85, -49, 81, 4, 3, 15, 121, 40, -31, -52, 81, -28, 26, 78, 6, -64, 89, 1}
, {-17, -73, -27, 85, 38, 10, -10, -17, 8, 81, -43, 89, -9, -45, 11, -23, -5, 65, -52, 38, -33, -86, 10, 16, -13, -14, -14, 65, 31, 53, -43, 10, 49, 31, -13, -13, 27, -28, 41, 39, -5, 28, -69, -94, -104, -40, -71, 54, -49, -75, 47, -83, 74, 48, -44, -2, -41, 77, 82, 29, 117, -94, -69, 81, -81, 61, -8, 18, -65, -74, 38, 1, 56, 28, 75, 32, 81, 28, 29, 18, 48, -89, 90, -48, -23, 70, -6, 85, -99, -79, -56, -3, 61, 43, 2, 34, 16, 6, 35, 33, -17, -5, 72, -122, -50, -17, -66, 61, 36, -26, 38, -82, 78, -66, -58, -32, -99, -34, -47, 89, -82, 43, 89, -68, 42, -28, -19, -56}
, {82, -62, -60, -50, 27, -51, -49, 8, 61, 87, 16, 33, -35, -63, -55, 69, -64, -51, -25, 45, -9, 42, 39, 98, -33, -40, 63, 7, -7, 35, 11, -86, 59, -91, -31, -16, -25, 69, 77, -80, -57, 4, -25, -65, 5, -41, -40, 82, 64, 104, 52, 39, 37, -103, 52, 18, 78, -62, 26, 39, -85, 67, -69, -5, 26, 66, -43, 11, 85, -18, -98, 30, -6, 0, 59, -7, -77, -18, 104, 27, -43, 50, 74, 60, -90, -56, -55, -36, -50, -65, 84, 46, 30, -21, 52, -49, 62, 55, -84, -68, 44, -17, 1, 122, -48, 39, -37, 42, -100, 36, -85, -61, -38, -94, 32, -29, -69, -4, 1, 48, -19, 14, 83, 23, 61, -25, -22, -18}
, {-55, -39, 20, -15, 80, 20, -7, -5, -66, -59, 33, 83, 80, -18, -93, 57, 1, 84, 63, -6, -12, 33, -82, -41, 65, -33, -45, 111, 33, -8, 53, -61, 74, -78, 85, -41, -64, -74, 60, -74, 29, -27, -32, 75, 16, 17, 4, -97, 23, -12, 11, -7, 66, -92, 51, 9, -85, -58, -70, -42, 9, 53, -5, -26, 90, -62, -68, -71, -23, 32, 44, -17, -65, 1, -70, 65, 0, 67, -45, -90, 35, -2, 9, 5, 27, -32, 60, 21, 53, 19, -63, -85, 89, 37, -46, 100, -102, -82, 13, -19, 28, -34, -58, -32, -83, -24, 90, 18, 55, -30, 93, 60, -77, -91, -83, 51, 135, 13, 12, 27, 60, -67, -29, 18, -65, 3, 38, 0}
, {-19, 23, 30, -20, 40, 84, -30, -30, 42, -50, 64, -86, 11, -71, 39, -81, -41, 59, 60, -27, -65, -16, -5, 64, 55, 10, -94, -39, -2, 50, -78, 98, 42, 11, 16, -47, 63, 70, -36, -50, -11, -89, 72, -28, -69, -21, 79, -49, -1, 55, 79, 44, 58, -88, 31, -89, 29, 15, -52, 54, 12, 43, 48, 88, 90, 22, -88, -10, 5, 91, -11, -94, -42, 9, 44, -92, -99, -36, -10, -74, 67, -72, -77, 10, -58, 90, 75, -87, 56, 61, 17, -29, -18, 43, 86, 94, -97, -39, -32, -55, 64, -36, -80, 95, -44, 56, -50, 76, -43, 51, 62, 99, -102, -28, -31, 0, 88, 71, 70, -18, 18, 22, 75, -36, -75, 36, 48, 80}
, {102, -84, -30, 43, -11, 62, 67, -66, -51, -19, 53, 91, 40, 36, 72, -46, -61, 93, -47, -11, -5, -80, 73, -20, 41, 50, 13, 47, -23, 102, 28, -69, 90, 37, 33, 46, 54, 52, -16, -68, 52, -70, -11, 1, 35, -42, 15, -77, 32, 6, 84, -104, 58, -21, -57, 14, 11, 55, -34, 92, -23, -20, 64, 68, -6, -41, 73, -18, 27, -12, -47, -69, 72, -48, 8, 54, 25, -61, -18, -62, -90, -48, -20, 79, -52, -13, 58, 89, -34, -5, 2, 97, -19, 83, 75, -11, -38, 73, 39, -60, 0, -38, -22, -63, -62, 26, 80, -44, 101, -99, 24, -51, -62, -21, 44, -46, 115, -86, -88, -57, 44, 7, -93, 17, 29, -1, -19, 8}
, {11, 28, -62, -37, -82, -30, 6, 12, 97, 96, -33, -39, 7, 57, -27, 77, -23, 72, -15, 86, -70, 90, -17, 94, 33, 70, 24, -10, -57, -11, 12, 87, 73, 1, 44, 37, -23, 47, -60, -27, 99, 11, 0, -96, 55, 15, -93, -42, -29, -57, -67, -38, 60, 89, -7, -66, 11, 46, -53, -1, -121, -47, 40, -33, 37, 65, 37, 33, 97, 72, -99, -72, -88, -33, 81, 9, 92, 77, 105, -79, -26, 53, -29, -25, 22, -20, -9, -56, 55, -97, -22, -78, -22, -27, 92, -91, -14, 52, 73, -85, -96, 58, 44, 95, 71, -55, -1, 52, -24, -6, 39, -13, -50, -74, 84, 34, -92, -44, -24, 68, -44, -46, -62, 1, -1, 41, -12, -19}
, {111, 4, 78, -46, -60, -72, 5, 112, 5, -66, -86, -18, -42, -61, -21, -92, -50, 52, -69, -69, 75, -62, 44, -32, 22, 62, -38, 114, -23, 3, -11, 13, 99, -47, -41, 42, -38, 51, 70, -35, 71, 55, 83, 3, -44, 20, 34, 15, 84, 12, 8, -82, 80, -87, -78, 88, 11, -50, 3, 33, 28, 69, -62, 73, 40, -21, 87, 45, -23, -81, -42, -38, -1, -54, -5, 55, 45, -19, -47, -67, 87, 11, -47, -53, 94, -21, 73, -56, -60, 48, 42, 0, 26, 33, 57, 78, -44, -74, -28, -48, -74, -58, 26, -92, -82, 13, 112, -4, 84, 7, -75, -109, 16, 37, -25, 31, 82, 29, -28, -54, 73, -32, -21, -92, 37, 36, 2, 61}
, {-40, 30, -107, -78, 25, 69, 51, 3, 68, -17, -72, -71, -78, 51, 11, -15, 52, -86, 95, 88, 68, 83, -67, -11, 14, 61, 13, -77, -88, 24, 20, -39, 85, 3, 27, -63, 7, 92, 80, -37, 35, -93, 33, -56, 54, 54, -1, -81, 33, 19, 72, 68, 14, 20, -4, -46, 52, 27, 32, 47, -114, 3, -101, -3, 115, -14, 79, 26, 85, 64, -47, 29, 23, 136, -28, 16, -41, 74, 47, -75, -10, -23, 77, -67, -64, 24, -70, 44, 34, -40, -13, -70, 21, -27, -39, -27, 1, 21, -75, 76, -13, 23, 84, -7, -41, -60, 23, -93, -90, 102, -64, 87, -85, 41, -18, -12, 83, -2, 69, -31, -34, -81, 25, 15, -72, 106, 5, 71}
, {80, -13, -23, -60, 46, -96, -10, 84, 19, -101, 41, 95, 52, -38, -51, -60, 90, -9, 21, -30, -43, 75, 63, -19, 2, -25, 54, 19, -1, -18, 83, 19, 17, 79, 53, 0, -61, 70, 89, -11, -83, -1, -32, -77, 47, 17, 95, 46, 70, 88, 67, 61, -20, 31, -20, 76, -19, 90, 70, 73, -37, 89, -4, -98, 39, -28, -21, 1, -45, -129, -23, -65, -11, -111, 74, -36, 64, 80, 9, -72, 20, 99, 31, -57, 28, 86, 50, -53, -7, -72, -83, 89, -95, 4, -57, -58, -22, -5, -79, 8, -30, -30, 100, 95, -42, -79, 23, 95, 63, 53, 7, -10, 74, -21, 61, 7, -11, -43, 52, 16, 85, 1, 39, 90, -50, 14, -67, 15}
, {1, -53, -28, 69, -62, 98, -6, -61, -4, 59, 47, -35, 88, 71, 115, -28, 57, 60, -85, 47, 8, -88, -99, -8, -85, -72, 90, -117, 53, -39, 58, 95, -98, -63, -60, -35, 85, -78, 5, 81, 7, 18, -89, 80, 56, 19, 9, 108, 57, 50, 46, 42, -69, -18, -56, -97, -38, -84, -41, -39, 8, -67, -38, -40, 76, 17, -75, 11, 105, -3, 5, 52, -10, 17, 25, 53, -9, 49, 102, -11, -41, 69, -4, -10, 107, -87, -77, -53, 9, -46, -29, -31, -79, -78, 34, 5, -18, 89, 68, 106, -54, -62, 2, -87, 0, 36, -73, -65, -29, -1, -16, 48, -14, 12, -38, -54, -38, 46, 40, 90, -52, 20, 54, -77, -3, 9, 25, 2}
, {-11, 86, -5, 15, 11, -3, 10, -27, -4, -21, -19, -14, -51, 89, 5, -11, -56, 99, 28, 75, 99, 12, 86, 112, -5, -64, -41, 98, -10, -112, -86, -67, 3, 96, -15, -75, 34, 83, -25, 31, -60, 85, -66, -41, 45, -27, 16, -54, -78, 72, -4, 13, -45, -125, 71, 25, -58, 16, -71, -63, -69, -88, -32, -17, -59, 92, -68, -28, -89, -82, 71, -80, -48, 69, -65, -85, -93, -76, 47, -47, -86, -2, -8, 35, -50, -37, 34, -52, -53, -44, -57, -16, -40, -89, 9, 1, -39, -55, 77, 77, 15, -26, -29, 1, -81, -3, 82, 16, -8, -1, 4, 0, 27, 22, -17, -32, 92, -81, -27, 83, -59, -44, -39, -102, -73, 55, 52, 88}
, {-80, 69, -48, -43, 88, 99, 64, -15, -110, -45, -65, 19, -19, 67, 74, 28, -47, -55, 17, -76, 46, 65, -46, -28, 14, -43, 72, 128, 24, 18, -62, 27, 82, 66, 51, -77, -41, -15, 58, -11, 7, -79, 99, -16, -87, -80, -28, -30, -30, 69, 78, 14, -13, 46, -92, -22, -69, 7, 25, -53, 50, -17, 106, 80, -78, -83, 73, -66, 82, 57, 82, 38, -12, 15, 92, -88, 33, -37, 71, 12, 94, 58, 67, 14, 27, 3, 48, 52, -63, -15, -78, 39, -39, 46, -63, -85, 100, 68, 41, -114, -32, -119, -2, 1, 61, 114, 17, -59, 84, 44, -48, -110, 83, -100, 73, -55, 58, 52, -73, 64, 92, -40, 12, -60, 49, -8, -58, 39}
, {-16, 39, -75, -25, -63, 77, -34, -101, 92, -15, -6, -82, 61, 72, 63, 101, 82, 70, -70, 10, 67, 56, -100, 109, -25, -80, 72, 24, 6, -124, 63, -5, 35, -87, 96, 41, 0, -3, 28, 48, 39, 46, -96, 52, 46, 59, -64, -75, -11, 99, -30, 104, -18, 44, 47, 48, 13, -70, -6, -109, 14, -14, -113, -43, -40, 15, 86, 74, 43, -37, 45, -29, -18, 7, 33, -68, -100, 39, 70, -83, 75, -7, 2, -27, -16, 49, 33, -79, 18, -18, 17, -84, 57, -44, -12, -58, -96, -7, 76, 63, 51, 104, -54, 18, 59, 63, -70, -21, -93, 67, 58, 57, -93, -85, 66, 62, 109, 57, 7, -85, 22, -44, 28, -1, 57, 30, 4, -43}
, {-23, 30, 23, 28, 65, 7, -19, 69, 83, -60, -85, 99, 89, -66, 60, -60, -27, -38, -2, 16, 2, -104, -22, -89, -66, 63, 12, -22, 66, 120, 99, 74, 35, 11, -32, -23, 77, -9, 50, 23, 7, -24, -87, 66, 67, -23, 36, -76, 81, -44, 88, 33, 32, 80, -62, -38, -16, 0, 109, 56, 49, -23, -4, -94, -95, -53, 55, -39, 92, 6, 112, -44, -40, 41, -76, -87, 4, 6, -16, -67, -4, -37, 29, 98, -30, 83, 84, 26, -7, -35, 53, 49, -40, 39, 40, 57, 42, 72, 85, 35, 11, -103, -56, 66, 74, -17, -76, -8, -11, -12, 5, 63, -80, -37, 47, -70, -61, -53, -76, 55, -15, 91, 11, -29, -18, 47, 84, 17}
, {83, 36, -7, -76, 55, 31, 100, 38, 14, 0, 27, -20, -77, 61, -11, 101, -98, -73, -79, -58, 79, 58, -44, 9, 23, 39, -59, 156, -8, -77, -31, -12, 55, 73, -58, -28, 50, 75, -13, 40, 52, -52, -35, 71, -69, -49, -48, 56, 37, 27, 42, 52, 79, 39, -58, 37, -50, 39, -2, -30, -52, 72, 56, 40, 37, 27, -68, -74, 75, -54, 87, 20, -63, -63, 80, 98, 96, -80, 1, 72, 88, -15, -98, -5, -43, -48, -79, -49, -88, -81, 38, -68, -53, -83, -36, -82, 35, 61, -90, 38, -59, 76, 107, -19, 42, 59, 43, 14, -5, 2, -75, 39, -76, 3, -66, -80, 4, -65, -33, 21, 66, 0, -71, 9, 71, -47, -18, -57}
, {38, 93, 19, 60, 2, 12, -68, 93, -121, 73, 7, 101, 36, -66, 20, -84, 26, 5, 18, 46, -44, 59, 0, -68, -91, -77, 84, 68, 95, -34, -42, -86, -79, -40, -90, -4, -68, 31, -102, -100, 2, -5, -49, -54, -67, 17, -30, 73, 19, 24, -57, -29, -49, -32, 22, -25, -55, -97, 1, 76, 51, -31, -34, -87, -55, -76, 0, 55, 7, -96, 80, 25, 28, -142, 80, -52, 12, -88, 22, -101, 58, -31, 9, -44, 5, 68, -9, 19, 92, -39, -62, 6, 87, 77, -72, 59, 108, -23, -32, 31, 97, -84, 76, -119, 95, -43, 41, -42, 51, 2, 76, -68, 82, -2, -91, 61, 75, -93, -67, 24, 87, 109, -22, -10, -10, 47, -8, -101}
, {-28, -62, 95, 97, 5, 29, 81, 6, 27, -70, 91, 35, 85, 41, -45, 47, 79, -18, 67, 29, 52, -29, 35, 69, 26, -29, 80, 135, -98, 79, 55, 25, 12, -60, -24, 82, -5, 69, -92, 47, 11, -41, 102, -103, 30, 84, -89, 61, 53, -14, -27, -21, -28, 114, 1, -2, 68, 43, 73, -14, 9, 70, 102, -71, 7, -107, 89, -73, 65, 0, -52, -72, -35, -29, -37, 20, -48, 49, -32, -17, -28, 0, -71, 52, -48, 80, -87, 48, 65, -25, 73, -11, 84, -51, -68, 76, -25, -57, -51, -28, 72, -72, 42, 4, 26, 21, 80, 60, -7, -33, -56, 70, 74, -43, 86, -66, -46, 21, -2, 79, -52, -47, 84, 92, -21, 5, -73, -34}
, {19, 22, -93, -81, -47, 91, -88, 29, -25, -113, -42, -2, 66, -20, -98, 89, -33, -27, 45, -105, 47, 74, -74, 91, 16, -24, -20, 14, 41, 70, 46, -60, -50, 53, 1, 38, -9, -13, 83, -12, 42, -31, 23, -50, -92, 32, -3, -38, -93, 86, 78, -63, -5, -71, -9, -2, 4, -38, 19, -67, -54, 21, -3, 7, 79, 25, 69, -42, -63, -175, 82, -21, -6, -18, 67, -44, -26, -95, -48, 28, 87, 67, 29, -11, 53, -18, 106, 34, -104, 23, -30, 86, 93, -21, 11, 90, 19, 5, 29, -69, 59, -54, 101, 22, -33, 47, -68, 69, -62, 15, 55, -35, -93, -12, -26, 42, -38, -79, 44, -5, -74, -50, -50, 54, -78, -59, -24, 34}
, {-124, -26, -31, -1, -88, 33, -41, -7, -90, -47, -81, 44, 73, 30, -47, 0, 45, -83, -70, 7, -48, 6, -60, 31, 64, -72, 1, 69, -75, -60, -20, -72, 37, 87, 30, -30, 16, -36, 72, -17, 76, -9, -9, -134, -54, -3, -99, -35, 6, -52, -73, -61, -67, 106, -17, 50, 37, -11, 81, 30, 83, -78, -24, -83, 54, -30, 16, -91, 25, -59, -64, 15, -35, 46, 97, 39, -4, 52, -48, 17, 48, 19, 60, -28, -96, 79, -42, 81, 18, 40, -43, 17, -91, 56, 0, -50, -26, -48, 20, 24, 75, 80, 2, -18, -1, 94, -53, -69, -25, -27, -7, 52, 75, -45, -69, 9, -59, 93, 48, -43, 70, -36, 41, 74, -50, 14, -69, 16}
, {-76, -55, 13, 31, -51, -49, -32, -15, 42, -96, -15, 97, -62, -56, -29, 7, 16, -65, -73, 5, 22, 87, 58, -24, 87, 51, -45, 128, 95, -75, 42, 46, -91, 66, -43, -16, -67, -33, -49, -61, -10, -27, -38, 21, 47, -22, 11, -61, -16, -77, 34, -28, 2, 93, -99, 83, -93, 58, 65, 14, -2, 60, 112, 78, -74, 12, 84, 40, 12, -110, 61, 68, -32, 4, 27, -54, -37, -55, -34, -80, -82, 81, -19, 92, -34, 21, -91, 91, -111, 9, 69, -98, 79, 68, -18, 29, 0, -77, -50, -110, -28, 3, 22, 64, 84, 47, 64, 82, 100, -24, -53, -69, 100, 76, 26, 16, 42, -90, -37, 89, 2, -49, -114, 14, -37, -81, -43, -86}
, {-62, 93, 41, 25, 21, -33, -1, 103, -50, 42, 46, 87, -68, -35, 27, 92, 85, 100, 12, -35, 13, -42, -1, 33, -30, -46, -60, 133, 42, 3, 5, 63, 97, 71, 57, 85, 37, 39, -71, 24, 39, -42, -36, -10, 93, -63, 1, -76, -36, 96, 14, 20, 29, 65, -66, 73, -78, 88, 40, -67, 40, -29, 38, 84, -61, 41, 23, -24, -39, -65, 16, -40, 3, -93, 47, 67, 66, -60, -96, -28, 54, 4, 77, 43, -82, 86, -56, 76, 20, 93, 79, -81, 22, -22, -59, 74, 0, 27, -75, 73, 90, -106, 86, -81, 68, -3, 59, 59, -14, 61, -54, -84, 33, -37, -59, 19, 11, 29, -78, 89, 42, 0, -18, -18, 41, 11, -78, 97}
, {-46, 71, 90, -55, -93, 3, -7, -55, -53, -54, -17, 68, -29, 94, 55, -47, 26, -82, 14, 36, -32, -15, -34, -59, 35, -97, 86, 38, 25, 73, 97, 63, -24, 7, 31, -67, 45, -34, -57, -74, -102, 57, 22, 51, -79, 17, 45, 76, -18, -82, -54, -78, -8, -29, 15, 85, -32, -93, 100, -70, -47, -42, -42, 33, -98, -55, 63, 11, 13, 102, 75, 72, 55, -34, 59, -70, -45, 6, -75, 92, 55, -83, -55, -8, 93, 60, -81, -71, -43, 78, 66, -60, 5, 76, -53, -33, -37, -66, 28, 97, 27, 26, -90, 40, -26, 91, 67, 80, -66, 52, -15, 34, -12, 28, 59, -19, 12, 15, 34, -67, -6, 105, 26, -78, -16, -91, -82, -122}
, {-87, 50, 14, -53, -31, -7, -66, -16, 95, 92, -18, 28, 13, 47, 66, -73, -45, 1, -67, 21, 84, 1, 6, -6, -40, -86, 83, -74, -71, 67, 6, 8, -31, 32, 26, -51, 74, -20, 18, 59, -91, -59, 43, 59, 49, -63, 59, -21, -23, -41, 2, -88, 85, -76, 81, 76, 78, 19, -89, 66, 31, 15, -74, -11, -25, 20, 2, 20, 10, 132, 0, 87, 22, -37, -68, 71, 54, 9, 87, 79, 8, -61, -41, 47, 52, -57, 37, -28, -34, 87, -94, -78, -38, 28, 95, 75, 22, -80, 8, -77, 26, 48, -50, 16, 92, -10, -70, 16, -96, 28, 15, -9, 82, -67, -80, 48, 26, 45, -52, 15, -35, 5, 81, -18, 39, 70, -72, -4}
, {-89, -35, 37, -65, 59, -58, 62, 39, -42, 53, -6, -52, -85, 17, 4, -19, 53, -107, 38, 28, 63, -15, -74, 39, 30, -78, 25, 33, -31, -79, 77, 0, 27, 76, 59, -80, 64, 76, -25, -42, -55, -83, 58, -82, 7, -60, 53, 8, 21, 6, 77, -29, -7, 91, 35, -63, 86, 1, 81, 35, -35, -55, 35, -72, -50, -67, -33, 43, 85, 108, 43, -47, -40, 6, -85, -16, 70, -27, -68, -77, -17, -8, 45, -52, 87, -50, 29, 28, -26, -61, -28, 9, 82, -61, -60, -42, 65, -36, 54, 18, 31, -36, 24, 64, -51, 1, -66, -3, 49, -63, -50, 102, -9, -52, 78, 3, -2, -32, 47, -58, 46, -97, 30, 83, 29, 91, 59, -118}
, {-2, 20, -46, 81, 11, -37, -48, -8, 38, -38, 71, 98, 63, -45, -62, 66, -68, -28, 58, 26, 78, -90, 26, -63, 9, 69, 12, -16, 23, 98, 3, 84, -90, 44, 17, 7, 9, -19, -35, -18, 57, -59, 68, 80, 65, -35, 50, -39, -81, -62, -20, -73, 29, 84, -7, -76, 34, -33, 137, 83, -42, -8, -57, -78, 23, -52, -42, 8, -70, -37, 41, 5, -17, -77, -93, 78, 30, 42, -80, -7, -5, -53, -64, -80, -17, 3, -44, 48, 108, 47, -47, 14, 51, 52, -93, 18, -66, -33, 97, -88, 62, -74, -47, -56, -67, 63, -40, 83, -47, -61, 87, 6, -32, -13, -50, 32, 40, 58, 20, 100, 75, 127, -48, -45, -79, -27, -30, -74}
, {76, 11, -58, -27, -11, 83, 77, -58, 9, -14, -80, -56, -88, -7, 54, -30, 63, -5, 32, 61, 6, -95, 48, 12, -90, -13, 58, 31, -47, 13, 43, 74, -29, 14, -74, 54, 94, 13, -45, 10, -22, 98, -63, -70, -28, 42, -32, -18, 30, 14, 23, -96, 70, 31, -39, 98, -3, -40, 58, 36, 29, -58, 20, -32, -62, 76, 25, -26, 56, -78, 11, 24, 25, 15, 83, 0, 63, -14, -67, 49, -24, -58, -96, -49, 1, -81, -9, -15, 0, 57, -69, -54, -54, 67, 74, -38, 96, 36, 98, 44, 91, -46, 74, -55, 69, -81, 101, -52, -60, -46, -56, -86, -40, 21, 78, 29, 53, 17, 39, 91, -25, 56, -95, 29, 50, -78, -9, -17}
, {-60, 34, -6, -51, -90, 22, -38, -54, -14, -81, 89, 20, 79, -3, 56, 9, 23, 39, 64, -99, 98, 16, 37, -20, -2, 89, 67, 55, -101, -3, -6, -20, -13, 32, 93, 86, -59, 58, 54, -48, 76, 58, 46, -83, -102, -47, -40, -19, 0, -27, 97, -26, -67, 13, 34, 25, 14, -1, -75, -31, -88, -99, 51, -60, -9, 12, 67, -15, 79, -49, -64, 28, 48, -38, 43, -86, -66, 57, -34, 46, -79, 97, 3, -38, -14, 45, -80, -80, 55, 65, -87, -38, -73, 58, 36, -18, -97, 51, 25, -54, 61, -49, 2, 83, 17, 30, -23, 21, 38, -103, 68, 48, -8, 16, 45, -87, 90, 93, -66, 74, -18, 55, -10, 43, -37, -77, -16, 0}
, {-66, 26, 47, 8, -89, 85, -54, 94, -38, -30, -90, 2, 87, -98, -53, -30, -47, 41, -71, -113, -20, 57, 86, 64, 10, -57, -85, 27, 88, 68, -47, -21, -48, 85, -39, -31, 83, 41, -1, 12, -71, -67, -47, -27, 11, -101, 12, -108, -51, 0, 29, 37, 64, -56, -84, 18, -28, 3, 125, 75, -25, -92, -35, -28, 73, 72, -81, 8, 15, -85, 31, -82, 26, -54, -42, -69, 14, -60, 53, 41, -84, 23, 34, -26, -20, 67, 51, 40, 76, -26, -69, -55, -59, 23, -25, 78, -62, 44, -32, 15, 27, -75, 35, -4, 34, 75, -57, 87, 83, -87, 45, -62, 12, 23, -61, 52, -29, -9, 78, -46, -24, 42, -110, -86, -25, -48, -18, -4}
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

typedef number_t dense_17_output_type[FC_UNITS];

static inline void dense_17(
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


const int16_t dense_17_bias[FC_UNITS] = {13, -18, -9, 7}
;

const int16_t dense_17_kernel[FC_UNITS][INPUT_SAMPLES] = {{-91, -140, 99, 0, 144, 56, 83, -77, -21, -52, 141, -147, -112, 149, 17, 46, -169, -65, -48, 180, -151, 186, 94, -155, 179, 14, -126, -141, -153, -150, -171, -168, 2, 75, 184, 31, 40, 50, 1, -186}
, {-82, 219, 83, 25, -241, 87, -157, 62, 83, -144, -91, -121, 179, 143, -10, -94, -72, -56, -176, -8, -57, -85, -113, 137, 176, -155, 83, -11, 187, -88, 216, -75, -76, -187, 123, 208, -9, 51, 139, -129}
, {56, 133, 100, -15, -79, -56, -189, 80, 97, -1, 131, 195, -157, 106, 162, 78, 38, -101, 117, 21, -4, -112, 179, -64, 115, -39, 126, -22, 16, 123, -168, -74, 128, -95, -82, -155, 34, 67, 156, -88}
, {-15, -30, -150, -129, 63, 101, 141, 25, -15, 154, -64, -123, 120, -84, 7, -165, 0, -149, 170, -134, -158, 17, -132, -8, -135, 53, -21, 112, 103, -109, -86, -53, 140, 201, 138, -42, 121, 132, -50, 67}
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
  //dense_17_output_type dense_17_output);
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
#include "max_pooling1d_40.c" // InputLayer is excluded
#include "conv1d_32.c"
#include "weights/conv1d_32.c" // InputLayer is excluded
#include "max_pooling1d_41.c" // InputLayer is excluded
#include "conv1d_33.c"
#include "weights/conv1d_33.c" // InputLayer is excluded
#include "max_pooling1d_42.c" // InputLayer is excluded
#include "conv1d_34.c"
#include "weights/conv1d_34.c" // InputLayer is excluded
#include "max_pooling1d_43.c" // InputLayer is excluded
#include "conv1d_35.c"
#include "weights/conv1d_35.c" // InputLayer is excluded
#include "max_pooling1d_44.c" // InputLayer is excluded
#include "average_pooling1d_8.c" // InputLayer is excluded
#include "flatten_8.c" // InputLayer is excluded
#include "dense_16.c"
#include "weights/dense_16.c" // InputLayer is excluded
#include "dense_17.c"
#include "weights/dense_17.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_17_output_type dense_17_output) {

  // Output array allocation
  static union {
    max_pooling1d_40_output_type max_pooling1d_40_output;
    max_pooling1d_41_output_type max_pooling1d_41_output;
    max_pooling1d_42_output_type max_pooling1d_42_output;
    max_pooling1d_43_output_type max_pooling1d_43_output;
    max_pooling1d_44_output_type max_pooling1d_44_output;
    dense_16_output_type dense_16_output;
  } activations1;

  static union {
    conv1d_32_output_type conv1d_32_output;
    conv1d_33_output_type conv1d_33_output;
    conv1d_34_output_type conv1d_34_output;
    conv1d_35_output_type conv1d_35_output;
    average_pooling1d_8_output_type average_pooling1d_8_output;
    flatten_8_output_type flatten_8_output;
  } activations2;


  //static union {
//
//    static input_9_output_type input_9_output;
//
//    static max_pooling1d_40_output_type max_pooling1d_40_output;
//
//    static conv1d_32_output_type conv1d_32_output;
//
//    static max_pooling1d_41_output_type max_pooling1d_41_output;
//
//    static conv1d_33_output_type conv1d_33_output;
//
//    static max_pooling1d_42_output_type max_pooling1d_42_output;
//
//    static conv1d_34_output_type conv1d_34_output;
//
//    static max_pooling1d_43_output_type max_pooling1d_43_output;
//
//    static conv1d_35_output_type conv1d_35_output;
//
//    static max_pooling1d_44_output_type max_pooling1d_44_output;
//
//    static average_pooling1d_8_output_type average_pooling1d_8_output;
//
//    static flatten_8_output_type flatten_8_output;
//
//    static dense_16_output_type dense_16_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d_40(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_40_output
  );
 // InputLayer is excluded 
  conv1d_32(
    
    activations1.max_pooling1d_40_output,
    conv1d_32_kernel,
    conv1d_32_bias,
    activations2.conv1d_32_output
  );
 // InputLayer is excluded 
  max_pooling1d_41(
    
    activations2.conv1d_32_output,
    activations1.max_pooling1d_41_output
  );
 // InputLayer is excluded 
  conv1d_33(
    
    activations1.max_pooling1d_41_output,
    conv1d_33_kernel,
    conv1d_33_bias,
    activations2.conv1d_33_output
  );
 // InputLayer is excluded 
  max_pooling1d_42(
    
    activations2.conv1d_33_output,
    activations1.max_pooling1d_42_output
  );
 // InputLayer is excluded 
  conv1d_34(
    
    activations1.max_pooling1d_42_output,
    conv1d_34_kernel,
    conv1d_34_bias,
    activations2.conv1d_34_output
  );
 // InputLayer is excluded 
  max_pooling1d_43(
    
    activations2.conv1d_34_output,
    activations1.max_pooling1d_43_output
  );
 // InputLayer is excluded 
  conv1d_35(
    
    activations1.max_pooling1d_43_output,
    conv1d_35_kernel,
    conv1d_35_bias,
    activations2.conv1d_35_output
  );
 // InputLayer is excluded 
  max_pooling1d_44(
    
    activations2.conv1d_35_output,
    activations1.max_pooling1d_44_output
  );
 // InputLayer is excluded 
  average_pooling1d_8(
    
    activations1.max_pooling1d_44_output,
    activations2.average_pooling1d_8_output
  );
 // InputLayer is excluded 
  flatten_8(
    
    activations2.average_pooling1d_8_output,
    activations2.flatten_8_output
  );
 // InputLayer is excluded 
  dense_16(
    
    activations2.flatten_8_output,
    dense_16_kernel,
    dense_16_bias,
    activations1.dense_16_output
  );
 // InputLayer is excluded 
  dense_17(
    
    activations1.dense_16_output,
    dense_17_kernel,
    dense_17_bias, // Last layer uses output passed as model parameter
    dense_17_output
  );

}
