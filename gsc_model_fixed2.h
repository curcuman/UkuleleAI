#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t max_pooling1d_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t conv1d_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    1
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  10


const int16_t conv1d_bias[CONV_FILTERS] = {3, 5, 5, 0, 5, -3, -3, 9, 4, -8, -11, 5, 5, 0, 1, 12}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{67, 62, -46, 90, -27, -63, 54, -52, -25, -69}
}
, {{40, 65, -9, 78, -62, 79, 75, 53, 28, 28}
}
, {{-93, -92, -62, -87, -10, 7, -4, -66, -33, 88}
}
, {{109, -73, 28, 53, 20, 98, 72, -39, 10, 31}
}
, {{-68, -6, -41, 8, 93, 43, -28, -44, 31, 88}
}
, {{-3, -74, 11, -72, 94, 77, -62, 74, 43, -9}
}
, {{27, 8, -77, 79, 29, 8, 40, 66, -47, 96}
}
, {{47, -82, -90, 55, 80, 60, 46, 19, 12, 39}
}
, {{-87, -75, 55, -27, 29, -4, 4, -65, -12, -88}
}
, {{-73, -36, -47, 71, -80, 82, 76, 3, 90, -7}
}
, {{-53, 49, -59, 44, -63, -97, 89, 68, 100, -58}
}
, {{-26, 47, 87, 7, 20, -17, 15, -102, 29, -91}
}
, {{-80, -95, -62, -31, -79, 5, 12, 55, 86, -72}
}
, {{10, 70, 62, -34, -50, 76, 94, -84, 102, -3}
}
, {{16, 45, 65, -78, -1, -94, 70, -93, -37, 43}
}
, {{45, -21, -74, -8, -89, -58, -67, -77, 0, 48}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t max_pooling1d_1_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_1(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t conv1d_1_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_1(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_1_bias[CONV_FILTERS] = {12, -4, 14, 9, 2, 0, 1, 3, 5, -6, 3, 0, -5, 13, -6, 1, 0, -8, 19, 0, -3, -1, -7, 9, 1, -10, 0, -4, 0, -6, -7, 7}
;

const int16_t conv1d_1_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{75, -16, -52}
, {-23, 19, -38}
, {-68, -9, 58}
, {55, -63, -29}
, {73, 90, 97}
, {57, 86, -57}
, {84, -27, -100}
, {75, 86, -9}
, {51, -80, -85}
, {55, -88, 97}
, {-46, -10, 110}
, {61, 56, -47}
, {-68, 2, 21}
, {-50, -7, -9}
, {-55, 73, 37}
, {-86, -52, -22}
}
, {{80, 29, 0}
, {60, -86, -37}
, {-54, 59, -3}
, {63, 27, 10}
, {-64, -106, 94}
, {30, -46, -73}
, {51, -102, 50}
, {-90, -59, -68}
, {-70, 14, 71}
, {9, 87, -82}
, {35, 10, 109}
, {49, -79, 21}
, {74, -82, 16}
, {-38, -23, 36}
, {-37, -32, -50}
, {-24, 13, 46}
}
, {{-55, -50, 106}
, {-84, 6, 110}
, {45, -86, 80}
, {74, 32, 25}
, {-6, -4, 41}
, {-72, -77, 64}
, {-85, -69, 51}
, {-104, -85, 6}
, {109, -50, 56}
, {21, 44, 23}
, {-96, 9, 1}
, {69, 75, -10}
, {5, 59, 53}
, {23, 56, 89}
, {54, 25, -6}
, {67, 103, 96}
}
, {{60, -85, 101}
, {-83, -52, -54}
, {37, 12, 57}
, {-66, 43, -85}
, {74, 44, 105}
, {-32, -33, 69}
, {45, 72, -50}
, {-73, 37, 87}
, {-46, 89, -84}
, {25, -63, -54}
, {-47, -13, 42}
, {14, 46, -64}
, {29, 27, 62}
, {73, 109, 76}
, {30, -69, -15}
, {4, 95, -57}
}
, {{66, 31, -7}
, {-69, 27, 66}
, {6, 4, -60}
, {13, -97, -70}
, {-46, -40, -34}
, {-73, 68, 92}
, {-100, 76, 68}
, {98, 57, -40}
, {103, -57, 11}
, {-53, 105, -92}
, {101, 77, -100}
, {71, -38, 114}
, {-97, 46, 54}
, {-42, 38, -102}
, {10, -10, -70}
, {-10, -44, -12}
}
, {{103, 7, -39}
, {46, -96, 78}
, {11, -56, 56}
, {64, -67, 14}
, {-1, -52, -33}
, {-86, 78, -106}
, {43, 81, -31}
, {1, -43, 84}
, {-54, -47, -80}
, {89, 8, -33}
, {29, 95, -16}
, {3, 92, 1}
, {0, 28, -5}
, {45, -31, 99}
, {-26, 25, -77}
, {64, -16, 60}
}
, {{14, 32, 17}
, {52, -50, 61}
, {-11, 7, -25}
, {-77, 86, 74}
, {88, -91, -10}
, {65, 98, 28}
, {-30, -34, -82}
, {32, 56, -30}
, {-11, 94, -100}
, {74, 83, -91}
, {7, 22, 51}
, {0, -56, 97}
, {-90, 16, 79}
, {-84, -85, -58}
, {52, 97, 103}
, {-9, -84, 24}
}
, {{14, -77, 59}
, {20, -32, -74}
, {-16, -86, 60}
, {22, -83, 51}
, {102, 23, -78}
, {-67, -91, -44}
, {87, 65, -62}
, {102, 5, 49}
, {-29, 33, -28}
, {-34, 17, -30}
, {-35, -52, -63}
, {-73, -54, 83}
, {-85, -17, 43}
, {-37, 94, 33}
, {69, -103, -95}
, {72, 92, 73}
}
, {{94, 14, 15}
, {-13, -31, 64}
, {-25, -29, -62}
, {-67, 27, -19}
, {-94, -4, 76}
, {-74, 84, -73}
, {-85, -64, -29}
, {19, 51, 67}
, {70, 5, 28}
, {82, 98, -42}
, {-80, 50, -45}
, {59, 21, 20}
, {-30, 17, -81}
, {-33, 75, 96}
, {-85, -19, -33}
, {-73, -61, 21}
}
, {{43, 28, 61}
, {-44, 6, -56}
, {-78, 21, -39}
, {91, 48, -20}
, {53, 83, -66}
, {67, 92, 90}
, {89, -94, -64}
, {69, -55, -39}
, {36, 47, -14}
, {34, 75, 59}
, {7, -97, -8}
, {-7, -82, 81}
, {-27, -10, -85}
, {31, -27, -7}
, {-96, -25, 80}
, {-52, 91, -71}
}
, {{-86, -57, 41}
, {46, 78, 13}
, {-55, -97, -52}
, {33, -61, 44}
, {-56, 79, -45}
, {27, -17, -54}
, {-52, -44, -68}
, {-81, -74, -26}
, {38, -51, 90}
, {101, -78, -30}
, {99, -84, -65}
, {-19, 55, -99}
, {-1, -73, 51}
, {-54, 5, -66}
, {-28, 32, 52}
, {3, -9, 6}
}
, {{10, 33, -34}
, {14, -1, 108}
, {-83, 71, 74}
, {-78, 42, 75}
, {70, -29, 19}
, {-92, 17, -30}
, {63, 31, -2}
, {105, -10, 2}
, {77, -102, -49}
, {-15, -37, -40}
, {-90, 93, -58}
, {-16, -4, 41}
, {55, 9, 46}
, {-78, 26, 74}
, {30, -66, 46}
, {15, -85, -53}
}
, {{2, 21, 52}
, {3, -12, 5}
, {-47, -28, 73}
, {85, 14, -70}
, {-14, 49, 68}
, {67, -75, 82}
, {27, 23, 47}
, {88, 102, -3}
, {-104, -100, -37}
, {-6, 107, 93}
, {-19, 34, 19}
, {-4, -102, 53}
, {-94, 68, 22}
, {97, 74, 32}
, {-45, 20, 17}
, {-83, 65, 11}
}
, {{62, -17, -104}
, {103, 7, 62}
, {-44, -21, -30}
, {23, 17, -52}
, {5, 84, -51}
, {32, 0, 45}
, {-28, -45, -39}
, {104, 64, -18}
, {-51, -73, 57}
, {77, 3, -21}
, {12, 45, 85}
, {-21, -10, -21}
, {-66, 81, -37}
, {94, -17, 22}
, {8, -64, -74}
, {-25, -23, -20}
}
, {{-16, -95, 22}
, {-13, -7, -60}
, {0, 7, 93}
, {30, 73, -43}
, {40, 8, 108}
, {83, -98, -68}
, {-83, -98, 20}
, {-23, 58, 0}
, {-49, 45, 84}
, {19, 23, 31}
, {65, 101, -12}
, {-53, -73, -92}
, {21, 89, 54}
, {0, 104, -87}
, {-70, 22, 37}
, {-19, -51, -15}
}
, {{-34, 41, 92}
, {69, 52, -57}
, {52, 0, -81}
, {-88, -102, 66}
, {59, 76, 103}
, {-75, -55, 10}
, {-76, 64, -50}
, {51, -71, -36}
, {-47, -104, 33}
, {-84, 104, 4}
, {32, -55, -9}
, {-72, -90, 29}
, {-50, 21, 54}
, {-80, 97, 83}
, {-59, -8, 20}
, {-98, 56, -23}
}
, {{7, 19, 95}
, {91, -26, 53}
, {54, 100, -89}
, {-73, -29, 4}
, {-67, 33, 75}
, {76, -106, -30}
, {106, -100, 68}
, {-55, -30, 84}
, {49, -40, 54}
, {-37, 85, -30}
, {97, 95, 24}
, {-39, 78, 49}
, {-89, -88, -111}
, {-53, -45, 49}
, {-45, -45, 85}
, {36, -33, -96}
}
, {{2, 43, -23}
, {10, 50, -98}
, {84, -100, 3}
, {94, 19, 40}
, {71, 69, 62}
, {105, -13, 99}
, {-44, 28, 53}
, {44, -53, 44}
, {-56, 33, 21}
, {41, 23, -10}
, {105, 58, -17}
, {82, 87, -29}
, {-26, 85, 71}
, {87, -56, 40}
, {-91, -25, -74}
, {-83, 31, -67}
}
, {{-49, 42, 61}
, {70, 26, -47}
, {-34, 4, -49}
, {66, 101, -16}
, {7, -34, 0}
, {-40, 18, 36}
, {-77, -46, 0}
, {47, 92, -74}
, {-45, 6, -71}
, {-18, -75, -11}
, {-61, 114, 16}
, {-12, 22, -52}
, {-19, 82, -89}
, {-17, -33, 12}
, {-95, -53, -43}
, {-99, -12, -9}
}
, {{-86, 13, -95}
, {56, 10, 53}
, {-22, -31, 53}
, {-46, -101, -33}
, {55, 78, -34}
, {-41, 47, -41}
, {-51, -4, 91}
, {75, -78, -18}
, {-56, 29, -94}
, {-41, 75, -18}
, {-82, 86, -26}
, {4, 1, 108}
, {80, 80, -99}
, {-84, 75, 68}
, {40, -89, -52}
, {-37, 21, -23}
}
, {{-14, -16, -34}
, {-8, -70, 20}
, {-11, -102, 94}
, {4, -49, 6}
, {91, -105, -1}
, {-32, -99, 35}
, {-104, 8, 15}
, {-7, 14, 3}
, {-9, -116, 93}
, {14, -9, -93}
, {-21, -49, 99}
, {68, 23, -76}
, {22, 91, -58}
, {-104, -55, 20}
, {-67, -54, -1}
, {81, -65, 56}
}
, {{-60, -93, 46}
, {32, 69, 45}
, {96, -3, -108}
, {53, 4, 29}
, {59, -6, 32}
, {6, 73, 47}
, {-35, -31, 37}
, {-1, 101, 36}
, {-25, -88, -32}
, {71, 30, -3}
, {12, 85, -39}
, {-53, 44, -8}
, {-47, -96, -31}
, {-83, 58, -31}
, {91, -53, 79}
, {38, 19, -10}
}
, {{-38, -56, 59}
, {-42, -16, 49}
, {88, -72, 3}
, {-68, -61, -31}
, {93, -72, 28}
, {31, -81, -3}
, {-69, 65, 69}
, {49, -31, -93}
, {-45, -25, 15}
, {-71, -107, 71}
, {69, -82, -34}
, {3, 7, -18}
, {1, 31, -23}
, {-101, -29, -45}
, {-77, 39, -107}
, {-44, -109, -66}
}
, {{-56, 34, -59}
, {-90, -3, 79}
, {33, -97, -77}
, {26, -101, -90}
, {4, -36, 44}
, {29, 104, 79}
, {11, -28, 22}
, {-36, 87, 104}
, {5, 92, -2}
, {0, -90, -83}
, {-12, -60, -77}
, {39, 12, 26}
, {105, 12, 27}
, {-42, -7, -68}
, {-47, -17, 66}
, {79, 30, -8}
}
, {{-26, 89, -28}
, {80, 65, -65}
, {56, -7, -38}
, {-56, 70, 13}
, {-97, -88, 39}
, {-98, -56, 22}
, {11, 59, -98}
, {-41, 53, 93}
, {-103, -18, 18}
, {65, -36, -49}
, {24, 96, -77}
, {7, 69, 44}
, {-69, -74, -27}
, {-63, -86, 30}
, {-32, 5, 48}
, {-6, 61, 95}
}
, {{-59, -90, -15}
, {72, -7, -59}
, {24, 15, -106}
, {-91, -55, -32}
, {93, -72, 38}
, {-35, 82, 70}
, {105, 3, 3}
, {38, 29, 4}
, {13, -7, 88}
, {-53, 39, -48}
, {1, 85, 91}
, {6, 39, -34}
, {18, 89, -23}
, {68, -80, 51}
, {98, 59, 52}
, {52, -5, 8}
}
, {{18, -15, -12}
, {92, -27, -73}
, {-47, -59, 85}
, {45, -82, 103}
, {-4, 85, -29}
, {25, 102, 41}
, {94, 20, 75}
, {-93, 20, -17}
, {-19, 103, -50}
, {16, 98, -100}
, {-38, 98, -19}
, {-82, -82, -17}
, {73, -69, -22}
, {-67, 61, -72}
, {66, 35, -8}
, {34, 80, -6}
}
, {{-65, -95, 5}
, {99, -36, 16}
, {-75, -56, -2}
, {-95, -45, 50}
, {9, -58, 80}
, {-81, -39, -46}
, {77, 4, -68}
, {50, 101, -43}
, {8, -97, -71}
, {-26, -25, -69}
, {54, 63, 66}
, {91, 77, 1}
, {-7, -6, -85}
, {51, -67, 7}
, {-83, 67, 58}
, {-92, -65, -18}
}
, {{-48, 9, 103}
, {-28, 4, 34}
, {8, 73, 86}
, {-38, 87, 0}
, {18, 31, 48}
, {-22, 19, 88}
, {-55, -23, -69}
, {-15, -7, 60}
, {-73, -84, -37}
, {-49, -33, 80}
, {-25, 102, 27}
, {42, -16, 49}
, {17, 84, -80}
, {6, 58, -99}
, {-7, -15, 20}
, {-33, 72, -102}
}
, {{-60, -66, 52}
, {95, -39, 72}
, {1, 18, -101}
, {-74, 83, -41}
, {-22, 3, -40}
, {-75, 70, -11}
, {1, -102, 82}
, {42, 16, -47}
, {54, -95, -43}
, {-34, 102, -54}
, {85, 9, 17}
, {24, -59, 46}
, {32, 104, 41}
, {-24, -20, -78}
, {-16, -108, 105}
, {2, 8, -14}
}
, {{102, 40, 26}
, {24, -47, 25}
, {-87, 64, -60}
, {-54, 93, -80}
, {-64, 69, -42}
, {-43, -88, 19}
, {-91, 46, -45}
, {27, -50, -45}
, {84, 86, 59}
, {23, 11, 41}
, {-101, 29, 23}
, {-46, -62, -4}
, {10, 54, 19}
, {36, 85, 90}
, {8, -77, 9}
, {7, 87, -106}
}
, {{94, 11, -62}
, {112, -24, -30}
, {-15, -52, 71}
, {-40, 7, -26}
, {84, -2, -20}
, {53, -26, 113}
, {32, -43, -75}
, {11, -10, -53}
, {2, -86, -12}
, {-77, -88, 49}
, {-51, -64, 27}
, {63, 17, -7}
, {52, 39, -26}
, {0, -80, -47}
, {59, -47, 110}
, {75, 73, -79}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t max_pooling1d_2_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_2(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t conv1d_2_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_2(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_2_bias[CONV_FILTERS] = {13, 8, 6, -11, -5, -11, 5, -1, -5, 1, -1, -8, -6, 18, 15, 8, 17, 1, -2, 11, 5, 7, 9, 1, -2, 1, 6, -7, 0, -8, 5, 2, 4, 0, 0, -2, -9, -8, -5, 0, 7, -5, 3, 0, -1, -7, 0, 7, -2, 17, -5, 6, 18, -6, 10, 1, 6, 20, -2, 13, 9, 5, 7, -1}
;

const int16_t conv1d_2_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{44, 8, 3}
, {-49, -69, -31}
, {62, 73, 21}
, {74, 28, 54}
, {18, -43, -29}
, {-60, -64, 4}
, {48, -14, 0}
, {-65, 50, -28}
, {38, 7, -63}
, {-28, -51, 0}
, {52, 0, 32}
, {-30, -22, 26}
, {-62, 74, 39}
, {70, -53, 8}
, {8, -29, 47}
, {54, -28, -46}
, {-51, 40, -13}
, {31, -55, 66}
, {81, 66, 55}
, {-58, 42, 47}
, {-1, -54, 85}
, {-9, -2, 25}
, {59, -53, -33}
, {-42, 79, 45}
, {19, 32, 31}
, {-70, -48, -59}
, {-38, -35, -24}
, {-1, 0, -9}
, {-62, -37, -51}
, {-57, 9, 19}
, {69, -3, -36}
, {16, -52, 9}
}
, {{-24, 39, 36}
, {63, 44, 19}
, {50, 71, 1}
, {22, -11, 20}
, {-42, 48, -55}
, {-48, -46, 20}
, {7, 41, 59}
, {-30, -35, 34}
, {-34, -41, 34}
, {55, 16, 7}
, {17, 53, 13}
, {-76, -65, -64}
, {23, -50, -36}
, {-17, 75, 71}
, {2, 43, -52}
, {46, 76, 44}
, {-1, 63, 42}
, {0, -4, 43}
, {-1, 54, -44}
, {-34, -62, 34}
, {69, -15, 19}
, {33, 19, 25}
, {-80, -37, -59}
, {-26, 36, -40}
, {50, 11, 1}
, {-18, -19, 10}
, {-43, 14, 43}
, {37, -34, 72}
, {-2, 71, -51}
, {30, 3, 61}
, {22, -15, 57}
, {21, 71, -54}
}
, {{-66, -42, -39}
, {53, -65, -59}
, {42, 8, 10}
, {0, 70, -43}
, {-10, -48, -57}
, {41, -17, 62}
, {39, -14, 0}
, {29, -16, 71}
, {26, 57, 4}
, {64, -62, -10}
, {0, -12, -39}
, {-47, -59, -71}
, {-76, -44, 71}
, {8, -68, 3}
, {66, -37, 44}
, {-71, 43, 13}
, {75, 80, -62}
, {60, -2, -32}
, {53, 34, 11}
, {-13, 67, 63}
, {27, 35, -33}
, {48, -40, 40}
, {-23, 11, 21}
, {-45, -37, 51}
, {42, 4, 0}
, {48, 73, -62}
, {8, 35, 14}
, {-14, 66, 88}
, {14, -56, -42}
, {18, -5, 33}
, {-73, -2, 8}
, {-62, 21, 9}
}
, {{-1, -7, -63}
, {-71, -46, 15}
, {-15, -55, -18}
, {-12, 59, -1}
, {-60, 56, 7}
, {32, 35, 33}
, {62, -26, 27}
, {10, -42, 40}
, {-40, 52, -54}
, {28, 81, 28}
, {36, 14, 0}
, {20, -27, 51}
, {69, 22, 26}
, {-74, -3, 4}
, {-55, 79, 15}
, {-83, -57, -26}
, {-78, 55, 78}
, {46, -32, 14}
, {67, 3, 60}
, {-74, 12, 65}
, {46, -16, -54}
, {-51, 26, 19}
, {-48, -32, 12}
, {24, -21, -39}
, {53, -32, -59}
, {-23, -44, -15}
, {1, -56, 2}
, {-47, -46, 71}
, {-34, -3, -45}
, {55, 1, 57}
, {-19, 60, -19}
, {51, 24, -9}
}
, {{-21, 27, -31}
, {-55, -19, -59}
, {58, 31, -70}
, {-3, 27, 61}
, {-52, -36, 15}
, {-41, -47, -8}
, {54, 11, -49}
, {-17, -29, 20}
, {-49, -80, -72}
, {0, 58, -65}
, {34, -3, -63}
, {10, 67, -39}
, {57, 67, -45}
, {-51, -39, -5}
, {-27, 30, -33}
, {-50, -43, -60}
, {-69, -32, 45}
, {37, 23, 59}
, {-11, -50, -58}
, {-70, 48, -68}
, {-21, -83, -48}
, {73, 23, -8}
, {54, 3, 44}
, {23, -65, -48}
, {-39, 28, -60}
, {21, 13, -31}
, {62, 29, -71}
, {45, 56, 26}
, {55, 76, -36}
, {-31, 56, -26}
, {59, -7, 48}
, {37, -7, 6}
}
, {{-27, 73, 60}
, {-17, 4, 53}
, {64, 44, -16}
, {-5, -15, 39}
, {5, 11, -41}
, {51, -59, -13}
, {-17, 51, 69}
, {35, -13, 4}
, {72, -30, -14}
, {-54, -72, 45}
, {0, 52, 52}
, {55, -45, 49}
, {-65, -18, 0}
, {60, -57, 60}
, {14, 64, 57}
, {0, -28, 47}
, {-30, -69, 64}
, {45, -49, 7}
, {38, 20, -37}
, {52, 3, -35}
, {61, -55, -16}
, {4, -37, 43}
, {49, -53, 2}
, {-32, -22, -6}
, {50, 22, 59}
, {-15, 61, -36}
, {54, -31, -5}
, {-39, -39, 45}
, {-27, 59, 39}
, {-19, -34, 50}
, {71, -3, -40}
, {41, 55, 10}
}
, {{-62, -9, -12}
, {55, 48, 26}
, {-7, 65, -52}
, {-66, -1, -12}
, {52, 76, 65}
, {-31, -16, 53}
, {-49, 60, 70}
, {65, 61, 13}
, {72, -48, 15}
, {28, 33, 22}
, {-35, 33, -64}
, {46, 11, -29}
, {-75, 25, -28}
, {47, 39, -25}
, {42, 20, 24}
, {-53, -38, 20}
, {13, -21, -45}
, {9, 14, -50}
, {37, 79, 4}
, {-48, 45, 61}
, {-30, -38, 54}
, {-54, 62, -23}
, {76, -2, 23}
, {1, 13, -13}
, {41, 70, 13}
, {-34, -16, 18}
, {-43, -3, 27}
, {34, 11, 63}
, {23, 30, 14}
, {62, 28, -32}
, {-25, -55, -57}
, {5, 55, -4}
}
, {{-30, 21, -48}
, {51, 57, 39}
, {-20, 16, 65}
, {50, -77, -73}
, {37, -29, 39}
, {-27, -16, 67}
, {-68, -13, -68}
, {36, 36, -17}
, {-30, 42, 37}
, {8, 1, 23}
, {6, 3, -18}
, {3, 3, 63}
, {15, 18, -41}
, {67, 67, -63}
, {-63, 49, 29}
, {23, 17, -43}
, {78, 15, -37}
, {31, -70, 3}
, {-34, 18, 38}
, {-5, 76, 79}
, {-47, -75, 1}
, {42, -62, -14}
, {33, 28, 61}
, {0, -71, -39}
, {20, 42, -68}
, {79, 6, 54}
, {-41, 0, -4}
, {-26, 1, -54}
, {32, -71, -61}
, {30, -42, 25}
, {-4, -2, 44}
, {52, -63, -49}
}
, {{-40, 18, -65}
, {-58, -31, 36}
, {52, -48, -66}
, {-23, 27, 54}
, {-53, 64, -42}
, {-8, 16, 26}
, {-20, -28, 45}
, {11, 9, 39}
, {50, -39, 28}
, {56, -48, 15}
, {49, -50, 27}
, {2, -54, 68}
, {-37, 57, 23}
, {47, -34, -57}
, {-28, -47, 15}
, {22, 6, 42}
, {-33, 32, -1}
, {0, -41, -50}
, {-28, 8, 16}
, {7, 9, 22}
, {52, 38, 49}
, {-44, 0, -48}
, {-68, 13, -18}
, {-71, -40, -71}
, {-45, -7, 36}
, {-19, 17, 45}
, {-71, -64, -31}
, {-36, 61, -62}
, {-41, -33, -36}
, {-77, 49, -69}
, {-17, 52, 52}
, {-23, 60, -18}
}
, {{-19, -12, -28}
, {50, 80, -5}
, {27, -11, -55}
, {-14, 65, 0}
, {60, 76, 19}
, {-52, 55, 1}
, {45, 29, 6}
, {36, 27, 28}
, {-53, 8, 72}
, {9, 32, -63}
, {60, 17, 21}
, {-31, 9, 22}
, {70, -33, 15}
, {66, -71, -48}
, {-2, -58, 11}
, {17, -37, 36}
, {58, 0, -28}
, {47, -35, -55}
, {-40, 49, 88}
, {-40, 2, 62}
, {-45, 57, 61}
, {8, -37, 35}
, {24, 26, -20}
, {30, -26, 61}
, {13, -2, -29}
, {-68, 55, 47}
, {62, 26, -23}
, {1, 18, -60}
, {-55, 37, -14}
, {44, 38, 70}
, {8, 32, 41}
, {45, -20, 80}
}
, {{-31, -56, -51}
, {-32, 6, -78}
, {62, 13, 41}
, {30, -77, 71}
, {-79, -22, 54}
, {-29, 49, 29}
, {5, 0, 55}
, {-10, 1, 40}
, {-15, 0, -29}
, {-49, 21, 9}
, {21, 36, -24}
, {-41, -33, 15}
, {61, 32, 68}
, {-49, 74, 14}
, {53, -78, 24}
, {-36, 12, 67}
, {60, 15, -60}
, {8, -57, -3}
, {31, -55, -62}
, {2, -65, 35}
, {-34, -36, 31}
, {-52, -67, 14}
, {61, 21, -47}
, {-44, -19, 42}
, {4, -11, -3}
, {-6, -4, 35}
, {-7, 25, -57}
, {-37, 32, -25}
, {11, -72, -43}
, {-41, -33, -39}
, {39, 26, -78}
, {-73, 1, -43}
}
, {{-69, -61, 18}
, {-38, 28, 40}
, {26, -45, -25}
, {80, -26, 27}
, {41, 50, -20}
, {34, -30, 48}
, {60, 78, 5}
, {-9, 52, -66}
, {-2, 23, 30}
, {43, 27, 56}
, {5, -13, -31}
, {-30, 31, 22}
, {28, 62, -6}
, {7, 15, 15}
, {59, -49, 45}
, {70, -7, -81}
, {-55, -29, 53}
, {-41, 9, -73}
, {18, 46, -19}
, {37, -42, -44}
, {52, -36, -70}
, {68, 67, 15}
, {36, 54, 47}
, {40, 48, 10}
, {47, -70, -16}
, {0, 17, -21}
, {58, -65, 23}
, {51, 51, -26}
, {18, 74, -72}
, {2, -59, 17}
, {-5, 4, 55}
, {-5, 21, 65}
}
, {{-55, 33, 55}
, {14, 47, 39}
, {-1, 1, -35}
, {-18, -17, 30}
, {-15, -26, -11}
, {58, -37, 48}
, {9, -4, 13}
, {55, -22, -36}
, {37, 59, -37}
, {9, 32, 65}
, {-14, -20, 44}
, {-51, 13, -13}
, {72, -47, -31}
, {36, -36, -39}
, {49, 49, 59}
, {1, 41, -41}
, {81, -51, -2}
, {-47, 49, -63}
, {-48, 21, -6}
, {16, 64, 67}
, {10, 45, 48}
, {50, -42, -34}
, {62, -28, 38}
, {58, 69, -16}
, {-11, -54, 22}
, {36, 46, -4}
, {37, -14, 64}
, {6, -36, -55}
, {63, -69, -12}
, {59, 53, -10}
, {78, -25, 69}
, {-17, -27, -18}
}
, {{51, -32, 68}
, {-41, 29, -1}
, {-62, -36, 46}
, {16, 61, 41}
, {36, 32, -68}
, {-23, -56, -73}
, {2, -51, 18}
, {36, 55, 26}
, {52, 16, 36}
, {-41, 75, -38}
, {46, -8, 31}
, {29, 0, 12}
, {9, -42, -43}
, {-61, -43, 71}
, {-24, -58, 44}
, {-33, -77, 0}
, {-10, -19, 59}
, {40, 53, -34}
, {-45, 43, 48}
, {8, 21, 9}
, {52, -47, -30}
, {20, 57, 81}
, {14, -40, 31}
, {47, 34, 4}
, {-5, -66, -45}
, {6, -56, -15}
, {-34, -41, 23}
, {0, -17, -53}
, {-6, 39, 39}
, {-41, -2, -56}
, {77, 29, 84}
, {-50, 41, -76}
}
, {{7, 27, 41}
, {-63, 55, -28}
, {30, 62, -65}
, {-27, -8, 30}
, {-21, -26, 45}
, {-74, 56, -17}
, {14, 69, -23}
, {-24, 26, -18}
, {31, 0, 1}
, {-52, 37, 73}
, {-4, -32, 2}
, {0, 72, 31}
, {-59, 72, -5}
, {45, -32, 31}
, {-28, -20, -7}
, {33, 22, -25}
, {55, 53, -24}
, {-71, -60, 53}
, {90, -50, -9}
, {54, 43, -32}
, {15, 54, -15}
, {43, 0, 34}
, {25, -3, -70}
, {-59, -13, 75}
, {-27, -2, -60}
, {-51, -63, 38}
, {30, -57, -35}
, {-61, 64, -15}
, {-24, -47, -49}
, {5, 9, -14}
, {40, -19, -10}
, {33, 10, -14}
}
, {{27, -54, 40}
, {48, -12, -60}
, {-66, 16, -5}
, {52, -25, -16}
, {45, 5, -72}
, {60, 15, -40}
, {81, -29, -46}
, {21, -16, 2}
, {-20, -17, 3}
, {-38, 12, -7}
, {25, 41, -4}
, {62, -64, 53}
, {-10, 3, 15}
, {12, -5, -35}
, {-3, 18, -49}
, {87, 13, -27}
, {-6, -2, -18}
, {36, 24, -14}
, {44, -18, 1}
, {-44, 19, 23}
, {47, -45, 82}
, {18, 41, -78}
, {27, -6, 35}
, {26, -15, 21}
, {-53, 39, -70}
, {24, 72, -44}
, {-7, 51, -47}
, {24, 55, -65}
, {13, -39, 2}
, {39, -47, -46}
, {4, -47, -34}
, {70, 1, 43}
}
, {{-18, -70, 42}
, {-25, 40, 35}
, {72, 80, -64}
, {4, -45, 8}
, {42, 33, -27}
, {1, 53, 60}
, {-31, 17, 73}
, {-20, 18, -18}
, {-26, -18, -2}
, {22, -62, -37}
, {0, 35, -52}
, {34, -56, 71}
, {-62, -36, 81}
, {0, 62, -66}
, {-12, -5, -55}
, {-31, 15, 91}
, {-45, -59, -15}
, {-69, 0, -23}
, {-25, -8, -12}
, {-68, -18, -74}
, {-40, 30, 50}
, {-39, 11, -14}
, {-8, 48, -64}
, {58, -66, 76}
, {37, 82, -28}
, {-12, -17, 22}
, {50, -64, 6}
, {7, -50, 46}
, {22, -2, 54}
, {-52, -9, 40}
, {54, 0, 24}
, {30, 56, 18}
}
, {{-54, 67, -11}
, {-29, 21, -43}
, {59, -40, 4}
, {54, 12, -62}
, {-34, 24, -38}
, {-74, 22, 2}
, {33, -49, 7}
, {-53, 40, 28}
, {18, -67, -47}
, {0, 44, -3}
, {-32, 7, 30}
, {49, 18, 7}
, {0, -43, 79}
, {-31, 23, 82}
, {-13, 77, 9}
, {-36, -3, -67}
, {-62, -40, 63}
, {-50, 16, 35}
, {83, 70, 38}
, {-52, 7, 29}
, {-38, 71, 28}
, {38, 51, 6}
, {51, 57, -5}
, {55, -56, -39}
, {-21, -21, -82}
, {17, 41, -18}
, {-9, -6, 49}
, {-36, 5, -20}
, {-17, 33, -46}
, {50, -54, -4}
, {-23, -11, 15}
, {-79, 2, -28}
}
, {{52, -61, 67}
, {-64, 46, 26}
, {-44, -34, 41}
, {-36, 64, 17}
, {26, 35, -57}
, {17, 22, 42}
, {-15, 56, 32}
, {-71, 25, -24}
, {56, -19, 48}
, {33, -13, -36}
, {-12, 33, -11}
, {55, -45, -61}
, {-52, 50, -8}
, {-12, -44, -15}
, {-37, -58, -27}
, {-32, 58, -63}
, {-20, -2, 57}
, {20, -53, -58}
, {-21, -33, 15}
, {0, -52, 22}
, {-29, -63, 67}
, {46, 68, 69}
, {-20, -24, 64}
, {40, 66, 66}
, {-23, -67, 10}
, {-57, 19, 64}
, {-3, -38, -4}
, {0, -21, 17}
, {0, -74, 3}
, {-49, 30, 0}
, {56, 9, 38}
, {28, -25, -65}
}
, {{22, 23, 20}
, {-46, -22, 61}
, {6, 25, 22}
, {6, 78, 80}
, {7, 45, -67}
, {7, -70, -19}
, {-63, 17, -54}
, {-7, 20, -22}
, {0, -10, -36}
, {22, 70, -1}
, {-43, -51, 57}
, {25, 61, -67}
, {52, -69, -32}
, {4, -43, -22}
, {-22, -40, -64}
, {4, 40, 1}
, {-37, -2, -2}
, {74, 32, -57}
, {58, 33, -10}
, {74, -36, -4}
, {-89, -2, 38}
, {-26, -45, 13}
, {17, 9, -9}
, {24, 34, -38}
, {-69, 41, -4}
, {23, -31, -2}
, {-14, -61, 59}
, {-4, 25, -52}
, {-63, 67, -24}
, {-33, -2, -2}
, {-59, 17, -36}
, {-11, 0, -66}
}
, {{58, -28, -2}
, {0, -31, -24}
, {46, -67, -65}
, {0, -1, 69}
, {56, 80, -46}
, {-35, 16, -33}
, {-33, -2, -47}
, {-40, 12, -54}
, {-16, -51, 18}
, {84, 78, 13}
, {3, 51, -22}
, {27, -43, -33}
, {30, 34, -40}
, {-29, -18, -63}
, {6, -10, -51}
, {-56, -47, 49}
, {-36, 78, 36}
, {6, 13, 14}
, {-51, -46, 23}
, {-36, -4, 43}
, {-33, -21, 58}
, {19, 10, -54}
, {13, 75, 31}
, {50, -12, 8}
, {51, 34, -35}
, {35, -41, 34}
, {-53, 26, -5}
, {-67, 48, 53}
, {-41, 41, 79}
, {20, 13, 44}
, {25, 79, 43}
, {12, 48, 68}
}
, {{20, -3, -19}
, {-46, 2, 9}
, {48, 25, 13}
, {-55, -14, -40}
, {38, -76, -53}
, {60, 64, -4}
, {-44, 37, 15}
, {8, -56, 51}
, {-50, -1, -77}
, {66, -45, -52}
, {-53, 60, -55}
, {49, -22, 15}
, {-14, 80, 24}
, {-50, 65, -26}
, {-5, -48, 27}
, {13, 23, -32}
, {-24, 64, -35}
, {-16, -14, 17}
, {-46, 76, 57}
, {61, 2, 3}
, {-7, 58, -35}
, {-57, -54, 26}
, {-19, 30, 41}
, {-65, -77, -60}
, {-70, 59, -39}
, {-9, 34, -52}
, {-66, 74, -14}
, {-5, 15, 2}
, {37, -6, -22}
, {-35, 49, -75}
, {-34, 22, -6}
, {30, 22, -3}
}
, {{-8, -8, 1}
, {-54, 46, 44}
, {-42, 50, -43}
, {-29, -46, -59}
, {-7, -36, 33}
, {57, 1, 9}
, {32, 66, 61}
, {-56, 65, -3}
, {80, 23, 67}
, {69, 64, 48}
, {-72, -14, -4}
, {-36, -49, 17}
, {28, 13, -51}
, {69, 5, 31}
, {14, 31, 11}
, {60, -56, 21}
, {-34, -49, -6}
, {-54, 67, 2}
, {66, 55, 35}
, {-21, -15, 70}
, {46, -51, -10}
, {68, -71, -13}
, {-42, 59, -8}
, {22, -50, 45}
, {10, -58, -32}
, {64, -64, 70}
, {38, -35, -45}
, {69, 45, 60}
, {46, 2, 29}
, {-17, -75, -52}
, {28, -77, 72}
, {-17, -44, -38}
}
, {{45, 36, 30}
, {65, 79, 57}
, {66, -36, 28}
, {-12, -63, 34}
, {43, 29, 22}
, {21, 74, -36}
, {19, -75, 37}
, {59, -25, 11}
, {-49, 18, 11}
, {6, -39, -4}
, {-2, 39, 39}
, {28, 45, -76}
, {-25, -57, -61}
, {-43, -45, -34}
, {14, -29, 2}
, {44, 0, -55}
, {-28, -13, -76}
, {47, 33, -12}
, {-19, -20, 21}
, {17, 7, 24}
, {0, -76, -3}
, {-10, 54, 35}
, {6, -49, -41}
, {-31, -4, -50}
, {36, -13, -45}
, {55, 8, -29}
, {-41, 52, 40}
, {-24, 50, -35}
, {16, -16, -64}
, {42, -23, -71}
, {-43, 63, 62}
, {-45, 1, -10}
}
, {{-4, 17, -9}
, {6, 39, 65}
, {-61, 66, -50}
, {-23, 18, -60}
, {70, 17, -33}
, {42, 65, 12}
, {-40, 26, 57}
, {56, 7, 16}
, {25, 9, 47}
, {70, 66, 42}
, {9, 58, -8}
, {-49, 42, -18}
, {70, 20, 16}
, {-67, -71, -23}
, {37, 75, -26}
, {73, -9, -31}
, {10, -17, -22}
, {67, 9, 61}
, {-55, -43, -43}
, {-2, 4, 7}
, {61, 15, 26}
, {-35, -9, -59}
, {19, 30, -34}
, {31, 0, 40}
, {-43, 9, 35}
, {21, -55, 19}
, {-18, -71, -57}
, {41, 33, 0}
, {-3, 19, 53}
, {-26, 14, -45}
, {51, -48, 26}
, {-62, 43, 1}
}
, {{51, -42, 40}
, {-15, 3, 11}
, {9, -75, 22}
, {52, 23, 44}
, {-22, 67, 39}
, {-16, -9, -58}
, {-15, 56, 77}
, {41, 27, 63}
, {-18, -41, 59}
, {75, 7, -55}
, {-64, -25, 61}
, {-26, 7, 8}
, {-70, -55, -31}
, {-64, 66, 71}
, {-18, -16, 63}
, {-72, 66, 10}
, {-57, 1, -65}
, {14, 61, 15}
, {-12, 55, -41}
, {44, 49, -17}
, {28, -17, 33}
, {-45, 56, 21}
, {-55, 76, -29}
, {59, 61, -21}
, {-11, 68, 49}
, {-17, 38, 1}
, {-31, 63, 66}
, {-1, 3, -55}
, {18, 71, 52}
, {-46, -11, 2}
, {44, 3, 36}
, {45, 60, 60}
}
, {{17, 12, 48}
, {21, 13, 0}
, {68, -29, -13}
, {-61, 46, 23}
, {23, -11, -60}
, {11, -29, 59}
, {-65, 29, -53}
, {28, 65, 68}
, {-7, 15, 71}
, {-26, -70, -53}
, {47, 13, -57}
, {-11, 20, 44}
, {-19, 1, -76}
, {-30, 56, -35}
, {35, 6, 9}
, {43, -22, -15}
, {-37, 42, -57}
, {33, -21, -53}
, {51, 46, 25}
, {36, -18, -8}
, {21, 0, 4}
, {27, -63, -25}
, {-33, 33, 13}
, {-44, -46, 53}
, {-45, 2, -25}
, {5, -20, -38}
, {2, -2, 47}
, {49, -64, -61}
, {-27, -8, -16}
, {29, 70, 33}
, {36, 42, 38}
, {80, 35, 46}
}
, {{17, -57, -16}
, {11, 34, -65}
, {41, -48, -20}
, {11, -71, 36}
, {-43, 40, -30}
, {32, -62, 6}
, {-3, 47, -43}
, {14, -20, -66}
, {27, 15, 3}
, {-33, -30, -18}
, {-65, 13, 65}
, {67, -41, -47}
, {65, 45, -42}
, {29, -62, -6}
, {-36, -70, -27}
, {-1, 3, 29}
, {-47, -38, -54}
, {-21, 59, -20}
, {1, -5, -55}
, {18, -45, -74}
, {41, -60, -18}
, {35, -59, -56}
, {-39, -49, -11}
, {-15, -46, 36}
, {68, -20, -77}
, {-54, 36, 13}
, {-45, 61, 14}
, {54, 50, 45}
, {11, -65, -52}
, {-5, -46, 23}
, {51, -22, -15}
, {66, -66, -12}
}
, {{-1, 35, 41}
, {52, 71, -7}
, {41, 45, 9}
, {64, -21, -17}
, {-11, 74, -7}
, {8, 19, 37}
, {10, 54, -36}
, {-40, -7, -63}
, {20, 71, 3}
, {8, 35, -60}
, {-19, 35, -9}
, {-39, 2, 48}
, {-14, -4, -69}
, {-42, 68, -24}
, {-70, -5, -67}
, {-65, -63, 49}
, {50, 11, -22}
, {-27, 14, -57}
, {3, -47, -46}
, {-46, -1, 31}
, {-5, 6, 45}
, {-41, 33, 36}
, {44, -18, -42}
, {-33, -62, -49}
, {-39, -18, -5}
, {-4, -50, 72}
, {51, 77, 1}
, {-18, -54, -69}
, {-56, 42, 64}
, {-19, -27, 39}
, {-52, -73, -12}
, {29, 85, 21}
}
, {{60, -16, 63}
, {35, -28, 33}
, {0, -72, -26}
, {5, 46, -43}
, {41, 41, -30}
, {39, 53, 49}
, {51, 63, 32}
, {39, -10, -65}
, {-11, -52, -47}
, {31, 50, -28}
, {24, -22, 7}
, {61, 20, 0}
, {24, -28, 17}
, {36, -73, 20}
, {2, 39, 9}
, {-28, -33, 49}
, {-76, -74, -54}
, {17, -73, 23}
, {8, 31, -23}
, {-24, -33, 26}
, {-1, -41, 41}
, {-54, -1, 63}
, {-9, 12, 47}
, {-2, 23, -4}
, {13, -1, 4}
, {57, -25, 69}
, {17, 49, 26}
, {59, -19, -13}
, {26, 29, -28}
, {-42, 8, -29}
, {-70, 65, 44}
, {30, 0, 32}
}
, {{34, 42, 49}
, {33, -70, -58}
, {28, 12, -30}
, {-11, -43, 46}
, {-62, 29, 30}
, {37, -63, 26}
, {57, -36, -22}
, {40, 32, 33}
, {-61, 81, 33}
, {29, 69, 30}
, {69, -52, -36}
, {27, -37, -64}
, {-28, 51, -51}
, {42, 3, 52}
, {-25, 37, -26}
, {-54, 26, 43}
, {27, 39, 24}
, {25, -44, 2}
, {42, 25, -76}
, {45, 5, 33}
, {39, 23, -61}
, {36, 57, -39}
, {9, -35, -39}
, {-35, 64, 68}
, {-19, 33, -17}
, {60, 77, -19}
, {-47, -52, 62}
, {59, 91, 68}
, {4, -53, 45}
, {-22, -7, -5}
, {27, 71, -74}
, {81, 58, 48}
}
, {{47, -52, -28}
, {9, -54, 22}
, {33, -51, 26}
, {23, -17, -61}
, {-24, 68, -2}
, {13, -45, -47}
, {76, 68, 61}
, {64, 23, -2}
, {60, 75, -60}
, {-47, 42, 28}
, {-67, -44, -46}
, {-7, -57, -11}
, {69, -54, 26}
, {-47, 38, 70}
, {-38, -43, -46}
, {-44, 60, -34}
, {53, 60, 0}
, {-45, 12, 70}
, {-55, 57, 63}
, {69, 52, 9}
, {-61, -28, 27}
, {24, 41, -64}
, {-11, 10, -38}
, {35, 45, 54}
, {-32, 5, 78}
, {10, 38, 24}
, {27, 51, 52}
, {40, 18, -17}
, {-37, 0, -70}
, {29, -69, 9}
, {-46, 35, -11}
, {16, 28, 65}
}
, {{6, -26, -71}
, {-21, 53, 42}
, {52, -2, 37}
, {-54, -30, 2}
, {16, -24, -48}
, {-56, -15, 8}
, {-49, 45, 20}
, {-56, 38, -14}
, {59, 33, 50}
, {43, 28, -8}
, {55, 37, 38}
, {6, 69, 11}
, {29, 57, -40}
, {58, -51, 45}
, {42, -41, 12}
, {12, -58, -32}
, {17, -11, -51}
, {73, -10, -32}
, {-41, 80, 18}
, {-20, -3, -27}
, {65, 64, -57}
, {-6, -10, 49}
, {3, 39, -42}
, {42, 0, 12}
, {-76, -3, 3}
, {73, 71, -41}
, {0, 24, -4}
, {-28, -70, -30}
, {0, -46, -45}
, {9, -39, -55}
, {-51, -56, -77}
, {-21, -60, 16}
}
, {{34, -11, -44}
, {-6, -10, -32}
, {-33, 1, -15}
, {-41, 3, 12}
, {7, 45, 28}
, {-28, -27, -40}
, {-14, 82, 58}
, {59, -27, 56}
, {-13, 59, -60}
, {-21, 49, 6}
, {-72, 11, 49}
, {47, -2, 20}
, {-24, 16, -54}
, {-63, -21, -1}
, {-45, 45, 7}
, {47, 46, -1}
, {70, 73, -14}
, {-51, -25, 70}
, {-41, -34, 48}
, {-41, -42, 87}
, {6, -40, -54}
, {54, -35, 13}
, {5, -17, -15}
, {62, 44, 0}
, {-12, -33, 7}
, {-1, 17, 83}
, {20, -29, -5}
, {-63, 24, -34}
, {24, 32, 56}
, {24, 86, -25}
, {0, 54, -49}
, {41, 41, 80}
}
, {{-21, 38, 40}
, {62, -19, 3}
, {54, -20, 45}
, {29, 13, -35}
, {76, -27, 28}
, {-26, -18, -41}
, {46, 34, -52}
, {78, 19, -68}
, {54, 33, 49}
, {-56, 32, -35}
, {77, 13, -73}
, {-21, 13, -33}
, {-30, 29, -38}
, {38, -14, 11}
, {20, -42, -11}
, {21, 59, 4}
, {-7, -15, -12}
, {-18, -8, 76}
, {-82, -66, 36}
, {80, 69, 18}
, {-62, 34, 6}
, {-13, 56, -33}
, {12, -25, -23}
, {3, -62, 49}
, {-41, -59, -30}
, {15, -8, -47}
, {-68, 40, -43}
, {-39, -63, 48}
, {77, -33, 11}
, {25, 50, -48}
, {22, 57, 10}
, {-36, -22, 43}
}
, {{76, 71, -35}
, {-63, 60, 42}
, {67, -32, -34}
, {-59, 10, -23}
, {74, 44, 68}
, {-41, 36, -22}
, {52, 73, 51}
, {-38, 40, -10}
, {41, 17, 33}
, {34, -52, -72}
, {51, 0, 4}
, {32, 62, 66}
, {-32, 3, -5}
, {-54, 18, -63}
, {53, 73, 1}
, {28, 8, -10}
, {25, -48, -54}
, {9, -15, 74}
, {39, -4, -42}
, {13, 73, -2}
, {22, 83, -20}
, {-19, -7, -29}
, {-13, 65, 62}
, {23, -59, 41}
, {22, -43, -20}
, {-66, -34, 10}
, {-53, 9, -75}
, {-83, 57, 32}
, {-27, 70, 39}
, {7, -1, 49}
, {-33, 10, -7}
, {40, -25, 3}
}
, {{51, 23, -22}
, {66, -64, -60}
, {25, -4, -3}
, {4, -30, 72}
, {27, 52, -3}
, {-16, 8, -41}
, {-28, 44, -46}
, {-67, -29, 42}
, {-37, 44, -53}
, {-56, -66, 63}
, {-79, 51, -26}
, {40, -7, -18}
, {55, -70, 16}
, {79, -82, 5}
, {-4, -12, 52}
, {46, -68, 44}
, {70, -1, 36}
, {47, -37, 28}
, {-17, 79, 63}
, {26, 15, 52}
, {-35, -26, 34}
, {-51, 30, 41}
, {-41, -59, 79}
, {-53, -27, 77}
, {0, 26, -18}
, {5, -17, 46}
, {32, 33, -60}
, {-74, 28, 37}
, {4, -68, -36}
, {-64, 64, 48}
, {2, -8, 38}
, {-61, -17, -37}
}
, {{-64, 22, -53}
, {44, 18, 13}
, {-36, 68, 52}
, {-43, -68, 14}
, {-2, -25, -8}
, {1, -15, 12}
, {38, 8, -26}
, {29, -7, -54}
, {50, -2, -36}
, {-17, 66, -57}
, {-33, 0, 34}
, {25, 32, -71}
, {-17, -10, 57}
, {7, 55, -61}
, {-72, -2, 55}
, {55, 24, -35}
, {-1, 53, 71}
, {24, -28, 41}
, {-28, 8, -5}
, {36, 17, 45}
, {39, -87, -40}
, {-68, 69, 63}
, {-25, -64, 0}
, {-13, -56, 60}
, {12, 56, 6}
, {46, -56, -52}
, {-10, 82, -66}
, {9, -42, 39}
, {-66, -39, 59}
, {46, -28, 8}
, {9, -17, -62}
, {23, -63, -49}
}
, {{-2, -56, 37}
, {-66, 49, -24}
, {62, -24, -51}
, {9, 17, -68}
, {-30, -61, 48}
, {-66, 70, 61}
, {51, 35, 24}
, {32, 49, -46}
, {63, 51, 42}
, {7, 11, -72}
, {-34, 65, -41}
, {-52, -63, -11}
, {38, 38, 61}
, {-18, 46, -5}
, {64, -21, -21}
, {-61, 57, -53}
, {-4, -27, 42}
, {18, -53, -45}
, {18, -47, 34}
, {-61, 58, -2}
, {46, -55, -13}
, {47, -57, -57}
, {-1, -38, 52}
, {-48, 53, -69}
, {-65, -44, -66}
, {-47, 48, 61}
, {2, -26, 64}
, {-78, -52, 56}
, {22, 72, -53}
, {60, -28, -55}
, {0, 16, -30}
, {47, 35, 51}
}
, {{6, 80, 0}
, {61, 6, 56}
, {22, -14, -65}
, {-55, 65, -44}
, {-69, -76, 39}
, {4, 23, 40}
, {52, -2, -26}
, {-36, 4, -65}
, {71, 6, 41}
, {54, -8, 1}
, {30, -7, -3}
, {32, -53, 55}
, {-37, -48, 60}
, {0, -58, 33}
, {61, -31, -77}
, {-51, 87, -48}
, {-60, -37, -23}
, {60, -6, 31}
, {57, 46, -79}
, {-9, -72, 11}
, {-33, 20, 11}
, {38, 52, 40}
, {19, 67, 61}
, {-74, 0, -45}
, {-61, 38, 31}
, {41, -3, -37}
, {62, -28, 19}
, {0, 46, 14}
, {68, 80, -41}
, {46, -12, -37}
, {-35, 27, -6}
, {-60, -32, -49}
}
, {{80, -28, -17}
, {8, -52, 61}
, {-73, -4, -73}
, {-45, -18, -19}
, {-49, 28, 72}
, {-31, -4, 39}
, {-41, -79, -46}
, {66, -20, -62}
, {-42, -41, 3}
, {-43, -59, 30}
, {-22, 57, 56}
, {32, 32, 20}
, {68, 40, -30}
, {21, -55, 2}
, {66, 20, 34}
, {16, -69, -70}
, {15, 26, -49}
, {29, 53, 0}
, {-54, 73, 5}
, {-5, 40, 40}
, {-53, -19, -26}
, {-25, -54, 44}
, {-55, 32, -42}
, {16, 5, -13}
, {-62, -18, -11}
, {36, 31, 82}
, {-59, -82, 29}
, {-39, -10, 30}
, {19, -16, 39}
, {64, -61, 69}
, {24, -72, -11}
, {28, 24, 11}
}
, {{34, -55, -11}
, {20, -52, -25}
, {-60, -67, -9}
, {10, -77, 21}
, {-56, 12, -51}
, {-64, -68, 10}
, {39, -41, 64}
, {37, -68, -61}
, {17, -24, -43}
, {47, 19, 45}
, {26, 51, 72}
, {-60, -72, -73}
, {28, -48, -49}
, {-13, -31, 27}
, {-59, 37, -41}
, {-2, 43, -67}
, {0, 27, 33}
, {-12, 65, 58}
, {13, -10, 21}
, {-37, 32, -22}
, {-52, 28, -8}
, {-44, 6, -41}
, {-55, 20, -50}
, {-67, 36, 50}
, {-55, -1, 4}
, {-77, 10, -57}
, {9, -72, 15}
, {-7, 72, 13}
, {56, 33, 30}
, {62, 26, -63}
, {-74, 22, -62}
, {43, -61, -45}
}
, {{39, 40, -51}
, {-28, 38, 52}
, {15, -14, -15}
, {17, -64, 58}
, {83, -22, 9}
, {7, 1, -55}
, {-50, 43, 24}
, {-22, 47, 31}
, {-44, -42, -1}
, {59, 11, 43}
, {25, 26, -7}
, {-9, 75, -21}
, {-29, 52, 7}
, {56, 70, 43}
, {6, -73, -59}
, {-17, -29, -47}
, {-10, -5, -61}
, {-64, -54, -57}
, {-65, -27, -29}
, {-31, 61, -24}
, {-14, 28, 47}
, {13, 26, 54}
, {55, -47, 38}
, {-38, -3, 9}
, {39, 72, 70}
, {0, 36, -8}
, {39, 54, -9}
, {-2, 46, 51}
, {14, -39, 38}
, {-50, -51, 47}
, {4, 82, 31}
, {23, -39, -24}
}
, {{-65, 5, -67}
, {-40, 36, -41}
, {1, -74, 56}
, {-68, 20, -20}
, {74, -21, -4}
, {46, -54, -65}
, {-33, 19, -36}
, {1, 71, -40}
, {-43, 8, -21}
, {-11, 48, -51}
, {53, -76, 7}
, {-8, 65, 57}
, {26, -68, 39}
, {-53, 15, 54}
, {66, 24, 74}
, {-25, -8, -43}
, {8, -29, 74}
, {50, 48, -68}
, {2, -28, -30}
, {14, 43, 29}
, {35, 61, 69}
, {-60, -35, 35}
, {68, 39, -10}
, {67, 48, -35}
, {4, -63, 59}
, {-50, -28, 25}
, {-36, -27, -25}
, {18, 15, 49}
, {-11, -61, -63}
, {28, -58, 54}
, {-53, 51, 41}
, {-2, -25, 25}
}
, {{72, -58, -1}
, {14, 7, 11}
, {1, 8, 53}
, {-48, 55, 41}
, {-65, -24, 15}
, {66, -24, -24}
, {59, 13, -42}
, {-18, -69, 66}
, {-26, 9, -76}
, {-39, 15, 44}
, {36, -2, -57}
, {-26, -56, 7}
, {-5, 48, 10}
, {60, -25, -67}
, {4, -46, 17}
, {55, 56, 46}
, {-23, -7, 18}
, {-35, 71, -32}
, {-83, -55, -66}
, {-30, -22, -79}
, {36, -49, -5}
, {69, 24, 37}
, {40, -68, 35}
, {-45, -58, 48}
, {-63, 62, 0}
, {59, 61, -11}
, {-52, 71, 22}
, {-30, -23, -1}
, {-67, -54, 67}
, {-20, 5, 38}
, {30, -33, 40}
, {-35, 60, -54}
}
, {{-72, -81, -57}
, {0, 35, 27}
, {-53, 7, -4}
, {-5, 65, -29}
, {4, 21, -64}
, {14, 23, -63}
, {-1, 40, -55}
, {-5, 12, -45}
, {-15, 55, -74}
, {-75, -5, 64}
, {-1, 70, 18}
, {72, 49, -22}
, {70, 0, 41}
, {56, 49, 46}
, {0, -7, 62}
, {-5, -32, -50}
, {72, -21, 40}
, {-11, 18, -65}
, {-29, 23, 12}
, {-16, 4, 19}
, {-32, -65, -4}
, {-62, 22, 9}
, {9, 2, -43}
, {-79, -49, -48}
, {13, -8, 44}
, {-19, -27, -7}
, {52, 42, 20}
, {43, 18, 29}
, {-54, 61, -28}
, {1, 66, -17}
, {-32, 41, 70}
, {-63, -5, -35}
}
, {{24, 25, -1}
, {-7, -23, -39}
, {-53, -52, 19}
, {-19, -46, -61}
, {41, 10, 51}
, {8, -52, -31}
, {29, -15, 48}
, {49, 80, 0}
, {31, -56, 59}
, {-15, -31, 76}
, {64, -69, -57}
, {45, -54, 21}
, {9, -41, 18}
, {10, 6, -5}
, {65, -36, 30}
, {-41, -21, 43}
, {34, 82, 35}
, {-53, -41, 36}
, {-73, -28, 16}
, {9, 38, 15}
, {-34, 51, -35}
, {-26, 75, -26}
, {-75, 12, -24}
, {-14, -49, -40}
, {-60, -11, -26}
, {48, 19, -4}
, {74, 42, -33}
, {18, 19, 46}
, {-42, 13, 23}
, {79, 25, 83}
, {-27, 86, 7}
, {45, -40, 4}
}
, {{-42, -2, 31}
, {-64, 55, 31}
, {-30, -42, -7}
, {-62, -43, 31}
, {-25, -10, 33}
, {67, 73, 71}
, {-23, -52, 56}
, {28, -47, -49}
, {-41, -40, -47}
, {0, -2, -19}
, {-4, 64, 12}
, {74, 35, 20}
, {-23, 60, 59}
, {-4, -63, 10}
, {-27, 5, 22}
, {-43, -19, 28}
, {-72, -40, 12}
, {18, 26, -68}
, {10, 19, 10}
, {25, -8, 64}
, {51, 24, 34}
, {19, -43, 76}
, {-5, 64, -12}
, {-11, -2, 74}
, {31, -24, -22}
, {65, -67, -36}
, {73, 2, -45}
, {-45, -64, -73}
, {54, -48, 24}
, {-47, 51, -47}
, {32, 19, 41}
, {-7, 41, -43}
}
, {{48, -68, 46}
, {9, -42, 26}
, {22, -61, 52}
, {-57, -20, -12}
, {52, 20, 49}
, {28, 20, 76}
, {-61, 4, -14}
, {51, 3, 48}
, {4, -42, -75}
, {45, -52, 31}
, {-52, 43, -42}
, {42, -38, -52}
, {28, 69, -36}
, {81, 46, -33}
, {-44, 12, -9}
, {23, 13, 23}
, {31, 9, 12}
, {30, 20, 68}
, {61, 12, -66}
, {-67, 8, -73}
, {52, -4, -64}
, {-26, -65, 35}
, {-30, 17, 33}
, {-39, -32, 2}
, {0, 60, 41}
, {66, 15, 59}
, {6, 36, 8}
, {11, -14, 30}
, {-55, -45, 56}
, {-46, 18, -26}
, {31, 59, 36}
, {7, -50, 47}
}
, {{-56, -56, 90}
, {46, -50, 56}
, {-29, 56, -22}
, {-23, 7, 43}
, {44, 0, -43}
, {4, -33, 70}
, {27, -58, -6}
, {75, -64, -45}
, {28, 27, -23}
, {-15, -4, 0}
, {-28, -39, -34}
, {49, -27, 34}
, {80, -58, -39}
, {-12, 34, -27}
, {44, 35, 55}
, {7, 44, 89}
, {-62, -32, -27}
, {-21, -23, 84}
, {44, -49, -8}
, {60, 6, -43}
, {8, -43, 76}
, {66, -51, 4}
, {-54, -45, -41}
, {-6, 0, -2}
, {-3, 35, -58}
, {34, -40, -19}
, {-34, -20, 25}
, {67, 48, 57}
, {-55, 0, -8}
, {-50, -9, 59}
, {4, 31, -76}
, {-8, 79, -52}
}
, {{-26, 26, 39}
, {-45, 16, 16}
, {61, -35, -22}
, {36, -31, -56}
, {50, 54, -4}
, {26, 28, -35}
, {-33, 67, 57}
, {39, 38, 64}
, {9, -43, 27}
, {-61, -51, 50}
, {55, -38, 54}
, {30, 40, -74}
, {55, 15, 41}
, {-44, 70, -47}
, {-4, -18, 40}
, {19, -5, -61}
, {-35, 42, -15}
, {-44, -70, 70}
, {-45, 61, 38}
, {30, 57, 16}
, {-32, -68, -37}
, {-72, 41, -75}
, {31, 5, -58}
, {-29, 61, 66}
, {39, -17, 36}
, {36, 26, -46}
, {-22, -22, 65}
, {-30, 13, 61}
, {16, -71, -25}
, {-29, -49, 25}
, {-17, -2, -68}
, {43, 36, 39}
}
, {{35, 44, 34}
, {-23, 34, -43}
, {-21, -3, -61}
, {17, 22, 55}
, {46, 55, 48}
, {32, 52, 16}
, {8, 39, 4}
, {-50, -58, 62}
, {47, 41, 2}
, {64, -44, -28}
, {-21, 35, 15}
, {58, 23, 39}
, {21, -7, -56}
, {31, 50, -48}
, {-82, 16, -5}
, {4, 64, -60}
, {19, 48, 3}
, {3, 27, -69}
, {-53, 32, -32}
, {-65, -38, -48}
, {60, -6, -35}
, {55, 30, -61}
, {1, 30, 44}
, {-44, 33, 42}
, {38, -22, -38}
, {-24, -20, 36}
, {-68, 76, 52}
, {-51, 66, -11}
, {30, -26, 31}
, {-13, 64, 57}
, {-55, 9, -44}
, {29, 19, -55}
}
, {{-38, -50, 48}
, {-38, -65, 70}
, {0, 19, -29}
, {-10, -7, 16}
, {-35, -73, -37}
, {-76, 13, -63}
, {57, -13, 51}
, {62, 46, -30}
, {-65, 34, 51}
, {73, 65, -53}
, {-2, -11, -36}
, {45, -5, -4}
, {-55, -67, 71}
, {-18, 65, -36}
, {75, 18, -18}
, {0, -32, 26}
, {-47, 58, 58}
, {16, -29, 59}
, {5, 69, 42}
, {19, -48, -49}
, {29, 16, -43}
, {-40, -37, 59}
, {0, -7, -58}
, {10, 24, 3}
, {-56, 43, 16}
, {-70, 25, 73}
, {45, -63, -41}
, {-37, 56, 80}
, {-47, -27, -57}
, {57, -25, -24}
, {40, 76, 40}
, {-16, 64, -27}
}
, {{2, -6, -37}
, {-12, -58, -63}
, {47, -70, 43}
, {-37, 14, 37}
, {-49, 23, -72}
, {33, -13, -31}
, {-36, -73, -38}
, {-44, 20, 56}
, {-39, 14, -27}
, {-70, 61, 41}
, {83, -35, -49}
, {54, 72, -24}
, {-1, 29, 13}
, {80, 13, -63}
, {-44, 21, 60}
, {30, 56, -10}
, {6, 9, 14}
, {31, -34, 52}
, {7, -43, 20}
, {68, -13, 36}
, {64, -2, -67}
, {23, -64, 30}
, {-29, -35, -81}
, {63, 6, 25}
, {29, 37, 47}
, {-29, 42, 39}
, {23, -59, 72}
, {89, -18, 38}
, {10, 28, -51}
, {29, 50, 64}
, {64, 18, 5}
, {-42, 23, -7}
}
, {{23, -14, -29}
, {-62, -14, -58}
, {51, -5, -4}
, {-58, -61, 5}
, {-42, 71, 43}
, {-1, 78, 64}
, {62, 48, 8}
, {37, 66, 55}
, {11, 40, -33}
, {-64, -29, 32}
, {65, 66, -50}
, {-55, -64, -3}
, {46, 36, -12}
, {67, -1, 39}
, {-58, -74, -66}
, {40, -34, 24}
, {-44, -20, 48}
, {-64, -41, 35}
, {56, -55, -3}
, {16, -64, 7}
, {-51, 64, -56}
, {50, 24, -46}
, {50, -43, 24}
, {-40, 9, 7}
, {-28, 2, 24}
, {-52, 39, 45}
, {-29, -52, 35}
, {-71, 0, 70}
, {18, -63, 27}
, {42, -21, 60}
, {49, -46, 34}
, {19, -52, -56}
}
, {{55, -4, 7}
, {-68, 11, 53}
, {27, -10, 72}
, {42, -19, 66}
, {38, -35, 58}
, {-11, 67, -47}
, {43, 13, 63}
, {-68, -27, 27}
, {-63, -23, 2}
, {17, 17, 0}
, {32, 62, 12}
, {-58, -60, 21}
, {-70, -53, -72}
, {-53, 25, 22}
, {37, 23, 1}
, {-49, 23, 45}
, {-47, -58, 25}
, {-12, 29, -63}
, {-48, -69, 55}
, {39, -26, 61}
, {74, -46, 22}
, {39, -56, 54}
, {63, 32, -69}
, {-16, -28, 49}
, {25, 32, 21}
, {38, -63, 48}
, {16, -25, 29}
, {3, -58, 20}
, {31, -32, 8}
, {34, 25, -71}
, {17, -24, -40}
, {-61, -34, 15}
}
, {{5, -69, 29}
, {18, -50, -22}
, {-19, 15, 40}
, {-32, 70, 38}
, {7, -10, -53}
, {-27, 24, 70}
, {-13, -28, -18}
, {35, -43, -26}
, {15, 37, 40}
, {-81, 51, 54}
, {8, 67, 42}
, {-9, -49, 18}
, {-23, -50, 25}
, {20, -38, -19}
, {18, 40, 55}
, {14, -44, 73}
, {34, -38, 7}
, {-66, 9, 22}
, {51, 18, 22}
, {10, -21, 63}
, {85, -31, -45}
, {-50, 4, -52}
, {-22, 52, 25}
, {-27, -49, -23}
, {39, 16, 7}
, {54, -29, 64}
, {-20, 58, -29}
, {17, 82, -42}
, {-29, 66, 20}
, {-51, -42, -12}
, {-64, 53, -57}
, {-44, 35, 29}
}
, {{-61, -31, -45}
, {-49, -37, 4}
, {-14, 0, 30}
, {-54, 31, -21}
, {5, -16, -56}
, {20, -33, 2}
, {-46, 2, -33}
, {-32, 71, -58}
, {-50, 12, -8}
, {-54, 15, -69}
, {38, -43, 8}
, {84, 46, 17}
, {14, 16, 55}
, {72, 7, 18}
, {79, -1, -49}
, {26, 17, -20}
, {-64, -32, -55}
, {55, -23, -53}
, {5, 3, -10}
, {-30, -14, 36}
, {1, 6, 20}
, {18, 60, -60}
, {-13, -67, -40}
, {58, 3, -13}
, {-40, -52, 16}
, {-5, 40, 49}
, {80, -44, 35}
, {30, -40, -50}
, {-53, 21, 0}
, {48, 39, 27}
, {-5, -41, -27}
, {70, 42, 69}
}
, {{22, 26, -50}
, {32, 0, 48}
, {45, -11, 31}
, {-42, -47, -77}
, {26, 81, -32}
, {-20, -39, -12}
, {26, -56, -65}
, {37, 70, 63}
, {-3, 74, 3}
, {1, -16, 25}
, {75, 67, -54}
, {71, -26, -55}
, {-30, 37, 45}
, {-18, -47, 67}
, {11, 54, -28}
, {-34, 42, -32}
, {-27, -41, 12}
, {-30, 8, 53}
, {58, -46, 52}
, {3, 53, 67}
, {-23, 61, -4}
, {70, -5, 10}
, {79, 55, 26}
, {-22, -70, -1}
, {63, 0, 64}
, {-20, 34, -42}
, {-31, 20, -14}
, {-85, 78, -32}
, {13, 59, 46}
, {41, 59, 72}
, {56, 13, 9}
, {-18, -31, 33}
}
, {{16, -24, 24}
, {54, -8, 76}
, {64, 2, -28}
, {-23, -35, 14}
, {-39, 12, 36}
, {-39, 7, 43}
, {-45, -62, 72}
, {10, 48, 5}
, {19, 50, -20}
, {-55, -18, -62}
, {42, 69, -36}
, {10, 26, -4}
, {62, 1, 6}
, {56, -22, -53}
, {16, 46, 0}
, {69, 24, 74}
, {-31, 53, 11}
, {-66, 27, 16}
, {-73, 38, -12}
, {-31, -18, -3}
, {61, 4, 32}
, {-10, -56, 6}
, {-5, -86, -81}
, {2, -58, 78}
, {50, 59, -54}
, {-65, 10, -59}
, {68, -54, -53}
, {-81, 12, 14}
, {3, 24, 63}
, {39, -25, -27}
, {-47, 13, -56}
, {47, 37, 72}
}
, {{-41, -38, -75}
, {-44, -25, -26}
, {14, 33, -58}
, {-43, 49, -56}
, {41, -15, 68}
, {62, 50, 47}
, {-40, -5, -2}
, {-26, 35, -22}
, {14, -10, 60}
, {60, -36, -25}
, {-56, -70, 17}
, {-29, 70, -67}
, {-31, 51, -27}
, {-34, 1, 53}
, {-22, 4, 51}
, {-64, 59, -13}
, {40, -55, -36}
, {-21, 7, -29}
, {71, -4, -10}
, {-44, 69, -12}
, {-37, -39, -3}
, {26, -31, -23}
, {-44, -57, 70}
, {65, -60, 32}
, {54, 22, 48}
, {14, 23, -49}
, {29, 65, 74}
, {78, -17, 18}
, {-23, -45, 9}
, {-8, -44, -52}
, {-63, -56, 47}
, {-51, -35, 52}
}
, {{67, 27, -64}
, {-24, -31, 40}
, {-33, -23, 61}
, {-40, -60, -40}
, {-68, 21, -82}
, {52, -53, 75}
, {5, 48, -75}
, {-35, -49, -31}
, {-20, -63, 15}
, {2, 50, -30}
, {-36, 45, -68}
, {26, 39, 44}
, {65, 20, -60}
, {36, 75, -55}
, {0, -37, 54}
, {66, -2, -8}
, {86, -63, 17}
, {57, 47, -32}
, {-51, 40, 4}
, {-44, -17, -52}
, {82, -70, -54}
, {27, 58, -40}
, {-14, -11, -19}
, {42, -67, -5}
, {24, 10, -48}
, {0, -42, -10}
, {37, 32, 66}
, {79, 29, 72}
, {39, 18, -33}
, {38, 32, -66}
, {-26, -65, -62}
, {57, -47, -2}
}
, {{-3, -25, -15}
, {-24, 76, 9}
, {56, 81, -66}
, {18, 51, 60}
, {-63, -60, -74}
, {-12, 54, 7}
, {66, 62, -22}
, {50, -19, -70}
, {-5, -39, -12}
, {-58, -20, 26}
, {-68, -17, 1}
, {33, 13, 1}
, {80, -31, -35}
, {-35, -31, 22}
, {-73, -36, 53}
, {66, 29, -44}
, {-67, -50, -30}
, {13, -22, -21}
, {49, 55, 17}
, {4, -79, -35}
, {-35, -85, -56}
, {-17, -47, -24}
, {71, -36, 20}
, {-40, 22, 21}
, {-13, 40, 68}
, {59, -14, 17}
, {77, -22, 23}
, {42, -29, -57}
, {-74, -19, 25}
, {45, 36, -71}
, {-45, 49, -18}
, {-13, -21, 4}
}
, {{69, -47, 30}
, {35, -65, -53}
, {56, 15, -22}
, {11, 5, 49}
, {-26, -46, 5}
, {-38, -11, 65}
, {42, 23, -29}
, {61, -19, -74}
, {36, 60, 29}
, {36, -66, 22}
, {13, 16, -27}
, {82, 51, -19}
, {56, 45, -57}
, {35, 2, 17}
, {-48, -46, 73}
, {16, -18, -36}
, {-16, -67, -62}
, {26, -53, 24}
, {-70, -20, -46}
, {-47, -66, -32}
, {-49, 26, -56}
, {62, -14, -48}
, {-8, 61, -7}
, {32, -45, 5}
, {0, -46, 70}
, {69, -63, -79}
, {-25, -4, -55}
, {-84, -14, 40}
, {52, 70, 51}
, {-33, -24, -10}
, {-66, -61, -11}
, {-35, -69, -50}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t max_pooling1d_3_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_3(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t conv1d_3_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_3(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    64
#define CONV_FILTERS      128
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_3_bias[CONV_FILTERS] = {-1, 8, -9, 9, -4, 2, -5, 6, -2, 0, 14, -8, -1, 11, 0, 0, 15, -7, -4, 13, 16, 17, 10, -2, -4, -10, -9, 8, 12, -4, -6, 0, 6, -10, 6, 4, 0, -4, -8, 3, -4, 5, -4, 10, 0, 11, -5, -5, -3, 0, -3, -6, -4, -4, 7, 5, -5, -5, -6, -9, 6, -5, 12, -4, -10, 15, 1, 8, 0, -6, -1, -5, -6, 4, 5, 6, -7, -9, 11, -4, -3, -1, -2, 13, -2, 9, -4, 9, -3, -7, -1, -3, 1, -5, 0, -8, 7, 11, -6, 5, -6, 15, -5, 2, -1, -3, -7, 8, -6, 4, 9, 9, -3, 8, 8, -4, -5, 14, 11, 13, -2, 11, 0, 3, 2, 6, -2, 8}
;

const int16_t conv1d_3_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-24, 58, 47}
, {41, -28, 12}
, {-19, -19, -5}
, {28, -12, 58}
, {-34, 4, 2}
, {5, 18, 48}
, {21, 20, -20}
, {-20, -39, 53}
, {-14, 41, -6}
, {-33, 13, 10}
, {-39, 15, 42}
, {27, 39, -12}
, {51, -4, -16}
, {44, 62, -40}
, {6, -51, -41}
, {-31, -65, -48}
, {8, 26, -38}
, {17, -2, -27}
, {-15, 20, 40}
, {60, 31, -3}
, {-9, -48, 27}
, {5, -38, -2}
, {-26, 13, 43}
, {19, 39, 36}
, {9, -44, 33}
, {-2, 48, -10}
, {8, 7, 3}
, {-24, 3, 3}
, {20, -46, -22}
, {20, -31, 7}
, {-33, -16, -50}
, {-18, -1, -31}
, {-20, 45, 0}
, {27, -51, -11}
, {-47, 11, -50}
, {12, -6, -1}
, {-12, 33, 35}
, {-14, 36, -7}
, {52, 39, 32}
, {-44, 2, 22}
, {-14, 3, 54}
, {-7, 42, 37}
, {-12, -50, 12}
, {-33, -35, 25}
, {-11, -3, 50}
, {-28, 46, -32}
, {-7, -47, -32}
, {-38, 21, 10}
, {1, -40, 42}
, {-53, 13, -44}
, {32, 50, 35}
, {-35, 6, 43}
, {-34, -5, -19}
, {23, -35, -19}
, {-16, 19, -44}
, {-57, 0, -57}
, {27, -31, -18}
, {32, 10, -30}
, {27, 18, 13}
, {10, -12, -41}
, {-19, -6, 22}
, {58, -26, -20}
, {59, -29, 4}
, {-21, -21, -18}
}
, {{56, 57, -38}
, {-43, 50, 47}
, {54, -32, -3}
, {-35, 40, -17}
, {-44, 47, 34}
, {8, 9, 26}
, {25, 20, -39}
, {14, -25, -24}
, {20, -41, -58}
, {-16, 9, -19}
, {8, 53, 7}
, {-46, -5, 37}
, {-6, -31, -11}
, {-31, 11, 46}
, {18, -27, -14}
, {-24, -10, -18}
, {-28, -22, -40}
, {-23, -1, 1}
, {-23, 44, 12}
, {37, 56, -29}
, {10, -18, -21}
, {38, -48, 9}
, {-39, 31, -13}
, {2, -26, 21}
, {-12, 38, -4}
, {-37, -36, -21}
, {-24, 18, 42}
, {-20, 10, -57}
, {38, -32, -37}
, {-41, -29, -41}
, {0, 8, -23}
, {-5, 5, 44}
, {3, -4, 19}
, {-52, -17, -44}
, {29, -50, 7}
, {33, 6, -45}
, {-5, -12, 11}
, {53, -12, 37}
, {26, -33, 37}
, {-10, 31, -29}
, {1, 16, 49}
, {38, -40, -1}
, {-28, -2, -50}
, {48, -13, 45}
, {39, 49, 16}
, {22, -14, -37}
, {-36, 43, -15}
, {-18, -46, -55}
, {45, 19, 0}
, {9, -26, -19}
, {2, 45, 51}
, {-3, 36, 23}
, {-39, 0, 7}
, {-19, 18, 48}
, {-11, 2, 4}
, {26, -34, 19}
, {10, -26, -7}
, {45, -15, 50}
, {25, 7, 52}
, {-2, 24, -37}
, {-10, 31, -16}
, {4, -39, 14}
, {12, 9, -48}
, {24, 5, 25}
}
, {{-2, 10, -30}
, {-26, -46, 32}
, {-37, 18, -52}
, {11, 28, 50}
, {-39, -24, -49}
, {11, 39, -26}
, {-19, -23, -27}
, {26, 26, -48}
, {32, 54, -34}
, {17, 59, -20}
, {1, -33, 0}
, {32, 29, -9}
, {-39, -23, 3}
, {10, 41, 32}
, {0, 19, 23}
, {1, -13, -8}
, {-25, -38, -11}
, {-33, -37, 1}
, {36, 41, 31}
, {-57, -53, 28}
, {-8, -14, -36}
, {-47, -21, 48}
, {-24, -4, -48}
, {54, 35, -47}
, {25, 52, 35}
, {44, 25, 26}
, {2, 51, -33}
, {12, -42, -14}
, {53, 12, -25}
, {0, 40, -46}
, {23, -43, -29}
, {5, 52, 40}
, {30, 19, 26}
, {45, 41, -36}
, {67, 11, 22}
, {4, -20, 50}
, {12, 41, 33}
, {-43, 6, -3}
, {39, -2, 25}
, {-33, -2, -7}
, {-15, 24, -10}
, {14, 3, 30}
, {-55, -55, -39}
, {32, -41, -48}
, {-24, -30, 27}
, {-51, -1, 37}
, {-33, 31, 64}
, {48, -14, 32}
, {14, -21, -22}
, {63, -31, -39}
, {6, 38, 37}
, {63, 52, 4}
, {-49, -42, -52}
, {-19, 24, 23}
, {-35, -14, 22}
, {12, 33, -39}
, {-15, -57, 31}
, {-31, 18, -24}
, {-42, 29, -38}
, {60, -32, 10}
, {-46, 11, -45}
, {-48, 12, -8}
, {9, -58, -51}
, {-12, -33, 7}
}
, {{35, 18, -50}
, {-6, 50, 49}
, {-43, 10, 48}
, {38, 2, -23}
, {-47, 20, 44}
, {-5, -11, -35}
, {27, 10, -3}
, {-10, -9, -34}
, {8, 28, 0}
, {47, 16, -2}
, {7, 9, -19}
, {-34, 31, -35}
, {-11, 5, -38}
, {4, 43, 17}
, {61, 36, 38}
, {27, 31, -26}
, {28, 30, 29}
, {-46, 25, 14}
, {11, -9, 10}
, {37, 25, -28}
, {11, 31, 6}
, {-41, -29, 26}
, {-37, 34, 46}
, {25, -43, 17}
, {-37, -38, -21}
, {35, 7, -29}
, {-32, 45, 67}
, {30, 3, 50}
, {14, 0, 45}
, {51, -29, -34}
, {50, 43, 51}
, {54, -1, -9}
, {20, -46, -34}
, {54, 41, -10}
, {-3, -25, 56}
, {-30, -42, -1}
, {-6, -41, -23}
, {10, -41, 2}
, {-10, 28, -55}
, {-20, 29, -25}
, {36, 40, -6}
, {16, 15, 3}
, {2, 8, -31}
, {-27, -18, -17}
, {42, 42, -37}
, {-44, 40, -16}
, {27, 5, 47}
, {20, 9, 27}
, {4, -36, -4}
, {57, 50, 20}
, {-23, 1, -20}
, {44, 8, 46}
, {-37, 57, 22}
, {14, -44, -37}
, {-9, -33, 13}
, {-7, -7, 37}
, {-45, 35, 25}
, {6, 31, 28}
, {-35, 5, 34}
, {-36, 40, -37}
, {44, 37, -18}
, {-50, -27, -28}
, {0, 28, 44}
, {18, 53, 22}
}
, {{-55, -26, -48}
, {5, 47, 39}
, {-24, 17, 4}
, {-4, -41, 17}
, {-6, -35, -45}
, {-3, -8, 30}
, {54, 54, 56}
, {33, 20, -14}
, {-20, -27, -31}
, {43, 61, 41}
, {45, -19, -47}
, {5, 36, -46}
, {-8, -8, 57}
, {-40, -16, 35}
, {-32, -45, -50}
, {36, 28, -11}
, {-41, -1, 24}
, {-47, 29, 6}
, {-40, 8, 11}
, {-46, -61, 15}
, {59, -9, 2}
, {-17, -2, -33}
, {46, 56, 56}
, {-27, -17, 22}
, {52, -31, -1}
, {51, -12, 45}
, {-12, 27, 45}
, {2, 46, 28}
, {18, 37, 10}
, {-3, 39, -43}
, {47, -11, -43}
, {58, -11, 37}
, {19, 6, -2}
, {8, 20, 20}
, {-6, -3, 25}
, {-2, 42, 60}
, {-24, 15, -14}
, {-45, -1, -2}
, {-9, -26, -31}
, {-18, 37, 44}
, {-7, -20, 45}
, {-23, -18, -25}
, {-40, 24, -8}
, {-12, 0, 11}
, {50, 8, -27}
, {-32, -2, -3}
, {16, 55, -11}
, {-27, 11, 30}
, {0, 47, 51}
, {-1, 16, 25}
, {-5, -25, 5}
, {41, -34, 5}
, {-10, -22, -44}
, {-42, 33, -42}
, {46, 35, 30}
, {23, 0, 51}
, {10, 19, 19}
, {-60, -68, 36}
, {-5, -42, -19}
, {-46, 34, -29}
, {0, -15, -44}
, {-9, -48, 6}
, {-5, -14, -22}
, {43, -8, -24}
}
, {{19, -15, 7}
, {-39, -49, -7}
, {21, 40, 6}
, {29, -2, 20}
, {-4, 51, 12}
, {-29, 24, 18}
, {35, -31, -5}
, {26, 33, 5}
, {27, 34, -32}
, {-31, 29, -39}
, {-44, -24, -9}
, {30, 8, 8}
, {-35, 52, 31}
, {-23, -51, 11}
, {-23, 23, -37}
, {60, 17, 39}
, {50, 26, 8}
, {-21, -51, 5}
, {55, -35, 24}
, {-1, -53, 8}
, {46, -38, 49}
, {-5, -47, -37}
, {-20, 22, -2}
, {44, 55, 50}
, {30, 21, -44}
, {-29, 6, 47}
, {34, -42, -33}
, {-3, -14, -28}
, {36, 0, 11}
, {46, 44, -35}
, {25, -40, 2}
, {-19, 50, 54}
, {40, -34, -16}
, {31, 44, 62}
, {-19, 49, -35}
, {-3, 23, 25}
, {13, -24, -51}
, {-10, -15, 26}
, {-11, 2, -2}
, {32, -18, -27}
, {-3, -48, -20}
, {-20, 4, 31}
, {35, 39, 45}
, {-4, 3, -24}
, {-35, 51, 34}
, {-19, 6, 14}
, {20, 62, 64}
, {55, 2, 13}
, {-18, -48, -46}
, {0, -16, -13}
, {-40, 12, 35}
, {-18, -33, 32}
, {36, -45, -8}
, {40, -50, 51}
, {15, -4, 9}
, {-34, 58, -3}
, {-37, -54, 26}
, {18, 30, -17}
, {33, -46, 7}
, {42, -10, 40}
, {-46, -5, -25}
, {23, -21, -22}
, {36, -49, -12}
, {-31, -4, 19}
}
, {{0, -49, -21}
, {24, 6, -50}
, {-2, -28, 35}
, {37, 17, 16}
, {1, -40, 9}
, {-25, 49, 0}
, {55, 50, -5}
, {1, -16, -28}
, {-34, -2, 51}
, {11, 2, 11}
, {8, 21, -13}
, {-27, 53, -25}
, {49, -11, 41}
, {-47, -2, -46}
, {22, 31, 38}
, {-45, 52, 35}
, {-5, 10, -21}
, {0, -47, 19}
, {14, 37, 43}
, {-4, 54, 20}
, {-37, -7, 26}
, {-28, 20, 30}
, {64, -11, -43}
, {23, -30, -36}
, {-28, 52, -12}
, {67, -12, 3}
, {-4, 22, -29}
, {46, 44, -56}
, {-7, 30, 36}
, {59, 20, -44}
, {3, -34, -43}
, {15, 22, -44}
, {-5, -4, 35}
, {55, 39, 39}
, {25, 44, 31}
, {38, 57, 12}
, {-26, 40, 36}
, {10, 4, -4}
, {50, -5, -7}
, {-4, -29, -20}
, {-1, 34, -20}
, {-24, 36, -4}
, {25, -39, 49}
, {-33, -36, 21}
, {0, 18, -29}
, {24, -34, 14}
, {4, 7, -18}
, {-24, 47, -32}
, {-20, 40, 31}
, {49, -34, -25}
, {28, -45, 7}
, {59, -5, -30}
, {19, -26, 2}
, {-24, 26, -48}
, {-28, -6, -1}
, {-30, 40, -32}
, {-1, -47, -49}
, {33, 0, -31}
, {-11, 16, -25}
, {43, -45, -18}
, {-12, 15, 31}
, {-59, -41, -8}
, {33, 6, 33}
, {7, 10, 37}
}
, {{20, -52, -43}
, {-11, -45, -32}
, {5, -11, 37}
, {41, -5, -11}
, {11, 41, -10}
, {24, -10, 39}
, {-18, 4, -40}
, {-45, 6, -2}
, {-36, 52, -36}
, {11, 54, 7}
, {26, -27, -34}
, {43, -17, -26}
, {47, 56, -15}
, {6, 14, -10}
, {-1, 11, 49}
, {-46, 31, 48}
, {12, 5, -39}
, {7, -14, -23}
, {-32, -17, 8}
, {42, -1, 10}
, {42, 31, -7}
, {18, -8, -36}
, {-43, 33, 8}
, {-4, 47, -51}
, {41, 0, -50}
, {-39, 46, 5}
, {46, 1, -7}
, {13, 18, -8}
, {50, -16, 1}
, {45, -17, 11}
, {6, -31, 49}
, {-8, -20, -20}
, {31, 1, -24}
, {41, -46, 62}
, {18, -23, 25}
, {-51, -27, 25}
, {48, -25, 38}
, {32, -46, -17}
, {17, -42, -23}
, {22, 1, 33}
, {53, 31, 21}
, {48, 0, 15}
, {-47, -24, -32}
, {-6, -40, -27}
, {-20, 39, -21}
, {-3, 6, 36}
, {35, 16, -62}
, {-11, -27, 37}
, {-7, 10, 48}
, {31, 22, 20}
, {-25, -21, -34}
, {-5, 37, 12}
, {65, 65, -36}
, {-44, -51, -6}
, {-20, -35, 40}
, {47, -12, -47}
, {32, -21, -40}
, {-36, -18, -29}
, {49, 0, -26}
, {-24, 7, -42}
, {12, 56, 24}
, {20, 8, -12}
, {-53, 9, 28}
, {-31, -34, 3}
}
, {{58, -16, 35}
, {-41, -14, 1}
, {-4, 32, 0}
, {-20, -16, -25}
, {38, -11, -11}
, {-25, 8, -28}
, {20, 31, 6}
, {-10, 33, -22}
, {40, -28, -35}
, {24, 23, -13}
, {3, 55, 0}
, {35, -18, -42}
, {28, -20, -5}
, {14, 49, 47}
, {26, 24, 5}
, {4, -35, -29}
, {39, -19, 42}
, {-35, 5, 34}
, {-54, 35, -30}
, {-27, 8, 43}
, {32, -11, -45}
, {27, 33, 26}
, {26, -9, 47}
, {41, -53, 28}
, {27, -42, 36}
, {28, 17, -57}
, {-3, -30, 1}
, {-43, -13, -4}
, {-55, 44, 32}
, {-9, -43, 4}
, {-28, -33, -9}
, {0, 4, -36}
, {-11, -45, -34}
, {22, -47, 13}
, {14, 38, 8}
, {35, -23, 42}
, {36, 53, 2}
, {26, 10, -3}
, {-35, -40, 35}
, {-42, 46, -27}
, {47, -35, -19}
, {-29, 32, 42}
, {14, -49, -45}
, {3, 36, 14}
, {50, 34, -9}
, {51, -9, 21}
, {3, -36, 36}
, {-16, 5, 47}
, {10, 14, -17}
, {-40, 1, -25}
, {-5, -32, 12}
, {-12, -48, -5}
, {-23, 22, 40}
, {5, 37, -12}
, {52, 35, 7}
, {-38, -24, 14}
, {58, 58, 33}
, {-8, 38, 41}
, {-15, -45, 49}
, {-60, -64, -49}
, {13, 35, 48}
, {-11, 21, 50}
, {7, 23, 45}
, {27, 4, 19}
}
, {{13, -35, -15}
, {28, -17, -16}
, {34, 0, -37}
, {21, 9, -26}
, {32, 33, 6}
, {57, 19, 47}
, {-37, -46, -21}
, {-7, -1, 46}
, {1, 36, 1}
, {18, -33, -4}
, {-39, 0, -5}
, {33, -35, -11}
, {26, -40, -37}
, {21, 24, -33}
, {0, 44, -20}
, {19, 1, -30}
, {18, -30, 14}
, {52, 15, -31}
, {14, -54, -2}
, {-53, -50, -54}
, {28, -19, -28}
, {-35, 49, 55}
, {26, -48, -20}
, {-28, 12, 30}
, {-1, -1, 22}
, {-6, 0, -7}
, {-56, 3, -64}
, {19, 19, 15}
, {52, 30, -48}
, {19, 24, 50}
, {-35, -22, -5}
, {-11, -53, 48}
, {51, -19, 14}
, {36, 16, -61}
, {12, -18, -27}
, {-48, -34, 6}
, {3, -2, -35}
, {49, -5, 25}
, {-6, -25, -38}
, {36, -30, -9}
, {22, 0, 13}
, {-35, -19, 24}
, {-33, 13, -27}
, {25, -36, -7}
, {53, 42, 49}
, {-30, 47, 5}
, {18, 25, 6}
, {59, 1, -5}
, {-10, 7, 0}
, {-40, -36, 10}
, {-48, -23, 33}
, {41, 7, 44}
, {22, -59, -40}
, {2, -29, -33}
, {5, -40, -28}
, {-28, -20, 31}
, {19, -27, 17}
, {28, -8, -24}
, {-18, 16, -38}
, {-3, 49, -24}
, {29, -31, 35}
, {35, 56, -6}
, {50, 44, 9}
, {36, -11, 34}
}
, {{-25, 26, -44}
, {-7, -29, 32}
, {55, 46, 24}
, {-35, -53, 38}
, {-18, -45, -33}
, {21, -10, 22}
, {-17, -2, 27}
, {-34, -1, -19}
, {36, 62, 7}
, {-47, -33, 47}
, {26, 15, 16}
, {-1, 14, -15}
, {7, 10, 36}
, {-39, -18, 47}
, {45, 56, 0}
, {11, -20, 51}
, {-19, 53, -31}
, {43, 22, -17}
, {19, -45, 23}
, {47, -13, 41}
, {49, -14, 42}
, {28, -33, -45}
, {-34, -3, 58}
, {17, 57, -21}
, {-51, 29, 2}
, {35, -43, 53}
, {32, 5, 30}
, {43, -38, -29}
, {6, -5, -31}
, {-11, -47, 13}
, {55, -38, 27}
, {56, -16, 28}
, {44, -3, 12}
, {12, 38, 10}
, {-21, -20, 32}
, {27, 14, 52}
, {47, 9, -23}
, {-7, -11, 41}
, {-7, -38, 8}
, {-55, 34, 22}
, {25, 53, 21}
, {-26, -20, -25}
, {-37, 8, 20}
, {60, 55, 3}
, {-13, -21, -32}
, {-10, -30, -22}
, {31, 28, -42}
, {-23, 13, 30}
, {14, 27, -7}
, {32, 32, 24}
, {20, -5, 54}
, {-29, 33, 19}
, {-19, -14, 12}
, {-38, -2, -39}
, {-35, 3, -28}
, {-26, 41, 38}
, {6, 33, -53}
, {-28, 2, 13}
, {-20, 44, 6}
, {52, -40, 23}
, {4, 29, -5}
, {-22, -26, -5}
, {42, -33, 40}
, {-36, 25, -35}
}
, {{-23, 3, -39}
, {-40, -23, 37}
, {34, -33, 31}
, {-26, 8, 1}
, {23, -42, -3}
, {26, 14, 40}
, {11, -31, -58}
, {53, 36, 38}
, {-40, -36, 38}
, {-40, 0, -48}
, {-2, 8, -42}
, {-48, -34, 37}
, {-46, 14, -31}
, {-33, -58, -25}
, {30, -43, 18}
, {25, -52, 1}
, {-8, -11, 29}
, {-11, -35, 8}
, {-17, 18, 7}
, {-25, -23, 49}
, {19, -33, 52}
, {53, -36, -49}
, {-47, 27, 18}
, {-43, 52, -40}
, {29, -40, 38}
, {23, 11, -50}
, {34, 52, -29}
, {-12, -23, -18}
, {-48, -42, 41}
, {-16, -54, -51}
, {-26, 51, -3}
, {2, -28, -25}
, {9, 43, -51}
, {-24, -2, 68}
, {0, 28, 38}
, {49, -1, 31}
, {14, -45, -16}
, {22, -3, 29}
, {-4, -5, 35}
, {-2, -50, 24}
, {38, 15, 35}
, {-41, 43, -3}
, {-50, -60, -58}
, {1, 18, -41}
, {2, -23, -35}
, {4, -3, 5}
, {-37, 52, -12}
, {7, 47, 24}
, {-11, 35, -29}
, {20, 8, -29}
, {18, -19, -46}
, {44, 0, 35}
, {-3, -34, -38}
, {-33, -10, 42}
, {-21, 23, -40}
, {-6, -54, -37}
, {2, 11, 14}
, {69, -37, -35}
, {36, -12, 6}
, {-1, -52, 14}
, {31, 21, -27}
, {-38, -25, -8}
, {42, 9, -32}
, {-11, -13, 46}
}
, {{56, 27, 54}
, {13, -7, -15}
, {-9, 27, -3}
, {23, 13, -3}
, {4, -47, -19}
, {-51, 41, -2}
, {-17, -56, 34}
, {17, 38, 27}
, {8, -59, 7}
, {-49, 47, 57}
, {32, -23, 40}
, {0, -37, 25}
, {38, 2, 41}
, {-28, 29, 59}
, {-2, -2, 3}
, {-43, -1, -28}
, {-39, 38, -10}
, {-26, 52, 27}
, {-19, -63, -2}
, {-33, -6, 48}
, {-3, 45, 38}
, {52, 34, -5}
, {-15, 48, -23}
, {45, 45, -21}
, {-22, 6, -34}
, {-41, 6, 9}
, {-6, 16, 54}
, {31, 13, 41}
, {-46, -31, 15}
, {35, -48, -21}
, {-44, -40, 51}
, {43, -49, -45}
, {42, -5, 25}
, {-25, 18, -1}
, {30, 30, -39}
, {33, -32, 20}
, {-17, 63, 58}
, {28, -44, 30}
, {-19, 41, 0}
, {-32, -7, 21}
, {30, 41, 52}
, {39, 11, -31}
, {42, 44, 12}
, {42, -18, -23}
, {38, 47, -42}
, {28, 4, 9}
, {-3, 45, 11}
, {-39, -55, 27}
, {19, -45, -44}
, {-24, -32, -64}
, {45, 21, 46}
, {31, -23, 34}
, {5, 48, -44}
, {21, -1, 34}
, {-4, 10, 57}
, {-61, 20, -19}
, {19, -25, 6}
, {29, 22, 53}
, {3, 35, 37}
, {-14, -45, -43}
, {37, 6, 65}
, {-22, 6, -42}
, {4, 37, 1}
, {-4, 48, -32}
}
, {{36, 10, -45}
, {-27, 10, 41}
, {48, -32, -53}
, {-32, -15, -11}
, {-39, 27, 20}
, {-32, -3, -6}
, {14, 0, 23}
, {49, 40, -2}
, {40, 24, -19}
, {35, -2, -3}
, {-2, 11, -46}
, {-36, -21, -10}
, {29, -15, 19}
, {3, 19, 59}
, {-29, 22, 10}
, {16, 35, -7}
, {60, -45, -36}
, {50, 5, 2}
, {-30, -18, -50}
, {-23, -7, -14}
, {36, 16, -16}
, {-27, -36, 40}
, {-17, 52, 35}
, {-3, 47, -8}
, {-19, 47, -11}
, {30, -48, 6}
, {-16, 3, 34}
, {-15, -4, 2}
, {-19, -2, 56}
, {17, -35, -6}
, {17, -45, -27}
, {-16, -40, 20}
, {-10, 32, 42}
, {-41, -28, 11}
, {-4, -62, -13}
, {31, -55, 29}
, {8, 44, -1}
, {-11, -21, 43}
, {-24, 24, 18}
, {28, -23, -46}
, {14, -25, 35}
, {28, 7, -39}
, {20, -39, 35}
, {-27, 23, 37}
, {47, -32, -19}
, {44, 27, 32}
, {-14, -28, -42}
, {-9, 34, 31}
, {52, 5, -49}
, {19, 10, -27}
, {45, -61, 46}
, {40, -40, -10}
, {57, -4, -42}
, {-6, -47, -18}
, {45, 40, 23}
, {11, 21, -41}
, {24, -17, -16}
, {-45, 24, -16}
, {-8, -18, 4}
, {57, 30, 37}
, {-40, 61, -31}
, {38, 9, -27}
, {-11, 28, 33}
, {37, -3, 30}
}
, {{34, 38, 17}
, {-51, -45, 18}
, {6, 30, -44}
, {-57, 16, -12}
, {31, -42, -42}
, {-4, 48, 27}
, {-10, 20, -15}
, {-35, -48, -27}
, {44, -33, -28}
, {-17, -32, 47}
, {-31, 13, -27}
, {-36, -17, 27}
, {38, -35, -33}
, {33, -62, 18}
, {-16, -1, -42}
, {57, 13, 34}
, {61, -28, 57}
, {-43, -38, -18}
, {15, -14, 61}
, {-66, 21, -55}
, {-32, 17, -3}
, {-40, -31, -13}
, {11, -18, 27}
, {-48, 35, -42}
, {6, 31, 44}
, {36, -25, -7}
, {-41, -31, 12}
, {31, -12, -37}
, {-1, -1, 61}
, {49, -29, -20}
, {-13, 24, 22}
, {16, 3, 38}
, {-1, 19, -13}
, {17, 36, 45}
, {41, 16, 56}
, {30, 29, 33}
, {-40, 5, 19}
, {30, 39, 53}
, {-13, 29, -1}
, {-3, 41, -44}
, {-7, 46, 34}
, {-3, 9, 17}
, {-28, 43, 35}
, {-49, -60, -8}
, {14, 46, 19}
, {-8, 31, -8}
, {13, 15, 23}
, {2, 43, -42}
, {-43, 32, 48}
, {15, -11, 28}
, {-16, -36, 11}
, {34, -17, -31}
, {33, -39, -18}
, {-35, 25, 15}
, {-50, -10, 38}
, {47, -1, -24}
, {34, -30, 9}
, {-39, 42, 26}
, {-35, -19, 15}
, {7, 35, 51}
, {-51, -65, -49}
, {47, -47, -25}
, {-19, 26, 9}
, {10, 31, -15}
}
, {{6, -39, 33}
, {10, -22, 47}
, {41, 34, 52}
, {-31, 39, 32}
, {0, 28, 31}
, {-2, 21, 45}
, {2, 44, 42}
, {37, 0, 1}
, {-21, 1, -35}
, {19, 33, 38}
, {-40, -19, -45}
, {7, -49, -50}
, {21, 38, -36}
, {-30, -12, 1}
, {-5, 18, -39}
, {11, -32, -40}
, {-14, 0, 33}
, {20, -40, -39}
, {-34, -24, 26}
, {-7, -11, 47}
, {-17, 30, 26}
, {14, -21, 10}
, {39, -29, -37}
, {4, 4, 23}
, {-43, 29, 1}
, {36, 14, 50}
, {-29, -22, 2}
, {-11, 18, 33}
, {17, -37, -7}
, {48, -6, 0}
, {50, -28, -16}
, {17, -19, -43}
, {-35, -46, -26}
, {20, 40, -19}
, {11, -4, 43}
, {-44, 41, 36}
, {-41, -12, -21}
, {-42, 48, -44}
, {-21, 18, 12}
, {1, -10, 41}
, {15, 9, 35}
, {41, -1, 45}
, {24, -40, 14}
, {-14, -1, -37}
, {17, -53, 0}
, {-48, -59, -12}
, {36, 28, -33}
, {22, -35, 33}
, {-22, 31, -22}
, {-39, -2, 42}
, {41, 39, -20}
, {15, -20, 33}
, {-10, 30, -4}
, {-27, 37, 48}
, {-24, -17, -16}
, {28, 11, 6}
, {14, -4, 13}
, {-71, 33, 8}
, {11, 40, -36}
, {25, -1, -42}
, {15, -59, -26}
, {30, -18, 4}
, {30, -25, -5}
, {-48, 6, 40}
}
, {{-5, -20, -9}
, {4, -29, -5}
, {-8, -8, 6}
, {18, -55, 12}
, {36, 46, -44}
, {36, -6, -39}
, {-43, -41, -8}
, {-17, 24, 19}
, {58, -21, -17}
, {-37, -22, 34}
, {49, -1, 11}
, {46, 11, 10}
, {-47, -49, -50}
, {6, -22, 30}
, {-39, -57, 47}
, {28, 61, 63}
, {48, -32, -3}
, {-17, 15, 63}
, {-26, 32, -26}
, {39, -64, -21}
, {-6, 2, 23}
, {-32, 1, -16}
, {20, 20, -14}
, {-52, -44, 29}
, {-49, 26, 16}
, {34, -15, 13}
, {45, -33, -4}
, {-29, 17, 51}
, {21, -8, 41}
, {-24, -4, 11}
, {-43, -44, -42}
, {15, -22, -5}
, {-3, 46, -15}
, {-29, 26, 9}
, {34, 30, 13}
, {-4, -7, 51}
, {-50, -42, 0}
, {3, -42, -1}
, {57, -7, 58}
, {-45, 27, -19}
, {23, 30, -38}
, {17, 24, -43}
, {3, 32, 9}
, {-3, 2, 21}
, {14, 0, -27}
, {8, -44, -36}
, {22, -32, -41}
, {13, 55, 57}
, {43, 24, -18}
, {0, 44, -45}
, {50, 0, 20}
, {-45, 11, 31}
, {-18, -19, 36}
, {-3, -18, 32}
, {-24, 23, 17}
, {47, -45, 35}
, {-21, -19, -16}
, {-22, 46, 31}
, {-42, -44, 44}
, {-25, 56, 16}
, {14, 7, 39}
, {12, 40, 61}
, {45, -40, -39}
, {45, -34, -29}
}
, {{-41, 25, -2}
, {6, -3, -4}
, {8, -11, -4}
, {-5, 21, -8}
, {-24, 34, 28}
, {14, 28, 12}
, {22, -18, -24}
, {53, 3, 7}
, {5, 32, -39}
, {-18, -18, 26}
, {33, 18, 10}
, {10, 19, 17}
, {34, 4, -27}
, {39, 44, 6}
, {-11, -51, -10}
, {31, 3, -1}
, {-1, -28, -46}
, {30, -25, 32}
, {-26, -23, 7}
, {-53, 21, -9}
, {17, 13, -31}
, {-50, 1, 49}
, {22, 5, -48}
, {-53, -47, 14}
, {43, 8, -45}
, {24, -32, 38}
, {-2, 58, -17}
, {14, -4, -49}
, {38, 45, 3}
, {-9, 48, -31}
, {-40, -32, 8}
, {1, -12, -53}
, {8, -41, 47}
, {35, 5, 16}
, {29, 4, -43}
, {-8, -33, -44}
, {45, -12, 21}
, {44, -22, 48}
, {29, -11, -8}
, {4, 11, -27}
, {5, -30, 41}
, {48, -27, 36}
, {-28, -55, 42}
, {36, 5, -36}
, {36, 19, 28}
, {-1, -29, 40}
, {-17, -16, -7}
, {33, 45, -17}
, {-13, -16, -35}
, {6, 8, -28}
, {10, -45, -27}
, {-4, -6, 5}
, {-37, -55, 5}
, {-7, 43, -37}
, {-19, -14, 37}
, {-49, 25, -7}
, {-45, 11, -16}
, {13, -51, 11}
, {43, 51, -32}
, {-66, -32, -13}
, {-13, -20, 39}
, {-20, -6, 20}
, {-36, -1, -24}
, {30, -41, 49}
}
, {{-15, 0, 16}
, {-2, -15, 41}
, {-42, 34, -31}
, {-4, 34, 23}
, {9, 5, 21}
, {-31, -31, 54}
, {13, -57, -42}
, {33, -2, -1}
, {-4, 49, 23}
, {25, 11, 21}
, {-31, 21, 45}
, {-48, -30, 22}
, {24, -21, 18}
, {-66, -40, -31}
, {-43, 6, 38}
, {23, 15, 21}
, {8, 40, -44}
, {-25, 0, -18}
, {-28, 23, -44}
, {-29, -61, 36}
, {20, -25, -61}
, {-1, 11, 3}
, {27, -4, -55}
, {25, -4, -49}
, {-17, -26, -44}
, {-40, 21, 3}
, {35, -31, -40}
, {-23, -9, 9}
, {51, 45, 31}
, {22, 44, 10}
, {-43, -26, -19}
, {-49, 18, 19}
, {35, 45, 3}
, {-28, -11, -36}
, {-25, -44, -55}
, {-35, -18, -12}
, {-11, -26, -11}
, {1, -12, 41}
, {-46, -29, 26}
, {35, -22, 48}
, {24, -52, -51}
, {16, -19, -27}
, {-20, 49, 11}
, {-57, -3, 29}
, {-38, -1, 39}
, {-13, 15, -8}
, {-41, -38, -39}
, {-1, 54, -38}
, {35, 34, 27}
, {-29, 0, 11}
, {36, -12, 46}
, {17, 4, 50}
, {-64, -50, -61}
, {-36, 41, -51}
, {18, 25, -3}
, {-17, 35, 12}
, {61, -33, -9}
, {38, 15, 23}
, {17, -29, 51}
, {42, 26, 13}
, {-44, -52, 29}
, {-33, 51, 58}
, {-19, 42, 26}
, {44, 54, 30}
}
, {{18, 4, 13}
, {-5, -23, 8}
, {14, -13, 49}
, {-28, -11, 4}
, {50, -2, -15}
, {13, 47, -51}
, {45, -41, 25}
, {22, -28, 36}
, {-33, -20, 33}
, {42, 42, -42}
, {-45, 22, 41}
, {-18, 7, 3}
, {48, 31, 31}
, {-33, 52, 50}
, {5, 64, 41}
, {13, -29, 20}
, {-18, 51, -30}
, {21, -11, 42}
, {12, 13, 41}
, {51, 0, 20}
, {-27, 19, 9}
, {34, 6, 35}
, {36, 30, -47}
, {11, -40, 50}
, {-13, 52, -13}
, {-3, -45, -31}
, {25, -43, -43}
, {15, 26, -30}
, {1, -18, 39}
, {-30, -36, -36}
, {-6, -5, -46}
, {-2, -46, 29}
, {1, 28, -21}
, {-5, 31, 22}
, {-62, -19, 35}
, {-55, -19, -41}
, {39, 5, 8}
, {-8, -9, -2}
, {-48, 42, 7}
, {-22, 39, 10}
, {46, 34, -54}
, {16, 54, 39}
, {32, -38, -45}
, {-19, 14, 46}
, {-15, 15, 1}
, {58, -14, 29}
, {1, -16, 27}
, {-5, -14, 43}
, {0, -35, 46}
, {51, 20, -41}
, {59, 2, 57}
, {-28, 16, -6}
, {46, 47, -40}
, {-50, 9, 24}
, {45, -39, -21}
, {-34, 36, -17}
, {22, -7, 25}
, {7, 18, 50}
, {-38, -35, 11}
, {13, 34, -26}
, {32, 56, -21}
, {-23, 43, -23}
, {-1, 3, 0}
, {42, -32, -42}
}
, {{51, 11, -10}
, {25, -30, -25}
, {35, 56, -29}
, {-36, 38, 9}
, {33, -11, 27}
, {20, -27, -22}
, {28, -29, 33}
, {57, 53, -39}
, {20, -9, 17}
, {-2, 46, -21}
, {0, 35, -39}
, {-14, -49, 1}
, {5, 21, -46}
, {57, 0, 8}
, {49, -29, -12}
, {11, -7, -11}
, {-43, -25, 6}
, {41, 28, -15}
, {33, 15, -2}
, {19, 58, 27}
, {24, 6, 53}
, {4, 25, -54}
, {24, 65, -13}
, {-31, -11, -8}
, {24, -45, -30}
, {-32, -18, 53}
, {17, -20, 18}
, {-19, -31, -10}
, {39, 43, -23}
, {42, 3, 40}
, {2, 22, 9}
, {-31, -23, -21}
, {-35, -13, 6}
, {10, 9, -18}
, {37, 26, -33}
, {-52, 27, -1}
, {-43, -4, -38}
, {-57, 27, -47}
, {32, 28, 36}
, {15, 8, 9}
, {-7, 61, -10}
, {46, -33, 26}
, {23, -27, 35}
, {27, -12, 14}
, {-38, -18, 14}
, {7, 19, 10}
, {-55, -66, 3}
, {-2, -40, 26}
, {37, -8, -40}
, {-2, 45, 7}
, {-39, 45, -1}
, {22, -37, 7}
, {34, 13, -19}
, {-7, 39, -6}
, {27, 4, -64}
, {47, 40, -10}
, {23, 18, 23}
, {19, 38, 0}
, {-40, -48, 45}
, {-41, 14, 20}
, {-40, 28, 32}
, {-43, 19, -43}
, {34, 25, 32}
, {21, -21, 10}
}
, {{-22, 35, 18}
, {-32, 29, 59}
, {-52, -47, 7}
, {-7, -20, 5}
, {15, -14, 29}
, {31, -14, 11}
, {-12, -44, -21}
, {-46, 46, -11}
, {16, 8, 22}
, {-44, -45, 31}
, {-10, 15, 61}
, {39, 40, 19}
, {-10, 27, -4}
, {8, 0, -25}
, {-31, 33, 34}
, {8, 49, 56}
, {37, 9, 53}
, {-40, -34, -14}
, {45, 64, 58}
, {28, 32, -12}
, {-56, 13, 22}
, {-18, -12, 67}
, {28, 4, 36}
, {-58, -19, -50}
, {18, -28, -19}
, {0, 19, 46}
, {-56, 34, 9}
, {-20, -19, 21}
, {3, -2, -2}
, {-38, 29, -1}
, {-59, -34, -2}
, {-38, -34, -16}
, {2, 60, 15}
, {-24, 36, 44}
, {-37, -15, -63}
, {0, 47, 27}
, {41, 14, 11}
, {17, -2, -8}
, {-2, 37, 39}
, {20, 30, 33}
, {-52, 8, -5}
, {-19, 2, 53}
, {52, 65, 2}
, {-35, 0, -21}
, {-47, -25, -12}
, {-48, -15, 11}
, {-22, 26, -40}
, {46, 14, -16}
, {-5, -38, 36}
, {32, -9, 67}
, {-34, -36, 30}
, {43, -37, 20}
, {-6, 11, 6}
, {-33, -34, -22}
, {-1, -4, 10}
, {-33, 48, 48}
, {12, -11, 1}
, {-20, -13, 52}
, {7, 45, 2}
, {64, -2, -2}
, {13, 24, 28}
, {2, -23, 42}
, {4, -21, -2}
, {-38, 53, 54}
}
, {{21, 41, 27}
, {-2, -19, 46}
, {10, -12, 14}
, {-19, 9, -49}
, {26, -2, -14}
, {-40, -20, -16}
, {50, 0, 37}
, {-19, 7, -13}
, {0, -38, 46}
, {-16, 31, -30}
, {-48, 24, 31}
, {21, -14, -15}
, {-14, 34, 2}
, {55, 27, 25}
, {-33, 9, -24}
, {-29, 2, -1}
, {-11, -22, 63}
, {-35, -14, 19}
, {37, 27, 0}
, {-7, 57, -4}
, {29, 41, 66}
, {41, -14, -7}
, {-18, -5, -13}
, {-35, 55, -18}
, {9, -22, -26}
, {14, 19, 40}
, {-37, 41, -36}
, {6, -29, -11}
, {32, 43, -45}
, {-34, -21, -17}
, {44, -7, -43}
, {-29, 25, 41}
, {-14, 39, -29}
, {-8, 2, 28}
, {53, -30, 3}
, {-35, -37, -7}
, {-38, -20, 21}
, {33, 45, -33}
, {-15, 20, 39}
, {33, -53, 20}
, {12, -17, -42}
, {20, -10, 15}
, {8, -50, 26}
, {-30, -2, -15}
, {-28, -37, -10}
, {-20, -31, -45}
, {6, 30, 37}
, {10, -35, 29}
, {-3, 50, 36}
, {60, 58, 4}
, {-16, 42, 17}
, {-22, -43, 33}
, {56, 43, 30}
, {7, 4, 45}
, {-13, 48, -31}
, {20, 49, -39}
, {10, 10, 39}
, {19, 26, -33}
, {-34, 36, 32}
, {26, -43, -23}
, {-29, 4, -33}
, {-7, -18, 3}
, {-24, -42, 28}
, {51, 22, -31}
}
, {{-33, 23, -45}
, {-34, -36, 53}
, {35, 24, 15}
, {7, 9, -16}
, {33, -20, -18}
, {-23, 7, 4}
, {16, 38, -30}
, {48, -4, 6}
, {-34, -19, -5}
, {31, 53, -30}
, {-24, 10, -37}
, {19, 19, -28}
, {-1, 27, -8}
, {-36, 40, -4}
, {-5, -23, -5}
, {22, -34, 8}
, {-37, -43, -31}
, {39, -18, -37}
, {-29, 4, 52}
, {-35, -22, 50}
, {55, 37, 48}
, {-15, -12, 16}
, {-34, 51, 11}
, {31, -23, 8}
, {-13, -20, 35}
, {-36, 55, 7}
, {53, -17, -21}
, {-57, -5, 4}
, {-20, -41, 13}
, {46, 48, -26}
, {25, 10, 10}
, {56, 17, 12}
, {-14, 0, 3}
, {38, -21, -3}
, {14, 49, 14}
, {23, 25, 0}
, {-41, 3, -37}
, {-13, -47, 32}
, {-3, 24, 22}
, {-44, 22, 46}
, {26, 39, 43}
, {25, 50, -42}
, {-10, 23, -35}
, {44, -29, -9}
, {43, -55, -17}
, {-31, 32, -40}
, {7, 19, 14}
, {24, 32, -20}
, {5, 29, -22}
, {10, -55, 34}
, {7, 43, -37}
, {16, -9, -38}
, {1, -27, 34}
, {-36, 25, -19}
, {-34, -36, -6}
, {-9, 9, -21}
, {-40, -10, -7}
, {30, 37, 38}
, {32, 54, -4}
, {1, -52, 13}
, {-18, -49, 29}
, {41, 34, -54}
, {30, 15, -35}
, {-29, -58, -7}
}
, {{-13, 31, 40}
, {-6, -40, 33}
, {-48, 0, -37}
, {4, 38, -11}
, {40, 34, 21}
, {-55, -54, -46}
, {33, -49, -16}
, {21, -8, -14}
, {0, -40, 16}
, {-39, -48, -2}
, {-15, 35, -44}
, {-18, 42, -39}
, {-33, 9, -16}
, {-17, -27, 1}
, {18, 17, -11}
, {24, -38, -29}
, {2, 37, -55}
, {25, -52, 17}
, {-5, -22, 24}
, {41, -49, -39}
, {4, -52, 44}
, {-22, -40, -18}
, {-24, 32, -2}
, {-33, 42, 40}
, {-24, -49, 2}
, {34, 46, -12}
, {56, -4, 23}
, {23, 28, -28}
, {-42, 6, 1}
, {11, 26, -6}
, {-52, -44, -19}
, {35, -7, 14}
, {41, 22, 31}
, {-22, 41, 17}
, {-55, -21, 38}
, {20, -53, -46}
, {-33, -26, -13}
, {21, -3, -39}
, {35, -29, -45}
, {-21, -32, 33}
, {40, 9, -12}
, {-46, 4, -40}
, {20, -30, -26}
, {-38, 6, 11}
, {-7, -18, -28}
, {-40, 3, -33}
, {26, 14, 29}
, {-42, -53, -48}
, {-46, 0, -22}
, {-34, -32, -40}
, {14, 32, -2}
, {46, 0, -28}
, {8, -19, -25}
, {-35, 14, 18}
, {5, 20, 18}
, {18, 29, -33}
, {13, 36, 12}
, {39, -33, 26}
, {-50, -9, 20}
, {-45, -16, 25}
, {4, -2, 46}
, {41, 0, -18}
, {13, -9, -57}
, {-51, 25, 15}
}
, {{11, 14, -23}
, {-17, 34, -22}
, {50, 4, 33}
, {-47, 7, 6}
, {42, -47, 52}
, {46, -35, 6}
, {36, 24, 46}
, {17, -34, 27}
, {43, 24, 4}
, {-12, 49, -12}
, {34, -38, 24}
, {-27, 48, -26}
, {39, -55, 12}
, {-37, -61, -14}
, {-58, -45, -58}
, {40, 15, -35}
, {-53, 22, -18}
, {27, -12, -20}
, {43, 51, 13}
, {13, -27, 30}
, {-49, -47, -17}
, {45, 29, -31}
, {40, -45, -24}
, {48, -11, 2}
, {-2, 26, -8}
, {38, -7, -4}
, {28, -19, -35}
, {-1, -15, 20}
, {33, -23, -54}
, {41, -42, -22}
, {-9, -4, 21}
, {25, 46, 52}
, {-34, 11, -59}
, {18, -2, 25}
, {62, 32, -2}
, {27, 22, -19}
, {-47, 17, 40}
, {11, 4, -11}
, {-31, -8, -28}
, {32, -42, 24}
, {25, -19, 5}
, {34, 34, 29}
, {-43, 10, 52}
, {7, -30, 35}
, {-32, 44, -12}
, {22, 48, 34}
, {-3, 54, 50}
, {31, 48, 0}
, {-29, 50, 35}
, {51, 13, -44}
, {8, -45, -1}
, {29, -40, 38}
, {1, 23, -5}
, {14, -4, 26}
, {-13, 38, -1}
, {-21, 13, -30}
, {31, -19, -42}
, {29, -14, -48}
, {-16, -47, -47}
, {18, 26, 12}
, {16, -66, 28}
, {-47, 3, 13}
, {-1, 51, -25}
, {-10, 49, -33}
}
, {{21, -45, -34}
, {29, -11, 9}
, {-44, -23, -44}
, {45, 18, 10}
, {45, 41, 38}
, {50, 22, 12}
, {18, 44, -9}
, {-22, 0, 16}
, {17, -38, -3}
, {-37, 32, 46}
, {47, 34, 11}
, {13, -1, 12}
, {-2, -27, 41}
, {-53, 43, 10}
, {37, -55, -27}
, {-8, -30, -20}
, {-49, -39, 17}
, {-17, 2, -17}
, {-40, -39, 45}
, {18, -65, -15}
, {-49, -38, -57}
, {-1, -50, 4}
, {28, 20, 27}
, {-37, -26, 19}
, {53, 46, -35}
, {-38, 20, 10}
, {-44, 22, 57}
, {21, -36, 40}
, {-42, 23, -11}
, {1, -4, 6}
, {42, 34, -37}
, {-8, -3, -32}
, {24, 42, -42}
, {36, -36, -12}
, {49, 29, 30}
, {16, 34, -6}
, {44, 57, -7}
, {47, -26, 21}
, {-8, -30, -16}
, {-6, 40, -17}
, {19, 32, -26}
, {26, -42, 32}
, {25, -40, 29}
, {34, -7, 4}
, {19, 40, 53}
, {-35, 13, 38}
, {-9, -20, -4}
, {-26, 35, 38}
, {7, 33, 7}
, {-35, -55, 11}
, {-7, 34, 35}
, {11, -38, 57}
, {-12, 15, -42}
, {23, 29, -43}
, {15, 35, 4}
, {-14, 47, 0}
, {9, -13, 30}
, {4, 8, -54}
, {-44, 2, -4}
, {37, -26, 38}
, {8, 27, -10}
, {11, 49, 45}
, {23, 1, 37}
, {-34, 17, -22}
}
, {{6, 14, -41}
, {15, 0, 23}
, {-26, -31, 42}
, {-50, -37, 42}
, {-7, -46, 45}
, {45, 32, 5}
, {-30, -21, 14}
, {-16, 37, -3}
, {28, -15, -4}
, {-51, -34, -39}
, {28, 27, 6}
, {44, -13, 56}
, {-32, 26, -26}
, {-14, 43, 42}
, {10, 0, 43}
, {26, 40, 1}
, {0, 18, 28}
, {-44, 17, -16}
, {-10, -17, 50}
, {-48, 13, 28}
, {-51, -13, 12}
, {25, 55, 51}
, {25, 27, -38}
, {-6, 14, 6}
, {44, 13, -23}
, {-32, -30, 50}
, {32, -57, 17}
, {41, -28, 61}
, {-35, -19, -29}
, {-47, 23, 4}
, {35, 1, 23}
, {-27, 22, -11}
, {-30, -42, 47}
, {-40, -40, -19}
, {-13, 21, 40}
, {19, 33, 28}
, {-54, -43, 35}
, {0, 55, 7}
, {-1, -10, -24}
, {-22, 16, -2}
, {8, 0, -17}
, {-37, -25, 15}
, {-2, 35, 2}
, {-35, -12, -25}
, {19, 23, -8}
, {-26, 2, 36}
, {41, 21, 47}
, {13, -28, 27}
, {-49, 23, -24}
, {8, 48, 23}
, {31, -44, 18}
, {-37, -17, -19}
, {10, 38, 17}
, {18, 3, 32}
, {-52, 55, -39}
, {12, 31, -26}
, {-5, 14, 36}
, {-47, -51, 55}
, {-34, -39, 22}
, {18, -30, 54}
, {-26, -5, -8}
, {-37, 40, -30}
, {-28, 48, 51}
, {-26, -7, 11}
}
, {{33, 52, 5}
, {-34, 33, -3}
, {27, 2, -50}
, {23, 52, -10}
, {23, 42, 24}
, {46, 26, 15}
, {43, 36, 53}
, {10, 30, 14}
, {20, 66, 49}
, {12, 18, 23}
, {-23, 22, -29}
, {-4, 33, 38}
, {7, -30, 23}
, {-16, -36, 17}
, {16, 32, -42}
, {21, 68, 64}
, {30, -21, 5}
, {11, -33, -43}
, {45, 6, 41}
, {34, -1, 27}
, {-64, -32, -2}
, {-21, 18, -11}
, {-38, 44, -43}
, {-33, -13, 46}
, {-26, 22, -46}
, {-35, -26, 35}
, {-16, -46, -26}
, {51, 43, -11}
, {26, 39, 26}
, {2, -30, 59}
, {-4, -26, -18}
, {-23, 9, -27}
, {59, 28, -20}
, {-34, -42, -30}
, {-17, -47, -63}
, {48, -27, -25}
, {30, -11, -5}
, {-46, -33, 17}
, {-26, 49, -24}
, {-22, 33, -37}
, {-26, 12, -21}
, {-23, 49, -51}
, {38, 11, -32}
, {-10, -38, -11}
, {31, 12, 31}
, {-15, -35, 2}
, {-11, -59, -58}
, {22, 12, 14}
, {-21, -33, -23}
, {30, -6, 20}
, {48, -6, 11}
, {50, 18, -3}
, {-52, 2, -38}
, {50, -26, -24}
, {-36, 1, 32}
, {16, -30, -43}
, {47, -11, -6}
, {18, 24, 19}
, {-40, -14, -24}
, {12, 29, 21}
, {-9, -18, -35}
, {35, -21, 65}
, {-20, 0, -38}
, {-2, 51, 47}
}
, {{-41, 2, -29}
, {15, -35, -3}
, {-2, 8, -4}
, {34, -29, -37}
, {-5, 2, 22}
, {50, -34, 1}
, {-8, -2, -41}
, {8, 61, -25}
, {5, 44, -42}
, {17, -46, -9}
, {-16, 26, 24}
, {18, -24, -36}
, {-30, 14, -37}
, {-14, -59, 47}
, {-9, 49, 8}
, {25, 35, -51}
, {22, 12, -28}
, {26, 52, 53}
, {-34, -55, -31}
, {28, 1, 61}
, {8, 16, -21}
, {-1, 14, 47}
, {-21, 2, 2}
, {20, -18, 47}
, {-10, 20, 45}
, {20, -14, 6}
, {-21, 0, -5}
, {5, -41, -33}
, {1, -49, 30}
, {-9, 1, 48}
, {-54, 32, -47}
, {7, -34, -36}
, {26, 39, 9}
, {52, -45, 21}
, {-45, -9, 3}
, {8, -39, -4}
, {-37, -39, -36}
, {-45, 27, -1}
, {21, 16, -4}
, {-41, -41, 21}
, {13, 3, 18}
, {-53, 14, 11}
, {38, 24, 22}
, {-6, -47, -41}
, {-36, 47, 3}
, {43, 62, -10}
, {-42, -15, -5}
, {-13, -30, 17}
, {5, 18, 37}
, {-8, 5, 11}
, {-52, 45, -1}
, {38, 44, -14}
, {0, 31, -41}
, {24, 25, -41}
, {2, -12, -34}
, {25, -17, -46}
, {-2, -37, 25}
, {37, 40, 3}
, {-35, 18, 21}
, {-8, -15, -51}
, {-16, -40, 35}
, {-8, 48, 5}
, {16, -10, -9}
, {23, -13, 1}
}
, {{14, 38, 29}
, {-28, 22, -25}
, {25, 15, -39}
, {11, -24, 2}
, {-12, 21, -24}
, {1, 32, 40}
, {-5, 20, 16}
, {23, 22, -4}
, {-21, -31, 10}
, {19, -7, -9}
, {-49, 40, -39}
, {38, 17, -52}
, {-54, -39, -17}
, {30, 42, 35}
, {19, -55, 28}
, {-11, -9, -53}
, {8, -21, 36}
, {-21, -19, -48}
, {-16, 45, 6}
, {-32, -49, -36}
, {19, -40, 34}
, {27, -45, 32}
, {-14, 43, 31}
, {1, -39, 40}
, {-28, 5, -31}
, {-20, -5, -21}
, {-15, 44, 29}
, {-52, 23, -13}
, {-25, -25, -9}
, {11, -29, 9}
, {-36, 19, 30}
, {-52, -22, -20}
, {-9, -32, -5}
, {-47, -8, -46}
, {-8, -56, -2}
, {44, -7, -33}
, {-33, -7, -45}
, {16, -36, -32}
, {-42, 17, 21}
, {26, -31, 7}
, {-13, -25, 27}
, {11, -29, 43}
, {37, -56, 7}
, {-22, -54, -41}
, {-15, 8, -13}
, {-27, 29, 11}
, {-40, 41, -45}
, {15, 39, 24}
, {39, 6, -10}
, {-2, 36, 35}
, {-52, -34, 39}
, {-43, 11, -5}
, {33, 39, 26}
, {-21, 33, -13}
, {3, -33, -52}
, {40, -35, -32}
, {-13, -46, 45}
, {-15, -47, 18}
, {8, -30, 44}
, {-45, -59, -25}
, {12, -7, -14}
, {-9, -27, -41}
, {-17, 44, -30}
, {10, -15, 6}
}
, {{16, 51, 20}
, {-40, -29, 29}
, {-10, 42, -16}
, {7, -5, -44}
, {0, -49, -37}
, {-21, 29, 6}
, {-19, -11, -37}
, {30, -30, 48}
, {-45, 12, -39}
, {-32, -47, -9}
, {-27, -42, -50}
, {-42, -27, -2}
, {-50, -9, -12}
, {19, 1, -32}
, {42, -13, 6}
, {38, 46, 42}
, {-21, 7, 35}
, {14, 21, 17}
, {-18, -35, -1}
, {-23, 5, 0}
, {48, 32, -51}
, {6, -7, -43}
, {0, -16, -46}
, {-35, -23, -32}
, {-33, 22, -5}
, {-49, -38, -25}
, {-35, -3, -26}
, {-37, -6, 37}
, {5, -31, 14}
, {-20, -30, -24}
, {-16, -6, 12}
, {-8, 20, -49}
, {26, -10, -47}
, {-27, -10, -10}
, {0, -8, -30}
, {27, -2, -49}
, {-49, -35, -10}
, {-9, -33, -44}
, {-21, 37, 44}
, {8, 51, -6}
, {49, -27, 51}
, {3, 20, 5}
, {26, -21, -9}
, {50, 23, 5}
, {3, 38, -39}
, {31, 45, 17}
, {29, -26, -25}
, {10, -1, -21}
, {-23, -13, -39}
, {-29, -32, 33}
, {16, 44, 36}
, {29, 20, -40}
, {-39, 15, 46}
, {-41, 34, 32}
, {-11, -26, 39}
, {22, -10, 44}
, {2, 6, 43}
, {-6, 6, 48}
, {0, -30, -51}
, {39, 28, 0}
, {12, 0, -5}
, {11, 49, 42}
, {-35, -19, 11}
, {41, -32, 39}
}
, {{-27, 9, -29}
, {-5, -4, 23}
, {10, -2, 16}
, {-16, 0, 39}
, {-39, -44, -39}
, {-20, 33, -18}
, {-13, 30, -16}
, {37, 30, 16}
, {36, -11, 53}
, {48, -5, -21}
, {-37, 28, 10}
, {43, -40, 50}
, {10, -26, 36}
, {57, 4, 34}
, {-32, 13, 10}
, {-18, 32, 25}
, {37, 53, 15}
, {23, 42, -36}
, {-43, -25, 15}
, {-7, 2, -9}
, {55, 39, -10}
, {29, -43, -52}
, {25, 51, 51}
, {18, 49, -8}
, {16, 6, 49}
, {-50, 23, 4}
, {-15, 13, 2}
, {18, -1, -32}
, {29, 20, 20}
, {-24, 5, 51}
, {-39, -8, -35}
, {42, 49, -2}
, {-39, -17, 9}
, {3, -32, -33}
, {-9, -34, 43}
, {-48, -31, -2}
, {-17, -53, -11}
, {26, -38, 44}
, {-30, -24, -3}
, {-8, 16, 8}
, {-37, 0, 26}
, {-6, -26, -31}
, {-29, 53, -21}
, {-22, -44, 27}
, {-19, -52, 10}
, {-6, -20, -44}
, {-34, 17, -15}
, {46, 48, 20}
, {7, 46, -8}
, {26, 17, 51}
, {-23, 32, 7}
, {47, -9, -35}
, {-11, 53, 26}
, {44, 10, -52}
, {16, 27, 32}
, {25, -37, -16}
, {-32, 12, 31}
, {-34, 38, 41}
, {0, 23, 38}
, {-14, 40, 46}
, {38, 44, -44}
, {31, -48, 24}
, {-53, -50, -28}
, {-22, -23, 7}
}
, {{-36, -8, -12}
, {-27, 20, -13}
, {33, 38, 36}
, {5, -42, -22}
, {-27, -26, -15}
, {23, 24, 34}
, {41, -42, -29}
, {10, -43, 32}
, {49, 39, 6}
, {12, -4, 27}
, {0, 23, 0}
, {39, -19, -25}
, {13, -46, -39}
, {-28, -63, -29}
, {-44, -44, -36}
, {15, 62, 41}
, {1, 54, -28}
, {32, -29, 35}
, {-27, 14, 20}
, {17, -4, -45}
, {-33, 38, -29}
, {35, 7, 34}
, {-16, -4, 10}
, {-26, -12, -19}
, {59, -22, -42}
, {39, -5, -23}
, {-1, 12, -39}
, {-35, 38, 21}
, {32, 18, -10}
, {27, 12, 2}
, {7, 31, 12}
, {19, 49, -31}
, {53, 33, 32}
, {18, 36, 19}
, {52, 42, -27}
, {-13, -6, 28}
, {-49, 20, 25}
, {-12, -19, 31}
, {47, 53, 36}
, {-7, 29, 51}
, {-46, -45, -47}
, {0, 12, 3}
, {17, 24, 20}
, {-30, -21, -32}
, {21, -4, -6}
, {22, -25, -44}
, {5, -6, -20}
, {-30, 17, -25}
, {-11, 34, -9}
, {49, 25, 45}
, {44, -52, 39}
, {-46, 42, 3}
, {-10, -42, -52}
, {26, 25, -29}
, {45, 24, -58}
, {-29, 0, 24}
, {-16, 22, -20}
, {-31, 18, 38}
, {51, -12, 49}
, {-28, 55, -20}
, {-48, 17, -39}
, {-34, 26, 35}
, {44, -10, -29}
, {-9, 34, 2}
}
, {{30, 35, 48}
, {-20, -3, 1}
, {44, 29, 33}
, {31, 48, 39}
, {37, 45, -4}
, {-31, -29, -47}
, {14, 51, 18}
, {-27, 42, -21}
, {-8, 27, -54}
, {0, 29, -29}
, {-5, 0, -10}
, {30, -6, 3}
, {-30, 48, 6}
, {-4, -12, 54}
, {30, -11, 3}
, {-6, -8, -52}
, {44, 39, -47}
, {51, 35, 2}
, {-36, -14, -47}
, {41, 28, -15}
, {-20, 44, 16}
, {-11, -30, -43}
, {52, 60, 26}
, {-30, 15, -45}
, {-45, -8, 31}
, {-3, -36, 7}
, {-28, -12, -24}
, {-6, -31, 25}
, {-2, -10, -28}
, {40, -26, -16}
, {40, 47, 10}
, {10, 50, -29}
, {-6, -9, -7}
, {15, 39, 0}
, {-21, 25, 26}
, {10, 23, 18}
, {39, 28, 31}
, {-16, 44, -17}
, {-39, -22, -42}
, {-5, -37, -33}
, {-12, 36, 22}
, {-26, -24, 0}
, {28, -10, 0}
, {11, -6, 31}
, {14, -36, -2}
, {-6, -33, 20}
, {54, -17, 18}
, {41, -50, -28}
, {41, 17, -8}
, {11, 17, -32}
, {1, -30, -44}
, {-12, 34, 50}
, {20, 31, 57}
, {-44, -20, -43}
, {-24, -24, -12}
, {37, -37, 34}
, {16, -36, 0}
, {-57, 0, -8}
, {24, -47, -44}
, {-41, -44, 39}
, {-8, 0, 29}
, {21, 48, 19}
, {-56, 46, 40}
, {-41, -55, -10}
}
, {{-34, 24, 45}
, {-11, -14, -38}
, {19, 0, -18}
, {21, 26, 21}
, {-5, -8, 4}
, {47, -37, 33}
, {23, -23, -2}
, {44, -25, -32}
, {-6, 11, -6}
, {8, 0, -49}
, {9, 30, -41}
, {-43, 28, 10}
, {9, 50, -15}
, {27, -29, -25}
, {-36, 44, 55}
, {5, -30, 27}
, {-30, 43, 35}
, {25, 12, -32}
, {11, -19, -58}
, {-9, 50, 35}
, {-3, -17, 28}
, {-1, 31, 32}
, {-8, 0, 18}
, {-6, -24, 23}
, {-18, 46, 20}
, {-14, 34, 34}
, {4, 24, -49}
, {51, 7, 16}
, {-38, -46, -39}
, {-6, 10, 35}
, {23, 2, 31}
, {18, -23, -12}
, {-20, 36, 0}
, {-35, -57, -29}
, {-4, 12, -56}
, {17, -9, -37}
, {38, 35, 24}
, {-27, 37, -14}
, {10, 30, 45}
, {47, -39, -27}
, {11, 47, -30}
, {-39, -39, 40}
, {-11, 4, -22}
, {46, 4, 50}
, {-22, 13, 43}
, {21, 17, 46}
, {-68, -59, -51}
, {-15, 18, -14}
, {-14, 51, -27}
, {2, -20, 40}
, {-48, 0, -15}
, {-3, 13, 19}
, {30, 12, -9}
, {-28, 21, -32}
, {-42, -44, 35}
, {-29, 22, -40}
, {25, 1, 30}
, {33, 31, 53}
, {-36, -13, -35}
, {4, -24, -33}
, {40, -14, -13}
, {33, 4, 58}
, {23, 66, 62}
, {48, 49, 36}
}
, {{-2, 0, 47}
, {-5, 1, 26}
, {40, 40, -36}
, {-35, -51, 21}
, {15, 38, 0}
, {-4, 38, 18}
, {32, -31, 10}
, {27, -40, -25}
, {0, 19, -11}
, {7, 53, 2}
, {22, -45, 30}
, {-25, -3, 39}
, {-1, 13, 57}
, {-5, -7, 59}
, {45, -34, 49}
, {1, -30, -4}
, {34, 3, -25}
, {-51, -29, -45}
, {28, 9, -38}
, {-54, 19, 41}
, {3, 33, 61}
, {-30, -28, -17}
, {10, 23, -38}
, {14, -2, -38}
, {-14, 25, 39}
, {43, 28, 48}
, {17, 9, 30}
, {-39, -10, -15}
, {26, 28, -10}
, {19, 32, 1}
, {20, -12, 46}
, {-35, -4, -25}
, {-19, 32, -38}
, {20, 3, 46}
, {-1, -32, -32}
, {20, 5, -18}
, {20, 28, 38}
, {-11, 28, 40}
, {1, -10, -18}
, {-42, -24, 33}
, {23, -31, -24}
, {-3, -42, 6}
, {-29, 22, 37}
, {26, 4, 25}
, {-42, 23, 3}
, {-44, -29, -41}
, {12, -32, 12}
, {26, 30, -34}
, {9, -47, -19}
, {11, -24, -38}
, {5, -26, -17}
, {1, -44, -13}
, {17, 39, -15}
, {-38, 0, 19}
, {34, -52, -36}
, {44, 0, 52}
, {-32, 24, -13}
, {3, -15, -20}
, {19, -4, 43}
, {-13, -60, 3}
, {20, -57, -8}
, {-26, -24, -51}
, {20, -34, 1}
, {48, 0, -33}
}
, {{-12, -23, 38}
, {-3, -39, -53}
, {14, -5, 18}
, {-4, 45, -4}
, {38, -53, 34}
, {17, -21, 9}
, {36, 17, -53}
, {-49, -47, -13}
, {19, -11, -18}
, {-19, 32, 5}
, {34, -16, 32}
, {26, 47, 9}
, {-25, -52, -10}
, {27, 36, -46}
, {-10, -11, 39}
, {-36, 11, -49}
, {-50, -42, -42}
, {-55, -9, -40}
, {29, -40, -18}
, {-37, 10, -19}
, {-52, 28, -39}
, {28, -37, -8}
, {28, 5, -48}
, {42, -54, 16}
, {-6, -5, 21}
, {23, 5, -18}
, {-16, 20, -15}
, {-30, 29, 14}
, {39, 6, 0}
, {-39, 13, 45}
, {-49, -49, 26}
, {-31, 9, -8}
, {33, 46, 0}
, {7, 30, 10}
, {-19, 34, 32}
, {0, -51, -36}
, {-45, 46, 14}
, {41, -47, -25}
, {-36, -48, -54}
, {-18, -17, 13}
, {2, -45, -40}
, {-53, 30, -33}
, {-25, -50, -43}
, {-28, -13, 9}
, {7, -45, 41}
, {24, -8, 19}
, {43, -5, -35}
, {44, -41, -30}
, {18, -34, -10}
, {-47, 31, 15}
, {-19, -3, -39}
, {-35, -34, 7}
, {39, -38, -6}
, {-50, -6, -17}
, {-55, -45, 20}
, {-49, -14, 44}
, {-28, 0, -31}
, {-28, 25, 47}
, {-50, 46, -6}
, {43, 23, 21}
, {-44, 0, -14}
, {-12, 21, -52}
, {-37, 23, -27}
, {-12, -4, -25}
}
, {{27, 4, -53}
, {-37, -36, 16}
, {-41, 49, 50}
, {-13, -38, -25}
, {9, -13, 6}
, {5, -5, 37}
, {34, -32, -36}
, {20, 21, -49}
, {-24, 21, 14}
, {39, 24, 6}
, {-39, -9, -45}
, {-7, 38, -25}
, {-37, -28, 22}
, {-29, 18, -36}
, {-15, 31, 41}
, {20, -52, 33}
, {-16, -48, -27}
, {-42, 27, -29}
, {34, 38, 19}
, {4, -9, 28}
, {31, -15, 13}
, {-25, -12, -5}
, {-11, 8, 23}
, {10, 41, -49}
, {41, 39, 27}
, {10, 45, 34}
, {-14, -4, 58}
, {22, -23, -20}
, {20, 19, 12}
, {-47, 38, -10}
, {33, -13, 13}
, {47, 0, 48}
, {-8, 11, -2}
, {-24, 11, 52}
, {41, 21, -5}
, {58, -37, -22}
, {-12, 25, 40}
, {-9, 34, -8}
, {-4, -11, -30}
, {-9, 16, -38}
, {36, 15, 42}
, {-39, -14, -29}
, {-32, -22, 7}
, {-8, 14, -4}
, {26, -50, 42}
, {-42, 6, 15}
, {26, 5, 16}
, {-49, 52, 28}
, {-33, -41, 25}
, {-48, -44, -29}
, {-44, -41, 1}
, {5, 32, 55}
, {-31, 28, 40}
, {50, -3, -2}
, {46, 39, 43}
, {-29, 12, 21}
, {-62, -25, -17}
, {-55, -29, -32}
, {28, 38, -35}
, {-7, -3, -55}
, {29, -35, 12}
, {41, 15, -17}
, {-56, -26, 32}
, {21, -32, -31}
}
, {{-29, -48, 29}
, {17, -33, -6}
, {-48, -45, -30}
, {10, -51, 28}
, {-28, -26, 0}
, {-40, 24, -13}
, {-21, 34, -28}
, {-17, 44, -27}
, {-28, 1, -44}
, {-21, 11, 28}
, {39, -48, 28}
, {16, -21, 41}
, {40, -6, -37}
, {18, 5, -10}
, {-38, 27, 31}
, {-3, 36, -40}
, {34, -43, 50}
, {-2, -19, -24}
, {48, -22, 19}
, {13, -30, -2}
, {26, 19, 39}
, {-24, -47, -3}
, {-14, 34, -35}
, {2, -28, -43}
, {-28, 29, 24}
, {13, 51, 47}
, {39, 10, -17}
, {-13, -19, -20}
, {54, 43, -41}
, {46, -13, 38}
, {-33, 51, 50}
, {21, -10, 13}
, {-54, 15, -40}
, {17, -38, 46}
, {52, 47, -6}
, {29, -44, 25}
, {-19, -46, 16}
, {-42, -41, -41}
, {-20, 37, -43}
, {23, 23, 9}
, {-28, 33, 24}
, {-19, 17, -6}
, {-20, -7, -3}
, {36, 0, 12}
, {10, 16, 43}
, {43, 24, 13}
, {-12, 31, 25}
, {2, -17, -40}
, {47, 38, -4}
, {-1, -16, -28}
, {23, -9, 33}
, {-15, 26, 16}
, {-8, 16, 27}
, {21, 18, -16}
, {7, -29, -30}
, {-27, 54, -22}
, {25, 13, 29}
, {42, -46, -51}
, {17, 9, -52}
, {-41, 8, -41}
, {-33, 43, 21}
, {-27, 32, -39}
, {19, 8, 30}
, {-5, -32, -1}
}
, {{-29, 14, -21}
, {-9, -34, -40}
, {34, -46, 14}
, {25, -5, -4}
, {-55, -44, -23}
, {15, 41, -35}
, {-53, 2, 6}
, {-8, -42, 45}
, {-44, -33, -20}
, {-28, -25, 6}
, {-26, 9, 43}
, {-17, -5, 3}
, {5, -24, 16}
, {-37, -9, -22}
, {41, -36, -2}
, {-52, -4, -10}
, {-19, -50, -40}
, {-45, -53, 39}
, {-9, 2, -13}
, {-34, 3, 1}
, {-51, 29, 10}
, {-29, 24, 10}
, {-20, 44, 32}
, {30, -9, 38}
, {-37, -21, 30}
, {46, 0, 13}
, {-10, 28, -2}
, {-12, 43, -26}
, {-7, -54, 37}
, {-18, -35, 47}
, {32, 44, -19}
, {-3, -42, -33}
, {-17, -24, -20}
, {12, -27, 29}
, {-11, -8, -10}
, {-14, 27, 28}
, {-13, -30, 30}
, {38, -4, -38}
, {-5, 0, -31}
, {13, 20, 19}
, {-33, -35, 26}
, {-53, 29, -55}
, {11, 41, 2}
, {-9, -50, 47}
, {22, -22, 43}
, {-36, 18, -21}
, {49, 20, -2}
, {-1, 44, -52}
, {30, -54, 34}
, {11, 2, -7}
, {37, -9, 10}
, {-22, 11, -23}
, {-10, 36, 14}
, {-49, 32, -48}
, {-10, -39, -18}
, {-17, 43, 10}
, {21, -26, 8}
, {-43, 32, -35}
, {-4, -3, -55}
, {44, -29, 20}
, {-41, 19, 15}
, {18, -46, 23}
, {26, -48, -37}
, {17, -12, 7}
}
, {{21, 19, -42}
, {19, 18, 32}
, {38, 39, 23}
, {24, 6, 30}
, {-11, -23, -9}
, {18, 1, -28}
, {6, -37, 36}
, {-29, 29, 34}
, {-45, -10, -24}
, {7, 21, 22}
, {-32, -40, 15}
, {12, -35, -3}
, {-43, 0, -6}
, {53, 33, -39}
, {1, 41, -20}
, {35, 9, 42}
, {1, -17, -32}
, {-10, -47, -11}
, {46, -3, -12}
, {-35, 0, 6}
, {-19, -33, -6}
, {-24, -33, 28}
, {-2, 6, -48}
, {44, 32, 37}
, {46, -37, 40}
, {19, 7, -24}
, {30, -25, 38}
, {10, 6, 0}
, {-1, -15, 54}
, {21, -44, 15}
, {52, -15, 38}
, {-33, 10, 49}
, {10, -41, 36}
, {-10, 55, 16}
, {30, 0, 39}
, {41, -25, -17}
, {35, -23, 5}
, {-32, 0, -31}
, {-50, -19, -18}
, {41, 37, 0}
, {10, -29, -10}
, {56, 30, 20}
, {9, 44, 1}
, {51, -34, -12}
, {-24, -22, 5}
, {-42, -61, -27}
, {-31, 56, 19}
, {-7, 48, 48}
, {45, -22, 48}
, {-36, 8, -17}
, {-8, 25, -28}
, {5, -27, 23}
, {-23, -19, 40}
, {-39, -20, -42}
, {-54, 49, 36}
, {-40, 39, 17}
, {-17, 25, 39}
, {48, 13, 32}
, {29, -43, -9}
, {49, -13, 11}
, {29, -23, -52}
, {29, -9, 40}
, {-7, 48, -50}
, {35, 8, -10}
}
, {{-35, 42, -19}
, {-5, 30, -1}
, {-25, 15, 35}
, {-7, 26, -22}
, {24, 5, -51}
, {19, 34, 21}
, {10, 46, -53}
, {-13, 31, -17}
, {25, 29, -4}
, {0, -48, -31}
, {34, -38, -4}
, {-5, 40, 0}
, {12, -49, 43}
, {-17, -32, -30}
, {19, 5, -51}
, {11, 2, -13}
, {-55, 15, 42}
, {-1, -21, -47}
, {-20, 44, 2}
, {11, 30, -18}
, {27, -7, 26}
, {-29, -38, -28}
, {-8, -47, 2}
, {19, 0, -24}
, {2, -34, -36}
, {45, 38, 26}
, {-28, -16, 16}
, {-17, 42, 24}
, {-27, -43, 43}
, {-37, 45, -52}
, {-48, 27, -13}
, {-13, -55, 22}
, {2, -28, 5}
, {-3, -22, -18}
, {-51, -14, -47}
, {-54, 8, -12}
, {19, -8, -9}
, {-56, -19, -23}
, {-17, -32, 24}
, {17, 32, -41}
, {12, 5, -2}
, {-29, 21, -45}
, {7, -38, 28}
, {26, 43, 21}
, {40, 11, -28}
, {25, 13, -18}
, {3, -17, -49}
, {1, 7, -22}
, {-42, -15, -23}
, {34, 31, 3}
, {-31, 46, 12}
, {42, 46, -12}
, {-11, -14, 0}
, {7, 41, -41}
, {44, -18, 0}
, {-17, -6, -4}
, {-49, 0, -2}
, {-34, -50, 49}
, {-39, -20, -2}
, {26, -49, -32}
, {43, 34, 5}
, {-56, 27, -38}
, {-5, -45, -11}
, {-20, -48, -28}
}
, {{32, -27, 58}
, {-11, 26, -25}
, {24, 56, 9}
, {-32, -21, 33}
, {-44, 3, -31}
, {-26, 39, 9}
, {23, 2, 13}
, {8, 58, 6}
, {30, -60, -17}
, {-32, -31, -35}
, {46, 21, -27}
, {52, -30, 53}
, {41, 47, 63}
, {22, 32, 46}
, {-1, 64, -30}
, {-35, -23, 17}
, {-34, 42, 41}
, {-15, -35, -11}
, {-23, -38, 3}
, {30, 50, 4}
, {61, -25, 28}
, {7, 5, 30}
, {16, -18, -7}
, {-29, 44, -17}
, {32, -21, 58}
, {7, 11, -2}
, {-4, 40, -35}
, {-38, -24, 6}
, {-36, 16, -21}
, {21, -19, -13}
, {49, 6, -35}
, {8, -16, -4}
, {-32, 34, -48}
, {7, 15, 19}
, {-10, 24, -6}
, {-4, 20, -14}
, {58, 45, 37}
, {12, 9, 30}
, {-27, 31, 45}
, {0, -49, 51}
, {63, 43, -34}
, {-38, 12, -13}
, {-44, -20, -17}
, {33, 19, 3}
, {-48, -10, -38}
, {-14, -4, 19}
, {-38, 20, 16}
, {-40, -5, -32}
, {-14, 28, 14}
, {-33, 0, -30}
, {13, -11, -50}
, {-25, -34, 34}
, {31, -11, -35}
, {24, -23, -43}
, {-12, 28, -20}
, {5, -38, -26}
, {-33, -16, 1}
, {-43, 14, -20}
, {10, 13, 7}
, {-43, -43, 43}
, {-24, 17, 41}
, {14, 58, 25}
, {-37, -7, 3}
, {24, -42, -19}
}
, {{28, -24, 35}
, {34, -18, 52}
, {43, -21, -3}
, {33, 37, 31}
, {-11, 0, -15}
, {-23, -7, -5}
, {-58, -46, -38}
, {32, 20, 36}
, {38, -62, 6}
, {-39, 0, 43}
, {-30, -18, -39}
, {13, 13, 3}
, {44, 3, 40}
, {5, 43, -42}
, {-22, -3, -46}
, {-16, 12, -4}
, {-53, -34, 35}
, {60, 17, 39}
, {-15, 28, -41}
, {1, 62, 58}
, {-46, 9, 44}
, {30, 16, 0}
, {-10, 17, 14}
, {-27, 18, 2}
, {-14, 25, 21}
, {7, -21, -16}
, {7, -17, 48}
, {2, 41, 2}
, {-45, -12, -22}
, {-48, 1, 15}
, {-12, -2, -45}
, {0, 37, -42}
, {-2, -29, 39}
, {0, -27, 32}
, {-48, 2, 29}
, {-46, 23, 41}
, {-3, 59, 9}
, {49, 14, 23}
, {24, -18, 18}
, {-37, 20, -24}
, {4, 0, 10}
, {-41, 2, -18}
, {25, -35, -24}
, {-37, 4, -17}
, {-1, 27, -30}
, {45, 59, 30}
, {0, 12, -54}
, {40, 42, -5}
, {36, 53, -34}
, {-60, 0, -4}
, {-24, 11, -10}
, {-33, -49, 11}
, {0, -9, 4}
, {-35, -24, 19}
, {25, 35, 14}
, {-22, 14, -29}
, {-6, 43, -23}
, {-29, 38, 37}
, {22, 9, -20}
, {-18, -35, 0}
, {21, 6, 33}
, {55, 36, -22}
, {42, -28, 31}
, {-2, 24, -22}
}
, {{-36, -4, -23}
, {15, 24, 27}
, {60, 32, 50}
, {-42, -46, 15}
, {-44, -29, 6}
, {19, 0, -13}
, {42, 39, 25}
, {-7, -31, 61}
, {-18, -28, -16}
, {-22, -41, -20}
, {22, -4, -22}
, {-13, -9, -17}
, {42, -9, -38}
, {57, -15, -6}
, {60, -3, -36}
, {-23, -8, -59}
, {34, 29, 44}
, {54, 43, 44}
, {-36, 37, 41}
, {47, 21, 17}
, {2, 19, 3}
, {-15, 24, 37}
, {47, 46, 49}
, {-18, -45, -18}
, {-22, 48, -47}
, {-4, 8, 43}
, {44, 51, 29}
, {23, 3, 12}
, {-41, 6, 10}
, {-41, 31, 32}
, {44, 50, -3}
, {-23, 29, 2}
, {-3, -48, 9}
, {-19, 12, -21}
, {-47, -27, 8}
, {-9, 50, -4}
, {16, 3, 18}
, {24, 2, -8}
, {1, 47, -57}
, {-11, 43, 1}
, {-12, 42, 59}
, {19, 12, 52}
, {6, -39, 56}
, {33, -2, -39}
, {9, -27, 17}
, {-38, -31, 22}
, {-17, -24, -28}
, {-18, -53, -43}
, {-22, 4, -8}
, {15, 22, 30}
, {-22, -41, 59}
, {-25, -24, -33}
, {-25, -24, -34}
, {55, -9, 35}
, {43, -44, -28}
, {29, 52, 28}
, {11, 35, -34}
, {-28, 47, 3}
, {46, -20, -33}
, {2, -34, -14}
, {50, 14, 68}
, {-39, 56, 50}
, {2, -28, -25}
, {-5, -49, -23}
}
, {{-33, 12, 21}
, {-33, 0, -8}
, {24, -7, 26}
, {24, 28, -13}
, {24, -33, 6}
, {-8, -28, 43}
, {-10, -15, 31}
, {37, -13, -38}
, {-4, -56, -8}
, {22, -31, -12}
, {0, -39, 0}
, {23, 1, 43}
, {-23, -35, 8}
, {4, -55, -40}
, {-52, 31, 26}
, {38, -60, 5}
, {-51, -38, -12}
, {39, -8, -27}
, {20, 16, -41}
, {47, 32, -27}
, {-60, 33, 0}
, {37, 50, 53}
, {43, 8, 11}
, {0, -30, 45}
, {48, 32, -39}
, {5, -13, 36}
, {-51, 19, 29}
, {26, -7, 23}
, {-41, 24, 0}
, {5, 51, 48}
, {-21, -51, 26}
, {26, 20, -20}
, {-27, -36, 43}
, {-36, -62, 15}
, {-47, -36, -35}
, {22, -39, -53}
, {-20, 46, 40}
, {-9, 30, 35}
, {9, -2, -12}
, {-14, -22, 13}
, {42, -46, 51}
, {-16, 2, -20}
, {-34, 29, -50}
, {14, 28, 13}
, {41, 42, 33}
, {29, -3, -41}
, {-41, -22, -23}
, {-7, -9, -13}
, {38, -47, -8}
, {22, -62, 8}
, {-8, -41, 33}
, {6, 19, -30}
, {38, -7, -33}
, {52, 35, -39}
, {-34, -9, -20}
, {33, 24, 5}
, {43, -33, 28}
, {-37, 17, 10}
, {25, 1, -34}
, {19, -15, -34}
, {34, -23, -31}
, {25, -38, -39}
, {15, 41, -2}
, {34, -37, 52}
}
, {{-27, 18, 19}
, {-8, -46, 48}
, {-38, 9, -21}
, {-35, -27, 7}
, {23, 47, 51}
, {44, 14, -10}
, {10, 18, 2}
, {3, 25, -25}
, {19, -1, -9}
, {40, 23, -5}
, {27, -17, -7}
, {20, -19, 9}
, {-20, -26, -13}
, {-9, -30, -29}
, {-33, -16, -31}
, {0, 27, 2}
, {8, -31, 28}
, {-37, 41, -23}
, {-29, 9, 37}
, {-22, -45, 36}
, {25, -48, -23}
, {-40, 52, 55}
, {15, -9, -3}
, {32, -41, -37}
, {44, 27, 35}
, {-49, -34, 4}
, {36, 32, -11}
, {30, 44, 18}
, {-23, -10, 13}
, {6, 44, -10}
, {21, -46, -29}
, {-9, 31, 32}
, {2, -5, -33}
, {-44, -11, 19}
, {-34, 8, -41}
, {30, 7, 27}
, {-35, -34, 42}
, {4, 44, 32}
, {25, -34, -3}
, {-28, 20, -33}
, {39, -15, 46}
, {46, 1, -14}
, {-31, 27, 1}
, {-39, -53, 26}
, {50, -42, 7}
, {49, -14, 47}
, {-1, 2, 48}
, {43, -39, 50}
, {17, 16, -1}
, {39, 1, -14}
, {-16, -26, -10}
, {42, 15, -23}
, {-18, -36, -58}
, {39, 35, -17}
, {-4, 35, -4}
, {3, 18, 35}
, {-26, 54, 42}
, {-5, 17, 53}
, {-15, -26, -12}
, {-37, -9, 26}
, {-39, -25, -56}
, {6, 26, 24}
, {-2, 7, -29}
, {53, -27, -42}
}
, {{11, -16, 18}
, {27, 46, -5}
, {-20, -19, -22}
, {4, -9, -8}
, {-3, -45, 9}
, {-3, 50, 42}
, {7, 10, 31}
, {-24, 40, 14}
, {-31, -19, -8}
, {46, 21, 54}
, {-2, -22, -45}
, {52, -40, 47}
, {51, 23, 47}
, {61, -17, 55}
, {-32, 43, 26}
, {29, -61, -44}
, {6, -50, -5}
, {-43, 23, 28}
, {-24, -2, 48}
, {-43, -15, -3}
, {22, -17, 10}
, {-32, 43, -30}
, {-8, -3, 23}
, {28, -36, -34}
, {-9, -7, -8}
, {-8, -13, 24}
, {46, -32, 60}
, {-44, -16, -13}
, {0, -22, 52}
, {16, -40, -26}
, {36, -33, 18}
, {8, 0, 40}
, {-13, 47, -7}
, {-18, -34, 19}
, {53, -33, 58}
, {33, 0, 41}
, {7, -35, 21}
, {1, 50, 49}
, {20, 6, -31}
, {-49, -10, -40}
, {4, -12, 10}
, {-18, -25, -38}
, {11, 7, 37}
, {13, 40, 10}
, {39, -40, -50}
, {-35, 10, 2}
, {44, 54, -12}
, {14, 14, -10}
, {-26, 19, 21}
, {-12, -11, -22}
, {12, 19, -13}
, {-14, 3, -21}
, {-45, -23, -41}
, {-31, -27, 25}
, {-24, -18, -37}
, {-36, 25, 33}
, {0, -45, -57}
, {27, 6, 23}
, {44, 12, 9}
, {-1, 33, 20}
, {-1, 41, -1}
, {-37, -57, -28}
, {-36, -12, 29}
, {2, -18, 38}
}
, {{51, 44, 25}
, {-20, -42, 43}
, {1, -33, 10}
, {-25, 3, 26}
, {-11, -53, 34}
, {-47, -33, 15}
, {52, 20, 46}
, {30, -47, 42}
, {-25, 21, -15}
, {-3, 10, -24}
, {19, -51, 18}
, {-3, -47, 28}
, {46, 54, -13}
, {25, 36, -22}
, {3, -22, -27}
, {6, 29, 32}
, {-5, 1, -3}
, {-45, -10, -48}
, {49, 38, 21}
, {61, -15, 27}
, {66, -5, 61}
, {35, -50, 22}
, {8, 46, 60}
, {41, 55, -24}
, {-37, 6, 11}
, {11, -29, 40}
, {-44, 56, 29}
, {-34, 30, 15}
, {-49, -2, 48}
, {11, -6, -5}
, {29, 51, 45}
, {-43, 14, 40}
, {7, -19, 39}
, {-21, -9, 47}
, {12, 19, 19}
, {41, -16, 34}
, {10, 37, 39}
, {33, 29, 24}
, {40, 39, 1}
, {44, -27, -9}
, {47, 50, 47}
, {5, 33, 10}
, {38, -26, -39}
, {40, -1, 44}
, {-34, 6, -9}
, {43, 4, 39}
, {-11, 35, 38}
, {9, 5, -21}
, {-33, 1, -10}
, {-26, 7, -30}
, {50, -12, 44}
, {24, -45, 34}
, {43, 48, -13}
, {-31, 2, 15}
, {46, 6, 13}
, {5, 2, 53}
, {-44, -64, -21}
, {-34, -38, -49}
, {11, -31, -12}
, {14, 12, 0}
, {-39, 3, 25}
, {-41, -52, -24}
, {-38, -12, 21}
, {43, 33, 11}
}
, {{-31, -28, 26}
, {48, 8, -3}
, {-41, -18, 0}
, {-19, 7, -22}
, {-5, 6, -24}
, {3, -33, -32}
, {-48, 1, -48}
, {40, -54, -18}
, {8, 4, -35}
, {-17, 2, -11}
, {-18, -12, 8}
, {44, 9, -45}
, {-34, -52, -7}
, {42, -19, 14}
, {-45, -43, 31}
, {-31, -37, -28}
, {33, 22, 26}
, {4, -47, 47}
, {28, 5, 46}
, {-32, -23, -14}
, {-15, 22, -14}
, {15, -48, 43}
, {-33, -20, -36}
, {9, 2, 48}
, {19, 25, -5}
, {46, -51, -22}
, {-15, -14, -26}
, {-29, -21, 26}
, {0, 24, -27}
, {12, -50, 27}
, {-27, -32, 1}
, {47, -13, 0}
, {0, 21, -19}
, {-18, 30, 28}
, {-19, -51, 41}
, {38, 14, 20}
, {-1, 6, 21}
, {-25, -28, -49}
, {-51, 20, -27}
, {14, 25, -29}
, {-38, -41, -24}
, {32, -38, 35}
, {-39, -44, -47}
, {-8, -7, 21}
, {-38, -16, -52}
, {37, -12, -18}
, {0, 32, -4}
, {-2, 38, 19}
, {24, -25, -18}
, {0, -37, 37}
, {30, 7, -52}
, {26, -50, 23}
, {-23, 18, 41}
, {-46, -5, -20}
, {14, -21, -37}
, {-24, 21, 3}
, {13, 31, -30}
, {5, 6, 13}
, {50, -31, -10}
, {-16, -32, 25}
, {5, 37, 49}
, {39, 1, 6}
, {41, -32, -30}
, {-31, 29, 33}
}
, {{-10, -9, 21}
, {-24, -44, -51}
, {-12, -21, -17}
, {31, 31, 44}
, {1, 0, 10}
, {31, -33, -48}
, {46, 21, -22}
, {28, -29, 43}
, {36, -36, 17}
, {22, 40, -41}
, {-36, -33, -55}
, {-37, 5, -14}
, {20, 29, 54}
, {12, 57, -15}
, {54, -8, -54}
, {-39, -22, 0}
, {-5, -26, -8}
, {22, -45, 47}
, {-24, -36, 10}
, {40, -3, -14}
, {42, -28, -21}
, {-36, 5, -34}
, {-20, -39, -35}
, {26, -39, -41}
, {-8, -9, 30}
, {49, -17, -6}
, {37, -23, -24}
, {-20, -45, 3}
, {-11, 46, 5}
, {-33, -26, 22}
, {37, 7, -30}
, {-43, 46, 49}
, {29, -21, 41}
, {29, 52, 11}
, {2, 35, 41}
, {-27, -1, 6}
, {-24, 53, 47}
, {39, -55, -18}
, {38, 50, -56}
, {-21, 13, -5}
, {23, -20, 43}
, {49, -19, 49}
, {-23, -55, 20}
, {-36, 43, -28}
, {15, -15, -26}
, {4, -44, 7}
, {21, -36, 27}
, {46, -30, 39}
, {-27, -53, -5}
, {10, 1, 28}
, {-10, 8, 26}
, {-6, -29, -17}
, {5, 13, 7}
, {0, -12, 32}
, {-51, -11, 26}
, {9, 11, 40}
, {-33, -54, 31}
, {-47, 15, 45}
, {50, 6, -37}
, {3, -47, 34}
, {-21, -3, 57}
, {-23, 35, 28}
, {0, -21, 48}
, {27, 33, 38}
}
, {{-12, 27, -20}
, {0, -10, 41}
, {46, -18, 0}
, {14, 19, 3}
, {17, -42, -46}
, {-12, -25, 1}
, {-39, -34, -52}
, {1, 6, 43}
, {-24, 35, 33}
, {24, -44, 25}
, {7, 6, 19}
, {26, 49, -30}
, {33, 0, 43}
, {-33, 6, 35}
, {-32, -12, -23}
, {-35, -56, -13}
, {31, 13, 1}
, {33, -14, 19}
, {8, -39, -16}
, {48, -49, 51}
, {36, -11, 0}
, {-36, 25, -33}
, {22, 16, -34}
, {-8, -50, -12}
, {44, 26, -39}
, {-45, -27, 38}
, {32, -5, -21}
, {-46, 13, 21}
, {-3, -14, -14}
, {-31, 26, 34}
, {20, -52, 16}
, {24, 32, -52}
, {21, -1, -34}
, {-12, -17, 23}
, {-2, -27, 42}
, {37, 15, -44}
, {23, 14, -42}
, {-43, 55, -1}
, {-17, 44, -34}
, {22, -19, 42}
, {-47, -29, -1}
, {-8, -30, -1}
, {-33, -48, 16}
, {40, -2, 13}
, {8, 54, -40}
, {51, 1, 18}
, {-11, 19, 8}
, {-2, 52, -8}
, {22, 7, 15}
, {-63, -4, 7}
, {-12, 2, -11}
, {24, 27, 20}
, {-19, -39, 50}
, {28, 44, -7}
, {-11, -30, 52}
, {-20, 37, -7}
, {-40, -16, -31}
, {-41, -42, -50}
, {18, 24, -37}
, {-46, 27, 7}
, {-40, -4, 46}
, {-43, -23, 18}
, {-43, 31, 1}
, {23, 44, -20}
}
, {{-32, -40, 43}
, {-52, 16, -9}
, {31, 33, 35}
, {4, -26, 8}
, {34, -21, 14}
, {-55, 30, -47}
, {-7, -28, -30}
, {-52, -26, -11}
, {-33, -29, -42}
, {-40, 2, -3}
, {-19, -4, -5}
, {21, 20, 6}
, {42, 45, -28}
, {3, 43, 19}
, {0, -27, -36}
, {-44, 26, 31}
, {38, -45, 8}
, {9, -26, 8}
, {4, -16, 22}
, {32, -20, 18}
, {-17, -43, 23}
, {39, -32, -10}
, {16, -14, 1}
, {-25, -50, -5}
, {-54, 2, -22}
, {5, 26, 4}
, {-15, 0, -53}
, {43, 42, 36}
, {-48, 27, 2}
, {23, 33, 15}
, {31, 21, 25}
, {-19, -17, -9}
, {-38, 39, -34}
, {-47, 1, 9}
, {-18, -37, 47}
, {-38, 9, 25}
, {-10, 36, 39}
, {14, -42, -22}
, {-5, -16, -19}
, {-23, -27, 38}
, {43, -40, -2}
, {-21, -19, 35}
, {-4, -14, -10}
, {-26, -30, 17}
, {-18, -42, 33}
, {21, 37, 51}
, {59, 26, -31}
, {25, -37, -34}
, {-19, -37, -17}
, {-13, -54, -8}
, {-32, 3, -19}
, {-37, -23, 21}
, {-26, -6, -37}
, {-41, -35, -4}
, {33, 44, -55}
, {-24, 48, 20}
, {-22, 0, -52}
, {-19, -11, 51}
, {30, 17, -18}
, {26, 40, 20}
, {5, 17, 45}
, {-26, -16, -52}
, {-5, 28, -30}
, {44, -19, -39}
}
, {{44, -6, 51}
, {44, -39, 19}
, {15, -17, -49}
, {-37, 1, -10}
, {-1, -39, 42}
, {31, -8, 11}
, {2, 24, -45}
, {0, 16, -25}
, {-9, -8, -17}
, {10, -49, -34}
, {36, -32, 32}
, {-27, -3, 12}
, {-39, -42, -55}
, {-6, 25, -37}
, {20, 25, 17}
, {25, -5, 0}
, {-8, -3, 61}
, {-8, -8, -41}
, {44, 58, -18}
, {-30, -19, -21}
, {-31, -60, -43}
, {31, 46, 3}
, {-17, -6, -11}
, {18, -43, -47}
, {38, 8, -2}
, {-46, 4, 21}
, {-16, 31, 28}
, {47, -7, 10}
, {-10, 40, 38}
, {44, 54, 20}
, {21, -16, 40}
, {43, 45, 17}
, {46, 39, -26}
, {-57, 41, -23}
, {-5, -25, -56}
, {-18, 52, 5}
, {33, -37, -32}
, {-10, -2, -34}
, {-45, 12, 14}
, {27, 51, 19}
, {33, -19, 23}
, {0, -42, 42}
, {-33, -15, -13}
, {35, 25, 40}
, {27, 22, 34}
, {-28, 44, 51}
, {-51, -40, 15}
, {-13, 26, 39}
, {-23, -39, -2}
, {17, 27, -32}
, {-46, -17, -15}
, {-35, -30, -44}
, {-47, 36, 22}
, {33, -48, 35}
, {30, -42, -18}
, {17, 39, 32}
, {63, 56, -12}
, {33, 16, 1}
, {-14, 41, 48}
, {0, 10, -35}
, {25, 36, -44}
, {49, -21, 39}
, {2, -38, 33}
, {-27, 46, 32}
}
, {{-32, 0, -31}
, {-19, -19, -42}
, {-16, 13, 31}
, {-51, -15, 29}
, {6, 37, -36}
, {43, -57, 39}
, {-2, -7, 15}
, {35, -39, -12}
, {-27, 14, -37}
, {-20, -12, -3}
, {-3, -1, 36}
, {33, 32, 33}
, {9, 16, -33}
, {0, 16, -33}
, {-42, 42, 39}
, {49, 35, 32}
, {-32, -26, -10}
, {-50, 36, -28}
, {17, -47, 15}
, {55, -33, 28}
, {36, -13, 3}
, {36, 31, -9}
, {-41, 52, 28}
, {-45, 45, -17}
, {20, 36, -41}
, {27, -36, 41}
, {28, 64, 29}
, {20, -50, -45}
, {40, -10, 20}
, {5, 7, 37}
, {6, -20, -14}
, {18, -13, 24}
, {22, -27, -27}
, {23, 50, 57}
, {16, 18, -35}
, {43, -29, 20}
, {18, 10, 1}
, {5, -54, -36}
, {-22, -23, 40}
, {11, -45, 9}
, {41, -41, 40}
, {-10, 14, -33}
, {30, -28, 7}
, {8, -10, 46}
, {43, -21, 19}
, {-27, 2, -8}
, {45, 51, -14}
, {-34, -42, 25}
, {-22, 27, -49}
, {-35, 25, -22}
, {30, -9, 47}
, {10, -51, 34}
, {13, -4, 41}
, {6, 34, 31}
, {-25, 48, 17}
, {40, 24, 42}
, {37, -28, -29}
, {44, -41, 32}
, {17, 24, 46}
, {13, 29, -31}
, {-13, -19, 22}
, {-42, -14, 29}
, {10, 40, 38}
, {-48, -46, 45}
}
, {{27, -41, -42}
, {-32, 5, -25}
, {-36, -34, -7}
, {42, -3, -22}
, {21, -10, 18}
, {41, 50, 17}
, {-13, -54, -21}
, {3, -8, 35}
, {-21, 5, -14}
, {27, 17, 38}
, {13, -34, -24}
, {-53, -18, -18}
, {-23, 1, 5}
, {36, -31, -43}
, {43, -42, 1}
, {49, 30, 36}
, {45, -53, 18}
, {31, 32, 44}
, {26, 26, -23}
, {33, -9, -54}
, {-17, -49, 31}
, {-9, 49, 12}
, {15, -5, 22}
, {25, 17, 26}
, {-19, -32, -37}
, {-5, -43, -16}
, {5, -28, -16}
, {22, -44, -14}
, {17, 20, -36}
, {-2, 12, -26}
, {-29, 25, 42}
, {28, -23, -7}
, {-6, -19, -26}
, {-32, -23, -33}
, {19, -44, 16}
, {21, -5, -1}
, {40, 9, -49}
, {28, 31, -37}
, {-30, -35, -50}
, {-19, 10, 42}
, {-16, 13, 2}
, {-35, -16, 5}
, {49, 0, -35}
, {-45, 37, -48}
, {-31, 17, 46}
, {5, 27, -42}
, {-39, 26, 3}
, {-12, -32, 29}
, {44, 24, -10}
, {19, -3, -43}
, {-38, -12, 37}
, {9, -19, 52}
, {23, -20, -19}
, {32, -28, -30}
, {48, 27, -10}
, {44, 22, -49}
, {3, -40, 51}
, {-12, -8, 3}
, {-22, -48, -33}
, {-5, 5, -34}
, {-37, -29, 31}
, {-32, -22, 27}
, {35, 42, -9}
, {-12, 34, 35}
}
, {{46, 30, -56}
, {26, 27, 24}
, {-35, -21, -56}
, {7, 7, 23}
, {-40, 25, -25}
, {-43, 5, 34}
, {4, -46, 31}
, {32, 8, 25}
, {40, -4, -32}
, {-12, 32, -38}
, {9, -31, -15}
, {42, -30, 6}
, {12, -49, 16}
, {-34, 44, -5}
, {8, -49, -34}
, {-35, 36, 24}
, {26, 7, -12}
, {-38, -19, 45}
, {16, 10, -14}
, {26, -17, -21}
, {-28, -6, -40}
, {-37, 13, 35}
, {-9, 18, 6}
, {2, -29, -21}
, {-52, 45, -47}
, {8, -19, -42}
, {-4, -18, 2}
, {-10, 14, 14}
, {-44, 7, -27}
, {0, -27, 21}
, {15, -40, -13}
, {4, 43, 12}
, {-6, 21, -26}
, {-15, -26, -2}
, {-41, 14, 1}
, {-31, 40, -15}
, {-14, 33, 33}
, {4, -12, -39}
, {-44, 31, -32}
, {17, -45, -54}
, {-31, -55, 26}
, {1, 50, -15}
, {37, -21, -18}
, {-46, -10, 38}
, {-53, 0, 7}
, {-27, -17, -39}
, {43, 43, 21}
, {-42, -19, -41}
, {18, -28, 24}
, {-46, 25, -2}
, {-38, -42, 8}
, {20, -7, 34}
, {-20, -38, -49}
, {-40, -24, 35}
, {-49, -27, -54}
, {0, -21, -29}
, {33, -49, -14}
, {0, -3, -17}
, {26, 32, -43}
, {-13, -27, 30}
, {-23, 3, 4}
, {17, 2, 20}
, {5, -5, -29}
, {-14, -45, 22}
}
, {{-20, 17, -29}
, {-5, -22, -39}
, {0, -46, -26}
, {40, -20, -25}
, {0, 38, -35}
, {32, 26, -36}
, {12, -30, 1}
, {-44, -13, 17}
, {7, 15, 28}
, {-3, -13, 44}
, {6, -16, 16}
, {-10, 41, -28}
, {32, 13, 0}
, {-22, -46, 8}
, {21, -48, -11}
, {-38, 21, -29}
, {46, 52, 39}
, {-55, -11, -41}
, {-9, -21, 45}
, {-21, 9, -39}
, {29, -27, -54}
, {19, -36, -33}
, {24, -1, 32}
, {42, -3, -25}
, {35, 28, -6}
, {-43, -54, -46}
, {56, -55, -47}
, {-12, -9, -28}
, {20, 35, 18}
, {-31, -2, -54}
, {24, -11, -29}
, {-50, -29, -26}
, {-54, -9, 13}
, {-31, 31, -19}
, {2, -17, -55}
, {40, 7, 45}
, {24, -47, -42}
, {26, -6, -35}
, {-19, -36, 43}
, {-48, 42, -23}
, {-45, -47, -6}
, {26, -35, 27}
, {11, 27, -12}
, {9, 13, -6}
, {-42, -56, 12}
, {38, -58, 45}
, {-40, 20, 29}
, {-1, -41, -18}
, {-2, 28, -37}
, {48, 51, 41}
, {-28, 38, -52}
, {32, 32, -37}
, {24, 28, 37}
, {-10, -12, 40}
, {-24, -38, 33}
, {10, 28, -20}
, {-12, -15, -16}
, {-39, 18, -43}
, {-57, 37, 23}
, {-41, -38, -1}
, {17, -44, -15}
, {-35, 32, 16}
, {-20, -14, -54}
, {27, -22, -18}
}
, {{-31, -40, -1}
, {23, 45, 44}
, {-42, -2, -45}
, {10, 44, 53}
, {-6, -19, -22}
, {17, 21, 42}
, {-42, -38, 47}
, {-33, 9, 35}
, {-9, 7, 6}
, {17, 56, -3}
, {14, 45, 4}
, {30, 0, 45}
, {-48, -2, 8}
, {55, 26, -8}
, {0, -61, 37}
, {-31, 18, 31}
, {34, 28, 36}
, {-9, 5, 12}
, {12, -36, -7}
, {-54, 2, -28}
, {25, 9, 40}
, {-16, 14, 20}
, {23, 9, 1}
, {-15, 53, 38}
, {-48, -39, 44}
, {18, -14, 19}
, {-14, -16, 1}
, {-10, 30, -5}
, {27, 6, 10}
, {4, -54, -14}
, {-11, -13, -29}
, {-30, -45, 48}
, {-27, 31, -4}
, {-2, 26, -30}
, {-39, 22, -44}
, {40, 22, 12}
, {24, 4, 24}
, {35, 33, -31}
, {-44, -30, -31}
, {21, 44, -14}
, {-27, 30, 50}
, {-19, 46, 23}
, {42, -6, 3}
, {-8, -30, -14}
, {-28, 21, 39}
, {-11, -31, 18}
, {-14, 27, -35}
, {-49, 28, 39}
, {-44, 39, 11}
, {4, 32, -10}
, {41, -38, 11}
, {-43, -7, -32}
, {3, 59, -21}
, {3, -39, -36}
, {48, 39, -27}
, {-18, -11, 48}
, {-15, -17, 23}
, {29, -6, 26}
, {-20, 17, 26}
, {-40, -4, -66}
, {-41, 16, 31}
, {13, 34, 36}
, {-32, 19, -56}
, {-14, 7, -17}
}
, {{32, -2, 45}
, {0, -46, 10}
, {2, -22, 21}
, {5, -40, -54}
, {48, 1, 27}
, {-29, 0, 6}
, {59, -43, -1}
, {-12, 9, 0}
, {38, 4, 6}
, {10, 20, 34}
, {-15, -52, -43}
, {-35, 27, -18}
, {28, 33, 12}
, {-1, 1, -58}
, {3, 49, -42}
, {64, -16, -18}
, {49, -47, -32}
, {15, -8, 40}
, {47, 18, 52}
, {-45, -27, 0}
, {-7, 44, -4}
, {-11, 23, 1}
, {0, 6, 28}
, {16, 49, -32}
, {0, -39, 22}
, {41, 61, 53}
, {27, 22, 28}
, {47, 44, -19}
, {-7, 14, 14}
, {46, -19, -13}
, {43, 45, -29}
, {-24, -40, 49}
, {36, -46, -3}
, {19, 15, 64}
, {2, 43, -18}
, {59, 14, 22}
, {-44, 16, -25}
, {-38, 30, -17}
, {-32, 28, 44}
, {-39, -45, 18}
, {-43, 40, -30}
, {5, -5, 35}
, {13, -16, 43}
, {31, -21, 39}
, {-18, -23, -26}
, {16, 11, 35}
, {1, 60, 12}
, {-40, -45, 46}
, {21, -26, 39}
, {11, -22, 62}
, {17, 25, 0}
, {56, 50, -8}
, {-41, 16, -17}
, {-22, -27, 34}
, {41, -17, 28}
, {-9, -38, 50}
, {35, -38, 23}
, {4, 36, 13}
, {-3, -27, 26}
, {-23, 30, 7}
, {-52, -8, -30}
, {-44, 35, -33}
, {21, 21, 31}
, {-41, 43, -31}
}
, {{-36, 35, 6}
, {-35, 50, 9}
, {13, 4, 12}
, {-46, 20, -45}
, {46, 51, 35}
, {-24, 9, 29}
, {1, -16, 21}
, {0, -37, 23}
, {-18, -18, -17}
, {-15, 1, -4}
, {-30, 44, -6}
, {6, 0, -1}
, {-15, 10, 16}
, {29, 29, 17}
, {-36, -47, 1}
, {26, -7, -62}
, {-15, 0, 24}
, {-1, 39, 26}
, {38, -7, 23}
, {22, 40, -4}
, {-17, 24, 23}
, {20, -47, 27}
, {31, -40, -28}
, {-30, -4, 20}
, {8, -51, 47}
, {-50, -53, 20}
, {-23, -10, 45}
, {12, 5, 28}
, {-26, -14, 2}
, {17, 42, 25}
, {7, 11, -25}
, {1, -19, 37}
, {-14, -22, 51}
, {32, 22, -46}
, {-16, -49, -45}
, {-13, -4, 14}
, {-36, 52, 57}
, {25, -48, -9}
, {45, 41, -12}
, {0, -38, -19}
, {13, 22, -35}
, {-4, -25, 45}
, {-9, 23, -38}
, {-5, -42, -4}
, {-50, -30, 33}
, {23, 14, -6}
, {-20, 1, 11}
, {49, 41, 17}
, {51, 11, 44}
, {36, -48, 37}
, {52, -21, 30}
, {-4, 17, -47}
, {-27, -5, 8}
, {20, 27, -9}
, {-47, 20, 33}
, {10, 38, 44}
, {-19, -28, -25}
, {27, -50, -22}
, {-2, -40, -9}
, {-14, -66, -25}
, {14, 21, 31}
, {27, -18, 10}
, {32, 2, 27}
, {14, 36, 45}
}
, {{-31, 58, 44}
, {-22, 58, -9}
, {25, 3, 57}
, {10, -24, 22}
, {-43, 48, -26}
, {-33, 27, 43}
, {50, 19, 54}
, {43, 10, 63}
, {4, 30, -53}
, {-13, -3, 52}
, {-16, -32, -33}
, {-45, 5, 10}
, {25, -23, 49}
, {-8, 45, -42}
, {-11, 18, 59}
, {8, -53, 42}
, {39, -35, -47}
, {-26, 26, 24}
, {30, -52, -11}
, {44, 0, -29}
, {23, 38, 36}
, {41, -8, 65}
, {-39, 0, -38}
, {-12, -23, -28}
, {-25, 40, -22}
, {6, -21, -58}
, {-20, -48, -15}
, {-17, 27, -32}
, {10, -39, 0}
, {9, -1, 35}
, {38, 51, 28}
, {-25, -8, 4}
, {20, 16, -23}
, {-29, 9, 0}
, {-44, 21, -58}
, {-44, -17, 40}
, {36, -17, -21}
, {-17, 42, -27}
, {3, -35, -19}
, {-28, 39, -5}
, {12, 19, -25}
, {-46, 28, -25}
, {-2, -11, -19}
, {34, 65, 62}
, {19, -39, -17}
, {-23, 23, -34}
, {-31, -26, -53}
, {-20, 40, -21}
, {15, -33, 30}
, {15, 7, -50}
, {52, 48, -10}
, {20, -44, 19}
, {-16, -10, -56}
, {-5, -38, -22}
, {-26, -7, 67}
, {-7, 37, -9}
, {-21, 37, 17}
, {32, 0, -32}
, {23, -8, 15}
, {40, 8, 10}
, {-24, -29, 33}
, {-3, -14, 32}
, {59, 25, 30}
, {-41, 7, -28}
}
, {{-3, -32, 2}
, {15, 26, -28}
, {-3, 39, -53}
, {-33, 38, 38}
, {-4, -34, -49}
, {16, -24, -47}
, {-33, -8, -19}
, {32, 22, -20}
, {10, -47, -34}
, {-53, 34, 31}
, {-42, -46, 45}
, {15, -11, -4}
, {-8, 24, -41}
, {45, -26, -30}
, {46, 21, -13}
, {43, -6, -36}
, {-37, 55, -3}
, {-55, -14, -21}
, {10, -15, 35}
, {45, -56, -44}
, {15, -37, 2}
, {-42, -4, -32}
, {-49, 40, 13}
, {-17, -36, 8}
, {20, 38, -42}
, {-32, 35, -22}
, {28, -26, -7}
, {-51, -41, -10}
, {-49, 35, -15}
, {-49, 20, -15}
, {23, 15, -17}
, {-22, -4, 6}
, {23, -15, 30}
, {47, 40, 17}
, {17, -47, -25}
, {21, -19, -43}
, {-38, 34, -28}
, {-8, -38, -13}
, {27, 19, -51}
, {-42, -23, -31}
, {-13, -16, 4}
, {-10, -6, 33}
, {-39, -25, 38}
, {9, -50, -21}
, {37, 31, 17}
, {-24, 8, -8}
, {-5, 37, 12}
, {47, -43, 38}
, {-9, -12, 45}
, {1, 46, 39}
, {-47, -48, 38}
, {-41, -22, 9}
, {30, -14, -42}
, {-15, -36, -43}
, {11, 43, 13}
, {38, 35, -34}
, {21, -46, -6}
, {-10, 43, 59}
, {-19, -30, 45}
, {-36, 10, 48}
, {-15, 23, 32}
, {11, -43, 0}
, {-7, -18, -22}
, {44, -51, 0}
}
, {{-23, -54, -30}
, {48, -44, 6}
, {-47, 41, -7}
, {-41, 4, 51}
, {4, 50, -22}
, {-7, -19, 37}
, {20, -43, 16}
, {20, 20, 28}
, {-46, -50, 17}
, {-4, -32, 17}
, {1, 43, 9}
, {0, -46, 53}
, {-27, 34, -37}
, {33, 21, 19}
, {-9, -52, 17}
, {13, 26, -5}
, {29, -65, -18}
, {4, -20, 1}
, {-20, -7, -22}
, {-3, -22, 25}
, {-7, 52, 16}
, {-25, -48, 28}
, {5, -1, 47}
, {41, -22, 26}
, {-21, 51, 21}
, {23, 17, -35}
, {55, -23, -3}
, {-39, 13, 20}
, {-31, 44, 24}
, {15, 48, -7}
, {1, 8, -46}
, {38, -8, 7}
, {36, -22, 40}
, {39, 43, 17}
, {8, -28, -18}
, {-9, 5, -11}
, {5, 33, -8}
, {-46, 10, -30}
, {16, -19, 32}
, {-37, 54, 26}
, {-11, 12, -22}
, {23, 25, 54}
, {40, -19, -40}
, {-11, -24, 40}
, {15, -10, -21}
, {-24, 27, 27}
, {60, 41, -8}
, {44, 42, -35}
, {4, 44, 7}
, {16, -31, -52}
, {19, -27, -31}
, {-23, 60, -18}
, {-3, -63, -50}
, {-21, 46, 11}
, {-10, 29, 24}
, {-10, -9, 22}
, {-7, -45, -49}
, {24, 16, -5}
, {3, 42, -26}
, {-31, -3, -27}
, {-32, -45, 31}
, {-53, -10, -2}
, {33, -51, 20}
, {19, 23, 34}
}
, {{-24, -20, -15}
, {-22, 36, -7}
, {28, 2, -5}
, {-4, 14, -17}
, {-7, 35, 43}
, {-5, -56, -8}
, {-12, 32, 20}
, {0, 70, 55}
, {-4, -50, 22}
, {-3, 19, -34}
, {3, -19, 37}
, {44, -42, -16}
, {-40, -4, 57}
, {62, 41, 4}
, {-42, -11, 0}
, {-35, -31, -9}
, {39, 39, -9}
, {-18, 44, 3}
, {20, 8, -61}
, {-31, -18, -14}
, {38, -24, 4}
, {-42, 47, 14}
, {7, 25, -40}
, {40, -13, 57}
, {-42, 19, 9}
, {35, -25, -9}
, {-27, 23, 35}
, {7, 20, 18}
, {-1, 41, -3}
, {-8, -49, 16}
, {26, 45, -29}
, {-18, 12, -4}
, {-4, 13, 49}
, {19, 10, -30}
, {4, 16, 6}
, {-32, -24, -34}
, {1, 8, -5}
, {-27, -9, 7}
, {-27, 25, -11}
, {-26, 44, -31}
, {42, 22, -28}
, {31, 35, 46}
, {14, -48, 23}
, {28, 2, -34}
, {52, 18, 6}
, {22, 55, 30}
, {-50, -44, 1}
, {-29, -55, 24}
, {-24, -14, 32}
, {-32, -23, -26}
, {-12, -42, -31}
, {-48, 60, -42}
, {-19, 15, -43}
, {35, 51, -4}
, {-62, -22, -15}
, {-48, -13, -45}
, {-33, -24, 22}
, {12, 27, 15}
, {58, 20, -34}
, {-41, 9, 10}
, {60, 54, 28}
, {-32, 62, -9}
, {37, -46, 33}
, {17, -57, 35}
}
, {{13, -20, 50}
, {-1, 13, 30}
, {13, 37, -20}
, {-32, 33, -3}
, {26, 13, 49}
, {-33, 29, 1}
, {35, -27, -55}
, {14, 9, 11}
, {-30, 13, -12}
, {-12, 26, 32}
, {10, -16, -5}
, {-11, -30, -47}
, {-55, 33, 0}
, {7, -59, -34}
, {27, -43, -5}
, {19, 66, 63}
, {16, 57, -28}
, {46, 34, -44}
, {38, -7, -33}
, {20, -3, -19}
, {-6, -50, -16}
, {-27, 52, 44}
, {-26, -8, -7}
, {-40, -23, -19}
, {55, 35, -24}
, {6, -43, -37}
, {-6, -39, -35}
, {35, 41, 7}
, {-37, -26, -10}
, {59, 54, -28}
, {42, 20, -3}
, {-39, -30, 46}
, {55, 6, 25}
, {-32, -37, 17}
, {0, -39, 15}
, {26, 32, 25}
, {-7, -32, -11}
, {45, 0, -25}
, {17, -5, -3}
, {44, 57, 26}
, {34, -50, -51}
, {-12, 31, -14}
, {5, 24, 11}
, {26, -7, 28}
, {-14, 56, -31}
, {14, 42, 16}
, {-37, 17, -27}
, {-7, -35, 19}
, {2, -1, -41}
, {12, 37, 47}
, {42, -41, 40}
, {12, -9, -11}
, {2, 14, 7}
, {0, -29, -13}
, {42, 18, -43}
, {-47, 6, -3}
, {53, 1, -17}
, {-12, 27, 15}
, {-42, -22, -17}
, {44, -3, 10}
, {-17, 8, 23}
, {2, 35, -3}
, {61, -38, -23}
, {9, 45, 42}
}
, {{2, -35, 11}
, {39, 36, 1}
, {-4, 9, -53}
, {-49, -18, -28}
, {2, -39, 42}
, {-29, -9, 21}
, {-16, -1, 41}
, {42, -17, -43}
, {-16, -26, -12}
, {-25, -26, 44}
, {11, 13, 3}
, {41, 14, 18}
, {-44, -19, -35}
, {-25, 29, -15}
, {-5, -42, 62}
, {65, 56, 61}
, {0, -32, 36}
, {22, -51, 14}
, {43, -8, -40}
, {26, -19, 28}
, {-28, -59, -36}
, {31, -42, -10}
, {-18, -19, 48}
, {-47, 35, -49}
, {24, 28, -23}
, {-2, 14, -18}
, {1, -42, -23}
, {48, -30, -33}
, {-25, -3, -11}
, {-34, 22, 27}
, {48, 29, 41}
, {-13, -48, 41}
, {-45, 50, -36}
, {6, -50, -48}
, {-9, -22, -34}
, {7, 21, 22}
, {26, 28, 15}
, {-53, 44, -44}
, {-20, 12, -37}
, {-5, 11, -20}
, {-22, -9, -50}
, {35, 35, 2}
, {36, 4, -26}
, {-3, -40, -32}
, {42, 18, -47}
, {-23, 33, -25}
, {-24, 19, 5}
, {32, 24, -34}
, {8, -35, 24}
, {34, -3, -1}
, {13, -40, -11}
, {-11, 27, 19}
, {27, 22, 36}
, {-40, 42, 8}
, {4, 4, -21}
, {42, 5, -22}
, {-21, 39, 32}
, {-7, 50, 36}
, {5, 22, 35}
, {41, 51, 6}
, {-47, 13, 41}
, {-8, 44, -3}
, {2, 53, -32}
, {-26, 52, 25}
}
, {{-20, 1, -47}
, {23, -26, -40}
, {26, -3, 23}
, {44, 34, 33}
, {-20, 25, 7}
, {3, -47, -29}
, {51, 47, 3}
, {-57, -38, -3}
, {47, 34, 16}
, {31, 59, 4}
, {-45, -41, -16}
, {14, 20, -49}
, {35, 49, 2}
, {42, -24, 56}
, {45, 5, -36}
, {33, -50, -27}
, {-10, 43, 33}
, {-50, -36, 14}
, {9, -14, -31}
, {4, 11, -21}
, {-5, 25, 33}
, {10, -25, -40}
, {-15, 42, 26}
, {-4, -41, 12}
, {23, 23, 44}
, {-2, 52, 20}
, {-12, 21, 18}
, {-7, 24, 34}
, {30, 7, -32}
, {-36, 33, 22}
, {24, 15, -1}
, {40, 30, 12}
, {-31, 44, 37}
, {-16, 13, 57}
, {-16, 3, -37}
, {10, 38, -33}
, {-41, -37, 51}
, {-35, 44, 24}
, {41, -45, -56}
, {-13, 7, 10}
, {-16, -17, 13}
, {-15, 40, -19}
, {-37, -14, -27}
, {-11, 51, 40}
, {20, -33, -46}
, {2, -3, 38}
, {-31, 54, 12}
, {-3, 12, -18}
, {-10, -20, 16}
, {15, -67, 47}
, {26, 38, 26}
, {7, -33, 37}
, {-2, -16, 22}
, {-31, 23, 10}
, {49, 1, -2}
, {-45, -25, -19}
, {-11, -4, 3}
, {-53, -12, 21}
, {-23, -14, 15}
, {-10, -31, 4}
, {-32, -33, -11}
, {19, 35, 26}
, {-7, -33, 37}
, {-14, -49, -21}
}
, {{22, -13, -41}
, {4, 0, -10}
, {-27, 21, -3}
, {-28, 34, 41}
, {-38, -31, 25}
, {22, -20, 26}
, {26, 15, -30}
, {11, 11, -53}
, {-41, -37, -55}
, {-3, 19, 1}
, {1, 37, -53}
, {-40, -28, 28}
, {-16, 23, -37}
, {-31, 4, -34}
, {-34, -46, -37}
, {54, 25, 24}
, {39, 23, 35}
, {24, 3, 22}
, {-7, -5, -14}
, {25, 35, -54}
, {23, 66, 38}
, {-23, -37, -10}
, {-5, -49, 10}
, {42, -7, -50}
, {42, 11, 8}
, {-53, 42, 50}
, {37, -19, 31}
, {-40, 27, -12}
, {18, -4, 46}
, {15, -9, -5}
, {-32, -30, 23}
, {20, -55, -33}
, {-26, -18, 15}
, {-30, 47, -29}
, {54, -18, -5}
, {49, 54, 26}
, {-39, 21, -24}
, {-49, -26, 33}
, {49, 9, 26}
, {38, -1, 34}
, {17, -41, -34}
, {-43, 0, 32}
, {49, 0, 0}
, {-39, 20, -35}
, {-19, -46, -2}
, {35, -6, 6}
, {57, 26, 24}
, {36, -9, 52}
, {-1, 2, 48}
, {-43, -24, -26}
, {10, -53, -51}
, {28, 2, -14}
, {41, 31, -14}
, {14, 45, -27}
, {-9, -6, -30}
, {-46, -19, -11}
, {0, 36, -28}
, {-46, -38, -28}
, {-28, -25, 33}
, {37, 40, 32}
, {-15, -12, 2}
, {-47, 8, 10}
, {32, -2, 21}
, {-53, -3, 30}
}
, {{11, 19, -28}
, {-24, -15, 8}
, {-37, -7, 22}
, {6, 0, -35}
, {-28, 50, 13}
, {-46, 29, 47}
, {14, -19, -46}
, {17, 58, 39}
, {-30, 7, 8}
, {-49, 18, -3}
, {11, -45, -15}
, {-25, 50, 34}
, {-4, -2, -28}
, {56, 54, 42}
, {17, 34, 26}
, {-55, 17, 28}
, {-12, 47, -14}
, {-31, 31, -13}
, {20, -32, 38}
, {10, 10, 51}
, {-49, 16, -25}
, {38, -49, -14}
, {33, 29, -31}
, {39, 10, 7}
, {-34, -2, 17}
, {-20, 28, -9}
, {-26, -22, -41}
, {-48, -32, 5}
, {-37, 1, 20}
, {27, -14, -10}
, {28, 11, 10}
, {5, -35, 40}
, {-27, 5, -5}
, {-41, -38, 11}
, {4, -13, 38}
, {7, -37, -60}
, {-29, -12, -26}
, {37, -30, 36}
, {-17, -37, -15}
, {6, 3, -23}
, {8, 43, -22}
, {45, -21, -51}
, {29, -39, -23}
, {-41, -19, 17}
, {-29, 20, 5}
, {23, -14, -15}
, {24, 26, -50}
, {25, 33, 9}
, {14, 36, 40}
, {6, 18, -36}
, {3, 42, -42}
, {39, -41, -13}
, {-2, -13, 37}
, {-28, 9, 5}
, {-38, -11, 32}
, {-42, -31, -10}
, {21, 19, 51}
, {19, 4, 37}
, {28, -25, 44}
, {5, -31, -64}
, {-11, 51, -19}
, {15, 51, 56}
, {-9, 36, -25}
, {31, 43, -15}
}
, {{-3, 38, -42}
, {7, 8, 3}
, {-14, 27, 62}
, {-2, -34, -23}
, {-32, 9, 4}
, {-19, 42, 26}
, {-3, 31, -34}
, {12, -26, -68}
, {9, -9, -20}
, {49, -17, -2}
, {-20, -31, -17}
, {-9, -11, 29}
, {5, -16, -15}
, {58, 34, -58}
, {23, -56, -6}
, {-48, 40, 30}
, {43, 33, 24}
, {1, 4, 20}
, {-14, 18, 34}
, {-42, 34, 32}
, {31, -18, 0}
, {-52, 36, -6}
, {-11, 49, 47}
, {-3, 17, 10}
, {54, -26, -15}
, {34, 61, 0}
, {-25, -34, 20}
, {0, 26, 51}
, {0, -19, 60}
, {-46, -4, 60}
, {10, -29, 49}
, {-2, -18, 3}
, {-60, -51, -31}
, {-33, 34, -39}
, {-6, 0, -2}
, {10, 18, 18}
, {17, 36, -37}
, {-37, -22, -27}
, {-15, 33, -38}
, {48, 2, -9}
, {-33, 39, 33}
, {23, -45, -22}
, {18, 53, 62}
, {-34, -22, 24}
, {-5, -38, -21}
, {48, -36, 24}
, {42, -25, 1}
, {4, 17, 24}
, {58, -32, -5}
, {-43, 13, 38}
, {-37, 41, -11}
, {18, -28, -1}
, {-36, -34, 4}
, {42, -41, -16}
, {-37, -7, -20}
, {26, -20, 9}
, {-16, 25, -39}
, {23, -27, 0}
, {4, -37, 30}
, {-21, -39, -34}
, {2, 39, 24}
, {20, -49, 7}
, {-58, -26, -35}
, {44, 24, -19}
}
, {{-38, -38, -20}
, {-33, 6, -28}
, {13, 11, -32}
, {19, 6, 32}
, {38, 43, -28}
, {10, -43, 31}
, {2, -19, 35}
, {-33, 26, 27}
, {-58, 31, -1}
, {-18, 18, 57}
, {-47, 26, 26}
, {20, 4, 13}
, {-21, 11, 48}
, {-24, -5, -55}
, {-50, 1, 4}
, {-54, 43, 35}
, {-67, 10, -16}
, {-28, 2, -36}
, {2, -38, -6}
, {26, -61, 1}
, {40, 19, 9}
, {-52, -64, 2}
, {7, -21, 4}
, {-43, -24, -3}
, {1, 7, 30}
, {-10, -29, -4}
, {53, -21, 68}
, {-2, 21, 39}
, {-7, -30, -13}
, {-25, -24, -8}
, {38, -9, 21}
, {49, 1, -15}
, {-19, -39, 1}
, {42, 12, -10}
, {39, -33, 60}
, {52, 53, 43}
, {42, -3, -30}
, {-40, -42, -42}
, {50, -41, 5}
, {52, 15, 41}
, {42, 30, 16}
, {47, -32, 42}
, {12, 55, -35}
, {-33, -20, 27}
, {-6, 46, -36}
, {-29, 7, 21}
, {56, 17, 29}
, {-55, -33, -10}
, {-1, 16, 9}
, {-14, -51, 9}
, {50, -29, 38}
, {11, 43, -20}
, {-21, -40, -54}
, {-5, 44, 33}
, {-27, -20, 26}
, {55, -4, 2}
, {34, -39, 0}
, {1, -11, -11}
, {17, 11, 6}
, {-20, -53, 11}
, {-24, -18, 31}
, {42, -26, -40}
, {-4, 15, -33}
, {-24, 47, -13}
}
, {{43, 0, -56}
, {52, -49, 29}
, {-11, 5, 28}
, {-58, 19, -28}
, {4, -4, -12}
, {-26, -47, 33}
, {-6, -18, 19}
, {3, -15, -13}
, {48, 2, 3}
, {21, -27, -40}
, {8, 4, -19}
, {18, -20, 27}
, {-24, 1, -59}
, {1, -37, 31}
, {14, 53, -3}
, {-39, 47, -17}
, {39, 31, -18}
, {-53, 0, -39}
, {41, -38, 28}
, {6, -54, -51}
, {35, -6, -14}
, {51, 41, 46}
, {42, -25, -1}
, {-16, 10, 22}
, {11, -24, -41}
, {-12, 33, -49}
, {23, 17, -25}
, {-31, 5, -12}
, {-55, 1, 41}
, {-31, 5, 48}
, {45, 51, -9}
, {41, 26, 44}
, {40, 0, 28}
, {-12, -4, 40}
, {25, -38, 35}
, {-16, -21, 1}
, {-2, 22, 8}
, {20, -36, -39}
, {-36, -8, -14}
, {-31, 34, 39}
, {5, -2, 16}
, {-8, -19, -6}
, {53, -37, 26}
, {-46, -47, -11}
, {-3, -7, -30}
, {11, -44, 35}
, {-20, -35, 4}
, {31, 54, 57}
, {7, -7, 41}
, {39, 25, 53}
, {-43, -2, 32}
, {-39, -31, 19}
, {-9, -1, -51}
, {-5, 28, 14}
, {20, -46, -6}
, {22, 37, 8}
, {19, -8, -2}
, {-25, -8, 30}
, {-24, -24, -39}
, {-33, 18, 19}
, {30, -56, 28}
, {-29, 9, -44}
, {-23, 59, -11}
, {49, 30, 11}
}
, {{4, -12, 8}
, {-23, 37, -19}
, {-40, -21, 38}
, {-2, 21, 0}
, {41, -42, 42}
, {36, -47, 21}
, {29, -20, 3}
, {32, -5, -52}
, {-7, -3, -42}
, {-45, 39, -38}
, {-2, 18, -45}
, {-42, 19, -4}
, {43, 12, -42}
, {-23, 12, 30}
, {48, -33, 55}
, {49, -9, 1}
, {-15, 55, 38}
, {-28, -23, 5}
, {-44, 30, 40}
, {19, -27, 25}
, {19, 17, 21}
, {-45, -12, 25}
, {49, 9, -20}
, {7, -4, -9}
, {46, -39, -48}
, {39, -20, -21}
, {52, -8, 33}
, {-15, -13, 51}
, {-3, -38, 0}
, {40, 8, -50}
, {0, 0, 1}
, {51, 6, 49}
, {-37, 0, -8}
, {-24, 18, 12}
, {21, 36, 43}
, {-16, -18, -6}
, {-36, -51, -51}
, {0, 42, 25}
, {-13, 17, -49}
, {37, -6, 7}
, {-17, -54, 29}
, {-2, 37, 10}
, {-25, 46, 11}
, {-31, -4, 51}
, {0, 19, -27}
, {-8, -13, 15}
, {29, -33, 62}
, {-38, 26, 26}
, {14, -31, 38}
, {6, -29, 9}
, {-21, 25, 11}
, {49, 21, 49}
, {-17, 48, -2}
, {51, -41, -12}
, {-12, 36, -10}
, {1, 51, 34}
, {-44, -44, -9}
, {34, -19, -7}
, {42, -8, -33}
, {-28, -13, 18}
, {-24, 15, 43}
, {-18, -21, 3}
, {-1, 6, 37}
, {34, 13, -50}
}
, {{-43, 18, -22}
, {-5, -30, -7}
, {40, -50, 18}
, {-17, 33, -6}
, {-14, -41, 30}
, {-5, -41, 24}
, {34, -4, -43}
, {-1, 61, 45}
, {7, 36, -11}
, {-2, 28, -50}
, {0, 21, 62}
, {-10, -27, 17}
, {-24, -20, -26}
, {-13, 38, 10}
, {23, 33, 41}
, {63, -22, 40}
, {18, -26, 8}
, {30, 17, 43}
, {-9, -48, 45}
, {23, -5, 9}
, {-22, -7, -27}
, {8, -6, 14}
, {24, -21, -17}
, {31, 42, -14}
, {40, 36, -9}
, {19, 24, -19}
, {-40, -31, -27}
, {12, -2, 45}
, {-20, -30, -21}
, {-2, 49, 16}
, {-54, -7, 7}
, {-46, 29, 10}
, {43, -8, 35}
, {7, -2, -47}
, {8, -23, -5}
, {31, -36, 16}
, {14, -14, -15}
, {20, -29, -20}
, {44, 34, -30}
, {-31, 44, 8}
, {58, 35, -31}
, {-28, 29, 38}
, {32, 21, -31}
, {14, -30, 9}
, {2, 41, -6}
, {19, 16, 33}
, {20, -60, -49}
, {5, -1, 29}
, {35, 47, -35}
, {11, 4, 55}
, {-41, -46, 44}
, {7, 7, 31}
, {27, -4, -7}
, {42, -47, 7}
, {-19, -3, -8}
, {32, 33, 23}
, {20, -2, 8}
, {-35, 3, -7}
, {-17, -37, -8}
, {-4, 46, 6}
, {-27, -4, -17}
, {40, -18, 51}
, {51, -26, -12}
, {47, 27, 9}
}
, {{52, 6, 15}
, {-47, 11, -28}
, {10, 45, 14}
, {-40, 26, 52}
, {0, 12, -41}
, {-25, 22, 9}
, {-37, 30, 38}
, {-22, 24, 47}
, {-33, 17, 43}
, {-49, 30, 44}
, {-9, 46, -5}
, {-23, -48, -3}
, {24, 38, 11}
, {49, -7, 42}
, {-22, 29, -57}
, {-52, -27, -34}
, {7, -33, 22}
, {-2, -24, 41}
, {35, 43, 26}
, {-23, 36, 1}
, {14, -16, 61}
, {-11, -48, -46}
, {35, 33, -29}
, {47, -48, 14}
, {-19, 40, -1}
, {-36, 30, -38}
, {-44, 0, 61}
, {6, 43, 19}
, {33, 4, 18}
, {22, -10, 46}
, {45, 41, 39}
, {-40, 51, 11}
, {-15, -29, -12}
, {47, 48, 21}
, {-8, 17, -22}
, {-21, 42, 30}
, {12, -22, 17}
, {17, 33, 3}
, {47, 10, -41}
, {24, -30, 2}
, {53, -33, 31}
, {1, -40, -21}
, {0, -9, -16}
, {-5, 11, 8}
, {37, -2, -12}
, {-8, -2, 23}
, {-40, 0, -46}
, {-19, 49, -16}
, {28, 22, -28}
, {-7, 22, -64}
, {30, -37, -41}
, {-44, 9, -44}
, {50, 9, -57}
, {-10, 27, 26}
, {-40, 27, -1}
, {14, 17, 22}
, {15, 22, -36}
, {16, 40, -36}
, {29, 4, -36}
, {7, -27, -21}
, {-27, 52, -32}
, {48, -51, 45}
, {28, 30, -55}
, {-34, 35, 23}
}
, {{-13, -13, -53}
, {-40, -10, -3}
, {-35, 1, -10}
, {40, 7, 48}
, {-6, -22, -5}
, {17, 41, -16}
, {-26, 10, -3}
, {29, -38, -4}
, {6, 7, 4}
, {-9, -15, -13}
, {-36, 42, -1}
, {-20, 23, -18}
, {50, -18, 27}
, {13, -24, -31}
, {-61, 14, -27}
, {11, 17, -4}
, {5, -46, 27}
, {-12, 31, 2}
, {-12, 30, 26}
, {-23, 9, -6}
, {-29, 2, -21}
, {18, -3, 34}
, {-26, -31, -39}
, {41, 22, -9}
, {33, 45, 12}
, {53, -21, 40}
, {-25, 67, -30}
, {-5, -51, 49}
, {46, -9, -20}
, {19, 11, -11}
, {-33, 28, 27}
, {-21, 9, 27}
, {-13, -34, -55}
, {44, -20, -25}
, {23, -30, -27}
, {47, 19, -34}
, {-5, 1, 18}
, {32, 11, -14}
, {16, -15, 56}
, {43, 7, -23}
, {-43, 38, -50}
, {8, 35, -42}
, {-42, 34, 12}
, {-29, 14, 40}
, {0, 25, 28}
, {47, 29, 52}
, {66, 39, -36}
, {42, -22, 29}
, {8, 43, 28}
, {-66, 35, -36}
, {2, -33, -5}
, {36, -39, -5}
, {35, 7, 5}
, {-33, 25, 23}
, {32, -32, 22}
, {-20, -24, -14}
, {44, 45, 35}
, {15, 39, -30}
, {37, -42, 47}
, {-59, 9, -21}
, {2, -53, -3}
, {-54, 27, -43}
, {-35, 5, -28}
, {48, -23, -34}
}
, {{28, 23, 53}
, {42, 17, 21}
, {11, 11, -10}
, {5, -20, -12}
, {-43, 53, -51}
, {-36, 11, 32}
, {23, 7, 21}
, {3, 41, -12}
, {9, 33, 11}
, {-23, 42, -3}
, {55, -44, -14}
, {-40, 38, -37}
, {-39, 59, -5}
, {-13, 59, 65}
, {11, -29, 3}
, {49, 22, 39}
, {30, 9, -25}
, {0, 19, -42}
, {-8, 16, 1}
, {40, 19, 65}
, {18, 29, 51}
, {-37, 35, 11}
, {-46, -37, 21}
, {30, 27, 23}
, {-32, 39, -15}
, {-37, -25, -36}
, {-25, -44, 35}
, {-3, -41, 26}
, {0, -11, -27}
, {29, -31, 30}
, {-28, -31, -43}
, {9, 22, -28}
, {53, 33, 29}
, {9, -49, -43}
, {-48, -27, -19}
, {37, -4, 43}
, {44, 19, -38}
, {29, 23, 0}
, {-50, -42, 26}
, {-21, -20, -31}
, {63, 46, 52}
, {0, -23, -25}
, {21, -18, -4}
, {6, 60, -29}
, {26, 27, -30}
, {-50, -40, -37}
, {-38, 16, 32}
, {-10, 37, -54}
, {12, 32, -10}
, {60, -12, -6}
, {-11, 17, -21}
, {-18, 38, -4}
, {19, 16, 28}
, {18, 7, 33}
, {53, -4, 21}
, {-41, -22, -4}
, {-29, -23, 23}
, {-14, 33, -39}
, {-16, -38, 23}
, {23, 44, -37}
, {24, 0, 61}
, {4, -51, 44}
, {-34, 51, -56}
, {40, 8, -18}
}
, {{-47, 53, 53}
, {-14, -38, 4}
, {51, 24, 0}
, {6, -44, -21}
, {47, 17, 7}
, {-20, 32, 12}
, {-11, 41, 41}
, {-37, 25, -37}
, {-47, 16, 45}
, {19, 14, -44}
, {-42, -40, 22}
, {50, 39, 12}
, {-19, 35, 34}
, {-31, 26, -4}
, {-2, -2, -41}
, {41, 46, -37}
, {34, 16, 14}
, {12, 29, 5}
, {13, -20, -50}
, {-64, -23, 0}
, {27, -5, 9}
, {44, -9, 7}
, {-39, -28, 4}
, {-1, 46, 39}
, {22, -33, -38}
, {2, -22, 43}
, {-4, -25, 6}
, {36, -40, 40}
, {1, -10, 19}
, {-45, -13, 39}
, {-33, -10, -26}
, {-41, -29, 22}
, {-31, 8, 0}
, {-30, -1, -51}
, {13, -34, 26}
, {46, -7, 38}
, {-27, -12, -47}
, {15, 40, 33}
, {35, 4, -25}
, {-1, 31, -34}
, {20, 0, 6}
, {10, 31, 24}
, {37, 11, -13}
, {25, -10, -13}
, {-28, 52, -21}
, {12, -9, -32}
, {-21, -51, 19}
, {50, 1, -39}
, {-24, 31, 14}
, {-33, -38, 38}
, {-25, -18, 41}
, {-47, 28, -36}
, {12, 28, -41}
, {16, 46, 34}
, {53, 22, -36}
, {-50, 25, -28}
, {27, 47, -5}
, {44, -47, 0}
, {-42, 28, 0}
, {-34, -27, 4}
, {21, -52, -20}
, {-9, 31, -19}
, {2, -17, 24}
, {51, -2, 44}
}
, {{-38, 25, -25}
, {-21, -39, 28}
, {-36, 6, 0}
, {-32, 37, 14}
, {34, -10, -9}
, {-13, 20, 11}
, {8, -25, 32}
, {-26, -31, -11}
, {-25, -12, 46}
, {-25, 11, -37}
, {-41, -6, -33}
, {-25, -11, -7}
, {31, -23, 41}
, {50, -7, -14}
, {14, -42, -14}
, {0, 38, 25}
, {31, -12, 23}
, {2, -39, 3}
, {-31, 9, 0}
, {-26, 1, -28}
, {21, -44, -53}
, {0, -33, 39}
, {37, -2, 0}
, {-30, -41, 43}
, {-16, 27, 38}
, {-4, -37, -42}
, {37, 14, 49}
, {-11, 11, 29}
, {28, -20, 23}
, {-41, -50, -16}
, {-3, -9, -14}
, {9, 16, -30}
, {-15, 32, -50}
, {63, -35, 55}
, {-47, 15, 19}
, {3, -26, 37}
, {0, -18, 40}
, {-3, -7, 35}
, {-46, 26, 19}
, {-3, -26, -50}
, {-43, 7, -40}
, {3, -25, 39}
, {-22, -52, -52}
, {37, 32, 29}
, {0, 45, -41}
, {18, 21, 9}
, {-3, -38, 2}
, {3, -14, -23}
, {-28, -22, 48}
, {-1, 48, -1}
, {13, 0, 0}
, {47, 49, 15}
, {1, -30, -41}
, {-50, 8, -53}
, {-24, 32, 31}
, {-1, 5, -26}
, {-23, -31, 41}
, {54, 45, -16}
, {2, 22, -42}
, {37, -24, 3}
, {29, -44, 49}
, {26, -8, -54}
, {41, -16, 6}
, {34, 8, 1}
}
, {{47, -22, -13}
, {26, -31, 33}
, {36, 49, -32}
, {5, 38, 36}
, {27, 49, -36}
, {-25, -21, -11}
, {-13, -30, -11}
, {-38, -6, -36}
, {35, 25, 45}
, {-47, 45, 16}
, {25, 34, -16}
, {-16, -14, -47}
, {-1, 14, -31}
, {44, 50, -33}
, {-24, -26, 47}
, {48, -26, 22}
, {17, -25, 52}
, {-30, -49, -33}
, {40, -15, 46}
, {-8, 7, 20}
, {13, 5, 6}
, {-48, -2, -8}
, {54, -31, -22}
, {29, 32, 24}
, {16, 37, -35}
, {28, -33, 9}
, {-46, 25, 7}
, {12, -22, 43}
, {-37, -19, 28}
, {34, -30, -21}
, {25, -45, -10}
, {-34, -8, 44}
, {39, 7, -26}
, {47, 21, -22}
, {-35, -37, -33}
, {9, -5, -10}
, {-34, -37, -50}
, {-53, 26, 17}
, {42, 29, -27}
, {-29, 0, -38}
, {-45, -54, -45}
, {-45, -41, -22}
, {12, -6, 40}
, {-7, -21, 47}
, {22, -27, -11}
, {-47, -55, -29}
, {4, 7, -31}
, {-46, 30, 17}
, {23, 0, 48}
, {50, -6, 69}
, {-34, -7, -24}
, {-37, 48, 16}
, {-20, 37, -25}
, {-9, -21, -2}
, {-38, -9, -34}
, {-26, -28, 10}
, {14, 34, 22}
, {4, 31, -18}
, {37, -7, -14}
, {17, 21, -15}
, {-52, -29, -42}
, {33, -48, 30}
, {14, 28, 42}
, {12, 33, -44}
}
, {{37, -45, -42}
, {42, -27, 37}
, {41, -49, -21}
, {5, 49, -51}
, {23, -38, -14}
, {44, 6, 10}
, {0, 16, 49}
, {24, 28, -28}
, {25, 41, 0}
, {-45, 27, 6}
, {-10, -50, 43}
, {26, -41, -4}
, {-21, -35, -31}
, {-9, -31, 39}
, {-23, 28, -31}
, {-13, 13, -40}
, {-12, -37, -12}
, {32, 16, 19}
, {41, -29, -11}
, {8, 10, -50}
, {37, -38, -49}
, {14, -31, -42}
, {-20, -37, 17}
, {-32, -52, 28}
, {40, -38, -10}
, {19, 2, -50}
, {28, -26, -22}
, {16, 44, 35}
, {-44, -3, 21}
, {-46, 12, 45}
, {-48, -29, -10}
, {-52, -42, 0}
, {10, 39, -18}
, {47, -37, 33}
, {-1, -6, -33}
, {32, 17, -9}
, {-14, -8, -45}
, {23, -28, -32}
, {-34, -46, 46}
, {16, -22, -34}
, {-20, -44, 3}
, {-13, -46, 46}
, {18, -17, -28}
, {-51, -19, -31}
, {32, -34, 9}
, {-33, 11, 11}
, {-40, 18, 26}
, {-23, 30, 36}
, {-46, -27, -44}
, {-10, -10, -25}
, {-51, -5, -15}
, {42, -33, 5}
, {35, 3, -40}
, {-40, -47, 28}
, {12, -38, 30}
, {-49, -31, -45}
, {31, 21, 32}
, {2, 18, -49}
, {-20, -27, -23}
, {-18, 21, 4}
, {-30, -39, -5}
, {40, -15, -21}
, {-13, 35, -39}
, {4, -28, 6}
}
, {{34, 23, -21}
, {-10, 53, 1}
, {5, 18, -45}
, {2, -21, 6}
, {33, -10, -19}
, {8, -54, 60}
, {-36, 3, -37}
, {17, 30, -36}
, {-10, 34, 30}
, {39, 28, -54}
, {27, 1, -27}
, {42, -13, -26}
, {-38, 49, -36}
, {52, 1, -30}
, {2, 54, 45}
, {43, -14, -33}
, {16, -35, 44}
, {34, 41, -17}
, {-49, -11, -5}
, {2, 20, 19}
, {-12, -54, 17}
, {-3, -7, -37}
, {35, 55, -32}
, {-10, -42, 35}
, {38, -41, 42}
, {7, -38, 31}
, {25, -72, 6}
, {-35, -34, -44}
, {0, 34, 14}
, {43, 21, -9}
, {2, -32, 28}
, {-13, -29, -50}
, {50, 56, 0}
, {-18, 17, -32}
, {26, -53, -11}
, {41, -45, 23}
, {-13, -35, -8}
, {0, -18, 0}
, {-50, -39, 20}
, {17, 37, 38}
, {-29, 39, -52}
, {40, 48, 19}
, {-48, 13, -26}
, {19, 25, 32}
, {-39, 31, 14}
, {48, 12, 12}
, {-29, -61, -46}
, {26, 45, -3}
, {19, -14, -37}
, {51, -9, -23}
, {-8, 5, 38}
, {24, 37, -4}
, {33, 10, 36}
, {39, -9, -38}
, {-45, 51, 19}
, {4, -57, -18}
, {46, 12, 27}
, {61, 61, -39}
, {0, -27, 10}
, {-7, -7, 38}
, {43, 63, 41}
, {-40, 50, 34}
, {37, -42, -35}
, {-32, -36, -2}
}
, {{60, -19, 4}
, {60, -12, -36}
, {28, 2, -20}
, {-11, 43, 24}
, {-34, -47, 4}
, {-14, 30, -30}
, {-19, 31, -32}
, {-22, 11, -15}
, {-1, 41, -48}
, {60, 59, 39}
, {44, -7, -24}
, {-31, 41, -30}
, {48, 43, 30}
, {-10, 4, 18}
, {0, -41, -9}
, {14, 12, -28}
, {32, 26, -9}
, {-17, 8, 19}
, {-19, -25, -33}
, {22, 43, -23}
, {39, 50, -33}
, {23, -29, -39}
, {39, 61, -29}
, {-24, 1, 50}
, {-16, -9, -32}
, {54, -49, 23}
, {2, -32, 28}
, {2, 11, -18}
, {13, -24, 43}
, {-35, -31, 39}
, {-35, -20, 24}
, {-11, 44, 34}
, {-34, 19, -41}
, {-13, -36, 51}
, {31, 59, 30}
, {-48, 1, 49}
, {35, -30, 12}
, {-35, 49, 44}
, {56, -30, 3}
, {32, -33, -27}
, {-37, 15, 8}
, {-27, 3, -40}
, {-25, -16, -17}
, {15, 16, -36}
, {42, -38, -42}
, {18, 42, -14}
, {-30, 39, 3}
, {7, -50, -1}
, {-19, 42, 10}
, {-57, -51, -27}
, {22, 11, 21}
, {-27, 10, 22}
, {15, -32, 4}
, {-32, -16, 39}
, {42, 44, -17}
, {13, 8, 25}
, {-26, -26, -29}
, {16, -4, -27}
, {-28, -22, 20}
, {-51, -41, -14}
, {18, 27, -28}
, {30, 38, 6}
, {17, 35, -6}
, {20, -20, 29}
}
, {{5, 11, 50}
, {-24, -47, 17}
, {-40, -5, 26}
, {-40, -9, 56}
, {-34, 4, 53}
, {22, -32, 7}
, {-13, -28, 46}
, {30, 7, 41}
, {-2, 21, 12}
, {31, -37, 12}
, {-49, 33, 52}
, {-43, -46, -28}
, {0, 29, 3}
, {3, -41, -11}
, {9, -43, -19}
, {-44, 51, -35}
, {37, 0, -36}
, {-15, 25, 48}
, {13, -7, 13}
, {-33, -49, -31}
, {-63, 40, -8}
, {11, 38, -24}
, {-17, -17, -42}
, {-6, 27, 15}
, {-48, -38, -21}
, {6, -25, 65}
, {2, 7, 11}
, {-27, 1, 15}
, {-12, -6, 56}
, {18, -22, 11}
, {41, 0, 41}
, {-18, 40, -31}
, {-21, 55, -28}
, {-40, 21, -36}
, {-27, 38, -13}
, {45, -16, -23}
, {27, -49, -11}
, {12, 48, -13}
, {20, 11, -22}
, {3, 12, 34}
, {34, -41, -21}
, {40, 0, 3}
, {38, -14, -27}
, {-52, -56, 60}
, {27, 27, 27}
, {-43, -8, 29}
, {33, 29, 1}
, {6, 15, 48}
, {-41, -43, -30}
, {2, 9, 11}
, {-3, -41, 53}
, {35, -13, 18}
, {34, 7, -28}
, {-50, -9, 13}
, {-48, -15, -3}
, {26, 50, 41}
, {7, 36, -11}
, {-39, 14, 42}
, {-34, -37, -22}
, {-23, 46, -18}
, {-55, -34, -12}
, {17, 44, -33}
, {-26, -40, -47}
, {28, 7, -1}
}
, {{-23, -16, -12}
, {-40, -55, -15}
, {39, 0, 16}
, {-54, 42, -21}
, {1, 10, -38}
, {8, 34, -34}
, {45, 15, 23}
, {-23, 0, 6}
, {-46, 12, 45}
, {-55, -44, -45}
, {18, -25, -49}
, {30, 27, 7}
, {-33, 40, 23}
, {19, -43, 10}
, {-23, 41, 42}
, {39, 2, -17}
, {-44, -34, -41}
, {-24, -42, 45}
, {17, -27, 3}
, {20, 45, -32}
, {11, -26, 11}
, {45, 41, 42}
, {-18, 24, -28}
, {46, -51, 42}
, {-50, -16, 4}
, {-14, -5, -37}
, {-26, 4, 7}
, {4, 7, 26}
, {43, 6, 20}
, {-54, -36, 47}
, {23, 43, 40}
, {17, -13, 7}
, {15, -52, -10}
, {12, 39, 34}
, {5, -27, 37}
, {-12, 24, -49}
, {-11, -28, -17}
, {-45, -52, -12}
, {-50, -22, 4}
, {-17, 0, -14}
, {16, 32, -41}
, {-32, 20, -13}
, {-46, 43, -49}
, {0, 12, 48}
, {-46, 41, -8}
, {5, 7, 1}
, {-50, 43, -44}
, {-2, 30, 26}
, {-42, -47, -42}
, {28, 6, -6}
, {-55, 5, 4}
, {33, -43, 32}
, {-21, 48, 3}
, {47, -29, -18}
, {-30, -33, 15}
, {-22, -42, -29}
, {-41, -1, -35}
, {-51, -44, 1}
, {-20, -16, -10}
, {12, -41, -28}
, {-23, 11, 0}
, {0, 8, 18}
, {19, -32, 24}
, {36, -21, 6}
}
, {{-31, 26, 46}
, {-17, 14, -50}
, {26, 46, -34}
, {40, -34, -15}
, {-24, -34, 24}
, {-39, -9, 36}
, {55, 47, 18}
, {-28, -59, -53}
, {-2, -22, -40}
, {-42, 24, -25}
, {-32, 29, -43}
, {27, -33, -4}
, {11, 2, -37}
, {-30, -5, -18}
, {48, 44, -21}
, {20, 61, 29}
, {34, -19, 56}
, {24, 21, 8}
, {28, -35, 36}
, {-16, -46, -7}
, {15, -48, -21}
, {49, -15, 36}
, {26, 5, 25}
, {7, 2, -40}
, {27, -6, -52}
, {-2, -21, 51}
, {-23, 13, 4}
, {57, 40, 28}
, {-41, 26, 29}
, {22, 36, 40}
, {38, -14, 18}
, {28, -5, 38}
, {-25, 24, 47}
, {-26, 52, -24}
, {48, -35, 39}
, {-21, -3, -26}
, {9, 38, 21}
, {-5, -40, -16}
, {-10, 22, 18}
, {-43, 16, 31}
, {-31, -16, 25}
, {18, 50, 31}
, {-10, -2, 56}
, {-45, 14, 28}
, {13, -13, -11}
, {-8, 1, 16}
, {-48, 10, 14}
, {-13, 3, 44}
, {-3, -9, -22}
, {65, 26, -17}
, {-5, 8, 27}
, {-1, -32, 21}
, {-1, -3, 57}
, {34, 0, 6}
, {-53, -29, 10}
, {-10, -24, 40}
, {-29, -22, -49}
, {-18, -38, 49}
, {-8, 20, -19}
, {48, -10, 20}
, {0, -44, -32}
, {-29, -5, 55}
, {-32, 41, -52}
, {53, 1, 11}
}
, {{53, -38, 30}
, {14, 13, -47}
, {-48, -34, 14}
, {33, 39, -36}
, {49, 45, -36}
, {53, 17, -37}
, {-29, -46, 15}
, {38, 2, -40}
, {-21, -33, 39}
, {0, -1, -51}
, {-33, 9, 32}
, {42, 28, 8}
, {11, 26, -48}
, {-61, 13, 26}
, {-48, -5, -51}
, {60, 10, 14}
, {-45, 49, 29}
, {-9, -36, -30}
, {25, 26, -4}
, {25, -50, -51}
, {24, -22, -18}
, {6, -5, 13}
, {7, 29, -37}
, {15, -13, -23}
, {-33, -20, -15}
, {-46, 4, 20}
, {15, -20, -29}
, {12, -20, -19}
, {-46, -37, -21}
, {-7, -3, -30}
, {-44, -42, 50}
, {-33, -44, 36}
, {-18, -29, 45}
, {2, -22, -25}
, {-34, -33, 44}
, {32, -35, -39}
, {-41, 29, 8}
, {-39, 18, 56}
, {13, -8, 28}
, {-12, 33, 4}
, {-2, -26, -42}
, {4, 13, -1}
, {24, 18, 11}
, {-29, -50, -3}
, {-6, 35, 37}
, {-40, -16, -3}
, {16, -22, 27}
, {36, 52, -30}
, {17, 46, 32}
, {-11, -20, 25}
, {53, -43, -8}
, {38, -3, -5}
, {-2, -9, 39}
, {16, 51, 26}
, {14, 30, -51}
, {38, 4, 41}
, {59, 56, 62}
, {32, 5, 38}
, {14, -16, 8}
, {-40, 51, 27}
, {-39, -27, -31}
, {-16, 0, 7}
, {29, 53, 11}
, {30, 49, 58}
}
, {{-36, 2, 41}
, {0, -56, -29}
, {-15, -47, 24}
, {22, 40, -29}
, {-37, -19, -9}
, {-52, -8, -6}
, {-17, 26, -55}
, {10, 6, -19}
, {-29, 21, 33}
, {-9, -37, 28}
, {-44, 9, 28}
, {44, -40, 21}
, {-49, -44, -5}
, {-40, 35, 45}
, {-18, -49, -19}
, {-5, -48, 13}
, {-15, -38, -49}
, {-17, -55, -32}
, {14, -21, -33}
, {20, -5, -9}
, {38, -21, -5}
, {4, -44, -26}
, {-42, -30, 34}
, {-38, -15, -42}
, {-12, -42, 9}
, {-27, -34, 25}
, {30, 25, -26}
, {-44, 25, -27}
, {1, -47, 9}
, {20, 23, -45}
, {16, 5, -59}
, {12, -40, -5}
, {-25, -25, -16}
, {-9, 51, 49}
, {25, -14, -38}
, {3, -24, 15}
, {-32, 27, 8}
, {-49, 0, 27}
, {30, 13, 4}
, {8, 28, 9}
, {4, 48, -52}
, {1, 30, -2}
, {-27, 34, 16}
, {-39, -12, -10}
, {-30, -37, 9}
, {46, 20, 23}
, {22, 35, -34}
, {-37, 8, 16}
, {-56, 26, -10}
, {40, -48, -20}
, {-1, -29, 44}
, {35, 20, 6}
, {22, -6, 11}
, {-8, -49, -11}
, {-42, -3, 31}
, {42, 17, 31}
, {-6, 18, -9}
, {-45, -4, 42}
, {-39, -51, -17}
, {-12, 49, -45}
, {50, -52, -44}
, {-14, 24, -31}
, {-36, 29, -44}
, {-53, -36, -46}
}
, {{-56, 19, -28}
, {30, -43, 36}
, {53, -44, 20}
, {-38, -50, -37}
, {-1, 17, 30}
, {32, 20, -5}
, {-24, 45, -32}
, {-49, -51, -44}
, {-36, -4, -17}
, {56, 22, -37}
, {39, -19, 21}
, {22, -2, 29}
, {40, 20, 20}
, {-13, -37, -5}
, {27, -22, -47}
, {20, 27, -18}
, {-34, 3, -19}
, {16, 18, -52}
, {-36, 16, -42}
, {-32, 32, 0}
, {58, -34, 16}
, {33, -17, -42}
, {26, 28, 11}
, {35, -6, 48}
, {24, 40, -42}
, {-41, -29, 51}
, {10, 3, 7}
, {47, -42, 23}
, {-35, 38, -3}
, {-31, -35, 41}
, {36, -37, -4}
, {-13, -18, 55}
, {9, -55, -28}
, {55, 59, 55}
, {38, 37, 38}
, {47, 15, 40}
, {-38, 13, 26}
, {-24, 40, -32}
, {-43, -10, 43}
, {0, -26, -35}
, {-6, 3, -5}
, {14, -14, -27}
, {-8, 53, -12}
, {0, 0, -21}
, {31, 20, 37}
, {-6, 28, 43}
, {46, -17, -17}
, {-49, 20, 43}
, {33, 31, 0}
, {-44, -28, -42}
, {47, 39, -53}
, {-41, 0, -13}
, {48, 11, -56}
, {-31, 33, 29}
, {8, 29, 15}
, {-25, -12, -11}
, {8, 27, 40}
, {-38, 17, -33}
, {21, 11, -39}
, {-51, -3, 32}
, {-31, -36, 11}
, {16, 19, -2}
, {13, -47, 34}
, {-21, -23, -37}
}
, {{-50, -35, 15}
, {-27, 17, -39}
, {-28, -11, -45}
, {-9, 6, -52}
, {24, 17, 33}
, {-28, -19, -20}
, {-38, -1, 32}
, {41, -47, -11}
, {9, -23, 34}
, {-37, 22, -44}
, {29, -21, -19}
, {4, 13, -13}
, {-47, 37, -18}
, {-19, -19, -9}
, {38, 5, 7}
, {-1, -28, 3}
, {59, 59, -26}
, {-15, 47, -8}
, {7, 57, 14}
, {-44, 7, -27}
, {-16, 2, -20}
, {-18, 34, 40}
, {-17, 16, 0}
, {-27, -35, 0}
, {1, 14, 16}
, {-28, 25, 34}
, {12, 14, -8}
, {-25, -4, -21}
, {16, 8, -11}
, {-12, 19, -46}
, {-28, 46, 7}
, {-7, 31, 29}
, {23, 37, 0}
, {-21, -10, 26}
, {41, -25, 32}
, {-44, -24, -5}
, {18, -56, -33}
, {-45, 33, 10}
, {-4, 7, -32}
, {-53, 26, 9}
, {-40, -30, 49}
, {-12, 25, -1}
, {-15, -27, 51}
, {-24, -24, -17}
, {-47, -29, 17}
, {17, 18, -2}
, {-30, 15, 41}
, {-15, 0, 41}
, {-38, -21, 30}
, {31, 52, 9}
, {-11, 37, 32}
, {-2, 26, -35}
, {-7, -49, -54}
, {0, 45, -11}
, {46, 19, -39}
, {-28, -3, 37}
, {-36, 52, -36}
, {9, -44, 18}
, {28, 1, 43}
, {49, -21, 29}
, {20, 3, -4}
, {47, -30, 14}
, {60, 13, 61}
, {-37, -46, 47}
}
, {{50, 28, 42}
, {-12, -52, 16}
, {-22, -15, -15}
, {4, 6, 7}
, {37, 2, 38}
, {-4, -37, 12}
, {-59, -42, 8}
, {61, -5, 22}
, {-23, -20, 4}
, {-47, -45, -6}
, {22, -1, 53}
, {-19, -44, 45}
, {-11, 42, -32}
, {-35, 9, 14}
, {46, -10, 30}
, {2, 15, -61}
, {47, 31, 35}
, {11, -1, 21}
, {-25, -45, 17}
, {-18, -7, -32}
, {-31, 0, 33}
, {-10, -13, 27}
, {-24, 23, -54}
, {-3, 46, 40}
, {23, 14, -43}
, {32, -18, 18}
, {-7, 13, 29}
, {2, 44, -41}
, {-42, 30, 0}
, {24, -50, -18}
, {16, -27, 27}
, {4, 33, -2}
, {-21, 26, 37}
, {-53, 31, 5}
, {36, 25, -16}
, {-48, 21, 35}
, {12, 4, -34}
, {27, 41, -9}
, {10, 12, 39}
, {-4, -13, 5}
, {40, -46, 44}
, {36, -37, 33}
, {-8, 14, -21}
, {10, 51, 19}
, {12, 22, -10}
, {31, -26, -34}
, {-9, 34, 2}
, {1, 44, 42}
, {41, -28, -38}
, {-12, -47, -17}
, {-38, -36, 46}
, {-7, 47, -33}
, {57, -8, -29}
, {-36, 50, 35}
, {-31, 35, -11}
, {-21, 2, 34}
, {-11, 10, 18}
, {-27, -20, -13}
, {-11, -27, 12}
, {3, 33, -17}
, {23, -25, -2}
, {35, 12, 53}
, {56, 2, 23}
, {6, 39, 40}
}
, {{-37, 23, 23}
, {-46, -10, 22}
, {-3, 34, -38}
, {25, 46, 8}
, {43, 20, 32}
, {55, -8, 49}
, {-5, 42, 44}
, {-22, 7, 15}
, {46, 2, -12}
, {-33, -24, -40}
, {2, 33, 43}
, {-48, 50, 29}
, {-20, 30, 22}
, {-65, 17, -61}
, {-28, -60, -50}
, {-8, -5, 36}
, {-17, 39, 12}
, {14, 33, -15}
, {1, 35, 20}
, {-36, 30, -43}
, {17, -24, -40}
, {-25, -1, -36}
, {-47, -55, -37}
, {-47, 2, -34}
, {-22, -36, 0}
, {41, -24, 4}
, {-26, -43, 25}
, {-45, -30, -24}
, {34, 2, 35}
, {47, 23, 15}
, {48, 34, -7}
, {-29, 20, -14}
, {26, 19, 39}
, {-62, -52, 1}
, {0, -42, -43}
, {3, 52, -20}
, {8, -46, 49}
, {24, -22, 24}
, {-22, 14, 22}
, {30, 7, 28}
, {-44, -32, 24}
, {-30, 38, -8}
, {38, -25, 24}
, {30, -38, 25}
, {-4, 17, -27}
, {-4, 13, 52}
, {16, 26, 6}
, {43, -15, 7}
, {-8, -9, -37}
, {-53, -7, 13}
, {-29, -44, 44}
, {42, 38, 33}
, {28, 23, 42}
, {13, -22, 51}
, {31, 48, -21}
, {-45, -31, 17}
, {-39, -26, 21}
, {40, -43, 19}
, {13, -49, -25}
, {-41, 40, -14}
, {-25, 12, -58}
, {30, 25, -11}
, {35, 27, 15}
, {-36, -40, 39}
}
, {{-27, 10, -41}
, {-47, 0, -20}
, {0, 37, -35}
, {-53, -41, -54}
, {38, 44, 28}
, {-12, -7, -34}
, {-39, 21, -41}
, {0, 9, -48}
, {35, -28, 11}
, {33, 23, -19}
, {15, -34, -11}
, {9, 20, -3}
, {27, -41, 50}
, {-17, -1, 38}
, {11, -17, 49}
, {42, 12, 3}
, {-41, 42, -44}
, {24, -6, -43}
, {-25, -17, -24}
, {-34, -53, 8}
, {-33, 7, -50}
, {16, -50, 3}
, {-47, 38, -37}
, {-32, 51, 29}
, {14, 15, -6}
, {0, -48, 0}
, {55, 29, -15}
, {43, 0, 9}
, {54, 45, 25}
, {-3, 42, -50}
, {-23, -36, -42}
, {-49, -6, 33}
, {14, 51, 29}
, {22, -32, -34}
, {-8, 34, -28}
, {-1, 3, 22}
, {18, 19, 13}
, {19, 34, -16}
, {-12, 1, -46}
, {42, -47, -42}
, {4, -7, -46}
, {12, 1, -10}
, {-4, -12, 2}
, {-5, 19, 19}
, {-9, -11, -45}
, {25, -20, -27}
, {-47, -39, 35}
, {-1, -26, 39}
, {0, 50, 46}
, {11, 62, -2}
, {-24, 4, 9}
, {48, -35, -12}
, {12, 31, 31}
, {44, -52, -24}
, {-37, -1, 27}
, {45, 56, 27}
, {-11, -58, 32}
, {-26, 39, 49}
, {1, -51, -40}
, {1, 17, 26}
, {27, 0, -55}
, {29, -7, -56}
, {16, 44, 13}
, {25, 49, -24}
}
, {{-43, -40, 35}
, {41, -18, 52}
, {33, -23, 45}
, {46, -19, 11}
, {23, 51, 21}
, {-45, -27, 3}
, {45, 30, -41}
, {-44, 54, -17}
, {-11, 11, -9}
, {-24, 12, -32}
, {36, 1, 41}
, {-35, 20, -31}
, {18, -24, 11}
, {-41, 7, 23}
, {-31, -26, -24}
, {-46, -45, -44}
, {22, 5, 28}
, {49, 52, -35}
, {-20, -43, 43}
, {-32, -53, -41}
, {6, 39, -19}
, {10, 27, 25}
, {-35, 28, -4}
, {-35, 41, 31}
, {25, 5, -34}
, {-45, 11, -4}
, {57, -32, -35}
, {7, -46, 27}
, {-15, -57, -5}
, {-11, -17, -16}
, {-31, -19, -44}
, {-42, 9, 3}
, {43, -57, -6}
, {22, 24, -32}
, {23, -25, 33}
, {17, 27, -38}
, {37, 13, -22}
, {-33, 45, 1}
, {28, -44, -10}
, {42, 16, 56}
, {-27, -11, -30}
, {-11, 0, 39}
, {10, 8, 61}
, {-18, 28, 1}
, {6, 13, 20}
, {-10, 25, -42}
, {14, -20, 21}
, {34, -15, -56}
, {-10, -13, 37}
, {-18, -37, -10}
, {-28, 0, 19}
, {10, 17, 36}
, {-22, 45, -12}
, {44, 9, 53}
, {27, 37, -4}
, {-25, -29, -23}
, {26, -3, 33}
, {-27, 34, -32}
, {13, -18, -31}
, {-22, 2, -14}
, {-36, -25, -26}
, {28, -45, 26}
, {0, -55, -11}
, {-5, 16, 14}
}
, {{35, -34, 46}
, {-46, -31, 32}
, {-6, -28, -50}
, {-4, 0, 17}
, {-26, 38, 38}
, {37, -24, 3}
, {-35, 10, -17}
, {-23, -4, -23}
, {32, 56, 25}
, {9, -47, 34}
, {-23, -26, 49}
, {-28, -14, -31}
, {-10, -34, -27}
, {19, 68, -16}
, {17, -13, -42}
, {24, -8, 53}
, {-39, -32, -13}
, {23, -37, 20}
, {7, 26, 39}
, {65, 4, 22}
, {35, 47, -34}
, {29, -47, 16}
, {53, -4, -18}
, {9, -38, -1}
, {14, 34, 20}
, {20, -16, -27}
, {43, 26, -26}
, {-29, 43, 42}
, {30, -36, -6}
, {11, 23, -26}
, {-8, -18, -48}
, {14, -22, -2}
, {20, -2, -29}
, {38, 37, -7}
, {-47, 53, 58}
, {37, 38, 19}
, {-22, 37, -6}
, {-37, -15, -17}
, {-40, -22, 19}
, {48, -3, -35}
, {1, -37, -44}
, {-20, 47, -22}
, {3, -12, 12}
, {-5, 57, 24}
, {3, 11, -46}
, {25, 44, 24}
, {-13, -25, 46}
, {-11, 36, 6}
, {-41, 24, -29}
, {56, -25, 10}
, {-30, 28, 29}
, {39, -2, 19}
, {5, -18, -36}
, {-13, -12, 50}
, {-32, -11, -48}
, {-36, 11, 12}
, {-33, 5, -33}
, {57, 31, 32}
, {-40, -2, 48}
, {-26, 37, -20}
, {22, -27, 50}
, {-51, -3, 35}
, {-16, 39, -2}
, {29, -52, 1}
}
, {{52, 1, 57}
, {13, -1, -24}
, {33, -41, 47}
, {-2, -24, 37}
, {-37, 41, -13}
, {-7, -31, 11}
, {0, -4, 38}
, {-7, 45, -23}
, {9, 19, 33}
, {23, 0, 51}
, {41, 17, 34}
, {26, 35, 11}
, {59, -25, -2}
, {33, 13, 61}
, {61, -5, -14}
, {23, 43, -47}
, {-8, -48, 15}
, {6, -40, 10}
, {-18, 0, -28}
, {-19, 42, 0}
, {66, -25, 25}
, {12, 13, 8}
, {-4, 55, -39}
, {-45, 24, 42}
, {21, -15, 0}
, {34, -37, 25}
, {-22, 33, 2}
, {-46, 6, -11}
, {-17, 33, -33}
, {-13, -47, -38}
, {-26, 54, 40}
, {51, 44, -18}
, {-16, 34, 39}
, {23, 39, 17}
, {-8, 0, 7}
, {52, -2, -8}
, {-18, -14, -29}
, {27, -43, 26}
, {-30, -24, -29}
, {-44, -54, 37}
, {4, 38, 3}
, {-8, 11, 9}
, {-8, -19, -42}
, {-36, 50, 18}
, {-24, -3, -42}
, {1, 12, -17}
, {-51, 17, -54}
, {-32, 10, -11}
, {-21, -5, 7}
, {16, -20, -15}
, {-18, -39, 59}
, {-45, 12, 32}
, {58, 62, -27}
, {39, 24, -4}
, {-47, -25, -47}
, {20, 6, 38}
, {-41, -45, 33}
, {-6, 7, -28}
, {-18, -7, -31}
, {10, -55, -11}
, {54, -26, 23}
, {23, -5, -13}
, {24, -52, -24}
, {15, -55, 25}
}
, {{-45, -4, 20}
, {26, 19, 37}
, {-9, -5, 8}
, {51, -31, 39}
, {12, 47, 22}
, {-27, 33, -13}
, {42, -27, -27}
, {31, 3, -12}
, {47, -29, 0}
, {-22, 35, -23}
, {1, 14, 16}
, {48, 50, 44}
, {-30, 34, 15}
, {29, 1, 32}
, {-40, -5, 30}
, {-44, 3, -3}
, {-51, -12, -23}
, {-40, -13, 34}
, {3, 5, 44}
, {-39, 33, -4}
, {-62, -34, -63}
, {-10, 39, 40}
, {6, -23, 25}
, {2, 19, -41}
, {33, 49, -19}
, {8, -19, 26}
, {-41, -57, -47}
, {37, 0, 14}
, {-37, -17, -41}
, {-35, -42, -43}
, {49, -21, 28}
, {-5, -36, -28}
, {-15, 24, 21}
, {38, -3, 30}
, {-18, 43, -48}
, {36, 16, 3}
, {-21, -36, -30}
, {32, 15, 18}
, {40, 57, -34}
, {-41, 28, 36}
, {-24, 27, -33}
, {-44, -16, 18}
, {17, 7, -27}
, {-19, -25, 5}
, {-28, 36, 55}
, {33, -28, 18}
, {3, -9, -43}
, {-4, 3, 5}
, {-33, 46, 20}
, {49, 6, 14}
, {46, -8, 30}
, {-6, -11, 52}
, {-32, -35, 13}
, {51, 5, -13}
, {26, 53, -29}
, {8, 28, 15}
, {3, 56, 23}
, {14, -12, 32}
, {-34, 23, -39}
, {-30, 12, 20}
, {-27, 6, 24}
, {36, 26, -2}
, {-8, -48, -6}
, {-12, -24, -42}
}
, {{29, -45, 5}
, {10, 35, -45}
, {50, -40, -22}
, {-1, -19, 3}
, {-41, 6, -49}
, {9, 26, -46}
, {25, -22, 14}
, {-7, -49, 5}
, {-27, 20, -41}
, {7, 13, -43}
, {37, -19, 17}
, {-48, -33, 40}
, {-44, -31, 49}
, {-11, 9, -45}
, {1, -14, -4}
, {64, 65, 52}
, {41, -31, 24}
, {-18, -11, -40}
, {50, -19, 30}
, {-17, 35, 24}
, {13, -21, 45}
, {34, 26, 36}
, {30, 47, -30}
, {28, -29, -10}
, {25, 24, -28}
, {-19, 27, 37}
, {-41, 53, 35}
, {40, -12, -41}
, {-32, -27, 14}
, {49, 19, 35}
, {32, 44, -34}
, {26, 31, 28}
, {-6, 8, -50}
, {62, 22, -21}
, {21, 22, 14}
, {-38, 14, 44}
, {-36, -3, -6}
, {49, -8, 39}
, {34, 2, 22}
, {-25, -46, -31}
, {-23, -1, -36}
, {57, 0, 40}
, {40, 0, -5}
, {36, -5, 13}
, {-2, 43, 41}
, {-44, 6, -24}
, {0, 53, 51}
, {8, 31, 29}
, {-31, 35, -9}
, {-6, 25, 45}
, {-2, 40, -26}
, {-4, -23, -2}
, {26, 13, -34}
, {-43, 52, -51}
, {-22, 2, -26}
, {-26, -13, -12}
, {-36, -4, 31}
, {-30, -47, -11}
, {-35, 9, 7}
, {50, 55, -19}
, {38, 14, 37}
, {-36, 44, -33}
, {-26, -24, -39}
, {52, 33, 22}
}
, {{42, -31, 13}
, {-8, 48, -8}
, {1, 2, -45}
, {55, -46, -26}
, {-38, -10, -18}
, {16, -18, 0}
, {12, -26, 0}
, {-46, -52, -22}
, {20, 49, -31}
, {-9, -45, -48}
, {27, -28, 39}
, {33, 49, -11}
, {-10, -41, 27}
, {-25, -2, -68}
, {-10, -26, -26}
, {-1, -24, -10}
, {-47, -16, -32}
, {34, -32, 49}
, {-39, 5, 3}
, {23, -64, -19}
, {-20, -39, 25}
, {-2, -2, -6}
, {41, -18, -32}
, {26, 27, 27}
, {17, 44, 0}
, {11, 4, -27}
, {-30, -5, 7}
, {-5, -48, -22}
, {2, -48, 38}
, {50, -34, -18}
, {28, 45, -24}
, {-29, 47, 53}
, {-28, -43, -13}
, {10, 1, -33}
, {13, -15, 21}
, {25, -36, 62}
, {19, 2, -41}
, {-41, 5, -42}
, {0, 51, 58}
, {42, 18, 34}
, {-31, -50, 14}
, {-22, -22, 10}
, {41, 16, -12}
, {31, -48, -16}
, {29, 59, 28}
, {-18, 54, -3}
, {-20, -10, -19}
, {-29, 35, 44}
, {-5, 18, -15}
, {-38, 40, -3}
, {13, -34, -2}
, {19, 32, 10}
, {14, -51, 0}
, {21, 1, 7}
, {-22, -38, -19}
, {19, 49, 53}
, {30, -24, -18}
, {34, -25, -28}
, {-10, 35, 14}
, {15, -17, 40}
, {-9, -65, -29}
, {-30, 12, 5}
, {48, 23, 28}
, {17, 47, 18}
}
, {{28, 2, -9}
, {26, -47, 33}
, {-11, -2, 19}
, {-20, 9, 5}
, {48, -42, 2}
, {5, 28, -16}
, {9, 15, 41}
, {-47, -36, 29}
, {36, 0, 47}
, {39, 33, -19}
, {-11, -47, 26}
, {-37, -47, 23}
, {-36, 23, -41}
, {26, 14, 59}
, {7, 8, 35}
, {23, 1, 14}
, {-9, 17, 17}
, {38, 22, -49}
, {11, -46, 42}
, {39, -18, 46}
, {-35, 27, -25}
, {-26, -43, 26}
, {-28, 50, 22}
, {-54, 21, 24}
, {-33, 17, 10}
, {1, -1, -14}
, {-38, -5, 35}
, {52, -9, -8}
, {-33, -21, -18}
, {44, 25, -14}
, {25, -1, 0}
, {4, -36, 36}
, {43, -35, -39}
, {6, -4, 12}
, {-7, -1, -16}
, {-11, -27, 26}
, {-42, -43, -1}
, {1, -4, -45}
, {-38, -30, -37}
, {5, 41, 39}
, {-8, 29, 45}
, {-34, -41, -1}
, {-8, 24, -10}
, {31, -5, 28}
, {-8, -2, 19}
, {39, -18, 28}
, {-19, 38, -38}
, {-39, 44, 10}
, {-43, 9, 16}
, {-32, 21, -4}
, {14, -34, 23}
, {33, -8, 36}
, {-12, 15, -7}
, {26, -18, -15}
, {-34, 10, -20}
, {27, 19, -22}
, {1, 34, -37}
, {-11, 37, 22}
, {-25, -9, 18}
, {66, 49, 51}
, {29, 52, 4}
, {-1, -23, 28}
, {-29, 22, 16}
, {29, 14, 21}
}
, {{-37, -39, 20}
, {43, 24, 54}
, {42, 5, 53}
, {25, 36, -22}
, {51, 23, 42}
, {-17, -1, 39}
, {-42, -42, 18}
, {34, -20, 42}
, {-47, -21, -51}
, {22, -18, 49}
, {55, 40, 35}
, {-43, -15, -16}
, {27, -27, -15}
, {-8, 50, 26}
, {-54, 0, -52}
, {-18, 29, -28}
, {-5, -51, -56}
, {8, 39, 1}
, {-36, 7, 27}
, {-34, -7, 24}
, {-48, -11, -43}
, {15, -10, -37}
, {-6, 1, 14}
, {25, 46, -3}
, {39, -1, 17}
, {40, -47, -38}
, {-27, -17, 53}
, {12, 14, -28}
, {-23, -1, -33}
, {19, 51, 44}
, {2, 36, -1}
, {-5, -34, 43}
, {52, -25, 13}
, {14, 4, -35}
, {-17, 46, 11}
, {2, -11, -17}
, {40, -36, 52}
, {-11, 32, 8}
, {49, -1, -1}
, {30, -47, -41}
, {31, 48, -26}
, {-35, 8, -8}
, {-35, 9, -38}
, {-47, 1, 48}
, {-19, -30, 52}
, {11, 50, 40}
, {-27, -35, -12}
, {45, 42, 39}
, {-49, -29, -24}
, {-2, -5, -39}
, {1, -25, -34}
, {-18, -11, -2}
, {13, -40, 5}
, {29, -10, -42}
, {-7, 23, -39}
, {-59, -17, 12}
, {-3, 52, -6}
, {-40, 17, 36}
, {3, -33, -42}
, {33, -41, -40}
, {-26, -19, -35}
, {17, 2, 12}
, {-24, -3, 29}
, {-47, 4, -46}
}
, {{49, -14, 38}
, {-16, 17, 15}
, {-21, -32, -20}
, {14, -41, 53}
, {40, 30, -21}
, {-20, -29, -47}
, {23, 11, -14}
, {25, -50, -42}
, {3, -49, 16}
, {54, 21, 51}
, {-46, -38, -1}
, {-4, 19, -6}
, {-39, 61, 44}
, {4, -3, 14}
, {-19, 49, 13}
, {-19, 30, 42}
, {44, -52, -52}
, {-4, 45, -15}
, {-2, -7, 23}
, {22, -5, 2}
, {61, 32, 64}
, {3, -54, 0}
, {47, -26, 26}
, {6, -37, -42}
, {-7, -43, -35}
, {13, -17, 37}
, {-6, 13, 5}
, {-20, -25, -38}
, {39, 18, -13}
, {28, 26, -29}
, {-22, -1, 29}
, {-9, 44, 17}
, {-44, 16, -40}
, {1, 67, 0}
, {61, 48, 5}
, {-1, 37, 7}
, {-7, 5, 32}
, {-24, 35, -35}
, {-24, -14, -14}
, {38, -36, 42}
, {-5, 5, 27}
, {-5, 44, -5}
, {-43, 40, 18}
, {24, 26, -20}
, {7, -6, 35}
, {-10, -35, 5}
, {-29, -14, -25}
, {38, -53, -45}
, {27, -35, -6}
, {11, -47, -38}
, {-35, -24, -25}
, {57, -20, 47}
, {57, 17, 18}
, {-33, 31, 38}
, {5, 53, 40}
, {-9, 48, 56}
, {-45, -49, -31}
, {28, 6, 5}
, {-45, 47, -35}
, {30, 7, 36}
, {-18, 20, -1}
, {4, 46, -17}
, {27, -44, -52}
, {47, 39, -21}
}
, {{-19, 48, -42}
, {-52, 25, -49}
, {-3, 22, 13}
, {-23, 39, -37}
, {29, 17, 15}
, {-46, -1, -47}
, {-12, -53, -39}
, {8, -34, 58}
, {-46, 44, -6}
, {24, -10, -48}
, {-23, -28, -31}
, {-54, 52, 46}
, {-47, 3, 46}
, {9, -18, -31}
, {33, -15, -12}
, {29, -45, -38}
, {5, -38, 0}
, {25, -12, 14}
, {39, -17, 13}
, {-1, 40, 27}
, {20, 12, 9}
, {57, 55, -21}
, {-50, 0, 27}
, {35, 32, 49}
, {31, 35, 32}
, {-39, 34, -53}
, {45, 18, 14}
, {-45, 16, 46}
, {-6, -55, 10}
, {20, -34, -4}
, {47, 26, 6}
, {44, 0, 17}
, {6, -6, 18}
, {-49, -5, 9}
, {26, 13, -59}
, {-25, -8, 3}
, {-1, -28, 40}
, {-21, 24, 2}
, {42, -18, 2}
, {9, -19, -45}
, {11, -6, 8}
, {-25, 3, 34}
, {-19, -58, 31}
, {33, 11, -9}
, {-5, -25, 48}
, {-8, 50, 28}
, {1, 0, 29}
, {56, -15, 48}
, {46, -50, 33}
, {-1, -33, 29}
, {13, 16, -15}
, {38, -8, 25}
, {16, 28, 4}
, {4, -17, 29}
, {7, -38, 36}
, {32, -7, -24}
, {6, 11, 4}
, {36, -65, 53}
, {-23, -22, 18}
, {-61, 29, -22}
, {-1, -20, 6}
, {36, 8, 38}
, {39, 1, 28}
, {31, 2, 50}
}
, {{1, -16, -43}
, {19, -56, -2}
, {-52, 19, 33}
, {2, -47, 37}
, {22, -5, 6}
, {-6, 41, -17}
, {-16, -53, -37}
, {-24, -16, 44}
, {46, -1, -11}
, {-28, -41, 12}
, {44, 14, -17}
, {-44, 41, 14}
, {12, 33, -4}
, {8, 22, 8}
, {-38, -40, 7}
, {-37, 20, 32}
, {-22, -11, -9}
, {17, -17, 43}
, {30, -28, -34}
, {8, -30, 1}
, {-26, 5, 14}
, {-27, 24, -25}
, {-42, -15, 36}
, {-26, -4, -26}
, {15, -43, -52}
, {-5, -7, -26}
, {-26, 43, 5}
, {15, -33, -45}
, {8, -42, 33}
, {-54, -28, 17}
, {32, 6, 28}
, {-7, 34, -55}
, {-3, 9, -25}
, {54, 0, 16}
, {9, -44, -30}
, {48, 29, 19}
, {9, 36, -14}
, {-14, 2, 9}
, {-19, 34, 12}
, {41, -6, -20}
, {33, -45, 8}
, {-37, -4, -45}
, {42, 8, 44}
, {-33, 32, -6}
, {-16, -22, 25}
, {-31, -20, 24}
, {33, 8, 20}
, {21, 0, -56}
, {10, -7, 0}
, {-53, -32, 52}
, {-4, 47, -29}
, {33, -48, -40}
, {41, -14, -30}
, {11, -24, 1}
, {-29, 46, -34}
, {-5, 28, 0}
, {27, -28, 18}
, {-16, 19, 36}
, {-52, -56, -17}
, {-33, 15, -34}
, {18, 22, -35}
, {35, -43, 18}
, {-51, -55, 47}
, {19, 12, 32}
}
, {{-17, 0, 22}
, {-2, 2, 29}
, {2, 25, 34}
, {15, 54, 30}
, {-41, 45, 17}
, {9, 13, 43}
, {-48, 15, -6}
, {46, -37, 1}
, {-51, -56, -18}
, {-36, 24, 33}
, {17, 52, -23}
, {-22, 2, -34}
, {-36, 39, 38}
, {-36, 20, -25}
, {20, -48, -57}
, {-21, 17, -11}
, {-3, 11, -60}
, {27, 5, 14}
, {14, 45, 0}
, {-26, -16, -9}
, {6, 26, -23}
, {-8, 0, -20}
, {48, 23, -25}
, {29, 4, 18}
, {48, 41, 4}
, {-27, 18, -33}
, {14, 15, 22}
, {21, 34, 18}
, {-31, -30, 27}
, {-42, -11, -39}
, {-1, -17, 28}
, {-26, 52, -1}
, {14, 27, -22}
, {15, 37, -19}
, {34, 49, -10}
, {-4, -35, 6}
, {6, 43, 36}
, {44, 40, 13}
, {40, -7, 8}
, {5, -21, 0}
, {-14, 19, 3}
, {27, 1, 16}
, {-48, 32, -17}
, {-48, 4, 26}
, {-13, -25, 10}
, {-14, -30, -26}
, {36, -7, -34}
, {-8, 17, 16}
, {16, 24, -12}
, {-42, -18, -57}
, {48, -40, -38}
, {-27, 32, -18}
, {10, -3, 40}
, {-41, -36, 12}
, {-9, -18, -3}
, {-57, 27, -37}
, {0, 4, 35}
, {-45, -7, 40}
, {25, -1, -29}
, {-2, -6, 3}
, {30, -7, -21}
, {35, 12, 0}
, {33, 0, -7}
, {32, -52, -36}
}
, {{-35, -25, 51}
, {-35, 0, -51}
, {-12, -30, 51}
, {50, -49, -54}
, {16, 41, -23}
, {-14, 42, 37}
, {52, 20, 56}
, {10, 35, -7}
, {22, -6, -38}
, {-34, -32, -9}
, {38, -41, -23}
, {-27, -7, 30}
, {55, 16, -7}
, {66, -6, 2}
, {55, 4, -34}
, {4, 12, 10}
, {8, 14, 23}
, {25, -47, 34}
, {-11, 52, 4}
, {65, 63, 38}
, {-19, 23, -2}
, {-52, -34, -1}
, {-9, 28, -25}
, {17, 42, -14}
, {27, 33, 40}
, {24, -16, -31}
, {17, 68, 52}
, {53, -50, 38}
, {10, 55, 60}
, {-7, -16, -31}
, {19, 40, -12}
, {22, 0, -39}
, {-34, -45, -33}
, {-15, 24, 11}
, {-30, 39, 3}
, {13, -42, 2}
, {-47, 48, 3}
, {24, -49, -31}
, {-6, 5, -54}
, {-41, -34, 2}
, {42, 23, -53}
, {-38, -7, -18}
, {3, 2, 43}
, {31, -24, 49}
, {4, 40, 44}
, {-24, -9, 45}
, {57, 6, -37}
, {-2, 33, 7}
, {53, 46, 16}
, {45, 34, 42}
, {2, 6, 1}
, {-25, 12, -28}
, {-22, -17, -39}
, {35, -42, -11}
, {-42, 2, 0}
, {-42, -38, 53}
, {-53, -10, -42}
, {5, 4, 1}
, {-9, -26, -34}
, {25, 30, -14}
, {16, -30, 59}
, {10, -3, -39}
, {0, -18, 37}
, {-49, 48, 9}
}
, {{0, -29, 55}
, {-12, -33, -9}
, {-45, 25, 49}
, {-21, 57, -4}
, {57, -38, -5}
, {-39, 44, 3}
, {18, -37, -5}
, {36, 42, -13}
, {-2, 7, 39}
, {-24, -35, -1}
, {36, -19, 41}
, {2, 10, 35}
, {27, 4, -10}
, {15, -21, 3}
, {17, -10, -35}
, {34, 19, 15}
, {-60, -29, -29}
, {46, 21, 20}
, {46, 15, -10}
, {-39, 49, 23}
, {57, 63, 44}
, {-63, -56, -42}
, {-28, 47, -20}
, {20, -12, -26}
, {30, 38, -17}
, {-15, 10, -19}
, {-23, 13, -8}
, {12, -40, -12}
, {35, -39, 24}
, {27, -15, 48}
, {-8, 2, -11}
, {39, -2, -39}
, {-37, -57, 41}
, {58, 32, -4}
, {65, 43, 11}
, {-17, -21, 18}
, {62, 13, 45}
, {32, 42, 51}
, {-27, -44, 19}
, {37, -30, -39}
, {29, 24, 41}
, {6, -21, 40}
, {30, 20, 1}
, {-3, -2, -7}
, {27, 11, -7}
, {-33, 15, 50}
, {19, -6, 7}
, {-60, -41, 49}
, {36, 18, -39}
, {-33, -34, -63}
, {41, 58, -13}
, {-2, 36, 13}
, {7, 44, 5}
, {-21, -40, -14}
, {-11, -6, -6}
, {14, -14, -35}
, {-38, 20, -49}
, {-5, 8, -23}
, {9, 47, 3}
, {43, 34, 42}
, {-46, -40, 35}
, {-24, -46, -17}
, {40, -52, -41}
, {39, 4, -24}
}
, {{47, -2, -7}
, {-32, -19, 45}
, {8, -39, 0}
, {29, 2, 4}
, {-8, -2, -43}
, {52, 50, -35}
, {-2, 20, -51}
, {-18, 20, 54}
, {-35, 11, -49}
, {-26, -31, 42}
, {54, 20, 30}
, {33, 9, -12}
, {8, 47, 6}
, {18, 27, 62}
, {26, 13, 15}
, {-9, 1, 30}
, {-24, -9, -12}
, {-26, -35, 16}
, {-45, 30, 25}
, {-5, 40, 59}
, {-41, -21, 16}
, {26, 46, 35}
, {20, 43, 9}
, {-12, 0, 50}
, {-29, -7, -41}
, {-28, -21, -15}
, {-61, 0, -30}
, {9, 32, 14}
, {22, -25, 11}
, {-1, 32, -40}
, {34, -6, 25}
, {4, -17, 25}
, {62, -41, 45}
, {31, -45, -61}
, {-47, 12, 24}
, {-47, 30, -23}
, {11, -42, 43}
, {-12, 39, 39}
, {-5, -37, 39}
, {-13, -3, 32}
, {0, -8, 12}
, {-45, -11, 24}
, {35, -45, -10}
, {16, 23, -6}
, {18, 9, 27}
, {-8, -29, -9}
, {-45, -32, -23}
, {52, 28, 40}
, {-4, -4, 53}
, {31, -12, -50}
, {29, 40, 14}
, {34, -39, 29}
, {-30, -13, -36}
, {-22, -42, 11}
, {23, 29, -12}
, {16, -58, -39}
, {1, -18, 30}
, {14, -11, -24}
, {-39, -37, 25}
, {-23, 25, -22}
, {-13, 29, 60}
, {52, -7, -4}
, {-13, 60, 50}
, {8, 35, -6}
}
, {{4, -31, 0}
, {-21, -40, -25}
, {-42, 46, -11}
, {-42, 18, -29}
, {0, 35, 36}
, {-37, -26, 32}
, {-10, -8, -14}
, {-33, 19, -51}
, {14, 4, -18}
, {-20, 41, 20}
, {0, 5, 10}
, {-41, 9, -17}
, {0, 28, -39}
, {5, -36, 26}
, {-31, -29, -29}
, {-14, 6, -32}
, {33, 19, 19}
, {-53, 34, -3}
, {-38, -19, 21}
, {29, -16, 15}
, {23, -60, -1}
, {48, -9, -23}
, {31, 17, 17}
, {-15, -19, -52}
, {42, 17, 8}
, {-10, 36, 42}
, {-34, 30, 26}
, {7, -10, -32}
, {-6, -38, 35}
, {-21, 52, 0}
, {-18, -19, 19}
, {-29, 17, 51}
, {42, -39, 5}
, {-8, -51, -50}
, {-52, -5, -53}
, {-5, 16, -44}
, {-34, 7, -46}
, {7, -4, -28}
, {-5, 8, -42}
, {26, 17, 18}
, {32, -17, 24}
, {34, 0, 15}
, {9, 6, -50}
, {34, 39, 48}
, {35, 26, 43}
, {-41, -23, -30}
, {-56, 15, 39}
, {53, -39, -31}
, {23, -48, -37}
, {17, 59, -30}
, {-40, 21, -31}
, {36, -7, 7}
, {-52, 30, 43}
, {42, 26, -23}
, {-44, -41, 43}
, {-40, 25, 27}
, {-31, 46, 28}
, {-23, 44, 54}
, {31, -15, 48}
, {1, 64, 2}
, {-3, 6, 7}
, {51, 43, -7}
, {55, 9, 32}
, {50, 13, 25}
}
, {{22, 5, -4}
, {-53, -9, -25}
, {-33, 24, 4}
, {-41, -1, -24}
, {-8, -27, 20}
, {-37, 7, 12}
, {-7, 44, -29}
, {9, 19, -48}
, {-25, 21, 49}
, {-13, 37, 20}
, {26, 25, -55}
, {29, 46, -15}
, {-31, 49, -6}
, {63, -42, -32}
, {41, -17, -6}
, {5, -7, 15}
, {53, 57, 6}
, {28, 13, 48}
, {7, -18, 14}
, {-29, 25, 28}
, {-40, 35, 25}
, {46, 48, 49}
, {33, -16, -14}
, {-39, 51, 26}
, {50, -22, -30}
, {25, 17, 46}
, {8, 16, -61}
, {-45, 20, -37}
, {47, -19, 27}
, {-38, -20, -27}
, {-4, 47, -19}
, {-34, 48, -31}
, {-32, 19, -14}
, {-27, -21, 49}
, {20, 0, 10}
, {55, 19, -1}
, {28, -7, -46}
, {-51, -49, -9}
, {33, 22, 34}
, {-25, -47, -49}
, {-12, 10, -33}
, {-26, 44, -17}
, {56, 52, -18}
, {-28, 13, -31}
, {6, -28, 41}
, {12, 24, -3}
, {27, 20, 11}
, {15, -9, -7}
, {-4, 25, -2}
, {20, 2, 3}
, {-51, 25, -37}
, {6, -6, -35}
, {-20, 38, 19}
, {-43, 8, 43}
, {24, 45, -25}
, {-22, 0, -9}
, {-38, 34, -15}
, {-49, 60, 7}
, {32, -45, 13}
, {-4, -12, -19}
, {-29, 27, 33}
, {-35, -3, -51}
, {0, 32, -40}
, {-25, -29, 30}
}
, {{31, -7, -41}
, {-14, -37, 40}
, {31, -31, -21}
, {-23, 26, -13}
, {-3, -7, 20}
, {39, 45, -14}
, {20, -43, 27}
, {-35, -43, 11}
, {-19, -39, -34}
, {34, 40, -53}
, {-29, 3, 50}
, {38, -7, -4}
, {-5, -3, -28}
, {-38, -26, -14}
, {42, 39, -51}
, {54, -28, 28}
, {-10, 22, 3}
, {28, 45, -10}
, {33, 27, -45}
, {-2, -2, -39}
, {-5, 31, -38}
, {53, 12, -17}
, {-32, -9, 15}
, {26, -35, -52}
, {-31, 30, 44}
, {19, 5, -43}
, {-34, -44, -40}
, {-23, 44, 20}
, {31, 45, 48}
, {46, 50, 43}
, {-50, 17, 16}
, {-16, 46, -23}
, {40, 59, -46}
, {-43, -48, 1}
, {36, -25, -27}
, {43, -20, -13}
, {-40, -24, -38}
, {-37, 9, 32}
, {45, 9, -4}
, {2, 30, -6}
, {29, 11, 33}
, {-8, -8, 34}
, {38, -28, 2}
, {37, -11, 26}
, {-4, 8, 49}
, {56, 37, -3}
, {22, -53, -15}
, {19, 54, -3}
, {24, 52, 37}
, {49, 16, -1}
, {-11, -4, -18}
, {4, 46, 14}
, {-59, 36, -38}
, {41, -17, 18}
, {-20, -22, -24}
, {26, -52, -15}
, {58, 55, 11}
, {-40, 27, 54}
, {-31, 7, 24}
, {-20, -12, 26}
, {-9, -30, -46}
, {8, -15, -13}
, {21, -4, 24}
, {5, -3, -36}
}
, {{-1, -45, 31}
, {22, -11, 40}
, {44, 1, -22}
, {-5, 23, -4}
, {15, 41, -40}
, {46, -40, 10}
, {-22, -10, 33}
, {-13, 21, 10}
, {59, -31, 47}
, {27, -57, 33}
, {-37, -17, 42}
, {28, 4, 22}
, {20, -13, -41}
, {-6, 2, 30}
, {-21, 4, -25}
, {-15, 13, 25}
, {3, -20, 61}
, {-26, -34, 30}
, {-16, -45, 23}
, {-27, -3, -19}
, {18, 20, 3}
, {18, -23, 23}
, {-21, 32, -35}
, {-29, 44, -26}
, {-12, -17, -35}
, {34, -46, 15}
, {-51, -60, 27}
, {32, -32, -34}
, {29, 50, -3}
, {-32, 4, -12}
, {29, -5, -17}
, {-52, 43, 5}
, {30, -2, 32}
, {5, 29, -49}
, {-48, -58, 6}
, {-31, 35, 45}
, {8, 35, 13}
, {-9, -42, 9}
, {-56, -22, 26}
, {28, -46, -41}
, {21, 34, 32}
, {-40, 4, -13}
, {-37, 15, -19}
, {-42, -24, -42}
, {15, 40, 0}
, {-12, -10, 42}
, {-24, -27, 26}
, {46, -48, 45}
, {10, 28, 13}
, {-11, 2, 2}
, {14, -51, -8}
, {11, 4, 26}
, {-17, 23, 12}
, {0, -18, -7}
, {-24, -1, 2}
, {40, 46, 52}
, {52, 34, 33}
, {10, -34, -21}
, {18, 17, -26}
, {66, 45, 49}
, {-44, -27, -31}
, {-34, -31, -8}
, {51, 6, -46}
, {55, 8, -11}
}
, {{-4, 3, 10}
, {21, 3, -36}
, {-36, 15, -44}
, {-14, 20, 0}
, {-22, -29, -53}
, {-16, 10, -18}
, {25, 55, -25}
, {-1, -13, -25}
, {-40, -26, -13}
, {6, -17, 6}
, {6, 0, 28}
, {13, 50, 23}
, {-10, 6, -43}
, {-41, -8, -37}
, {60, -3, -9}
, {48, 30, 9}
, {55, 41, -7}
, {31, 19, -51}
, {22, -37, -20}
, {-20, -38, -21}
, {-22, -16, 27}
, {50, 36, -2}
, {33, 24, 5}
, {42, -12, -42}
, {41, -38, -54}
, {2, -1, -13}
, {46, -52, 27}
, {55, 34, 8}
, {6, 0, -21}
, {-29, 35, -27}
, {29, 0, 15}
, {-41, 31, 24}
, {30, 21, 52}
, {-22, 39, -38}
, {-20, 20, 25}
, {51, 3, 31}
, {-10, 24, -37}
, {-37, -54, -1}
, {44, -28, 16}
, {-2, 9, -15}
, {-28, 23, -7}
, {-51, 27, 25}
, {25, 20, 12}
, {49, 9, 40}
, {4, -34, -28}
, {39, -1, 1}
, {-28, 34, 10}
, {-34, 42, 5}
, {-18, 37, -12}
, {59, 50, 55}
, {-22, -18, -8}
, {40, -24, 0}
, {32, -11, -52}
, {0, 38, 38}
, {2, 37, -42}
, {-38, -14, -41}
, {44, -28, 32}
, {67, 5, 35}
, {21, 8, -39}
, {-33, 65, 63}
, {27, 23, -50}
, {5, -4, 3}
, {-31, 19, 58}
, {3, 15, 34}
}
, {{-32, -36, -54}
, {13, -39, 18}
, {-29, -15, 38}
, {10, 14, 9}
, {24, 31, -45}
, {-27, -42, 24}
, {22, 23, 48}
, {-31, -48, -14}
, {-19, 40, -23}
, {17, -26, -13}
, {-50, -28, -44}
, {-21, 14, 0}
, {31, -41, 13}
, {2, 52, -5}
, {-44, 22, -33}
, {-11, -2, 20}
, {43, -6, -1}
, {-25, -28, -36}
, {41, 15, 38}
, {-11, 43, -9}
, {-22, -45, -17}
, {42, -55, 24}
, {-55, -25, 4}
, {-20, -12, -11}
, {-42, -31, -44}
, {-8, -44, 12}
, {-10, -9, 2}
, {-3, 23, 9}
, {-29, -11, -38}
, {0, -35, -29}
, {15, -4, -23}
, {30, -47, 17}
, {18, 44, -39}
, {33, 23, 16}
, {42, 9, -5}
, {-22, 45, 44}
, {24, 0, 1}
, {36, -41, 20}
, {31, 19, -14}
, {-44, -21, 40}
, {-18, -17, 30}
, {37, 0, -23}
, {-10, -29, 25}
, {0, -6, 8}
, {38, -19, 27}
, {20, 4, 18}
, {-2, 20, -14}
, {-18, -9, 14}
, {40, -28, 19}
, {47, 61, 26}
, {27, 11, 6}
, {46, -3, 45}
, {-52, 13, 29}
, {7, -50, 31}
, {-49, 1, -33}
, {23, -11, -33}
, {-29, 39, -14}
, {37, 43, -33}
, {-22, -49, 21}
, {-2, -28, -5}
, {16, -20, 30}
, {-18, 26, 16}
, {-42, 24, 40}
, {-36, 14, 10}
}
, {{16, 37, -19}
, {-7, 8, -32}
, {15, 5, -10}
, {-39, -43, 19}
, {-39, -13, 41}
, {-41, -40, -48}
, {-7, -40, 21}
, {5, -44, -54}
, {26, -52, 36}
, {-55, -12, -21}
, {-32, -43, 45}
, {-28, -40, 15}
, {-28, 17, 32}
, {31, 7, -38}
, {2, -35, -30}
, {9, 2, -49}
, {28, -45, 26}
, {-38, 10, 20}
, {-32, -27, -7}
, {-30, 11, 32}
, {8, -4, 28}
, {-33, -4, -20}
, {39, 14, -47}
, {-36, 26, 22}
, {26, 47, 7}
, {-1, 40, -38}
, {20, 35, 18}
, {23, -17, 44}
, {13, -18, 0}
, {5, 31, -39}
, {-8, -31, -53}
, {-37, -17, -7}
, {-14, -25, 39}
, {-47, 39, -30}
, {20, 45, -33}
, {-54, 8, -56}
, {33, 17, -50}
, {-41, -7, -22}
, {35, 1, -16}
, {-6, 44, 22}
, {-41, 46, 30}
, {-9, 13, -1}
, {-11, -25, 20}
, {-25, -4, -42}
, {15, 17, -4}
, {-50, -13, -15}
, {-40, -22, 27}
, {-8, 12, 20}
, {-22, -4, 14}
, {-10, 28, 32}
, {9, 11, 15}
, {-14, -3, 46}
, {15, 8, 30}
, {-50, -53, -27}
, {-16, 15, -50}
, {12, 5, -17}
, {-23, -47, -5}
, {-7, 43, 48}
, {-50, 5, 35}
, {1, 13, -5}
, {26, -39, 40}
, {-54, 28, -30}
, {-8, -21, 10}
, {19, 40, 34}
}
, {{5, -7, -29}
, {-13, -13, 34}
, {-28, 2, -41}
, {0, 0, 35}
, {-41, -13, 46}
, {0, -1, -37}
, {13, 59, 28}
, {-24, 39, 31}
, {30, -18, 0}
, {34, -33, 38}
, {-22, 20, -4}
, {10, -29, 40}
, {-4, -15, 22}
, {16, 3, 36}
, {-21, 66, -17}
, {-1, -14, -13}
, {12, 20, -29}
, {25, -46, -1}
, {-34, -33, 4}
, {-9, 31, -25}
, {5, 14, 0}
, {31, 30, 53}
, {-12, -35, 55}
, {53, -26, 2}
, {14, -4, 34}
, {-6, 45, 56}
, {17, -18, 31}
, {2, -20, 28}
, {-17, 62, 28}
, {-17, 4, -12}
, {51, -39, -36}
, {32, 1, 35}
, {18, 55, 8}
, {-21, 23, -3}
, {-44, -49, 40}
, {13, 10, -15}
, {-41, 9, 41}
, {-21, -8, -16}
, {-16, -36, 7}
, {-35, -44, 18}
, {-32, -33, 16}
, {7, -42, 28}
, {46, -8, -18}
, {22, 6, 10}
, {-49, 8, 10}
, {49, 46, -25}
, {-42, -59, -36}
, {31, 12, -30}
, {11, -17, -32}
, {17, 11, -16}
, {5, 40, -43}
, {17, 33, 30}
, {-18, 30, 31}
, {15, 10, -40}
, {-7, -31, 0}
, {-22, -30, 41}
, {-27, -48, -20}
, {-41, -22, -9}
, {26, -36, 29}
, {-16, 22, 10}
, {-6, 1, 9}
, {-34, 57, 19}
, {-10, 14, -45}
, {-50, -34, 46}
}
, {{49, 24, 39}
, {28, 32, 16}
, {0, 62, -31}
, {9, -4, 23}
, {31, -23, -25}
, {-1, -47, 41}
, {46, -34, 6}
, {-5, 5, 41}
, {5, 20, -48}
, {-26, 39, 47}
, {4, 20, -47}
, {-8, -30, -18}
, {48, -27, 40}
, {11, -51, 64}
, {42, 65, 60}
, {-12, 34, -56}
, {43, 37, 54}
, {27, -28, 4}
, {2, 40, -47}
, {54, 11, 58}
, {-6, -37, -1}
, {-44, 39, -6}
, {-3, 9, -30}
, {-42, 8, 54}
, {41, -14, 48}
, {50, 15, -24}
, {-53, 20, -18}
, {-11, 39, 31}
, {8, -9, -42}
, {-1, 30, 0}
, {-34, 33, -20}
, {16, -31, -5}
, {-43, 50, -46}
, {56, 14, -22}
, {-18, -29, -29}
, {16, 11, -54}
, {44, -35, -36}
, {41, -31, -30}
, {-9, -39, 15}
, {-54, -57, -38}
, {-16, -19, -39}
, {49, 35, -5}
, {37, -39, 9}
, {22, 34, -31}
, {-2, -35, 34}
, {10, 33, -40}
, {13, 28, -52}
, {0, 15, -37}
, {29, 54, -44}
, {2, 9, -34}
, {-33, 29, -42}
, {-53, 36, -15}
, {7, -28, -18}
, {-24, -45, 45}
, {10, 53, -52}
, {-41, -6, 4}
, {21, -51, 28}
, {-6, 51, -41}
, {-8, 0, -37}
, {6, -25, -29}
, {7, -18, 17}
, {47, 7, 36}
, {14, 22, 46}
, {31, -53, -43}
}
, {{17, -39, -1}
, {-15, 20, 14}
, {-33, -30, 26}
, {47, -53, 35}
, {-20, 4, -17}
, {12, -16, -22}
, {-35, 30, 21}
, {9, 30, 22}
, {54, -25, 53}
, {-12, -19, 24}
, {-23, 40, 28}
, {50, -31, -10}
, {-17, -45, 15}
, {39, 0, -25}
, {55, 56, -1}
, {13, 29, 56}
, {-11, 38, 23}
, {-36, -4, 40}
, {4, -44, 33}
, {16, 26, 23}
, {9, 2, -18}
, {34, -2, 31}
, {-34, 3, 14}
, {-20, -9, 10}
, {-26, -37, -11}
, {35, -37, -5}
, {-55, -32, 1}
, {-24, 13, -8}
, {-2, -5, -2}
, {9, -14, -18}
, {-47, -2, 54}
, {24, -21, -23}
, {34, 5, -15}
, {-18, 19, 8}
, {-38, 46, 42}
, {-21, 58, 0}
, {20, 4, 31}
, {-27, 48, -24}
, {-44, -36, -39}
, {-52, -5, -41}
, {-42, 26, -3}
, {-25, 45, 52}
, {-3, 28, -14}
, {-28, -10, -27}
, {21, -14, 48}
, {12, -9, -48}
, {48, -61, -26}
, {-34, 4, 0}
, {-31, 47, 15}
, {31, 53, 24}
, {-44, -42, -22}
, {-17, 42, 11}
, {42, 48, -11}
, {-18, 27, -1}
, {-33, 46, 14}
, {45, -40, 16}
, {11, 45, -7}
, {42, 10, -36}
, {-43, 25, 4}
, {2, -3, -32}
, {16, 62, -15}
, {-19, 28, 53}
, {-39, 22, -1}
, {-36, -9, 23}
}
, {{20, -42, 23}
, {18, -35, 46}
, {-4, 20, -47}
, {47, 21, -10}
, {-7, 29, 7}
, {9, -29, -43}
, {-29, -23, -53}
, {25, -8, 5}
, {49, 26, 30}
, {-51, -23, 9}
, {46, 13, -34}
, {-20, 21, -13}
, {3, 27, 17}
, {-24, 4, -50}
, {27, 49, -12}
, {-17, 39, -31}
, {-16, 17, 33}
, {-11, 37, 13}
, {-22, 37, -15}
, {2, 21, 15}
, {30, 16, 30}
, {-27, -40, -46}
, {-42, -19, -20}
, {-10, -14, -40}
, {9, 33, -39}
, {-11, -42, -1}
, {6, -28, 32}
, {49, -1, -14}
, {44, -22, -38}
, {-14, -17, -29}
, {-42, -45, -45}
, {47, -29, -18}
, {-46, -54, -11}
, {-6, -15, 15}
, {44, -23, 23}
, {-51, -16, -39}
, {-13, 17, -40}
, {-22, 44, -46}
, {-30, 0, -45}
, {-50, 10, -46}
, {5, -18, -15}
, {46, -21, -35}
, {40, -49, -3}
, {44, -45, -21}
, {41, -48, 36}
, {29, 18, -43}
, {35, 30, 14}
, {-47, 37, 8}
, {43, -26, -7}
, {-50, -35, 51}
, {7, 10, 35}
, {-22, -31, 28}
, {-20, 4, 39}
, {-46, -33, -17}
, {43, -38, -45}
, {0, -36, 7}
, {-32, 7, 15}
, {31, -6, 49}
, {-30, -41, 38}
, {-23, 23, 15}
, {-26, 19, 47}
, {-3, 48, -36}
, {-51, 19, -44}
, {-25, -27, -7}
}
, {{-12, -8, 18}
, {-36, 25, 45}
, {38, -46, 3}
, {-19, 46, -24}
, {-36, 21, -9}
, {-50, 45, -16}
, {28, 47, 44}
, {-15, 9, -57}
, {-35, 37, -26}
, {-41, 34, 24}
, {-48, -6, -47}
, {-36, 34, 21}
, {-30, -10, -2}
, {29, -4, -36}
, {60, -3, 19}
, {19, 60, -22}
, {-4, 50, 56}
, {24, 45, 40}
, {48, -31, 14}
, {24, 36, -2}
, {-35, 35, 49}
, {-41, 42, -21}
, {-35, -36, 2}
, {3, 44, 5}
, {32, -45, -7}
, {-11, 16, -24}
, {35, -6, 24}
, {39, -42, -27}
, {41, -29, 17}
, {-1, -5, -30}
, {42, -26, 48}
, {0, -10, 53}
, {27, 0, 41}
, {38, 3, -41}
, {-47, -19, 19}
, {22, 51, 44}
, {-8, 46, -41}
, {42, 38, 37}
, {-1, -15, -3}
, {4, 36, -2}
, {39, 16, -8}
, {-15, 43, -26}
, {53, -37, 0}
, {53, -30, 23}
, {21, -21, -15}
, {-4, -33, 10}
, {-24, -44, -19}
, {-47, 10, -9}
, {-40, -49, -40}
, {-25, 60, -7}
, {-40, -35, -17}
, {-42, -23, 27}
, {-10, -12, 39}
, {41, -43, -24}
, {18, 1, 10}
, {23, 19, -4}
, {3, -14, -53}
, {32, 41, 20}
, {-35, 6, 36}
, {-27, 40, 40}
, {-31, -28, -16}
, {30, 44, -4}
, {42, 10, 11}
, {12, 16, -26}
}
, {{34, 5, 11}
, {-49, -39, -9}
, {-10, 2, -10}
, {-11, 6, -54}
, {-31, -22, -19}
, {-7, -42, 27}
, {44, 21, 48}
, {35, 21, 19}
, {19, -45, 38}
, {-9, 54, -20}
, {-30, 23, -44}
, {-8, 45, 14}
, {47, -21, 50}
, {-40, 0, 27}
, {0, 1, -39}
, {38, -22, -34}
, {43, -51, 36}
, {34, -2, 44}
, {27, 27, 47}
, {15, -36, -43}
, {-17, 56, 32}
, {-13, -7, -51}
, {52, -19, -47}
, {-24, -32, 17}
, {-33, 42, 51}
, {45, 51, 35}
, {32, -33, -3}
, {32, 13, -29}
, {3, 42, 8}
, {2, 33, -6}
, {18, 30, 3}
, {43, -4, -7}
, {-4, 29, 4}
, {-19, 30, 19}
, {31, 14, 0}
, {49, 56, -2}
, {16, 8, -10}
, {34, 47, -34}
, {7, -50, -29}
, {32, -12, -3}
, {-49, -46, 19}
, {41, 35, 1}
, {-1, 34, -22}
, {46, -4, -40}
, {-21, -43, 38}
, {-9, -3, -38}
, {-29, -16, 55}
, {-32, -8, -31}
, {-27, 8, -31}
, {-12, 41, 42}
, {46, -14, -10}
, {43, 19, -31}
, {-3, 54, 42}
, {-14, -30, 42}
, {-4, -16, -39}
, {-35, 53, 23}
, {26, 6, -27}
, {-47, 38, -20}
, {-3, 13, 41}
, {46, 10, 23}
, {1, -38, 34}
, {25, 34, 30}
, {-36, -43, -35}
, {49, 36, -16}
}
, {{2, 3, 9}
, {-40, -45, -9}
, {45, 51, 9}
, {-12, 42, -9}
, {-7, -48, 33}
, {5, 26, -54}
, {-6, -23, 56}
, {39, -25, -11}
, {-30, -21, 41}
, {36, -26, -19}
, {9, 15, -5}
, {30, 42, 0}
, {-41, 32, -23}
, {-25, -28, -39}
, {-36, -17, 13}
, {-39, 31, 44}
, {5, 27, 27}
, {-26, 33, -2}
, {25, -23, -37}
, {-49, 47, -7}
, {38, -35, -18}
, {-7, 10, 18}
, {20, 15, -5}
, {17, 15, 44}
, {14, 38, -7}
, {-19, -20, -13}
, {41, 68, 64}
, {-2, -46, 18}
, {7, 52, 12}
, {10, -44, -24}
, {18, -27, -19}
, {-44, 51, 52}
, {48, -14, -4}
, {28, 61, 23}
, {58, 60, 45}
, {-1, -5, 27}
, {-45, -52, -22}
, {34, 51, 38}
, {-13, -50, -50}
, {15, -49, 17}
, {7, -49, -39}
, {0, 25, -14}
, {54, 2, 0}
, {-23, -25, -10}
, {-44, -11, 49}
, {-6, 22, 21}
, {-26, 52, 38}
, {43, 42, 34}
, {-45, -10, -2}
, {-9, -4, 11}
, {23, -12, 32}
, {-4, -14, -43}
, {30, 63, -9}
, {28, -5, -42}
, {-33, 38, 43}
, {55, 0, 34}
, {-26, 25, 21}
, {-22, -16, -29}
, {45, -16, -46}
, {24, 19, 46}
, {-31, 19, -19}
, {7, -46, -55}
, {-33, -31, 24}
, {38, -9, 46}
}
, {{38, -56, -36}
, {47, 47, 10}
, {-11, -14, 25}
, {-14, 38, 43}
, {-12, 48, -38}
, {40, -46, -46}
, {-14, -20, -24}
, {-10, -28, -1}
, {52, -42, -47}
, {7, -1, 55}
, {31, -53, 22}
, {35, 50, -48}
, {58, -17, 7}
, {-31, 9, -28}
, {39, 0, 24}
, {-19, 19, 30}
, {46, -38, -27}
, {8, -26, -25}
, {-30, 42, 10}
, {-13, -12, -42}
, {59, 24, 14}
, {43, 3, 28}
, {-19, -20, 9}
, {41, -30, 33}
, {51, 36, 16}
, {-21, 0, 32}
, {31, -29, -20}
, {41, -12, 12}
, {-21, 11, 19}
, {32, -46, 1}
, {28, 16, 4}
, {1, 38, 24}
, {-7, 0, -47}
, {-28, 45, 43}
, {21, 62, 51}
, {-34, 0, 17}
, {-19, -13, -19}
, {-31, 4, -52}
, {-9, -15, -9}
, {-48, -13, -21}
, {-52, 0, -1}
, {-8, 49, -9}
, {9, 37, 57}
, {34, -2, 8}
, {-32, 23, 0}
, {31, 2, 24}
, {-8, 15, 14}
, {-53, 45, -50}
, {5, -18, -31}
, {17, -50, -28}
, {13, 22, 4}
, {-12, 0, 17}
, {-27, -20, -21}
, {5, 31, 37}
, {15, -32, -49}
, {52, -20, 0}
, {-40, -33, -46}
, {-32, -57, -30}
, {29, -26, -10}
, {-6, -54, -38}
, {-34, 21, 11}
, {-34, 23, 11}
, {-61, -13, -23}
, {47, -32, 5}
}
, {{-9, 38, 52}
, {42, -18, -11}
, {25, 14, -31}
, {-41, 6, 5}
, {45, -35, 1}
, {36, -14, 20}
, {-47, -24, 18}
, {37, 1, -25}
, {-4, -4, 15}
, {-16, -28, 49}
, {55, 4, 53}
, {-28, 0, -10}
, {-33, -33, -20}
, {29, 23, -24}
, {43, 28, 30}
, {2, 23, 55}
, {49, -11, -31}
, {-33, 17, 13}
, {19, -46, 10}
, {-34, 8, 7}
, {-5, 27, -64}
, {40, 10, 56}
, {31, -34, 51}
, {-51, 7, 27}
, {-14, 48, 40}
, {-22, 20, -19}
, {-1, -45, -69}
, {-29, -33, -32}
, {-42, -29, 8}
, {-29, -20, -19}
, {-36, 46, 13}
, {-50, -45, -48}
, {5, -1, -33}
, {7, -23, -24}
, {-49, -52, -6}
, {-19, 16, 51}
, {39, -18, 47}
, {11, -28, 45}
, {-10, -11, 2}
, {-12, 36, 29}
, {-9, -37, 36}
, {-57, -43, 43}
, {-5, 0, 6}
, {38, -29, -43}
, {9, 21, -41}
, {-25, -28, 18}
, {-27, 46, 0}
, {6, 1, 58}
, {15, 0, 26}
, {55, -2, 54}
, {-34, -33, -23}
, {-26, 29, 5}
, {2, 38, -21}
, {-14, 29, 48}
, {-36, -20, 36}
, {-44, -17, 2}
, {-20, 20, 47}
, {59, -42, 27}
, {9, 6, 39}
, {47, 33, -34}
, {-29, -40, 40}
, {32, 23, -24}
, {11, 31, 45}
, {34, -36, -8}
}
, {{-7, -20, -30}
, {-22, -11, 1}
, {-52, -7, -19}
, {-15, 18, 15}
, {33, 31, -21}
, {-35, 32, -42}
, {32, -26, -52}
, {-32, 7, 4}
, {40, -39, -41}
, {39, 37, -49}
, {-42, -27, -4}
, {-5, -43, 41}
, {47, -29, -42}
, {-46, -14, -1}
, {-32, 24, 48}
, {40, -37, -6}
, {32, -12, 21}
, {-48, -42, -41}
, {-21, 22, 21}
, {-9, -49, -44}
, {-37, 12, 4}
, {11, 28, 18}
, {22, -23, -9}
, {-40, -48, 30}
, {-41, 49, 36}
, {-47, 44, 36}
, {0, -24, -35}
, {-36, -50, 13}
, {-22, 4, 14}
, {23, 16, -29}
, {49, -27, 48}
, {-32, -38, -13}
, {23, -51, 13}
, {38, 21, -30}
, {10, 48, 10}
, {-21, -47, -23}
, {12, -26, 44}
, {-8, 40, -37}
, {-22, -17, 26}
, {19, -36, -12}
, {44, -25, 15}
, {-14, -31, -16}
, {33, -13, -42}
, {-25, -22, 23}
, {1, -36, -22}
, {-16, -43, 1}
, {38, 3, -21}
, {-49, -32, -50}
, {-30, -2, -53}
, {32, -10, -33}
, {8, 45, -40}
, {-31, 5, -12}
, {-49, 40, -26}
, {2, -50, -12}
, {47, -22, 43}
, {-27, 1, 35}
, {2, -38, 38}
, {-6, -19, 44}
, {-29, 24, -31}
, {16, 36, 1}
, {-33, -45, -48}
, {0, -13, -9}
, {4, 23, 29}
, {-26, -52, -16}
}
, {{7, 62, 1}
, {-49, 46, 4}
, {53, -6, 40}
, {-5, 20, -5}
, {-11, -34, 3}
, {-1, 8, 21}
, {-2, -34, 19}
, {54, -5, 53}
, {-57, 7, -12}
, {48, 24, -14}
, {17, 42, -36}
, {37, 17, -23}
, {-10, -26, 34}
, {1, 14, 49}
, {21, 14, 12}
, {-21, -31, -54}
, {-22, -32, -30}
, {-36, -15, 8}
, {18, -15, 26}
, {42, 8, -1}
, {-10, -29, -41}
, {-35, 14, 56}
, {-1, 2, 28}
, {59, -48, -18}
, {5, -27, -45}
, {30, 14, -18}
, {-38, 60, 19}
, {6, 18, 26}
, {0, -52, -12}
, {-25, -39, -41}
, {15, 48, 2}
, {-12, -25, 45}
, {-31, -29, -24}
, {33, 55, -41}
, {21, -2, -33}
, {-27, -24, 34}
, {47, 23, 22}
, {28, -28, 4}
, {-36, 1, 11}
, {9, 43, 7}
, {24, -33, 61}
, {-28, -13, -20}
, {-3, -23, -35}
, {58, 46, -36}
, {18, 15, 0}
, {1, -5, -21}
, {-49, -23, -28}
, {-44, -60, 7}
, {34, -7, 29}
, {24, -40, 9}
, {-29, 3, -34}
, {46, -45, 4}
, {-3, 23, 31}
, {10, 40, -23}
, {43, -7, 62}
, {-47, -23, -27}
, {-31, 1, -19}
, {8, 44, -30}
, {33, 56, 0}
, {-30, 16, -16}
, {20, -28, 15}
, {-11, -18, 24}
, {-4, 38, -5}
, {35, -4, -51}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t max_pooling1d_4_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_4(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t average_pooling1d_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_DIM [1][128]
#define OUTPUT_DIM 128

//typedef number_t *flatten_output_type;
typedef number_t flatten_output_type[OUTPUT_DIM];

#define flatten //noop (IN, OUT)  OUT = (number_t*)IN

#undef INPUT_DIM
#undef OUTPUT_DIM

/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t dense_output_type[FC_UNITS];

static inline void dense(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 128
#define FC_UNITS 40


const int16_t dense_bias[FC_UNITS] = {7, -1, -3, -7, 7, 9, 7, -1, 1, 8, 2, 6, 7, -8, -6, 6, -8, 6, 7, -10, 8, 0, 9, -2, -4, -1, -8, -10, 9, 1, -7, -7, 4, 6, 8, 1, -4, -2, -6, -8}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-39, -21, 73, 37, 73, 40, 87, -63, -82, 93, -14, 31, -27, 62, 0, 74, 16, -74, -4, 9, -72, 104, -37, -6, -31, 30, -52, 18, 59, -22, 22, 35, -63, 96, -102, -77, 1, -34, 45, 71, 59, 62, -18, -36, 27, 16, 60, -25, -83, -97, -87, 65, -94, 6, -63, 2, -64, 10, 48, -35, -57, 66, -17, 30, -41, -44, 99, -46, 56, 92, -70, -10, 13, 71, 24, -69, -38, 84, -60, -12, 78, -72, 37, 66, -56, 81, -62, 62, 78, -25, 6, 53, -16, 88, -80, 28, -55, 45, -43, 41, 35, 25, -64, -40, -84, -50, 8, 89, -75, -83, 95, 13, 85, -32, 90, 0, -24, -70, -62, -45, -62, 49, -16, -58, -72, 0, -47, -95}
, {22, -37, -72, 55, 39, 103, -28, 51, 4, 30, 15, -87, -86, 35, -43, 91, -29, -89, 42, -5, -50, 25, 62, 54, -51, 73, 14, -41, -60, 51, 37, 19, 47, 10, 26, 42, 6, 34, 81, 77, 80, 7, -83, -39, -22, -21, 16, -11, 24, 38, -40, 59, -82, 28, -3, 10, 25, 55, 28, 83, 72, -34, -49, 83, 70, 22, -63, -68, -29, 57, -75, 99, 64, 65, -29, -81, -15, 73, -59, 7, -59, 8, -22, -63, 44, 55, 57, -13, -48, -44, 42, 4, -1, -61, 49, 65, -49, 68, -80, 81, 30, -96, 10, 100, -35, -13, 22, 22, 93, -95, -74, -18, -43, -64, 30, -76, -30, 39, 14, -55, 9, -16, -71, 80, 24, -11, 25, -21}
, {75, -31, -13, -6, 38, -13, 57, -14, -68, 90, 43, 52, -69, -28, 99, -49, 78, -76, 13, -72, -41, 6, 67, -49, -23, -21, 97, -45, 56, -9, 13, 69, -4, -19, -32, -38, -100, -31, -59, 6, 11, -68, 57, -80, -98, -48, 25, -92, -52, 32, -49, -101, -8, 46, 38, 42, 47, 12, 39, -73, -33, -50, -89, 76, -85, -67, -32, 67, -86, -31, -25, -32, -98, 95, -49, -51, -96, 11, -70, 71, 68, 58, 81, 35, -15, 96, -38, -59, 105, 9, -79, 68, -73, 26, -33, -53, -59, -29, -69, 31, 97, -29, 51, 0, 28, -12, -84, -42, 16, 54, -75, -39, 79, -49, 75, 43, 48, 42, -71, 6, 75, 2, 58, 9, -96, -36, -24, 15}
, {1, 92, 27, 11, 46, -99, -47, 56, 19, 95, -5, 83, -25, 11, -90, 87, -91, 100, -55, -48, -7, 59, -44, -12, 39, -80, 26, 52, 63, 54, -20, 10, -22, -78, 1, -56, 85, -58, 86, -95, -17, 49, -23, -96, 47, -1, 35, 17, -67, 62, 78, 8, -57, 25, -54, 8, 53, 22, -78, 84, -25, 29, 81, 30, 0, -96, 12, -36, -18, 58, 10, 20, 88, -72, -65, 25, 20, -34, -48, 39, 2, -46, 83, -23, 12, -105, 91, 39, 30, 61, 83, -62, 61, 27, -56, -68, 24, -12, -59, -83, 81, -44, 74, -27, 63, 41, 53, -95, -1, 66, -17, 46, -49, -59, -62, -25, -87, -27, -71, 35, 65, -98, -87, -23, -47, 13, 92, 92}
, {-47, -24, 5, -16, 83, 41, 42, -83, -38, 58, -69, 23, -89, 70, -35, -90, 18, 24, 69, -79, 30, -2, -82, -43, 58, 22, -68, -32, -75, -54, -68, -81, -63, 58, -3, -25, 3, -45, 22, -5, -8, 59, 28, -66, -35, -39, -60, 87, -33, -74, 36, -64, -65, -91, -7, 70, 14, 73, -87, -90, 74, 93, -4, -84, -12, -44, -41, 47, 22, -7, -2, 57, -48, 64, -70, -69, 86, 36, -78, -8, 16, -4, -56, -77, 33, 2, -12, 92, -17, 49, 86, -61, 62, 29, 5, 50, 50, -90, -81, 76, 79, 95, -19, 8, 8, -69, -7, -23, -96, 62, 86, 0, -48, -23, -32, -23, -46, 64, 65, 5, -27, -27, 13, 81, -26, 47, 6, -96}
, {28, 54, -84, 89, -42, -80, -77, 55, -19, 43, -41, -48, 50, 93, -17, 78, 49, -46, -68, 77, 7, 64, 83, -50, 76, -49, -53, 67, 113, 41, 36, -31, -8, -31, 39, 82, -83, -51, -95, 35, -46, -2, 44, 10, 5, -77, -17, -77, -68, -17, 67, -89, 0, 8, 94, -36, 73, -48, 18, -51, 8, -26, -17, 55, -31, 44, 84, 9, 4, 1, 98, -10, -38, 22, -57, 68, -7, -26, 66, 86, 10, 92, -76, 97, 33, -24, -18, 24, 9, -56, 10, -27, 82, -41, -51, -74, 38, 89, -88, -6, -83, 39, 70, 45, 56, -71, 19, -63, 23, 6, 83, -19, -61, -31, 23, -100, 8, 61, -73, 79, 25, -11, -93, 51, -103, 61, 80, -74}
, {5, -84, 47, 39, -108, 66, -93, -50, -43, 96, 12, 3, 8, -46, -37, -53, 88, -32, 4, -24, -33, -28, 21, 7, 60, -52, -1, -9, 74, 64, 27, 81, 75, -62, 23, -34, 28, 41, -81, 47, 12, 52, -56, 32, -27, 77, 40, -60, -62, 76, -78, -4, 46, -64, 64, -87, -21, -31, -39, 72, -62, -52, 101, -81, -19, -12, 59, -46, -66, 21, 45, -45, -86, -28, 33, -16, 95, -52, 13, 57, 44, -63, 50, 23, 12, -56, 65, -54, -17, -64, 61, -40, 70, 91, -14, 0, 13, -53, -64, 41, 57, -59, 82, 11, 2, -73, 25, -18, -91, 76, 32, 50, 4, 12, -81, -30, 31, -5, 96, 43, 21, 66, -74, -25, -20, 39, -81, -30}
, {89, -47, -27, -44, 26, 67, -91, -50, 73, 96, -13, -92, -79, -20, -83, -85, -55, 89, -7, 55, 69, -51, -11, 17, -8, -13, 44, 75, 18, -81, -73, 74, 50, -57, -82, -48, -93, 62, 16, -91, 63, -38, 70, -80, 89, 57, 1, 39, -49, -50, -12, 0, 75, -77, 51, -92, -55, -59, 22, 74, -29, -80, -35, -87, -52, 52, -14, 70, -68, 50, 21, 24, -62, 78, 25, 94, -46, 59, -34, -27, 34, 43, -44, -50, -18, -5, 7, 72, 14, -41, -64, -64, -31, 33, -6, -5, -76, -73, 58, -5, 64, 95, -51, -73, 26, -21, 94, -56, -66, 97, 27, -68, -94, 40, 5, 25, -83, -88, -54, -12, -15, -70, 11, -76, -50, -31, -59, -19}
, {66, 7, -83, -51, 0, -20, -25, 56, -57, 76, 47, -61, -11, 113, -68, -99, -99, -52, 98, 112, 20, -20, 24, 7, 13, 79, 0, 35, -20, -57, 42, -94, -33, -94, 83, -13, -87, 58, 57, 86, 36, -79, -47, -15, -39, 81, 94, 15, 44, -49, -48, -88, -81, -58, -38, 78, 49, -5, -89, -57, -75, 75, -10, -35, -67, 28, -17, 67, -32, -3, 75, -23, -45, 11, -60, 80, 60, 36, -52, -29, -34, -101, 45, 49, 13, -75, -61, 25, 50, -19, -72, 71, 36, 17, -51, 3, -11, 2, -86, -42, -68, -41, -10, 10, 10, 63, 15, 35, 34, 4, -52, -20, 72, -68, -2, 61, 4, 67, 77, -23, -6, -73, -40, -41, 10, -82, 69, 82}
, {-48, 28, -85, -29, -56, 77, 0, 3, 73, 17, 20, 5, -83, 69, -35, -58, 33, -9, 91, 19, 71, -16, 72, -31, 89, 79, -70, 18, 64, 27, -11, -88, 76, -28, 68, 28, 46, -14, 66, -87, 0, 89, 40, 34, -33, 84, -55, -38, -81, -54, -49, 73, 67, -41, 93, 20, 45, -84, 22, 86, 35, -35, -71, -67, -49, 47, -78, 1, -54, 91, 35, -57, 42, 81, -79, 31, -19, -48, 88, -18, 78, -41, -36, 72, -14, -35, 19, -15, 53, 27, 82, 86, 86, -32, 39, -69, 33, 49, 71, 95, -45, -25, 37, -21, 61, -12, -27, 97, 27, 17, 81, 24, 77, -50, 81, -57, -85, -90, 58, 31, -39, 39, -30, -47, -4, 90, -60, -106}
, {-61, -28, 66, 54, 13, -13, 60, -34, -50, -30, 23, -54, 85, 81, -45, 73, 4, -34, -84, 57, -96, 13, -44, 59, -41, -50, 70, -96, 7, -11, -14, -79, 95, 73, -47, -42, 76, 45, -48, 85, -8, -41, -43, -56, 40, 50, -49, -93, 85, -84, -23, -19, -78, 67, 51, -4, -68, -43, -10, 53, 100, 23, 7, 82, -6, -65, 66, -53, 103, 81, 81, 10, 63, 76, -69, -99, -40, 16, 0, -17, -38, 89, 34, -41, 107, 72, 86, 52, -51, -23, -71, -34, -44, -70, -87, -90, -22, 7, -92, 24, -82, 89, 49, -61, -28, 73, -90, 7, 77, -80, -70, 28, -80, 72, -88, -10, -21, 61, -77, 88, 0, 51, 100, 50, -5, -10, 23, 47}
, {-87, 28, -83, 9, -10, -30, 48, 84, 40, -58, 5, 12, -6, -21, 19, 23, 5, 2, -94, 48, -30, -26, 44, 59, 92, -2, -95, -43, 7, 13, -66, 66, 64, -51, -2, -67, 28, -75, -91, 93, -56, -48, 35, 64, -55, -18, 37, 67, -23, 10, 55, -54, -36, 9, -94, 71, 38, -15, -21, 0, -69, -16, 22, -82, -94, 96, -22, 58, 7, -30, -63, 73, -2, 33, -67, -86, -1, -94, -79, -23, -5, -43, -19, 39, -34, 51, -28, -14, -41, 100, -94, 64, 26, 24, 94, -1, 47, 103, -77, -14, 62, 104, 79, 17, -40, -37, -96, 38, -58, -85, 54, 25, -60, 66, 68, 44, 40, 96, 13, 54, 10, 43, -39, 59, 45, 57, 81, -72}
, {79, 48, -41, -41, -66, -63, -108, -98, 92, 17, -52, -91, -73, 115, -68, -38, 29, -68, -3, 107, 43, -77, -93, 4, -30, -55, 58, 44, 26, -59, -63, 72, 24, -65, -65, -7, 7, 17, 38, 20, -67, 4, -49, 26, 85, 90, 50, -57, 23, -31, -41, -57, 68, 10, 28, -13, 55, 40, 9, -10, -56, -11, -7, -9, -18, 46, -81, -27, 63, 41, -57, 19, -27, 17, 75, 64, -6, -65, 93, 11, -10, 84, -21, 84, 59, 8, 83, -16, -84, 91, 35, -49, -37, 29, 14, 74, -3, 25, -12, -12, -98, -40, -14, -44, 12, -45, -56, 37, -105, -52, 34, 14, -88, -5, -70, 65, -2, -40, 12, 66, -55, 66, -60, -80, 12, 83, -10, 66}
, {-48, -64, -56, -97, 87, -42, 5, -62, 7, 11, 38, -22, -20, 66, 38, -6, -97, 45, 64, -52, -83, -33, -90, -60, 51, 96, 38, -35, 24, -17, -62, 92, -45, -23, -50, 58, -57, 2, 28, 48, -97, -87, -59, -64, -43, 56, -46, -87, -18, 24, 85, -67, 23, -51, 71, 83, 6, 90, 22, 18, 46, -24, -6, -41, 61, -24, -21, -77, -98, 66, 60, -80, 102, -85, -10, -41, 37, 82, -1, 60, 50, 17, 10, 11, -60, -35, 81, 0, -8, 18, 32, -52, 31, 52, -8, 46, -90, -47, 51, 58, -66, -65, 83, 74, -51, -80, 83, -7, 25, -1, 48, -49, 30, -93, -52, 22, -79, -103, 61, -13, 81, 67, 31, -92, -85, 72, 46, 17}
, {11, 71, 14, -78, 84, 15, 92, -65, 20, -19, -77, 7, -10, 5, -33, 49, 11, -33, 29, -75, 27, 56, -44, 0, -7, -33, -77, -107, -60, -47, -61, -20, 0, -51, 57, 17, 83, -73, -3, -26, 35, -84, -79, 49, -72, -44, 5, -42, -83, 48, -75, -80, 50, 59, 61, 93, -46, -31, 19, -53, -1, -29, 98, -69, -83, -17, 28, -74, -86, -77, -64, -4, 63, -68, -33, -54, 93, 26, 58, -59, 62, -89, 18, 32, -37, -96, -74, 85, 42, -30, -21, -91, 83, 42, 72, -78, 7, -3, -23, -6, -20, -83, -10, -79, 76, -20, -33, 69, 81, 60, 19, 51, 58, 41, 36, 15, -2, -57, 33, 17, 53, 23, -61, 1, -28, -57, -60, -14}
, {-89, -14, -33, -68, -3, -72, -83, -77, 55, -25, 53, 73, -26, -78, 91, -76, 49, 18, 65, 12, -53, -28, 23, -43, -88, -52, 72, 35, -71, -77, 42, 86, 77, -19, -33, 73, 55, 18, 16, 10, -43, -38, 74, -94, 49, -87, 40, -41, -65, 35, -69, 90, -96, -59, 1, 62, 81, -87, 29, 69, -12, -20, -59, -10, 24, 1, 74, -2, -47, 92, -87, -18, 34, -57, -60, -19, 46, 82, 4, -87, -41, 42, 95, -64, -83, 101, -11, 64, -44, 4, 66, 19, 27, -53, 20, -49, 20, 42, -44, 96, -64, 93, 80, 87, -27, 61, 48, 48, -34, -7, -59, 41, 31, 36, -52, 80, -79, -40, -58, 79, 73, -24, 65, -2, 86, -74, 22, 22}
, {53, -42, 77, 11, -2, -62, 41, 26, -91, -37, -61, 91, -56, -45, -29, 43, -111, 57, -12, -50, 10, -54, -52, -90, 13, -68, 4, -40, -65, 29, 60, 51, -60, 53, -35, -92, -94, 55, -23, 69, 61, -58, 33, 53, 109, -71, 29, -52, 75, -75, 65, -58, 45, -41, -99, 73, 30, -29, -74, -29, -94, -61, 5, -93, 21, 67, -39, 96, -9, 41, 33, -4, 51, -27, -74, -82, -49, 30, -41, -33, 38, -46, 90, 39, 78, -54, -37, 30, -96, -81, -11, -35, 13, 0, 5, 35, -3, 12, 39, -30, 90, -94, 92, 68, 79, -49, 85, 76, -46, -84, -15, 0, -68, 30, -26, -96, 5, -14, -93, -26, -49, 64, -85, 36, 70, -67, -19, 27}
, {29, -17, 99, 102, 19, 12, -29, -3, -34, -6, 54, -78, -76, -33, 14, -3, 50, 12, -35, -11, 41, 0, 94, 34, 31, -21, 16, 98, 26, 96, 8, 91, 61, -33, 5, 0, 39, -81, 35, -61, -28, 5, -81, 65, -73, 101, 3, -31, 23, 57, 7, 39, -12, -75, -75, 69, -43, 86, -74, -71, 58, -63, -96, -26, -43, -86, 9, 22, 48, -86, -33, 93, 89, -8, -18, -59, -20, 30, 16, -32, -80, 24, 30, 68, -73, 64, 68, 43, 16, -65, -40, -45, 2, 34, -24, 28, 52, -49, 59, 42, -92, -10, -98, 7, -13, -11, -62, 1, -90, -5, -85, 70, -88, 38, 90, -67, -88, -3, -75, -53, -15, 33, 95, 1, -76, -19, -58, 85}
, {15, -78, 4, 15, -28, -50, 41, -61, 50, 85, -73, -85, -60, 64, 54, 90, 27, -19, -37, -59, -33, 113, -35, 34, -68, -3, -12, -66, -67, 60, -66, 8, 6, 51, -51, 63, 36, 53, -24, -27, 34, 78, 0, -35, 10, 13, -26, 19, -19, -6, 59, 25, 8, 57, 47, -79, -75, -14, -68, -18, 14, -97, 101, -77, -29, -19, -97, 70, 7, -5, -45, -43, -18, -67, -79, 92, 62, -35, 77, 17, -21, 30, 96, 36, -55, 13, -85, 105, -4, -38, 80, 46, 78, 29, -30, -98, 37, 101, 52, -89, 80, 76, -99, -76, -66, -47, -49, 58, 0, -85, -24, -72, -26, 30, 59, 72, -76, 66, -19, 10, -63, 107, 38, 39, 95, -50, -95, 25}
, {-15, 31, 41, 54, 31, 95, 66, -10, -48, -40, -18, -92, -46, -48, 81, 52, -37, -55, 43, -39, 28, -67, 84, -31, 72, -35, -49, 26, -73, 42, 5, -16, -48, 64, 2, -59, -15, -79, 45, -63, -47, 94, -82, 17, 28, -41, -78, -71, -14, 16, 80, 12, -17, 72, -90, -4, 70, 91, -24, -56, -44, 49, -111, 6, -60, -53, 76, 62, 64, 95, 35, -42, 67, -64, -45, -38, 12, 13, -5, -85, 9, -97, 36, -107, -28, -46, -36, 52, 49, -63, -29, 26, 69, -24, -45, -51, 65, 7, 0, 86, 11, -66, -7, -40, -51, 23, 24, -63, 93, 0, 71, 0, 94, 59, -57, -45, -18, 39, -12, 48, -10, -30, 33, -28, 62, 19, -41, 11}
, {90, -4, -43, -5, 74, -31, -26, -44, 28, -49, 61, 13, 86, -49, 6, 57, -77, 72, -11, -4, -3, 76, 41, -85, -67, -36, -15, 5, -49, 60, -34, 38, -69, -96, -12, -31, 54, -30, -77, -89, -10, -41, 19, 108, 69, 4, 18, -55, 102, 38, -32, 34, 95, -69, -71, 55, -89, 47, 89, 91, -40, -44, -8, -22, -60, -67, 11, -95, -81, -7, 49, 48, -87, -3, 81, 48, -5, 75, 98, -93, -9, -66, 62, 87, 73, -16, -12, 89, -36, 0, 2, -61, -63, -56, -50, -70, 80, -18, -17, -63, -64, 73, 60, -2, 22, 63, -18, 37, 23, 21, 63, 77, -62, 18, 53, 34, -41, -33, 91, -47, -45, 4, -28, -70, -61, 49, 55, 106}
, {92, -7, 19, 45, -51, -19, 76, -49, 10, 75, 27, -63, -54, 19, 60, 21, 40, 91, 75, -6, -72, 91, -69, 24, 77, 77, -9, 94, 85, 98, 34, 37, -9, 26, -80, 102, -21, 27, -73, -77, 8, -30, -44, 1, 88, -73, -12, 14, 17, -35, -5, -87, 21, -24, -82, 20, -30, 65, -99, 79, 26, 93, 41, 40, -36, -79, 24, -23, -99, 25, -62, -53, -23, 92, -1, 46, 78, 7, 97, 85, 78, -96, 29, 92, -94, 26, -64, 43, 87, -33, 5, 78, 87, 55, -19, -37, -53, 44, 89, -25, 81, -24, 22, 56, -53, 89, -20, -50, 11, -8, 59, 45, 53, 92, 73, -14, 51, -95, -84, -55, 38, 90, -54, -84, -66, 102, 29, 59}
, {38, -17, -91, -5, 8, -15, -89, 76, -43, -65, 55, -72, 68, -33, -42, -84, -44, -55, -66, 16, 98, 78, 87, -97, 36, 17, -2, 91, 42, -37, -49, 87, 66, 29, -23, -16, -77, -6, 18, 63, -11, -8, 13, 87, 7, 40, -11, 8, -67, 25, 8, -81, -56, 20, -57, 61, -86, -89, 88, -55, 73, -22, 60, 86, 75, 91, 81, 31, -73, 58, -13, -107, -65, -41, 65, 25, -66, -101, 31, 25, -24, 72, 53, 96, -35, 70, -72, 100, -95, -54, -72, -81, 20, -91, 21, 42, 54, -67, 4, 20, -39, 94, 8, 43, 57, 76, 39, 76, -104, 93, 84, 57, -37, 88, -24, 46, -52, 95, 64, 8, 25, 86, -30, 24, 11, 60, 85, 86}
, {21, -2, -102, -72, -23, -62, 73, -91, 25, -60, -24, -86, 71, 10, -38, -76, 89, -18, -49, 9, -71, 93, -84, -58, -12, -33, -60, 64, -11, -2, 17, 14, 6, 77, -70, -8, 53, 27, -6, -50, 13, -63, 69, 23, 34, -14, 81, 80, -52, -53, 31, 16, -18, 41, -11, -81, 4, -94, -8, 61, -35, -80, -94, -32, -38, 4, 26, 27, 59, -83, 58, 17, -75, -36, 88, 71, -25, 68, 63, 60, -26, -23, 69, 0, -34, 78, -89, 12, 4, 27, -81, -30, 25, -16, 65, 92, -25, 70, 51, -34, 14, -78, 19, 38, -34, -37, -69, 84, -64, -33, -86, 62, 24, 4, -24, -30, 22, -2, 1, -51, 48, -87, -73, -67, -59, -10, 57, -20}
, {23, 86, -77, -106, -107, -73, 37, -69, 0, -71, 56, -38, 81, -9, -60, -92, -75, -52, -59, 31, 33, -86, -95, 20, -46, -21, 21, 30, -14, 67, -69, 2, -6, -28, -93, 84, -56, -75, -99, -45, 24, -94, 65, 84, 8, 2, 71, 1, 27, -65, 81, 0, 67, 82, -32, -61, -49, -21, -65, -51, -37, -23, -80, -17, 72, 31, -70, -46, 42, 13, 76, -80, 55, -100, -53, -67, -91, -86, -78, 76, -34, 47, 80, -6, 60, 47, -98, -93, 31, 24, -49, 68, 29, 89, -22, 95, -79, -49, -2, -82, 63, -51, -53, -39, 33, 7, -89, -92, 12, -87, -68, 10, 78, 48, -69, 63, -46, -52, 70, 48, 60, 71, 44, 88, -19, 77, -40, 87}
, {35, -84, 37, 48, 86, 45, -48, -63, -63, 56, 56, 70, 41, 75, 68, 15, -63, -20, 18, 22, -9, -3, 4, -11, -13, 87, -66, -14, -72, -70, 34, -31, 71, 95, 45, 55, -59, -74, 84, 79, 6, -71, 76, -25, 49, 71, 43, -5, 3, 0, 56, 46, -64, -89, -30, -69, 62, -75, -78, 53, 73, -95, -50, -61, 57, -35, -12, -90, 25, -25, -95, 26, -8, 59, 88, -2, 76, 48, 90, 91, -82, 47, 54, -78, -16, -60, 89, 81, 13, 86, 72, -8, 27, 56, 2, -73, 84, 0, -60, 30, -22, 79, 10, 101, -87, 59, -40, -10, -59, -103, 17, 13, 1, 13, -47, 35, 72, 1, -83, 42, -14, 20, 69, 55, -61, 58, -49, -13}
, {85, -19, 66, -24, -80, -5, 43, -43, 66, -22, 91, -32, 40, -97, -69, -93, -104, -15, -57, 33, 94, 48, -28, 64, 67, 92, -92, -12, 19, -6, -32, 52, 50, 54, -81, 61, 52, -76, -80, 60, -47, 40, -52, -14, 18, 55, 58, -7, -32, 32, -59, 10, 56, -18, 19, -5, -21, 37, -41, 13, 50, -10, -71, 32, -75, -27, -91, 0, 58, 75, -14, 41, -28, -40, -92, -58, 76, 34, -26, -16, 4, 76, -17, -16, 41, 59, -1, 4, -74, -81, -89, 69, 39, -11, 4, 99, 10, -69, 79, 51, 83, -62, 85, -23, -17, 26, 39, 80, 19, 66, -19, -27, -91, -52, 0, -62, 61, -21, -46, -45, 45, 47, 50, -87, 40, -43, -68, -44}
, {-99, -28, 89, -52, 83, -42, 83, -97, -5, -41, -85, -43, -102, -51, 101, 6, 36, 66, 90, -67, -58, -55, -26, -38, -49, 38, -58, 47, -59, -95, 40, 11, -66, -80, -90, -53, 6, 48, 24, -22, 36, -8, 84, -14, -17, -109, 38, 86, 30, -32, 91, -97, 31, -74, -87, -5, -67, 0, -40, -10, 24, -42, 55, -57, 0, -96, 33, 15, -40, 36, -78, 88, 25, 66, -83, -28, 12, 24, -55, 66, 15, -92, 23, -45, 6, 80, -98, 89, 91, 24, -27, -84, -78, 82, -29, -36, -28, -56, 33, -81, -6, 16, -76, 69, -71, -47, 53, 62, 92, -80, 88, -105, -10, 78, -54, 81, 71, -32, -108, -83, -72, 0, 5, 19, 85, -82, 26, -70}
, {23, -36, -26, 73, 37, -30, 87, 76, 37, -73, 45, 3, -34, 8, 69, 12, -58, 71, 86, -49, -58, -53, -68, -69, -51, -16, 29, 95, 13, -67, -56, 59, -8, 27, 28, 11, -88, -59, 46, 22, -41, 20, -37, 32, -16, -45, -40, -6, -28, -23, 31, 44, 69, 50, -72, 39, -67, -21, -14, 38, 43, -39, 36, -86, -39, -4, -23, 43, -53, 65, 1, 40, 9, 8, -51, 64, 0, -96, -42, -88, 93, -64, -42, -68, -37, -53, 36, 30, 32, 49, 62, -76, -32, 60, -23, 63, -58, -4, 15, 46, -95, 32, -50, -87, 98, -45, 7, 50, -34, 89, -27, -24, -92, 92, -12, 1, -73, 85, 69, 1, -51, 56, 73, 5, 58, -44, 28, 46}
, {-44, 18, -37, 17, -41, -31, 6, 75, 19, 7, -48, 44, 22, 95, -57, -79, 73, 11, 13, -26, 88, 0, 0, 83, -66, 92, -28, -34, -21, 56, 29, 75, -30, 29, -94, 82, -26, 87, -14, -31, -57, -70, 46, -65, 99, -14, -44, 83, -88, -7, -21, 16, -51, -10, 33, 5, -54, -72, 41, 8, 75, 45, 98, -37, -19, 55, -57, -41, -100, -60, -46, -55, -61, 36, -97, 80, -85, 4, -40, 89, -74, 37, -78, 0, -13, -81, 35, -58, 64, -91, 22, -20, -86, -84, 90, -85, -27, -2, -58, -27, -63, 20, 8, 0, -50, 4, -68, -61, -2, -8, -2, 2, 44, -54, 46, 18, 16, 57, 88, -2, -3, 28, -50, -36, -95, -9, -24, -87}
, {87, -21, 72, -77, -2, 24, -54, -30, 63, 93, -102, 15, -13, 12, 78, 94, 32, -60, -93, 102, -46, -7, -12, 7, -58, -37, -17, -106, -94, 75, 7, 24, 48, 37, 80, -67, -23, 53, 69, 82, 83, -19, -6, 91, 52, 5, -90, 67, -51, -80, 73, 43, 68, 64, 0, -39, 0, -49, 47, 46, -30, 0, -51, -100, 79, 35, 71, 0, 40, 39, -7, 91, -36, 62, -47, 62, 14, -30, 68, 79, -37, -1, -19, 39, 21, -10, 91, -10, -68, 50, 95, 8, 50, -19, -95, 19, 93, -13, 57, -60, 63, -75, 52, -91, -6, -34, -51, -44, 62, -45, 4, -99, -49, -70, -25, 24, -67, -53, -47, -44, -93, -49, -57, 5, 92, 77, -51, 37}
, {87, 97, -79, 52, 1, -101, 101, -24, -21, 73, -32, 44, 107, -88, 32, -53, -84, 88, -58, -26, -19, -29, 90, -11, 41, -54, 13, -80, -98, 25, 88, -92, -60, -92, 55, -65, 73, 22, -26, -49, 4, 76, -34, 84, 88, -87, 86, -13, 50, -25, -46, 66, 99, 79, -75, -33, 9, -43, 20, -9, -87, -79, -1, -36, -57, 81, -101, 83, -55, 76, 51, -65, -71, -84, -48, -107, 24, 9, 80, 80, 39, 45, 60, -36, 6, -89, -70, -98, -56, -4, 95, -60, -34, 20, -55, 61, 21, 79, 7, -15, -92, 84, 53, 87, 30, 87, 69, 79, 89, 72, -75, -2, -93, -39, -88, -2, -19, -2, 49, -59, 0, -44, 12, -43, 24, -88, 29, -14}
, {54, -22, -35, 63, 49, 59, 86, -50, 58, -6, 45, -9, -75, -29, -95, -40, -71, 87, 51, -74, 12, -83, 43, 82, -33, -46, -74, 50, 26, 13, 73, -49, 61, 2, 68, -48, 27, 81, 27, 48, -30, 72, 57, -73, 19, 73, -96, -37, 74, 66, -59, 28, 8, 0, 37, -19, 54, -1, -83, -15, -35, -30, 74, -48, -48, -29, -61, -46, 101, 84, 0, -2, 28, -69, 82, -35, -23, 59, 73, 34, -46, 23, -84, 97, -77, -23, -46, 20, -62, -50, -39, 70, -12, 17, -2, -16, -62, 44, -59, 85, 67, -5, -81, 43, -3, 24, 79, 84, 91, 85, 1, -72, -33, -68, -43, 34, -22, 58, 77, -59, 47, 58, -15, 42, -88, -87, 64, 17}
, {-36, 89, 3, 40, -9, -81, 73, 46, 28, 0, -27, -75, -55, -22, 52, 59, 53, 88, 18, 82, 37, -2, 12, 0, -45, 74, 16, -18, 52, -30, 43, 41, 79, -89, 18, -11, -29, 37, 18, 80, -19, 54, -28, 0, -90, 98, 56, -70, -26, 72, 39, 5, -55, 99, -37, -57, 34, 41, -8, 32, 90, 64, -62, 65, -34, 69, 33, 60, 0, 51, 43, -48, -8, 38, 55, -91, -90, -40, 82, 85, -37, -77, 91, -39, 5, 22, -89, -19, -60, 38, -51, -2, -53, 76, -71, -6, 10, -50, 89, 35, -103, 60, 14, 88, -73, 81, 1, 68, -8, 17, 11, 56, -29, -76, 83, -79, -79, 89, 38, 72, 47, -1, 88, 62, 25, -54, 46, 72}
, {-81, 45, -8, 19, -37, 44, -81, -49, -8, -18, 97, -67, -90, -21, 24, 63, -30, -29, 28, 86, -75, 86, 92, 50, 81, 9, -77, 18, -36, 36, 47, -35, -5, -72, 94, -34, -70, -14, 20, -81, 87, -59, 83, -30, -90, -9, -59, -36, -43, 27, -78, 76, 21, -92, 7, 13, -39, -52, 20, -8, -11, -66, 36, -8, 89, -66, -83, -6, -68, 52, -22, 3, -90, 73, -21, 57, 74, 18, 83, 68, -11, -79, 76, 83, 25, 14, -75, 12, -35, -28, 82, -13, 54, 7, -73, -57, -15, 72, 28, 68, 12, -18, -52, -26, -42, 39, -96, -27, 84, 44, -78, 0, 50, -16, -77, -50, 44, 48, 89, 94, -75, 77, 49, -50, 39, 70, 73, 79}
, {-69, 32, -91, -100, -9, -53, 5, 5, 7, 82, -27, 93, 53, -59, 85, -80, 45, 44, 52, -32, -57, 105, 40, -88, 68, -96, -69, 66, -62, 71, -47, 0, 6, 41, 65, -24, 51, -70, -30, 79, 63, -63, 35, -16, 7, -83, 78, 28, -37, -69, 3, -20, -59, 88, 59, 25, -30, -41, 88, -81, -93, -60, -49, 45, 47, -40, -58, 92, 75, 77, -45, 65, -33, 9, -18, 72, -51, 51, 0, 36, -66, -20, -39, -29, -88, 85, -44, -67, 95, -8, -68, 23, -57, 52, 36, 38, 11, -54, 16, -55, 53, 8, -67, -88, 67, -70, -63, 40, -67, 81, 32, -53, 47, -94, -75, -14, -73, 22, 66, 26, -38, -19, -44, 51, -57, 22, 82, 75}
, {-9, -30, -37, 62, -23, -6, -36, -60, 55, 2, -52, -84, -30, 84, -54, -31, 10, 98, 82, -11, 0, 16, -87, -2, 11, 66, -30, 72, -41, 75, 75, -49, -80, 88, 49, 40, -34, 79, 61, -7, 83, -22, 82, 7, 13, -47, 17, -53, -49, 79, 66, 58, 89, -55, -19, -55, -35, 25, -52, 41, -56, 91, 27, 54, 19, 87, 94, -6, 33, 56, -81, -66, -64, -92, -58, -63, 2, 51, 76, -45, -98, -7, -4, 51, 1, 20, 93, 86, 28, -43, 32, 18, 33, 68, -90, -42, -89, 33, 2, -10, -63, -87, 98, -8, 78, 39, 14, -22, -13, -10, -50, -92, -68, 58, 55, 41, 12, 7, -70, 64, -5, -36, -4, 61, -86, 9, 80, -32}
, {15, 0, 71, -96, 23, -4, -57, -68, -84, -74, -10, 7, 60, 76, 74, 82, -3, -18, 60, 68, -75, -42, 53, -18, 42, -6, -22, 35, -79, 1, -44, 58, -2, 91, 60, 73, -6, 29, -81, 18, -90, -38, -22, 35, -11, -41, 76, 62, -37, 55, 83, 33, -69, 86, -29, 26, 85, 13, 86, -76, 29, -62, 37, 33, 25, 34, 64, -85, -25, -52, -89, -50, 65, 32, -38, 72, 95, 48, 42, -58, -70, -88, -77, 36, -102, -23, -18, -27, -25, -74, -72, -42, 95, 40, -34, -87, 19, -5, -17, -32, -90, 16, 3, -19, 38, 69, 86, 23, 20, 33, 89, -79, -8, 87, -57, -74, 33, 73, 22, -37, 32, -80, 63, -16, -65, -15, 16, -84}
, {28, -56, 84, -22, -7, -63, 9, 37, 50, -76, 56, -56, 26, -66, 41, -24, -60, 96, 96, 29, -53, 84, -102, 43, -28, 38, -92, 40, -83, 30, 42, 87, -66, 70, 50, 40, -60, 85, -26, 63, 68, -12, 72, -22, -46, -108, 53, 46, -51, -16, 26, -77, 59, -8, 99, 25, 87, -98, 10, -19, 71, -83, 10, -56, 74, 61, -14, -18, -63, 49, 39, 50, 5, -48, 76, -49, 78, -44, 53, 29, -5, 66, -65, 39, -53, 11, 81, 60, 29, 100, 87, 53, 82, 13, 80, 79, 28, -100, -3, 27, -37, -18, 76, -14, -72, -25, -84, 27, -13, -44, -62, 44, -84, -45, 46, 72, 86, 41, 20, 23, -86, 40, 31, -2, 61, 99, -63, 5}
, {74, 79, 88, -105, -73, -20, 49, -76, 11, -16, 40, -74, 50, -16, -47, -70, -29, -13, -65, 37, -72, -91, -97, -8, -28, 56, -74, -73, 35, 38, -43, -70, 74, 81, -61, -95, -63, -69, -92, 16, 21, 43, -71, 33, 67, 9, 72, -84, 81, -95, -7, -24, 99, 65, -65, 17, -69, -3, 26, 75, -67, 71, -105, -81, -7, 39, 2, -40, -35, 40, -87, -44, 48, 33, 65, 90, -60, 91, -52, -37, -43, -8, 96, -36, -64, -47, 27, 66, 61, 39, 35, 44, -31, -69, 80, -13, 63, 93, 39, 24, -83, 18, -51, 5, -59, 6, 20, -30, 9, 39, -76, -25, 59, 5, 74, 5, 81, 69, 70, 66, 36, 8, 90, -100, -29, 21, 26, 15}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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

typedef number_t dense_1_output_type[FC_UNITS];

static inline void dense_1(
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
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 40
#define FC_UNITS 4


const int16_t dense_1_bias[FC_UNITS] = {10, -5, -5, 7}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{-181, -123, -197, -101, -139, 188, 113, -82, 204, -47, -22, 159, 183, -39, 61, -27, -78, -22, 86, 20, 150, -24, 176, 21, -18, -79, -105, -152, 139, 33, -7, 108, 92, 22, 102, 134, -40, 12, -92, 65}
, {-187, -173, 132, 177, -125, 36, 87, 146, 64, -69, -114, -157, 128, 148, 16, -119, -44, -106, -178, 146, -70, 156, -125, 150, 130, -148, -32, 181, 24, 37, 170, 51, -98, -182, -95, 193, 150, 177, 161, 105}
, {-141, 79, -67, -47, -124, -131, -95, -186, -147, -137, 70, 77, 0, 55, 53, 9, -52, 23, -21, 165, -22, -183, -104, -102, -88, 177, -43, 179, 32, -161, 28, 79, 72, -7, -21, 64, -177, -122, 76, 86}
, {72, -110, 166, -163, 59, 161, 126, 44, -116, 103, -5, 52, 124, -80, -73, 87, -120, 81, 118, 115, -41, 55, 82, 11, -80, -52, -184, 174, 80, -49, -89, -182, -19, 0, -7, 188, -185, -23, 175, 41}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
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
  //dense_1_output_type dense_1_output);
  number_t output[MODEL_OUTPUT_SAMPLES]);

#endif//__MODEL_H__
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"

 // InputLayer is excluded
#include "max_pooling1d.c" // InputLayer is excluded
#include "conv1d.c"
#include "weights/conv1d.c" // InputLayer is excluded
#include "max_pooling1d_1.c" // InputLayer is excluded
#include "conv1d_1.c"
#include "weights/conv1d_1.c" // InputLayer is excluded
#include "max_pooling1d_2.c" // InputLayer is excluded
#include "conv1d_2.c"
#include "weights/conv1d_2.c" // InputLayer is excluded
#include "max_pooling1d_3.c" // InputLayer is excluded
#include "conv1d_3.c"
#include "weights/conv1d_3.c" // InputLayer is excluded
#include "max_pooling1d_4.c" // InputLayer is excluded
#include "average_pooling1d.c" // InputLayer is excluded
#include "flatten.c" // InputLayer is excluded
#include "dense.c"
#include "weights/dense.c" // InputLayer is excluded
#include "dense_1.c"
#include "weights/dense_1.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_1_output_type dense_1_output) {

  // Output array allocation
  static union {
    max_pooling1d_output_type max_pooling1d_output;
    max_pooling1d_1_output_type max_pooling1d_1_output;
    max_pooling1d_2_output_type max_pooling1d_2_output;
    max_pooling1d_3_output_type max_pooling1d_3_output;
    max_pooling1d_4_output_type max_pooling1d_4_output;
    dense_output_type dense_output;
  } activations1;

  static union {
    conv1d_output_type conv1d_output;
    conv1d_1_output_type conv1d_1_output;
    conv1d_2_output_type conv1d_2_output;
    conv1d_3_output_type conv1d_3_output;
    average_pooling1d_output_type average_pooling1d_output;
    flatten_output_type flatten_output;
  } activations2;


  //static union {
//
//    static input_1_output_type input_1_output;
//
//    static max_pooling1d_output_type max_pooling1d_output;
//
//    static conv1d_output_type conv1d_output;
//
//    static max_pooling1d_1_output_type max_pooling1d_1_output;
//
//    static conv1d_1_output_type conv1d_1_output;
//
//    static max_pooling1d_2_output_type max_pooling1d_2_output;
//
//    static conv1d_2_output_type conv1d_2_output;
//
//    static max_pooling1d_3_output_type max_pooling1d_3_output;
//
//    static conv1d_3_output_type conv1d_3_output;
//
//    static max_pooling1d_4_output_type max_pooling1d_4_output;
//
//    static average_pooling1d_output_type average_pooling1d_output;
//
//    static flatten_output_type flatten_output;
//
//    static dense_output_type dense_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_output
  );
 // InputLayer is excluded 
  conv1d(
    
    activations1.max_pooling1d_output,
    conv1d_kernel,
    conv1d_bias,
    activations2.conv1d_output
  );
 // InputLayer is excluded 
  max_pooling1d_1(
    
    activations2.conv1d_output,
    activations1.max_pooling1d_1_output
  );
 // InputLayer is excluded 
  conv1d_1(
    
    activations1.max_pooling1d_1_output,
    conv1d_1_kernel,
    conv1d_1_bias,
    activations2.conv1d_1_output
  );
 // InputLayer is excluded 
  max_pooling1d_2(
    
    activations2.conv1d_1_output,
    activations1.max_pooling1d_2_output
  );
 // InputLayer is excluded 
  conv1d_2(
    
    activations1.max_pooling1d_2_output,
    conv1d_2_kernel,
    conv1d_2_bias,
    activations2.conv1d_2_output
  );
 // InputLayer is excluded 
  max_pooling1d_3(
    
    activations2.conv1d_2_output,
    activations1.max_pooling1d_3_output
  );
 // InputLayer is excluded 
  conv1d_3(
    
    activations1.max_pooling1d_3_output,
    conv1d_3_kernel,
    conv1d_3_bias,
    activations2.conv1d_3_output
  );
 // InputLayer is excluded 
  max_pooling1d_4(
    
    activations2.conv1d_3_output,
    activations1.max_pooling1d_4_output
  );
 // InputLayer is excluded 
  average_pooling1d(
    
    activations1.max_pooling1d_4_output,
    activations2.average_pooling1d_output
  );
 // InputLayer is excluded 
  flatten(
    
    activations2.average_pooling1d_output,
    activations2.flatten_output
  );
 // InputLayer is excluded 
  dense(
    
    activations2.flatten_output,
    dense_kernel,
    dense_bias,
    activations1.dense_output
  );
 // InputLayer is excluded 
  dense_1(
    
    activations1.dense_output,
    dense_1_kernel,
    dense_1_bias, // Last layer uses output passed as model parameter
    dense_1_output
  );

}
