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

typedef number_t max_pooling1d_145_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_145(
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
#define CONV_KERNEL_SIZE    20
#define CONV_STRIDE         8

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_116_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_116(
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
#define CONV_KERNEL_SIZE  20


const int16_t conv1d_116_bias[CONV_FILTERS] = {9, 10, -4, 18, 21, -2, -24, 14, 21, 6, -7, 0, 8, 6, -2, 0}
;

const int16_t conv1d_116_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{76, 50, 5, 69, -10, 50, -29, -29, 35, 27, -8, 66, -22, -19, -7, -59, -60, 15, 6, 33}
}
, {{-17, -13, -49, -55, 28, -18, -27, 18, 65, 4, -22, -6, -37, -18, 35, 75, -21, 51, 40, -62}
}
, {{70, 89, 15, -28, 38, 20, -3, 17, 28, -50, -23, 38, 46, -73, -64, -52, -23, 26, -8, -81}
}
, {{-15, 30, 36, -5, 72, -1, -42, 60, 25, 65, -52, -53, 8, 22, -12, -74, -36, -18, -31, -36}
}
, {{-68, -43, -28, 21, -46, 40, 39, 40, -62, -9, 68, 43, 62, 66, 4, -7, 86, 31, 66, -14}
}
, {{-67, -67, -76, 38, -45, -3, -43, -68, 18, 31, -26, -27, -12, -24, -61, -22, 66, -47, 60, 34}
}
, {{30, 38, -58, 20, -19, 0, 72, -18, -2, 7, -1, 18, 7, -18, 5, 1, -30, 71, 3, -45}
}
, {{9, -35, 22, 40, -4, -82, -49, -81, -34, -36, 76, 63, 39, 73, 64, 22, -23, -52, -77, -37}
}
, {{-53, 65, -29, 79, 67, -15, -38, 44, -10, -47, -60, -4, -16, -59, 7, -19, -82, -35, -45, -58}
}
, {{-30, 61, 49, 40, -55, -60, -5, -1, 3, 19, 38, 63, 36, 73, 58, 14, -68, -9, -43, 4}
}
, {{-23, 16, -12, 1, 55, 65, 65, 4, 22, -3, -5, -10, -39, 47, 66, 41, 62, -19, 28, 8}
}
, {{-54, -23, -34, 44, -52, -24, 69, -35, -43, 9, 46, 46, -45, -52, -32, 11, -32, 61, -38, 81}
}
, {{47, 15, -2, 14, 30, 66, -2, -28, 31, 44, -44, -3, 23, -49, -59, 1, 52, -17, 57, 39}
}
, {{-44, 49, 29, 30, 14, 84, 80, 49, 8, 34, -12, 77, 68, -39, 10, -17, -4, 30, -58, 17}
}
, {{48, -9, 30, 38, 72, 24, -50, 67, -47, 76, 64, 63, 8, -4, -17, 39, 38, -31, -3, 15}
}
, {{-52, 38, -13, -33, 70, 34, -14, 71, 68, 21, 48, 51, 4, 41, -38, 74, 46, 56, -5, -31}
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
#define INPUT_SAMPLES   998
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_146_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_146(
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
#define INPUT_SAMPLES       499
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_117_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_117(
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


const int16_t conv1d_117_bias[CONV_FILTERS] = {8, 37, 0, 7, 13, -8, -5, -3, 20, 3, 0, 17, 23, -4, 10, -1, 4, -9, 18, 0, -17, 7, 36, 9, 19, 1, 15, 0, -21, 10, -12, -8}
;

const int16_t conv1d_117_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{7, 50, -66}
, {35, -98, 50}
, {-17, 10, -31}
, {27, -24, -53}
, {-9, -102, 60}
, {110, 79, -11}
, {-23, -60, -74}
, {-43, 91, 66}
, {47, 14, 51}
, {-3, 33, -19}
, {38, 63, 97}
, {54, 29, 36}
, {89, -40, -55}
, {-12, 13, 67}
, {-45, 64, -43}
, {-89, -45, -38}
}
, {{-23, 62, 6}
, {74, -12, -15}
, {-49, -51, -62}
, {90, -79, 0}
, {91, -68, -45}
, {82, -47, 121}
, {-27, -55, -61}
, {79, -69, -11}
, {-36, 108, -88}
, {22, -93, 31}
, {76, -65, -99}
, {19, 65, 30}
, {56, -78, 86}
, {-77, -40, -13}
, {-23, 8, 23}
, {-82, -59, 62}
}
, {{70, 17, 81}
, {69, -62, 12}
, {-22, -27, -108}
, {49, 77, 0}
, {65, -24, 42}
, {16, 4, 83}
, {0, -99, 1}
, {-52, 43, -80}
, {60, 71, -71}
, {70, -75, -10}
, {61, -100, -33}
, {0, 35, 29}
, {85, -72, 41}
, {-115, 104, -103}
, {-99, 73, 70}
, {11, 50, -89}
}
, {{14, -60, -94}
, {11, -98, 34}
, {105, -6, -6}
, {49, -43, 24}
, {-74, -79, -83}
, {29, 102, 26}
, {-49, 58, 42}
, {-63, 35, -11}
, {83, 27, -85}
, {-51, -92, 23}
, {-67, -36, -42}
, {-69, 49, -70}
, {-60, 35, 33}
, {-88, -22, 66}
, {38, 68, -18}
, {0, 89, -57}
}
, {{19, -29, -64}
, {-94, 65, -2}
, {-15, -63, -92}
, {-29, 63, 57}
, {-57, 60, 12}
, {-84, -47, 103}
, {-65, 24, -56}
, {111, -4, 98}
, {14, -38, -38}
, {28, 46, 4}
, {-29, 71, 77}
, {43, 15, -51}
, {-47, 88, 94}
, {-57, 18, -61}
, {70, -38, 13}
, {-66, -35, -46}
}
, {{-59, 41, -99}
, {-62, 37, -77}
, {73, -56, -90}
, {-33, 96, -44}
, {-76, -5, 107}
, {103, 50, -19}
, {-17, 49, -58}
, {-8, -95, -81}
, {-30, 83, -78}
, {-17, 67, 43}
, {10, -25, 72}
, {63, -74, -40}
, {-21, 59, 22}
, {106, 34, -28}
, {101, -83, 43}
, {49, 95, -37}
}
, {{-90, -71, 27}
, {-86, -9, -4}
, {78, 29, -26}
, {-24, 20, 63}
, {-56, -79, 11}
, {79, 3, -8}
, {7, 65, 107}
, {36, -82, 53}
, {16, 37, 79}
, {108, 31, 51}
, {-8, -76, 19}
, {62, 77, 44}
, {-20, -88, -26}
, {-45, 9, 106}
, {-77, -34, 7}
, {-36, -72, 27}
}
, {{81, -30, 109}
, {67, 125, -84}
, {-19, -85, 95}
, {117, 91, -83}
, {-14, -7, -77}
, {-112, 47, -15}
, {34, -29, 79}
, {2, 5, 10}
, {107, -81, 22}
, {8, -98, -1}
, {-77, -53, 94}
, {-120, -4, 14}
, {21, 8, -94}
, {65, 103, 5}
, {101, 31, -3}
, {22, 91, -26}
}
, {{-72, 11, -115}
, {53, -72, 60}
, {-22, 79, -62}
, {-78, 79, 59}
, {90, -39, -98}
, {-7, 29, -76}
, {61, -22, -79}
, {-15, 92, -59}
, {14, 72, -81}
, {-83, -30, 0}
, {-103, 70, 25}
, {-85, -100, -63}
, {89, 68, 32}
, {-75, -90, -46}
, {-124, -22, 66}
, {-38, -20, -17}
}
, {{18, 0, -54}
, {-19, -34, 31}
, {27, 45, -30}
, {56, 81, -85}
, {-111, 76, -49}
, {2, -81, -31}
, {-31, 0, -5}
, {14, 63, 2}
, {10, -13, 57}
, {78, 71, 100}
, {-87, -55, -3}
, {-52, 52, -41}
, {-6, 0, -52}
, {-54, 63, 89}
, {-36, 94, 30}
, {50, -10, -48}
}
, {{59, -17, -34}
, {17, 55, -5}
, {-31, 52, -32}
, {-67, 86, 28}
, {41, 9, 18}
, {1, -44, 48}
, {-22, -49, -86}
, {95, 0, 73}
, {12, 13, -12}
, {28, -58, 73}
, {34, -86, -90}
, {106, 27, -34}
, {-61, 93, 69}
, {-36, -19, 104}
, {-15, -68, -63}
, {105, 32, 17}
}
, {{-21, -70, 45}
, {60, -19, 97}
, {5, -76, -14}
, {-8, 1, 49}
, {57, 106, -86}
, {70, -12, -9}
, {75, 12, -81}
, {80, -17, -69}
, {-40, 103, 81}
, {81, -73, 30}
, {-34, -15, 40}
, {-82, 26, -16}
, {-92, -34, 21}
, {56, -98, -62}
, {-66, 76, 90}
, {-91, 15, 81}
}
, {{53, -73, -51}
, {73, -15, 100}
, {-23, 71, -88}
, {-70, 3, -100}
, {33, -19, 96}
, {103, -103, 53}
, {-17, 71, -82}
, {-30, -3, 97}
, {49, 55, -42}
, {-67, 96, -48}
, {-61, 65, -55}
, {53, 47, -73}
, {-36, -36, -24}
, {56, 16, 71}
, {-53, -84, 36}
, {42, -58, 86}
}
, {{-95, -36, -99}
, {109, -100, 23}
, {-28, 56, -42}
, {55, 60, 8}
, {62, -60, 77}
, {92, -101, -12}
, {-54, -34, 55}
, {86, -30, -54}
, {-103, -62, 17}
, {-77, 19, 115}
, {109, 25, -25}
, {93, 86, -30}
, {-38, -72, -85}
, {97, -51, 66}
, {30, 54, 71}
, {23, -27, 113}
}
, {{92, 11, 75}
, {76, -41, -87}
, {-58, -44, 94}
, {102, -14, -78}
, {33, -11, -4}
, {-27, -89, 54}
, {13, 24, -9}
, {-52, -98, 63}
, {25, -23, 59}
, {-49, 84, -15}
, {-48, -3, 80}
, {-34, -68, -24}
, {53, -57, 1}
, {94, -70, -84}
, {40, 23, 60}
, {32, 58, -86}
}
, {{75, 47, -37}
, {-28, 34, 80}
, {29, -39, -40}
, {4, 77, 35}
, {78, -36, -94}
, {-82, 29, 58}
, {36, -40, -91}
, {-61, 1, -73}
, {79, -17, 6}
, {-43, -90, -100}
, {83, 47, -9}
, {17, -5, -53}
, {-32, -22, 22}
, {-86, 21, 67}
, {-76, 69, 99}
, {-39, 90, 73}
}
, {{13, -47, 70}
, {53, -14, -38}
, {-69, 87, 62}
, {31, 54, -92}
, {-48, 52, 95}
, {-26, -13, -29}
, {-59, 40, 60}
, {-104, 95, 23}
, {97, 15, 10}
, {-114, 103, -80}
, {-26, -89, 8}
, {-75, -101, 34}
, {-35, -18, -66}
, {35, 35, 86}
, {0, 39, -9}
, {15, -40, 48}
}
, {{-71, 38, 55}
, {-86, 61, -54}
, {104, -82, 4}
, {98, 10, -83}
, {-82, -63, -70}
, {61, 71, -63}
, {84, -95, -85}
, {54, 30, 62}
, {68, 23, 6}
, {48, 21, -54}
, {9, -88, 32}
, {-70, 110, 50}
, {94, 71, 111}
, {19, -49, 69}
, {-83, -72, -49}
, {-34, -30, -33}
}
, {{69, 38, 46}
, {51, -17, 26}
, {45, 60, 31}
, {-32, 75, 104}
, {-33, -22, -45}
, {-24, -57, -78}
, {7, -57, -1}
, {-77, 88, -32}
, {-68, -34, 73}
, {80, 5, -70}
, {-39, -75, -9}
, {-70, -77, -8}
, {69, 83, -10}
, {83, -40, 108}
, {1, 6, -91}
, {64, -85, 23}
}
, {{-49, -8, 93}
, {-92, 75, -67}
, {-85, 91, -85}
, {-50, -53, 96}
, {-53, 12, -9}
, {-78, 97, 91}
, {-13, 2, -52}
, {-58, -58, 9}
, {-57, 32, -117}
, {6, -95, 24}
, {-12, -7, 68}
, {73, -24, 71}
, {-88, -70, -74}
, {-53, -13, -15}
, {-7, 69, -9}
, {55, 3, 36}
}
, {{-34, 61, 1}
, {86, 85, -95}
, {41, 40, -36}
, {-38, -49, -39}
, {10, 71, -43}
, {-18, 94, 70}
, {65, 63, -75}
, {-75, 36, -107}
, {14, 84, -89}
, {93, -31, -82}
, {23, -94, 102}
, {-43, 58, -14}
, {-78, -105, 72}
, {57, -74, 98}
, {91, 40, 62}
, {-5, 20, -68}
}
, {{81, -102, 111}
, {-38, 77, 64}
, {23, -84, 73}
, {-33, -75, 98}
, {-106, -24, -92}
, {-34, 106, -59}
, {-83, 0, -30}
, {-65, -31, 61}
, {-62, -33, 111}
, {-58, -50, 45}
, {-1, -80, 67}
, {-54, 123, -2}
, {16, -7, 72}
, {77, -61, 108}
, {3, 54, 57}
, {54, 91, 34}
}
, {{72, 34, -54}
, {-25, 44, 26}
, {-36, 46, 78}
, {79, -7, -51}
, {-84, -36, 100}
, {-16, -34, 42}
, {-70, -71, 15}
, {-79, 68, 9}
, {67, 41, 60}
, {-67, 35, 68}
, {-40, 72, -67}
, {70, 110, -81}
, {32, -84, 41}
, {34, 45, -74}
, {-85, 28, -39}
, {-60, 1, -52}
}
, {{-3, -51, 70}
, {81, -74, -71}
, {-45, -97, -72}
, {14, -31, 86}
, {44, -74, -67}
, {18, 96, 96}
, {91, -45, -29}
, {95, 42, 106}
, {43, -52, 89}
, {27, -3, -77}
, {46, -112, -25}
, {-34, -48, -44}
, {-34, 71, -88}
, {98, 73, 7}
, {53, -5, -92}
, {-12, -28, -51}
}
, {{-54, 62, -86}
, {-24, 16, 30}
, {33, -78, 20}
, {-73, -85, 48}
, {11, 32, 34}
, {62, 37, 95}
, {-104, -36, -94}
, {80, 3, 88}
, {18, 38, -81}
, {-69, -63, 98}
, {1, -60, 20}
, {-17, -39, -77}
, {-100, -21, 106}
, {-69, 66, -88}
, {-17, 88, -53}
, {-66, -67, 34}
}
, {{-54, -86, 34}
, {-58, -77, -59}
, {-2, -96, 51}
, {89, 98, -73}
, {30, 21, 16}
, {25, -92, -16}
, {-16, -105, 99}
, {49, 96, 112}
, {11, -32, 68}
, {60, 5, 63}
, {8, 5, 47}
, {44, -60, -3}
, {-74, -80, -93}
, {2, -87, -44}
, {51, 35, -52}
, {-71, 73, 18}
}
, {{-81, 104, -83}
, {-62, -75, -42}
, {-20, -39, 125}
, {-70, -85, 60}
, {77, 78, 32}
, {-23, 92, -54}
, {94, -63, -75}
, {0, -19, 30}
, {79, 34, 9}
, {-82, 42, 36}
, {-24, 122, -68}
, {88, 35, 29}
, {-68, -21, -16}
, {48, 12, -49}
, {-2, 65, -45}
, {-98, -39, -36}
}
, {{-79, 41, -33}
, {-21, -104, 11}
, {-22, 43, 66}
, {-2, -34, -11}
, {-53, -7, -29}
, {-67, 75, 78}
, {-17, -21, -2}
, {103, 88, -15}
, {-69, -93, 19}
, {103, 61, -68}
, {100, 69, -87}
, {-80, -70, 52}
, {-75, 10, 30}
, {104, 100, -95}
, {-49, 58, -6}
, {-74, 33, -81}
}
, {{65, -64, -55}
, {39, -100, 73}
, {60, 68, -14}
, {-77, -49, -50}
, {-106, 42, 9}
, {67, 34, 16}
, {-5, -67, -2}
, {23, -100, 68}
, {-66, 50, 84}
, {23, -4, 20}
, {49, 75, 13}
, {-109, -85, 17}
, {10, -51, -110}
, {70, -24, 46}
, {68, 91, 10}
, {90, 105, -4}
}
, {{-13, -90, 78}
, {99, -93, 37}
, {-44, 42, -84}
, {16, -56, 32}
, {87, -83, 26}
, {59, 39, -43}
, {81, -20, 56}
, {2, -35, -86}
, {-32, 90, 58}
, {67, 49, 44}
, {-70, -26, 9}
, {-74, 20, 68}
, {27, 37, -70}
, {40, -46, 58}
, {85, -15, 122}
, {73, 12, 120}
}
, {{-39, 95, -22}
, {22, 14, 26}
, {42, 41, 98}
, {85, 94, 90}
, {47, -95, 20}
, {113, 53, 77}
, {10, -56, -48}
, {28, -26, -48}
, {77, 2, 26}
, {-62, 59, 77}
, {71, -92, 40}
, {-58, 94, 106}
, {-33, -90, 49}
, {-103, 6, 27}
, {52, -33, -39}
, {-82, -90, 11}
}
, {{27, 31, -15}
, {-72, -2, -47}
, {98, -24, -35}
, {6, -59, 84}
, {-46, -64, -23}
, {35, -13, 72}
, {-1, -82, -73}
, {-43, 90, -11}
, {89, 20, -51}
, {93, -12, 95}
, {11, -9, -61}
, {-26, -50, -27}
, {-37, -90, -82}
, {-53, 42, -79}
, {83, -29, -86}
, {105, 45, -92}
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
#define INPUT_SAMPLES   249
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_147_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_147(
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
#define INPUT_SAMPLES       124
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_118_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_118(
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


const int16_t conv1d_118_bias[CONV_FILTERS] = {8, 11, 15, -4, 12, 10, 0, -6, 10, -13, 10, 6, -1, 21, 19, 10, 8, 11, -1, 16, -1, -6, -20, -4, 16, 21, 22, -3, 14, 0, 18, 15, 6, 11, -10, 14, -3, -1, 0, 24, 4, 0, -18, 8, 11, -10, 0, 36, 23, 6, 0, 14, 0, 1, 4, 26, 4, 14, 15, -18, -7, -12, -15, 9}
;

const int16_t conv1d_118_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{19, -55, -41}
, {64, 68, 77}
, {-62, -1, -7}
, {20, 41, -4}
, {-51, 86, 40}
, {49, 35, -54}
, {-82, 59, -52}
, {-66, 47, -4}
, {37, -67, 48}
, {12, -41, -1}
, {68, 1, -35}
, {-71, 31, 28}
, {40, -37, 49}
, {-60, 57, 52}
, {-15, 44, 46}
, {60, 63, 18}
, {60, -32, -74}
, {-48, 75, 58}
, {-57, -21, -3}
, {59, -9, 41}
, {-23, -4, -65}
, {36, 47, -50}
, {26, -44, -25}
, {-25, -65, 72}
, {-74, 70, 59}
, {-48, 11, -64}
, {33, -63, -9}
, {52, -63, 39}
, {22, -19, -41}
, {28, -38, -52}
, {57, 9, 47}
, {62, -62, 71}
}
, {{36, 19, 43}
, {72, 42, 59}
, {8, -19, 59}
, {-40, -61, -32}
, {8, -12, 14}
, {-73, -43, -61}
, {46, -32, -59}
, {18, -21, 19}
, {20, 13, 64}
, {-21, -45, 62}
, {65, 11, 16}
, {70, 9, 45}
, {52, -18, -25}
, {54, 31, -64}
, {36, 28, -31}
, {3, 67, 69}
, {-69, 54, -65}
, {-19, 68, -77}
, {-46, -18, 20}
, {7, -14, 19}
, {11, -22, -35}
, {-39, -8, 34}
, {-4, 4, 58}
, {61, -55, -43}
, {65, 4, 46}
, {74, 75, -33}
, {59, -46, 53}
, {-40, -73, -2}
, {-30, -23, -71}
, {58, 12, 60}
, {40, 69, 12}
, {62, -75, -36}
}
, {{41, 52, 66}
, {-4, -8, 41}
, {-21, 77, -26}
, {62, 39, -91}
, {14, -23, -52}
, {3, 53, -28}
, {-22, -6, 53}
, {32, 30, -43}
, {21, -2, 70}
, {9, -50, -9}
, {-32, -39, 38}
, {-53, -1, 7}
, {57, -24, 44}
, {-48, 44, 4}
, {-36, 12, 18}
, {24, -62, -36}
, {-31, 6, -52}
, {52, 12, -56}
, {-50, -49, 30}
, {25, -47, 0}
, {-40, -1, -26}
, {20, -39, 20}
, {10, -38, 40}
, {27, -4, -62}
, {43, 7, 52}
, {-46, -45, 33}
, {72, 33, 16}
, {72, 61, -48}
, {64, -4, -51}
, {-24, -72, 65}
, {1, -23, -3}
, {-24, -20, -52}
}
, {{-1, 66, -6}
, {27, 23, 31}
, {64, -32, -68}
, {-39, 50, -62}
, {-13, -23, 39}
, {10, 41, 12}
, {29, 58, 45}
, {-77, -74, 2}
, {8, -60, 2}
, {16, -31, -13}
, {47, 62, 29}
, {71, -17, -4}
, {46, -55, 6}
, {32, 22, -46}
, {7, 43, -30}
, {-73, -69, 47}
, {15, 33, 52}
, {-40, -20, 0}
, {-67, -75, 61}
, {40, -30, -39}
, {67, 46, -9}
, {-80, -45, -25}
, {27, 40, -47}
, {-36, -49, -42}
, {-33, 40, 37}
, {-57, -56, 71}
, {23, 28, -68}
, {-47, 7, 0}
, {37, -64, 56}
, {24, -34, 62}
, {-19, 68, -25}
, {-2, -58, -55}
}
, {{2, 44, -12}
, {-59, -19, 23}
, {16, 57, -44}
, {-74, -56, -48}
, {57, 43, 33}
, {63, -31, -34}
, {-23, 21, 6}
, {43, 8, 9}
, {37, -51, 32}
, {30, 5, 5}
, {53, 50, -4}
, {13, 0, 27}
, {13, 46, 27}
, {39, -9, -50}
, {0, 45, -43}
, {-40, -77, 41}
, {-8, 0, -59}
, {-42, -45, -79}
, {-45, 22, 18}
, {63, -74, -16}
, {64, -5, -16}
, {-59, 52, 36}
, {-69, -59, -63}
, {54, -70, -58}
, {32, 8, -76}
, {-69, -55, -39}
, {58, -12, -3}
, {-30, -2, 3}
, {-17, -37, 14}
, {80, 75, 40}
, {19, -66, 64}
, {-36, -39, -8}
}
, {{-61, 61, -25}
, {7, -40, -62}
, {-36, -19, -30}
, {-25, 13, 24}
, {-65, -35, 47}
, {-45, 19, 57}
, {25, 7, -34}
, {-67, -24, -71}
, {-63, 70, 31}
, {58, 11, 78}
, {-51, -61, 56}
, {33, 29, 0}
, {74, 0, -34}
, {-4, 14, 48}
, {8, -57, 52}
, {-13, 53, 14}
, {-51, 28, -18}
, {-22, -62, -30}
, {-21, -15, 4}
, {-40, 76, 76}
, {-1, 66, 17}
, {-21, 60, -14}
, {-16, -46, 62}
, {39, 34, -37}
, {79, 24, -19}
, {-31, 30, 22}
, {15, 64, 27}
, {-46, 62, 47}
, {74, -51, 57}
, {-69, -4, -2}
, {11, 4, 69}
, {1, -39, 14}
}
, {{-34, -68, -64}
, {38, 29, 30}
, {70, 68, 74}
, {29, 49, -23}
, {-44, 49, -29}
, {-36, 45, -71}
, {48, -25, -71}
, {-23, -56, 33}
, {18, -52, -73}
, {28, -15, -4}
, {64, -9, -27}
, {61, 66, 57}
, {-65, 19, 41}
, {-39, -4, 61}
, {-73, 28, -39}
, {-9, -34, 63}
, {62, -23, 57}
, {22, -47, 14}
, {-15, 11, -39}
, {24, -19, 26}
, {-51, -47, -29}
, {-68, 33, -65}
, {-36, 19, -62}
, {50, -3, 17}
, {-69, -43, 69}
, {-47, 15, -58}
, {64, 64, 59}
, {-64, 15, -20}
, {50, -23, 0}
, {71, -39, -1}
, {2, 21, -47}
, {44, -1, -28}
}
, {{22, -48, -54}
, {-25, 49, 47}
, {49, -30, 53}
, {-7, 28, 50}
, {-20, 5, -61}
, {23, -12, -60}
, {2, 27, -63}
, {75, 15, 54}
, {45, 20, -1}
, {20, -11, -36}
, {62, 38, -21}
, {69, -51, -69}
, {-48, 10, -33}
, {39, 24, -22}
, {-3, -30, -21}
, {34, 33, 23}
, {23, 39, 2}
, {51, -50, -53}
, {-28, -79, 82}
, {65, -42, -29}
, {-47, -56, -4}
, {51, -33, 40}
, {-45, 67, -62}
, {70, -65, -54}
, {-64, -66, -62}
, {18, -74, -39}
, {-29, 8, 42}
, {28, -79, 46}
, {21, 26, 1}
, {74, -18, 35}
, {-24, 20, 69}
, {-13, 0, -36}
}
, {{-4, 8, -63}
, {63, 81, -70}
, {69, -69, 41}
, {-43, 57, 85}
, {-38, 15, -41}
, {-50, -53, -69}
, {-75, -42, -71}
, {-43, 38, 31}
, {28, -26, 72}
, {0, -65, -32}
, {16, 52, -19}
, {43, 15, 12}
, {-57, 60, -56}
, {4, -29, -8}
, {37, 38, 60}
, {66, -41, -21}
, {-36, 9, 20}
, {43, 1, -34}
, {12, 31, 73}
, {-36, -23, 54}
, {-8, -17, 53}
, {-64, -9, 27}
, {5, -8, -29}
, {-24, 47, 3}
, {-33, -70, 1}
, {-72, -71, 38}
, {-9, 2, 40}
, {65, 55, -35}
, {-53, -73, 34}
, {57, 41, 21}
, {-23, 13, -11}
, {40, -22, 12}
}
, {{67, -6, -16}
, {-81, -69, -12}
, {-27, 44, -50}
, {-53, -27, -7}
, {25, -28, 50}
, {-61, -47, 67}
, {58, 50, 44}
, {56, 34, -1}
, {-47, -9, 26}
, {-42, 81, -57}
, {-3, -12, -23}
, {-12, -61, -9}
, {-49, -8, 75}
, {70, -8, -49}
, {-47, -62, -59}
, {-41, -44, -32}
, {-43, -4, -42}
, {17, -27, 16}
, {15, -7, -70}
, {-84, 80, -2}
, {-30, -70, -10}
, {-8, 30, -28}
, {18, 13, 20}
, {75, 14, 45}
, {-11, -38, 28}
, {59, 52, -20}
, {-7, -54, -25}
, {35, -13, -22}
, {-40, 26, 43}
, {-26, 51, -3}
, {37, 27, -39}
, {15, -60, -58}
}
, {{-4, -21, -5}
, {-10, 79, 72}
, {69, -59, -32}
, {6, -20, -29}
, {27, 83, -5}
, {27, -48, -65}
, {-30, -1, -53}
, {13, -77, -49}
, {-53, -88, 31}
, {26, -47, 17}
, {-34, 33, -3}
, {-69, -36, -53}
, {31, -24, 30}
, {15, 15, -35}
, {-81, -59, 40}
, {38, 56, -75}
, {-26, -69, -64}
, {31, 37, 11}
, {56, 51, -43}
, {39, 54, 63}
, {36, 31, 61}
, {49, -71, -67}
, {-15, 6, 24}
, {42, 19, 68}
, {-33, -32, -15}
, {39, -27, 82}
, {87, 16, -15}
, {-63, 83, -48}
, {41, 39, 4}
, {24, -22, -66}
, {4, 26, 75}
, {-55, 56, -29}
}
, {{-28, 3, -38}
, {-67, 0, 8}
, {57, -24, 35}
, {4, -39, -9}
, {5, -29, 79}
, {-71, 40, -9}
, {43, -44, 23}
, {-14, 16, -36}
, {-45, 13, 35}
, {25, 0, -33}
, {75, -2, 52}
, {-69, -28, 41}
, {68, 18, -56}
, {-61, -51, 51}
, {-58, 1, 28}
, {62, -5, -67}
, {48, 42, 36}
, {75, 29, 71}
, {-47, -61, -26}
, {-60, -50, 46}
, {21, -49, -48}
, {-31, -12, 6}
, {-23, -61, 44}
, {20, 24, 20}
, {42, -66, -12}
, {6, 30, 32}
, {-16, 34, 71}
, {39, 16, 28}
, {-46, 21, 8}
, {74, -17, 14}
, {-36, -50, 39}
, {-27, -52, -7}
}
, {{-54, 35, -4}
, {-64, -2, 34}
, {66, 7, -8}
, {-53, 0, -16}
, {36, -52, 5}
, {-60, 4, -55}
, {13, 66, 14}
, {1, 52, -17}
, {-31, 13, 50}
, {-50, -53, 10}
, {25, -19, -20}
, {-28, 7, -37}
, {-69, -56, 53}
, {-61, -36, -54}
, {-43, 31, -66}
, {-63, -10, 24}
, {43, 39, 32}
, {59, -35, -42}
, {-66, -2, -74}
, {-70, -64, 39}
, {-63, -32, 0}
, {44, -27, 53}
, {-15, 26, -58}
, {11, -38, -60}
, {70, -44, 58}
, {55, 0, -11}
, {-41, -28, -66}
, {-48, -6, 27}
, {2, -17, -47}
, {-50, -43, 1}
, {-38, -52, -21}
, {-72, 66, 25}
}
, {{9, 44, 31}
, {36, 29, -28}
, {-52, -57, 61}
, {-51, 21, -33}
, {59, -13, 3}
, {-59, -62, -37}
, {53, 42, 14}
, {-45, -28, 69}
, {33, 45, -22}
, {34, -26, -13}
, {-13, 33, 19}
, {-24, 51, -40}
, {-34, 36, -42}
, {64, -44, -55}
, {23, 38, -24}
, {-40, -31, 32}
, {44, 37, 77}
, {-6, -13, 40}
, {58, 38, 41}
, {-59, -42, -72}
, {8, -29, -54}
, {-10, -47, 15}
, {24, 23, 14}
, {32, 1, 0}
, {20, -67, -35}
, {-59, 5, -3}
, {47, -25, -45}
, {-67, -55, -37}
, {14, 22, 18}
, {55, -61, 39}
, {-26, -48, -58}
, {36, -46, -61}
}
, {{-55, -53, -74}
, {-35, 49, -76}
, {-26, -63, -34}
, {59, 45, -76}
, {65, 37, 11}
, {-21, -50, 45}
, {63, -31, 67}
, {32, -75, 57}
, {62, 24, -37}
, {-22, -66, -3}
, {0, 74, 66}
, {-54, 38, 64}
, {-26, -34, -77}
, {-5, -48, 56}
, {13, 6, 74}
, {-2, 52, 2}
, {62, 29, 16}
, {-75, -23, -20}
, {47, 45, 79}
, {32, 13, 63}
, {-57, -59, 48}
, {-67, -5, 54}
, {79, -32, -36}
, {64, -5, -29}
, {-9, -61, 42}
, {68, 56, -38}
, {22, 57, 18}
, {-25, 7, -31}
, {-8, -72, 0}
, {-27, -26, 0}
, {19, -37, -3}
, {10, 0, -65}
}
, {{4, -38, 8}
, {-18, -9, 50}
, {74, -19, 25}
, {-53, -8, 72}
, {29, 5, -60}
, {-12, -73, 11}
, {52, -9, -6}
, {21, -22, -36}
, {70, -6, 48}
, {71, -29, -44}
, {-49, 35, -72}
, {64, 26, 27}
, {-28, 23, -48}
, {65, 13, 1}
, {-3, 52, 14}
, {-7, -15, -48}
, {-54, -56, 26}
, {-70, 48, 7}
, {49, 29, 2}
, {-7, 30, 54}
, {39, -42, 17}
, {-66, 32, 66}
, {4, 52, 31}
, {-27, 34, 55}
, {16, 28, 66}
, {50, -31, 6}
, {-62, -24, 50}
, {-62, 36, -17}
, {-6, 48, 21}
, {-32, 66, 7}
, {-4, 8, -73}
, {-82, 1, -24}
}
, {{18, 37, -55}
, {45, 33, -53}
, {-39, 15, -4}
, {4, -35, 23}
, {-36, -40, -84}
, {-21, -48, 61}
, {3, 64, 74}
, {40, -80, 12}
, {-48, 66, -30}
, {-22, 58, -66}
, {-51, 47, 27}
, {-55, -5, 52}
, {-46, 67, -1}
, {37, 59, 2}
, {28, -15, -38}
, {48, 52, -48}
, {-42, -38, -35}
, {48, -23, -55}
, {-7, 16, 73}
, {-15, 45, 40}
, {45, -5, -43}
, {23, -80, 53}
, {-52, -60, 32}
, {-41, -17, 25}
, {49, -43, -66}
, {39, 17, -12}
, {-50, 71, 27}
, {72, 13, -53}
, {24, -53, 0}
, {16, 43, 55}
, {15, -65, -28}
, {34, -52, 7}
}
, {{-27, -1, -25}
, {-25, 69, -12}
, {-15, 74, -64}
, {63, -44, 8}
, {1, -28, 38}
, {16, -44, 18}
, {47, -24, -36}
, {85, 73, -60}
, {-33, 44, 27}
, {74, -29, -64}
, {-42, -50, 46}
, {-5, -13, -53}
, {-23, 89, -4}
, {62, -19, -36}
, {85, -37, -66}
, {-21, 4, 63}
, {-70, 25, 27}
, {8, 76, -45}
, {-5, 18, -2}
, {13, -86, -26}
, {-5, 17, -56}
, {-19, -43, -62}
, {10, 74, 27}
, {-11, -3, 16}
, {47, 22, 32}
, {15, 48, -49}
, {-6, -23, -1}
, {-10, 38, 15}
, {44, 1, -60}
, {-12, -37, 72}
, {-22, -16, -13}
, {3, 0, -66}
}
, {{-32, 37, 41}
, {-16, -59, -48}
, {36, -13, -66}
, {18, -78, -28}
, {0, -10, 52}
, {1, 33, -34}
, {32, 44, -38}
, {45, -38, -29}
, {-64, 37, -1}
, {67, 13, -2}
, {60, 54, 74}
, {-58, -70, 35}
, {22, 21, -21}
, {9, -42, 2}
, {47, 52, 55}
, {13, 12, 48}
, {-18, -3, -10}
, {-6, -23, 73}
, {26, 9, -56}
, {-34, 36, -46}
, {-38, -4, -62}
, {-1, 70, 45}
, {60, 59, 9}
, {54, -10, -8}
, {-35, -47, -47}
, {-67, 62, -12}
, {-61, -42, 5}
, {-20, 67, 56}
, {66, -50, 55}
, {-12, -2, 41}
, {-11, -10, -60}
, {12, 17, -36}
}
, {{-15, 38, -14}
, {-22, 21, -18}
, {-48, 3, -32}
, {24, -26, 73}
, {54, 70, -22}
, {-18, 29, 13}
, {-17, 78, -24}
, {10, 20, -51}
, {-21, 56, -52}
, {13, -33, -44}
, {-68, 42, 83}
, {27, 25, -31}
, {43, -28, 11}
, {-40, 66, 54}
, {14, -28, -24}
, {24, 29, -46}
, {3, 35, -62}
, {80, -57, -17}
, {59, 42, -55}
, {50, 12, 1}
, {-66, -20, -48}
, {26, -33, 17}
, {-28, 23, -6}
, {68, 15, -37}
, {52, 49, 71}
, {-29, -48, 33}
, {54, -49, -24}
, {26, -46, 42}
, {23, 16, 51}
, {42, -21, 48}
, {-55, -2, -13}
, {-21, -39, 21}
}
, {{-19, 18, 8}
, {47, -39, -33}
, {22, -35, 4}
, {69, 24, 29}
, {-43, -56, -23}
, {-26, 36, -75}
, {20, 1, 41}
, {81, 60, 8}
, {-36, -40, -8}
, {47, -37, 65}
, {-56, 1, -26}
, {-57, 34, 12}
, {-60, 44, -28}
, {-63, 34, -53}
, {60, 44, 10}
, {-36, 43, -36}
, {45, -35, -48}
, {29, 66, 58}
, {31, 34, 30}
, {0, 46, -5}
, {-33, -50, 42}
, {-17, 51, -40}
, {40, 0, -2}
, {-11, 22, 18}
, {-29, -7, -40}
, {32, -57, 47}
, {45, 53, 58}
, {-20, 10, -31}
, {-52, 61, 68}
, {-64, 3, -53}
, {39, 34, 45}
, {-18, 49, 58}
}
, {{-56, -36, -37}
, {-65, 45, 55}
, {-62, -8, 61}
, {-43, -40, 20}
, {50, -55, 39}
, {-23, 49, 39}
, {10, -64, -62}
, {28, -63, -28}
, {-15, -21, -94}
, {23, -31, 8}
, {-22, -13, 28}
, {-38, 18, 57}
, {67, 16, -14}
, {-52, 19, -36}
, {-38, 1, -55}
, {71, 42, -46}
, {3, 20, 4}
, {16, -66, -37}
, {-63, -49, -75}
, {51, 37, -6}
, {18, 64, 41}
, {-12, -39, -50}
, {-51, -23, -25}
, {-26, 26, -75}
, {-26, 42, -13}
, {51, -27, 11}
, {50, 68, 69}
, {2, 55, -40}
, {36, -30, 54}
, {-68, -55, 12}
, {74, 66, -73}
, {-60, 30, 25}
}
, {{65, 68, -66}
, {-15, 41, 38}
, {6, 58, -35}
, {-30, 10, -54}
, {-22, -43, 26}
, {-65, 61, -24}
, {27, -68, 49}
, {28, 57, 64}
, {12, -19, 52}
, {-40, -79, -79}
, {27, 69, 38}
, {-31, -80, 43}
, {54, 12, -62}
, {40, 57, -42}
, {-70, -56, 22}
, {9, -43, -35}
, {54, 0, -74}
, {-54, -32, 0}
, {-36, 62, -43}
, {56, 29, -53}
, {22, -33, 25}
, {-6, 53, 59}
, {29, 32, 60}
, {30, -37, -26}
, {66, -74, -1}
, {6, -71, -32}
, {-66, -62, -65}
, {-5, -65, 40}
, {56, 67, 69}
, {53, -20, 51}
, {44, 46, -34}
, {34, 25, -23}
}
, {{-46, 69, 45}
, {27, -50, -59}
, {-45, -46, -26}
, {-26, 5, -14}
, {74, 82, 38}
, {-10, 35, -16}
, {61, -50, -49}
, {2, -42, -55}
, {-12, 28, 53}
, {10, 78, -23}
, {54, 8, -35}
, {1, -28, 0}
, {31, 4, -50}
, {44, 19, -34}
, {-22, -6, -3}
, {-69, -75, -2}
, {-17, -50, -35}
, {46, 56, 21}
, {1, -6, 50}
, {-14, -72, 17}
, {-55, 36, 41}
, {0, -15, -54}
, {-47, 26, 9}
, {-12, 0, 17}
, {6, 64, -57}
, {-36, 34, 26}
, {55, -62, -1}
, {62, 35, 77}
, {-64, -21, -21}
, {-64, 35, -14}
, {28, -32, 68}
, {-8, -4, 46}
}
, {{71, -23, -17}
, {-12, 78, 43}
, {-25, -24, -62}
, {-36, 10, -47}
, {-36, 35, 68}
, {40, 26, -33}
, {7, -45, -12}
, {-43, -13, -7}
, {43, 0, 0}
, {-34, -61, 27}
, {42, -27, 28}
, {-40, -53, 60}
, {60, -44, -49}
, {34, -4, 81}
, {-75, -19, -55}
, {-40, 3, 37}
, {-68, -72, -10}
, {18, 63, 24}
, {-6, -18, 46}
, {28, 5, -41}
, {46, -63, 55}
, {-8, -16, -11}
, {-68, 26, -89}
, {12, 57, -21}
, {-8, -2, 23}
, {-2, 41, 3}
, {59, 30, -56}
, {24, 52, 39}
, {23, 2, 1}
, {11, 11, 12}
, {11, -50, 75}
, {45, -70, 42}
}
, {{-1, -1, 0}
, {4, 75, 60}
, {74, 28, 62}
, {5, -7, -59}
, {47, -10, -7}
, {-17, -4, -17}
, {-55, 0, -45}
, {-32, -87, 67}
, {-64, 66, -61}
, {43, 2, -43}
, {-42, -34, 46}
, {-4, -59, 73}
, {50, 87, -18}
, {-1, 53, -21}
, {-26, -65, -64}
, {55, -16, 1}
, {29, -14, 49}
, {-53, -47, -36}
, {-56, -30, 22}
, {38, 7, 93}
, {-56, -35, -4}
, {11, -23, 63}
, {86, -12, -46}
, {19, 18, 25}
, {-33, 60, 72}
, {3, -24, -66}
, {-3, 52, 60}
, {29, 10, 1}
, {-28, 65, -24}
, {-78, 49, 14}
, {67, -49, 36}
, {-51, -18, -71}
}
, {{62, 60, -61}
, {-7, 37, 26}
, {-54, -46, 15}
, {-46, 70, -40}
, {22, 4, -50}
, {7, -43, -6}
, {54, 68, 58}
, {60, 66, -26}
, {54, 24, 44}
, {15, -58, 36}
, {8, -60, 9}
, {16, 54, 34}
, {-40, 54, -1}
, {-33, -4, 6}
, {-23, -26, -41}
, {-66, -67, -32}
, {-23, 3, -27}
, {78, -3, -52}
, {27, -16, 31}
, {80, 33, 17}
, {37, 16, -27}
, {-14, 46, 13}
, {12, 57, 90}
, {83, -34, -59}
, {50, 4, -49}
, {47, -49, 34}
, {81, -40, 67}
, {-35, 3, 14}
, {-54, -78, -12}
, {-46, 3, -66}
, {60, 58, -60}
, {-1, -70, -14}
}
, {{-24, -66, -31}
, {69, 73, -11}
, {-2, 40, -76}
, {-22, -23, 50}
, {-26, -3, 32}
, {37, -25, -2}
, {29, 22, 71}
, {-42, -44, 47}
, {-79, -24, -87}
, {31, 4, -59}
, {-37, 18, 58}
, {-32, 40, 53}
, {13, -34, 26}
, {10, 19, 4}
, {39, -44, -71}
, {-15, 40, 64}
, {-24, -7, -1}
, {44, 56, 21}
, {-60, -65, -27}
, {19, -12, 17}
, {-49, -44, 75}
, {1, 43, 59}
, {36, -38, 46}
, {-10, 42, 22}
, {-61, -53, 1}
, {47, 63, 57}
, {4, 18, 78}
, {-1, 22, -30}
, {51, -37, 7}
, {32, -11, -71}
, {62, -4, 70}
, {32, 21, 28}
}
, {{53, -47, -33}
, {-7, -42, 18}
, {56, 17, 23}
, {3, 11, -60}
, {-61, 0, -49}
, {-45, -23, -39}
, {-12, 42, -41}
, {-45, -84, 9}
, {-45, 2, 36}
, {0, -46, -31}
, {-47, -57, -66}
, {53, -21, 46}
, {52, -29, 5}
, {-82, -3, -65}
, {32, -14, 12}
, {-6, -42, -15}
, {45, 23, -51}
, {63, -16, 5}
, {6, -40, 17}
, {-58, 41, 5}
, {-48, -33, -47}
, {-31, 31, 41}
, {80, 8, 43}
, {31, 56, 48}
, {-11, 10, -46}
, {-11, -24, 29}
, {80, 15, 75}
, {70, -40, -63}
, {0, 12, 33}
, {31, 3, 62}
, {6, -5, 55}
, {-25, -37, 58}
}
, {{-4, 2, -33}
, {66, 6, -48}
, {-27, -3, 26}
, {65, -25, 7}
, {50, 29, 17}
, {55, -59, 28}
, {-43, 47, 19}
, {1, -50, 72}
, {72, 76, 34}
, {-4, 45, -9}
, {-51, -55, 36}
, {-42, 63, 57}
, {60, 50, 17}
, {-65, -26, -68}
, {-51, 54, 32}
, {-36, -49, -61}
, {-58, 36, -35}
, {73, -62, -25}
, {-73, -24, 18}
, {8, 45, -64}
, {-47, 57, 29}
, {-80, -27, -43}
, {7, 44, -63}
, {16, 39, 50}
, {36, 11, 18}
, {72, 56, -45}
, {-72, 71, 36}
, {-23, -38, -75}
, {-16, -35, -24}
, {47, 62, -58}
, {39, 26, 10}
, {-54, 38, 53}
}
, {{-22, 31, 8}
, {-4, -9, -5}
, {-53, 21, -9}
, {-38, 80, -16}
, {-2, -34, -60}
, {10, 9, 18}
, {-58, -12, 41}
, {-25, -22, -21}
, {0, -9, -42}
, {21, 6, 31}
, {59, 2, 58}
, {69, 35, 24}
, {-1, 41, -47}
, {-58, -25, -5}
, {-50, 15, 27}
, {-38, -62, -5}
, {-47, 37, 78}
, {12, -43, -57}
, {73, 38, 69}
, {-67, -13, -7}
, {-36, -33, -77}
, {5, 6, -43}
, {54, -37, -23}
, {3, -63, 38}
, {0, -24, -59}
, {43, -57, 29}
, {30, -80, -32}
, {-80, 60, 10}
, {-35, 50, 46}
, {76, -36, -22}
, {59, 50, -52}
, {12, -43, -77}
}
, {{-7, -49, -37}
, {34, -14, -54}
, {53, -4, 21}
, {87, -48, -12}
, {14, 56, -32}
, {15, 45, 24}
, {-31, -61, 32}
, {-55, -36, 20}
, {65, 53, 20}
, {-44, -43, 0}
, {-66, 24, -37}
, {38, 56, -66}
, {-62, 32, 36}
, {-23, 75, -37}
, {-45, 26, -55}
, {-33, 0, 49}
, {-41, 53, 5}
, {83, -22, 12}
, {37, 8, 61}
, {-26, -60, -44}
, {-32, 65, -46}
, {38, 34, 4}
, {73, 48, -21}
, {-28, -56, -2}
, {-55, 14, -10}
, {-24, 6, -50}
, {-40, -42, 66}
, {86, 57, 8}
, {30, -4, 22}
, {11, -60, 66}
, {-22, -8, -70}
, {-3, -22, -51}
}
, {{-7, -48, 39}
, {20, 39, 10}
, {-44, -23, -1}
, {-11, 4, 54}
, {-47, -3, 31}
, {-43, -36, 79}
, {56, -50, 45}
, {35, -32, -18}
, {37, 26, -83}
, {4, 19, 83}
, {-68, 32, 39}
, {61, -21, -37}
, {-34, 44, -72}
, {-47, -46, -58}
, {70, 47, 25}
, {75, 64, 58}
, {-26, 66, 43}
, {-15, 0, -28}
, {-74, 55, -46}
, {-8, 57, -55}
, {48, -31, 10}
, {35, 24, 75}
, {17, -42, 85}
, {54, -41, -38}
, {-68, -60, 52}
, {74, -26, 11}
, {60, 0, 8}
, {-35, 29, 85}
, {-28, -8, 78}
, {31, 47, 3}
, {-24, -62, -71}
, {58, -37, -66}
}
, {{-15, 55, 6}
, {-29, -4, 45}
, {5, 21, 44}
, {66, 0, -5}
, {-68, -63, 39}
, {3, -57, 62}
, {-30, -35, 5}
, {-66, -16, 43}
, {-51, -32, 43}
, {1, -52, -11}
, {-63, 52, -42}
, {51, 70, 41}
, {31, 60, 4}
, {-16, -70, -33}
, {-61, -21, 54}
, {-10, -58, 0}
, {-60, -10, -3}
, {4, 5, -42}
, {60, 62, -74}
, {-59, -32, -32}
, {-32, 2, 41}
, {78, 15, 6}
, {-12, 19, 59}
, {79, 40, -46}
, {24, 43, -46}
, {56, -35, 44}
, {65, 35, 41}
, {60, 4, -64}
, {-63, -72, 7}
, {11, -20, 18}
, {-48, 58, 62}
, {-36, -35, -2}
}
, {{38, 40, -15}
, {-31, -69, -69}
, {-3, -22, -55}
, {61, 60, -3}
, {65, 53, 59}
, {38, -62, 43}
, {-50, 53, -50}
, {-12, 41, 34}
, {50, -37, 37}
, {44, -17, 25}
, {37, 18, 11}
, {-67, -60, -23}
, {-54, 75, -57}
, {61, 2, 56}
, {-30, -39, -26}
, {-23, -33, -56}
, {-31, -30, -8}
, {70, 88, -25}
, {62, 19, 11}
, {49, 13, 42}
, {-31, -1, 39}
, {-28, 42, -9}
, {-25, -39, -49}
, {-66, 63, -12}
, {54, 10, 55}
, {5, 59, 76}
, {-1, -3, -64}
, {4, -43, 10}
, {-58, 26, -6}
, {0, -66, -19}
, {7, -16, -8}
, {-6, 15, 63}
}
, {{-16, -68, 22}
, {-8, 60, 49}
, {31, -32, 58}
, {-46, -59, 68}
, {6, 11, 67}
, {-9, -69, 15}
, {10, -39, -15}
, {39, 16, 69}
, {72, 9, 87}
, {-61, 52, 26}
, {-33, -6, -11}
, {78, -23, -71}
, {-21, 31, 25}
, {-47, -66, 26}
, {12, 21, -17}
, {-23, 73, 54}
, {-25, 52, -23}
, {-38, -60, 19}
, {51, -18, -16}
, {52, 23, 18}
, {-61, -36, -21}
, {49, 20, -13}
, {-35, 70, 0}
, {44, -24, 9}
, {-77, -59, 75}
, {54, 64, -47}
, {-13, -73, -69}
, {63, -56, -35}
, {-21, 38, -24}
, {50, -20, -29}
, {-48, -7, 0}
, {58, -17, 33}
}
, {{-45, -9, -57}
, {26, 41, -47}
, {14, 79, 62}
, {-27, -17, 61}
, {-45, 72, 65}
, {26, 58, -61}
, {-2, 53, -65}
, {47, 28, 60}
, {4, -6, -14}
, {-17, 75, -33}
, {-60, 58, 30}
, {-17, 30, 35}
, {30, 21, 65}
, {51, 15, -4}
, {36, -61, 20}
, {-82, -25, 48}
, {1, 72, 70}
, {-9, -55, -65}
, {-43, 45, 63}
, {25, -50, -15}
, {-65, -46, 37}
, {14, 34, -59}
, {-42, 49, -31}
, {-19, -37, 59}
, {-16, -16, -11}
, {34, -39, -55}
, {-68, -45, -36}
, {64, 15, -61}
, {-7, 24, -51}
, {-9, -52, 31}
, {22, -46, -44}
, {-79, 43, 33}
}
, {{56, 29, 24}
, {26, 55, -61}
, {-14, -38, -25}
, {-18, 26, 49}
, {47, -62, 51}
, {-8, -31, 45}
, {-32, 40, -66}
, {39, -64, 49}
, {6, -36, -51}
, {39, 37, 56}
, {-20, -15, 33}
, {-1, 32, 60}
, {10, -53, 54}
, {-52, -2, 25}
, {63, 79, 17}
, {70, 60, -29}
, {37, -69, -21}
, {-63, 36, -83}
, {-55, 69, 64}
, {-14, 0, 49}
, {55, -65, -54}
, {-21, 0, 40}
, {50, -35, -10}
, {-70, 46, -20}
, {50, 12, 66}
, {68, -63, -55}
, {24, -34, 11}
, {18, 50, 62}
, {8, 73, -62}
, {-24, 10, -69}
, {52, -51, 1}
, {68, -34, 43}
}
, {{8, -20, 68}
, {26, 31, -5}
, {-46, -15, -63}
, {-8, -66, 36}
, {12, 90, -23}
, {37, 44, -66}
, {-63, 32, -22}
, {-66, -59, 30}
, {-56, -78, 47}
, {-23, 8, 20}
, {48, -50, 66}
, {-65, 29, -12}
, {11, 48, -9}
, {46, 68, 0}
, {-22, -82, 53}
, {5, -28, -20}
, {-66, -67, 34}
, {16, 68, 5}
, {-66, -41, -26}
, {20, 63, -62}
, {-11, -32, -42}
, {26, -73, -19}
, {-59, 31, -42}
, {11, 53, 22}
, {24, -2, -6}
, {-2, 77, 33}
, {-43, 30, -10}
, {51, 27, 10}
, {34, -33, 60}
, {18, -48, -19}
, {53, -2, -9}
, {48, 62, -21}
}
, {{-45, 37, 9}
, {-72, 46, 19}
, {-29, 4, 67}
, {52, 38, 3}
, {-21, 77, 45}
, {40, 12, 14}
, {-38, -34, -34}
, {11, 37, -64}
, {-63, 35, -46}
, {15, 49, 44}
, {70, -28, -60}
, {-32, -50, 57}
, {-8, -67, 35}
, {-69, -64, 36}
, {18, 50, 15}
, {-35, -60, 14}
, {-53, 79, 63}
, {21, -46, -65}
, {71, 66, 30}
, {18, 43, 30}
, {-21, 3, 54}
, {71, 10, -19}
, {-78, 39, 73}
, {-20, -23, 49}
, {-59, 35, 30}
, {-4, 0, 61}
, {59, 17, -9}
, {-54, -51, -49}
, {-66, 62, 50}
, {-18, -53, -6}
, {-31, -3, -39}
, {32, 48, -65}
}
, {{-2, -38, 56}
, {-49, -40, -74}
, {-52, -60, 66}
, {49, -1, -46}
, {-12, 64, -4}
, {-43, 32, -31}
, {7, 44, -62}
, {-37, 20, 69}
, {-47, 27, -71}
, {-11, 9, -22}
, {0, -70, -81}
, {-42, 17, 47}
, {-61, -23, -47}
, {24, 35, 12}
, {-56, 49, -10}
, {-5, -16, -32}
, {-52, 38, 35}
, {69, -13, 17}
, {64, -35, 6}
, {-35, -11, 28}
, {-68, 37, 65}
, {10, 14, -27}
, {-25, -42, -59}
, {43, 64, -54}
, {0, 13, 6}
, {-35, -7, -46}
, {-61, 46, -37}
, {31, -11, -20}
, {34, -14, 57}
, {70, 66, -69}
, {49, 38, -62}
, {-44, -1, 52}
}
, {{-7, 11, 55}
, {30, -21, 9}
, {12, 34, 7}
, {-62, -54, 70}
, {19, -33, 1}
, {71, 24, 42}
, {-62, 74, -12}
, {-18, 26, 52}
, {18, 45, -8}
, {-46, 30, -30}
, {-47, -55, 4}
, {-18, 40, 58}
, {69, 60, 18}
, {-28, -68, -12}
, {46, -49, -62}
, {-51, -19, -29}
, {28, 49, 53}
, {3, -45, -64}
, {-41, 34, -17}
, {-13, -40, 25}
, {80, -41, 57}
, {53, 50, -17}
, {-28, 62, -40}
, {-14, -44, 0}
, {26, -76, -9}
, {-66, -9, -31}
, {-51, 57, 2}
, {-29, -35, -27}
, {53, -35, 53}
, {-7, 25, 31}
, {-61, -13, -65}
, {40, -19, -38}
}
, {{-33, -18, -41}
, {-49, -65, -71}
, {13, 66, 59}
, {30, 1, -20}
, {-4, -6, 11}
, {-26, -45, 45}
, {13, 9, -27}
, {33, 57, 51}
, {45, 46, 9}
, {-10, 69, 26}
, {-14, -12, -71}
, {-36, 31, -43}
, {-62, 40, 8}
, {55, 37, -58}
, {79, -13, -19}
, {43, 55, 43}
, {72, -27, 29}
, {-35, -5, 37}
, {22, -49, 70}
, {-35, 28, 20}
, {-65, 60, 47}
, {43, 59, 29}
, {-55, -1, 31}
, {23, -24, 45}
, {-35, -72, -79}
, {-11, -38, -36}
, {-23, 3, 8}
, {38, 21, -52}
, {-56, 76, 20}
, {12, 7, 29}
, {-9, 67, 16}
, {72, -24, 71}
}
, {{58, 43, 48}
, {63, 29, -23}
, {-70, 13, 16}
, {-19, -30, -23}
, {26, 0, 7}
, {36, 44, -76}
, {-38, 62, -36}
, {1, -43, -37}
, {-9, 23, -65}
, {-50, -17, 39}
, {-68, 67, -43}
, {14, 49, 63}
, {30, -56, 0}
, {20, 0, 34}
, {46, 12, 16}
, {44, 68, -67}
, {-45, 32, 28}
, {74, -70, 32}
, {44, -51, 30}
, {-41, -79, -33}
, {33, 29, 16}
, {-19, -60, 24}
, {56, -69, -38}
, {50, -4, -61}
, {51, -61, -9}
, {49, -55, -25}
, {-30, 26, -12}
, {68, 30, -47}
, {56, -66, -65}
, {-73, 49, 8}
, {33, -37, 7}
, {-32, 49, 5}
}
, {{20, -43, -32}
, {-47, -49, 56}
, {-71, 72, 70}
, {-9, -18, -65}
, {52, -14, -5}
, {-18, -7, 64}
, {18, -26, 59}
, {75, 50, -44}
, {-30, -43, -61}
, {66, 1, 74}
, {10, 60, 32}
, {52, -63, -25}
, {0, -15, 85}
, {-33, -25, -57}
, {74, 48, 59}
, {-18, -27, -32}
, {-4, -36, 43}
, {63, -49, -53}
, {-2, 66, -67}
, {25, -78, -22}
, {17, -46, -53}
, {77, -40, -20}
, {-61, 47, 74}
, {-28, -31, 56}
, {-45, -10, -9}
, {30, 26, -41}
, {-58, 59, 17}
, {5, -9, 12}
, {-73, 61, -68}
, {-45, 26, 8}
, {-60, 11, 11}
, {50, 46, -4}
}
, {{-45, 50, 18}
, {-82, -7, 26}
, {-50, 10, -12}
, {-18, -42, 2}
, {-64, -55, 9}
, {11, 7, -61}
, {-79, 48, 39}
, {42, -21, 55}
, {64, 33, -21}
, {43, -46, 9}
, {-4, -6, 23}
, {33, 65, -34}
, {-12, 84, -23}
, {-67, 55, -37}
, {20, -39, 55}
, {16, -17, 71}
, {75, 8, 20}
, {40, -3, 36}
, {50, -52, -33}
, {-68, 34, 55}
, {-24, 29, 66}
, {-59, 51, -6}
, {52, -7, 13}
, {23, -36, -62}
, {37, 86, 46}
, {-71, -60, -30}
, {37, -42, 32}
, {6, 3, 15}
, {-7, 5, 74}
, {-51, 39, 41}
, {-71, -40, 13}
, {65, 75, 42}
}
, {{31, 53, -37}
, {-19, -60, 0}
, {36, 45, 64}
, {0, -11, -3}
, {48, 22, -70}
, {53, 30, 48}
, {0, -41, 2}
, {-71, -39, -65}
, {-59, 14, 55}
, {-51, 65, -63}
, {25, -54, 62}
, {-56, -65, -18}
, {-48, 13, 23}
, {38, 47, 22}
, {41, 38, -76}
, {-16, -64, 22}
, {3, 65, 62}
, {12, 23, -4}
, {-12, -3, -24}
, {-18, -32, -43}
, {-60, -8, 44}
, {-41, 36, -4}
, {-48, -61, 40}
, {-73, 17, -42}
, {-61, -30, 8}
, {-27, 55, 1}
, {37, 10, 54}
, {-9, -38, -4}
, {66, -35, 15}
, {62, 4, 49}
, {-7, -18, -67}
, {33, 23, -51}
}
, {{52, 59, 39}
, {-24, 50, 12}
, {-4, 52, -3}
, {-16, 26, 101}
, {-57, -21, 62}
, {-12, 23, 29}
, {42, 65, 46}
, {63, -45, -20}
, {67, 66, -40}
, {31, 5, -55}
, {16, 67, -66}
, {-35, 10, -17}
, {36, 15, -35}
, {-29, -20, -69}
, {-56, 4, -36}
, {-8, 37, -50}
, {26, -65, 58}
, {-68, -18, 6}
, {-59, 44, -40}
, {25, 29, 68}
, {22, -59, -21}
, {-23, -41, 11}
, {-27, 53, 46}
, {-56, 38, 42}
, {49, 3, -46}
, {-17, -8, 1}
, {38, 18, 0}
, {-45, 80, -21}
, {-26, 31, -78}
, {14, 3, 7}
, {-69, 5, -12}
, {70, -59, 65}
}
, {{-27, -2, 35}
, {62, -68, 66}
, {-28, 34, 45}
, {-35, 46, -24}
, {3, 5, 60}
, {-33, -14, 37}
, {-31, -35, -63}
, {-8, -38, 59}
, {-41, 23, -7}
, {36, -30, -13}
, {-58, 20, 1}
, {-61, -9, 15}
, {18, -26, 20}
, {41, 20, -57}
, {-33, -12, 6}
, {-62, -60, 59}
, {26, 39, -61}
, {8, -40, 61}
, {52, -42, 48}
, {-1, 43, 34}
, {8, 28, -18}
, {72, -56, 53}
, {-13, 41, 28}
, {-40, 23, 57}
, {-74, -18, 21}
, {27, -56, -4}
, {-43, -3, -62}
, {-79, 56, 5}
, {-17, -3, 4}
, {39, -11, -34}
, {-4, -43, -14}
, {27, -7, 40}
}
, {{48, -13, -68}
, {37, 66, -33}
, {-17, 2, 28}
, {-66, -55, -30}
, {-61, 18, -62}
, {76, 69, -16}
, {25, -52, 22}
, {57, 52, 37}
, {34, 0, 30}
, {-26, -21, -44}
, {17, -19, -62}
, {35, 43, 43}
, {-11, -60, 47}
, {6, 12, 32}
, {6, -41, -27}
, {49, -46, 29}
, {20, -52, 26}
, {-65, 7, 10}
, {-34, -61, 54}
, {-59, 59, -19}
, {37, -46, 68}
, {43, 25, -40}
, {9, 41, -57}
, {-27, 34, 41}
, {-52, -26, 25}
, {64, -24, 16}
, {37, 12, -8}
, {-26, 43, -60}
, {-61, 66, 39}
, {-49, -70, 19}
, {-42, -65, -32}
, {65, -4, 41}
}
, {{6, -43, 50}
, {15, 0, -34}
, {-31, -28, -69}
, {-29, 71, 45}
, {53, -65, -9}
, {70, 51, 2}
, {-68, -50, -32}
, {65, -59, -29}
, {39, 0, 34}
, {49, 17, 23}
, {36, 53, 47}
, {-20, -8, 55}
, {66, -76, 69}
, {10, -8, -68}
, {-13, 18, 6}
, {68, 1, -48}
, {-12, -9, 26}
, {-61, 63, -54}
, {-44, 1, 68}
, {43, 54, 29}
, {60, -48, 21}
, {3, -66, -45}
, {63, 32, -29}
, {-5, -11, 13}
, {-46, 25, 24}
, {-15, -78, -24}
, {47, -68, -51}
, {-53, -79, -12}
, {60, 61, 19}
, {-40, -8, 21}
, {-11, 0, 63}
, {45, 10, -9}
}
, {{61, 14, 38}
, {-38, 40, -72}
, {-41, -53, -76}
, {-40, -26, -44}
, {45, -57, 38}
, {-42, 31, 51}
, {-16, -1, 54}
, {36, -21, 68}
, {71, -39, -86}
, {67, -29, -71}
, {-69, 72, -70}
, {-3, -35, 46}
, {53, 5, -76}
, {-29, -23, -71}
, {-47, -27, 48}
, {-38, 18, -62}
, {66, -19, 51}
, {30, 41, 40}
, {-52, -58, 26}
, {60, 46, 38}
, {-59, -44, 11}
, {16, 27, -17}
, {93, -86, 57}
, {-28, -36, -3}
, {78, 18, -51}
, {-66, -20, 51}
, {84, -56, 65}
, {82, -62, 0}
, {46, 37, 31}
, {-29, 17, 28}
, {7, -5, -45}
, {22, -71, 75}
}
, {{48, -72, 5}
, {15, -35, 28}
, {-36, -9, 50}
, {54, -27, 49}
, {-74, -45, 49}
, {-41, 44, -65}
, {-13, -60, -4}
, {0, -51, 68}
, {31, -66, -42}
, {12, 68, -30}
, {-74, -33, -31}
, {-36, 43, -39}
, {-44, 60, -16}
, {26, 59, -59}
, {28, -29, -27}
, {29, -77, 15}
, {5, -13, 62}
, {-49, 57, 34}
, {23, 7, 13}
, {50, -45, 49}
, {30, 1, 64}
, {-54, 30, -4}
, {-62, 68, 4}
, {70, 11, 66}
, {24, -10, 9}
, {-31, -69, 46}
, {39, 12, -20}
, {-60, 2, 9}
, {-31, -19, 59}
, {12, 33, 71}
, {-4, 19, 53}
, {-10, -63, -16}
}
, {{74, 79, 51}
, {60, -34, -1}
, {64, -19, -45}
, {-72, -74, -50}
, {32, 67, 14}
, {75, 6, -28}
, {14, -17, -26}
, {38, -47, 23}
, {-48, 70, -25}
, {-31, -70, -59}
, {51, 28, 18}
, {-57, 53, -14}
, {28, -50, 54}
, {48, 89, 80}
, {-70, 42, 1}
, {-34, -39, 49}
, {-58, 21, -77}
, {13, 36, -45}
, {59, 28, -25}
, {-51, 30, 36}
, {-62, 30, 49}
, {-83, -58, -32}
, {48, -34, -54}
, {-70, -28, 65}
, {-30, -45, 77}
, {-52, 15, -52}
, {-55, -9, -61}
, {25, -68, 1}
, {-31, -13, 28}
, {-34, -6, -44}
, {-14, 57, 77}
, {39, -8, -26}
}
, {{7, 4, -17}
, {39, -50, 91}
, {-24, 61, 38}
, {8, -36, 42}
, {-37, -22, 43}
, {-11, 56, -30}
, {-58, -63, 54}
, {83, -46, 17}
, {-15, -44, 71}
, {-2, 46, 43}
, {12, 59, 71}
, {78, -50, 28}
, {4, -39, 29}
, {28, -24, -6}
, {32, -49, -17}
, {56, 66, -5}
, {47, 48, 27}
, {-31, 26, 41}
, {80, 72, 2}
, {-93, -27, 7}
, {-75, -36, -70}
, {-17, -19, -78}
, {-61, 52, -19}
, {78, -66, -53}
, {74, 4, 82}
, {-28, 11, 73}
, {-26, -73, -50}
, {-28, 35, -73}
, {-53, 5, -41}
, {54, 45, 32}
, {-46, -47, -49}
, {77, -31, 26}
}
, {{-53, 72, 49}
, {-30, -67, 49}
, {-60, -61, 59}
, {86, -63, -3}
, {53, 61, 58}
, {-69, -32, -8}
, {-53, -29, -41}
, {34, -15, 13}
, {-15, 42, 49}
, {-57, 60, 1}
, {39, -32, 51}
, {32, -72, -81}
, {33, 14, 34}
, {-5, -14, 76}
, {-33, 41, 1}
, {-49, 74, 76}
, {-12, 43, 62}
, {49, -48, 20}
, {-53, -78, -13}
, {47, -74, 2}
, {-14, 70, -34}
, {-63, -45, 1}
, {78, -3, -53}
, {-58, -56, -45}
, {39, -21, -65}
, {37, 4, -64}
, {-48, 52, 6}
, {-48, -51, -7}
, {-31, -18, 8}
, {-19, 67, 35}
, {74, 16, -28}
, {-59, 42, -8}
}
, {{-50, 50, 25}
, {-55, 37, -29}
, {-11, -29, 17}
, {22, -37, 0}
, {-23, -43, 27}
, {55, 21, 75}
, {8, -70, 41}
, {47, -31, -61}
, {-69, -33, -52}
, {-35, 75, 63}
, {65, -14, 62}
, {37, 30, 1}
, {-25, -45, 46}
, {30, 35, 67}
, {57, -79, 14}
, {32, 20, -45}
, {-45, 20, -38}
, {79, 34, -20}
, {-76, 50, 40}
, {-20, -21, 10}
, {-68, -40, 34}
, {16, 37, 32}
, {36, 1, -53}
, {0, 12, 62}
, {12, -24, 35}
, {2, -27, 57}
, {42, -19, 10}
, {27, 60, 83}
, {8, 70, 55}
, {-60, 15, 39}
, {64, -2, -63}
, {-38, -61, 44}
}
, {{-17, -28, -69}
, {-30, 57, 62}
, {0, 28, -29}
, {87, -10, -39}
, {30, -43, 33}
, {13, 37, 28}
, {-44, 21, 3}
, {18, -60, -23}
, {52, 60, -74}
, {-2, 50, -18}
, {-72, 33, 59}
, {-24, -50, 54}
, {15, -21, -52}
, {45, 44, 1}
, {10, -11, -68}
, {58, -47, 21}
, {26, 75, -53}
, {67, 34, -22}
, {39, -71, 55}
, {31, 31, 29}
, {86, -42, 37}
, {62, -59, 38}
, {12, 28, -22}
, {-29, 62, 71}
, {-34, 49, -65}
, {-47, -66, -3}
, {-47, -17, -27}
, {77, 62, -47}
, {4, -11, -41}
, {-61, 36, 42}
, {-73, -56, -27}
, {-12, -86, 55}
}
, {{-1, -70, 12}
, {-55, -26, -60}
, {-68, -37, 37}
, {-76, -78, -9}
, {-26, -2, 22}
, {4, 64, 21}
, {-61, -23, 0}
, {79, 73, 59}
, {-23, -50, -33}
, {-36, -6, -3}
, {65, 68, -54}
, {8, 23, 70}
, {50, -67, -75}
, {-2, 7, -4}
, {43, -23, -18}
, {-38, 48, -25}
, {-58, -38, -5}
, {6, -75, 13}
, {51, 65, 66}
, {62, 64, 34}
, {9, 53, -34}
, {35, 57, -16}
, {-90, -25, 2}
, {29, -74, 25}
, {56, -62, 2}
, {-38, 63, -65}
, {3, -27, 39}
, {17, 19, 72}
, {-54, 35, -30}
, {10, -2, -5}
, {-48, -68, -71}
, {51, -48, 36}
}
, {{-43, -26, 47}
, {-63, 73, -53}
, {26, 17, -42}
, {-74, 65, 11}
, {-54, 41, 68}
, {-9, 64, -69}
, {-60, 27, -11}
, {-3, 15, 57}
, {-6, -28, 79}
, {-42, -33, -36}
, {-50, 6, 68}
, {15, -65, 28}
, {-58, -18, -10}
, {-41, 63, -57}
, {43, -23, 67}
, {-24, 64, -66}
, {24, 51, 14}
, {48, 51, 25}
, {13, -24, -26}
, {-46, -5, 34}
, {75, 13, 1}
, {-36, -32, -26}
, {13, -42, -70}
, {51, 42, 28}
, {-2, -53, -15}
, {-42, 61, 24}
, {24, 33, 27}
, {57, -36, -68}
, {46, 28, 59}
, {-5, 26, -26}
, {69, -24, 51}
, {12, -28, -17}
}
, {{-18, -38, 7}
, {64, -14, -53}
, {54, -58, 21}
, {44, -69, 70}
, {33, 6, 23}
, {-53, 42, -29}
, {-16, 68, 76}
, {58, -60, 0}
, {22, 20, 50}
, {-35, -3, -28}
, {48, 18, 63}
, {20, -2, 11}
, {-68, 5, 27}
, {-43, 73, 31}
, {62, 10, -60}
, {40, -21, 24}
, {-12, 60, -59}
, {72, 75, 33}
, {-13, 5, 31}
, {-84, 42, -31}
, {54, -43, 11}
, {40, 16, -1}
, {-25, 28, -2}
, {55, -17, 67}
, {-79, -13, 39}
, {58, -56, -45}
, {-21, 10, -53}
, {45, -41, -25}
, {-12, 24, -1}
, {7, 29, -56}
, {22, -66, -31}
, {64, -76, 19}
}
, {{18, 74, 42}
, {65, 67, -56}
, {37, 58, 57}
, {-27, 46, -75}
, {-67, 63, 49}
, {61, 71, -30}
, {59, 21, 67}
, {66, 3, -48}
, {56, 63, -2}
, {19, 39, -46}
, {-34, 40, 71}
, {72, -56, 51}
, {23, -41, 68}
, {35, 30, 15}
, {-73, 56, 55}
, {-11, -69, 6}
, {20, 10, -63}
, {-7, 42, -27}
, {9, 28, -10}
, {-77, -26, -59}
, {-11, -34, -6}
, {66, -62, 36}
, {52, -3, -49}
, {-48, -12, -14}
, {0, -54, 34}
, {-13, -13, -22}
, {15, -16, 30}
, {17, 50, 5}
, {8, -2, -30}
, {64, 54, 61}
, {36, -17, 57}
, {-68, 1, 81}
}
, {{-52, 59, -58}
, {28, -31, -80}
, {-57, -6, -14}
, {-58, 17, 63}
, {-69, -4, -6}
, {-7, 70, 50}
, {-69, -64, 16}
, {31, -43, 55}
, {41, -74, 49}
, {-45, 47, 78}
, {13, 23, 62}
, {-1, -68, 34}
, {-4, -43, -11}
, {33, 3, -30}
, {-20, 8, 18}
, {-25, -64, 63}
, {-49, 14, 0}
, {47, 33, 17}
, {13, -73, 44}
, {38, 14, 20}
, {-4, 70, -23}
, {-54, -23, 68}
, {-58, 49, 4}
, {38, 38, -30}
, {0, 43, 13}
, {-23, -21, -43}
, {79, 7, -39}
, {-23, 11, 60}
, {33, 6, 7}
, {-53, -18, -32}
, {3, 22, 56}
, {-32, 21, -25}
}
, {{50, 10, -58}
, {64, -28, 19}
, {39, 32, -26}
, {-31, 69, 39}
, {28, -63, -9}
, {-30, 11, -15}
, {-51, -29, -5}
, {-5, -12, 57}
, {18, 27, 48}
, {57, -37, -57}
, {-50, 71, -34}
, {-72, -9, 10}
, {39, -17, 0}
, {28, -38, -68}
, {18, 15, 74}
, {7, -68, -64}
, {77, 45, -31}
, {25, -43, 33}
, {15, 78, 49}
, {-44, -64, 25}
, {-76, 47, 13}
, {-73, -13, -25}
, {-48, 4, 64}
, {39, -6, 0}
, {15, 66, -11}
, {-59, -40, -52}
, {-15, -63, -22}
, {62, 30, -41}
, {48, 25, 5}
, {-77, 58, 34}
, {-7, 68, 52}
, {2, -75, -16}
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
#define INPUT_SAMPLES   61
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_148_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_148(
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
#define INPUT_SAMPLES       30
#define CONV_FILTERS        128
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_119_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_119(
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


const int16_t conv1d_119_bias[CONV_FILTERS] = {-2, -10, 14, 0, 9, -11, -5, 20, 3, 4, 10, 6, -16, 0, 6, 12, -4, -15, 0, -5, 19, -16, 0, 8, 4, 14, -8, 2, -10, 1, 15, 12, 0, 0, 1, 6, 1, -9, 22, 2, 6, -6, -13, 7, 3, 12, 6, -12, 17, 7, -14, 4, 0, 12, -5, -8, 22, -1, 1, -5, 5, 9, -10, 4, -1, 16, 0, 19, -7, 3, 21, -6, -2, -15, 0, 6, 6, -8, 9, 9, 2, -3, -16, -9, -6, -7, 5, 22, -2, -5, 10, -4, 22, 5, 13, 4, 8, -15, 17, -3, 0, 17, 3, -6, 14, -8, 12, -4, 2, -5, 4, -3, 6, 2, -4, -3, 7, -2, 11, 15, -3, 0, -4, 4, -3, 2, -4, -8}
;

const int16_t conv1d_119_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-32, -2, 5}
, {-40, 52, -37}
, {-21, 59, 36}
, {-41, -11, -17}
, {57, 41, 26}
, {24, 43, -46}
, {-5, -43, -51}
, {-38, -30, 21}
, {-34, 25, -6}
, {0, -30, 44}
, {-53, -34, -30}
, {-26, 29, 44}
, {-27, -26, 29}
, {-37, 33, -5}
, {-53, 47, -27}
, {15, -42, 29}
, {41, 49, 30}
, {-48, 29, 18}
, {-33, 27, -8}
, {-24, 6, 50}
, {-21, -24, -23}
, {32, -31, -24}
, {39, -20, 40}
, {15, 41, -21}
, {59, 5, -30}
, {33, 5, -55}
, {-29, -18, -2}
, {0, -17, -19}
, {-7, -29, -45}
, {35, 8, 40}
, {9, 10, -46}
, {-6, 31, -49}
, {-41, -38, 43}
, {-58, -50, 42}
, {-32, 61, 12}
, {0, -3, 50}
, {9, 14, -46}
, {25, -37, -3}
, {45, 57, 15}
, {-22, 48, 46}
, {-9, -34, 59}
, {19, -41, -37}
, {21, -3, -18}
, {4, -47, 3}
, {18, -33, 50}
, {-32, -28, -32}
, {30, 38, 32}
, {24, -39, -39}
, {-10, 15, 37}
, {-60, -51, -42}
, {39, 29, -46}
, {-60, -13, -31}
, {49, -37, 56}
, {41, 10, -2}
, {-5, 21, 30}
, {-2, -31, -31}
, {-21, -3, -44}
, {-43, 32, -6}
, {-6, 14, 23}
, {43, -11, 51}
, {0, 16, 9}
, {54, -20, 5}
, {27, -35, 27}
, {-2, -14, -38}
}
, {{13, 53, 22}
, {19, -16, 16}
, {-23, -36, -61}
, {46, -38, 21}
, {-5, -47, -41}
, {23, -20, 30}
, {29, -34, 13}
, {-38, -2, 13}
, {23, -26, -32}
, {-30, -7, 10}
, {5, -48, 20}
, {5, 48, -33}
, {-47, 26, -11}
, {-17, 41, 2}
, {-48, 8, -24}
, {30, -30, 19}
, {-45, -47, 16}
, {-44, -33, -41}
, {37, 31, 37}
, {44, -24, 47}
, {8, 27, 47}
, {12, -17, 22}
, {-9, -2, 53}
, {-25, -27, 39}
, {29, -9, -8}
, {-2, 21, 20}
, {-46, -48, -18}
, {28, 20, -27}
, {37, 28, -31}
, {-47, -36, 37}
, {22, -24, 47}
, {3, 25, -30}
, {-28, -42, 12}
, {-37, 11, 25}
, {11, 5, 12}
, {-46, 63, -42}
, {-10, 42, 18}
, {53, -19, 20}
, {-10, -5, -29}
, {48, 6, 51}
, {-41, -31, 15}
, {-48, -48, 1}
, {-30, 54, -20}
, {31, 21, 17}
, {-16, -40, -22}
, {-41, 41, 16}
, {-33, -7, -50}
, {20, -45, -46}
, {-37, -2, -10}
, {32, -64, -45}
, {33, 55, 50}
, {-23, -60, -8}
, {-32, 40, 12}
, {-24, -22, -47}
, {-10, 36, 37}
, {28, -29, -25}
, {-48, 37, -20}
, {37, -48, 37}
, {17, 37, -3}
, {-39, 30, 15}
, {19, 33, -27}
, {-29, 13, 10}
, {13, 21, -5}
, {0, 23, -35}
}
, {{-7, 27, 20}
, {-41, 14, 6}
, {23, 7, -48}
, {-41, -62, 17}
, {-38, -29, -50}
, {-10, 40, 13}
, {26, 53, -40}
, {41, 28, 14}
, {49, -27, -32}
, {41, 53, 1}
, {33, 34, 51}
, {-18, 47, 16}
, {-4, -17, -52}
, {14, 10, -6}
, {-43, 48, 11}
, {-29, -11, -23}
, {-14, 25, 48}
, {-19, 15, -60}
, {27, 45, -14}
, {0, -29, 5}
, {-35, -44, 49}
, {41, 25, -8}
, {3, 38, -24}
, {41, 44, -7}
, {9, 22, 60}
, {-57, -2, -10}
, {-16, 13, 6}
, {-9, -10, 20}
, {34, -23, -11}
, {-37, 2, 29}
, {9, 20, 6}
, {31, -20, -19}
, {-4, 28, 25}
, {1, 12, -39}
, {56, -44, -20}
, {-42, -23, -62}
, {-47, -44, -3}
, {9, -32, 12}
, {61, -36, 6}
, {-65, 18, -50}
, {42, -8, -28}
, {-5, 16, -21}
, {-3, -37, 8}
, {11, 1, -41}
, {-28, -25, -8}
, {30, 0, 39}
, {9, 48, -15}
, {-20, 35, 44}
, {49, -21, 52}
, {29, 24, 22}
, {-55, -50, -42}
, {59, 2, -7}
, {33, 43, -28}
, {-9, -45, 22}
, {44, -47, 37}
, {0, 10, -23}
, {13, -23, -35}
, {13, 63, 60}
, {-27, 7, 25}
, {-21, 21, -30}
, {0, 58, 50}
, {17, 26, 32}
, {9, -29, -37}
, {-11, 9, -4}
}
, {{0, -6, 0}
, {-36, 25, 46}
, {17, -47, 3}
, {-1, 9, 40}
, {41, -41, -2}
, {28, 40, -8}
, {28, -6, 10}
, {29, -30, 19}
, {-26, 46, -18}
, {0, 8, -22}
, {-18, 52, 49}
, {6, 13, 46}
, {42, -21, 38}
, {0, 30, 46}
, {-41, -48, 8}
, {24, 14, -42}
, {-46, 18, -23}
, {27, -25, 10}
, {-3, 51, 0}
, {-51, 46, 47}
, {25, -6, -35}
, {-3, 20, -38}
, {23, 28, 25}
, {14, 67, 41}
, {24, 55, 52}
, {-61, -10, -31}
, {18, 63, 32}
, {-37, 33, 9}
, {37, 17, 6}
, {42, -38, 31}
, {-53, -4, -20}
, {28, 19, -2}
, {52, 0, -3}
, {24, 16, 28}
, {31, 57, 53}
, {-1, -41, -35}
, {3, -16, -53}
, {42, 28, -45}
, {-45, -32, -7}
, {0, -15, -18}
, {-56, -49, 22}
, {39, -1, 19}
, {12, 23, 9}
, {21, 21, -28}
, {48, 21, 57}
, {41, -33, -35}
, {-24, 24, 22}
, {-34, -32, 3}
, {1, 31, 48}
, {-7, 33, 32}
, {45, -36, 31}
, {27, -11, 42}
, {-33, -43, 39}
, {46, -31, 6}
, {-31, -2, 15}
, {15, -26, 0}
, {33, -2, 51}
, {3, -9, -2}
, {-40, -36, 50}
, {5, 24, 37}
, {-6, -5, 4}
, {-25, -42, -17}
, {51, 36, -10}
, {-47, -12, -15}
}
, {{28, -11, 35}
, {-24, -3, 32}
, {33, 49, -18}
, {-42, -56, -30}
, {2, -38, 19}
, {-3, 5, -40}
, {13, 29, 13}
, {-10, 32, -28}
, {36, -35, -22}
, {-14, 12, 31}
, {-61, 15, -8}
, {-9, 0, 26}
, {-28, 13, -49}
, {-41, 26, -49}
, {25, 36, -16}
, {57, -30, -14}
, {5, 29, 48}
, {29, -16, 1}
, {-10, -17, -42}
, {-41, 39, 41}
, {-22, -41, 33}
, {-30, 17, 30}
, {-43, -23, 50}
, {0, -42, -36}
, {-2, 14, 7}
, {54, 19, -28}
, {-8, 12, -13}
, {28, 39, -42}
, {29, -10, 53}
, {41, -18, 7}
, {39, 25, 5}
, {-14, -16, -13}
, {-33, 43, 33}
, {-9, 8, -47}
, {24, 1, 17}
, {33, -5, -31}
, {-39, -41, 46}
, {48, 18, -45}
, {-30, 20, -25}
, {-32, 40, 24}
, {-40, -5, -21}
, {-33, -56, -28}
, {18, -18, -4}
, {19, 12, 29}
, {-20, 8, 26}
, {-30, -15, 60}
, {-27, 37, -17}
, {-4, 12, 4}
, {-22, 26, -25}
, {33, -18, 33}
, {-37, 29, 44}
, {9, 43, 29}
, {-35, 45, 11}
, {-7, -17, 13}
, {44, 39, -3}
, {14, -41, -16}
, {-31, -35, 46}
, {-21, 51, 62}
, {30, -37, -36}
, {-22, -38, 43}
, {45, -7, 54}
, {-9, 41, -7}
, {-11, -37, -54}
, {30, 23, -24}
}
, {{1, 40, -7}
, {40, -53, 17}
, {-46, 4, 21}
, {56, 19, 36}
, {-35, -38, -33}
, {38, 42, 36}
, {31, 19, 36}
, {32, -35, 43}
, {46, -46, -8}
, {-24, 1, -27}
, {11, 16, 3}
, {-40, 30, 0}
, {-40, -13, 38}
, {-21, 27, 44}
, {9, -2, -45}
, {-19, -44, -6}
, {-41, -25, 43}
, {-53, -12, 28}
, {-10, -32, 38}
, {-9, 57, -21}
, {0, -37, -36}
, {58, -29, 9}
, {-39, 17, 34}
, {44, -41, 0}
, {40, 55, 42}
, {-4, 37, -28}
, {37, -49, 24}
, {10, -16, 33}
, {39, -14, 7}
, {33, -7, -22}
, {-7, 15, 17}
, {-12, -11, -3}
, {67, 43, -20}
, {-21, 38, -14}
, {24, 32, 33}
, {36, -32, 13}
, {16, 48, 37}
, {-13, 12, 40}
, {-18, -13, 19}
, {2, -69, -9}
, {-28, -62, -7}
, {36, 9, 0}
, {24, -21, 40}
, {-63, -5, 13}
, {18, 13, 27}
, {42, -26, -38}
, {-13, -6, 31}
, {-54, 39, -38}
, {-14, -57, 22}
, {-65, 27, -44}
, {5, -31, -28}
, {15, 14, -57}
, {-25, -18, 56}
, {-1, 45, 34}
, {-45, 49, -15}
, {15, -17, 10}
, {-47, 40, -38}
, {-40, -38, -35}
, {-16, 17, 13}
, {49, 9, -2}
, {11, -41, -29}
, {-47, 36, 46}
, {38, -40, 58}
, {35, 6, -58}
}
, {{-28, -1, 48}
, {-42, 28, -27}
, {-34, 33, -61}
, {15, -21, -18}
, {-41, 2, -48}
, {43, -43, 33}
, {0, 21, 47}
, {41, 19, -33}
, {-36, -11, -18}
, {-40, 38, -31}
, {-18, 7, -31}
, {-42, -10, 3}
, {19, 47, 28}
, {-13, -19, 5}
, {-17, 3, 44}
, {-45, 21, -42}
, {5, -40, 45}
, {-34, 41, 20}
, {19, -38, 28}
, {-9, -51, 14}
, {34, 36, -43}
, {-38, 0, -10}
, {32, -19, -50}
, {-37, -32, 36}
, {6, -43, -46}
, {-50, 4, -38}
, {11, 25, 33}
, {-48, 11, 34}
, {-14, -18, -7}
, {28, 36, 18}
, {-21, 31, 2}
, {23, -13, 20}
, {-22, -43, -6}
, {35, 13, 47}
, {20, 25, 44}
, {-18, 36, 25}
, {-30, -24, -1}
, {0, 0, 9}
, {46, -47, 8}
, {53, 0, 8}
, {19, 42, 47}
, {-31, -36, 24}
, {13, 42, -12}
, {-44, 9, 25}
, {-32, -18, 28}
, {-50, -22, -43}
, {-1, 37, -11}
, {3, -57, 27}
, {-50, -1, 22}
, {-38, 14, -39}
, {15, -43, 4}
, {-30, -25, -1}
, {-4, 18, -18}
, {30, 10, -16}
, {-40, -24, 29}
, {-15, 0, -46}
, {-18, -16, -22}
, {-56, 43, 0}
, {-31, -5, 34}
, {36, 34, 47}
, {-40, -51, -31}
, {-44, 3, -25}
, {-25, -7, 4}
, {-4, -17, 7}
}
, {{46, 5, -7}
, {0, 36, 30}
, {-18, -9, -36}
, {24, -17, -29}
, {-30, 44, -38}
, {18, 9, -2}
, {-17, 45, -27}
, {4, -8, -8}
, {9, -51, 35}
, {-21, 10, 57}
, {48, 50, 19}
, {16, -36, -5}
, {-4, -48, 42}
, {-61, 16, 33}
, {-2, -49, -31}
, {-14, 4, 22}
, {-31, -2, 49}
, {-11, 48, -10}
, {0, 1, 57}
, {-30, 45, 2}
, {-1, 36, 38}
, {0, 22, 0}
, {-44, 22, 27}
, {-2, 27, 16}
, {11, 51, 39}
, {5, -28, -32}
, {14, 0, 49}
, {6, 44, 16}
, {-33, -40, -34}
, {-2, -20, 35}
, {-50, -49, -27}
, {-43, 9, -20}
, {-27, 47, -49}
, {-36, 39, 16}
, {52, 15, 20}
, {18, -6, -54}
, {54, 28, -29}
, {-22, -19, 23}
, {-18, -1, 64}
, {4, -33, -43}
, {30, -20, -45}
, {-2, 3, -55}
, {-51, -41, -35}
, {18, 18, 20}
, {-24, -24, 18}
, {-49, -36, -3}
, {50, -38, -15}
, {49, 35, 20}
, {-18, 62, 69}
, {1, 33, 9}
, {-44, -25, -57}
, {55, 48, -17}
, {9, 43, -21}
, {51, 25, -22}
, {-21, 27, 42}
, {10, 14, 8}
, {-34, -23, 15}
, {-29, -3, 44}
, {-23, 23, -22}
, {49, 44, 2}
, {20, 33, 18}
, {-38, 14, -22}
, {29, -2, -1}
, {12, -43, -27}
}
, {{7, -32, 18}
, {-43, 21, -3}
, {-36, -20, 46}
, {18, 51, 43}
, {48, 3, 19}
, {-27, 33, 50}
, {2, -17, 9}
, {53, -13, 8}
, {-24, -8, 8}
, {-24, -9, 0}
, {38, -25, 37}
, {-28, 47, -2}
, {0, 22, 23}
, {25, 28, -42}
, {20, 8, -17}
, {47, -45, 6}
, {-5, 23, -26}
, {-2, -38, 42}
, {-42, 41, 7}
, {-41, -32, -46}
, {-37, 33, -29}
, {-15, -22, 43}
, {28, 44, 48}
, {35, -11, 35}
, {18, 15, 2}
, {22, 17, -19}
, {34, 78, 32}
, {38, 39, 50}
, {36, -22, -46}
, {8, -42, 42}
, {19, -17, -20}
, {5, -49, 65}
, {31, -6, 53}
, {64, 51, 14}
, {-21, 37, -8}
, {-43, -11, 3}
, {-48, -19, -12}
, {47, -29, -21}
, {-11, -20, 50}
, {49, -35, 37}
, {0, 20, 21}
, {41, -30, 27}
, {29, 38, -29}
, {19, -12, -29}
, {-29, 11, -41}
, {-42, -30, -2}
, {48, 31, 41}
, {37, -32, 46}
, {-33, -3, -31}
, {41, 36, -34}
, {-36, -32, -34}
, {-38, -49, -5}
, {22, -9, -49}
, {51, 20, 16}
, {14, -26, 8}
, {37, 32, 2}
, {13, -21, -9}
, {2, -41, 16}
, {33, 23, 8}
, {-19, -21, -15}
, {-20, -47, -45}
, {-15, 12, -7}
, {45, 53, 4}
, {-8, -20, -29}
}
, {{11, -18, -14}
, {47, 22, 34}
, {19, -7, 57}
, {-6, 43, -28}
, {-39, 14, -40}
, {-9, 34, -56}
, {17, -37, -57}
, {-50, 17, 42}
, {40, -33, 41}
, {4, -39, -19}
, {-26, -43, -9}
, {-26, -24, -7}
, {-18, -5, -29}
, {39, 65, -4}
, {-16, 21, 12}
, {25, 29, -33}
, {43, 1, -36}
, {-13, -23, -4}
, {-10, -35, 45}
, {27, 0, 8}
, {25, 14, -37}
, {-10, -38, -7}
, {-34, 28, -19}
, {-11, -35, -12}
, {-30, 43, -43}
, {21, 26, -10}
, {53, 5, -51}
, {17, 17, -51}
, {28, 33, -33}
, {-35, 16, 20}
, {25, 61, 0}
, {-15, -9, -31}
, {-21, 24, -4}
, {0, -57, 33}
, {27, 22, 38}
, {36, 20, 28}
, {45, 47, 4}
, {51, -32, -9}
, {50, 58, 0}
, {-15, 5, 33}
, {-12, -32, -22}
, {33, -28, -55}
, {-53, 24, -5}
, {-14, 15, -45}
, {-39, -40, -33}
, {11, -37, 11}
, {0, -2, 31}
, {16, 43, 7}
, {40, 62, 16}
, {16, -27, -28}
, {19, 53, 37}
, {-26, 17, -35}
, {43, -1, 47}
, {45, 1, -3}
, {-26, -8, 20}
, {42, -15, -18}
, {-20, -25, -17}
, {-50, -35, 30}
, {37, 40, -19}
, {-8, -37, -1}
, {-23, 31, -31}
, {-7, 52, 0}
, {-27, 35, -49}
, {42, -27, -12}
}
, {{-28, 61, -31}
, {-26, 24, 24}
, {15, -18, 38}
, {-29, -13, 2}
, {-20, -33, 5}
, {22, -43, 0}
, {-5, -24, -37}
, {-41, -18, 56}
, {-1, -15, -61}
, {1, -29, -16}
, {37, -9, 36}
, {-30, -37, 43}
, {53, -7, -18}
, {-8, 3, 3}
, {21, -37, 41}
, {-7, 42, 29}
, {-3, 55, -18}
, {14, -50, 24}
, {-3, 59, 35}
, {36, -18, 1}
, {-32, -7, -36}
, {34, 34, -31}
, {-47, -52, -22}
, {23, 64, 26}
, {37, -10, 27}
, {-26, 45, 46}
, {-51, -15, 30}
, {-6, -12, 41}
, {2, 25, -23}
, {-34, -27, 2}
, {8, 17, -37}
, {15, 44, 25}
, {29, 8, 43}
, {2, 19, -30}
, {-32, 32, 60}
, {17, -75, -47}
, {1, -43, -22}
, {-38, 7, 43}
, {-6, 37, -1}
, {27, -47, -19}
, {-14, 7, -34}
, {45, 40, 67}
, {-35, 21, 22}
, {-27, 46, -32}
, {-31, -4, -18}
, {50, 0, -5}
, {35, -12, 31}
, {2, 1, 38}
, {2, 16, 37}
, {14, -27, 38}
, {15, -48, 14}
, {19, 21, 34}
, {-27, -31, -33}
, {22, -36, -17}
, {-8, -1, -55}
, {-33, -17, 27}
, {38, 33, 19}
, {70, 47, 26}
, {-73, 13, 25}
, {-22, -40, -21}
, {42, 8, 32}
, {-46, 19, 0}
, {-50, 12, 36}
, {30, -32, 28}
}
, {{11, 56, -29}
, {-22, -25, -33}
, {-18, 46, 43}
, {-5, 11, -52}
, {-39, -19, -50}
, {2, 43, 11}
, {23, 31, -9}
, {-33, 49, -17}
, {11, 27, 23}
, {43, -20, 48}
, {-16, -14, 40}
, {17, -30, -43}
, {9, -43, 46}
, {3, -13, -53}
, {-37, -4, 2}
, {38, 25, -11}
, {53, 51, 13}
, {5, 24, 37}
, {44, -37, 15}
, {35, -22, 27}
, {-34, -22, -22}
, {-17, 11, -40}
, {-41, 3, 21}
, {-48, -16, -8}
, {-17, 7, -9}
, {14, -43, 4}
, {-33, -33, 6}
, {-29, -6, 42}
, {40, -10, -35}
, {-5, -33, -25}
, {-13, -36, 4}
, {-35, 20, -3}
, {39, 44, 39}
, {0, -6, 24}
, {23, 69, 33}
, {-28, 12, 39}
, {-50, 15, -12}
, {8, 38, -15}
, {-34, -24, -37}
, {48, 17, -40}
, {-45, -45, -42}
, {11, 13, -32}
, {5, -24, 7}
, {-46, 55, -1}
, {-3, 45, 12}
, {46, -27, -52}
, {21, 9, -39}
, {15, 21, -26}
, {-16, 10, -47}
, {13, 28, -42}
, {21, 20, 0}
, {-20, 23, -44}
, {35, -38, -24}
, {34, -25, 34}
, {40, 0, -8}
, {-23, 26, -21}
, {-40, -38, -19}
, {19, -37, 8}
, {6, -17, 27}
, {22, -25, -46}
, {36, 13, 53}
, {26, -41, 33}
, {38, -8, 48}
, {43, 0, -3}
}
, {{15, -11, 4}
, {33, 23, -17}
, {-60, 31, 33}
, {-52, 10, 8}
, {-26, 18, 0}
, {-1, 4, 29}
, {13, -4, 16}
, {-46, 37, 12}
, {-28, -59, 1}
, {21, -44, -18}
, {-33, -16, 41}
, {16, 53, -5}
, {-50, -34, -15}
, {35, -22, 29}
, {2, -59, -22}
, {14, -16, 47}
, {-6, 9, -28}
, {29, 51, -58}
, {17, 14, 13}
, {-47, -43, 46}
, {6, -35, 30}
, {0, 25, 14}
, {21, -49, 13}
, {47, 1, 13}
, {48, -35, 28}
, {-46, 11, 13}
, {-2, 10, -41}
, {0, -35, -15}
, {-33, -10, 15}
, {-28, 43, -49}
, {1, 16, 22}
, {0, -16, -5}
, {41, -30, -17}
, {-57, -3, 42}
, {28, 47, 48}
, {-13, 2, 40}
, {7, 42, 29}
, {48, 48, -5}
, {-25, 26, 22}
, {25, 22, -40}
, {-39, -12, 19}
, {38, 39, -61}
, {11, 5, -18}
, {14, -52, 36}
, {33, 15, -51}
, {4, 40, 18}
, {-50, 40, 38}
, {-3, -32, -45}
, {-38, -31, 26}
, {4, -30, 39}
, {39, -45, 16}
, {44, 0, 42}
, {-13, -4, 33}
, {46, 2, -5}
, {-15, -12, 4}
, {-31, 41, 17}
, {22, 41, -42}
, {-42, -11, 1}
, {27, -60, -2}
, {19, 14, -12}
, {33, 35, 45}
, {14, -11, -17}
, {38, -14, -30}
, {-50, 26, 0}
}
, {{53, -39, 41}
, {32, -43, 30}
, {-8, -1, -46}
, {-16, 0, -3}
, {-39, -55, -3}
, {14, -10, 3}
, {40, -23, 49}
, {52, -25, 11}
, {35, 47, 34}
, {26, -14, -33}
, {2, -15, -14}
, {25, 23, 5}
, {-46, 14, 34}
, {12, 30, 33}
, {-7, 15, 9}
, {-41, 21, 24}
, {43, 33, -40}
, {-31, -13, 14}
, {27, -44, 28}
, {46, -17, 45}
, {26, 17, -15}
, {55, 3, 3}
, {-19, -8, 30}
, {3, -64, 4}
, {-25, -31, 28}
, {0, -41, 56}
, {-62, -32, 16}
, {-23, 22, -41}
, {17, 23, 2}
, {-33, 35, -11}
, {24, 48, 1}
, {39, -34, -22}
, {-46, -17, -6}
, {32, -52, 41}
, {-24, -18, -57}
, {45, 29, 17}
, {-13, -9, 42}
, {43, 55, -42}
, {37, -29, 29}
, {-22, 0, 37}
, {13, 7, -46}
, {33, 22, 16}
, {-47, -24, -8}
, {9, 0, 6}
, {-26, -42, -44}
, {45, 32, -4}
, {-6, 17, 1}
, {25, -5, 41}
, {-10, -52, 7}
, {18, -31, -15}
, {-17, 40, -24}
, {-2, 22, 1}
, {28, 51, -14}
, {-37, -20, -52}
, {-3, -1, -50}
, {-40, -25, -26}
, {46, 49, 32}
, {57, 48, -43}
, {-17, -22, -4}
, {-48, 7, 11}
, {4, -34, -1}
, {45, -14, -8}
, {20, 26, -35}
, {8, 40, 7}
}
, {{19, 14, -56}
, {24, -15, -12}
, {34, 5, 76}
, {37, 17, 47}
, {29, 22, -44}
, {59, 0, 39}
, {5, -44, 53}
, {9, -33, -40}
, {-8, -3, -48}
, {-49, -14, 15}
, {4, 12, 56}
, {4, 19, -7}
, {-35, 8, 49}
, {19, 7, 25}
, {36, 42, 27}
, {43, 35, 6}
, {33, 2, 31}
, {-7, 1, -35}
, {2, 34, -26}
, {5, -51, 21}
, {-38, 24, -17}
, {-6, 49, 15}
, {6, -49, 2}
, {37, -15, -45}
, {52, -4, 33}
, {-12, 9, 53}
, {-1, 9, -30}
, {-44, -48, -45}
, {28, 41, -40}
, {30, -40, -32}
, {1, 5, 30}
, {-28, -12, 29}
, {36, 12, -32}
, {-39, -9, -39}
, {2, -5, 2}
, {-11, 3, 42}
, {-23, -16, 28}
, {-32, -17, 49}
, {-6, 25, -44}
, {-24, -34, 22}
, {-16, 21, -34}
, {2, 24, -1}
, {16, -37, 29}
, {13, 24, 36}
, {-48, -41, 50}
, {38, 0, 39}
, {6, 32, 61}
, {-3, 60, 55}
, {-16, 46, 32}
, {-29, 10, 36}
, {46, -23, -38}
, {29, 43, -15}
, {0, -1, 18}
, {-42, -49, -8}
, {30, -56, 48}
, {-29, 30, 54}
, {43, 1, -20}
, {0, 49, 56}
, {15, 74, 20}
, {24, 30, -26}
, {-19, -12, -50}
, {48, -50, 13}
, {-16, 0, 9}
, {-1, 0, -6}
}
, {{31, 7, 13}
, {-44, 31, -49}
, {51, -16, 48}
, {-21, -1, 9}
, {-16, -29, 49}
, {34, 40, -26}
, {-34, 12, 9}
, {23, -4, -20}
, {-9, 33, 35}
, {-51, -21, -3}
, {-41, -55, -36}
, {-52, -5, -9}
, {-28, 46, 36}
, {-40, -11, -3}
, {11, 35, 0}
, {2, -14, 10}
, {40, 26, 1}
, {12, 8, 8}
, {-17, 13, -6}
, {13, 6, 12}
, {-41, 5, -9}
, {18, 52, 25}
, {7, 48, 19}
, {3, 1, -27}
, {11, -5, 31}
, {29, 48, -38}
, {42, -30, -27}
, {-34, -26, -54}
, {-9, -34, 37}
, {-6, 18, -47}
, {8, -15, 40}
, {60, -33, 12}
, {12, -32, 21}
, {-17, 1, -23}
, {11, 20, -26}
, {28, 31, 56}
, {7, -32, 2}
, {-20, 27, -46}
, {-35, -52, -31}
, {-10, 45, 61}
, {-16, -17, -6}
, {3, -7, -27}
, {46, 11, -40}
, {-23, -43, -30}
, {-23, 13, 51}
, {18, -3, -45}
, {28, -39, -17}
, {1, 41, 21}
, {-5, 35, -1}
, {-3, 20, -55}
, {-32, -45, 21}
, {-28, -13, -19}
, {-19, 47, 13}
, {24, -43, 0}
, {-3, 41, -23}
, {3, -32, 4}
, {40, -43, 14}
, {31, 30, 56}
, {17, 30, -38}
, {51, -31, 1}
, {41, -8, 33}
, {-8, 38, 43}
, {-5, 14, -3}
, {-34, 40, -17}
}
, {{39, -34, -15}
, {-30, -8, -11}
, {20, -41, -6}
, {-55, -20, -6}
, {-52, -24, 49}
, {18, 21, 9}
, {21, 13, -20}
, {-54, 35, 53}
, {0, 23, 52}
, {2, 38, 23}
, {0, -24, -24}
, {-18, 0, -47}
, {20, 28, 19}
, {-3, 61, 25}
, {-39, -34, 51}
, {-42, -39, 21}
, {26, 28, -15}
, {6, -44, 41}
, {-66, -6, 16}
, {44, -44, -11}
, {22, 40, -2}
, {44, -35, 14}
, {-25, -50, -32}
, {39, -17, 43}
, {10, -25, -1}
, {23, 14, 44}
, {35, 32, -32}
, {-28, -29, 38}
, {-27, 32, 0}
, {-15, -33, 0}
, {18, 27, -8}
, {-11, -4, 28}
, {-11, -1, 22}
, {-18, 31, -47}
, {19, 42, -27}
, {49, -49, -10}
, {29, -24, 36}
, {-38, -44, -12}
, {23, 2, 37}
, {30, 49, 52}
, {-24, 11, -50}
, {-7, 38, -43}
, {-40, -16, -30}
, {-3, 19, -47}
, {-54, -34, 45}
, {43, 15, -6}
, {7, 12, -13}
, {-47, 0, -20}
, {-43, -16, 33}
, {-46, 3, 16}
, {-7, -27, 21}
, {-27, 47, -39}
, {-10, 36, -13}
, {-31, 37, -19}
, {-44, -55, 6}
, {-48, -2, -23}
, {-21, 6, 15}
, {-42, 10, -18}
, {5, 10, 34}
, {-41, 15, 40}
, {-18, -51, 14}
, {17, 5, -3}
, {-62, 11, 19}
, {-4, -45, -21}
}
, {{42, -28, -38}
, {-45, 32, 29}
, {-15, -65, -54}
, {54, 47, 52}
, {-29, 44, 38}
, {-32, -49, 15}
, {36, -48, 47}
, {-32, -23, 54}
, {15, -28, 11}
, {30, 0, -24}
, {-34, 36, -11}
, {42, -6, -6}
, {-29, 51, -15}
, {13, 20, -6}
, {1, -18, -27}
, {5, 2, -23}
, {18, -24, -16}
, {51, 30, -53}
, {41, 44, -13}
, {37, 19, -30}
, {-47, -12, -41}
, {-23, -49, 4}
, {49, 47, -24}
, {6, 39, -11}
, {41, -42, 15}
, {-12, -32, -17}
, {14, -30, 46}
, {45, 4, 14}
, {9, 40, -48}
, {-8, -20, -7}
, {54, -55, 5}
, {-39, -2, -13}
, {-15, -1, -48}
, {45, -11, 51}
, {46, 10, 1}
, {32, -2, 21}
, {-38, -11, 8}
, {51, 24, -46}
, {-40, -12, 0}
, {49, -28, 13}
, {-19, -49, -17}
, {24, 12, 16}
, {54, -16, 46}
, {9, -42, 2}
, {47, -35, -22}
, {-24, -12, -36}
, {-45, 47, 21}
, {-41, 17, 0}
, {-34, -20, 28}
, {-22, -34, 5}
, {-24, 1, -19}
, {38, -34, 29}
, {-34, 0, -44}
, {40, 18, -9}
, {-8, 40, -19}
, {12, -18, -22}
, {-3, -15, -50}
, {-8, 36, -57}
, {28, -53, 42}
, {40, 49, 8}
, {8, -14, 20}
, {26, 33, -13}
, {-41, -35, -21}
, {-20, -5, -30}
}
, {{1, 17, -36}
, {-15, -18, -8}
, {0, -9, 32}
, {42, 39, 19}
, {26, -31, 47}
, {-33, -43, 28}
, {3, 19, 13}
, {-15, -18, 36}
, {3, 21, 8}
, {42, 5, 55}
, {-10, 26, 29}
, {-25, -2, 12}
, {-22, -41, 9}
, {-51, -9, 2}
, {-46, 13, -39}
, {39, -27, -42}
, {-34, 38, -19}
, {-29, -35, -36}
, {13, 46, 26}
, {-12, -12, -27}
, {13, 46, 21}
, {-47, -4, -45}
, {-37, 50, 3}
, {-22, -22, 33}
, {42, 56, 36}
, {-10, 7, 35}
, {-1, 52, 40}
, {-29, 17, 35}
, {-15, 25, 6}
, {-7, 18, 14}
, {-9, -64, 1}
, {-25, -31, 41}
, {39, 53, 44}
, {52, -10, 4}
, {-29, -38, -36}
, {-37, -48, 37}
, {15, -51, 39}
, {8, -39, 0}
, {-26, 7, 42}
, {-52, 31, -18}
, {-25, -4, -4}
, {47, 8, 12}
, {24, 0, 3}
, {51, 31, -46}
, {-21, -2, 51}
, {-30, -19, -27}
, {47, 4, -27}
, {-2, 46, 39}
, {4, -50, 37}
, {13, -13, 26}
, {4, -51, -48}
, {51, 31, -11}
, {-12, -34, -22}
, {-40, 1, -36}
, {5, 14, -10}
, {21, 17, 10}
, {-33, -19, -1}
, {-20, 51, -2}
, {-16, -30, 47}
, {41, 48, -33}
, {38, 5, -5}
, {-15, -37, 31}
, {15, 54, 12}
, {-38, -32, -31}
}
, {{-27, 40, -37}
, {-28, -40, -13}
, {-39, 9, 39}
, {12, -29, -3}
, {-1, 0, 4}
, {-11, 27, -54}
, {24, 37, -34}
, {-50, 25, -23}
, {-34, 9, -47}
, {-20, 37, -15}
, {-57, -54, 4}
, {9, -36, 22}
, {-43, -12, -39}
, {53, -40, 10}
, {30, -51, -39}
, {-34, -24, 3}
, {-6, 30, 40}
, {14, -25, 35}
, {-27, 6, -52}
, {-21, -43, -22}
, {-33, -28, 6}
, {-11, 33, 20}
, {-8, 0, 5}
, {33, 23, -20}
, {-25, 9, 24}
, {16, 48, 3}
, {-62, 36, 21}
, {-12, -32, 46}
, {-27, 51, -6}
, {-15, 33, -38}
, {13, 30, -9}
, {19, -11, -11}
, {-13, 17, 3}
, {42, -36, -40}
, {-66, 20, 22}
, {-39, -10, 13}
, {-41, -18, 28}
, {32, 49, -27}
, {-9, 4, 0}
, {-7, -12, -34}
, {10, 16, -16}
, {22, 7, -30}
, {-7, 45, 29}
, {10, -9, -29}
, {-4, 23, -19}
, {40, -52, -23}
, {-26, 38, 50}
, {45, -36, 0}
, {-18, 22, 48}
, {14, 22, 38}
, {52, 5, -24}
, {49, 10, 42}
, {10, 13, 0}
, {12, -28, -28}
, {28, 47, 2}
, {34, 44, -55}
, {-14, -45, 11}
, {-42, 27, 6}
, {-25, -19, 29}
, {-14, -26, 11}
, {-56, -49, 21}
, {24, 12, 2}
, {31, 21, -19}
, {12, -44, 32}
}
, {{10, -13, -10}
, {46, 18, -30}
, {-25, -10, 7}
, {35, -2, -38}
, {17, 0, 36}
, {31, 40, -32}
, {-49, -8, -34}
, {-20, 8, 54}
, {9, 40, -24}
, {13, 37, 10}
, {13, -49, 31}
, {9, 25, -20}
, {-34, 31, -12}
, {-32, 40, 28}
, {-28, -42, -10}
, {-1, 38, 0}
, {52, 1, 42}
, {22, 29, -24}
, {-39, 33, -33}
, {-8, 48, 34}
, {16, 10, -51}
, {51, -25, 11}
, {-18, 44, -27}
, {-43, -38, -14}
, {17, -37, -42}
, {42, 22, 12}
, {28, 20, 39}
, {26, -25, -15}
, {33, 58, 49}
, {-33, 13, -55}
, {36, 26, -27}
, {77, 26, 45}
, {-50, 34, -22}
, {-35, -7, 3}
, {35, -37, -9}
, {-25, -54, 38}
, {19, 23, 19}
, {-8, -11, -49}
, {-30, -19, 37}
, {2, 37, 23}
, {-42, 8, -42}
, {-13, -25, 33}
, {1, -12, -25}
, {46, -7, -38}
, {-34, -2, 35}
, {-8, -13, -34}
, {-47, 14, -45}
, {-28, 6, 48}
, {-32, 26, 65}
, {-42, 55, 52}
, {7, 30, 3}
, {64, 46, 9}
, {-27, -18, 36}
, {-49, -37, 2}
, {5, -5, 40}
, {13, 31, 43}
, {5, -47, 48}
, {64, -22, 68}
, {-27, -24, 66}
, {-30, 48, 34}
, {-10, 16, -1}
, {-47, 26, 22}
, {-14, -7, -43}
, {31, 27, 20}
}
, {{63, 54, 58}
, {42, -5, 18}
, {27, -28, 16}
, {-45, 43, -38}
, {-28, -16, 1}
, {-27, -57, -39}
, {-8, -26, -22}
, {-56, -33, 49}
, {6, 5, -23}
, {20, 49, 3}
, {13, 35, 23}
, {6, 49, -52}
, {-52, -12, 4}
, {41, 24, 51}
, {4, -11, -12}
, {38, 7, 18}
, {26, 25, 14}
, {10, 0, 31}
, {30, -9, -40}
, {-28, -25, 47}
, {-14, 18, 10}
, {52, 49, -68}
, {2, -4, 12}
, {17, 49, 19}
, {0, 10, -26}
, {3, -33, -25}
, {-8, -5, -6}
, {-40, -6, -5}
, {-5, -5, -18}
, {-41, 7, -26}
, {38, 16, -6}
, {-35, 15, 26}
, {-40, 26, -1}
, {-39, -23, -1}
, {15, -32, 13}
, {6, 34, 35}
, {-27, 13, 15}
, {-39, -21, -27}
, {36, 21, 34}
, {45, -24, 11}
, {39, 42, -24}
, {-38, -8, 0}
, {22, 17, -5}
, {34, 34, -37}
, {-33, 9, -23}
, {-39, 12, -9}
, {-24, 14, -35}
, {31, -23, -34}
, {-16, -23, 4}
, {28, 13, -25}
, {8, -13, -38}
, {26, 42, -38}
, {-39, 9, -39}
, {-20, 52, 21}
, {53, 24, -21}
, {29, -41, 0}
, {18, 32, -40}
, {0, 37, 19}
, {-17, -21, -22}
, {15, 1, 43}
, {0, 39, -30}
, {-23, -31, 33}
, {45, -28, -21}
, {22, -45, -26}
}
, {{8, 43, 26}
, {-27, 0, -36}
, {-4, 51, -1}
, {33, -14, 12}
, {-1, -39, -23}
, {-29, 4, 0}
, {-34, -48, 45}
, {0, 45, 46}
, {-4, 40, -23}
, {-8, -13, 21}
, {1, -31, 18}
, {38, -1, -25}
, {36, 17, 43}
, {-28, 12, -17}
, {-1, 17, -23}
, {53, 10, -47}
, {5, 17, -51}
, {-13, 28, 34}
, {-38, 2, 44}
, {30, -32, 27}
, {-49, 0, 49}
, {-12, 50, -9}
, {25, -40, -20}
, {-51, 12, 49}
, {-24, 50, 49}
, {3, -22, 8}
, {23, -24, 23}
, {-41, 33, 12}
, {45, 27, -39}
, {0, -26, -43}
, {-35, 17, 2}
, {38, -17, -33}
, {9, 16, 19}
, {7, -3, 5}
, {-21, 28, 59}
, {13, -14, 55}
, {45, 10, -14}
, {-1, 43, -32}
, {16, -28, 29}
, {14, 31, -35}
, {-3, -38, 20}
, {46, -11, -45}
, {-48, 14, -22}
, {6, -30, 0}
, {-44, 45, 19}
, {-6, -40, -50}
, {16, 32, 17}
, {33, -24, -23}
, {-12, 49, -31}
, {-2, -4, -41}
, {-51, -34, 10}
, {-20, -22, -49}
, {23, 43, -5}
, {32, -31, -37}
, {-47, 11, 21}
, {16, 52, -20}
, {10, 20, 3}
, {-31, -23, -14}
, {-75, -52, 3}
, {-48, -32, -6}
, {38, 13, -48}
, {-3, 7, -16}
, {20, -20, 8}
, {-20, 2, -29}
}
, {{4, -27, -46}
, {-33, -19, 53}
, {20, -32, 42}
, {-11, 31, -10}
, {-6, 48, 29}
, {-38, -55, -11}
, {-26, -5, 32}
, {-41, 2, -51}
, {34, 29, -31}
, {-11, 66, -53}
, {1, -27, 8}
, {10, 7, 41}
, {39, 5, -36}
, {53, 41, 58}
, {45, 25, 37}
, {-29, -39, -11}
, {-52, 2, -38}
, {24, 15, 28}
, {5, -11, 55}
, {8, -37, 20}
, {36, -8, 0}
, {27, -47, 24}
, {34, 16, 53}
, {32, 22, -3}
, {-44, 12, -34}
, {-7, -17, 18}
, {40, -41, 42}
, {-50, 4, 27}
, {4, -53, -52}
, {1, 38, 8}
, {47, -29, 51}
, {-11, -11, -40}
, {3, -6, 41}
, {31, 3, 6}
, {14, 36, 27}
, {-9, -5, 3}
, {11, 31, 44}
, {-27, 2, -30}
, {1, -21, 19}
, {-1, -26, 39}
, {-30, 9, -34}
, {-38, 20, 9}
, {17, 48, -16}
, {7, -20, -21}
, {30, -8, -27}
, {-30, 1, 13}
, {-28, 0, -47}
, {0, 0, -52}
, {-9, -21, -5}
, {3, 45, 28}
, {-18, 8, 14}
, {-19, -16, 42}
, {-49, -17, -27}
, {53, -11, 53}
, {-35, -34, 12}
, {7, 2, 10}
, {26, -21, 13}
, {-55, -11, -2}
, {-32, 46, -12}
, {30, 17, -8}
, {10, 2, -4}
, {-30, -26, -43}
, {-13, 34, -7}
, {26, -30, 5}
}
, {{5, 8, -23}
, {-2, 48, -15}
, {0, -26, 20}
, {55, 36, 60}
, {19, -34, -1}
, {-29, -36, 35}
, {-32, -2, -19}
, {2, -1, -13}
, {28, -47, -10}
, {18, -5, 35}
, {20, -19, -42}
, {-33, -2, 27}
, {-26, 51, 31}
, {-18, -23, -21}
, {64, 60, -38}
, {-22, -15, 9}
, {-4, -33, -37}
, {54, 28, 82}
, {3, -46, 42}
, {41, 17, -21}
, {13, 36, -11}
, {-11, -8, 11}
, {-1, 36, 11}
, {48, -40, -41}
, {-21, 41, -5}
, {21, 18, 17}
, {35, 39, -29}
, {49, 0, -39}
, {-52, -48, -31}
, {60, -5, -47}
, {52, -12, -11}
, {-36, -1, -20}
, {-40, -44, 29}
, {-24, -28, 20}
, {-34, 31, -1}
, {2, 32, 61}
, {30, -46, -19}
, {18, -23, 42}
, {-20, -15, 29}
, {35, 40, 11}
, {53, -3, 48}
, {-23, 24, -45}
, {-50, -19, -12}
, {24, -1, 12}
, {-16, 60, 54}
, {15, -14, 28}
, {-30, -31, -37}
, {45, 38, 5}
, {-21, -4, 22}
, {-36, -17, -42}
, {42, 31, 16}
, {2, -2, 7}
, {44, -37, 33}
, {65, 56, -24}
, {-35, -4, 19}
, {-18, 58, 67}
, {-37, 18, 47}
, {-19, 19, -1}
, {68, 1, -26}
, {-32, 4, -12}
, {22, -11, 7}
, {-1, -53, 20}
, {31, -46, -53}
, {55, -47, 38}
}
, {{13, -41, 12}
, {25, 54, -30}
, {25, -16, -10}
, {-15, -21, -37}
, {-7, 50, 47}
, {9, 6, 17}
, {10, -38, 23}
, {-26, 29, 22}
, {-51, 2, -8}
, {-30, 23, -30}
, {-16, 8, -20}
, {11, -20, 26}
, {15, -45, -51}
, {-22, 73, 9}
, {46, 10, 5}
, {58, 10, 20}
, {29, 37, -29}
, {35, 31, 11}
, {-42, 15, 0}
, {48, -10, -39}
, {1, 7, -28}
, {-52, -55, -61}
, {11, -30, -18}
, {-26, -2, 35}
, {-32, -11, -24}
, {25, 22, 26}
, {67, 48, 72}
, {52, 22, -13}
, {-31, -42, 42}
, {12, 6, 30}
, {66, 32, -4}
, {-34, -11, 37}
, {51, 29, 0}
, {-26, 37, 19}
, {-27, -20, -30}
, {10, 34, -30}
, {41, -43, 34}
, {-5, 40, -7}
, {14, -14, 19}
, {-1, 33, 41}
, {-10, 11, 22}
, {-34, -17, 25}
, {-29, -36, 23}
, {43, -9, -25}
, {42, 64, 4}
, {7, -57, -55}
, {6, 27, 47}
, {-34, 12, 6}
, {15, 38, 15}
, {-1, 23, -7}
, {-33, 38, 14}
, {4, -11, 1}
, {41, 18, 21}
, {-52, -48, 43}
, {63, -30, 52}
, {5, 0, -20}
, {-16, -32, -46}
, {10, -48, -1}
, {-9, 33, 55}
, {-5, -40, 0}
, {-21, -32, -9}
, {-46, 21, -25}
, {-6, 33, 9}
, {-34, 8, 28}
}
, {{-12, -33, -25}
, {-13, -13, -33}
, {-4, -37, 33}
, {46, -42, 20}
, {46, -32, -9}
, {35, 46, 22}
, {-11, 34, 14}
, {-20, 43, -48}
, {-16, -31, 2}
, {4, 16, -18}
, {22, -4, -41}
, {9, -18, -37}
, {-2, -6, -11}
, {-33, -50, -1}
, {-54, -2, 15}
, {9, 10, -47}
, {-23, 46, -39}
, {39, -31, 60}
, {-25, -23, -1}
, {47, -32, -40}
, {-41, 49, 16}
, {-3, 20, -53}
, {45, 9, 16}
, {-47, -54, 8}
, {6, -2, -33}
, {44, 31, -42}
, {-33, 7, -5}
, {-45, 16, 10}
, {-24, 40, 23}
, {-42, 39, 40}
, {-33, 42, 5}
, {-60, 5, 26}
, {-20, 14, 46}
, {35, 39, -58}
, {13, 20, 32}
, {9, 10, 36}
, {-12, 13, -9}
, {-12, 45, 17}
, {-27, -35, -29}
, {10, -7, 32}
, {-14, 45, 34}
, {0, 11, -40}
, {-19, 1, -28}
, {-16, 56, -10}
, {23, -45, -5}
, {24, 55, -35}
, {-1, -40, 53}
, {-58, 14, -48}
, {0, 12, -45}
, {-8, 17, -26}
, {40, -8, -3}
, {-48, 9, 13}
, {47, -9, -38}
, {33, 7, 39}
, {-8, -9, 32}
, {44, -1, -3}
, {-2, 43, 41}
, {-17, 10, -43}
, {-33, 8, -33}
, {-27, -15, -41}
, {-9, 11, -3}
, {1, 14, 13}
, {-5, -14, -34}
, {0, 16, 17}
}
, {{36, -33, 51}
, {38, 28, 26}
, {25, 20, 32}
, {50, -51, -32}
, {25, -54, 2}
, {8, 41, -46}
, {-32, 53, 29}
, {27, -43, -4}
, {57, 39, -23}
, {-15, -1, 34}
, {-10, 25, -44}
, {10, -20, 39}
, {7, 37, 38}
, {45, -9, -45}
, {32, 22, 12}
, {-19, 6, 31}
, {55, 35, -4}
, {-23, 49, 2}
, {-27, 16, -8}
, {39, -10, 50}
, {1, 3, 33}
, {-19, 33, 21}
, {25, -48, 48}
, {-19, 44, 5}
, {45, -8, 6}
, {40, -24, -8}
, {5, 10, -8}
, {20, 27, -16}
, {30, -40, -26}
, {-32, 14, 5}
, {-39, -12, 8}
, {-54, -27, 1}
, {-10, 1, -28}
, {-8, 4, -35}
, {37, -10, -41}
, {-41, -39, -29}
, {15, 2, -36}
, {41, 49, 15}
, {13, -7, -14}
, {-50, 44, -7}
, {7, 54, 9}
, {17, -49, -25}
, {31, 10, 42}
, {-19, -1, 30}
, {-41, 16, -45}
, {-35, 48, -23}
, {-42, -3, -36}
, {16, 25, -44}
, {11, -17, -1}
, {20, -15, 38}
, {-1, -44, -51}
, {21, 32, -6}
, {41, -19, -53}
, {46, 54, 0}
, {-37, -41, -7}
, {17, -5, -16}
, {-36, 18, -52}
, {-5, 25, -30}
, {-40, -4, -58}
, {-38, -42, 49}
, {54, -51, 21}
, {-16, 20, 9}
, {35, 19, 33}
, {-4, -38, -31}
}
, {{33, 9, 25}
, {51, 18, -47}
, {55, 46, 59}
, {35, -17, 24}
, {-10, 0, -6}
, {-47, 30, 49}
, {-13, 2, 39}
, {25, 42, -41}
, {42, -40, 33}
, {22, -11, 36}
, {42, -40, -21}
, {14, -5, 13}
, {-14, 41, -25}
, {-38, -22, 39}
, {51, 7, 16}
, {-34, 31, 42}
, {19, -43, -23}
, {23, 16, 44}
, {24, -42, -4}
, {-29, 24, 18}
, {13, -20, 19}
, {-6, -5, -38}
, {5, -46, -2}
, {23, -41, 3}
, {59, 30, -13}
, {34, 45, 35}
, {-58, -17, -32}
, {31, -50, 15}
, {-4, 14, -22}
, {19, 1, 42}
, {-20, 16, -45}
, {17, -40, -44}
, {37, 24, -37}
, {-24, 48, -57}
, {44, 45, -13}
, {0, 30, -41}
, {-14, -38, 49}
, {-44, -2, 6}
, {34, 46, 15}
, {-20, 42, -43}
, {-15, -25, -38}
, {-52, -5, -41}
, {5, 19, -6}
, {-14, -39, 43}
, {-39, 38, 30}
, {-22, 17, 19}
, {36, 36, 44}
, {-50, -9, -31}
, {10, -16, -7}
, {30, 6, 21}
, {15, -42, 48}
, {-51, -34, -43}
, {-40, 19, -21}
, {26, -36, 26}
, {48, -14, 49}
, {47, -16, -23}
, {0, 0, 25}
, {-28, -11, -23}
, {45, 55, -28}
, {12, -14, 17}
, {-39, 20, -53}
, {-1, -15, 27}
, {-12, 31, 4}
, {10, -12, 41}
}
, {{-31, -20, -3}
, {-10, 10, 0}
, {-26, 33, 4}
, {-40, 44, -10}
, {-44, -3, -42}
, {-14, -28, -40}
, {-30, -8, 42}
, {-9, -35, 4}
, {-32, 5, -34}
, {64, 33, 0}
, {58, 83, 9}
, {14, 41, 27}
, {53, -50, 39}
, {-15, 15, -37}
, {42, -28, 36}
, {40, -50, 20}
, {6, 38, -18}
, {44, -4, -4}
, {25, 43, 34}
, {29, -3, -22}
, {-48, 1, -43}
, {4, -38, -30}
, {-8, 49, 10}
, {32, 11, 97}
, {-12, 10, 39}
, {-3, 8, 28}
, {-13, 25, 28}
, {-14, 20, -38}
, {-31, -64, 3}
, {47, 43, 6}
, {17, -48, 37}
, {52, 31, 3}
, {-48, 5, -1}
, {-17, 30, -37}
, {69, 93, 19}
, {0, -58, -30}
, {10, -3, -35}
, {-18, -22, -16}
, {-17, 18, 0}
, {-38, 22, -38}
, {9, -19, 25}
, {-24, 26, 6}
, {-2, -54, -30}
, {32, 3, 6}
, {-41, 35, -26}
, {42, -48, 4}
, {-1, -33, 38}
, {45, 28, -38}
, {-4, 62, -12}
, {-3, -48, 16}
, {25, -4, -27}
, {34, 22, -30}
, {-27, 32, -17}
, {51, -32, -13}
, {-9, 52, -15}
, {-13, 51, -34}
, {-25, 60, 38}
, {20, 17, 26}
, {-42, -51, -33}
, {-5, 17, -29}
, {-25, 11, -20}
, {-9, -13, 11}
, {0, 43, 54}
, {8, 5, 13}
}
, {{8, -18, -35}
, {-46, -8, 14}
, {-54, 23, -16}
, {9, 9, -1}
, {12, -38, 57}
, {2, 14, -7}
, {31, 1, 18}
, {-29, -21, 15}
, {-38, -44, 0}
, {17, 12, 18}
, {3, -36, 19}
, {10, 47, 38}
, {14, -29, 18}
, {-49, 46, -31}
, {30, 19, 22}
, {-8, 41, 24}
, {28, -28, -41}
, {-54, 0, -45}
, {17, 42, -17}
, {-26, 1, 6}
, {-34, -7, 51}
, {-28, 34, -2}
, {41, 29, -19}
, {41, -42, 32}
, {-3, 23, 27}
, {-39, 25, -63}
, {54, 47, 8}
, {9, 5, -32}
, {56, 31, -49}
, {-20, -22, 33}
, {55, 25, -28}
, {-33, -40, -21}
, {-44, 43, -42}
, {-30, 1, -12}
, {-22, -50, 22}
, {25, -32, -30}
, {-16, -4, -17}
, {-2, -22, 3}
, {-33, -16, 32}
, {-18, -26, 26}
, {0, 45, 26}
, {9, 39, 13}
, {56, 45, -49}
, {0, -27, -51}
, {20, 18, 10}
, {-41, -60, -44}
, {42, 55, 16}
, {0, -23, 25}
, {-3, -40, 33}
, {22, 71, 25}
, {-11, 42, 6}
, {-12, -19, -5}
, {-1, 2, -36}
, {-57, -15, -44}
, {48, -24, -31}
, {-28, 53, 37}
, {30, -27, 57}
, {3, 42, 12}
, {0, -10, -32}
, {-40, -50, -1}
, {-2, 19, -9}
, {4, 27, -30}
, {-14, 52, 38}
, {36, -44, -32}
}
, {{-13, -36, -17}
, {39, -22, 5}
, {23, -30, -17}
, {3, -36, -24}
, {-3, 50, 0}
, {49, 0, 18}
, {-31, -19, 49}
, {-7, 58, 29}
, {13, 61, 20}
, {-1, -44, -36}
, {34, 7, 37}
, {0, -32, 3}
, {-34, 50, -22}
, {-13, 56, -23}
, {-3, -17, 11}
, {19, 34, 51}
, {-40, -6, -18}
, {26, 56, -46}
, {-36, 24, -32}
, {-43, -44, 13}
, {-9, -4, -9}
, {-32, 29, -50}
, {-23, -8, -46}
, {-16, -5, 29}
, {-3, -28, -71}
, {48, 32, -25}
, {53, 11, 19}
, {48, 0, -20}
, {35, -27, 26}
, {25, -15, -1}
, {67, 43, 20}
, {-56, 63, 42}
, {13, -30, 5}
, {23, 54, 24}
, {8, 36, 27}
, {54, 62, 63}
, {-43, 27, 45}
, {43, -31, -9}
, {-45, 34, -61}
, {50, -27, 16}
, {-22, -40, 2}
, {15, 0, -16}
, {-30, -40, -30}
, {-28, 2, 24}
, {44, 60, -34}
, {23, -28, -17}
, {-1, 24, -47}
, {0, 33, 10}
, {-22, -33, 35}
, {-49, -23, -19}
, {3, 5, 3}
, {-13, -48, -34}
, {24, -24, -45}
, {9, -5, -1}
, {-22, 41, 40}
, {-33, -29, -34}
, {-42, 7, 19}
, {24, -55, 17}
, {14, 47, -9}
, {-9, 30, 39}
, {-26, 7, 41}
, {10, 11, -11}
, {-39, -44, 25}
, {33, -7, -14}
}
, {{39, 44, 16}
, {16, 23, 0}
, {-6, 8, 5}
, {-27, 4, 27}
, {-48, -14, -50}
, {-58, 14, -57}
, {-38, 26, -29}
, {2, -12, 21}
, {42, 20, -7}
, {46, 14, 20}
, {-2, -3, -6}
, {-3, -23, -21}
, {-38, 44, -40}
, {-5, -16, 21}
, {7, 9, 12}
, {13, 35, -23}
, {-43, -32, -37}
, {-24, 22, -5}
, {-34, -28, 10}
, {37, 34, 16}
, {-5, -17, -53}
, {4, -9, -17}
, {52, 22, 31}
, {56, 6, 58}
, {-27, -13, 29}
, {15, 5, -33}
, {47, -30, -40}
, {46, 48, 3}
, {-24, -44, 18}
, {20, 29, 16}
, {3, -38, 9}
, {-17, -37, -21}
, {18, -14, 27}
, {-49, 26, -35}
, {60, 46, -19}
, {30, 8, 0}
, {51, -5, 52}
, {41, 16, 20}
, {-35, 52, -7}
, {2, -10, 24}
, {24, 10, -39}
, {-7, -19, 12}
, {5, -49, -1}
, {25, -15, 43}
, {50, 20, -30}
, {-5, -5, 28}
, {-18, 1, 36}
, {-27, -14, -10}
, {7, 7, 38}
, {-4, -1, -52}
, {-8, -44, 1}
, {-22, -51, 17}
, {-46, 26, 36}
, {21, 38, -19}
, {-29, -10, 17}
, {53, -43, 12}
, {31, -44, 21}
, {-37, 5, -24}
, {-56, -11, 30}
, {22, 50, 14}
, {-37, -34, -6}
, {-46, 1, -11}
, {49, -5, 32}
, {55, -23, 13}
}
, {{23, -11, -31}
, {0, 36, -11}
, {-1, -3, 4}
, {-19, -43, -17}
, {4, -39, 17}
, {8, 35, -40}
, {42, 20, -9}
, {2, -4, 32}
, {-43, 36, 8}
, {-7, 2, -22}
, {-13, 33, -2}
, {3, -4, 18}
, {-22, 3, -9}
, {37, 53, -38}
, {-40, -32, 31}
, {-40, 37, 44}
, {4, -29, 23}
, {15, -40, -26}
, {-40, -6, -11}
, {54, 33, 43}
, {2, 15, -11}
, {48, -9, 27}
, {20, 11, 50}
, {-35, -50, 39}
, {-41, 23, 7}
, {-14, 49, -26}
, {0, -56, -33}
, {-29, -9, -37}
, {-27, 1, 36}
, {23, -13, -28}
, {44, 5, 53}
, {-21, -47, -52}
, {-16, -7, 0}
, {-38, -39, 22}
, {32, 37, 35}
, {13, 18, -27}
, {-35, 4, 31}
, {-13, 47, 15}
, {-14, -2, 3}
, {-22, 18, 3}
, {-47, 28, 50}
, {-47, -4, -39}
, {-19, 18, 11}
, {-8, -50, 54}
, {24, -9, 37}
, {17, 48, 43}
, {-21, 40, -17}
, {-7, 4, 27}
, {-29, 1, -16}
, {-3, -48, -23}
, {0, 30, -3}
, {36, -34, -46}
, {32, 18, 2}
, {47, -23, -24}
, {35, 43, 35}
, {42, 7, 25}
, {50, 46, 44}
, {-12, -26, -35}
, {-68, 14, 19}
, {-36, 50, 7}
, {4, 32, 39}
, {-9, 35, 17}
, {9, 34, 7}
, {9, 37, 52}
}
, {{-13, -45, 34}
, {-37, 11, -3}
, {53, -1, 54}
, {-3, -8, 30}
, {50, 11, 26}
, {28, 26, -34}
, {43, 45, 44}
, {32, -1, 1}
, {47, 64, -7}
, {23, -62, -25}
, {-54, -31, 28}
, {-36, 23, 28}
, {10, 44, 7}
, {17, -36, 24}
, {-33, -44, 23}
, {8, 30, -23}
, {-4, 27, 19}
, {-43, -44, -21}
, {24, -17, -38}
, {-37, -8, 22}
, {46, -39, 54}
, {-5, 18, -29}
, {-6, 35, 24}
, {-5, -1, -13}
, {20, 39, 15}
, {43, 9, 26}
, {-24, -26, 1}
, {-8, 50, 3}
, {-11, 5, -44}
, {-51, -14, -26}
, {-9, 30, 42}
, {49, -1, 15}
, {-15, -58, 23}
, {31, -39, 8}
, {35, 61, -5}
, {-41, -17, 33}
, {-54, 33, 54}
, {-47, 11, -44}
, {-36, 13, 6}
, {46, 28, 46}
, {33, -36, 9}
, {-54, -33, -54}
, {8, -51, -9}
, {28, 5, -12}
, {3, 35, 25}
, {53, -30, 14}
, {-20, -24, -28}
, {-14, -9, -40}
, {31, 10, 13}
, {-25, -49, 22}
, {-31, -3, 33}
, {-43, -44, 11}
, {-38, -20, 31}
, {9, -25, -10}
, {-6, 11, -34}
, {-50, -33, 27}
, {-2, -3, -43}
, {-6, 32, 26}
, {-47, -12, -9}
, {-14, 37, 49}
, {3, -21, 41}
, {-15, -37, -1}
, {26, -46, -50}
, {-44, 46, 0}
}
, {{-33, -8, -50}
, {4, 3, 22}
, {9, -31, 17}
, {30, 4, -23}
, {-51, 39, -4}
, {19, -30, 21}
, {49, 24, -46}
, {7, 40, -12}
, {-48, -11, -3}
, {-53, -59, 3}
, {-44, 48, 46}
, {-11, 11, -41}
, {23, -11, 36}
, {9, -12, 37}
, {30, 35, 35}
, {2, -2, -2}
, {16, 26, -31}
, {-24, 14, -48}
, {-18, -14, 23}
, {-9, 0, 7}
, {-11, 0, -1}
, {-6, 58, -42}
, {-37, 6, -11}
, {17, 11, -42}
, {21, 0, 36}
, {15, 40, 21}
, {-5, -9, 0}
, {40, 27, 34}
, {41, 30, -44}
, {-19, -41, 24}
, {5, 11, 23}
, {20, 28, -20}
, {3, -16, -34}
, {4, -27, -23}
, {37, 36, -30}
, {16, -16, 26}
, {27, 46, 24}
, {-1, -35, 41}
, {-7, -35, 3}
, {-28, -39, 30}
, {6, -39, -42}
, {11, 37, 8}
, {34, -8, 46}
, {-45, -33, -49}
, {-42, -24, 2}
, {0, -20, 13}
, {20, 27, -21}
, {23, 43, 14}
, {-20, -11, 60}
, {-40, -33, 8}
, {47, 26, 13}
, {34, -49, -49}
, {9, 32, -2}
, {9, 53, -16}
, {-38, -31, -56}
, {26, 44, -20}
, {12, 47, 51}
, {24, 13, -29}
, {-9, -4, -14}
, {13, -4, -48}
, {45, 0, -18}
, {48, 19, -40}
, {37, -19, -34}
, {-34, -37, 42}
}
, {{34, -28, 2}
, {18, -22, -37}
, {60, 10, 38}
, {50, 43, 30}
, {-25, -10, -20}
, {-21, 39, 37}
, {-8, -12, 1}
, {-47, -22, -46}
, {-57, 62, -19}
, {-31, -52, 54}
, {-33, -32, 47}
, {-52, -27, -34}
, {-43, -41, 31}
, {-41, 2, 4}
, {10, 49, -22}
, {16, 6, -21}
, {18, -8, -27}
, {-25, -20, 37}
, {41, -50, 33}
, {0, 25, -21}
, {11, -34, 36}
, {5, 49, 19}
, {11, 13, 20}
, {32, -16, -40}
, {9, 37, -51}
, {63, 4, -19}
, {34, 42, 35}
, {46, -45, -28}
, {37, 36, -18}
, {-21, 37, -25}
, {47, 22, 34}
, {-34, -54, 8}
, {8, -24, 49}
, {14, -46, -3}
, {16, 14, 48}
, {39, 11, 77}
, {38, 20, -34}
, {35, -22, 10}
, {37, 50, 35}
, {41, -17, 22}
, {-53, -36, 57}
, {16, -43, -43}
, {-34, -23, -9}
, {43, -7, -49}
, {-24, 15, -2}
, {21, -15, 23}
, {-8, 46, 8}
, {-18, -17, -46}
, {11, 25, 38}
, {-69, 20, 13}
, {21, 9, 6}
, {-55, -10, 5}
, {17, 49, 17}
, {31, 66, -47}
, {50, -36, 10}
, {56, -14, 15}
, {-10, 18, -37}
, {-30, 8, 17}
, {44, -8, 33}
, {9, 7, 19}
, {17, 5, 24}
, {31, -16, -24}
, {-32, -35, 41}
, {-8, -44, -22}
}
, {{34, -34, -52}
, {21, 6, -38}
, {12, -66, -2}
, {44, -12, 50}
, {-51, 48, 31}
, {-54, 27, -19}
, {41, -16, 51}
, {1, -30, -47}
, {-19, -38, -42}
, {22, -46, 29}
, {29, 10, -22}
, {-5, 20, -35}
, {-28, -32, 38}
, {-2, -30, -2}
, {-29, -44, 12}
, {15, -4, -11}
, {-50, 14, -26}
, {-34, -4, -54}
, {-23, 14, 23}
, {-50, 17, -36}
, {50, -19, 46}
, {-42, -3, -25}
, {19, -9, -7}
, {23, 5, -8}
, {30, -34, 37}
, {-16, -57, 0}
, {14, 43, -20}
, {8, 3, 14}
, {-8, 20, -52}
, {-31, 42, 28}
, {-44, -2, -20}
, {-54, 41, 15}
, {-45, 47, 32}
, {-1, 51, 38}
, {14, 21, 35}
, {-53, -27, -4}
, {-38, 12, -42}
, {-42, -30, 34}
, {-49, 31, 13}
, {-42, -11, -29}
, {-27, 23, -17}
, {65, 61, 21}
, {54, 52, -3}
, {-46, -1, 4}
, {53, 2, -29}
, {-15, 22, -55}
, {-34, 39, 50}
, {24, -29, -10}
, {-44, 26, 60}
, {53, -7, 50}
, {-39, 29, 45}
, {-29, 21, -59}
, {13, -9, 34}
, {-44, 16, -31}
, {-34, 39, 39}
, {-36, -20, -16}
, {-1, 37, 15}
, {-54, -11, 35}
, {-2, 2, -10}
, {10, -41, 38}
, {6, -19, 50}
, {26, 32, -28}
, {40, -10, 47}
, {-19, 38, 38}
}
, {{-33, 2, -16}
, {13, -3, 43}
, {-21, 22, 2}
, {20, 27, 27}
, {47, -41, 28}
, {-43, -16, 4}
, {-22, -20, 15}
, {-53, 0, -45}
, {-3, 14, -11}
, {-31, 50, 51}
, {0, 61, -11}
, {65, -35, 12}
, {-7, -46, -11}
, {33, 0, 14}
, {-32, -22, -13}
, {-14, -5, 50}
, {44, -47, -42}
, {44, 39, 6}
, {-28, -27, 36}
, {45, 1, 9}
, {37, -51, -51}
, {-69, 23, -13}
, {-45, -41, 6}
, {-37, -28, 18}
, {-42, 26, -6}
, {-12, -20, -37}
, {60, 35, 46}
, {-42, -2, -21}
, {41, -16, 38}
, {-23, -32, 19}
, {-17, 33, 35}
, {28, 32, 39}
, {9, 63, -39}
, {43, 49, -9}
, {-18, -16, 1}
, {21, 21, 50}
, {15, 23, -9}
, {18, 31, 1}
, {-18, -9, -44}
, {1, -30, 58}
, {-8, 8, -3}
, {0, -14, -13}
, {6, -3, 24}
, {-28, 2, 3}
, {27, 43, 28}
, {-28, -38, -46}
, {-21, 23, 11}
, {67, 46, 30}
, {-33, -30, 19}
, {43, -19, 17}
, {31, -17, 8}
, {22, 27, -22}
, {-58, 2, 26}
, {-8, 42, -38}
, {-41, -5, -33}
, {-13, -6, 24}
, {32, -24, -9}
, {-14, -3, -27}
, {55, 50, 45}
, {-27, 39, 10}
, {12, 17, -12}
, {-42, 31, 25}
, {8, 39, -31}
, {23, 23, 23}
}
, {{15, 6, -7}
, {23, 47, -41}
, {26, -40, 30}
, {25, 10, 9}
, {-2, -40, 44}
, {9, 0, -26}
, {-4, 61, 9}
, {48, 24, -11}
, {19, 43, -33}
, {13, 34, 51}
, {8, -13, -9}
, {23, -23, 16}
, {37, -21, 14}
, {51, 63, 10}
, {55, 1, 27}
, {-39, 56, -36}
, {-12, -5, 32}
, {23, -20, 18}
, {41, -41, 0}
, {-19, -36, 10}
, {23, 37, -12}
, {26, -19, -42}
, {-16, -40, -49}
, {-19, 12, -24}
, {-83, 0, -39}
, {36, 22, -2}
, {-1, -3, -27}
, {-41, -36, -19}
, {-2, 11, -38}
, {30, -18, -35}
, {-18, -9, 66}
, {54, -53, 46}
, {25, -44, 4}
, {39, 0, -35}
, {19, -11, -1}
, {23, 25, 11}
, {17, 52, -27}
, {50, 38, 12}
, {-20, -7, -56}
, {25, 31, -11}
, {-3, 3, -34}
, {-2, -59, 23}
, {48, 36, 0}
, {34, -44, 45}
, {52, -8, 16}
, {-25, 2, 13}
, {1, -29, 34}
, {-37, -27, 12}
, {-36, 0, 19}
, {27, -11, -28}
, {11, 5, 0}
, {-30, -19, -48}
, {-36, 27, -43}
, {25, -33, -46}
, {25, 8, 51}
, {-44, 2, -7}
, {0, -23, -37}
, {28, -33, -27}
, {42, 26, -37}
, {0, -14, -36}
, {0, -11, 56}
, {45, -3, 9}
, {31, 9, 49}
, {-29, 46, 14}
}
, {{5, 5, -16}
, {33, -16, -44}
, {-36, 50, 75}
, {9, 2, 35}
, {-28, 44, 26}
, {24, 2, -30}
, {51, -43, -12}
, {50, -49, 8}
, {-19, -22, 12}
, {-59, -40, -39}
, {39, -15, -42}
, {45, -41, -20}
, {29, -48, -13}
, {-21, 58, 19}
, {53, 24, 17}
, {1, -22, 19}
, {0, 44, 42}
, {9, -40, 26}
, {-20, 1, -54}
, {33, 8, 23}
, {57, 52, 13}
, {-11, 38, -36}
, {-7, 5, -14}
, {-12, -40, -45}
, {3, -16, 27}
, {-3, 41, -39}
, {-18, -13, 21}
, {-11, -45, 25}
, {17, -45, -4}
, {-64, 37, -19}
, {31, 57, 13}
, {47, 12, -69}
, {-43, -37, -42}
, {-40, -14, -47}
, {20, -13, 8}
, {17, 4, 54}
, {-28, 32, 11}
, {-10, 25, -38}
, {27, 37, 0}
, {-8, -13, 8}
, {29, 51, -21}
, {-35, -21, -39}
, {-23, -48, -20}
, {58, 50, -41}
, {50, 5, -41}
, {53, 52, 10}
, {54, -5, -6}
, {33, -15, 44}
, {-21, -18, -43}
, {-25, 29, -29}
, {27, -3, -35}
, {-10, 45, 18}
, {2, 13, -45}
, {-18, 38, 55}
, {17, -43, -2}
, {-22, -29, -25}
, {-18, 45, 35}
, {36, -7, -40}
, {44, -34, -53}
, {28, -32, -2}
, {-40, -2, 37}
, {-25, -29, 51}
, {32, -33, 6}
, {45, -23, 41}
}
, {{24, -43, -22}
, {-4, 8, 23}
, {2, -22, -9}
, {-22, -32, 9}
, {10, 23, 21}
, {-50, 24, -60}
, {22, -17, 0}
, {10, 8, 16}
, {45, -40, -16}
, {32, -41, 5}
, {49, -11, 5}
, {45, 20, -16}
, {-15, 34, -13}
, {-37, 27, -44}
, {0, -20, 2}
, {8, -15, -12}
, {41, 0, -45}
, {-67, 20, 39}
, {5, 55, 23}
, {6, 40, 25}
, {-50, 5, -49}
, {38, 36, 0}
, {2, -10, 29}
, {40, -21, 32}
, {5, 29, 40}
, {21, 13, 32}
, {37, 13, -19}
, {32, 45, -31}
, {36, -39, -23}
, {-28, -47, 43}
, {26, -51, 14}
, {48, -20, -36}
, {23, 52, 42}
, {29, -42, 3}
, {-12, -32, 57}
, {-46, -33, 18}
, {16, 35, 34}
, {13, 8, -33}
, {-22, -2, 28}
, {35, 1, -12}
, {-27, 34, -45}
, {-34, 56, -25}
, {-3, -19, 5}
, {33, -4, 23}
, {-8, 35, 2}
, {-22, -26, -36}
, {53, 40, -29}
, {37, 6, -24}
, {-41, -4, 52}
, {-15, 20, -2}
, {18, -20, 35}
, {2, 5, 14}
, {-27, -11, -49}
, {27, -28, -27}
, {24, 13, -43}
, {3, -59, -53}
, {38, 40, -28}
, {-20, -20, 11}
, {22, -26, 8}
, {36, 49, 40}
, {-23, 27, 17}
, {-45, 43, 53}
, {9, -33, 16}
, {-19, 13, -35}
}
, {{27, -34, -18}
, {-23, -36, -46}
, {-21, -52, 14}
, {-25, -4, 48}
, {-3, -38, 48}
, {-48, 6, 2}
, {-9, 32, -32}
, {-35, 42, -6}
, {-3, 18, 31}
, {4, -9, -21}
, {-34, -46, -53}
, {44, -35, 50}
, {-50, -42, 27}
, {21, -9, -6}
, {-61, -48, -50}
, {-29, 16, 14}
, {3, -6, 33}
, {-59, -78, -33}
, {7, 50, 39}
, {34, 33, -10}
, {-45, -30, -14}
, {-43, -1, -26}
, {32, -34, 39}
, {26, 45, 25}
, {18, 2, -29}
, {-36, 35, -53}
, {-44, 22, -6}
, {-22, -39, 45}
, {-12, -39, -10}
, {8, -23, -41}
, {22, -44, 34}
, {-4, -8, -55}
, {8, -19, -45}
, {-21, 7, -24}
, {63, -35, -28}
, {-9, -45, -37}
, {-32, -21, -29}
, {37, 59, -38}
, {44, 7, -5}
, {-34, 2, -45}
, {-4, 3, -5}
, {32, 31, -16}
, {-4, 9, 42}
, {28, -10, -39}
, {37, -9, 36}
, {41, 26, 45}
, {-5, -14, -46}
, {-26, -54, -51}
, {18, 0, 1}
, {-17, -18, 41}
, {-13, 41, -34}
, {-22, 22, -26}
, {45, -1, 11}
, {-29, 52, 1}
, {-35, 41, -8}
, {-62, -30, -44}
, {17, 46, 11}
, {-15, -38, -49}
, {-31, -68, 11}
, {9, 35, 55}
, {51, 56, 19}
, {-11, -46, 49}
, {50, 43, 39}
, {-14, 48, -38}
}
, {{33, -27, -31}
, {52, 5, 54}
, {-47, -33, -46}
, {-10, -36, 1}
, {27, -6, -26}
, {16, 16, -43}
, {0, -48, 37}
, {6, 0, -30}
, {23, -33, -41}
, {10, -44, -34}
, {-29, 32, 48}
, {54, -51, -37}
, {-33, 44, 27}
, {-18, 28, 49}
, {45, 40, 35}
, {-38, 36, 25}
, {4, 28, 39}
, {-46, -3, -5}
, {-52, -9, -40}
, {16, 18, 45}
, {-28, 50, 2}
, {10, -15, 33}
, {-8, 30, -34}
, {24, -15, -17}
, {17, -27, -68}
, {2, -17, 32}
, {-23, -8, 65}
, {20, -46, -11}
, {57, 27, -2}
, {49, -49, 13}
, {36, 32, -11}
, {-37, -19, 2}
, {37, 61, 38}
, {18, -3, 9}
, {-12, -13, -11}
, {18, 38, 23}
, {17, 32, 55}
, {21, 22, -47}
, {-2, -35, -48}
, {-14, -35, -31}
, {-46, 54, -47}
, {-39, -4, 34}
, {-5, -4, -13}
, {-41, -43, -12}
, {17, -42, 30}
, {-30, 11, 37}
, {-11, 51, 6}
, {42, 27, 33}
, {39, -10, 15}
, {1, -16, 11}
, {25, 39, -6}
, {45, -24, 13}
, {-15, -27, 29}
, {7, 7, -51}
, {13, 19, 38}
, {-19, 15, -3}
, {-5, 39, -10}
, {-40, -47, 7}
, {0, 58, 20}
, {-42, 23, 40}
, {29, 15, -46}
, {5, 12, 17}
, {-53, -43, 43}
, {-46, 50, 12}
}
, {{13, -10, 29}
, {20, 4, 13}
, {-40, 18, -9}
, {-34, -20, 25}
, {-48, 32, 39}
, {-47, -49, -4}
, {38, -21, 9}
, {37, -18, 43}
, {7, 6, 0}
, {11, 22, 17}
, {16, 47, -29}
, {40, -25, 47}
, {-39, 50, -6}
, {5, -40, 8}
, {37, 14, -22}
, {45, -33, -45}
, {37, 6, -48}
, {-30, 67, 50}
, {-1, 49, -44}
, {13, -35, 35}
, {-14, -29, -40}
, {34, -37, -56}
, {46, 6, -8}
, {36, 40, -24}
, {38, -37, -15}
, {6, 13, 31}
, {-7, -21, -25}
, {-8, 11, -10}
, {-21, 41, 39}
, {-28, -33, 45}
, {21, -5, -66}
, {49, -4, -40}
, {39, 23, 12}
, {-10, -8, -24}
, {13, -12, 17}
, {8, -37, 49}
, {11, 44, 17}
, {45, 36, -11}
, {31, 10, -24}
, {42, -31, 14}
, {-10, -16, -26}
, {30, 30, 31}
, {-21, -42, 21}
, {-50, 14, 11}
, {47, 32, 43}
, {-23, -31, 29}
, {19, -14, -20}
, {-53, 10, -46}
, {-12, 5, 16}
, {58, -20, 45}
, {14, 14, 33}
, {0, -10, 19}
, {33, -21, -15}
, {-2, 37, 32}
, {-23, 10, -5}
, {0, 56, -1}
, {-12, -13, -7}
, {23, 22, -10}
, {-41, 40, 56}
, {-26, 24, 36}
, {28, 0, 31}
, {2, -35, -42}
, {-41, 23, -10}
, {-48, -13, 3}
}
, {{-33, 0, -28}
, {-37, 0, 45}
, {-29, -20, -47}
, {25, -41, -48}
, {34, 25, -41}
, {31, -12, -57}
, {15, 43, 23}
, {27, -9, 17}
, {2, -25, -8}
, {-32, 53, 23}
, {-5, -43, -55}
, {42, 35, -12}
, {-26, -39, 10}
, {63, 50, -30}
, {33, 45, -24}
, {9, 3, -27}
, {-46, -38, -32}
, {24, 37, 37}
, {49, 5, -21}
, {10, -24, 3}
, {7, 6, -10}
, {-5, 23, 25}
, {-20, 13, -24}
, {51, -35, 22}
, {51, 16, 21}
, {-42, 21, 24}
, {-37, 21, -37}
, {3, 18, -43}
, {-26, -37, 2}
, {55, 13, 20}
, {13, 7, -43}
, {23, -37, -51}
, {-36, 19, 16}
, {46, -35, -14}
, {-21, -33, -16}
, {44, 33, -25}
, {35, 33, -13}
, {-48, -22, 44}
, {-1, -31, -22}
, {11, 2, -25}
, {1, -54, -11}
, {-50, -43, -42}
, {26, -32, 10}
, {46, -1, 15}
, {61, 4, -39}
, {48, 35, -12}
, {-42, -48, -53}
, {16, 31, -5}
, {46, 0, 47}
, {0, 0, -41}
, {-12, -6, 14}
, {-9, 2, 1}
, {16, 8, -47}
, {-18, -48, -18}
, {66, 9, 0}
, {5, -35, -8}
, {47, 51, 4}
, {12, -23, -9}
, {-46, -60, -59}
, {-33, -32, 12}
, {2, -46, -22}
, {14, 23, 15}
, {10, -2, -23}
, {-38, 45, 42}
}
, {{-26, 25, 42}
, {-2, -20, 21}
, {-57, 51, 20}
, {-20, -26, 38}
, {-1, 35, -41}
, {38, 40, 24}
, {42, 44, 25}
, {-29, -15, -29}
, {-3, -23, -34}
, {-39, -20, -42}
, {-17, 21, -52}
, {26, 41, -57}
, {-43, -17, 40}
, {-32, -42, -4}
, {26, 36, 8}
, {-32, -18, 19}
, {-13, -3, 47}
, {-65, -2, 5}
, {-42, -15, -31}
, {48, 60, 14}
, {-16, 5, 24}
, {-41, 17, -39}
, {30, 23, 45}
, {43, -21, -29}
, {-28, -42, 41}
, {42, -25, 35}
, {24, -49, -52}
, {9, -38, 3}
, {-23, -1, -10}
, {-29, 0, -17}
, {39, 52, -9}
, {-17, -22, 49}
, {-43, -44, -51}
, {-17, 30, -27}
, {22, -38, -18}
, {-6, 43, 22}
, {4, -23, -14}
, {-7, 41, 6}
, {0, 37, -21}
, {-47, 66, -48}
, {-10, 52, 25}
, {16, 24, 38}
, {1, 12, 30}
, {-36, 47, 43}
, {-21, -37, -7}
, {-18, 3, 29}
, {-38, 27, 13}
, {-37, 21, -25}
, {29, -35, 4}
, {25, 52, -3}
, {-2, -28, -25}
, {61, 9, 67}
, {38, -23, 57}
, {3, -38, -21}
, {11, 6, 27}
, {-29, 37, -56}
, {30, -9, 17}
, {4, 72, 57}
, {-5, -71, -24}
, {-45, -6, 53}
, {-28, 63, -9}
, {-28, 14, -16}
, {-33, 24, 34}
, {-10, -35, -27}
}
, {{19, -26, -11}
, {-34, 6, 35}
, {-8, -34, -54}
, {-14, 30, -51}
, {-8, 26, -7}
, {-11, -11, -30}
, {29, -20, 30}
, {-32, 40, 5}
, {13, 39, -20}
, {-48, 20, -20}
, {15, 8, 41}
, {-33, -25, 22}
, {49, 45, -30}
, {10, -37, -37}
, {59, -58, 49}
, {-1, -31, -49}
, {62, 22, 65}
, {-51, 15, 22}
, {0, 3, 57}
, {-22, 33, -11}
, {24, 52, -4}
, {-29, -43, -57}
, {5, 40, -38}
, {17, 39, 2}
, {-1, -7, -8}
, {37, -47, 27}
, {-38, -5, -8}
, {-9, 27, 11}
, {32, 37, 52}
, {-12, -46, -34}
, {-12, 11, -18}
, {-1, -55, -18}
, {9, -1, 32}
, {-25, -36, -34}
, {9, 46, -10}
, {-12, -1, -58}
, {-14, 24, 60}
, {3, -10, 43}
, {-47, 58, -11}
, {-15, -10, -46}
, {30, -30, 7}
, {19, -5, -2}
, {-40, -19, -15}
, {-15, -62, -5}
, {-6, 23, 33}
, {66, -39, -19}
, {-32, 33, 20}
, {-46, -28, 44}
, {-46, 14, 11}
, {-9, 64, -12}
, {42, -17, 10}
, {55, -50, -30}
, {50, -10, 44}
, {-21, 28, -42}
, {-2, -36, -8}
, {17, 12, 13}
, {-50, 26, 18}
, {-17, -17, -15}
, {2, -35, -62}
, {-38, -39, -35}
, {44, 46, 31}
, {-31, 0, 35}
, {-44, 9, 47}
, {26, 13, 25}
}
, {{1, 31, 21}
, {10, 45, 47}
, {-43, 39, 16}
, {-18, -13, -30}
, {7, 44, -43}
, {-11, 8, -14}
, {16, -46, 18}
, {12, -8, -36}
, {44, -43, 46}
, {-28, 58, 27}
, {0, -27, 5}
, {10, 12, 35}
, {-1, -47, 43}
, {-14, -15, -6}
, {31, -6, 21}
, {45, -24, -13}
, {22, 19, -30}
, {15, 61, -23}
, {-46, 40, 46}
, {-12, -19, 8}
, {31, 33, 46}
, {-4, -18, -70}
, {3, -24, 36}
, {-38, -15, 22}
, {-23, -3, -6}
, {-14, 30, 55}
, {42, -20, -22}
, {12, -40, 39}
, {65, -29, 50}
, {56, -6, 26}
, {-42, 47, 1}
, {21, -11, 0}
, {18, -1, 22}
, {-7, -1, 4}
, {38, 14, -42}
, {0, -26, -6}
, {6, -31, 16}
, {-39, 0, 0}
, {-19, 43, -51}
, {14, 39, -31}
, {-18, 18, -15}
, {5, 7, -17}
, {-44, -24, 48}
, {0, 25, -9}
, {-38, 49, 22}
, {56, -8, -20}
, {-42, 21, -50}
, {-28, -39, 21}
, {30, -24, 15}
, {-13, 15, 44}
, {-48, 16, -4}
, {86, -9, 80}
, {-42, -13, -26}
, {17, -22, 0}
, {-13, 9, 19}
, {51, -19, 38}
, {0, -33, -8}
, {45, 52, -5}
, {27, -2, 22}
, {-23, -10, -42}
, {19, -11, -46}
, {44, -23, -24}
, {24, -18, 9}
, {-2, -39, 51}
}
, {{0, 57, -22}
, {-55, -15, 37}
, {-24, -2, -37}
, {29, 41, 17}
, {38, -35, 16}
, {-48, 0, -42}
, {-40, 3, -3}
, {-5, 39, -24}
, {-1, 5, -24}
, {39, 22, -14}
, {44, 26, 56}
, {24, 15, 38}
, {-43, -48, -45}
, {32, 30, -6}
, {-22, -42, -49}
, {16, 33, -2}
, {21, 45, -1}
, {-28, 7, 4}
, {-36, 24, -47}
, {26, 34, -7}
, {-32, 30, 18}
, {-7, -25, -20}
, {37, -6, -29}
, {13, 21, 4}
, {49, -27, -34}
, {35, 8, 40}
, {72, -28, 41}
, {14, 45, -34}
, {-31, 37, -4}
, {25, -28, -20}
, {-43, 39, 43}
, {-39, 13, 22}
, {-40, -47, -57}
, {38, -23, 11}
, {2, 16, 36}
, {11, 21, -12}
, {-4, 4, 28}
, {-16, -6, 20}
, {-44, 41, -8}
, {-11, -22, -43}
, {40, -6, 48}
, {42, 11, -6}
, {47, -52, -28}
, {-3, 37, -10}
, {-20, -5, 13}
, {-28, 6, -25}
, {20, 11, 47}
, {-52, 63, 44}
, {-9, -21, 6}
, {-32, 54, 21}
, {6, 5, 38}
, {-15, -8, 3}
, {10, -3, 23}
, {1, 31, -30}
, {-42, -14, 16}
, {47, 50, 21}
, {-42, 0, -24}
, {23, 30, -47}
, {-16, 8, -58}
, {12, -7, -15}
, {13, -30, 10}
, {-2, -43, 11}
, {-45, 5, 19}
, {-8, -1, 19}
}
, {{-24, 61, 10}
, {-34, -13, -34}
, {-67, 35, 33}
, {55, -42, 33}
, {1, -29, 46}
, {-42, -39, -58}
, {-36, -15, -37}
, {25, -27, 34}
, {20, -45, 40}
, {49, 0, -18}
, {1, -29, -38}
, {-47, -21, -34}
, {30, -5, -2}
, {34, -53, -15}
, {46, -34, -13}
, {0, -54, 24}
, {-29, 3, 29}
, {26, 5, 45}
, {-25, -2, -49}
, {-29, 31, -8}
, {-38, 12, 31}
, {24, 33, -15}
, {52, 27, -24}
, {41, 52, 32}
, {46, 17, 55}
, {-33, -55, -37}
, {19, 35, -20}
, {4, 51, 35}
, {0, 26, -59}
, {-32, -20, 14}
, {38, -10, 0}
, {-41, 8, -16}
, {-40, 20, -3}
, {11, 25, -27}
, {-17, 5, 56}
, {-42, 0, -17}
, {52, 40, 16}
, {-10, -24, 29}
, {7, -37, 42}
, {-33, 28, -29}
, {48, 18, 19}
, {-13, 0, -18}
, {44, 8, -25}
, {29, 48, -49}
, {-7, -12, -20}
, {-15, 49, 3}
, {4, 48, 17}
, {-48, -49, -54}
, {-31, -48, -54}
, {30, 39, 24}
, {-29, -11, -30}
, {3, 4, -18}
, {-13, 19, 47}
, {0, -35, 1}
, {11, 32, -12}
, {-47, -19, 30}
, {-50, 38, 21}
, {-19, -24, 25}
, {-27, -10, -8}
, {46, -49, -13}
, {47, 35, 32}
, {31, -32, 15}
, {53, 41, 40}
, {-2, -47, 21}
}
, {{-22, 10, -25}
, {4, 47, -17}
, {-22, 48, 32}
, {-3, 20, -65}
, {-52, -4, 36}
, {48, 18, -33}
, {3, 2, 31}
, {37, -35, -30}
, {19, -34, 43}
, {-40, -11, -33}
, {22, 37, 4}
, {-16, -23, 41}
, {15, 39, -49}
, {-17, -15, 26}
, {14, 14, -12}
, {14, 3, 6}
, {46, 19, -6}
, {-38, 30, 27}
, {47, -28, 3}
, {47, -18, -21}
, {-22, 45, 37}
, {3, 45, 11}
, {-19, -22, -11}
, {-28, 25, -15}
, {-58, -7, 23}
, {-10, -15, -45}
, {-10, -55, -42}
, {-45, 27, 9}
, {50, 14, 60}
, {9, 2, 48}
, {-10, -31, 22}
, {-31, 50, -24}
, {-22, 32, 31}
, {-54, 42, 44}
, {-37, -74, 30}
, {-52, 41, 34}
, {-4, -31, -22}
, {32, 9, 40}
, {19, -33, 0}
, {51, 6, 32}
, {-33, -12, -42}
, {-2, -47, 18}
, {39, 31, -23}
, {15, 24, 15}
, {35, 24, -9}
, {55, -37, 13}
, {-20, 22, 54}
, {-41, 18, -20}
, {-35, -11, 1}
, {-31, 57, -21}
, {-24, -34, -42}
, {66, -40, 56}
, {33, -29, 34}
, {1, 16, -4}
, {-36, -20, 8}
, {-37, 19, -43}
, {19, -46, -8}
, {52, -29, 40}
, {-30, 39, 19}
, {21, -7, -37}
, {47, -12, -1}
, {21, -42, -18}
, {16, 28, -48}
, {-10, 21, -21}
}
, {{-42, -10, 60}
, {4, -25, -43}
, {-23, -33, 41}
, {-57, -25, -55}
, {48, 29, 0}
, {50, 17, -24}
, {-22, 27, -32}
, {0, -48, -10}
, {0, -4, 50}
, {39, 19, 41}
, {-39, -41, 0}
, {-8, -40, -47}
, {-45, -49, 37}
, {-41, 41, 6}
, {30, 35, 14}
, {-11, 50, -21}
, {11, 16, 6}
, {42, -44, 25}
, {1, 19, -8}
, {42, 24, -46}
, {0, 50, 21}
, {-68, 21, -55}
, {1, -15, -37}
, {44, 32, 38}
, {-48, 39, 3}
, {55, -1, -6}
, {26, -27, -44}
, {-29, 14, 32}
, {13, -34, -34}
, {-20, 55, -10}
, {25, -37, 61}
, {11, -43, -37}
, {-50, -40, -36}
, {-24, -36, -27}
, {0, -25, -25}
, {-16, 51, 4}
, {-37, -48, -6}
, {40, 49, -46}
, {-41, -52, -5}
, {0, 24, -18}
, {6, -30, 23}
, {24, -21, -8}
, {-36, 20, 43}
, {-8, -17, -17}
, {44, -51, -9}
, {-8, 28, 21}
, {-42, -16, 49}
, {-22, 21, -55}
, {21, -16, -20}
, {68, 51, -18}
, {-21, -50, 33}
, {22, 61, 52}
, {-8, 26, 33}
, {18, -1, -32}
, {16, 58, -30}
, {-3, -1, -20}
, {26, 35, 24}
, {34, 40, -8}
, {-85, -59, 4}
, {27, 52, -42}
, {40, -40, 7}
, {6, -45, 9}
, {-27, 26, 48}
, {29, -24, 36}
}
, {{47, -32, 39}
, {-8, -24, -25}
, {-30, 49, -54}
, {-1, -11, 39}
, {33, -51, -42}
, {2, -12, -17}
, {-1, -50, 48}
, {-48, 12, -43}
, {-35, -30, 40}
, {-17, -26, -27}
, {-22, -6, 40}
, {-25, 0, 25}
, {-33, 33, -24}
, {13, -34, -25}
, {20, -2, 41}
, {22, -35, 51}
, {-10, 67, 48}
, {60, -2, -49}
, {32, 24, -32}
, {36, -2, -46}
, {3, 0, 0}
, {28, -46, 36}
, {-20, -20, 31}
, {-38, -50, 11}
, {-5, 27, -12}
, {7, -13, -25}
, {48, 33, 59}
, {-31, 42, 55}
, {-35, 49, 26}
, {6, 13, 0}
, {-50, -8, -51}
, {-20, 31, 45}
, {-42, 47, 44}
, {59, -43, 59}
, {-42, -53, 32}
, {-28, -35, 10}
, {41, -3, -38}
, {-36, 13, -44}
, {31, -26, -29}
, {60, 33, -53}
, {29, -11, -20}
, {-17, 37, 13}
, {14, 4, 11}
, {37, 29, -28}
, {-13, 49, -39}
, {14, 33, 3}
, {25, 37, 45}
, {16, -11, -14}
, {0, -71, 36}
, {42, 22, -18}
, {4, 43, -18}
, {15, 1, 82}
, {-33, -15, 0}
, {3, -63, 12}
, {21, 19, 32}
, {36, 15, 35}
, {-32, -27, -21}
, {0, 66, 53}
, {-25, -10, -6}
, {5, 25, 22}
, {-16, 35, -46}
, {31, 52, -19}
, {-49, -50, -21}
, {41, -37, -22}
}
, {{-26, -33, 55}
, {10, 0, 11}
, {0, -25, 9}
, {5, -5, -45}
, {17, -21, 0}
, {16, -29, 22}
, {-55, 34, -21}
, {26, 32, -34}
, {-7, -35, 11}
, {57, -23, 5}
, {41, -35, 5}
, {52, 33, -27}
, {-45, 1, 10}
, {-45, -30, -17}
, {-39, 15, -4}
, {33, -31, 3}
, {-33, -44, -22}
, {39, -21, -44}
, {17, 49, -34}
, {-49, 45, -38}
, {-28, -11, -3}
, {27, -41, 49}
, {44, 10, -4}
, {74, -24, -14}
, {18, 12, 62}
, {3, 29, -38}
, {-20, -26, -24}
, {48, -19, 43}
, {18, 31, -21}
, {-5, 38, 51}
, {25, 20, -5}
, {-5, -47, -56}
, {-10, -39, -25}
, {9, -7, -44}
, {-3, 55, 44}
, {-4, -12, -52}
, {-4, 40, 43}
, {-23, 3, 5}
, {37, 22, -25}
, {-5, 24, -19}
, {42, -20, 38}
, {-43, 29, 16}
, {23, -28, -18}
, {-46, 47, -20}
, {-40, 33, -38}
, {-9, 46, -51}
, {11, 0, -47}
, {29, 31, -50}
, {33, -30, 28}
, {13, -3, 11}
, {39, -3, -21}
, {18, 6, -4}
, {-39, 44, 40}
, {46, 23, 62}
, {46, 29, 31}
, {-40, 50, -48}
, {-38, -25, 1}
, {40, 10, -44}
, {-6, -9, -37}
, {-4, 0, 23}
, {52, -43, 12}
, {22, 46, 51}
, {21, -31, -13}
, {28, -17, -43}
}
, {{9, -29, -14}
, {-5, -27, -49}
, {15, -18, -39}
, {1, 36, 51}
, {-49, 11, -17}
, {-1, -29, 5}
, {47, 23, 36}
, {29, 16, 24}
, {-42, 22, 51}
, {2, 7, 40}
, {24, -25, -56}
, {32, -32, -41}
, {-40, 42, 10}
, {-47, -16, 37}
, {37, 27, 25}
, {4, -40, 52}
, {44, -30, 34}
, {-1, 5, -35}
, {-36, -42, 42}
, {-18, 12, -22}
, {39, 32, 23}
, {-2, 6, 47}
, {-19, -4, -11}
, {25, -26, -6}
, {34, 32, -23}
, {-15, 30, 43}
, {-54, -60, 7}
, {28, -28, -44}
, {-16, -42, -26}
, {12, 47, 31}
, {-11, 22, 31}
, {-6, -62, -41}
, {8, -64, 16}
, {-30, 25, 1}
, {49, 38, 13}
, {-8, -26, 29}
, {-40, -8, -47}
, {56, -17, 8}
, {-42, 0, 6}
, {-15, 54, -37}
, {-38, 0, 1}
, {2, -45, 14}
, {26, 24, 40}
, {27, 20, 38}
, {28, 31, -50}
, {-6, -38, 13}
, {-17, -8, -60}
, {17, 8, -53}
, {0, 2, -46}
, {-42, 0, -53}
, {17, 32, 12}
, {-15, -34, 15}
, {5, -31, -9}
, {43, -22, 27}
, {33, 35, -21}
, {1, -24, -29}
, {-28, 36, 34}
, {36, 0, 17}
, {36, 5, 24}
, {7, 29, 49}
, {36, -20, 42}
, {-47, 42, 9}
, {-37, -43, 7}
, {27, 18, 56}
}
, {{-21, 38, -18}
, {17, -41, -37}
, {-19, -15, -2}
, {-41, 17, -30}
, {37, 6, 56}
, {8, 7, 6}
, {-10, -37, 3}
, {32, -44, 30}
, {-9, 11, 18}
, {-1, -12, -7}
, {30, -15, 43}
, {30, -20, -43}
, {-16, -37, -9}
, {3, 19, 13}
, {41, 42, 28}
, {-55, -14, -10}
, {51, -4, 27}
, {-7, 34, 21}
, {12, -48, -32}
, {44, 51, 38}
, {47, -34, -51}
, {-46, -17, -46}
, {-23, -30, -21}
, {-34, -6, -3}
, {-25, 32, 58}
, {21, 35, 7}
, {-10, 64, 7}
, {57, 36, -1}
, {-9, 17, 3}
, {38, 45, 2}
, {-61, 9, 7}
, {51, 76, -2}
, {8, 9, 61}
, {4, 42, 25}
, {-1, 51, 40}
, {29, -20, 34}
, {-5, 7, 9}
, {-28, -2, -17}
, {51, 27, 32}
, {49, -5, -35}
, {-27, 18, 5}
, {66, 35, 7}
, {-14, 8, 36}
, {-32, 43, 45}
, {31, 1, -41}
, {19, 46, 32}
, {35, 5, 34}
, {-2, 9, 53}
, {-11, 18, -24}
, {30, 62, -7}
, {5, -50, 22}
, {-15, 43, 14}
, {-50, -22, 44}
, {23, -35, 11}
, {34, 3, -4}
, {-13, -16, 0}
, {2, -51, 26}
, {25, 15, 11}
, {35, 15, 27}
, {-46, 4, -6}
, {-49, -43, -24}
, {-18, -27, -33}
, {-22, 21, 32}
, {38, 10, 0}
}
, {{-16, -35, -38}
, {34, -20, -1}
, {-39, -6, 36}
, {10, -51, 24}
, {-27, 36, -12}
, {20, -22, -49}
, {46, 13, -30}
, {46, -40, 3}
, {-12, 4, -6}
, {-18, -22, 0}
, {28, 3, 19}
, {34, -17, 21}
, {8, 33, -26}
, {37, 9, -38}
, {-46, -30, 0}
, {-30, 24, -32}
, {-34, -15, 21}
, {1, 18, 2}
, {43, -38, -17}
, {5, -14, 2}
, {-18, -50, -40}
, {-7, -34, -10}
, {-20, -49, -57}
, {33, 0, -37}
, {40, 33, 36}
, {-37, -50, 40}
, {2, 15, 15}
, {0, -47, -8}
, {41, 0, 34}
, {4, 9, -24}
, {-14, -12, -32}
, {49, 4, 10}
, {-28, 7, 33}
, {57, -19, -39}
, {-45, -18, -27}
, {-41, 12, -2}
, {13, 47, 18}
, {26, -36, -47}
, {-25, -3, -46}
, {-11, -17, -52}
, {-3, -25, 6}
, {9, 4, -15}
, {9, 1, -30}
, {30, 50, 11}
, {-1, 20, 31}
, {3, -22, -20}
, {-23, -28, -57}
, {46, -32, 15}
, {-7, -48, -6}
, {-30, -51, 43}
, {-12, -10, 12}
, {-25, -15, -53}
, {-29, 15, -14}
, {20, -50, 31}
, {-24, -21, -13}
, {-16, -21, 44}
, {41, -36, 2}
, {9, -39, -56}
, {-39, 7, 3}
, {28, -11, 31}
, {-8, 32, -26}
, {-8, -2, -5}
, {-39, 41, 25}
, {-40, -29, 44}
}
, {{26, -24, 43}
, {11, 24, 52}
, {-11, 15, 43}
, {24, 18, 45}
, {5, -22, 18}
, {-34, -6, 38}
, {-48, 30, 5}
, {22, -39, 22}
, {-29, -14, -17}
, {24, -40, -17}
, {-15, -33, -18}
, {16, 20, 5}
, {33, -26, -14}
, {52, -40, 36}
, {39, -20, -14}
, {-13, 31, 27}
, {-26, -34, 3}
, {54, 52, -14}
, {9, -39, 33}
, {-29, 0, -15}
, {29, -23, 50}
, {-30, -1, -54}
, {-23, 43, 21}
, {-25, -2, -61}
, {-21, -15, 20}
, {-42, 53, -41}
, {-4, -19, 23}
, {-46, -14, 2}
, {-50, -48, 19}
, {16, 29, -42}
, {27, 26, 18}
, {45, 11, 24}
, {33, 42, 46}
, {0, 9, -12}
, {-46, -58, -31}
, {25, 56, 20}
, {-35, 46, -29}
, {33, 8, 40}
, {2, -13, -4}
, {58, 36, -28}
, {49, -8, 51}
, {-48, 22, -48}
, {38, 26, -6}
, {1, 21, 44}
, {26, 41, -35}
, {1, 24, 11}
, {10, -12, -27}
, {44, 18, -35}
, {-46, -44, -11}
, {-54, -41, -2}
, {42, 12, -30}
, {-8, -3, 48}
, {50, 15, -2}
, {19, 13, 25}
, {10, 4, 0}
, {-32, -25, -31}
, {27, -22, -13}
, {35, 4, 47}
, {43, 27, 40}
, {9, -52, 53}
, {4, 25, -32}
, {-40, -29, -48}
, {10, -40, -16}
, {47, 16, 1}
}
, {{24, -35, 47}
, {-9, -10, 24}
, {51, -27, 10}
, {-10, -22, -8}
, {12, -27, 34}
, {44, 5, 32}
, {22, 32, 43}
, {2, -34, 3}
, {-48, -33, -36}
, {-32, -47, 49}
, {-4, 42, 39}
, {-24, -48, -14}
, {-39, 38, 49}
, {-48, -46, -56}
, {18, -30, -4}
, {30, -19, 24}
, {40, 19, 36}
, {25, -39, 26}
, {-13, -32, -29}
, {-36, -44, -23}
, {-10, -40, 15}
, {40, -16, -16}
, {35, -23, -23}
, {7, -49, -24}
, {-4, 44, -40}
, {-14, -36, 6}
, {29, -4, -3}
, {41, -5, 37}
, {-12, 35, 0}
, {-32, -36, 40}
, {21, -44, -4}
, {4, 19, -30}
, {44, -34, -8}
, {-36, -48, 18}
, {51, 56, -33}
, {-6, -30, -38}
, {9, -3, 36}
, {-27, 17, 24}
, {-9, -2, -23}
, {-33, 0, 32}
, {-49, -13, -35}
, {35, -39, 8}
, {49, -5, -16}
, {16, -53, 4}
, {16, 45, 10}
, {26, -48, 29}
, {19, 33, 29}
, {7, 33, 42}
, {-44, 7, -19}
, {35, -33, -53}
, {-46, 39, 24}
, {-7, -40, 6}
, {-14, -49, -17}
, {-49, -32, -2}
, {48, 39, -33}
, {25, 6, -41}
, {13, -14, 27}
, {42, 2, -11}
, {-50, 17, -10}
, {-23, 39, 14}
, {-35, -23, 14}
, {-40, -6, 14}
, {8, -2, -6}
, {-31, 0, -54}
}
, {{-50, 9, 12}
, {-37, -47, 26}
, {-28, 26, 3}
, {-54, -33, -23}
, {-38, -32, -50}
, {33, 47, -27}
, {55, 36, -31}
, {23, 7, 2}
, {-16, -20, 54}
, {-27, -47, -2}
, {37, -42, -9}
, {-19, -56, 30}
, {-48, 26, 8}
, {40, -29, 28}
, {-46, -7, 25}
, {9, -18, 9}
, {37, 56, 61}
, {-25, -12, 10}
, {14, 53, 28}
, {0, -27, 41}
, {5, 32, 29}
, {-6, -25, -48}
, {-19, -17, -49}
, {-39, -44, -8}
, {-20, -16, -30}
, {38, 3, 38}
, {-39, -7, 25}
, {-2, 21, 50}
, {-31, 7, -20}
, {-19, 21, -18}
, {9, -30, -43}
, {52, -9, -20}
, {8, 24, -3}
, {34, 19, -5}
, {-28, 4, -34}
, {-58, 11, -50}
, {-23, 31, 53}
, {23, 19, -26}
, {16, 32, -46}
, {-24, 32, -36}
, {2, 10, -27}
, {29, -14, 23}
, {-12, 34, -34}
, {52, -16, 52}
, {44, -25, 27}
, {35, -16, 58}
, {39, 21, 43}
, {-21, -37, 55}
, {9, 40, -6}
, {-19, 28, 35}
, {-42, -19, 40}
, {2, 8, 35}
, {-24, -5, 0}
, {-53, -47, -53}
, {32, -21, 7}
, {-53, 21, 33}
, {27, -7, 18}
, {25, 1, 49}
, {-12, -42, -5}
, {-15, 3, 59}
, {-24, -1, -24}
, {0, -26, 7}
, {-29, -23, 27}
, {-17, 2, 9}
}
, {{32, 3, 61}
, {54, 2, -13}
, {-14, 13, -3}
, {32, -37, -48}
, {31, -30, 11}
, {5, -12, -40}
, {10, -22, -41}
, {53, 35, 14}
, {22, 9, -13}
, {68, 4, 15}
, {-4, 38, 65}
, {45, 50, 21}
, {14, -32, -16}
, {-56, -43, 15}
, {29, 28, -13}
, {16, -35, -29}
, {45, 38, 45}
, {35, 20, 23}
, {51, -18, 10}
, {-37, 58, -38}
, {3, -44, -6}
, {-38, 36, -39}
, {-50, -1, -17}
, {27, 53, 11}
, {-8, 68, 61}
, {-12, 50, -11}
, {42, 10, 7}
, {25, 32, 40}
, {67, 44, -15}
, {-30, -19, 14}
, {-20, 10, -75}
, {60, -11, -31}
, {37, -19, 32}
, {-6, 37, 24}
, {47, 30, -1}
, {-30, 19, 33}
, {-38, -42, 36}
, {-30, -31, 43}
, {-9, 23, 2}
, {-23, 30, 11}
, {4, -22, -31}
, {55, 12, 76}
, {-46, 11, -38}
, {41, -2, 48}
, {-29, 0, -48}
, {15, -37, -15}
, {14, 45, -12}
, {24, 45, -9}
, {10, 37, -48}
, {-8, 3, 37}
, {16, 26, -43}
, {80, -6, 59}
, {-9, 32, -25}
, {-15, -3, 27}
, {-47, -38, -44}
, {14, 24, -33}
, {34, -30, 21}
, {57, 42, 5}
, {20, -32, 36}
, {-41, 33, -6}
, {40, 44, -18}
, {-36, -22, 11}
, {-17, 26, -13}
, {19, 20, -26}
}
, {{-49, -41, -55}
, {29, 31, -25}
, {-66, -58, 0}
, {23, -5, -52}
, {8, 50, 14}
, {-26, 49, 50}
, {-5, -43, 4}
, {12, 8, 21}
, {2, -26, -46}
, {-30, -37, -38}
, {-26, 16, 0}
, {0, -20, 54}
, {0, -25, -27}
, {-36, -42, -56}
, {3, -10, 22}
, {-38, 18, 4}
, {-39, -36, -44}
, {33, 16, -25}
, {-9, 50, 8}
, {33, -12, 1}
, {-13, -44, 35}
, {42, 9, -56}
, {-7, 20, 4}
, {11, 37, -46}
, {-36, -52, -28}
, {-35, -45, 2}
, {41, -1, -9}
, {-12, 36, 4}
, {35, -16, 47}
, {44, 38, -1}
, {-29, 5, -9}
, {28, 1, 45}
, {-26, 49, -37}
, {-10, 29, -6}
, {-25, 4, -54}
, {23, 4, -9}
, {53, -2, 3}
, {-27, -4, -37}
, {-13, -20, -31}
, {-43, 33, -46}
, {35, 9, -41}
, {-9, -18, -43}
, {27, 51, -6}
, {8, -43, 19}
, {-20, 47, -46}
, {-39, 17, 31}
, {9, -11, 18}
, {46, 32, 32}
, {17, 21, -41}
, {-19, 27, -33}
, {40, -38, 28}
, {-30, 12, -34}
, {-34, 21, 51}
, {-4, 15, 14}
, {-5, 17, -56}
, {10, -42, -43}
, {52, -14, 0}
, {10, 12, -32}
, {19, 48, -68}
, {32, 23, 53}
, {-43, -7, -29}
, {51, 27, -16}
, {52, -22, 44}
, {17, -16, -28}
}
, {{-37, -2, -51}
, {46, 39, -47}
, {16, 18, -58}
, {56, -43, 28}
, {8, 56, 31}
, {0, -30, -16}
, {0, -18, -53}
, {0, -38, 33}
, {34, -29, -27}
, {-24, -35, 0}
, {-5, 3, 38}
, {-8, 20, -8}
, {-35, 45, 2}
, {49, 26, -22}
, {30, 24, -28}
, {-27, 3, 41}
, {-36, -4, -24}
, {-53, 45, -11}
, {0, 46, -13}
, {28, -56, 5}
, {36, -17, -22}
, {44, -41, -17}
, {8, 37, -40}
, {43, -24, 34}
, {10, -71, -1}
, {-10, -27, 11}
, {67, 66, 56}
, {-30, -54, -46}
, {-16, 29, 32}
, {-7, -35, 35}
, {-23, 34, 31}
, {36, 4, -10}
, {21, -42, 33}
, {48, 24, 26}
, {39, -15, -1}
, {28, 58, 22}
, {33, -10, 16}
, {22, -35, 23}
, {-12, 9, 10}
, {61, -36, -6}
, {0, 20, 28}
, {36, 53, -52}
, {47, 43, 1}
, {-8, -12, 14}
, {-10, 4, 37}
, {48, 12, 9}
, {-27, -55, -45}
, {36, -19, 27}
, {-27, 0, 50}
, {-36, 43, 0}
, {-18, 5, 46}
, {-34, 36, 42}
, {-24, -51, -35}
, {-61, 15, -19}
, {21, 25, -43}
, {-31, -7, -22}
, {-5, -44, 13}
, {-46, -46, -20}
, {22, 54, -32}
, {35, -36, -13}
, {0, -4, -23}
, {-10, -3, -33}
, {-4, 29, 25}
, {37, 47, 7}
}
, {{10, 52, -14}
, {-50, 6, -27}
, {19, 26, 29}
, {-33, -4, 25}
, {26, -9, 13}
, {-7, -20, -27}
, {20, -25, 5}
, {13, 28, 24}
, {39, -40, -29}
, {-24, 33, -50}
, {9, -40, 3}
, {-26, 48, 31}
, {-31, 47, 5}
, {22, -33, -49}
, {20, -34, -31}
, {-3, 51, -28}
, {-3, -13, -12}
, {33, -60, 24}
, {18, 42, -31}
, {-25, -39, -48}
, {35, -46, 10}
, {16, 14, -50}
, {21, -22, -52}
, {14, -33, 7}
, {28, 12, -28}
, {20, 5, 2}
, {-42, 8, -32}
, {8, 28, 39}
, {-29, 14, -26}
, {2, 19, 9}
, {46, 10, -34}
, {-19, -5, 40}
, {-5, -37, -38}
, {-21, 38, 10}
, {26, 2, -30}
, {1, -19, -18}
, {-6, 41, 38}
, {2, 47, 23}
, {33, 5, 14}
, {-4, -43, -56}
, {6, 1, -15}
, {50, -25, -51}
, {18, -50, 45}
, {37, 15, -5}
, {-32, 2, -44}
, {-28, -1, -39}
, {13, 0, -31}
, {23, -15, -44}
, {-4, 34, 43}
, {-31, -32, 31}
, {50, -12, -37}
, {-37, 31, -4}
, {30, -46, -50}
, {2, -13, 12}
, {-46, 3, -42}
, {-10, 43, 13}
, {-32, -40, -37}
, {-25, 38, 28}
, {-46, -47, -40}
, {0, 50, 19}
, {-46, -49, 22}
, {36, -21, 17}
, {-43, 11, -46}
, {-38, -20, -27}
}
, {{32, -4, 55}
, {49, 45, -31}
, {30, -10, -13}
, {38, 46, 10}
, {-17, 17, -37}
, {13, 22, 48}
, {6, 20, -7}
, {20, -47, -41}
, {-30, 23, -4}
, {12, -39, 13}
, {7, 63, 10}
, {-30, -14, -34}
, {-39, -26, 0}
, {38, -1, -33}
, {12, 15, 1}
, {-31, 26, 38}
, {9, -30, 13}
, {48, -21, 5}
, {-18, 29, -18}
, {-31, 6, 33}
, {-9, 13, 10}
, {-2, 13, 10}
, {51, -27, 38}
, {45, -17, 42}
, {12, -23, 17}
, {-20, 17, 5}
, {24, 59, 3}
, {47, -46, 21}
, {-26, -28, 47}
, {-7, 32, -13}
, {23, 40, 5}
, {18, 41, 36}
, {-2, 11, -26}
, {24, 15, 25}
, {-22, -46, -9}
, {-15, -7, 30}
, {-40, 29, -11}
, {31, 34, -16}
, {41, 0, -24}
, {32, -4, -58}
, {42, -29, 40}
, {-5, 16, -18}
, {-53, 31, -5}
, {-9, -32, 24}
, {7, 8, -44}
, {5, -15, -24}
, {-33, 38, 45}
, {50, 0, 18}
, {-9, -2, 56}
, {-3, 5, 4}
, {-46, -36, 8}
, {25, 28, 34}
, {-27, -55, -13}
, {22, 21, -25}
, {-21, -32, -26}
, {-5, -10, -13}
, {-36, 44, 33}
, {-17, 38, 69}
, {-5, 39, 26}
, {11, -11, 45}
, {-40, -25, -2}
, {-11, 5, -48}
, {6, 0, -15}
, {-15, -46, 40}
}
, {{29, -39, 28}
, {-30, 23, -37}
, {9, -5, 41}
, {-31, -3, -48}
, {-20, 35, -23}
, {33, -42, 8}
, {18, 2, -31}
, {30, 21, 33}
, {37, 45, -8}
, {-18, -58, -18}
, {-11, 32, 0}
, {-38, 4, 20}
, {-13, -17, 41}
, {21, 13, -14}
, {-12, 14, -47}
, {54, 48, 22}
, {54, -3, -46}
, {-28, 37, 5}
, {-42, -13, 36}
, {-7, 5, 38}
, {12, -29, 45}
, {45, 18, 53}
, {-20, -48, 50}
, {10, -26, 2}
, {36, -3, 15}
, {45, -10, -44}
, {-60, -38, -53}
, {-39, -29, 27}
, {-23, -11, -17}
, {39, 10, -21}
, {21, 62, 1}
, {0, 13, 29}
, {-20, -30, 34}
, {-38, -1, -4}
, {7, 52, 17}
, {-51, 25, -11}
, {-5, -8, 7}
, {0, 9, -23}
, {-17, 55, 12}
, {23, 32, 9}
, {-25, -12, -23}
, {-14, -39, 5}
, {43, 20, 24}
, {15, 36, -21}
, {-34, 10, 39}
, {0, 15, 50}
, {19, 39, 5}
, {-53, -35, -17}
, {33, -27, -34}
, {-31, -17, -40}
, {6, -49, 20}
, {37, -6, 4}
, {44, -10, 21}
, {10, -20, -45}
, {21, 20, 4}
, {-58, 2, 28}
, {-30, -17, -5}
, {-25, 0, -35}
, {15, 31, -30}
, {-51, -50, -3}
, {9, 37, 3}
, {-7, -7, 22}
, {-33, -53, 3}
, {38, 38, 22}
}
, {{-7, -39, 38}
, {46, 47, 25}
, {6, -45, 29}
, {-50, -24, 10}
, {46, -26, -12}
, {45, -26, 47}
, {-20, 20, -3}
, {-50, 14, 43}
, {10, 4, 45}
, {50, -20, 4}
, {-15, 24, 17}
, {13, -12, 1}
, {-36, -29, 30}
, {48, -16, -18}
, {19, 57, 8}
, {17, 6, -29}
, {-33, 28, -32}
, {-5, -14, 40}
, {-9, 9, -10}
, {-33, 5, -28}
, {-20, -39, -25}
, {-3, -43, -32}
, {9, -51, -40}
, {7, 25, -25}
, {-2, -26, 22}
, {29, 15, 17}
, {70, 44, -27}
, {-16, 43, 47}
, {-33, -21, 31}
, {-43, 29, 22}
, {-3, 57, 5}
, {42, -26, 43}
, {47, -46, 18}
, {54, 48, -17}
, {-50, 0, -27}
, {-45, 35, -6}
, {-33, 31, -41}
, {-43, 18, -12}
, {2, -13, -24}
, {-10, -41, 36}
, {-6, 49, 32}
, {3, 29, 46}
, {-38, -18, 46}
, {16, 46, 45}
, {-12, 24, -14}
, {18, 13, -9}
, {1, 14, 18}
, {-22, 68, 54}
, {0, -52, -22}
, {-6, 33, 39}
, {3, 30, -16}
, {-9, 53, 65}
, {-41, -12, 54}
, {-33, -37, 8}
, {-9, 43, -19}
, {-40, -20, 51}
, {12, -15, 5}
, {-26, 32, -25}
, {35, 57, 79}
, {-27, -13, 39}
, {36, -32, 26}
, {-2, -49, 22}
, {-11, -39, -42}
, {48, 16, 51}
}
, {{0, 40, 57}
, {-48, 35, -28}
, {-22, 29, 16}
, {32, 53, 3}
, {-3, 23, -27}
, {0, -33, 36}
, {12, -44, 28}
, {46, 41, 39}
, {41, -3, -12}
, {-33, 11, 28}
, {1, -26, -10}
, {-11, 24, -18}
, {50, -1, 19}
, {-12, 24, -7}
, {6, 27, -23}
, {43, -9, -39}
, {-11, 9, 2}
, {-36, 35, -23}
, {-6, 2, -41}
, {59, 11, -21}
, {16, 6, 31}
, {3, 32, 53}
, {-1, -26, -36}
, {38, 1, 35}
, {-10, 52, -34}
, {54, -9, -35}
, {35, -76, -62}
, {-18, -11, 37}
, {-36, 34, -35}
, {12, 5, -50}
, {-49, -10, 15}
, {-69, -70, -43}
, {-49, -16, 30}
, {-14, -44, 13}
, {-21, -11, 34}
, {33, 4, -25}
, {27, -2, -49}
, {57, 3, 22}
, {21, 6, -45}
, {-54, -15, -4}
, {-42, 31, 41}
, {16, 3, -6}
, {-17, -29, 8}
, {48, 32, -12}
, {-48, -28, 4}
, {26, -49, -6}
, {50, -8, -43}
, {-3, -4, -62}
, {-44, 34, 27}
, {-21, -14, -47}
, {54, 9, 11}
, {36, -41, 36}
, {-5, -23, 44}
, {39, 0, 37}
, {16, -5, 31}
, {-53, 35, -24}
, {-18, 27, 26}
, {-23, 17, 6}
, {-65, 4, 12}
, {-34, 40, 49}
, {-28, 20, 35}
, {29, 18, -54}
, {36, 39, 3}
, {0, 41, 6}
}
, {{9, 16, -9}
, {39, -1, 28}
, {-39, -51, -31}
, {39, 32, 35}
, {-21, 49, -52}
, {15, -9, 0}
, {-34, 38, 34}
, {-6, 13, 52}
, {0, -4, -50}
, {32, 9, -39}
, {26, 61, 60}
, {20, 51, -20}
, {10, -35, -35}
, {-76, -2, 41}
, {-46, 24, -36}
, {33, -12, 50}
, {-19, 26, 2}
, {8, -35, 14}
, {-48, 48, 17}
, {40, -29, -6}
, {-47, -38, 26}
, {19, -41, -21}
, {43, 38, -38}
, {48, 7, -38}
, {-31, 63, 41}
, {57, -17, -50}
, {21, -9, -5}
, {-32, -8, -17}
, {21, 47, 30}
, {-31, 41, -28}
, {-26, -30, -23}
, {51, -9, 42}
, {39, -16, 6}
, {-25, -37, 17}
, {-9, 23, 48}
, {1, -22, 12}
, {48, -37, -46}
, {-45, -45, 37}
, {-25, -20, 38}
, {-27, -29, -7}
, {-43, 52, -33}
, {38, -8, -10}
, {-33, -5, 41}
, {-8, 47, 13}
, {38, -55, -16}
, {-35, -21, -48}
, {38, -17, -11}
, {-32, -1, 16}
, {-28, 33, 8}
, {27, -2, 71}
, {-16, -45, -47}
, {-19, 51, -31}
, {-12, 31, -11}
, {-30, 30, 26}
, {-24, 26, -30}
, {38, 14, -25}
, {-26, 35, 19}
, {12, -8, -16}
, {3, 13, 35}
, {52, -22, 10}
, {-5, -15, -12}
, {1, -3, 13}
, {-46, -24, 20}
, {21, -37, 36}
}
, {{-18, -15, -5}
, {38, 3, 56}
, {0, 51, -38}
, {-16, -43, -25}
, {40, 1, 24}
, {36, -53, 27}
, {21, 21, -10}
, {-51, -37, -5}
, {27, 42, 2}
, {-25, -38, -24}
, {-47, 33, -17}
, {-32, -36, 17}
, {4, 3, 24}
, {18, 25, -17}
, {49, 33, -16}
, {-30, -37, 45}
, {-25, 10, -19}
, {-28, 55, -29}
, {45, -19, 43}
, {41, -38, 39}
, {-16, 3, 45}
, {-17, -20, 22}
, {-21, -3, -48}
, {-11, -1, -19}
, {32, 18, 30}
, {27, 29, 36}
, {-4, -9, 45}
, {-35, 7, -31}
, {54, 14, 23}
, {-29, 7, 53}
, {49, 20, 15}
, {33, 7, -19}
, {-7, 26, 2}
, {57, 8, 29}
, {-24, 29, 12}
, {-18, -2, 4}
, {-2, -28, 31}
, {11, 33, 31}
, {-11, -47, -15}
, {61, 51, 3}
, {23, -25, -48}
, {-36, -39, -22}
, {13, -21, -15}
, {13, 39, 8}
, {37, 47, -26}
, {-48, -8, 31}
, {-46, 10, -34}
, {16, 54, -27}
, {56, -19, 19}
, {29, -52, -7}
, {43, 23, 5}
, {-23, 27, 14}
, {-50, -53, 36}
, {27, 19, 22}
, {47, 33, 49}
, {22, 1, -3}
, {-35, -9, -25}
, {0, -14, 46}
, {61, 24, 18}
, {-49, 39, -28}
, {23, -50, -10}
, {27, -25, -11}
, {-38, 35, -49}
, {-49, -9, -3}
}
, {{-57, -40, 0}
, {-42, -20, -34}
, {39, -33, 29}
, {64, -11, 47}
, {54, 0, -26}
, {47, -42, 45}
, {-44, 51, -36}
, {21, -38, 35}
, {-23, 57, -16}
, {8, -48, -34}
, {-30, 4, 12}
, {-40, 17, -7}
, {-5, -20, 39}
, {55, 56, -21}
, {0, -12, -38}
, {-37, -37, -35}
, {-7, -17, 15}
, {-55, -5, -52}
, {41, 32, -49}
, {-47, 20, 53}
, {9, 33, -42}
, {-41, -19, 20}
, {50, -32, 20}
, {-3, -47, 2}
, {57, 38, 29}
, {-64, 15, -17}
, {-29, 47, 45}
, {53, 47, -27}
, {-46, 2, 1}
, {-6, -54, -44}
, {-35, 28, 19}
, {-2, -25, 12}
, {4, 55, -3}
, {56, 59, 53}
, {-33, -50, -39}
, {-25, 29, -27}
, {37, 20, 36}
, {38, -2, 0}
, {24, 36, -6}
, {-19, 37, 11}
, {51, 47, -35}
, {-22, 7, 32}
, {52, 17, -41}
, {-10, -4, -35}
, {37, -13, 50}
, {42, -39, -51}
, {-17, -14, 54}
, {21, 13, -35}
, {19, 34, -9}
, {-1, 8, -46}
, {-37, 34, 30}
, {-56, -22, 7}
, {53, 6, 41}
, {3, 37, 44}
, {-46, 30, 21}
, {46, 41, 47}
, {16, -34, 53}
, {23, -45, -10}
, {13, 59, 46}
, {49, 44, -12}
, {-32, -36, -23}
, {-10, -29, -14}
, {-19, 7, 31}
, {-23, -41, -24}
}
, {{24, -21, -29}
, {-20, 50, -40}
, {48, 7, -5}
, {41, 30, 46}
, {-20, -40, -24}
, {58, -4, 7}
, {-12, -44, -33}
, {29, -22, 19}
, {-2, 9, 0}
, {-50, -8, -10}
, {11, -34, -45}
, {-48, -55, -30}
, {-10, -26, 46}
, {42, 16, -5}
, {27, -44, -8}
, {6, -27, -20}
, {-3, 57, -10}
, {-14, 8, 25}
, {-2, 37, -18}
, {8, -27, -43}
, {11, 7, 19}
, {51, -2, 50}
, {0, -16, 39}
, {18, 0, -10}
, {12, 36, 44}
, {25, 35, -29}
, {-38, 12, 14}
, {-5, -47, -6}
, {-6, 4, 12}
, {-24, -51, -5}
, {-24, -41, -17}
, {-18, 48, -41}
, {9, 0, 18}
, {0, -43, 26}
, {7, 17, -10}
, {27, -17, -23}
, {-36, 22, -20}
, {15, -40, 5}
, {0, -16, -30}
, {-35, 0, -30}
, {-21, 22, 12}
, {-27, -34, 24}
, {45, 18, -4}
, {22, -51, -1}
, {7, -53, 44}
, {17, -37, -25}
, {-26, -39, 0}
, {-2, 26, -14}
, {37, 27, 24}
, {3, -4, 19}
, {-40, 52, -13}
, {-71, 29, 30}
, {-38, 0, 0}
, {58, 43, 44}
, {-1, -27, -34}
, {1, 19, -17}
, {35, -23, -39}
, {-50, 27, 15}
, {8, -27, 23}
, {-19, 45, 7}
, {27, 33, 26}
, {-8, 0, -30}
, {33, 47, -15}
, {43, -13, -4}
}
, {{-10, -6, -12}
, {-15, -24, -8}
, {-7, -9, -31}
, {18, 4, -32}
, {20, -31, 20}
, {-44, -55, -4}
, {14, -8, -3}
, {44, -44, 46}
, {10, 32, -35}
, {-4, 6, 38}
, {28, -28, -4}
, {-13, 29, 29}
, {41, 28, -10}
, {2, 60, 55}
, {25, 3, 39}
, {-48, 8, -21}
, {-3, 3, -32}
, {2, 65, 29}
, {33, 33, -13}
, {-3, -38, -19}
, {10, -1, 9}
, {-50, -50, -20}
, {-20, 45, 18}
, {-28, -42, -22}
, {-6, 0, -50}
, {-33, 23, -34}
, {-38, 0, -26}
, {-36, 33, 4}
, {-52, 21, -62}
, {-29, 42, -15}
, {50, 52, -8}
, {-16, -24, 5}
, {5, 15, 52}
, {48, 23, 46}
, {50, 57, 4}
, {21, -47, 33}
, {12, 34, 43}
, {42, 6, -27}
, {-18, 32, -37}
, {-22, 20, 1}
, {46, -16, -7}
, {-36, -21, -67}
, {-14, 6, -25}
, {-38, -26, -16}
, {4, 15, 28}
, {-58, 8, 47}
, {-45, -17, 20}
, {-3, -19, 7}
, {-2, 36, 12}
, {-21, 15, -26}
, {-48, 11, -20}
, {38, 4, -2}
, {49, -57, 14}
, {-41, 14, -24}
, {53, -19, 13}
, {27, 7, 3}
, {-49, 10, 29}
, {-28, -26, 4}
, {-4, 8, 41}
, {-19, 16, 35}
, {23, -30, 6}
, {-5, -3, -52}
, {20, 26, 9}
, {-44, -21, -19}
}
, {{-37, 55, 45}
, {-32, -47, -16}
, {10, 19, -33}
, {-48, 22, 29}
, {43, 16, 51}
, {24, 24, 54}
, {49, 26, 29}
, {27, 40, 32}
, {-33, 50, 54}
, {48, -22, -27}
, {10, -54, -37}
, {22, -43, -16}
, {-10, -52, -47}
, {16, 9, 49}
, {-40, 27, -38}
, {-5, 54, -29}
, {-34, -20, 11}
, {-6, -7, 34}
, {-38, -32, -3}
, {-43, -35, 10}
, {19, 51, 17}
, {46, 21, -22}
, {33, -47, 39}
, {10, -33, -45}
, {-25, 4, 37}
, {-51, -18, 2}
, {-39, 2, -59}
, {-31, -8, -48}
, {-41, 9, -61}
, {-38, -44, -36}
, {12, 32, -7}
, {38, -46, 13}
, {2, -42, -51}
, {-46, -56, -41}
, {5, -26, -17}
, {-7, -37, 15}
, {20, -33, -24}
, {42, 58, -38}
, {-23, -8, -34}
, {49, -4, 33}
, {12, -28, -48}
, {-22, -44, -22}
, {34, 39, 43}
, {52, 23, -19}
, {38, -27, -37}
, {30, 54, -25}
, {-32, -51, 40}
, {20, -4, -28}
, {24, 20, -6}
, {5, 7, -2}
, {44, 18, -46}
, {-24, 23, -28}
, {-21, -18, 15}
, {-36, -11, -21}
, {6, 35, -6}
, {-17, -44, -38}
, {37, -1, 35}
, {-2, 20, -22}
, {-19, -12, -1}
, {17, -17, 30}
, {31, 17, 13}
, {27, -34, 23}
, {39, 8, -21}
, {45, 52, -40}
}
, {{24, 26, 9}
, {28, -17, 47}
, {32, -43, -3}
, {29, -56, 44}
, {15, -33, -38}
, {-43, -59, -3}
, {-17, -9, -1}
, {43, -39, 24}
, {-16, 7, 18}
, {14, 60, -37}
, {-5, 14, 10}
, {-30, -5, -14}
, {13, -24, -39}
, {40, -7, 29}
, {-40, -36, -12}
, {-46, 21, 39}
, {12, 25, 32}
, {27, 36, 27}
, {8, -28, 32}
, {21, -50, 37}
, {21, -37, -32}
, {43, 7, 17}
, {1, 28, -20}
, {-11, 56, -22}
, {-1, -39, -4}
, {-45, -18, 3}
, {32, -23, 2}
, {-14, 2, 14}
, {39, -45, 26}
, {34, -9, -42}
, {-15, -3, -11}
, {-30, 52, 2}
, {4, -40, -48}
, {-7, -46, 15}
, {37, 63, 23}
, {5, 14, -41}
, {49, -15, -16}
, {45, 29, -14}
, {-9, -27, -26}
, {-8, -21, 27}
, {9, 12, 7}
, {32, -33, -20}
, {0, 39, 22}
, {24, -6, -8}
, {41, 13, 48}
, {-14, -11, -28}
, {-9, 2, -51}
, {36, -11, -12}
, {-12, 1, 56}
, {-20, -57, -8}
, {-25, 0, 42}
, {-27, 22, -13}
, {32, 3, -19}
, {-41, -9, 21}
, {45, 28, -46}
, {28, -27, -21}
, {52, 34, 11}
, {-19, 30, -36}
, {-44, 5, -23}
, {17, -40, 1}
, {46, 34, 36}
, {28, 22, -42}
, {-34, -6, -41}
, {-8, 12, 47}
}
, {{-34, -21, 10}
, {24, -51, -43}
, {45, 48, 51}
, {-24, 40, 1}
, {10, -10, -40}
, {16, 1, 47}
, {6, -17, 30}
, {9, 10, -29}
, {-30, 29, -23}
, {20, 30, 53}
, {-17, 0, -14}
, {-39, -11, 47}
, {18, -9, -42}
, {26, -55, -51}
, {29, 8, -50}
, {37, -11, -22}
, {-30, 18, -21}
, {35, 12, 24}
, {6, 49, -2}
, {14, -23, -2}
, {28, 11, -28}
, {-18, -24, 6}
, {-26, 5, -21}
, {-1, 2, 65}
, {63, 57, 2}
, {37, 14, -14}
, {46, 22, 21}
, {11, 36, -29}
, {34, 38, 25}
, {32, 23, -5}
, {-46, -26, 8}
, {12, -12, 23}
, {27, -47, 42}
, {23, -15, 32}
, {-25, 53, 62}
, {-12, 0, -12}
, {-32, -20, -35}
, {-28, -20, -31}
, {0, 61, -16}
, {36, 16, -64}
, {-22, 22, -27}
, {37, -12, -62}
, {-55, -38, 18}
, {48, 13, -10}
, {-40, -20, 18}
, {2, 21, -10}
, {1, -16, -57}
, {53, 35, 38}
, {24, 64, -25}
, {41, -46, 39}
, {11, 0, -26}
, {35, -35, 45}
, {-32, -32, 0}
, {-26, 20, 59}
, {4, 10, 33}
, {-23, 25, 32}
, {-44, -30, -9}
, {37, 51, 35}
, {28, 38, -22}
, {29, 1, 14}
, {-43, 35, 31}
, {26, -18, 36}
, {-18, 37, 43}
, {-55, 2, -3}
}
, {{1, -14, -33}
, {-37, -24, 27}
, {14, -32, -5}
, {-14, -5, -22}
, {21, -18, 20}
, {0, -13, -27}
, {-51, 39, -16}
, {21, -40, 24}
, {-34, 23, 37}
, {-56, 12, 12}
, {0, -42, 39}
, {-45, -29, -42}
, {-29, -11, 45}
, {7, -9, -28}
, {1, -27, 48}
, {-42, 23, -37}
, {-51, -24, -45}
, {-43, 15, 5}
, {-15, 53, 10}
, {-13, 10, -7}
, {-25, -30, 6}
, {-28, 8, 26}
, {17, 21, -17}
, {-59, 4, -36}
, {-21, 15, 45}
, {-53, 33, -50}
, {-31, 6, 25}
, {47, 46, 55}
, {-37, -57, 33}
, {21, 22, 7}
, {23, -5, 10}
, {44, -10, 28}
, {1, 30, 57}
, {-14, 19, 58}
, {-40, -21, 38}
, {-6, 22, 42}
, {-7, 38, 19}
, {-22, -18, 40}
, {24, 42, 11}
, {-32, -18, 50}
, {13, -13, -38}
, {2, 18, 33}
, {27, 3, -38}
, {-51, 39, 24}
, {-7, 30, 49}
, {9, 31, 23}
, {15, 8, 12}
, {-70, -40, 29}
, {-22, 11, -36}
, {41, -8, 7}
, {51, 54, 12}
, {-36, 30, -15}
, {1, 26, -25}
, {-19, 11, -28}
, {-46, 23, 9}
, {-20, 15, 11}
, {-50, -17, 44}
, {-59, -27, 1}
, {-32, 12, -4}
, {-18, 1, 52}
, {12, -13, -33}
, {1, -26, 54}
, {4, 15, -29}
, {44, -39, -7}
}
, {{-3, -32, 14}
, {43, 26, -39}
, {3, -25, -7}
, {19, 38, -35}
, {-2, 48, 45}
, {-26, 20, 9}
, {21, -37, -42}
, {-4, -27, 2}
, {-46, 26, 48}
, {-7, -37, -49}
, {-35, 26, 34}
, {-22, -28, 32}
, {-35, 10, 37}
, {29, 35, -26}
, {-45, 43, -14}
, {-49, 8, -25}
, {-8, -53, 34}
, {-59, -60, -42}
, {34, 29, 53}
, {25, -54, 13}
, {-35, 14, -15}
, {-48, 18, 9}
, {-38, -28, -14}
, {37, -32, 29}
, {-43, -60, -30}
, {-58, -36, 11}
, {6, 11, 43}
, {-25, -2, 15}
, {-20, -51, 40}
, {27, -27, 33}
, {6, 26, 34}
, {-43, -28, 34}
, {-5, -15, -28}
, {26, 37, -24}
, {-17, -32, -5}
, {0, -31, -63}
, {-27, -8, -32}
, {-6, 41, 8}
, {45, 43, 47}
, {34, -26, 29}
, {37, -25, 10}
, {-40, -34, 34}
, {1, -18, -5}
, {46, 37, 25}
, {-9, -11, 49}
, {-37, 32, -8}
, {-32, 29, -29}
, {-10, -31, 1}
, {52, 9, 35}
, {-10, -5, -11}
, {25, 22, -4}
, {6, 27, -53}
, {-23, -11, 47}
, {16, 22, -56}
, {-10, 38, 40}
, {-52, -11, -1}
, {27, 11, 38}
, {-71, -17, 34}
, {36, 52, -28}
, {-11, -21, -35}
, {37, -43, 48}
, {-35, 55, -14}
, {-7, 8, 34}
, {40, 41, 44}
}
, {{48, 16, 7}
, {18, -46, -38}
, {52, -23, 47}
, {-35, 22, -3}
, {-50, -17, 52}
, {18, -12, -34}
, {-37, -27, -55}
, {38, -26, 6}
, {19, 21, 1}
, {0, -22, 53}
, {-33, -23, -20}
, {29, 38, 9}
, {-30, 20, -50}
, {-3, -33, -2}
, {-42, 40, -18}
, {-35, -5, 39}
, {13, 4, -15}
, {-5, -49, -54}
, {-13, -18, 6}
, {51, 26, -50}
, {0, 0, 49}
, {20, -24, -11}
, {-53, -25, 42}
, {0, 55, -29}
, {-27, 2, 26}
, {38, 60, 42}
, {2, -36, 16}
, {40, -26, -45}
, {39, 39, 38}
, {-14, -32, -22}
, {-55, 1, -13}
, {-45, -45, -15}
, {8, 0, -11}
, {-35, -29, -24}
, {-39, 18, 19}
, {-56, -52, 25}
, {-9, -8, 18}
, {-24, 29, 38}
, {51, 15, 39}
, {13, 4, -3}
, {-31, 12, -48}
, {29, 55, 16}
, {-24, -21, -40}
, {-44, -26, -53}
, {-3, 11, 27}
, {-42, 43, 39}
, {0, 30, -18}
, {-27, 50, 9}
, {-33, -21, 51}
, {0, 2, 33}
, {-31, -5, -44}
, {-25, -43, -14}
, {5, 39, -4}
, {10, 52, -25}
, {-55, -54, -4}
, {13, -26, -35}
, {6, 47, 49}
, {5, 12, -13}
, {35, 26, 40}
, {8, 16, -2}
, {22, -42, -15}
, {11, 47, 24}
, {12, -19, -22}
, {20, -25, -15}
}
, {{20, 33, -25}
, {3, 22, 6}
, {-10, 54, -32}
, {-39, 22, -36}
, {0, -45, 4}
, {-30, 33, -18}
, {-8, -5, 27}
, {-16, -13, -10}
, {-26, 10, -21}
, {-14, -34, 27}
, {51, 5, -2}
, {1, 50, 37}
, {32, 19, -39}
, {8, 43, 33}
, {39, 14, 26}
, {28, 45, 15}
, {25, 12, -37}
, {-51, 9, -20}
, {-8, 28, 23}
, {-15, 11, 1}
, {40, 43, -41}
, {26, -11, 29}
, {36, 15, 19}
, {-24, -44, 14}
, {-24, 29, 3}
, {29, -38, -4}
, {11, 20, 29}
, {-51, -14, -2}
, {-40, 3, 47}
, {25, 35, -1}
, {-35, -31, -18}
, {34, -32, 45}
, {38, 21, -3}
, {-24, 50, -43}
, {4, -38, 28}
, {-31, -10, -51}
, {13, 36, 16}
, {-20, 9, -2}
, {-42, 10, 25}
, {-2, 39, -11}
, {-28, 29, -26}
, {-13, -16, -47}
, {17, 41, 4}
, {-44, -39, -2}
, {2, 0, 21}
, {40, 0, -40}
, {35, 19, 11}
, {-32, -10, -36}
, {43, 7, 20}
, {-51, -29, -43}
, {-23, -36, 39}
, {18, 7, -57}
, {-18, 37, -19}
, {-32, -47, -12}
, {8, 41, -24}
, {-54, 29, -5}
, {39, -35, 42}
, {-29, 17, 18}
, {-6, 6, 37}
, {-40, 3, -20}
, {9, 43, -40}
, {45, 11, 47}
, {28, -20, -43}
, {17, 7, -55}
}
, {{30, -60, 18}
, {-16, 13, 10}
, {35, 19, -14}
, {39, -35, 60}
, {-7, -19, 55}
, {52, 34, 2}
, {33, 52, -47}
, {5, -3, -13}
, {28, -44, 53}
, {38, 0, -47}
, {53, 19, -32}
, {2, -45, -4}
, {9, -48, 37}
, {0, 42, -25}
, {-17, -54, 8}
, {-19, -45, -11}
, {3, -11, 55}
, {36, 16, -10}
, {-44, -37, -1}
, {6, 32, -17}
, {8, 43, -27}
, {-30, -28, -7}
, {42, 13, 43}
, {29, -49, 13}
, {11, -12, 6}
, {-32, 59, 5}
, {-17, -8, -17}
, {-28, 18, -49}
, {-17, 44, 41}
, {-32, -30, -17}
, {-5, -22, 0}
, {-42, 7, -76}
, {55, 12, -41}
, {39, -29, 26}
, {-47, -31, -13}
, {40, -17, -34}
, {22, -48, -20}
, {-30, 50, -51}
, {-53, 9, -8}
, {30, 22, -27}
, {34, 44, -43}
, {11, 8, -1}
, {29, -33, 27}
, {-30, 31, -42}
, {-39, 28, -14}
, {16, -48, -6}
, {-18, -12, -22}
, {30, -32, 62}
, {50, -59, 20}
, {30, 0, 6}
, {-38, 51, -10}
, {-10, -2, 37}
, {-46, -49, -29}
, {-32, 49, 34}
, {5, 33, -36}
, {-11, 47, -24}
, {-21, -8, -41}
, {14, -20, -35}
, {-33, 1, -17}
, {45, -26, 31}
, {23, 43, 3}
, {-34, 46, 29}
, {29, 13, -11}
, {0, -5, 4}
}
, {{15, -8, -55}
, {44, 7, 0}
, {-63, -61, 8}
, {-24, -43, 7}
, {35, -38, -17}
, {-7, -62, -61}
, {-4, 5, 29}
, {22, -34, 50}
, {-55, -31, 18}
, {-49, 55, 53}
, {48, 4, -15}
, {-26, 29, -35}
, {26, -48, 24}
, {-42, -14, 46}
, {26, -29, -8}
, {-24, 16, -13}
, {2, -3, 46}
, {-48, -6, -70}
, {0, 34, 40}
, {-20, 21, 49}
, {-13, -31, 21}
, {11, 10, 2}
, {-23, 17, -20}
, {-1, 31, 8}
, {-24, -35, -39}
, {0, -47, -42}
, {-66, -36, -4}
, {31, -8, 47}
, {-25, 26, -5}
, {-23, -11, 46}
, {31, -7, 26}
, {27, 27, -5}
, {40, -11, 10}
, {26, -29, -15}
, {49, -5, 43}
, {-51, -25, -5}
, {-10, 57, -30}
, {-33, 29, -5}
, {-30, -20, -4}
, {0, 24, 11}
, {3, -26, 15}
, {-10, -49, 27}
, {-37, 32, 20}
, {-49, -34, 29}
, {2, -29, 46}
, {-21, 28, -14}
, {-34, -29, 34}
, {27, -19, 1}
, {-1, 3, -39}
, {-2, 33, 31}
, {-23, 43, -37}
, {-7, -43, 42}
, {21, -11, -31}
, {2, -15, 24}
, {27, -36, -27}
, {-22, 30, -32}
, {40, 46, -3}
, {26, 12, -17}
, {-38, -36, -59}
, {-40, 9, 37}
, {36, -34, -11}
, {6, 53, -42}
, {61, 46, -29}
, {-24, -41, -1}
}
, {{-38, -32, 40}
, {-10, 33, 22}
, {12, -55, 0}
, {15, 0, -41}
, {-58, -25, -29}
, {-39, -44, -44}
, {38, -16, -13}
, {-22, -34, -6}
, {10, 21, 32}
, {-29, -42, 53}
, {-31, 12, 3}
, {-19, 14, 5}
, {-43, 20, -49}
, {1, -24, -38}
, {28, -37, 39}
, {46, -6, 4}
, {31, 60, 52}
, {-21, -78, -35}
, {-17, 40, 7}
, {27, 42, -32}
, {-7, 52, 13}
, {-17, -6, 5}
, {20, 16, -19}
, {33, -16, -3}
, {-32, -17, 12}
, {-49, -3, 9}
, {2, -8, 0}
, {-30, -43, -23}
, {40, -37, -32}
, {-12, -18, 7}
, {-55, -51, -47}
, {-9, 15, -18}
, {-22, 24, 32}
, {17, 29, 22}
, {-24, 11, 23}
, {-61, -59, 9}
, {-23, -17, 20}
, {24, 49, -37}
, {47, 18, 42}
, {-9, -24, -48}
, {-39, 6, -27}
, {21, 1, 36}
, {18, -15, 3}
, {20, -14, 20}
, {-12, -14, 47}
, {-1, -23, 38}
, {-31, -30, -46}
, {15, 25, 16}
, {39, -28, 36}
, {-38, 53, -36}
, {1, -42, 6}
, {36, -28, 55}
, {-19, 20, -31}
, {-41, -25, 8}
, {30, -16, -27}
, {-69, -30, -32}
, {1, -30, 53}
, {51, -14, 52}
, {-9, -33, 0}
, {50, 32, -4}
, {5, 23, 30}
, {-19, 35, 33}
, {52, 47, -25}
, {-31, 0, 8}
}
, {{-25, 30, 36}
, {31, 38, -44}
, {0, -37, 18}
, {-15, -50, -36}
, {51, -2, 5}
, {41, 27, 1}
, {40, -17, 51}
, {-37, 2, -17}
, {-48, -51, 26}
, {15, 22, -28}
, {20, 12, 45}
, {-6, -26, -13}
, {50, -44, -2}
, {17, -31, 50}
, {-18, 12, -26}
, {29, -15, 8}
, {-33, 37, 25}
, {-45, -29, 28}
, {14, -6, 8}
, {21, 32, -2}
, {-14, 49, -8}
, {-1, 6, -8}
, {41, 14, -19}
, {-58, -22, 26}
, {-53, -6, 28}
, {-33, -40, 22}
, {7, -20, 22}
, {-1, 40, -22}
, {-20, -25, -30}
, {0, 7, -31}
, {17, 56, -18}
, {-11, -2, 56}
, {35, -20, 8}
, {-8, -16, -34}
, {-61, -42, -24}
, {-36, -34, 45}
, {9, 25, -16}
, {20, 24, -39}
, {8, -36, 21}
, {-8, 50, -8}
, {53, -42, -7}
, {-18, 7, 38}
, {-43, 6, 8}
, {44, -43, 30}
, {-46, 29, -34}
, {-3, 51, 35}
, {-11, -23, 4}
, {-39, -25, 60}
, {21, -5, -43}
, {53, -5, -21}
, {-38, -45, 8}
, {-18, 57, -1}
, {-44, 0, -30}
, {-31, -18, 5}
, {-22, -40, 1}
, {-19, 0, -45}
, {45, -22, -26}
, {38, 46, -1}
, {1, 20, 42}
, {-44, 20, 15}
, {30, 7, -24}
, {35, 43, 31}
, {-40, 45, -3}
, {50, 47, 0}
}
, {{19, 2, 4}
, {0, 11, -11}
, {-17, -11, -36}
, {36, 35, -63}
, {-36, -21, -28}
, {22, -11, 26}
, {6, 58, 34}
, {54, -19, 40}
, {51, -21, -36}
, {32, 37, 12}
, {29, 17, -6}
, {51, -41, -34}
, {15, 1, 32}
, {25, 51, 5}
, {47, 16, 55}
, {46, -16, 0}
, {-31, -24, -37}
, {-54, 30, -26}
, {-42, 38, -32}
, {38, -14, 49}
, {28, -5, 0}
, {-56, -61, -64}
, {-38, -20, 10}
, {13, -54, -42}
, {-43, -43, -63}
, {39, -1, 45}
, {15, 31, 47}
, {45, 43, -24}
, {-6, -8, -33}
, {-1, 42, 2}
, {-4, -28, -32}
, {47, -28, -20}
, {43, 12, 28}
, {18, 22, 25}
, {12, -19, 1}
, {-8, -50, -56}
, {46, 27, 39}
, {48, 0, 50}
, {-12, -33, -41}
, {-40, -22, -26}
, {49, -45, -45}
, {19, 52, 29}
, {22, -44, -41}
, {47, -40, -1}
, {-42, -25, 10}
, {34, -35, -27}
, {-26, 41, -50}
, {-18, -50, -10}
, {0, -65, -36}
, {-16, 9, 46}
, {22, 35, 48}
, {-2, -30, 50}
, {13, -7, -27}
, {-33, 19, -14}
, {52, -42, 47}
, {-6, -37, -54}
, {52, -19, -32}
, {-27, 39, 18}
, {-67, -2, 29}
, {-1, -4, 20}
, {43, 32, -35}
, {12, 24, 19}
, {30, 5, 25}
, {48, -5, -41}
}
, {{-36, 29, -17}
, {-6, 25, -32}
, {23, 10, 43}
, {-21, 29, -14}
, {-38, 37, 51}
, {-42, -16, 24}
, {0, 0, -3}
, {-35, 45, -8}
, {-54, 10, -7}
, {39, -10, 60}
, {-26, 54, 65}
, {-38, 38, 25}
, {18, 16, -14}
, {-5, 30, 24}
, {8, -3, 17}
, {-11, 28, -23}
, {-46, -47, -39}
, {-28, 34, -28}
, {-33, 46, 13}
, {26, -28, -7}
, {44, 11, -14}
, {-13, -8, -20}
, {-27, -43, 16}
, {19, -6, 55}
, {-31, 18, -25}
, {32, -16, -47}
, {16, -20, -16}
, {-24, -24, 15}
, {-32, -34, 24}
, {6, 61, -1}
, {-17, -2, 3}
, {0, 16, -23}
, {-42, 45, -32}
, {18, 53, 19}
, {46, 73, 20}
, {-3, -20, 14}
, {-13, 48, -13}
, {48, -23, -42}
, {-4, 19, -7}
, {-19, 31, -37}
, {-34, 14, 22}
, {-68, -54, -8}
, {-16, 7, -27}
, {10, 38, 13}
, {21, -3, 34}
, {-60, -51, -20}
, {6, 0, -46}
, {34, -43, -36}
, {64, 3, -16}
, {-30, -53, 35}
, {-42, -24, -29}
, {16, 38, 38}
, {43, -25, 30}
, {18, -28, -34}
, {24, 43, -34}
, {14, 42, 51}
, {39, -6, 11}
, {-40, 39, -27}
, {78, 8, 49}
, {49, -35, 0}
, {51, 40, -43}
, {4, -14, 0}
, {-34, 19, 45}
, {38, 35, 8}
}
, {{7, 36, -19}
, {15, -45, -28}
, {16, -33, 54}
, {3, -9, 56}
, {4, 50, -2}
, {35, 15, -13}
, {10, -4, 19}
, {56, 29, 15}
, {-22, -9, 35}
, {35, -51, -1}
, {43, 56, -23}
, {-54, -4, 37}
, {29, 40, 18}
, {11, 53, -15}
, {-34, -35, -2}
, {23, 47, -21}
, {-19, -45, 13}
, {-39, 43, 39}
, {33, -5, 17}
, {17, -41, 47}
, {-54, -25, 42}
, {-25, 30, 40}
, {46, -32, -10}
, {-44, 29, 34}
, {45, 18, 20}
, {10, 12, -4}
, {-13, -50, 21}
, {10, -10, 53}
, {-11, -21, -7}
, {-43, 16, -12}
, {-22, -7, -46}
, {-20, -32, 45}
, {-43, 11, 35}
, {31, 3, -28}
, {5, 31, -2}
, {39, 61, -32}
, {25, -12, -61}
, {3, 3, 35}
, {49, -27, 9}
, {21, -22, 32}
, {10, -6, -30}
, {22, 22, -24}
, {-16, 23, -35}
, {-10, 37, -44}
, {-40, 11, -10}
, {-41, 42, -37}
, {-17, 47, 23}
, {0, -8, 33}
, {58, -12, 24}
, {-6, 7, 26}
, {-15, -33, 12}
, {-23, -43, -41}
, {4, 37, 28}
, {-22, -24, 12}
, {31, -12, -39}
, {1, 28, -25}
, {52, -50, 29}
, {43, 31, -54}
, {34, 11, -30}
, {-15, -30, -22}
, {-44, 0, -43}
, {-40, 6, 38}
, {16, -26, -11}
, {36, -46, -39}
}
, {{24, 40, -35}
, {-14, 0, 37}
, {-38, -63, 8}
, {-9, 22, -58}
, {1, -16, 39}
, {33, -35, -51}
, {-16, 14, 10}
, {-5, 34, 15}
, {8, 8, -11}
, {37, -34, 9}
, {-20, 31, -14}
, {16, 51, 51}
, {1, 45, 5}
, {-22, -39, 4}
, {-38, -47, 44}
, {35, -20, 16}
, {-37, -4, 51}
, {51, -28, -31}
, {-10, 11, 4}
, {-34, -20, -14}
, {23, -4, -29}
, {-54, -46, 24}
, {14, 9, -23}
, {13, -23, 56}
, {-41, 41, 44}
, {25, -2, -19}
, {-18, -46, 5}
, {12, 43, 49}
, {-46, -31, -8}
, {28, 57, 9}
, {3, -42, 26}
, {-54, -24, -40}
, {-13, -34, -34}
, {33, -55, -4}
, {-15, 54, -24}
, {-38, -32, -34}
, {-6, -12, 4}
, {-19, 36, 17}
, {-1, 19, 18}
, {7, -28, -14}
, {-5, -28, 39}
, {32, 49, 2}
, {17, 0, -19}
, {-41, -15, -44}
, {-43, 40, 44}
, {-3, -11, -17}
, {-35, -21, 38}
, {-37, 30, -34}
, {43, 33, 51}
, {-26, 24, -37}
, {-32, -3, -7}
, {-1, 16, -45}
, {-35, -11, 10}
, {-38, -17, -36}
, {-16, -39, -33}
, {-19, -42, 29}
, {27, -19, 39}
, {43, 55, 48}
, {-45, 29, 5}
, {14, -20, -34}
, {25, -40, 52}
, {-3, -6, 49}
, {54, -26, -12}
, {28, -5, -52}
}
, {{-40, 16, -39}
, {-42, -17, 26}
, {29, 23, 13}
, {40, -12, 49}
, {-35, 27, 37}
, {30, 22, -51}
, {14, 20, -21}
, {-28, 23, 29}
, {-49, -19, -38}
, {20, 6, -8}
, {-7, 9, 16}
, {58, 15, -1}
, {32, -8, -31}
, {16, -24, -47}
, {11, -3, 20}
, {-54, 34, 11}
, {48, -31, 39}
, {10, -9, 13}
, {25, -6, -17}
, {-13, 20, -3}
, {4, -41, 7}
, {42, 37, 24}
, {-8, 0, 28}
, {-18, 57, -14}
, {26, -26, 33}
, {19, -17, 6}
, {35, 20, 45}
, {-31, 32, -27}
, {31, -17, 0}
, {13, -17, -45}
, {-14, -38, -48}
, {-2, 0, 12}
, {-13, -47, 45}
, {-4, 15, 30}
, {-3, 44, 15}
, {-22, -6, 0}
, {-36, 17, 30}
, {35, -46, -19}
, {61, 25, 57}
, {-22, -50, 6}
, {-2, 40, 33}
, {16, -10, -36}
, {-13, 37, -34}
, {14, -16, 20}
, {52, -34, 47}
, {50, -21, 0}
, {-20, -41, -39}
, {19, -31, 9}
, {-26, -29, -31}
, {4, 45, 21}
, {-41, -36, 9}
, {63, -24, 52}
, {30, -44, 47}
, {-37, 33, -21}
, {15, -36, -43}
, {-16, 34, 12}
, {19, 20, -28}
, {48, -8, 12}
, {-38, -27, 44}
, {36, 47, -46}
, {37, 29, 31}
, {-20, 4, -23}
, {6, -38, 1}
, {29, -35, -11}
}
, {{47, -19, -43}
, {-2, -19, 47}
, {40, 1, -2}
, {44, -21, 8}
, {5, -40, 25}
, {32, -5, -39}
, {-46, 52, 33}
, {20, -44, 4}
, {34, 41, -43}
, {28, -25, -30}
, {25, 34, 35}
, {41, 18, -43}
, {8, 0, 25}
, {-30, 41, -43}
, {10, -37, -36}
, {-24, -8, -26}
, {51, -44, -13}
, {-34, 7, 14}
, {-12, 22, -27}
, {-1, -16, 6}
, {29, 33, 18}
, {-6, 26, -56}
, {49, 30, 11}
, {2, -17, -34}
, {8, 28, 44}
, {51, -2, 48}
, {37, 69, -37}
, {50, 26, -37}
, {50, -4, -10}
, {8, -11, -48}
, {-48, -37, -40}
, {-8, 38, 71}
, {-28, 28, 31}
, {20, 51, 24}
, {22, 44, 8}
, {14, -36, -37}
, {-6, -7, 49}
, {-43, -23, -12}
, {-40, -37, 33}
, {-47, -7, 16}
, {32, 48, 2}
, {-25, -25, -12}
, {16, 32, -49}
, {-40, -1, -9}
, {-9, -42, 39}
, {25, -12, -13}
, {10, 37, -32}
, {68, 82, 4}
, {-31, -20, -34}
, {5, 0, 32}
, {-36, 5, -35}
, {1, 0, 65}
, {-18, 33, -35}
, {11, 0, 11}
, {-46, 8, -27}
, {-17, 47, 59}
, {48, 39, 30}
, {-28, 31, 70}
, {-23, 28, -23}
, {16, -32, 40}
, {-49, 0, -15}
, {42, 0, -27}
, {0, 2, 26}
, {29, -56, -13}
}
, {{55, -2, 5}
, {-6, 50, -7}
, {20, -16, 7}
, {50, 33, -23}
, {-45, -5, 2}
, {-40, 24, -38}
, {12, -26, 8}
, {26, -24, -16}
, {1, -57, -5}
, {-23, 18, 12}
, {5, -39, 5}
, {27, -29, 0}
, {-38, -23, 48}
, {-53, 33, -48}
, {-26, 4, -37}
, {-43, -16, -43}
, {-39, -14, 34}
, {6, 44, -30}
, {32, -47, -12}
, {9, -35, -3}
, {11, 33, 13}
, {-40, -52, -50}
, {-40, -2, -22}
, {10, 10, -23}
, {-40, 0, 7}
, {-33, 24, 21}
, {-53, -33, 38}
, {5, -42, -3}
, {13, 8, 12}
, {37, 2, 17}
, {-10, 16, -1}
, {-4, 14, 4}
, {-46, -8, 19}
, {-6, -21, 27}
, {-3, -41, -12}
, {-31, 0, 22}
, {-5, 13, 27}
, {21, -51, 46}
, {12, -16, 46}
, {-35, 44, 41}
, {-19, 42, -54}
, {-33, 39, -21}
, {40, 43, -2}
, {31, -31, -22}
, {49, -5, 44}
, {-27, 49, -37}
, {-13, -25, 12}
, {-21, 35, 0}
, {-29, 0, -19}
, {8, -20, -37}
, {36, -51, 26}
, {35, -35, 39}
, {-27, 8, -27}
, {-5, -41, -14}
, {-36, -43, -43}
, {33, -19, -21}
, {34, 36, 22}
, {21, 32, 55}
, {-30, 31, 19}
, {-52, 32, 31}
, {9, 53, -49}
, {37, -39, 33}
, {-21, 19, 26}
, {14, -24, 3}
}
, {{-64, -11, -2}
, {17, 26, 33}
, {45, 19, 32}
, {39, 21, 43}
, {6, 46, 25}
, {48, 53, -10}
, {23, 11, 12}
, {1, 2, 3}
, {-7, -2, -36}
, {-26, -56, -44}
, {45, -34, -20}
, {5, -15, -18}
, {-16, 7, -37}
, {10, 38, 47}
, {41, -34, 27}
, {-3, 15, 15}
, {-25, 33, 20}
, {52, -17, -19}
, {-27, 5, 18}
, {22, 18, -11}
, {-16, -40, 40}
, {-8, 5, -56}
, {-22, -45, -25}
, {22, 10, 3}
, {32, -36, -24}
, {61, -9, 8}
, {73, 16, -17}
, {-17, -14, -44}
, {15, 36, 55}
, {15, -24, -44}
, {32, -5, -19}
, {74, 28, 28}
, {-37, 9, -2}
, {-14, 21, -18}
, {26, 33, -57}
, {3, -35, 34}
, {9, 47, -5}
, {45, -13, -16}
, {24, -60, -29}
, {-18, 55, 67}
, {28, -18, 56}
, {45, 7, -31}
, {-37, -32, -40}
, {12, 44, 45}
, {0, 51, -15}
, {-20, -33, -56}
, {-34, 26, 25}
, {-22, 16, -4}
, {30, 39, 46}
, {-17, 25, -38}
, {-53, 8, 47}
, {-41, 17, -17}
, {-31, 46, -47}
, {26, -7, -31}
, {-28, -24, 6}
, {-2, 8, -34}
, {28, 50, 21}
, {-20, -31, 38}
, {47, 71, 25}
, {-32, 15, 11}
, {-12, 9, -10}
, {-39, 28, 37}
, {8, -10, 37}
, {-27, -10, -21}
}
, {{43, 46, -27}
, {-29, 50, 47}
, {-54, -52, -29}
, {9, 15, -38}
, {21, -5, 4}
, {-18, 8, -34}
, {33, 23, 37}
, {32, 13, 39}
, {-47, -9, 15}
, {37, 56, 51}
, {39, 53, 27}
, {-13, -11, 56}
, {17, -50, -48}
, {36, -11, 42}
, {32, -9, 25}
, {33, -26, -39}
, {21, 9, -41}
, {12, -8, -23}
, {8, 36, 50}
, {10, 3, 4}
, {-41, -43, -13}
, {-34, -4, -62}
, {-29, -9, -47}
, {52, 21, 5}
, {43, -49, -47}
, {37, -20, 35}
, {39, 16, -25}
, {-26, 54, 39}
, {-8, 15, 23}
, {22, 50, 48}
, {52, 52, 18}
, {2, -19, 37}
, {-22, -27, -35}
, {-8, -35, 2}
, {12, 2, 39}
, {42, 53, -33}
, {-26, 18, -24}
, {-20, 42, 38}
, {27, 10, 54}
, {0, -14, 36}
, {32, 32, 39}
, {-12, 14, 46}
, {-37, -9, -37}
, {25, 41, -42}
, {54, 39, 21}
, {-24, -40, -55}
, {-6, 41, -20}
, {-18, 47, 6}
, {-10, 16, 5}
, {49, 56, 51}
, {-8, 24, 3}
, {-19, 24, 8}
, {-1, 27, 5}
, {-48, 2, 32}
, {52, -26, -16}
, {-1, -38, -38}
, {50, 46, -20}
, {40, -14, -21}
, {62, -16, -15}
, {-32, 36, 34}
, {-27, -17, 48}
, {-29, -42, 37}
, {39, 19, 9}
, {8, -38, 21}
}
, {{45, 49, 7}
, {-34, -6, 18}
, {27, -35, 6}
, {-62, -32, -33}
, {30, 8, 39}
, {41, -51, -11}
, {-12, -31, 21}
, {54, -15, 42}
, {33, -5, 5}
, {-17, -6, 23}
, {-51, 27, -44}
, {-22, -4, -3}
, {-6, 3, 33}
, {-25, -26, -7}
, {29, -17, 40}
, {-38, 15, 41}
, {9, -33, -48}
, {-41, 22, -27}
, {-2, -33, 28}
, {41, 51, 34}
, {35, -3, 10}
, {-33, 15, 10}
, {-31, -30, -42}
, {24, 28, 43}
, {53, -34, -35}
, {-41, -51, 6}
, {38, -16, -51}
, {19, 17, -47}
, {-9, -22, 1}
, {8, 6, 26}
, {20, 51, 5}
, {-9, -6, 0}
, {-8, 37, 28}
, {31, -37, 49}
, {53, 0, -29}
, {19, -36, -51}
, {4, 26, 0}
, {36, 13, -9}
, {25, 31, -42}
, {28, -43, 28}
, {-48, 0, -18}
, {53, 50, -29}
, {25, -38, 31}
, {-1, 5, 48}
, {17, -5, 42}
, {-12, -15, -19}
, {-20, -50, 16}
, {-7, 11, 8}
, {19, -19, 51}
, {-25, 37, 51}
, {-39, -13, -15}
, {1, -30, 38}
, {-29, 50, 8}
, {25, -2, 24}
, {48, 34, 46}
, {-2, 14, 49}
, {45, 12, 43}
, {37, 3, 50}
, {19, 1, 28}
, {-46, -27, 10}
, {55, 24, -35}
, {-13, -42, 36}
, {22, 26, -50}
, {24, 28, 27}
}
, {{-48, -28, 14}
, {53, -22, -19}
, {7, 19, 17}
, {31, 32, 65}
, {57, -30, -24}
, {50, 3, 4}
, {43, 9, 32}
, {7, 50, -27}
, {19, 11, -41}
, {-39, -33, 16}
, {-7, 39, -19}
, {19, 0, 56}
, {-2, -50, -12}
, {-2, 7, -57}
, {2, -28, -6}
, {20, -42, -39}
, {23, -32, 15}
, {-60, -43, 8}
, {4, 5, 22}
, {54, -37, -26}
, {-55, -14, 26}
, {-1, 3, 44}
, {18, -10, 26}
, {13, 44, -34}
, {-11, -37, 11}
, {-5, 17, 12}
, {-21, 48, 47}
, {49, 50, 52}
, {36, 10, 18}
, {22, -51, -30}
, {-12, 3, -61}
, {20, -5, 34}
, {31, 55, 15}
, {-22, -2, 12}
, {10, 37, -17}
, {-8, -9, 6}
, {-33, -20, -24}
, {-55, -31, -34}
, {52, 9, -9}
, {16, 25, -29}
, {38, -1, -9}
, {-15, -31, 41}
, {17, 28, 43}
, {-23, -49, 16}
, {-47, -39, 41}
, {-28, -7, -22}
, {25, -40, 42}
, {1, -20, 38}
, {-24, 59, -18}
, {-35, 46, 8}
, {42, 19, 45}
, {20, -21, 36}
, {34, -36, 6}
, {13, -14, 8}
, {-29, -46, -10}
, {26, -27, -6}
, {-41, -33, 41}
, {-7, -7, -42}
, {-24, 16, -21}
, {31, -12, -45}
, {15, 17, -9}
, {18, 30, -38}
, {33, -2, 4}
, {-39, -56, -33}
}
, {{3, 10, -10}
, {-43, -10, 42}
, {0, 22, 8}
, {-61, 4, 12}
, {12, 10, -3}
, {41, -33, -27}
, {-25, 33, 43}
, {-44, -26, -31}
, {43, 9, 24}
, {18, -15, -43}
, {-4, -38, -3}
, {48, 39, -36}
, {-44, -40, 29}
, {-1, -33, 46}
, {48, 23, 23}
, {25, 12, -11}
, {13, 50, -41}
, {-15, 9, 60}
, {11, -9, -27}
, {8, 54, -24}
, {-44, -12, -16}
, {-55, 22, 30}
, {51, 3, 34}
, {-7, 14, 8}
, {14, -2, -48}
, {45, -8, 17}
, {-45, 8, -14}
, {44, 19, -33}
, {-40, 9, 14}
, {-4, -36, 24}
, {36, -7, 49}
, {-21, 15, 38}
, {-23, -20, 16}
, {24, 19, 5}
, {-34, -41, -34}
, {-2, 32, 55}
, {-32, -25, 5}
, {53, 11, 43}
, {11, -10, -48}
, {67, 25, 16}
, {51, 49, 25}
, {-3, 22, -48}
, {-17, -39, 34}
, {34, 36, 16}
, {-22, -7, 46}
, {24, -26, -17}
, {-32, -22, 26}
, {-31, -10, 14}
, {-45, -45, -27}
, {20, -16, 18}
, {42, -40, 9}
, {-28, 43, 5}
, {-27, -45, 45}
, {-51, 23, -43}
, {-9, 40, 8}
, {55, -15, 41}
, {-32, -31, -3}
, {-2, -12, 30}
, {-49, -61, -49}
, {-11, 29, -15}
, {42, 40, -30}
, {7, -33, 46}
, {-40, -23, 28}
, {36, 51, 34}
}
, {{-43, 13, -15}
, {21, -42, -11}
, {-66, 54, -5}
, {16, 12, 55}
, {-8, -13, 34}
, {-51, -28, 2}
, {-25, 0, -51}
, {34, -42, -34}
, {10, -23, -13}
, {45, 36, 25}
, {-48, 48, 25}
, {-42, -47, 48}
, {-4, 29, 44}
, {-6, -25, 28}
, {14, -9, 29}
, {-30, 31, -13}
, {47, -52, -26}
, {7, 30, 18}
, {-36, 40, 7}
, {-45, 40, 43}
, {39, -22, -8}
, {-35, 35, 10}
, {-14, 34, 10}
, {52, -24, -7}
, {-31, -38, 18}
, {13, -31, 11}
, {39, 0, -38}
, {-16, 29, -36}
, {-25, -27, -46}
, {-12, 1, 13}
, {18, -55, -21}
, {-43, -66, 13}
, {6, 9, -32}
, {-33, -12, 14}
, {55, 5, -40}
, {48, -14, 13}
, {16, -39, 16}
, {-52, 18, 2}
, {-9, 24, -30}
, {-15, 0, 37}
, {30, -25, -10}
, {19, -42, -45}
, {23, 32, -9}
, {-47, -7, 12}
, {45, 26, -2}
, {4, 42, 22}
, {42, -55, 26}
, {-30, -59, -44}
, {-19, 44, 42}
, {-33, -24, 32}
, {0, 34, -23}
, {-13, -55, -27}
, {6, 12, 50}
, {47, 26, 41}
, {-8, 18, 28}
, {47, -42, 40}
, {-39, 5, 35}
, {8, 11, -4}
, {14, 58, 38}
, {5, 11, -40}
, {-5, 36, 3}
, {27, -22, -9}
, {54, -5, 54}
, {-22, -55, -14}
}
, {{2, -28, 2}
, {2, 0, 21}
, {20, -1, 5}
, {-29, -26, 12}
, {29, -10, -20}
, {42, -1, 15}
, {31, -13, -14}
, {4, -28, -51}
, {17, 48, 18}
, {-18, -38, -55}
, {27, -2, 45}
, {-36, -52, 38}
, {28, -23, -45}
, {-1, 7, -42}
, {-47, 36, -31}
, {-8, 16, 52}
, {-3, 33, 37}
, {-55, 0, -43}
, {-15, 29, -13}
, {44, 43, 12}
, {27, -37, 18}
, {37, -19, -3}
, {-51, -37, 30}
, {35, 32, -14}
, {-49, -29, 15}
, {-41, 12, 58}
, {42, 48, -8}
, {-10, 39, -25}
, {-20, 33, 58}
, {2, -10, 5}
, {7, -57, 17}
, {25, 1, 69}
, {-11, 37, 40}
, {-18, 20, -28}
, {24, -16, 11}
, {-23, -39, -5}
, {-41, -50, -10}
, {52, -8, -13}
, {25, -8, 46}
, {-47, 34, -6}
, {-29, -20, -29}
, {-22, -13, 42}
, {-14, -45, 25}
, {17, 53, 18}
, {-7, -25, 41}
, {-11, 27, 28}
, {-10, -6, 40}
, {-31, 57, 55}
, {33, 40, 18}
, {-16, -38, 9}
, {-24, 26, 1}
, {37, 20, 8}
, {11, -25, 51}
, {44, -18, -19}
, {27, -26, -10}
, {-57, 2, 16}
, {3, -25, 26}
, {-17, -17, -15}
, {-3, -17, 16}
, {26, -46, -43}
, {-47, -45, 28}
, {-42, 7, 18}
, {25, 11, 26}
, {0, 56, -28}
}
, {{31, -3, -37}
, {-39, -36, 18}
, {-1, 20, 28}
, {8, 28, -28}
, {2, 16, 50}
, {-21, 24, -44}
, {-46, -29, 48}
, {41, -46, 28}
, {14, 5, 58}
, {28, -10, 30}
, {-29, -7, -44}
, {17, 8, -17}
, {-10, -25, 33}
, {17, 9, -27}
, {36, 40, 21}
, {-14, -47, -31}
, {-9, 37, -53}
, {-32, -6, -3}
, {10, -13, 8}
, {-15, -47, -23}
, {-8, -44, -5}
, {23, 15, 30}
, {38, 16, -23}
, {-20, 2, -23}
, {15, -4, -25}
, {31, 40, -41}
, {7, 62, 10}
, {3, 28, 45}
, {55, 18, 6}
, {-6, -9, -48}
, {-19, -31, 53}
, {17, 5, 46}
, {-28, 15, 31}
, {-30, -20, 11}
, {-41, 32, -35}
, {-26, -6, 37}
, {0, -43, -1}
, {3, 1, -43}
, {38, -22, 35}
, {21, -18, 0}
, {22, 10, -52}
, {0, -35, -39}
, {48, 20, -12}
, {-23, -6, -25}
, {50, 45, -28}
, {-9, -30, -11}
, {-58, -48, 9}
, {17, -12, -28}
, {-46, -32, -11}
, {-27, 29, -11}
, {32, 3, 25}
, {23, 1, 22}
, {-5, -5, -49}
, {12, -34, -61}
, {15, 36, -4}
, {2, -7, 24}
, {21, -16, -27}
, {40, -18, -41}
, {35, 39, 16}
, {-31, 22, 17}
, {18, 16, -3}
, {-36, 10, 35}
, {-26, -23, -9}
, {-43, 23, 0}
}
, {{8, 41, -12}
, {19, -9, 25}
, {22, 30, 25}
, {55, 51, -47}
, {-27, 53, -54}
, {19, -25, 58}
, {32, 26, -43}
, {-24, 49, -27}
, {5, 27, -46}
, {16, 33, -8}
, {6, 38, 20}
, {-14, 22, 3}
, {42, 29, 12}
, {-71, -22, -50}
, {-21, 52, 50}
, {0, -43, 22}
, {-29, 5, 29}
, {-18, -23, 65}
, {-18, 36, -24}
, {-8, 2, 30}
, {-29, 6, 52}
, {33, 2, -10}
, {42, -43, -21}
, {0, 33, 42}
, {34, -4, -19}
, {37, 26, -21}
, {16, 30, 60}
, {-11, 21, 34}
, {35, 29, 54}
, {-9, 53, 47}
, {1, 2, -45}
, {17, -24, 11}
, {3, -43, 37}
, {0, 25, 23}
, {-28, 23, -10}
, {36, -45, 21}
, {-31, -52, 40}
, {-7, -41, -3}
, {43, -17, -5}
, {-18, 33, 28}
, {-51, -38, -16}
, {26, -15, -37}
, {45, -40, 20}
, {34, 14, 44}
, {-5, -23, -15}
, {-2, 10, -11}
, {23, 53, -43}
, {66, 27, -32}
, {-65, -18, -32}
, {-20, -25, -26}
, {4, -53, -34}
, {50, 62, 34}
, {43, 25, -21}
, {-17, 38, -5}
, {-54, -5, -46}
, {24, -3, 55}
, {41, -41, -44}
, {62, 31, 11}
, {46, -1, -4}
, {-2, 2, -45}
, {3, -22, -18}
, {19, 0, 42}
, {-35, -2, 1}
, {-17, 13, 13}
}
, {{34, 16, 61}
, {50, -28, -10}
, {-7, 18, 22}
, {17, -35, 37}
, {-36, 11, -46}
, {-24, -15, 17}
, {-9, -42, 22}
, {-29, -4, 10}
, {6, -53, -14}
, {32, -28, 11}
, {55, 13, 48}
, {-27, -35, 8}
, {2, -32, -44}
, {-17, 19, -1}
, {46, 36, 23}
, {50, 9, -27}
, {38, -7, -17}
, {-21, 38, 21}
, {-13, -18, 36}
, {30, 46, 41}
, {44, -46, -26}
, {25, -34, -50}
, {-23, -47, -3}
, {36, 39, -41}
, {49, -21, -15}
, {67, 15, 65}
, {54, 58, 63}
, {59, -23, 58}
, {39, 20, 53}
, {46, 33, 61}
, {-42, 31, -32}
, {-41, 54, 0}
, {1, -25, -46}
, {-38, 0, 62}
, {-17, -38, -6}
, {-27, 42, 1}
, {27, -35, -36}
, {-39, 4, -43}
, {52, 1, -24}
, {-34, 40, -38}
, {-40, -15, 43}
, {0, 32, 34}
, {-2, -24, -40}
, {-40, 39, -42}
, {25, -5, -5}
, {-32, 29, -40}
, {25, -3, -52}
, {-20, -24, 53}
, {-50, -38, -2}
, {20, 7, 53}
, {34, -30, -1}
, {7, 49, 37}
, {16, -29, -10}
, {11, 22, 33}
, {-21, 29, 38}
, {-18, 3, 21}
, {18, 13, -28}
, {-16, 76, 73}
, {50, -36, 8}
, {2, 22, 51}
, {-11, 31, 48}
, {-46, 45, -28}
, {-33, 32, -34}
, {-12, 45, 26}
}
, {{-37, -36, 49}
, {28, 53, -29}
, {22, 59, 51}
, {-24, -10, 0}
, {44, -40, 2}
, {-16, -43, -45}
, {-36, -3, -34}
, {-42, 2, 41}
, {-26, -30, 32}
, {33, -38, -19}
, {-45, -25, 36}
, {-6, -33, -21}
, {36, -36, 19}
, {-42, -14, -18}
, {-47, 20, 3}
, {51, 31, 33}
, {21, -16, -7}
, {45, 2, -13}
, {11, -22, -6}
, {48, 21, -18}
, {-25, 39, 2}
, {-17, -36, -29}
, {-27, 28, -32}
, {-35, -27, -16}
, {51, 60, -17}
, {17, 1, 18}
, {5, -47, 27}
, {-14, 7, -10}
, {6, -30, 0}
, {-5, -37, -43}
, {28, 33, 23}
, {-56, -45, 12}
, {41, 5, 5}
, {-54, 9, 26}
, {-42, 40, -23}
, {-10, -12, 6}
, {14, -22, 2}
, {-17, -38, 27}
, {41, 55, -23}
, {3, 57, 40}
, {-14, 21, 22}
, {31, 0, -15}
, {-8, 31, 17}
, {30, -12, -41}
, {42, 20, 42}
, {-36, 41, 6}
, {-1, 9, -31}
, {42, -1, -18}
, {25, 51, 52}
, {-14, -10, -25}
, {47, 1, -6}
, {-51, -40, -42}
, {41, 35, 28}
, {81, 53, 7}
, {-19, -48, -22}
, {60, 25, 44}
, {-31, -11, -21}
, {24, 25, -25}
, {19, 77, -24}
, {-44, 30, 22}
, {30, -17, -21}
, {-11, -48, 6}
, {44, -39, 38}
, {-34, 45, -48}
}
, {{-46, 5, -23}
, {29, 45, 37}
, {5, -48, 34}
, {6, 42, -29}
, {-26, -50, 47}
, {14, -8, 51}
, {-4, -48, 49}
, {15, 42, 47}
, {60, 18, 32}
, {-58, -20, -2}
, {-13, 43, 11}
, {-51, -37, 0}
, {-32, -2, 34}
, {44, -18, -46}
, {44, -5, -62}
, {-2, -39, -25}
, {-11, -47, 13}
, {23, -15, -46}
, {29, 16, -50}
, {49, 10, -12}
, {2, -24, 3}
, {-13, 35, 33}
, {23, -49, -19}
, {-27, 19, -30}
, {-19, 0, -19}
, {-18, -22, 22}
, {-50, -2, -38}
, {0, 41, -49}
, {39, -44, -51}
, {-37, -50, -1}
, {-15, -9, -47}
, {12, -36, -53}
, {-11, 7, 26}
, {7, -44, 13}
, {15, 60, -6}
, {9, 40, -40}
, {24, 17, 28}
, {37, -8, 31}
, {-33, -4, -43}
, {-42, 5, -19}
, {-11, -22, -8}
, {12, 35, -18}
, {46, 52, -16}
, {51, 24, -39}
, {-35, 34, -16}
, {-47, 9, 15}
, {18, -10, 29}
, {-22, 29, 9}
, {-20, -14, -24}
, {-11, -57, 5}
, {36, -14, -24}
, {-37, 10, -49}
, {-25, 49, 41}
, {34, 45, 9}
, {25, 11, -42}
, {-46, 38, -51}
, {0, 29, -15}
, {27, 24, -21}
, {-50, 32, -31}
, {43, -5, -40}
, {34, -25, 44}
, {38, -36, -40}
, {28, 41, 15}
, {-40, 14, 5}
}
, {{-49, -65, -55}
, {0, -29, 28}
, {-7, 56, 25}
, {44, -37, 36}
, {-14, -29, 47}
, {46, 23, -39}
, {-15, -17, -32}
, {19, 18, -49}
, {-44, -48, 25}
, {-17, -34, 50}
, {-12, -20, 37}
, {8, 23, 37}
, {-26, 48, 32}
, {55, -26, 39}
, {26, 32, -20}
, {-20, 40, 44}
, {10, 10, -33}
, {40, 6, 15}
, {39, -15, 25}
, {2, 14, -14}
, {35, -16, 36}
, {56, 45, -12}
, {30, -47, -20}
, {-31, 41, 2}
, {-1, 24, 7}
, {-20, -40, 8}
, {21, 58, 63}
, {40, 19, -23}
, {29, -42, -20}
, {-29, -30, -26}
, {35, 14, 36}
, {10, 40, -5}
, {40, -12, -1}
, {31, 24, -33}
, {-20, -29, 1}
, {-17, 58, -25}
, {28, -36, -44}
, {44, -42, 47}
, {1, -8, -51}
, {20, 6, -44}
, {-27, 42, -45}
, {40, 47, 28}
, {48, -8, 35}
, {0, -6, -13}
, {60, -16, 5}
, {25, -32, -46}
, {-52, -34, 16}
, {2, 16, 13}
, {4, -4, 59}
, {-9, 34, 18}
, {22, 7, 12}
, {-18, -11, -51}
, {14, 46, 15}
, {36, -34, -29}
, {51, -25, 16}
, {-2, 29, 16}
, {24, -42, -12}
, {-27, 3, -38}
, {-16, 71, 32}
, {-30, 28, -51}
, {-4, 20, 2}
, {50, -10, -25}
, {-27, 24, -44}
, {-48, -2, -8}
}
, {{-48, 12, 7}
, {44, 29, 13}
, {1, 43, 20}
, {-12, 7, 1}
, {-23, 18, -22}
, {22, 18, 37}
, {-24, 15, -20}
, {-44, 39, 36}
, {-17, -12, 25}
, {-16, -23, -30}
, {-51, 21, -48}
, {7, -29, -14}
, {-22, -47, -30}
, {-22, -29, 51}
, {2, -37, 7}
, {16, 3, 38}
, {-33, 43, 48}
, {-51, -40, 21}
, {27, 15, -25}
, {-14, 33, -10}
, {-37, 0, 50}
, {36, 5, 10}
, {-25, 14, -12}
, {-62, -28, -54}
, {36, 22, -37}
, {-2, -50, -39}
, {29, -8, -2}
, {-30, 49, 32}
, {3, -10, -7}
, {7, -43, 39}
, {-33, 0, 15}
, {1, 10, 8}
, {50, -52, 41}
, {31, -7, -43}
, {-44, -58, 32}
, {-35, 55, -36}
, {7, -11, -43}
, {-35, -7, 19}
, {-3, -12, -51}
, {43, 57, 26}
, {36, 11, -20}
, {-6, 33, 2}
, {-46, 37, 24}
, {37, 6, -5}
, {19, -34, 0}
, {45, -43, 60}
, {-7, -27, 30}
, {-10, -53, 26}
, {-15, 9, -5}
, {-46, 3, -54}
, {47, -35, 6}
, {-36, -41, 13}
, {0, 13, 40}
, {46, -2, 20}
, {-22, 25, 49}
, {-27, 13, -54}
, {22, -45, -26}
, {13, -12, 14}
, {-36, -32, 3}
, {29, -14, -42}
, {-46, -24, 16}
, {13, 28, 35}
, {48, 53, -29}
, {37, -4, 39}
}
, {{-23, -1, -38}
, {8, -48, 28}
, {-21, -5, -16}
, {9, 17, -34}
, {4, 55, 26}
, {19, 60, 14}
, {-11, 31, 0}
, {-9, -23, 2}
, {32, 14, 33}
, {-59, 14, -20}
, {8, 21, 9}
, {48, 27, 13}
, {9, -9, -51}
, {30, -18, 16}
, {51, -2, -45}
, {-16, -40, 29}
, {34, 19, -7}
, {37, 11, -22}
, {-44, -43, -14}
, {19, -2, -46}
, {-6, -27, 41}
, {-4, -18, 11}
, {-16, 9, -34}
, {-16, 36, -38}
, {2, 10, 9}
, {10, 50, 24}
, {19, 57, -5}
, {18, 41, 23}
, {-8, 12, -25}
, {-51, 29, -28}
, {18, -23, -31}
, {-13, 75, 22}
, {36, 0, -2}
, {-45, -28, -6}
, {-37, 27, 15}
, {-44, -21, 6}
, {-56, -35, 1}
, {14, -47, -8}
, {29, 5, 1}
, {-37, 5, 40}
, {54, 14, -3}
, {-23, 48, 29}
, {37, 45, -26}
, {19, 57, -44}
, {21, -51, 28}
, {-7, 10, -14}
, {-29, 16, -15}
, {-9, 63, 33}
, {9, 13, 4}
, {-32, 23, 6}
, {50, -8, -14}
, {50, 12, 48}
, {29, 8, 21}
, {50, -18, 8}
, {-44, -3, -48}
, {-15, 23, 27}
, {20, 43, -49}
, {30, 23, 42}
, {3, 32, 28}
, {-10, -22, 43}
, {11, -50, 45}
, {4, -49, 47}
, {15, -53, 4}
, {40, -45, -30}
}
, {{-17, -5, 13}
, {-23, 44, -29}
, {35, -26, 18}
, {-11, -23, 51}
, {-26, -44, -7}
, {10, -60, 6}
, {-25, 19, -37}
, {4, 5, -25}
, {-24, 48, -18}
, {46, -8, -18}
, {-34, 35, -37}
, {-34, 48, 46}
, {-14, 4, 6}
, {21, 21, -19}
, {-36, -47, 26}
, {30, -49, 43}
, {25, -50, -60}
, {-25, -12, -42}
, {35, 47, 16}
, {23, -25, 45}
, {29, -10, -6}
, {-10, -34, -38}
, {18, -13, 37}
, {-2, -31, -31}
, {-39, -38, 49}
, {-44, -53, 15}
, {-26, 18, 60}
, {-2, -3, 24}
, {-45, 17, 2}
, {28, 18, -3}
, {-2, 44, 35}
, {-40, -38, 3}
, {38, 52, -43}
, {-34, 6, -2}
, {67, 7, 11}
, {31, -4, -6}
, {7, 21, -5}
, {-42, -2, 10}
, {12, -41, -44}
, {-31, -37, -11}
, {32, 39, 11}
, {48, -2, 33}
, {6, 28, -33}
, {23, 43, 27}
, {56, 7, 25}
, {-8, -17, -31}
, {21, 31, 46}
, {0, -36, -36}
, {0, 6, 51}
, {-34, 33, -10}
, {-11, 20, 36}
, {-69, -61, -12}
, {-20, 18, 54}
, {-19, -42, -26}
, {34, 40, 27}
, {-7, -42, 35}
, {19, 1, -43}
, {3, -53, -47}
, {-9, -18, 10}
, {-8, 0, -43}
, {-26, -8, -2}
, {-1, -4, 8}
, {54, 49, 38}
, {27, -16, -5}
}
, {{-10, 36, 3}
, {4, 17, -43}
, {86, -21, -11}
, {-2, -16, -32}
, {1, -13, -30}
, {27, 15, -33}
, {52, -32, -24}
, {14, 51, -39}
, {25, -18, 66}
, {-8, -22, -2}
, {49, -54, -9}
, {39, 47, -17}
, {13, -44, 33}
, {-23, 4, 68}
, {-34, 2, 30}
, {-29, 63, 21}
, {48, 35, 8}
, {-7, -41, 33}
, {-12, -49, -17}
, {60, -38, 36}
, {56, -20, 46}
, {-6, -31, -11}
, {-29, -17, 20}
, {23, 41, -45}
, {-32, -22, -22}
, {58, 47, -7}
, {-36, 12, 10}
, {-8, 35, 45}
, {-18, -25, -10}
, {-48, 3, -58}
, {-5, -2, 51}
, {17, -63, 38}
, {-1, -27, -29}
, {-12, -20, -17}
, {-22, 13, -4}
, {37, 54, 11}
, {-35, 53, 23}
, {59, 33, 55}
, {-37, 48, 16}
, {44, -41, 7}
, {-28, 54, 53}
, {11, -29, 14}
, {-12, -37, -1}
, {50, 51, 26}
, {13, -59, 47}
, {46, 0, -39}
, {31, 7, -34}
, {-37, 33, -57}
, {11, 43, 24}
, {-35, 34, -33}
, {-46, 43, -52}
, {-20, -59, -5}
, {-26, 23, 35}
, {-13, 55, -51}
, {-26, 29, 31}
, {3, 32, -8}
, {-12, 25, 46}
, {-50, -5, 13}
, {-23, -34, 62}
, {-35, 39, -38}
, {-30, 7, -27}
, {-9, -11, -47}
, {-1, -43, -22}
, {17, 52, -51}
}
, {{14, 33, 21}
, {-21, 22, -26}
, {18, -21, -11}
, {14, -33, -21}
, {-1, -21, -38}
, {50, -40, -43}
, {-27, -21, 17}
, {-2, -2, 45}
, {-19, -9, -1}
, {-30, 57, -32}
, {-21, 4, -4}
, {39, -21, -34}
, {-30, -10, -29}
, {-5, 4, 9}
, {38, -8, 14}
, {-5, 54, -29}
, {-36, -10, -24}
, {-3, 37, -3}
, {-35, -38, -37}
, {17, 52, 40}
, {-37, -2, 54}
, {14, 46, -33}
, {39, 14, 48}
, {-33, -24, 3}
, {44, -34, 29}
, {27, 14, -7}
, {-11, -69, -44}
, {16, -11, -35}
, {39, -54, -27}
, {-12, -7, -19}
, {40, -5, 8}
, {-18, -51, -35}
, {-40, -27, -23}
, {-50, 10, -59}
, {62, 12, 21}
, {-9, 12, -43}
, {12, 0, 6}
, {-5, -31, 16}
, {-14, 56, -3}
, {-40, -14, -31}
, {-39, -22, -33}
, {-33, 31, -56}
, {42, -16, 39}
, {0, -39, 25}
, {-6, -12, 8}
, {60, -4, 39}
, {-45, -3, -2}
, {-34, -40, 10}
, {34, -5, 18}
, {26, -16, 16}
, {19, -42, -17}
, {-24, 30, 57}
, {26, 41, 36}
, {-11, 48, 0}
, {-34, 60, -37}
, {-37, -30, -60}
, {0, 45, -31}
, {47, -23, 38}
, {-57, -51, -39}
, {8, -25, 25}
, {-12, -40, 21}
, {43, 28, 34}
, {-48, -44, 8}
, {52, -39, 35}
}
, {{-60, -53, 12}
, {-53, -9, 18}
, {0, 21, 5}
, {18, -19, -46}
, {45, 32, 19}
, {12, -11, -1}
, {16, -41, -39}
, {-29, -24, -19}
, {-25, 32, -39}
, {17, 29, 34}
, {0, -27, 18}
, {-30, 28, 36}
, {49, 7, -23}
, {-12, 13, 40}
, {57, 40, 31}
, {7, -12, 51}
, {-5, 9, 34}
, {17, -10, -15}
, {-34, -44, 33}
, {0, -18, 31}
, {29, 41, 45}
, {29, -46, -50}
, {-1, 42, 0}
, {-50, -2, -25}
, {31, 20, -31}
, {-37, -41, -25}
, {-21, 50, -3}
, {-4, -15, 15}
, {33, -40, 61}
, {-9, 6, -28}
, {18, -33, -8}
, {4, -26, -10}
, {-36, -22, -43}
, {-14, 61, -35}
, {-53, 21, 18}
, {-29, -28, 41}
, {25, -7, 53}
, {31, -10, 3}
, {27, -9, -44}
, {57, 10, 36}
, {-46, 23, -35}
, {34, 48, 11}
, {31, -32, 8}
, {-13, -15, -31}
, {2, -34, -10}
, {0, 8, -47}
, {-44, 31, 8}
, {36, 18, -20}
, {-45, -46, -5}
, {7, 23, 53}
, {-34, 47, 10}
, {12, -44, 32}
, {8, 11, -18}
, {1, -59, -58}
, {17, -31, -48}
, {53, 31, 37}
, {21, -32, 48}
, {49, -23, 48}
, {29, -13, -6}
, {-30, 47, 37}
, {-3, 40, 11}
, {-38, -20, 4}
, {46, 26, -15}
, {25, -11, 6}
}
, {{-28, 5, -32}
, {-42, -50, -22}
, {-24, -32, -6}
, {33, 52, 9}
, {-43, -3, 0}
, {13, 32, 22}
, {-2, -22, -18}
, {37, -46, -16}
, {3, -43, 35}
, {-10, -41, -24}
, {-17, -31, 46}
, {-18, 0, -58}
, {-36, -21, 46}
, {-18, -11, -29}
, {3, 11, 25}
, {26, 31, -16}
, {-8, 15, -35}
, {59, -32, 7}
, {-36, -31, 36}
, {-42, -29, -35}
, {51, 39, -25}
, {17, -36, 48}
, {45, -45, 5}
, {48, 6, 12}
, {-51, -10, -26}
, {-5, -12, -21}
, {-38, 29, -48}
, {29, -42, 10}
, {15, 45, 1}
, {38, 41, -32}
, {24, -9, 27}
, {33, 22, 8}
, {57, -48, -38}
, {33, 17, 0}
, {52, -18, -2}
, {-28, -39, 22}
, {-26, 0, 8}
, {-16, 9, -37}
, {6, 45, 50}
, {-22, -44, -49}
, {-16, -31, 10}
, {-29, -6, -10}
, {55, 15, 38}
, {15, 14, -20}
, {-12, -38, 33}
, {-20, 39, -42}
, {2, -1, 4}
, {37, 19, -14}
, {-2, -69, -13}
, {-48, 31, -56}
, {46, 27, 29}
, {28, -43, 35}
, {41, 5, -37}
, {16, 56, -34}
, {29, 19, 36}
, {-18, -19, -56}
, {8, 52, 7}
, {16, -25, 41}
, {20, -19, 40}
, {-7, -41, -4}
, {-25, 24, 26}
, {-15, -41, -42}
, {43, -14, -1}
, {-22, 34, -1}
}
, {{12, -9, -55}
, {25, 10, 49}
, {8, 11, 11}
, {39, 52, 16}
, {41, 46, -38}
, {35, -40, 7}
, {-30, 7, 7}
, {15, 2, -16}
, {-30, -37, -44}
, {-7, 19, 42}
, {28, 47, 43}
, {-30, 18, 47}
, {-23, -35, 18}
, {-11, 29, -31}
, {-16, 9, -40}
, {-23, 25, 18}
, {-3, 37, -47}
, {8, -3, 22}
, {-18, 34, -47}
, {-31, 34, 14}
, {-53, -31, 24}
, {-4, -18, -21}
, {29, 33, -7}
, {-56, -1, -25}
, {-14, -11, 31}
, {61, 7, -6}
, {61, 60, 56}
, {-40, 12, -22}
, {-3, -22, -6}
, {44, 25, -7}
, {-36, -37, 25}
, {4, 25, 34}
, {-1, -8, 42}
, {51, 18, 56}
, {-25, 27, -44}
, {60, 11, -13}
, {-22, -38, 42}
, {-25, 23, 17}
, {45, -28, -54}
, {38, -8, 22}
, {-11, 46, 9}
, {0, -18, 20}
, {-6, 23, -11}
, {46, -10, -46}
, {-9, -48, 38}
, {-20, -33, 19}
, {40, 35, -1}
, {-3, -39, 42}
, {6, 24, 1}
, {-18, -19, 17}
, {0, -37, -7}
, {42, 5, -16}
, {10, 48, -34}
, {41, 24, 3}
, {7, -23, -26}
, {12, -8, 25}
, {10, -38, 33}
, {-15, 18, -16}
, {-5, 53, 71}
, {-18, -8, -13}
, {5, -17, -9}
, {47, -16, 29}
, {-52, -32, -40}
, {-6, 24, -5}
}
, {{54, 17, -10}
, {-29, -11, -18}
, {57, 21, -7}
, {-14, 43, -32}
, {7, 1, -37}
, {-13, 14, 29}
, {-30, 5, 28}
, {-31, -28, 14}
, {-29, -34, -33}
, {12, -23, -37}
, {-8, -6, 45}
, {-52, -43, 47}
, {12, 17, 5}
, {54, -5, -27}
, {-3, 17, 37}
, {27, 28, 49}
, {36, 6, -7}
, {43, 18, 1}
, {-21, 33, 35}
, {-34, -34, 46}
, {41, 21, -30}
, {-19, -1, -45}
, {-34, 48, 12}
, {34, 25, 33}
, {-34, -6, -14}
, {39, -53, -8}
, {-58, -2, 38}
, {-44, 35, -37}
, {-42, -1, -44}
, {3, -16, -17}
, {18, -16, -45}
, {-30, 16, 24}
, {24, -43, -42}
, {20, 17, 5}
, {-8, 2, 25}
, {26, 38, 63}
, {38, 30, -43}
, {-54, -35, 9}
, {11, 15, -30}
, {20, 60, -23}
, {23, 4, -55}
, {-52, 12, 30}
, {18, -33, 13}
, {37, -11, 21}
, {8, -1, 35}
, {18, -50, -31}
, {-1, -41, -26}
, {-26, 7, 15}
, {41, -1, 6}
, {21, 35, -34}
, {-12, 46, 24}
, {29, 14, 54}
, {5, -15, -42}
, {27, 21, 25}
, {21, 47, 10}
, {-42, -45, 5}
, {50, -24, -9}
, {-14, 38, 35}
, {26, -31, 29}
, {-47, 17, -40}
, {1, 36, -44}
, {-14, -32, 27}
, {19, -4, -31}
, {50, -46, 35}
}
, {{-3, -42, -45}
, {1, -14, 22}
, {-28, 5, 18}
, {3, -47, 36}
, {-18, 14, 1}
, {-39, -8, 31}
, {-3, 45, -27}
, {2, 11, 21}
, {-38, 11, -18}
, {-21, 29, -4}
, {-13, -25, -50}
, {-35, -31, -11}
, {-11, -34, -7}
, {55, 64, -26}
, {14, -25, 0}
, {23, 22, 31}
, {-44, -39, -12}
, {-14, -36, -20}
, {-21, 53, 19}
, {-4, 3, 5}
, {6, -49, 27}
, {43, 28, 24}
, {18, 25, -18}
, {-42, -42, 53}
, {-69, 19, -71}
, {-58, 5, 36}
, {3, 30, 38}
, {-32, 28, -50}
, {52, 18, 22}
, {-25, 2, -54}
, {46, 28, -22}
, {53, -34, -27}
, {41, 33, -18}
, {40, -19, 24}
, {3, 16, 45}
, {35, -23, 3}
, {-35, 40, 23}
, {55, 34, -9}
, {43, 46, 39}
, {21, 4, 45}
, {-36, 20, -10}
, {57, 41, 35}
, {-19, 14, -19}
, {-10, -35, 34}
, {-7, 40, 58}
, {-40, 32, 33}
, {-16, -57, 0}
, {-46, 20, -52}
, {-34, 38, 30}
, {6, -60, 17}
, {35, 36, 11}
, {15, 18, 48}
, {3, -32, 36}
, {18, -34, 8}
, {34, 7, 49}
, {-4, 12, 23}
, {-1, -44, 26}
, {4, 13, -56}
, {15, -7, -14}
, {20, 16, 46}
, {11, -28, 35}
, {-10, 8, 46}
, {47, -37, 12}
, {-1, -13, -21}
}
, {{-11, 11, 50}
, {-27, -10, 46}
, {-37, 6, 29}
, {40, -2, 57}
, {44, -26, 5}
, {-32, 19, 34}
, {-11, -45, -21}
, {9, 42, -14}
, {-23, -18, 39}
, {65, -38, 3}
, {25, 6, 55}
, {40, 29, 55}
, {-36, -38, 45}
, {-20, -11, -17}
, {-30, 25, -29}
, {-42, -48, -47}
, {-24, 44, -28}
, {-24, 24, 35}
, {-21, -44, 11}
, {46, -46, 20}
, {7, -26, 42}
, {4, 31, -44}
, {32, 52, -9}
, {68, -20, 8}
, {13, -18, -16}
, {-37, -14, 19}
, {-21, 13, 3}
, {44, 22, -25}
, {14, -47, 11}
, {-37, -39, -18}
, {-18, -15, -2}
, {2, 15, -63}
, {3, 44, 48}
, {-41, -34, -10}
, {-20, -15, 55}
, {-13, 39, 0}
, {8, -42, -48}
, {37, -24, 47}
, {55, 18, -31}
, {-32, -22, 4}
, {-20, -47, -32}
, {-19, 30, -13}
, {-7, 51, -36}
, {13, -32, 7}
, {-29, -35, 36}
, {33, -6, 40}
, {-22, 22, 12}
, {13, -44, -9}
, {51, 7, -34}
, {-19, 39, -42}
, {51, 47, 39}
, {21, 7, 13}
, {16, 28, -29}
, {-32, 0, -42}
, {-40, -43, 18}
, {-1, 0, -6}
, {-47, -6, 29}
, {2, -17, 35}
, {39, 1, 0}
, {-46, 24, 48}
, {48, 5, 44}
, {12, -46, 16}
, {-15, -8, 2}
, {-43, 16, -46}
}
, {{1, 14, -30}
, {20, -40, -24}
, {26, 22, 22}
, {-32, 46, 1}
, {-3, 6, 5}
, {-6, -28, -17}
, {44, -10, -25}
, {-21, -33, 22}
, {27, 58, -19}
, {0, -37, -1}
, {7, 16, 9}
, {1, 21, 28}
, {21, -31, 2}
, {35, 8, 54}
, {17, -23, 46}
, {38, -3, -42}
, {-5, -39, -13}
, {-6, -5, -27}
, {-30, 0, -21}
, {14, 50, -34}
, {-15, 1, -32}
, {2, 12, 52}
, {29, 41, 42}
, {-46, 26, -17}
, {35, 10, 34}
, {-17, -22, 4}
, {-38, -41, 16}
, {38, -10, -21}
, {-54, -18, 16}
, {-42, -57, -24}
, {-31, -23, -14}
, {-56, 14, -9}
, {-48, 19, 19}
, {-48, 50, -17}
, {4, -26, 38}
, {-37, 39, 37}
, {34, -21, -12}
, {-38, -42, 47}
, {23, 10, 25}
, {-22, -20, -15}
, {-19, -42, 36}
, {-1, -5, -35}
, {-29, 27, -12}
, {-16, -24, 14}
, {-43, -9, 52}
, {45, 45, -38}
, {26, -34, 18}
, {-4, 17, -41}
, {-40, 10, -19}
, {-16, 1, -18}
, {54, 43, 29}
, {-50, -25, 24}
, {46, -12, -25}
, {15, -35, 31}
, {11, 50, -37}
, {-25, 16, 53}
, {40, -46, -22}
, {-25, -14, -59}
, {-53, 51, 0}
, {44, -40, 22}
, {22, -11, 40}
, {24, -3, -37}
, {-23, 37, 48}
, {38, 11, 9}
}
, {{0, -27, 71}
, {-38, -22, 25}
, {18, 39, -50}
, {-40, -7, -10}
, {18, -37, 50}
, {4, 38, 51}
, {20, -36, -16}
, {-16, -33, 24}
, {-10, 53, -32}
, {36, -13, 21}
, {14, -57, -69}
, {20, -44, 3}
, {-16, -20, -3}
, {-30, 31, 48}
, {3, 52, 22}
, {42, -22, 30}
, {49, -34, 2}
, {-11, 10, -28}
, {-23, -29, -29}
, {50, 28, 2}
, {-53, 6, -6}
, {11, 30, -5}
, {45, 28, 3}
, {-48, -8, -42}
, {16, 28, -31}
, {-21, 55, -2}
, {-40, 25, -41}
, {-8, -6, 4}
, {-18, -5, -37}
, {-14, 41, 24}
, {54, -3, 7}
, {34, 20, -39}
, {-27, 30, -14}
, {-13, -54, 30}
, {24, -19, 57}
, {38, -31, -16}
, {-15, 7, 17}
, {8, -14, -24}
, {-2, 32, -18}
, {34, 55, -22}
, {-38, -3, 4}
, {-5, 25, -9}
, {-11, 25, 49}
, {-15, 10, -33}
, {10, -22, -2}
, {-40, 31, 21}
, {11, 12, 27}
, {14, 23, 39}
, {-27, -28, 17}
, {-59, -5, 2}
, {29, -7, 13}
, {-26, -35, 62}
, {8, 44, 1}
, {0, 3, -26}
, {4, 38, -43}
, {39, -32, 34}
, {-14, 17, -38}
, {-22, 71, 44}
, {9, -23, 12}
, {-28, 33, -24}
, {-15, 7, -10}
, {-7, -35, -32}
, {-10, 32, 22}
, {-37, -22, -27}
}
, {{-41, -49, 13}
, {-41, 10, -22}
, {-20, -37, -7}
, {-34, 16, -17}
, {2, -35, -2}
, {46, -29, 37}
, {-29, -23, 0}
, {-37, -17, -23}
, {-52, -12, -4}
, {-28, 14, 19}
, {-38, 31, 3}
, {-14, -3, -27}
, {-20, 36, -5}
, {-58, 16, -13}
, {28, -21, -42}
, {4, 26, 5}
, {-53, -34, -47}
, {3, 39, 58}
, {-16, -16, -37}
, {-12, -60, 41}
, {-37, 24, 25}
, {24, -18, 10}
, {-56, -30, 0}
, {-41, -54, -44}
, {13, 5, -48}
, {2, -10, 42}
, {44, 25, -36}
, {-22, 39, -6}
, {-31, 14, 52}
, {-38, -7, -32}
, {4, 11, 48}
, {-51, 26, 37}
, {-22, 11, 21}
, {13, 2, 11}
, {39, 39, 18}
, {-20, 54, -41}
, {-16, -38, 20}
, {-33, -28, -13}
, {-41, 1, -50}
, {7, 7, -30}
, {35, 32, 52}
, {35, 18, -45}
, {-6, 29, 54}
, {13, -35, 31}
, {-40, -25, 37}
, {3, 5, -25}
, {-31, 36, -28}
, {50, -29, 45}
, {-25, -21, 13}
, {34, 42, 37}
, {-16, -38, 11}
, {-45, 23, -26}
, {-26, -22, -36}
, {-40, -4, -54}
, {-45, -25, 29}
, {12, 6, 20}
, {6, 44, 0}
, {-7, -31, -18}
, {24, 44, 38}
, {18, -50, 44}
, {34, -34, 37}
, {-43, 43, 2}
, {49, 9, 43}
, {-42, 0, -55}
}
, {{-45, -18, 28}
, {-11, 12, -54}
, {-24, 18, -26}
, {41, -36, 45}
, {-11, 11, 26}
, {38, -33, 50}
, {42, 12, -28}
, {-13, 25, 37}
, {-46, 43, -9}
, {21, -52, -54}
, {65, 44, 7}
, {12, 20, 29}
, {-5, 52, 10}
, {-12, 34, -64}
, {48, 40, 4}
, {14, -51, 31}
, {-28, 11, -7}
, {13, 13, 52}
, {44, -26, -53}
, {52, -17, 0}
, {25, 18, 5}
, {-37, -10, -2}
, {15, -45, 37}
, {4, -52, 36}
, {0, -19, 50}
, {54, 54, 62}
, {-16, 23, -1}
, {32, 40, -25}
, {1, 37, 29}
, {-47, 44, 2}
, {-27, -10, -36}
, {-4, 39, -21}
, {2, -27, -46}
, {-9, 19, -19}
, {-51, -17, -61}
, {-47, -22, -13}
, {-4, -30, 18}
, {18, -51, 40}
, {5, 39, -35}
, {27, 23, 49}
, {-6, -5, 5}
, {-41, -37, -5}
, {-20, -3, 42}
, {-38, 0, -34}
, {-31, 18, 36}
, {-10, -54, -35}
, {54, -33, 32}
, {48, -3, 12}
, {-44, -39, 40}
, {10, 35, 31}
, {1, -37, 10}
, {40, 9, 61}
, {1, 9, 50}
, {44, 4, 22}
, {-51, 38, -38}
, {-34, 42, 39}
, {46, -36, 21}
, {63, -31, 70}
, {-2, 2, -18}
, {32, 11, -31}
, {-15, 37, -23}
, {20, -29, 21}
, {34, -23, -4}
, {-35, -7, -22}
}
, {{30, 24, 21}
, {-38, 35, 12}
, {-31, -8, 55}
, {19, -11, -8}
, {13, -37, 32}
, {9, -33, -4}
, {0, -20, -24}
, {25, -41, -41}
, {13, -15, 39}
, {-26, 4, -44}
, {-8, 3, -21}
, {-13, 45, 7}
, {28, 2, 24}
, {-13, 5, -32}
, {19, -22, -41}
, {39, 9, 18}
, {4, 1, 18}
, {46, -32, 12}
, {0, 32, -10}
, {14, 53, -35}
, {-33, 40, 46}
, {2, 23, -24}
, {17, -33, 51}
, {52, -13, -48}
, {-3, 15, 25}
, {16, -44, 8}
, {-18, 27, -36}
, {23, -10, -42}
, {-28, -40, -37}
, {-25, 33, -15}
, {40, 48, 3}
, {-15, -47, -61}
, {-2, 46, -18}
, {25, 44, 4}
, {52, -18, 1}
, {-2, -39, -18}
, {48, 27, -27}
, {19, 0, 48}
, {43, 47, -36}
, {7, -27, -36}
, {17, -34, 26}
, {-11, -59, -60}
, {-12, -43, -42}
, {-50, -2, 14}
, {-30, 48, 28}
, {33, 48, -47}
, {16, -23, -28}
, {25, -60, -14}
, {52, 48, 64}
, {18, -63, 13}
, {42, -25, 8}
, {-59, -66, 24}
, {11, 1, -20}
, {-18, 27, -25}
, {25, -4, 27}
, {-52, -48, 32}
, {4, 21, 41}
, {-9, -49, -26}
, {0, -46, 0}
, {32, -40, 5}
, {-3, 17, 51}
, {34, 28, -27}
, {-24, 37, 27}
, {-45, 37, -38}
}
, {{53, 16, -4}
, {40, -32, 42}
, {-28, -14, -39}
, {-54, 22, -46}
, {-40, 19, -15}
, {32, 38, -28}
, {48, -37, -33}
, {2, 57, 47}
, {-26, -2, -36}
, {-17, 49, 47}
, {8, 25, 57}
, {-21, 58, 42}
, {-37, 5, 42}
, {-16, 0, -50}
, {-25, -24, 4}
, {24, -15, -15}
, {-3, -4, -38}
, {-23, 35, -61}
, {21, 13, -15}
, {15, 32, 43}
, {-8, 16, 44}
, {19, -11, 0}
, {3, -5, 43}
, {53, 40, -13}
, {9, 39, 29}
, {-29, 0, 46}
, {30, -55, 25}
, {17, -22, 26}
, {0, -28, -9}
, {-19, 53, -24}
, {14, -42, 23}
, {-28, 39, 0}
, {-16, -18, 12}
, {0, -17, 0}
, {72, -27, -30}
, {13, 6, -15}
, {-36, -16, -23}
, {-22, 39, 28}
, {17, 48, 60}
, {-52, -33, -34}
, {37, -19, -13}
, {13, 20, 24}
, {-39, 34, -39}
, {38, 32, -7}
, {-52, -11, -55}
, {39, -43, -20}
, {48, 6, 49}
, {25, 4, -28}
, {-45, -45, -28}
, {54, -14, 52}
, {44, 5, -19}
, {-31, -16, 18}
, {25, -4, -48}
, {-49, -54, 1}
, {-31, -17, -35}
, {18, 29, -42}
, {46, 55, -39}
, {50, 31, -11}
, {30, 28, -18}
, {-41, -23, 41}
, {4, 35, 25}
, {-13, 23, 2}
, {-23, -28, 4}
, {27, 23, -7}
}
, {{-17, -8, 24}
, {34, 40, 39}
, {47, -41, -42}
, {53, 39, -5}
, {43, 48, 39}
, {35, -45, 21}
, {-26, 19, 34}
, {9, -31, -47}
, {-43, -28, 17}
, {40, 29, 49}
, {-2, 39, 59}
, {40, 19, -37}
, {47, 47, 7}
, {-71, -36, 12}
, {-39, -26, -17}
, {-11, -24, 26}
, {-14, -20, -30}
, {-46, 44, 53}
, {-37, -27, -27}
, {-8, -35, -51}
, {-42, -40, -38}
, {20, 10, -31}
, {-39, 35, 40}
, {-41, 42, 28}
, {15, -24, -22}
, {33, 27, -28}
, {-10, 47, 40}
, {4, 46, -43}
, {-44, -10, -36}
, {14, -30, 3}
, {-24, -22, 22}
, {13, 9, 32}
, {-45, 15, -15}
, {16, -45, -38}
, {18, 62, 64}
, {-9, 28, -23}
, {14, -43, 2}
, {14, -4, -25}
, {-2, -12, 2}
, {-8, -58, 0}
, {44, -3, -40}
, {-26, 55, -27}
, {9, 47, -28}
, {-4, -47, 11}
, {-53, 10, 45}
, {32, -28, -10}
, {56, 33, -14}
, {-9, 40, -26}
, {44, -47, 25}
, {37, 67, 16}
, {-29, 0, -47}
, {43, -23, -52}
, {1, 35, -8}
, {55, -1, 15}
, {7, 17, -44}
, {15, -45, -11}
, {9, -20, 42}
, {-22, -38, 17}
, {16, -2, 37}
, {22, 19, 29}
, {-12, 1, -31}
, {-20, 20, 37}
, {52, 26, 53}
, {-7, -51, -3}
}
, {{54, 60, 52}
, {14, -40, -8}
, {49, 57, -19}
, {49, 31, -51}
, {-21, 15, 42}
, {-40, 17, 15}
, {-19, -49, 10}
, {-13, 23, 24}
, {38, 50, 16}
, {-11, 14, -12}
, {-32, 15, -53}
, {20, -26, 45}
, {35, -50, -4}
, {13, 54, 20}
, {-28, -10, 13}
, {-15, 22, 30}
, {24, -37, 1}
, {63, 51, -5}
, {39, -49, -16}
, {34, -25, -32}
, {0, 36, 40}
, {-39, 36, -20}
, {43, 41, -49}
, {-21, -3, 4}
, {43, 2, 26}
, {11, -42, 51}
, {8, -11, -9}
, {-12, -50, -48}
, {41, -19, 28}
, {-40, 40, -8}
, {5, 61, -18}
, {-49, -27, 32}
, {18, -14, -12}
, {29, 37, 27}
, {11, 4, 33}
, {23, -30, 60}
, {2, -38, 27}
, {33, 8, 15}
, {48, -31, -24}
, {20, -33, 57}
, {-44, -19, -8}
, {27, -41, 18}
, {-25, -24, -15}
, {-23, 8, -12}
, {21, 44, 8}
, {17, -34, 0}
, {0, -39, -22}
, {-27, 31, -53}
, {1, -3, -23}
, {-16, 29, -33}
, {9, 3, -32}
, {15, -37, -42}
, {-8, -52, 17}
, {30, 63, -23}
, {-26, 32, 52}
, {-2, 47, 5}
, {-27, 45, -34}
, {-40, 30, 13}
, {-40, 35, -20}
, {8, 7, 32}
, {30, 4, 35}
, {16, -32, -11}
, {26, -34, 28}
, {-23, 47, -31}
}
, {{-8, 35, -46}
, {38, -11, -50}
, {-53, -6, -1}
, {21, 40, 20}
, {-9, 2, -7}
, {3, -52, -10}
, {22, -5, 45}
, {-13, 15, -23}
, {-20, 1, -9}
, {41, -2, -17}
, {16, -14, 0}
, {-8, 22, 48}
, {43, 24, -8}
, {12, -34, 11}
, {-38, 35, 17}
, {-31, 13, 49}
, {22, -9, -44}
, {69, -6, 5}
, {-41, -24, -27}
, {44, 35, -48}
, {0, 23, -49}
, {-52, 12, 6}
, {-50, -30, -8}
, {24, -40, -15}
, {17, 17, 22}
, {-31, 19, 45}
, {-37, 54, 9}
, {-14, 35, -8}
, {30, -22, -48}
, {2, -25, 47}
, {14, -18, 1}
, {43, 0, 45}
, {12, 30, -39}
, {-29, 10, 47}
, {-38, -27, -28}
, {-28, -36, 16}
, {50, 14, 47}
, {-41, 1, -7}
, {24, 41, 24}
, {17, -30, 10}
, {-27, 28, 16}
, {50, -22, -21}
, {44, 22, 3}
, {-24, -14, -26}
, {-26, -54, 31}
, {15, 9, 13}
, {-33, 7, 29}
, {14, 22, -34}
, {24, 28, -30}
, {18, 39, -28}
, {-4, -9, 33}
, {19, 59, 26}
, {-51, -32, 28}
, {27, -3, -8}
, {45, 8, 41}
, {-39, 13, 9}
, {34, 42, 46}
, {34, 19, -33}
, {-35, 23, 19}
, {-35, -6, -16}
, {-24, 9, -29}
, {-16, 4, -50}
, {-47, 38, 15}
, {-29, 32, -56}
}
, {{43, 37, -11}
, {29, -18, -20}
, {60, -20, -34}
, {1, 6, 32}
, {36, 11, 1}
, {-46, -40, -47}
, {30, 49, 30}
, {-46, 21, -17}
, {-37, 1, 24}
, {2, 26, 26}
, {16, 55, -26}
, {12, 45, -45}
, {3, -13, 38}
, {-14, 22, 46}
, {-53, -40, 1}
, {2, 27, 38}
, {-16, -32, -29}
, {-47, -12, -30}
, {55, -5, 17}
, {52, 25, -25}
, {-30, 6, -53}
, {17, 46, -10}
, {25, -13, 25}
, {62, 37, 20}
, {-29, -14, -26}
, {-2, -61, 8}
, {31, 0, 53}
, {43, -9, 35}
, {31, -12, -4}
, {-38, 44, 6}
, {-48, -26, -8}
, {-26, -12, 19}
, {-48, 7, -15}
, {-28, -15, -49}
, {53, 28, -12}
, {28, -10, 23}
, {42, 47, -40}
, {11, -9, 6}
, {14, -16, -19}
, {38, 17, -11}
, {-15, -34, 43}
, {-39, -18, 42}
, {0, -11, 26}
, {-47, 35, -20}
, {-3, 52, 0}
, {-34, -10, -14}
, {18, -3, 17}
, {-19, -12, -10}
, {21, 42, 65}
, {35, -23, 27}
, {41, -4, -45}
, {21, 21, -36}
, {21, -33, -7}
, {-25, 9, 39}
, {-30, -6, 22}
, {37, -15, -34}
, {47, 19, 28}
, {-32, 21, -37}
, {-24, 19, -7}
, {34, -53, -38}
, {41, -17, 41}
, {8, 32, 29}
, {-4, -1, -46}
, {-11, -41, 20}
}
, {{-2, 35, 29}
, {14, -8, -31}
, {-23, 43, -6}
, {33, 26, 32}
, {9, 16, 29}
, {0, 52, -6}
, {14, 51, 47}
, {-14, -4, 10}
, {54, -16, -38}
, {-45, 25, 40}
, {5, 44, 18}
, {-13, -2, -25}
, {5, 46, 41}
, {46, -28, 3}
, {2, -47, -19}
, {-33, 37, 4}
, {-14, 11, -20}
, {-30, -23, -65}
, {15, 16, -10}
, {-17, 42, 22}
, {54, 55, -48}
, {-25, 72, -38}
, {-46, -48, -2}
, {-39, 6, 0}
, {64, -32, 73}
, {-3, 17, 42}
, {13, 41, -34}
, {53, -9, 1}
, {-8, -48, -11}
, {-46, -6, -44}
, {-62, -70, -54}
, {-13, -42, -58}
, {48, 13, -51}
, {1, 46, -35}
, {27, 11, 1}
, {0, 6, 3}
, {-54, -28, 14}
, {48, 55, 11}
, {-38, 49, 40}
, {29, -32, -2}
, {6, -53, -27}
, {44, 4, 23}
, {54, -26, -23}
, {-44, -49, 11}
, {40, 26, -61}
, {-10, -11, 18}
, {19, 58, 36}
, {-16, 32, 24}
, {4, 30, -42}
, {18, -53, -15}
, {-5, 26, -6}
, {-18, -28, -25}
, {-17, -1, -28}
, {-41, -19, 69}
, {-49, -28, 10}
, {-3, -1, -14}
, {38, 9, -3}
, {5, 0, -27}
, {27, 30, 39}
, {-34, -43, -3}
, {34, -45, -16}
, {19, 21, 27}
, {50, -41, 8}
, {-43, -34, -5}
}
, {{-28, -33, 38}
, {15, -18, 28}
, {-7, -46, -36}
, {11, 14, -11}
, {-30, -52, 20}
, {45, 9, -25}
, {56, 18, -7}
, {13, 16, -26}
, {-11, -43, 3}
, {-52, -27, 4}
, {47, 42, -38}
, {33, 0, -9}
, {4, 13, 0}
, {10, -18, -60}
, {22, 29, -3}
, {6, 15, 25}
, {10, -27, -8}
, {-10, 18, 57}
, {11, -36, 29}
, {34, -21, 18}
, {-4, 46, 10}
, {3, -16, -36}
, {18, 25, -42}
, {34, 32, 16}
, {-37, -7, 41}
, {35, -38, 20}
, {-15, 35, -52}
, {49, 55, 49}
, {-23, 34, -53}
, {4, -30, -24}
, {-48, 37, 18}
, {2, 11, 30}
, {-20, -10, 9}
, {-33, -41, -24}
, {-52, 35, -53}
, {32, 40, -58}
, {-52, 50, -20}
, {35, -12, 6}
, {-14, -33, -3}
, {-52, -19, -54}
, {-38, -27, -50}
, {8, -17, 15}
, {32, 31, 7}
, {-53, 20, -53}
, {-8, -14, -22}
, {46, 2, 13}
, {33, -17, 36}
, {46, -51, 19}
, {11, -9, -29}
, {28, 8, -5}
, {-28, 0, 29}
, {21, -34, -35}
, {-35, 4, 3}
, {36, 14, 38}
, {-18, -58, -30}
, {-19, 36, 29}
, {-36, 13, -22}
, {-6, -9, 8}
, {-25, 41, -16}
, {42, 4, -21}
, {43, 27, -40}
, {-13, -5, 44}
, {31, 48, -26}
, {18, -1, -11}
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
#define INPUT_SAMPLES   28
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_149_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_149(
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
#define INPUT_SAMPLES   14
#define POOL_SIZE       14
#define POOL_STRIDE     14
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t average_pooling1d_29_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d_29(
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

//typedef number_t *flatten_29_output_type;
typedef number_t flatten_29_output_type[OUTPUT_DIM];

#define flatten_29 //noop (IN, OUT)  OUT = (number_t*)IN

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

typedef number_t dense_58_output_type[FC_UNITS];

static inline void dense_58(
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


const int16_t dense_58_bias[FC_UNITS] = {-8, 10, 14, 4, 4, 11, 13, 0, -2, -9, -4, 2, -9, -4, 0, 9, -8, 4, 9, 2, -5, 14, 7, 14, 4, -3, -1, 11, -6, 0, -10, 1, 14, 0, 0, -8, -8, -3, 1, -8}
;

const int16_t dense_58_kernel[FC_UNITS][INPUT_SAMPLES] = {{49, 31, -50, 62, -23, 4, 48, -8, -75, -13, 91, -10, -50, 40, -7, -85, -45, -61, 44, 15, 38, -9, 127, -59, -102, -57, 68, 86, -71, -95, 6, 17, -54, 65, 31, -65, -25, 45, -84, 64, 35, -95, 21, 86, -14, -84, 116, 80, -46, 87, 87, -58, -21, -34, -74, 57, -9, -47, -70, 0, 115, -33, 56, 16, 14, -2, -50, 83, 75, -89, -94, 9, -29, 7, -43, 38, 64, -32, -79, -5, 66, -58, -70, 110, -28, 33, -25, -12, 96, -7, 44, -58, -77, -96, 40, 17, -44, -25, 4, -65, 38, 20, 18, 24, -48, 82, -53, 14, -29, 23, 14, 62, -56, 55, -83, -53, -36, 51, 36, 27, 36, 84, 2, -94, 44, -74, -50, 72}
, {-71, 81, 60, -27, 87, -34, 18, -69, -8, 48, -50, 15, 58, 80, -49, 22, 0, 80, -44, -16, -3, 11, 89, 80, -61, -1, -60, 55, -44, 30, 91, -41, -46, 32, 9, 92, -5, 42, 42, -16, -87, -3, 5, 76, 16, -60, -23, -25, -27, -71, -36, 33, 74, 22, -86, -6, 68, -29, 17, -30, 66, -26, -67, 67, 19, -20, 72, 30, -23, -70, 73, 8, 58, 22, 49, 33, 68, 16, 21, 16, 11, -40, -77, 57, -86, 21, 93, 53, 80, -41, 46, -27, 106, -42, 77, -53, 15, -60, 56, -34, -20, 74, -2, 93, 34, -6, 81, -5, -75, -66, -13, 110, -64, -76, 47, 31, 36, -96, 19, -22, -29, -83, 50, 42, -46, 17, 77, 90}
, {-60, 13, 111, 71, 66, 25, -80, 74, -47, 70, 58, -12, -82, -11, -25, 95, -57, -25, 59, 46, 82, -5, 83, -63, -68, -102, -76, 76, 7, 90, 34, -100, -15, -49, -2, -21, 24, 82, 60, -55, 76, 59, -3, 69, -82, 65, 20, -21, -36, -51, -73, -34, 40, -33, -20, -56, 19, 44, -94, 108, -74, 17, -33, -13, -43, -11, 63, -57, 20, 2, 47, -52, -44, -37, -77, -28, 40, -65, 39, 78, 34, 12, 21, 14, -80, -11, 27, -54, -13, -85, 21, -45, -74, 38, 12, 57, -62, -64, -41, 63, -4, -15, -46, 22, -74, -77, 48, 56, -18, 86, -54, -109, -2, 16, 48, 40, -20, -42, 64, 11, -70, 93, -31, 63, -56, 46, 94, -39}
, {-68, -8, 15, -40, 36, -48, -80, 29, 75, -75, -51, 48, -59, 24, 82, 90, 69, -32, -59, -3, 82, -1, 33, -77, -90, 85, -12, 43, 5, 22, -25, 92, -20, -21, -94, 67, 39, -73, 53, 52, 6, -96, -102, 85, -89, -75, 111, 55, 68, -2, -100, -4, 70, -49, -45, 71, 17, 52, 92, 65, -45, 74, 15, -29, -38, 21, 53, 20, -88, 12, 60, -102, -44, 32, -89, -21, -24, -40, 48, -35, -82, -14, 3, -66, -7, -19, 79, -32, -71, 6, 79, 14, 30, 7, -4, -4, 58, -95, -62, 41, 25, 93, 73, -78, 31, 49, -16, -85, 89, 35, -31, -36, 79, 62, -18, 38, 26, 56, 21, 24, 37, -32, -19, 84, 109, -94, -74, -33}
, {-27, -8, -57, -58, 61, -119, 19, -16, -71, -33, 75, 70, -50, -79, -77, -81, -65, 56, -53, 90, -72, -53, -15, -100, 42, -87, 29, -6, -27, -66, -43, 106, -81, 5, -86, 46, 71, -20, 27, 102, 41, -25, -48, 77, 52, 42, 106, -32, 42, 2, -83, 95, 51, 91, 59, -88, 17, 74, -41, 57, 7, 41, -49, 94, 73, 80, -71, 42, -42, -9, 60, -63, 4, 14, 9, 46, -50, 44, -42, -60, 42, -90, -68, 75, 55, 53, 29, -63, 8, -18, -98, -36, 9, 29, 37, -71, 10, -35, -28, 5, -86, 30, -9, -100, -15, 19, 62, -10, -70, 73, -34, 41, 69, 28, 11, -41, -83, 65, -48, 59, 48, 74, -44, -9, 15, -78, -47, -22}
, {-23, -58, -80, 60, -13, -66, -22, -10, 102, 35, 21, 83, -82, 84, -62, -72, -36, -6, 28, 71, 58, 53, -108, 66, 55, 105, 66, 17, -49, 62, -79, 54, 32, 89, 34, -3, 89, -79, 15, 12, -28, -74, -64, 74, -28, 37, -76, -66, -3, -35, -60, -25, -89, -63, -94, 8, 101, -85, 70, -57, 19, -101, 11, 20, 39, -56, -96, 50, -57, 31, -63, 78, 15, 101, -35, 68, 41, -41, -61, -55, -37, -56, 66, 42, 89, 74, -16, -11, -53, 63, 40, 60, 105, -20, 98, 22, 31, 17, -58, 47, -74, 63, -15, -42, 19, -52, 36, 58, 66, -18, 36, 37, 87, 2, 12, 48, -28, -75, -4, 4, 36, -75, 70, -34, 92, -53, -9, -5}
, {37, -64, -69, 76, -29, -32, -19, -26, 3, 18, -85, -79, -57, -72, 61, -53, 25, -81, 102, -102, 88, 2, -25, -77, 41, 32, -8, -31, -30, -12, 93, 42, 6, -90, 91, 12, 40, 42, 31, 35, 7, 14, -67, 66, -65, -83, -62, -92, 10, 50, -80, -41, 45, 76, 3, -50, 72, -33, -67, -50, 43, 62, -2, 54, -60, 76, -28, 85, -106, 44, 88, -75, 66, -71, -108, -26, -46, -9, 92, 55, 45, 28, 99, -59, -43, 52, 54, -41, -77, 69, 2, -65, 54, -48, 12, 66, -80, -41, 3, 81, 81, 5, 88, -94, 71, 17, 23, 69, -79, -75, 92, 56, 49, -46, -43, -33, -32, -64, -38, 59, 13, -36, 27, 89, -69, 12, -20, 19}
, {55, -7, 82, -39, 92, -21, 13, 75, -26, 67, 22, 88, -58, 58, -71, 75, 35, -33, -88, -70, 90, -44, -26, -23, 59, -25, -24, 10, 73, 69, 23, -8, 95, 59, 37, -71, -42, 17, 24, -37, -63, 64, -17, -96, -30, 9, 88, -8, 77, -48, 55, 87, -20, -46, -59, 13, 20, 79, 51, 23, -53, 32, -24, 54, 59, -52, -45, 70, 30, 15, -16, -52, 8, 60, -1, 0, 79, 12, -66, -33, -74, -100, -79, -77, 11, 1, 12, -43, -90, -78, -31, -10, -85, -92, 15, -54, 27, -56, -59, -93, -12, 23, -25, -3, -89, -24, -91, -27, -79, 73, -64, -69, -18, -10, 85, 24, -60, 69, -43, -59, -83, -19, 30, 69, -30, 53, 1, -2}
, {-1, 72, -107, 11, 88, -21, 36, -71, -107, 90, 43, 83, -17, 38, 27, 87, 53, -75, -5, 64, 21, -55, -26, -28, 34, 85, 80, -58, -55, -60, -56, 100, -66, -11, 53, -90, 92, -83, -54, 83, 62, -20, -22, 84, 37, 79, 65, -97, -31, -3, -47, 32, 42, -57, -76, 47, -14, 59, 6, -5, -92, -33, 2, 34, -23, -30, 61, -37, 98, -41, 107, 52, 17, 63, -82, 30, -73, -23, 61, -89, -61, -42, -21, -62, -81, -61, -37, 27, -39, -68, -38, -22, -53, 32, -38, -94, 32, 83, -24, 67, 44, -17, -33, 59, 74, 82, -23, 76, 83, 28, 5, -28, 60, 9, 70, -68, 84, 116, 17, -58, -12, -103, -68, 6, -80, -17, -8, -31}
, {-13, -16, -66, -14, -68, 91, 16, 21, 30, 59, 59, 32, 76, -21, 82, -43, -36, 67, 80, 42, -93, -12, 51, 23, -62, 77, 81, -15, 17, -90, -17, -17, 39, -85, -80, -62, 8, 67, 82, 9, 38, -80, 17, -40, -46, -43, 6, -44, 35, -68, 94, 4, -44, -79, -5, 92, 19, -19, -93, 60, -18, 71, 58, 70, 42, 67, -25, -17, 3, -9, -59, 16, 50, 1, -68, 69, 80, 88, -6, 62, -42, -38, -74, 2, 40, -8, 21, -11, -74, 86, -10, -13, -51, 7, -85, 28, -93, 72, -77, 83, -89, 3, 55, -87, -40, -41, 70, -14, -71, 18, 96, -57, -53, -30, 21, 52, 50, -103, 14, -34, 35, 56, 87, 58, 68, 45, 6, -20}
, {50, 62, -83, 27, 10, 100, 64, -76, 39, -13, 21, 21, -31, -10, -51, -95, -100, 27, -35, 0, 10, 93, -89, -50, 1, -70, -82, -103, 48, -30, 46, -59, -30, -36, -41, -35, -53, 87, 39, -28, -121, 90, -77, 87, 7, 87, 65, 78, -78, 3, 38, -82, -53, -62, 77, -31, -77, -43, -77, 17, -97, 64, 57, 54, 69, -88, -93, -17, 39, -6, -66, -34, 67, 85, -53, 38, 33, 16, -15, -57, 17, -99, 75, -35, 38, -41, 97, -11, 32, 91, -86, -76, 23, 100, -33, -73, 4, -1, -75, 0, -29, -81, 31, -45, 30, 28, -65, 69, 70, 14, -19, 15, 48, 35, 88, 84, 1, -1, 18, -78, 30, -88, -86, -75, 68, 47, 35, -35}
, {-78, -29, 74, 27, -69, -12, 61, 75, 4, -40, 36, -81, 68, -51, 15, 10, -20, 48, 4, 58, 95, -32, -29, -58, 12, 24, 68, -21, 10, 69, 33, 91, -73, -8, -45, -42, -70, 71, -2, -61, -8, 75, 87, -92, 11, 64, -99, 29, -28, 51, -95, -90, -85, 5, 88, -95, -33, -57, -53, 106, 99, 22, 90, -83, -38, -57, 12, -45, -88, 56, 11, 51, -108, -53, -71, 88, -28, -61, 30, -43, 37, -45, 98, 69, 89, -75, -57, -50, -11, -25, 89, 56, -95, 3, 8, 20, -46, -99, -39, -70, -20, -9, -78, 70, -53, 37, -19, -92, -49, 48, 12, 3, 5, 57, 6, -85, 20, 57, -48, 39, 0, -23, 30, -49, 0, 47, -72, -38}
, {-54, 78, 62, -24, 50, -41, 36, -53, -69, 26, -4, 21, 68, 35, 85, 72, 75, -4, 44, 49, 0, 79, 30, 69, 14, 40, 31, 28, -96, 44, 63, -107, -38, 9, -55, -95, -6, 54, -41, -51, 52, -33, 57, 93, 68, -76, 82, 110, -71, 15, -81, 27, 84, 72, 90, -57, -12, 50, -79, -79, 21, 90, 46, 18, 35, -32, 51, -91, 43, 36, 52, 60, -40, -47, -52, 18, -9, -48, 44, -1, -13, 29, 79, 14, -45, 28, -13, -96, 59, -20, 53, 43, 32, 64, 53, -101, -46, -37, -73, 18, 72, -2, 68, -82, -90, 25, -86, 64, -23, -40, -53, -106, -63, -72, 79, 33, -96, 29, 19, 40, -21, 74, 3, 24, 59, 68, 43, 27}
, {52, 63, -50, -6, -23, 57, 65, -2, -12, 91, -32, -17, -27, 37, -62, 21, -47, 49, -38, -20, 18, -60, 0, -26, 22, 106, -74, -101, 6, 0, 63, 107, 68, -41, -7, -52, 0, 18, -60, 86, 61, 27, -88, 29, -40, -5, -90, 47, -65, -15, -95, -59, -62, -97, -10, -76, -105, -94, -51, -39, -76, -74, 85, 72, 90, -26, 15, 67, -21, -57, 90, 17, -43, 115, 31, 57, -116, 23, 32, 21, -10, -90, 61, -82, 85, 83, 78, -86, -8, -84, -71, 82, -43, 79, 45, -88, -13, 35, 90, 68, -17, -70, -7, -3, -67, -25, -91, 80, -84, 10, 22, 90, -50, 97, 83, 58, -77, 25, -21, -89, 51, -43, -93, -16, -91, 1, -100, 22}
, {5, 10, 87, -70, -28, 28, 100, 47, 61, 2, 54, -36, 66, 94, 88, 15, -47, 55, 92, 8, -59, 50, 29, 47, -100, -63, -61, 64, -72, 30, 70, -65, 58, -53, -13, 81, 20, -76, 11, 45, -15, -72, 67, 33, -92, 12, 84, 46, 57, 58, 16, 35, -56, 14, 74, 12, -73, -2, -28, 0, -43, 84, 31, 45, -82, 44, 31, -58, -37, -71, -17, -7, -34, -12, -53, 29, -33, 0, 9, -39, -69, -31, 100, 3, 4, 63, -59, -27, -44, -34, 24, -65, 98, -75, -15, -43, -1, 28, 106, 39, -40, -97, -21, -26, -9, -42, -28, -48, -15, -3, -26, 29, 82, 7, -44, 60, 8, 81, -25, 80, 35, -22, 10, 24, -97, -1, 4, -69}
, {30, -35, -71, -68, 79, -10, -19, 29, 87, -76, 17, 100, 68, 12, 106, 65, -57, 38, 22, 49, 22, 67, -74, -70, 69, 30, 57, 24, -83, -72, 9, -24, 57, -83, -17, 74, 15, -82, -92, -81, 37, 17, -22, -17, 89, -101, -6, 1, -30, -49, -66, 66, 4, 43, -67, 66, -12, -14, 93, 34, -49, 56, 41, -28, 15, 106, -21, 83, 86, -15, 15, -80, 30, 29, 80, 26, 51, 48, -92, 89, -41, -55, 13, -61, 3, -83, 43, 94, -6, 58, -71, -93, 38, -36, -71, 101, 30, -68, -56, -79, 87, -67, 34, -25, -38, -54, 65, -107, 77, 53, 2, 10, -3, -12, -86, -26, -6, 81, 70, 51, -92, -37, -56, 87, 10, 27, -51, -35}
, {71, 43, 59, -68, -47, 60, 82, 61, 41, 27, -86, 70, -35, 37, 75, -26, -39, -63, -26, 27, -38, 36, 52, -96, -6, -20, 4, 5, -6, -27, 51, 53, 54, -40, 50, 95, 90, 46, -56, 5, 57, -95, 94, -51, -100, 80, 9, -83, -87, -98, 66, -53, 17, -79, -15, -49, -69, 27, -66, 24, -22, -30, 48, -96, -7, 54, 92, -48, 64, 89, -90, 43, 36, -95, 100, 3, -3, 0, 27, -32, 87, 27, 78, -32, 4, -69, -21, 49, -10, 83, -88, -3, -37, -68, -26, 94, -30, 60, -75, 49, 78, -106, 67, -37, -58, 77, 68, 45, -17, 74, -62, -53, 66, -72, 69, 82, 88, -52, 79, -67, 15, 43, 71, -49, -17, 17, 83, 77}
, {-79, 0, 58, -42, -80, -30, -25, -25, -55, -94, 92, -48, 13, -42, 31, -19, -53, -67, 58, -55, 91, -91, 48, -89, -45, -22, -41, -81, -30, -81, 88, 33, -97, -6, -76, 48, -101, -60, -22, 43, 32, -17, -59, 40, -81, -22, 69, 84, -45, 23, -60, 69, -62, -29, 9, -18, 93, 33, -79, -63, 83, 91, 2, 25, -47, -57, 66, 90, 9, 27, 68, 17, -65, -95, -4, 32, 22, 1, 90, 90, -2, -75, -48, 17, 76, 73, -78, -19, 4, 50, -82, 27, 36, 51, 68, -80, 81, -83, -44, 1, -24, -17, -92, 51, 53, 49, -17, -37, -68, 101, 102, -60, -27, -54, 91, -65, -14, 16, 90, 30, -85, 64, 39, -48, -62, -33, -11, -32}
, {-36, -109, 85, 72, -6, -30, -43, 103, 36, 35, 82, 85, -57, 56, -24, 13, 14, 18, -3, 13, 74, -17, -10, 66, 79, 90, 26, -22, -16, 110, 28, -11, -44, -7, -35, -50, 84, -68, -58, -54, -10, 26, -29, 27, 76, -21, 23, -98, -42, 50, -27, 94, 45, 20, 87, -3, 76, 45, -79, 14, -51, 96, -1, -7, 46, 84, 27, -10, 52, 73, 43, 35, -24, 15, 63, 42, 74, -61, -58, 60, -89, -49, -29, -71, -8, 4, 17, 76, 105, 47, 43, 26, 41, 50, -43, -50, 74, 65, 95, -86, 53, 66, 85, -77, 64, 24, -10, 57, -87, -48, -90, -25, -56, 64, 37, 65, 50, -30, -99, 87, 0, 98, 70, -93, 36, 13, 55, 43}
, {25, -84, -12, -68, -14, -35, -80, 137, 19, -28, 17, 49, 37, -60, -92, -67, 85, 36, 85, 42, -27, 78, -26, 22, -94, 30, -96, -24, -32, 87, 98, -73, 49, -72, -99, -59, 72, 80, -37, -63, -109, -84, 24, 4, 97, 75, 50, 0, 71, 39, 38, -90, 33, -52, -7, -31, 66, 45, -86, 99, 98, 105, -51, -41, 38, 81, -42, -8, -69, 43, -29, 16, -85, -14, -70, 38, -17, 77, 51, -64, -84, -86, 86, -73, -42, -7, 67, -55, 75, 65, 41, 89, -67, 83, 83, 65, -1, 83, -76, 13, 99, -16, -10, 20, 12, -95, -58, 12, -66, -95, 4, -44, 39, 14, -70, 73, -85, -40, 10, 17, -30, 90, 27, 90, 92, -24, -42, -27}
, {62, 19, -11, 81, 16, -17, -14, 19, -43, -4, -63, 77, -38, 33, -63, -40, -39, 67, -7, 99, 30, 111, 72, 1, -18, -53, -16, -2, -21, -19, -23, -22, 72, 96, 94, 21, 48, -24, -15, -7, 24, 69, 95, -55, -79, 64, 53, 38, 8, 45, 59, -38, -48, -50, -27, -78, -33, -12, 43, 96, -29, 81, 4, -49, -57, -18, 92, -15, -51, -88, -41, 59, 79, 92, 88, 88, 64, 6, 51, 43, 34, -27, 94, -24, -78, 82, 11, 20, -30, -39, -91, 64, -48, -35, 94, 25, -9, -40, -57, 15, -82, -31, 30, 18, -15, -41, -90, 22, 93, -1, -58, -38, -101, -71, 42, 14, -49, 94, 113, 41, 89, -40, 54, -14, 51, -2, 13, -85}
, {-24, -30, 34, 30, -55, 91, -41, 26, 44, -31, -22, -25, 3, 50, 4, -85, -59, -86, 6, -7, -52, -63, -29, 24, 6, 40, 17, -69, -27, -4, 10, -58, 26, 34, 97, 39, 4, 8, 52, 4, 61, 41, -55, -50, 39, 9, 21, 30, 61, 91, -5, 95, -101, -14, -65, -46, 86, 85, -88, 63, -77, -48, -37, 52, -15, 56, -51, -75, 3, 18, -14, -82, -8, -71, -82, 36, -47, -69, 77, 5, 56, -10, -44, -36, -88, -88, 53, 94, 2, -12, 46, 25, -18, -1, 33, -7, 47, -34, 74, -26, 88, 76, -82, -16, -91, 47, 5, 41, -13, -72, -3, 15, 9, 37, 62, 16, -4, 24, 39, 47, -59, 75, -68, -93, 10, 5, 74, -47}
, {-61, -21, -46, -91, 66, 95, -99, -55, -12, -2, -101, 31, 61, 8, 34, 52, -104, 71, -51, -27, -16, 67, 15, 0, -20, -72, -50, 55, 2, -33, 26, 35, -62, 66, 22, 3, 107, 82, 26, 2, 48, -59, -39, 92, -34, -25, 0, -111, 3, 14, 35, -51, -53, 55, -28, -38, -34, -71, 79, -83, -22, 7, -86, -72, -58, 37, 62, 47, -25, 11, 59, 95, 34, 65, -31, -6, -54, -68, 10, 11, 48, -6, 33, -111, -35, 36, -90, 58, -3, -89, 52, 56, -2, -46, -20, -7, 40, -1, 0, -26, 27, 11, 74, -24, -25, 76, -9, -55, 104, -104, 55, -26, 40, -3, 4, 66, -28, -24, -26, 5, 25, -90, -86, 70, 116, -82, -4, -99}
, {20, -63, -82, -85, 88, -105, 87, 85, -35, 78, 6, 26, -42, -26, 37, 91, 105, 34, -109, -66, -29, -74, 90, -14, 90, 46, -32, -3, -101, -64, 80, -53, 69, 63, 80, 54, -58, -66, 49, -88, -26, -47, -90, 65, 86, 60, 86, 19, 46, 17, -96, -56, 14, 54, -24, -16, 19, 8, 92, -84, -90, -54, 43, 26, -40, 48, -48, -60, 13, 14, -69, -81, -21, -74, -26, 74, 99, 22, 15, -69, 12, -100, -28, 38, 50, -71, 43, 14, 17, 25, -46, 86, -54, 80, 21, -34, 57, -60, 33, -99, -6, 103, 0, -58, 66, -95, 83, -89, 6, 78, -37, 68, -78, 58, -37, -56, 35, -40, 43, 50, 67, -65, 64, -22, -82, 93, -99, -54}
, {45, -28, 17, -27, 78, 23, -75, 74, -75, 87, -24, 4, 14, -97, -15, 10, 71, -50, -63, -50, -16, 74, 39, 93, -20, 45, -87, 38, 79, -72, 39, 72, -6, 74, 4, -82, 86, -82, 47, 100, 88, -64, -34, -64, -39, 55, -79, 0, -82, -60, 12, -54, 73, -71, 47, 49, -69, -71, -70, -65, 27, -6, -33, -22, -79, 47, 82, 91, 21, -51, 73, 20, -77, -4, -58, -6, 29, 10, 54, 32, 2, 19, 23, -87, -94, -67, 67, -83, -18, 36, -26, 37, 21, 69, 90, -104, 59, 5, 27, -39, -57, 26, 32, -36, 61, -16, -48, -9, -45, 35, 10, 20, -10, 72, 18, -30, -27, 98, 92, 85, -47, -23, -98, 82, -97, 72, -61, -66}
, {2, -51, -43, -95, -14, -75, 36, 77, 29, -6, 13, 53, 82, 56, 35, -39, -12, -53, -69, -41, 65, 70, 76, -74, -17, -37, -87, 103, 25, -75, -13, -111, 72, -26, 34, 21, -18, -26, 26, -95, -44, -36, -95, 37, 45, -45, -19, 69, 77, -82, -79, -87, -94, -15, 42, 80, 81, 8, 9, -82, 20, -3, -56, -64, -36, 3, 87, -98, -18, 35, -70, 61, -80, 38, 14, 59, 16, 18, -57, 49, -9, -82, -19, 81, 67, 62, 38, -83, -39, 62, -20, 11, 2, 54, -44, 49, -57, -7, 26, -69, -72, 48, -25, 74, -86, -72, -67, -24, 98, -62, -95, 60, 74, -53, -12, 13, 83, -27, 37, -35, -25, -7, 66, 60, 11, -77, 45, 47}
, {39, -1, 78, -51, 78, -15, -48, 65, -76, 22, 12, 26, -57, 0, -54, -51, 24, 32, 82, -44, 27, -16, 47, 14, -64, -72, 28, 3, 38, 94, -25, -43, 4, -11, -5, 20, -23, -40, -47, -26, -8, 75, -72, -86, -45, -104, 29, -1, 15, 44, 33, -66, -38, 67, 92, 35, 60, -69, -44, -81, -3, -32, 31, -69, 37, -70, 19, 70, 32, -42, -32, 67, 50, -6, -82, 2, 95, -100, -75, 4, -97, -8, -78, -13, 31, -90, -13, -68, -35, 111, 82, 63, -42, 59, -15, 27, -30, -83, -18, 9, 96, 82, 105, 90, -79, -4, 17, -22, 38, 19, -87, -40, -15, -8, -86, -43, -1, 93, -55, 35, 29, -12, 91, -75, 68, -16, 0, 98}
, {19, 0, 48, 6, 88, 54, 46, 9, -29, 72, 87, 37, -95, 70, 19, 69, 57, -62, 33, 82, -13, -42, 75, 48, -88, 74, -16, -56, 31, 40, -80, 3, -80, 37, -86, -32, -51, 53, -8, -100, 22, -14, -27, -34, -1, 21, 57, -67, -20, 27, -70, 6, 2, -84, 28, -98, 4, -56, -53, -18, 86, -18, 21, -5, 77, -63, -1, 90, -75, -79, 71, 69, 81, -53, -6, 87, 64, -22, 61, 87, 27, 22, -50, -88, -64, -93, 78, 78, 41, -36, 0, -89, -8, 40, 5, -62, -50, 31, 18, -90, 0, -49, 31, 33, -57, -5, -20, -19, -1, -90, -26, 54, 27, 39, -62, -44, -35, -31, 88, -18, 46, 41, -82, -19, -17, 70, 68, -45}
, {72, 21, -95, -62, 37, 40, -50, -67, -31, 24, 25, 37, 4, 40, -27, 30, 78, -29, -76, -52, -10, 48, 79, -59, 46, 63, -3, 96, 40, -20, 83, -86, 54, 60, -55, 25, -76, -58, -5, 99, -37, 75, 88, -45, -53, -40, -70, 10, 33, 4, -14, -5, 27, -38, 34, 94, -95, 43, -95, 27, 64, 42, -99, -44, 16, -76, -69, -64, 55, -74, -94, -88, 18, -14, 0, 89, 15, -29, 32, -97, 15, 16, -13, 71, -70, 90, -21, -84, -12, 30, -99, -8, -42, 55, 69, -23, 75, 24, -2, -58, -96, -88, 53, 48, -24, 78, 57, -20, -46, 6, -91, -102, -102, 69, -33, -37, 7, 28, 56, -19, 67, 46, -82, 20, -78, -39, -22, 86}
, {15, -34, 51, 73, -81, -63, 17, -55, 0, -44, -74, 33, 72, -29, -54, -68, 0, 35, 20, -45, -101, 55, -5, 33, 23, -53, 30, 21, -38, 66, 15, 56, -78, -29, -80, -18, 54, -19, 92, -27, -54, -39, 67, -20, 73, -34, -73, -15, -94, -55, 5, -23, -53, -99, 7, -52, -51, -41, 21, -32, -79, 85, -4, 41, 42, -67, 21, 8, 37, 26, 14, 100, -46, 27, -81, -44, 104, 3, -68, 91, 74, -68, -35, 73, -17, -87, -36, 87, 87, -37, 47, 42, -24, -81, -53, -2, -73, 101, 42, -69, 21, -6, -68, 28, 59, -21, 1, 49, 4, -56, 9, -55, 43, -91, 35, 49, 46, -13, 12, 84, 4, 84, 59, -97, 45, 34, 55, -27}
, {86, 42, -45, 87, -55, 75, -65, 79, 58, 25, 58, -106, -71, -53, -58, -34, -67, 6, 40, -48, -49, -6, 55, -78, 72, 22, 9, 54, 99, 11, 31, 82, -57, -13, 78, 46, 69, -75, -86, -54, 94, 59, 69, 31, 20, -90, -79, 26, -113, 59, 66, -112, 54, -91, 26, 56, -90, -8, 36, -15, 17, -90, 50, -33, -95, -3, -41, -107, 6, 64, -42, 92, -35, 88, -79, -47, 21, -24, 93, 41, 12, 34, -11, 13, 21, 74, 25, -41, -84, 53, 0, -43, -70, -84, 21, 69, -80, 0, -89, -49, 19, -82, -22, 100, 81, 31, -55, 67, 73, 24, -99, 93, -35, -78, -47, 47, 65, 4, -15, -67, -76, -47, -10, 72, -41, 11, 119, 96}
, {-5, -48, 122, 38, 64, 7, 2, 57, 39, -106, 58, 74, 31, 33, -55, 19, 79, -47, 12, -37, -47, -63, -31, 39, -27, 69, 51, -44, 12, 49, 22, -2, -56, 0, 45, -50, -73, 58, -113, -100, -97, 86, 25, -90, 25, 83, -13, 13, 36, 46, 53, 111, 97, -31, -88, -14, -31, 23, 0, -84, 52, -1, -30, -9, -11, 88, -40, 85, -59, 18, 60, 23, 11, -89, 0, -78, 34, -14, -29, 4, -15, -19, 78, 29, -71, 26, -61, 81, -33, 29, 97, 28, -87, -33, 50, -11, 8, 60, 113, -34, -3, 98, -84, 14, 52, -76, 37, -55, -86, -59, 62, -81, 17, -63, 29, 94, 0, 61, 9, 18, -46, 35, -75, -33, 54, -25, -54, 73}
, {55, -19, -44, 24, -20, -22, -26, -6, -13, -24, 82, -75, -70, -97, 51, 72, -33, -64, 102, -85, 64, -9, -33, 50, 41, 76, -20, 1, -67, -54, 83, -87, 8, -28, 9, 6, -66, -57, -11, -25, -70, -72, -29, -9, 31, 75, -69, 40, 48, 54, 37, -49, -91, 40, 85, -36, 5, 45, -18, -23, 34, 9, -22, -53, -30, 97, -64, 96, -14, -28, 33, -13, -93, -64, -117, 63, 31, 77, 52, -17, 89, -74, -27, -10, -51, 75, -14, 76, -31, -70, 2, -40, 98, 82, 92, 57, -65, -7, 17, 75, -33, -3, -3, -97, 93, -25, -24, -12, -62, -93, -20, 23, 50, -87, -72, 34, -21, -47, 80, -91, -6, 47, 55, -95, -70, 15, -67, 31}
, {-90, -37, -28, 95, 75, 43, 73, 12, -60, 12, -3, -1, -31, 92, 28, -32, -28, 5, 15, -74, -69, 54, 79, -71, 43, -61, 14, 84, 24, 53, -3, 54, 64, 81, -26, 35, -88, -74, 60, -18, -6, 2, -79, 32, -77, -56, -66, -13, 33, 58, 58, 71, 58, 100, -70, -46, -69, 82, 90, -3, -29, -6, 78, -50, -59, -69, 30, 31, -94, 24, -66, 41, -80, -81, 20, 10, -59, -49, 55, -95, -92, 82, -13, -5, 62, 37, -62, -56, -75, 60, 41, 29, 12, -56, -33, 83, -40, -17, -90, 109, 54, 113, -95, -36, -51, -48, -32, -90, -74, 72, 0, -7, 2, -44, 1, -56, -85, 90, -97, -86, -39, 86, 51, 23, 84, -63, -38, -16}
, {-29, 20, -71, -18, 32, 83, 23, -52, 62, 29, -28, 29, 23, 97, 74, -45, -19, 0, 76, -85, 91, 37, 32, -46, -28, -105, 90, -6, 30, -22, -57, 1, 25, -67, -55, -97, -14, -64, 18, 53, 8, -76, 72, -95, 62, 82, 65, 100, -70, 34, -50, -19, 66, -35, 83, 86, 40, -3, -47, 6, 79, 45, 31, -42, 51, 85, -89, 16, -86, 40, -59, -70, 58, 18, -55, 34, -77, -88, -3, 76, -72, 31, 73, 32, 68, 61, 0, -52, 57, 43, 55, 42, -89, 21, -73, -9, 45, 37, -46, 32, 8, 15, -10, 45, -105, 37, 89, 44, 0, -70, 87, -28, -72, 70, -32, -73, -102, -52, -18, 24, -56, 19, 10, -57, 30, -22, 14, 66}
, {-91, 40, 12, 3, 30, -27, 8, 1, 67, -24, -24, -58, -48, -54, -94, 37, -49, 38, 60, -92, -72, 106, -3, 76, -46, 20, 0, -50, -75, 74, -55, -74, 26, 29, 49, 15, -81, 67, -47, -42, -41, 79, -14, 50, 90, -23, 38, 59, 72, 16, 78, -17, -34, 72, -24, -30, -43, -43, -4, -52, -31, -88, 15, -70, 97, 4, 10, -17, -70, -83, 16, -119, -23, 104, 62, 23, -109, 94, 44, -62, 44, -69, 100, -82, -53, 99, 47, -90, 59, 53, 12, -38, 0, 38, 58, -41, 60, 66, -73, 0, -5, 44, 37, -36, 66, 41, -60, -55, -77, -17, -59, 51, 12, -34, -33, 55, -56, -2, 64, -75, -5, 61, 19, -51, -27, 55, -3, -39}
, {86, 35, 27, 81, -1, 19, -42, -12, -80, -72, 18, 74, 59, -24, -89, -93, 2, 78, 48, 75, 1, 103, 41, 37, -12, 20, 89, 43, -32, 96, -30, -63, -83, 80, 4, 10, 39, 84, -82, -77, 33, -64, -72, -101, -67, 26, -26, 17, 2, -45, 72, 10, 100, 65, 90, 62, -44, 24, -6, 85, 5, -51, -2, -9, 61, 26, 71, -80, 101, 7, 44, -12, -25, 5, 53, 18, 55, 7, 2, 84, 26, -59, 31, -53, -40, 57, -14, -7, 32, 37, -52, -32, -82, -77, 73, -39, -79, -33, 2, 55, 58, -45, -53, -73, -40, -5, 17, 1, -74, 94, 77, 24, 72, -10, -71, 31, 30, -65, -58, -91, 11, 84, 78, 69, 34, 2, -18, -76}
, {-33, -75, -38, -74, 42, -76, -71, -15, 37, -51, -6, 26, -75, -23, 29, -46, -101, 55, 91, 90, -22, -20, -3, -92, -20, 79, -72, -53, -34, -38, 70, -68, -49, -62, -59, 68, 37, 74, 82, 69, -57, 36, -15, 38, -96, 8, -13, 20, 41, -86, -62, 87, 63, -12, -74, 81, 13, 11, 95, -96, 76, -8, 75, -21, 59, 77, 0, -51, -40, 4, -59, 72, -46, -19, -21, 55, -58, 63, 55, 21, 69, 9, -7, -92, 32, 80, 36, -54, -44, 16, -90, -8, -82, 84, 2, -25, 19, -5, -45, 29, 47, -44, -66, 17, 2, 56, 12, -15, 37, -50, -12, 1, 54, -92, -62, -62, -59, -4, -111, -18, -97, 31, 23, -70, -3, -18, 80, 64}
, {-41, 7, 82, 59, -3, -48, -22, 33, -94, -76, -58, -69, -47, 82, 0, 47, 5, -11, 82, -75, -59, -58, 72, -11, 32, 29, -1, 8, -3, 68, -60, 70, 71, -28, 18, -9, -101, -89, 35, 26, 66, 59, 56, -86, -67, 74, 63, -40, 39, 29, 51, 29, 22, 2, -22, 20, -79, 75, 32, 27, 29, 40, -46, -36, -33, 1, 22, 42, -97, -85, 45, -96, -6, -12, -54, -67, 106, -97, -47, -61, 88, 7, -5, 70, 92, 83, -13, -2, 39, 49, -11, -35, 16, 5, 75, -86, 75, -36, -94, 20, -29, 83, -26, 40, -72, 51, -71, 18, 77, -35, 34, -86, 19, -10, 43, 67, 66, 6, -12, -3, 32, 0, -45, -62, 109, 2, 26, 9}
, {-62, -66, -44, -30, 1, 68, 48, 19, 50, 4, 43, 78, 36, -73, -47, -54, -87, 59, 38, -79, -56, -8, -36, -70, 52, -78, 52, -70, 72, 69, -89, -51, -69, -17, 5, -29, 84, 20, -52, -17, -71, 27, 24, -32, 65, -81, 47, -30, -100, -47, -4, -34, -21, -89, 28, 15, 90, 40, 89, 24, -92, 7, -78, 10, -6, -52, -98, -30, 11, 54, 47, -50, -13, 54, 35, 2, 74, -21, -98, -67, -39, -58, 28, 16, -49, -66, 66, -77, 94, -56, -44, 70, -54, 75, -51, 31, 70, 85, -55, -91, 22, 49, 72, 33, -21, -33, -50, -64, 32, 82, -56, 54, 97, 17, -29, 64, 68, 42, -2, 61, -11, -81, 58, -42, -44, 5, 33, 65}
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
#define FC_UNITS 7
#define ACTIVATION_LINEAR

typedef number_t dense_59_output_type[FC_UNITS];

static inline void dense_59(
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
#define FC_UNITS 7


const int16_t dense_59_bias[FC_UNITS] = {9, 7, -1, -2, -19, -2, 3}
;

const int16_t dense_59_kernel[FC_UNITS][INPUT_SAMPLES] = {{-202, 178, -36, 48, 46, 186, 220, -1, 187, -81, 36, 42, -179, 132, -22, -18, -190, -127, -42, 37, -33, 73, 114, 137, 109, -86, -127, 112, -4, -16, -21, -155, 133, -62, -187, -16, -199, -108, -61, -167}
, {42, 169, 124, -162, -84, -37, 164, -31, -179, -39, 54, 154, -75, 31, 156, 0, -16, 205, 34, 156, 116, 93, -142, 3, -207, -108, 7, 45, -1, 98, 67, 98, 187, 14, -51, -22, -107, -8, -129, -158}
, {-14, -153, -151, -2, -174, 17, 114, -3, -106, 211, -29, -49, 65, -153, -85, 144, -155, -96, 122, 153, -111, 74, 141, -139, -181, 21, 165, -159, -123, 39, 134, -37, 52, 123, 31, -118, -129, 166, -6, 182}
, {61, -24, -177, 123, 170, 108, -53, 36, 195, -113, -151, -26, 130, 5, -14, 129, -125, 169, -161, -96, -78, -112, 153, -5, -82, -166, -36, -113, -6, -192, -107, 61, -128, 154, 45, -52, -172, 197, 44, -181}
, {14, -54, -160, -106, -48, -106, -142, 26, 75, 96, 142, 87, 90, 159, 34, -154, 6, -117, -138, 86, 167, -102, -157, -93, -53, -16, -65, -153, 150, 17, 138, -83, -130, -22, -80, 142, 113, -66, -67, -9}
, {-12, -72, -127, -7, -106, -65, -10, -8, 201, -78, -182, -130, -146, -110, 65, 96, 169, -108, -76, -176, 129, -6, 96, 10, -1, 95, 56, 124, 7, -33, 169, -173, -124, -151, -95, -204, 12, -124, -168, 57}
, {9, -160, -23, 149, 127, -60, -165, 194, 129, -90, -139, -6, 149, -181, -41, 146, -140, -40, 37, 123, 90, 2, 36, 81, 161, 80, 40, -84, 150, -134, -79, -14, -65, 105, 127, 67, 160, -180, 145, 20}
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

#define MODEL_OUTPUT_SAMPLES 7
#define MODEL_INPUT_SAMPLES 16000 // node 0 is InputLayer so use its output shape as input shape of the model
#define MODEL_INPUT_CHANNELS 1

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  //dense_59_output_type dense_59_output);
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
#include "max_pooling1d_145.c" // InputLayer is excluded
#include "conv1d_116.c"
#include "weights/conv1d_116.c" // InputLayer is excluded
#include "max_pooling1d_146.c" // InputLayer is excluded
#include "conv1d_117.c"
#include "weights/conv1d_117.c" // InputLayer is excluded
#include "max_pooling1d_147.c" // InputLayer is excluded
#include "conv1d_118.c"
#include "weights/conv1d_118.c" // InputLayer is excluded
#include "max_pooling1d_148.c" // InputLayer is excluded
#include "conv1d_119.c"
#include "weights/conv1d_119.c" // InputLayer is excluded
#include "max_pooling1d_149.c" // InputLayer is excluded
#include "average_pooling1d_29.c" // InputLayer is excluded
#include "flatten_29.c" // InputLayer is excluded
#include "dense_58.c"
#include "weights/dense_58.c" // InputLayer is excluded
#include "dense_59.c"
#include "weights/dense_59.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_59_output_type dense_59_output) {

  // Output array allocation
  static union {
    max_pooling1d_145_output_type max_pooling1d_145_output;
    max_pooling1d_146_output_type max_pooling1d_146_output;
    max_pooling1d_147_output_type max_pooling1d_147_output;
    max_pooling1d_148_output_type max_pooling1d_148_output;
    max_pooling1d_149_output_type max_pooling1d_149_output;
    dense_58_output_type dense_58_output;
  } activations1;

  static union {
    conv1d_116_output_type conv1d_116_output;
    conv1d_117_output_type conv1d_117_output;
    conv1d_118_output_type conv1d_118_output;
    conv1d_119_output_type conv1d_119_output;
    average_pooling1d_29_output_type average_pooling1d_29_output;
    flatten_29_output_type flatten_29_output;
  } activations2;


  //static union {
//
//    static input_30_output_type input_30_output;
//
//    static max_pooling1d_145_output_type max_pooling1d_145_output;
//
//    static conv1d_116_output_type conv1d_116_output;
//
//    static max_pooling1d_146_output_type max_pooling1d_146_output;
//
//    static conv1d_117_output_type conv1d_117_output;
//
//    static max_pooling1d_147_output_type max_pooling1d_147_output;
//
//    static conv1d_118_output_type conv1d_118_output;
//
//    static max_pooling1d_148_output_type max_pooling1d_148_output;
//
//    static conv1d_119_output_type conv1d_119_output;
//
//    static max_pooling1d_149_output_type max_pooling1d_149_output;
//
//    static average_pooling1d_29_output_type average_pooling1d_29_output;
//
//    static flatten_29_output_type flatten_29_output;
//
//    static dense_58_output_type dense_58_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d_145(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_145_output
  );
 // InputLayer is excluded 
  conv1d_116(
    
    activations1.max_pooling1d_145_output,
    conv1d_116_kernel,
    conv1d_116_bias,
    activations2.conv1d_116_output
  );
 // InputLayer is excluded 
  max_pooling1d_146(
    
    activations2.conv1d_116_output,
    activations1.max_pooling1d_146_output
  );
 // InputLayer is excluded 
  conv1d_117(
    
    activations1.max_pooling1d_146_output,
    conv1d_117_kernel,
    conv1d_117_bias,
    activations2.conv1d_117_output
  );
 // InputLayer is excluded 
  max_pooling1d_147(
    
    activations2.conv1d_117_output,
    activations1.max_pooling1d_147_output
  );
 // InputLayer is excluded 
  conv1d_118(
    
    activations1.max_pooling1d_147_output,
    conv1d_118_kernel,
    conv1d_118_bias,
    activations2.conv1d_118_output
  );
 // InputLayer is excluded 
  max_pooling1d_148(
    
    activations2.conv1d_118_output,
    activations1.max_pooling1d_148_output
  );
 // InputLayer is excluded 
  conv1d_119(
    
    activations1.max_pooling1d_148_output,
    conv1d_119_kernel,
    conv1d_119_bias,
    activations2.conv1d_119_output
  );
 // InputLayer is excluded 
  max_pooling1d_149(
    
    activations2.conv1d_119_output,
    activations1.max_pooling1d_149_output
  );
 // InputLayer is excluded 
  average_pooling1d_29(
    
    activations1.max_pooling1d_149_output,
    activations2.average_pooling1d_29_output
  );
 // InputLayer is excluded 
  flatten_29(
    
    activations2.average_pooling1d_29_output,
    activations2.flatten_29_output
  );
 // InputLayer is excluded 
  dense_58(
    
    activations2.flatten_29_output,
    dense_58_kernel,
    dense_58_bias,
    activations1.dense_58_output
  );
 // InputLayer is excluded 
  dense_59(
    
    activations1.dense_58_output,
    dense_59_kernel,
    dense_59_bias, // Last layer uses output passed as model parameter
    dense_59_output
  );

}
