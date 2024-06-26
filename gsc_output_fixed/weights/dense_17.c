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