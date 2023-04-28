/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 40
#define FC_UNITS 7


const int16_t dense_37_bias[FC_UNITS] = {12, 0, -10, 7, -28, 3, 9}
;

const int16_t dense_37_kernel[FC_UNITS][INPUT_SAMPLES] = {{-53, 128, -6, 133, -29, -165, -192, 91, -156, 98, 25, -200, -35, 151, 83, 165, 206, -14, 184, -104, -173, 110, -94, -57, -152, 65, -179, 141, 136, -48, -98, -13, -10, 177, -58, -112, 226, -165, -171, 48}
, {-103, 176, 166, -61, -189, -69, -146, 179, 71, 46, -110, -141, 156, 0, 77, -64, 49, -73, 99, 123, -90, 134, -97, 188, -149, 133, -176, 204, -182, -174, -171, 35, -136, -185, 212, 58, 190, 82, -150, -170}
, {136, -141, 65, -57, -182, 97, -169, 17, -111, -153, 113, 8, 69, -40, 201, 75, -169, 10, 188, -45, 146, 17, 54, 62, -100, -196, -113, 171, 96, 105, 133, 170, -127, -90, -181, -103, -80, 148, -155, 108}
, {-39, 133, -19, -170, 166, -142, 136, 106, 42, 31, -145, 89, -86, 42, 29, 50, -34, -50, 139, 44, -160, 103, 153, 118, -142, -37, 97, -176, -143, -201, 119, -134, 103, 83, -171, -71, -132, -74, -136, -81}
, {186, 60, 47, 143, 145, -192, -129, 102, 183, -61, 7, -7, 57, 132, -191, -165, -166, -176, 136, -7, -173, 166, 45, -161, -190, 182, -140, -22, -90, -21, -40, -23, -187, -28, -138, 90, -150, -214, 65, -185}
, {132, 110, -56, 78, -87, 105, 102, -89, -112, 143, -49, 167, 24, -53, -114, -120, 14, 23, 101, 26, 163, -52, 166, -175, 119, -133, 169, -77, -109, 228, -138, -157, 120, 144, -32, 171, -124, 134, -118, -156}
, {173, 180, 39, -126, -134, 64, 174, -73, 108, -174, 177, 25, -166, -152, -6, 21, 77, 126, -71, -139, 45, -99, -92, 70, 56, 89, -141, 72, 112, 106, -210, 79, 165, -172, 113, 77, -9, -38, -157, -117}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS