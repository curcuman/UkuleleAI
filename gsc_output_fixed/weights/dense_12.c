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


const int16_t dense_12_bias[FC_UNITS] = {-3, 0, -3, 1, -1, -3, -2, 1, 1, 0, 2, -2, 1, 2, 2, 0, 3, 2, 1, -3, 1, 3, 3, 1, -2, 0, 2, 3, -2, 2, 0, 2, 0, 0, 2, -2, 3, 0, 1, -2}
;

const int16_t dense_12_kernel[FC_UNITS][INPUT_SAMPLES] = {{-75, -74, 101, 29, -29, 62, -64, -12, -44, 1, 3, 97, 101, 51, 49, -54, -87, 50, 59, -35, -70, 11, -72, -60, 34, -60, 16, 83, -85, 23, 81, 9, 96, 28, -69, 84, 51, -68, 79, 15, 86, 51, 67, -6, -29, 10, -69, 21, 1, -19, -48, 91, 72, 70, 99, -98, -75, 41, 0, 3, -14, 54, 85, -8, -20, 27, 2, 21, -3, 46, 7, -3, 30, -58, -64, -75, 75, -80, 60, 23, -19, -13, 82, 69, -53, -9, 30, 63, -91, 52, 71, 35, 10, -38, -54, 0, 4, 11, -37, -84, 93, -27, -3, -32, 90, 16, -43, 46, -79, 84, -74, -15, 0, 54, -50, 94, -6, -56, 85, -75, -14, 8, -58, 80, 53, -51, 49, -58}
, {-23, 52, 41, -90, -66, -21, 6, -57, 1, 31, 20, 34, 76, 26, 57, -44, 12, 29, -80, -51, -27, 3, 58, -73, -28, -69, 4, -61, 99, 28, -33, -32, -9, -15, -62, -9, 52, -36, -49, 28, 41, -66, 38, -90, 7, 98, -20, -92, 103, -34, -82, -32, 66, 89, -2, 81, -55, 80, -41, -37, 45, 51, 62, -45, -40, 69, -19, -78, 87, -56, -91, 50, 24, 96, 41, -22, -62, 93, -73, -75, -78, -46, 77, 84, 55, 94, -8, 90, -91, 6, 32, 33, -88, -90, -24, -8, -95, -51, 42, 27, -56, 109, -9, -76, 45, -39, -35, -45, 63, 34, 43, -91, -89, 96, -71, 94, -36, -64, 40, -66, 69, -87, 49, 75, -73, 6, -41, 76}
, {-19, -40, -8, -57, -55, 46, 56, -53, 11, -61, 82, 85, -27, -84, -60, 35, 13, -85, 23, 52, -77, -61, -89, 86, -4, 7, -41, 59, 11, -61, 18, 62, 94, -76, 92, -77, 6, 3, 75, 64, -9, -14, 48, 60, 55, 71, -83, -2, -83, -78, 42, -9, 82, -1, -48, 27, -10, -15, 43, -51, 11, 5, -55, 68, -33, 27, -45, -60, 37, -15, -91, -33, 35, -57, 0, -77, 13, 79, 60, 97, 53, -15, -34, -81, -84, 80, 56, 46, 75, -17, 94, -71, 17, 26, -57, -73, 41, -66, 68, -1, -62, -58, 90, -4, -38, -54, -78, -64, -81, 23, -65, 9, -8, -22, 37, -52, -94, 22, 24, 0, 94, 12, -24, 86, -53, 41, 82, -25}
, {-9, 41, 26, 88, -55, -79, 0, 36, 63, -17, 12, -46, 19, 20, 21, 88, 56, -70, -4, 12, 57, 17, 46, -46, -47, -43, 12, 21, 57, -74, 1, 47, -41, -89, 6, -60, 51, -8, 25, -70, 2, 77, 29, -70, 2, 21, 64, 38, 10, -65, 53, 34, 78, 65, -42, 6, 65, -61, -71, 50, 91, 4, 38, -24, -39, -31, 54, 11, -74, -48, -79, 64, 90, -8, -22, -7, 55, -80, -16, 63, 51, 31, 9, 6, 52, 51, 62, -5, -2, 31, -98, -20, 66, -51, -61, -94, -17, 41, -70, 65, 3, 71, 26, 68, 60, -35, 2, -47, 91, -19, 82, -27, 55, 84, 48, -30, -90, -52, -77, -58, 12, -90, 45, -93, 10, -74, -9, 83}
, {3, -38, 10, -23, 5, -79, 87, -14, -73, 63, 1, 23, -3, -62, -32, 99, 0, -45, -66, 52, 5, -51, -36, -82, -29, 7, 9, 83, 93, 1, -32, -20, 54, 1, 75, -2, 65, -28, 74, -25, -67, 88, -82, 14, -32, -19, 65, 4, 41, 72, -90, -8, 33, -65, 23, 99, 15, 54, 41, 78, -40, -33, 12, -6, 68, 83, 41, -56, 12, 58, 23, -60, -79, 0, 48, -58, 13, 63, -40, 17, 29, 94, -6, -18, -11, -29, 77, 51, -75, 81, 8, -34, 14, 51, 85, 52, -92, 64, 34, 64, -74, 84, 66, 54, 18, 52, -54, 12, 6, -5, 51, 2, -27, 30, -81, 102, -77, 11, -64, -18, 33, 6, -92, 95, 72, 23, -75, -18}
, {-91, 15, 78, 84, 21, 77, -73, -72, 75, -13, -12, -17, 30, 13, 20, 3, -31, -62, 61, -76, 94, 24, 64, -43, 20, -56, -22, 90, 8, 67, 90, 1, -16, 10, -51, 61, 15, -23, -64, 5, -50, -95, 88, -9, -13, 102, -83, -87, 62, -93, 94, 59, 48, -4, -9, 21, 60, -17, -19, -88, 78, -71, 90, 58, -80, 51, -98, 34, 40, -30, 80, -25, -40, -61, 86, 29, 65, -52, 94, -54, 64, -44, 60, -40, 35, 72, -12, -67, -85, 55, 18, -8, 51, 16, -6, -42, -52, -7, 78, -36, -1, 53, 12, -94, -83, 20, 30, -3, -64, 80, 24, -50, 0, 65, 34, 56, 74, -21, -24, 19, 24, 38, -92, -25, 24, -65, -8, 50}
, {0, -44, 40, -41, -60, 2, 81, -54, -57, -85, -44, 7, 100, 60, 32, -62, -50, -19, -53, -57, -65, -85, 27, 20, -29, -89, 48, -19, 76, 0, -65, -7, -88, 30, -16, 46, 89, 15, -65, 14, 3, -63, 53, -82, -65, 28, -31, 87, -59, -11, -24, -6, -59, -99, -25, -26, -17, -93, 17, 92, 59, -85, 10, 78, -70, -67, -45, 21, -28, 7, 42, -67, 43, 72, -92, -10, 17, 70, 8, 85, 85, -25, 20, -2, 23, -4, 67, 46, 5, 93, -39, 15, 64, 45, -49, 29, 60, 52, -87, -93, -42, -19, 59, 41, -69, -62, 84, -14, -33, 9, 83, 49, 56, -94, -71, 32, 30, 40, -18, -79, 12, -83, -7, -63, 7, -14, -58, 86}
, {-23, -76, 55, -93, 57, 9, 49, 4, 79, -51, -79, -68, -41, 41, 53, 34, -6, 60, 53, -81, -4, 86, 26, 95, -33, 65, 6, 100, 32, 92, -57, -71, 80, 26, 50, 19, -92, 2, 2, -47, -70, -67, 47, -68, 33, -46, -34, -13, 49, 47, -67, 43, -37, 67, 84, -73, -41, 99, -82, 85, -22, -84, -4, 90, -16, 49, 55, 68, 42, -86, -74, 96, -41, 42, -57, -67, 2, 7, 12, 11, -49, 42, 8, 40, 3, -56, 0, -33, 88, 98, -86, -45, -24, 53, -47, 69, 33, -54, 89, -24, 0, -81, -30, 49, 95, -45, -85, 32, 86, 11, 49, -59, 74, 41, 91, 62, 70, 56, 46, -62, -24, 77, 5, -67, 44, -20, -10, -68}
, {52, 78, 51, 96, -14, 86, -94, -14, -6, 72, 73, 80, 72, -30, -37, -52, 69, -93, -28, -67, -10, 18, 63, 83, -30, -80, -99, -62, 91, -49, 77, -53, -16, -73, 71, 6, -13, -12, 89, -11, -23, 78, -42, -20, 44, 84, 9, -45, 68, -49, 59, -35, 13, -78, 50, -23, -9, 90, -62, -18, 11, -82, -2, 71, 3, -73, -55, -66, 5, -24, -76, -40, 33, -89, -18, 40, 34, -50, -66, 59, 57, 52, -49, -64, 43, -16, -8, -40, -26, -67, -49, 34, -19, -50, 82, 75, -91, -47, -74, -10, 42, -29, -69, 14, -52, -48, -13, -75, 92, 31, 17, 13, -89, -88, -29, 63, 36, 58, -72, -75, -90, -29, 42, 86, -65, -44, -45, 62}
, {-30, 72, 73, -56, -21, 49, -10, 46, 91, 85, 14, -21, -12, 79, -74, -7, -94, -61, -21, 87, 67, 74, -71, -64, 9, 83, 74, 6, -71, -71, 76, -3, 36, 85, -84, 17, -65, 20, 60, 33, 11, 35, 51, 26, 13, 11, 4, 35, 88, -34, -51, -81, -5, -78, -16, 2, -29, 5, 86, -3, 82, -60, 38, -37, 17, 65, -45, 14, 34, 48, 6, -58, -69, 61, 55, -63, 52, -52, -2, -33, -36, -9, -5, -21, -78, -92, -76, -42, -15, 31, 80, 28, 26, -49, -51, -12, 6, 99, -83, -10, 8, 49, -95, -87, -78, -43, -63, -8, 45, 43, 41, 25, 83, -66, -9, 23, -9, -19, -1, -57, -50, -24, -6, 86, -36, 44, -21, -98}
, {-41, 80, 64, -76, 73, 10, 5, 15, -48, -11, 24, -25, -23, 53, -69, 63, -53, -85, -52, -76, 32, 71, 65, -25, -52, -24, -11, -63, 74, -44, 21, 68, 3, 84, -68, -43, 4, -34, -80, -82, -69, -69, -47, -62, 12, 94, -37, -72, 54, 79, 47, 19, 36, -78, 71, 56, 34, -29, -4, -27, 15, -24, -93, -48, 86, -26, 84, -7, -55, 74, 57, -61, -13, 76, 84, -76, -51, 14, 65, 19, 83, 97, -72, -78, 67, -19, 67, -56, 2, -35, 30, -11, 55, -7, -23, 49, -78, -15, 30, -85, -91, -57, 47, 42, -36, 25, -23, 73, 4, 3, -62, 21, 29, 97, 95, 61, 96, 45, -20, -29, -45, 61, 17, 86, -39, -31, 18, -93}
, {-31, -37, -37, 99, 7, -88, -20, 43, -60, 52, 35, -80, 12, 11, 90, 93, 61, -73, -93, -38, 57, 90, -82, -36, 43, 29, 46, -18, -50, 42, -43, 66, 11, -15, 88, 34, 29, -85, 6, -88, -2, 33, -9, 100, 71, 92, -59, -49, -69, -46, -88, -84, -5, -87, -80, 86, -34, 32, 65, 1, 37, -6, -10, -50, 9, 56, 40, -12, 15, 80, -2, 36, -81, 46, -77, 32, -12, 75, -20, 1, -45, -73, 55, 51, 14, -93, -42, 2, 90, 94, 72, 95, 36, -39, 29, 20, 68, -70, 13, 64, 87, -83, 56, 73, -67, 0, -49, 25, -43, -41, 19, 10, 30, 0, 38, -20, -90, 95, 43, -43, -42, 56, 35, -43, 58, -6, 60, -42}
, {-95, 11, -40, -16, 0, -80, -44, -20, -44, 6, -63, -74, -21, 61, 48, -5, 25, -2, -77, -49, -35, 66, 92, 61, 18, 55, -83, -67, -12, -51, -1, 47, -88, -47, -67, 21, 15, -13, -33, 76, -12, 36, -63, 76, -9, -36, 80, -8, 29, 65, -50, -21, -84, 27, -90, 75, 73, -11, 90, -97, -25, 46, -30, 7, 6, 55, -66, -54, 32, -48, 53, -16, 92, 24, -22, 91, -97, -22, 42, -50, -4, -40, 78, 43, 61, 8, -50, -34, -55, -93, -49, 72, -59, -14, 74, -41, -12, -5, -32, -70, -47, -58, -61, -60, -61, 98, 65, 39, 70, -103, 58, -71, 38, -83, 28, -8, -85, 37, 76, 28, -95, -37, 7, 75, -102, 97, -78, 35}
, {56, -2, 22, 13, -53, 83, 35, 15, 51, -58, 23, -49, 28, 79, 76, 94, -51, -69, -92, -69, -92, -45, -56, 19, -97, 56, -26, -29, -2, 69, -52, 10, -21, -39, 44, 22, -81, 15, -52, -31, -20, 66, 40, 85, 44, 36, -68, -40, -31, 5, -86, -21, -67, 51, 13, -15, -63, -9, -7, -66, -51, 51, -65, -81, -91, 3, 13, 27, 94, -23, -63, 52, -5, 101, 20, -22, -35, -62, 76, 67, 13, 85, -70, 1, 18, -74, 69, 28, 19, 49, 0, 59, 34, 36, 15, -19, -21, 66, 63, 44, -60, 86, -31, 55, -92, -9, -1, 61, -86, 31, 39, -78, -29, -19, 79, 0, -23, 11, -31, 53, 18, -27, -17, 1, 75, -3, 2, 90}
, {-4, 93, -45, -90, 82, 20, 49, 15, -61, 6, -53, 25, -8, 64, 70, -73, 68, -27, 39, 62, -65, -57, 8, 77, 5, 18, -50, -28, -5, -25, -68, 5, -15, 92, 21, -10, -54, -38, 14, -79, 10, -72, 74, -19, -44, -3, 98, 52, 89, -81, 69, -4, 65, 42, -76, 40, -15, -14, -61, -85, 12, 47, 85, 5, 35, -20, 7, 80, 73, 2, 16, -2, -38, 31, -69, 13, 61, 68, 78, 68, 88, 27, 84, 82, -19, -79, -23, 42, -35, -21, -93, -15, -49, 30, 31, 10, 46, 80, -39, 78, -50, -29, 46, -75, -49, 14, -17, -12, 19, -24, 41, 40, -91, -56, -25, 39, 24, 44, 88, -65, -33, 79, 92, 90, 47, -12, 30, -9}
, {-20, 86, 67, -3, -78, -69, 37, -15, 42, 33, 32, 53, -17, 89, 84, -30, 48, -85, -10, -91, -43, -22, 9, -94, -24, -88, -68, -17, -76, 13, -44, -51, 63, -69, 75, 57, 33, -14, 63, -12, 63, 82, 16, 18, -60, 84, -74, 22, -80, 89, 15, 1, -38, -55, 21, 75, 87, 60, -47, -63, -49, -91, 27, 49, -29, -68, 29, -14, 15, -26, -32, 26, -53, 81, -89, -35, 15, -20, -93, -5, 78, 19, -29, 61, 44, -35, -68, -30, 16, 75, 37, -23, -26, -17, -75, 16, 86, -10, 28, -29, 12, 4, -32, 76, 32, -14, -71, 80, 30, 71, -6, 82, -101, -92, -25, 59, 72, 43, 36, 3, 71, 35, -68, -82, 82, 54, -48, 16}
, {89, -81, -26, 81, -35, 62, 5, -2, -76, -92, -4, -49, -99, -12, -23, 1, 7, -25, -50, -2, -14, 61, 2, 28, 28, 5, -90, 50, 67, -3, -21, 8, -70, -92, 88, -83, 7, -25, -63, -17, 101, -26, 38, 45, 2, -16, -61, -3, 30, -37, 12, 34, -61, 63, -21, 25, -63, 1, -82, -14, 78, -29, 4, -10, -63, -87, -19, 39, -26, -76, -31, 71, 67, -8, -60, -99, -44, 77, -38, -46, -62, -76, -42, -22, 27, 58, 71, 89, -32, 28, 70, -34, 42, 3, -40, -23, -34, -88, -76, -83, 61, 85, 94, 67, -79, -42, 35, 101, -27, -17, -62, -38, 41, -73, 3, -20, 22, 54, -79, -70, 20, 28, 30, 30, 1, 13, -70, -31}
, {78, -53, -12, -17, -67, 78, 74, -88, -25, 15, -60, 10, -60, 90, 52, -46, -10, 21, -48, 79, -36, -13, -54, -30, -44, 67, -77, 45, -40, -6, -42, -55, 13, 86, -61, 48, 50, -43, -77, -83, 34, -7, 66, -6, -60, 28, 54, -11, -85, -26, 14, -84, -27, 66, -81, 99, 62, -97, -52, 51, 61, -73, 6, 1, -21, 83, 31, -94, -6, -66, 82, -60, -4, 23, 54, 38, -45, 21, -49, -77, 2, -38, 79, 52, 86, 79, 6, 27, 55, 8, 44, 90, 77, -93, 7, -29, -25, -61, 24, 26, 80, 82, -43, -86, -58, 61, 55, 68, 45, 25, -32, 75, -21, 28, -47, -3, -48, 25, 0, -59, 9, 64, 71, -90, 12, 31, 75, 71}
, {34, 42, 15, 13, 81, 12, 88, 79, -8, -57, -8, -23, 52, -83, 94, -53, 48, 86, 91, 94, -34, -62, 8, -52, -49, -47, 89, 78, 68, -97, -77, -24, -79, -51, -65, 71, 53, 8, 2, -16, -6, -66, 28, 36, 32, -99, 42, 62, 65, -56, 30, 5, -90, 78, 22, -31, -17, 49, 66, 24, -59, 76, -58, -33, 75, 29, -57, 11, 0, -99, 63, 18, -70, 62, 8, -53, -30, -69, -65, 43, 91, -95, 64, -28, -88, -51, 10, -20, -81, -72, 72, -94, -1, 4, -65, 78, -64, 44, -61, -93, -84, -67, -43, 39, 53, -41, 59, 91, 10, -82, -22, 5, 75, -29, 55, 43, -39, -66, 4, 70, 16, 5, 38, 41, 25, 37, 0, 52}
, {-61, 72, 96, 51, 47, -79, -70, 29, -58, -60, -72, 74, 49, -74, 29, -87, 80, -43, 39, -45, 60, -52, -3, -27, 91, 57, 1, -73, 71, -27, 74, 58, 81, 50, -39, 0, -8, -5, 99, 57, 61, -74, 39, 38, -46, 61, -19, 87, 42, -12, 44, 63, -32, -54, 84, 65, 38, 45, -37, -84, 40, -59, 31, 67, 17, 46, 26, -63, -18, -42, -40, 31, -12, -14, -17, 42, 47, 75, 29, -94, 69, 9, -94, -74, 21, -83, -71, 34, -32, -47, 69, -22, 62, 43, 57, -76, -60, 10, 30, 62, -2, 55, -9, -20, 89, -23, -39, 62, -85, -43, 33, -47, 80, -93, -84, 50, -77, -61, 30, 4, 27, -19, -71, -65, -11, 34, 47, 50}
, {-83, -22, 46, -54, 78, 82, 58, 28, 25, -75, -98, 63, 64, 27, 87, -49, 28, -98, 9, -91, 83, 47, 74, -52, -47, -37, 73, -84, 81, -3, -36, -51, 44, -37, 8, 39, -88, 70, -65, 18, -33, 22, -5, -31, 89, 79, 33, 31, -76, -65, 46, -95, 9, 53, -58, -74, 88, 56, 77, -7, -88, -1, -22, -52, -43, 18, 82, -58, -2, 74, -66, -4, -85, -32, 78, 18, 11, -15, -19, 35, 84, 25, 50, 1, -66, -25, 36, -86, 34, -45, 38, -73, 47, 4, 58, -81, 89, -17, -71, 75, 46, -62, -54, 0, -17, -83, -82, 48, -18, -97, -38, 69, 7, 25, -50, 9, 12, 22, -49, -10, -16, 63, -79, -25, 32, -33, 9, -77}
, {-16, -22, -11, 0, 56, -28, 9, 82, -16, 23, 51, -9, 80, -22, -8, -29, -58, -85, -20, -42, -12, 85, 1, 54, 24, 37, 58, 75, 35, -30, 59, -87, -76, 79, 49, -64, -72, 12, -66, -27, 55, 56, 1, -26, -22, -35, -20, 27, -64, -46, 86, -21, 23, -85, 46, -26, 24, 63, 33, 77, 59, 4, -61, -66, 8, 6, -53, -25, 86, -77, 89, 92, 85, 75, 70, 40, -76, -87, -33, -90, 12, 65, -60, 7, -58, 90, 35, -14, 0, -10, 23, -62, -101, -58, 13, -47, -62, -91, -85, 86, 27, -52, -25, 12, 82, 25, 52, 55, 48, -64, 84, 68, -11, -5, 0, -27, -67, 24, 96, -95, 78, 53, -86, -65, 22, 9, -20, -40}
, {60, -20, 13, 65, -75, 37, -14, -52, -42, 23, 31, 19, 10, 82, -14, -32, 40, 54, -2, -86, 83, -2, -69, 86, 64, -18, -84, 99, 16, -67, 75, 51, 40, -92, 98, -75, 65, 36, -17, -96, 80, 70, -64, 24, -12, -85, 71, -74, -87, -39, 53, -19, 36, -43, -77, 88, 10, -17, -1, -56, 46, -42, 25, 83, -66, 48, -58, 0, -53, 82, -36, -90, -60, 91, 1, -3, -93, -84, 29, 55, 72, -29, -17, -33, 75, -50, 8, 76, 45, 75, 49, 8, -83, 27, 78, 13, -56, 43, 55, 39, -56, 67, 8, 71, -12, 17, -43, 43, 59, -81, -16, -37, -66, -52, -68, -68, 78, -29, -80, -54, -16, 46, -24, -54, -19, -28, -67, 18}
, {18, -60, -48, 58, -81, -55, 69, -77, 7, -58, -49, 89, 22, -4, -90, 39, -52, -23, 68, -86, 65, 67, 28, -67, -87, -16, 57, -1, -20, 90, -61, -68, 71, -70, 93, -91, -25, -85, -29, -98, -86, 11, 53, 74, -15, -80, 65, -78, 3, 14, 4, -77, 59, 37, -9, -20, 31, 75, 8, -60, -20, -7, 72, -98, -31, -65, 39, -33, 5, -19, 71, 77, -50, -8, -69, -37, 93, -43, -21, -82, -55, -58, -48, 50, 0, 97, -30, 50, -90, -72, 17, 68, 31, -68, -31, 94, -81, 93, -30, -20, 67, -16, 89, 62, 89, -37, 31, -26, 19, -50, -45, -22, 94, 20, 28, 86, -58, 53, 69, -8, -52, -45, -32, -5, 22, 97, -40, 70}
, {11, 28, -65, 48, 97, 52, -87, 52, 37, -78, -43, -90, -9, 69, 26, -30, -83, 63, 9, -66, -66, 31, 57, -30, 47, -77, -67, -3, -23, -64, -11, -36, 84, -82, 0, -36, 14, -95, -24, 45, -95, 83, 20, -20, 92, -67, 71, 73, -41, -35, -72, -49, -40, -13, -56, 83, -54, -61, 2, -81, 50, -5, -44, 58, 1, -32, 6, -19, 88, 48, 70, 26, -49, -16, 92, -78, 5, -74, 29, -34, -60, -6, 84, 67, -20, 69, 40, -91, -62, 35, 52, 78, -68, 68, -27, 17, -77, -26, 48, -79, 19, 104, -79, 4, -54, -84, -42, 92, -20, -14, -75, -7, 38, 34, -8, 62, -88, 8, -92, 39, -61, 46, 51, -92, 51, -3, -17, -24}
, {-50, 7, 49, 24, 86, 44, -3, 41, -47, 94, 61, 56, -15, -49, 24, 71, 51, 89, -72, 85, -26, 24, 58, 31, -1, 81, 59, -22, -91, 61, 23, -66, 46, -63, 70, -57, -93, 19, -96, -86, -18, -82, 69, 50, -31, 13, 46, -47, 12, -54, 92, 83, -17, 46, -88, 27, 41, -53, -23, 65, 84, 8, 41, -59, 27, 94, -88, -15, 95, -53, 14, 93, 67, 23, -37, 74, -28, 78, -96, -72, 87, 52, 64, -3, 83, -34, -23, 48, 89, 57, -22, 94, -84, 16, -30, 28, 63, -79, -24, 14, 77, -49, -43, -92, 69, 41, 63, 12, -1, 65, 60, 49, -16, -58, -23, 17, -10, -5, -73, -24, -89, 52, 51, 91, -15, 18, -5, -4}
, {15, 81, -77, -84, 39, 33, -92, 63, 82, -85, -70, -55, -98, 90, -73, 91, 0, -38, -23, -14, 83, 54, -18, 67, -42, 78, 102, 103, -22, 66, -85, -45, 14, 16, -10, -56, -94, 80, 29, 54, 82, -86, -38, 15, 40, -31, 13, -13, 4, -2, 41, 77, 42, -82, -38, 86, 87, -30, -64, 81, -54, -25, 32, 64, -29, -28, 54, 46, -96, 47, 68, 60, 44, 89, -88, 56, 89, 76, 17, -55, -25, -71, -84, -11, 47, -70, -90, -80, 1, 73, -89, -86, 71, -45, 0, -46, -19, -43, -48, -7, 20, 49, 49, 72, -33, -58, 47, 94, 51, 73, 1, -96, -45, -44, 84, 25, -76, 7, 75, -25, 42, 15, 76, -23, -44, -33, 65, -14}
, {-93, -74, -76, -66, -81, 88, 81, -6, 69, -85, -13, 46, -42, -50, 0, 80, 15, 0, 31, -58, 14, 78, -95, -65, 82, -68, 72, 38, -77, -77, 16, 67, -25, -32, -77, -35, 58, -28, 54, -87, -9, -88, -92, -52, 31, 3, -32, 94, -65, 38, -21, -72, 54, -46, 89, 56, -46, 29, 38, -56, -27, -82, 63, 63, 20, -70, 82, -39, 79, -82, -79, -74, 59, 64, 96, -91, 88, 36, -12, 14, -5, -86, -51, 21, -21, 8, -51, 82, -23, -45, 92, 49, 80, 24, -98, 23, 59, -97, -40, 47, 70, 0, -64, 76, -62, -8, -4, -60, -81, 69, -70, -51, 78, 38, -52, 54, -4, -35, -88, -68, -14, -12, -37, 88, 81, -21, 14, -94}
, {84, 29, -55, 83, 67, 56, -61, -34, 48, 2, -39, 88, 53, 57, -94, -3, -69, 49, -42, 22, 36, -79, -91, 88, 51, -34, 36, 83, 83, -95, -24, -44, 75, -44, -97, -21, -30, -2, -15, -46, 41, -26, -58, -40, -31, 83, -32, -25, 0, -44, -53, 74, -74, 2, -61, 65, -75, 23, 36, -60, -35, -9, 25, -43, 8, 63, -90, 61, -92, 79, -85, 49, -63, 50, -86, -89, -73, 3, 2, 73, -68, 99, -82, 35, -68, -31, 51, -95, 60, 22, -57, 48, 85, -35, 87, -5, -58, -25, -85, -22, 44, 66, -1, -9, -41, -11, -61, -18, 65, 5, -83, 23, -39, -40, 8, 84, -69, 34, -17, 50, 7, -36, 82, -26, 10, -79, -62, -90}
, {-83, -58, -13, -35, -54, 67, -69, -56, -95, 28, -69, 1, 47, -20, 42, -96, -77, 92, 32, 46, 75, -5, 32, 91, 44, 11, -97, -26, -43, -6, 66, -87, 70, 41, 47, 15, 7, -31, -3, -26, 46, 20, 69, 4, -81, -48, -71, 39, 65, 9, 77, -22, 93, 17, -101, -78, 63, 93, -2, -53, 0, -51, -62, -75, 49, 36, -73, 23, -63, -35, -24, -86, -88, 15, 87, 2, -93, 27, -3, 4, -55, -12, 2, 8, -55, -63, -51, 39, 63, -88, 75, 2, 24, -56, -17, 33, -92, -59, 19, -68, -21, -71, 69, -34, 28, 35, 26, -71, 62, 43, 45, -67, 59, -78, -75, -24, -3, -16, 82, -1, -99, -12, -11, 44, 22, 52, -76, -10}
, {50, -2, -12, 32, -51, 73, -23, -81, 42, 96, 57, -53, -85, 34, 86, -6, -6, -32, 45, 37, 14, -17, 47, 78, -76, 24, -25, 72, 33, -84, 9, 46, 20, -73, -86, -64, -81, 45, 0, -95, -32, -24, 25, -30, -14, 31, 0, -66, -20, 92, -45, -20, 54, 46, -27, 68, -90, 35, 24, -51, 33, 47, -39, 58, -1, -43, -40, 3, -37, 58, -16, 40, 20, -33, 54, -45, -88, 72, 91, 11, 32, 0, -61, -50, -8, 53, -71, 81, 31, -28, 0, -41, -13, 62, 101, 8, 87, 60, -10, 70, -18, -11, 68, -83, -12, 86, -77, 62, 40, 42, 73, 94, -32, 73, 68, 84, -47, 27, -41, 27, -53, 84, -13, 82, -58, 33, 92, -92}
, {-29, -2, 17, 65, -82, 57, -36, 2, 20, 20, -90, -62, 39, -70, -40, -7, 6, 59, 45, 90, -20, 40, 37, 28, -41, 60, 25, 90, -37, -15, -10, 1, -77, 2, 28, 56, 96, -72, -66, -5, -68, 49, -68, -48, 28, -27, -17, 88, 8, 93, -86, 77, -6, 73, 18, 51, 14, 46, 80, 10, -76, -70, -20, 87, 6, -16, 10, 64, 35, 53, -45, -92, -78, -70, -94, 46, 3, -78, -44, 40, -68, 13, -22, 13, 71, 1, 20, -49, -95, -23, 14, 2, -30, 67, -21, 96, 13, -20, 0, 34, 10, -39, -93, -90, -64, 57, -93, 35, -60, -21, -83, 11, -29, 18, -19, -28, -51, 25, 5, -18, -68, -62, -80, 45, -24, -85, 12, -92}
, {-71, -77, -94, 73, 60, 3, -38, 69, -62, 10, 54, -73, 9, -39, 26, -20, 75, 36, -96, 62, 78, 10, 24, 41, 78, -47, -39, 99, -19, -38, -95, -53, -48, 89, -80, -28, -30, -21, 23, 85, 68, -90, 28, 49, 58, 66, -93, -59, -30, -75, 80, 54, -16, 41, 40, 76, -84, 79, 2, 40, -69, 45, 45, 7, 29, 72, -6, 24, 77, 58, -56, -60, 5, -1, -82, -66, 4, -19, -67, -33, 0, 7, -23, -74, -46, -43, 80, -53, 55, 85, -7, -1, 0, 63, 81, -79, -3, 10, 4, 72, 49, 64, 42, -39, 11, 71, -36, 1, 10, -83, 93, -71, -25, -78, 27, -29, 82, -41, -18, -23, 60, 26, -80, 29, 8, 22, -31, 34}
, {-26, 83, 35, -50, 39, -29, -89, -7, 23, 42, 74, -59, 80, -54, -91, 46, -17, 26, -23, 21, -22, 51, 67, 51, 10, 72, 7, -21, 34, 88, 65, 2, -79, -3, -51, 67, -64, -19, -9, 87, -48, 38, 27, 53, -13, 20, -76, -35, -26, 26, 36, 1, 3, 49, 71, 54, -5, -35, -41, 21, -68, 45, 56, -16, -5, 40, -64, 14, 68, -36, 77, -75, 18, -86, -19, 77, -87, 44, 30, -12, 75, -36, -53, -52, -73, -87, 45, 9, -54, -52, -7, 75, 73, 91, -28, 57, 55, -59, -88, 13, -77, -43, 83, -88, -53, 6, -26, -32, 53, 68, -57, -80, -83, 40, -85, 16, 85, -4, 33, 73, -37, 74, 81, -20, -15, -92, 38, 14}
, {-92, -52, -35, -37, -44, -31, 0, 40, 36, 54, 33, 4, -100, -8, -68, 80, -83, 36, -76, 29, 22, -84, 20, 89, 49, 50, -11, 73, 65, 71, -43, -24, -62, 87, 4, -59, -45, 66, -79, 22, -60, 25, 84, -13, -44, 52, -1, -61, 96, 27, 12, -15, 12, 86, 93, 67, 8, 98, -46, 74, -3, 72, -39, -43, 79, 99, 101, -33, -60, 40, -20, 44, 6, 21, 81, 52, -9, -69, -75, 43, -38, 80, -20, 71, 23, 11, -71, 69, 61, 18, 34, -57, -5, -24, -96, -36, 94, 36, 1, 21, 98, 62, -42, -4, 57, -36, -29, -79, 33, -85, 76, -96, -81, 15, 41, -23, 22, 6, 43, -77, -37, 43, -92, 83, 0, -96, 50, 40}
, {-68, -63, 17, -21, 76, 80, 61, -8, 57, 56, 0, 3, -16, -53, 90, -62, -18, 77, -94, -58, 28, -11, -35, 17, -89, 52, -38, 50, 45, -43, 76, -24, 32, 87, -45, 44, 26, -82, 12, 36, -74, -6, -14, 55, 0, 63, -85, -2, -10, -2, -84, 86, -23, -80, -50, -17, 81, 56, -94, -23, 93, 80, -58, 2, -51, 36, -37, 61, -53, 32, -72, -72, -56, -61, -69, -43, -74, 46, 82, -84, 11, -30, 15, 81, -7, 33, -52, -32, 64, -30, -62, 27, 15, 62, -63, -26, -58, 17, 76, -7, 72, -77, 35, -35, 25, -3, -22, -81, -17, -44, 57, -43, -65, 81, -63, 80, -94, 88, -52, -91, 95, 76, 89, 57, 61, -12, -60, 50}
, {-25, -52, -33, 68, 36, -35, 52, 84, 2, -10, -24, -39, 46, -40, 20, -27, -36, 55, -2, -91, -42, 90, 88, -15, 11, 19, -67, -85, -33, -26, -14, -70, 75, 8, 68, -32, -2, -15, 22, 94, -6, 22, -65, -40, 76, 64, -39, -62, 101, 63, 73, -12, -67, -92, 26, 27, 6, -54, 37, -90, -92, 81, -17, 7, 17, -15, 29, -91, -73, 65, 28, -3, 52, -19, 66, 45, -21, -44, 88, -5, 74, 28, -38, 22, -81, 39, 24, -32, 76, -44, 19, 25, -62, -65, -65, -83, 46, 93, -71, 71, -22, 65, 43, -71, -77, -50, -46, 58, -75, -69, 77, -36, -79, -53, -73, 39, 87, 35, 40, -1, 31, 34, -77, -26, 10, 82, -4, 22}
, {-63, 15, -58, -5, 62, 25, 67, -20, -55, 61, -38, -5, -55, -8, 89, -86, -93, 68, 4, -46, 7, -17, -95, 6, 18, -25, -14, -15, 42, -95, -85, 28, 57, -59, 15, -15, -85, 51, -4, 17, 22, -39, -28, -100, 7, -65, 64, 62, 25, 36, 1, -3, 37, 59, 8, 21, 28, -74, 11, -66, 17, -12, -68, -66, 73, -61, -34, -2, 73, -22, 0, 50, -76, 19, -42, -17, 84, 30, 74, -22, 70, -48, -86, -78, 45, -59, 26, -37, 8, 83, 28, -89, -2, -58, 52, -79, -19, -62, 59, 30, -17, 41, -57, -37, -90, 4, -41, -10, -49, -4, 53, 2, 22, -42, -46, -28, 78, 51, -65, 37, 58, 73, -44, 8, -68, 44, 78, 26}
, {-77, -34, -59, -83, -38, -74, 69, -17, -70, -2, 1, -11, 34, 24, -86, -29, 2, 89, -29, -9, -37, 82, 47, -38, -44, -35, -61, -68, 5, 75, -31, 1, 4, 4, 21, 56, -54, 89, 52, -83, -84, 37, 0, -12, 7, -76, 51, 72, 17, -9, -3, 41, 54, 36, -88, 47, 13, 23, 81, 45, 64, -62, 87, -1, 71, 42, -72, 23, -14, 90, -19, 52, -6, 31, -44, -97, 12, -18, -4, -42, -8, 92, -84, -34, -74, -90, -46, -40, -85, -65, -66, -78, 80, -88, 22, 68, 30, 27, 21, 56, -64, -60, 42, -3, -79, 46, -40, 22, -51, -4, -67, 33, 74, -78, -25, 78, 65, -19, -71, -29, 93, 28, 39, 17, -70, -55, -68, -9}
, {-11, -85, 34, 58, 56, 56, 20, 31, 90, -18, -2, 19, 11, -54, -17, 70, 47, -59, -33, 83, 38, -69, 23, 5, 62, 14, -102, -19, -73, -94, 36, 92, -63, -50, -82, 36, -35, 37, -80, 76, 61, 26, -10, 36, -72, -43, -89, -82, -65, 54, -87, -8, -74, -49, -46, 75, 58, -65, -53, -90, 27, 58, 89, 88, 22, -63, -68, 70, 18, 71, -89, 67, -76, -76, -94, -50, -27, 59, -92, 83, 31, -28, -83, -9, 22, -16, -44, -16, -56, -101, 4, 55, 8, 77, 83, -71, 0, -63, 67, 53, 16, 57, -7, -35, -81, -7, 20, 52, -63, 83, -69, 58, 7, -56, 67, 36, 17, -94, -63, -22, -34, 0, -15, -60, -91, -55, 0, -7}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS