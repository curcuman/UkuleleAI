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


const int16_t dense_bias[FC_UNITS] = {-37, 22, 1, -34, 17, 24, -16, 23, 12, 1, -16, 14, 24, -9, -18, 20, 14, -30, -21, -10, 5, -14, 0, -25, -25, -13, -34, -16, -12, 38, -35, 19, 6, 11, 21, -38, 6, -2, -20, -11}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{25, -43, 80, -15, -28, 56, 6, 36, -36, 14, 94, 86, 2, 114, -4, 28, 51, 120, -15, -42, -106, -21, -67, 15, 49, -32, -80, 59, -94, -16, 6, -87, -114, -89, -4, -52, -48, 19, -85, -52, -75, 69, -52, -4, 54, 108, -16, -80, -102, -41, 94, 69, 76, -106, 38, -53, -95, 92, 81, 95, 95, -46, -40, -103, 36, -11, 17, 42, -87, 36, 44, 15, 2, 37, 71, 57, 77, -12, -64, -44, 31, 44, -17, 107, -38, -64, 67, -158, 97, -85, -121, 31, -56, 120, 1, 6, -89, -72, 77, 36, 88, -60, 56, 30, 17, 12, 16, -38, -48, 32, 30, 68, 25, 44, -49, 17, 78, -38, -46, -7, 42, 72, 29, 13, -19, -55, -27, -29}
, {-66, 104, 60, 85, -81, -122, 94, -143, 44, -33, 15, -45, -19, 53, -88, -60, 82, 94, -43, -46, -47, -31, -9, 2, -17, 35, 40, -86, 124, 21, 49, -57, 81, 121, 86, -97, -11, 30, 93, 18, -39, -79, 93, 45, 5, -73, -53, -57, -10, -79, -70, -87, 17, 58, 50, -10, 78, -52, 58, 58, -100, 32, 84, 68, -79, -8, -77, -96, 7, 32, 29, 14, -68, -31, 17, -51, 38, -36, 91, 92, -14, -23, -6, -10, -86, -25, 42, -20, -63, -18, 126, -50, 52, 4, -50, 51, 68, 93, -48, -39, 26, -116, 28, -80, -20, 54, 69, 33, -61, 78, -69, 70, 4, 105, 73, 3, 1, -56, -111, 79, -96, 10, 60, 44, 26, 10, -47, -60}
, {2, 77, 54, 26, 84, -91, -88, -71, 47, 27, 51, -115, 75, -111, 16, 61, -41, -61, 23, -67, -15, 100, -55, -79, 91, 82, 123, -69, 6, 74, -35, 48, -61, 19, 10, 8, -17, -80, 13, 82, -3, 24, 0, -70, 50, -63, 67, -78, 52, -67, -49, 28, 64, -72, 72, 28, 60, 47, 90, -40, -55, 54, -11, -79, -4, 37, -11, 48, 31, 92, -80, 50, -20, -2, -80, 0, -49, 60, 5, -45, 41, -22, -46, -73, -69, -63, 72, -15, 14, 21, 48, -52, 42, -130, 3, -9, -96, 85, -110, 91, -11, 40, 12, -74, 11, -27, 79, 101, 95, 17, 11, 15, 79, 106, 48, 50, 60, 7, -58, -26, 24, 145, -41, 6, 28, -69, 6, 5}
, {21, -10, -76, -73, 82, 86, -16, 132, 102, 28, -39, -65, 97, 123, 73, 32, -59, -42, 12, -70, -101, 2, -98, -25, 120, 7, 20, -85, 36, 93, -24, -4, -113, 57, 35, 101, 30, -2, -21, 87, 70, -48, -72, -3, -35, 82, 60, -82, 81, 76, 5, 2, -83, -108, -23, 11, 60, -75, 56, 31, -45, 11, -101, -5, 38, -44, -59, 0, 37, 91, 54, -41, 41, -34, 32, 87, 21, 44, 31, -65, 78, -66, 31, -11, -51, -70, 3, -33, -17, -30, -129, 125, -81, -58, 73, 30, 16, -5, 58, 33, -32, 64, 76, 20, 68, 84, 18, -64, 53, -104, -45, 95, -45, 92, 31, -37, 1, -67, -17, -78, 72, -32, 19, 23, -40, 28, -42, -17}
, {-12, 23, 1, 74, 93, -171, -50, -101, -13, -16, -86, -86, 9, 50, 0, 0, 62, -21, -71, -66, 4, 53, 95, 46, -6, 47, -26, -97, 57, -18, -31, 66, -73, 42, -25, -6, -4, -82, -20, -19, 0, 80, -21, 63, 28, -49, 45, 25, 7, -63, -81, 86, 44, 23, -84, 47, 51, -57, 21, 53, 2, 95, 50, -80, 47, 99, 25, -80, 8, 72, 2, 46, -55, 106, -62, -17, -86, 25, 29, -86, -87, 33, -26, -30, -61, 9, -18, 62, 98, 91, -40, 51, 96, 31, -8, 76, 2, -75, -63, 132, 45, 151, -40, 8, -54, 91, -28, -38, -37, 77, -79, 38, -14, -56, -5, -38, -95, -31, 1, -18, -10, 105, 14, -13, 31, -69, 14, -41}
, {-52, 53, -6, 78, 64, -128, 44, -156, -30, -66, 76, -39, -21, -77, 35, -12, -37, 79, 87, -103, 63, -43, 93, 21, 69, -43, -43, -7, -62, 26, -118, 84, 2, 112, -31, -13, 31, 70, 5, -99, 42, -97, 97, 1, -110, 43, -7, 93, -14, 71, -47, 25, -62, 136, 99, 7, -84, 11, 44, 68, -55, -15, 40, 29, -93, -25, 20, 3, 48, 81, 12, -63, 79, 94, -77, 44, 62, -56, -46, 23, -114, 110, -106, -22, 30, 86, 11, -41, 8, -41, 113, -89, -56, -57, 85, 73, -77, 37, -102, -16, -65, 70, 36, 19, -102, 50, 42, -16, 48, -99, 94, 76, -5, 40, -89, -10, -111, 107, -41, 86, -83, 5, -11, 83, -18, 36, -10, 32}
, {-93, -52, -69, 5, -8, 32, 84, 7, -66, -4, 35, 94, 22, 122, 43, 82, 60, -7, 10, -70, 43, 77, -80, -96, -44, -91, -31, 84, 13, 30, 64, -11, 73, 49, -78, 13, -110, 72, -51, 46, 53, 61, 0, -19, 119, 24, -33, 78, -48, 74, 71, 52, -77, -23, 78, -101, -65, 86, -26, -6, -76, -3, -48, 40, 1, -83, 5, 83, 6, -83, -9, 25, -41, 66, -83, 32, 62, -70, -24, -58, 48, -82, -19, -75, 47, -90, -19, 85, 67, -29, -56, 89, -68, 98, 75, -43, 60, 47, 68, -142, 28, -175, -45, -48, 0, -50, 17, 42, 8, 37, 69, -14, -50, -22, -108, -61, 34, 100, 11, 24, 31, -67, -74, -39, 56, 31, 85, -34}
, {21, 107, -43, -52, 8, -76, 97, -43, -94, 108, -83, -62, -73, -12, -83, 43, 95, -60, 66, 1, -89, -25, 51, -78, 26, -75, -94, 68, 95, 39, -37, -7, 78, -94, -53, 69, 41, -61, 92, -82, 58, -26, -41, -95, -103, -4, -72, -60, 36, -73, -46, -52, 116, 63, -27, 60, 69, 84, -55, 14, -71, 27, 79, 71, 70, 81, -56, -25, 104, -110, -86, -9, 52, -52, 22, 37, -51, -5, -18, -80, 53, -118, -6, -56, 73, 75, -78, 57, 16, 76, -40, 67, 38, -112, -28, -4, 57, -27, 40, 51, 18, 125, 42, 69, -72, 52, 22, -40, 85, 95, 12, -55, -113, 5, 145, -46, -107, -32, 53, -32, -28, 51, 43, 99, -66, -50, 52, -71}
, {-88, 35, 45, 94, -57, 72, 108, 81, -11, -85, -13, 57, -1, 26, 47, -30, -24, -59, -63, 15, 5, 70, 98, -51, -87, -70, -39, -2, 37, 9, 12, -82, -41, -14, -99, -74, 8, 80, 93, 18, 55, -22, 34, -3, 22, -41, -80, -49, 12, -35, -57, -29, -52, 10, -86, -17, -91, -32, 41, -4, -69, 67, -47, 37, 0, -72, 65, -50, 83, 73, 92, -61, -84, 59, 79, -72, 52, -12, 27, 33, -62, -59, 92, 51, 111, -60, -77, -9, 36, 18, 70, -17, -48, 84, 52, -14, -30, 50, -69, -118, -69, 63, 112, 37, 84, 71, 35, 62, -52, 90, 5, 18, -117, -27, -27, 39, -32, 65, -1, -20, 8, -45, 53, 41, -19, 27, 76, -6}
, {64, -10, 75, -66, 36, 13, 88, -38, -50, 20, -15, 8, 36, -60, -34, -19, -27, -59, 61, 74, 43, -32, 49, 32, -43, 10, 17, 56, -62, -2, 68, -37, -60, 98, 110, 28, 115, 44, 56, -44, 72, -20, 46, -26, -3, 112, -11, 118, -51, 61, -77, -5, 23, 103, 5, -5, -22, 89, 82, -21, 43, 10, -70, -8, 31, -33, -84, 32, -102, 104, 57, -32, 12, 83, -59, 61, -83, -40, -62, 42, 31, 55, -49, 81, -84, -75, 10, -58, 10, -69, 21, -90, -81, -6, 44, 63, 57, -39, 92, 23, -79, -121, 11, 91, -77, -103, -15, -93, -17, -56, 55, -46, 108, -19, -36, -111, -31, 119, 54, 107, -65, 16, 14, -88, 60, 34, -52, 49}
, {41, -106, -26, 2, -48, -60, -90, -61, 103, 89, 41, -45, -37, -32, -38, -29, 60, 52, -49, 90, 46, 70, -74, -112, -32, -65, 98, -8, -20, 13, -129, 71, -10, -49, -11, 33, 39, 26, -109, 54, -37, -41, 25, -85, 18, -67, -88, -72, -48, 20, -97, 1, 14, -64, -16, -57, 21, -16, -49, 8, -31, 87, -107, 18, 17, 102, -39, -73, -36, -57, -102, 115, 80, 89, -2, 52, 91, -79, -9, 9, -31, 13, -122, -97, -37, -87, 80, -41, -37, 77, 4, 1, 63, 54, -39, 96, -42, 80, 56, 4, 23, 47, -45, -103, -55, 69, -22, 77, 46, -14, 13, -82, 14, 24, 5, 17, -76, 35, -24, 77, 54, 73, -93, -105, 78, 30, -74, 21}
, {4, -35, -65, 116, 36, 59, -70, -98, -112, -6, -76, -57, -14, -52, 51, -10, 4, -83, -63, -35, 27, 46, -57, 100, 58, 57, -45, -70, 129, -50, 44, -11, 5, -33, 57, -76, 16, -23, 87, -5, 56, -92, 16, 28, 16, 45, 31, 61, -49, 83, -57, -27, -95, -38, 41, 99, -94, 28, -36, -9, 42, 34, -25, 84, -66, 4, -118, 4, -39, -14, 87, -101, -74, -55, -33, -48, -20, -65, 51, -56, -11, 117, 78, -81, -74, 68, -85, 105, -3, -17, -34, 59, -88, 73, -82, -99, 25, -8, 91, 21, 13, -18, 27, 83, 22, -8, -79, 33, 6, 66, 94, -58, 0, 18, -76, 3, 11, -32, 57, 32, -53, -35, 41, -39, -35, -68, 51, 23}
, {33, 34, 60, -35, 46, -39, -14, -53, 58, 120, 25, -48, 49, 6, 16, -48, -44, -65, 74, -102, 97, 18, 5, 63, -128, 37, 34, 81, 85, 44, -54, -45, 28, -24, -16, 24, -63, -71, 29, 50, -69, 70, -39, -36, -9, -28, -45, -68, -5, -67, -78, 62, -24, -24, 95, 85, -9, 31, -136, -27, -13, 70, 79, 80, 83, -63, -82, 91, -45, -116, 64, 81, -99, 64, 118, 44, -36, 60, 72, 50, 7, -84, -42, -55, 86, 64, 77, -36, 68, 12, -56, 4, -71, -49, -47, -86, -85, -88, -89, -34, -48, 113, -13, -111, 62, 79, -65, 57, -97, -45, -51, -8, -98, -78, 98, 93, -50, -5, 32, -95, 4, -23, -76, 67, -88, -85, -11, -84}
, {102, 80, 28, -78, -33, -83, -88, -96, 55, 25, -69, -89, -2, 57, -6, -10, -48, -28, 51, -9, 83, -87, 50, -59, 81, -26, 54, 7, -25, 63, 58, 58, -106, -82, -69, 3, 21, -91, 72, 95, 14, 3, -43, 56, -69, 28, -39, -115, -89, 66, 79, 13, -37, 20, 6, -80, 33, 73, -9, 65, 35, 30, -47, -35, 36, 0, -43, -83, 60, 34, 26, 25, -2, 50, -19, -64, 72, 32, -40, -74, -15, 19, -74, 54, 15, 31, 102, -42, 54, 117, 13, 65, 26, -123, -5, 108, -27, 70, -86, 98, -7, 119, -4, -34, 24, 1, -39, 38, -12, -31, 79, 56, 24, 11, 27, -61, -72, -115, 61, 42, 44, -1, 45, 32, 73, 53, -49, -77}
, {58, 33, 24, 49, 94, 81, 74, -36, 6, 124, 108, -88, 87, -24, -57, -77, 88, -34, 68, -32, 9, 66, 46, -40, 124, 73, 85, 38, -133, 31, 52, -58, -113, 66, -21, 10, -56, -78, -79, -5, 37, 61, -54, 56, 29, 70, 39, -46, -5, 13, -32, 84, 53, -19, -24, 47, 136, 63, 9, -64, 20, 8, 59, -35, 87, 111, 99, -21, -42, 14, -5, -33, 34, -33, -64, 14, 64, 54, -7, 20, 88, 70, -53, -17, -26, 59, 44, -63, -17, -10, -61, -57, 97, 42, 38, -11, -107, 47, 28, 43, -59, 129, -10, -109, -15, 108, -58, -24, -67, 33, 30, -59, 63, 108, 84, 75, 67, -103, -6, -78, 51, 96, 65, 43, 21, -35, -5, -19}
, {-74, -1, 2, -63, -89, 83, 56, 81, -49, -14, 68, -37, -30, 63, -54, 23, -38, -103, 56, 62, -24, 91, 6, 20, 13, 64, -24, 66, -20, -103, 72, -47, 34, -56, 62, 14, 52, -17, -38, -15, -17, 84, 114, -110, -43, -81, 84, -61, -80, -94, 32, 63, -67, 46, 77, -30, -136, -67, -108, -80, 93, 7, -34, -24, 14, 27, 54, 4, -38, -44, -30, -132, -54, 0, 54, 81, 7, -20, 25, -18, -73, -97, 108, 19, 97, 85, -122, 113, -79, 84, 76, -81, -100, 5, 70, -2, 4, 72, 96, 17, -88, -32, 97, -63, -47, -102, -15, -113, -86, -25, 0, -107, -125, -114, -29, -6, 74, 124, -69, 10, 10, -134, -47, 52, 54, -55, 15, 43}
, {-37, 48, 24, -99, -45, -39, 37, -9, -46, 59, -85, -133, -46, 73, -83, 73, -5, -36, -80, -74, 83, 1, 69, 26, -92, 75, 26, 55, 18, -77, -96, -79, -30, -46, 10, 43, 13, -97, -29, -86, -3, -58, -6, 32, -20, -53, 77, -36, 24, 40, -20, 86, 27, 62, 10, -83, 35, -4, -44, -36, 86, 33, 45, 23, -36, 64, 98, 38, 116, 27, -87, -58, -99, 11, -19, 25, 27, 89, 75, 18, 75, -97, 77, -20, 27, -79, 74, 55, -65, 81, -38, 26, -82, -21, -6, 27, -70, 29, -39, 19, 65, 121, -33, 24, -6, 21, 7, 69, 39, -68, -9, -13, -114, -61, -9, 85, -60, -10, -108, 24, -12, -84, -55, -58, 57, -94, 51, 40}
, {-8, 51, 11, 37, 88, 41, -62, 182, 1, -90, -41, 19, 48, 103, 84, 69, 36, 72, 60, -63, -33, -10, -105, -94, 32, -91, 69, 57, 58, -65, 51, -48, -68, -23, -71, 12, 0, 32, -76, -32, -53, -105, 2, -21, 47, 58, 51, -67, 65, -11, 117, 80, -25, 3, 84, -41, 15, 89, 109, 48, 1, -56, 73, -64, 103, -101, 51, -23, -93, -20, 79, -68, 6, 1, -73, 67, -12, -38, -12, 24, -40, -68, -92, 8, -85, -25, -8, -48, -76, -53, -88, 65, 45, 26, 98, -3, 26, 62, 105, 39, -83, -83, -41, 84, -43, -45, 67, 8, 44, -102, -38, 10, -34, 78, -159, -80, -41, -29, -12, 46, 45, -40, 99, -103, -45, 79, 3, -35}
, {43, -102, -68, -52, -5, -82, -103, 27, 121, -43, 109, -14, -5, 8, 82, -30, 81, 115, -62, -74, 70, -65, -12, 27, 64, -6, -33, 25, -80, -37, -79, 62, 26, 47, -46, -28, 75, -42, -72, 50, 25, 40, 7, 55, 71, 47, -84, -74, -29, -31, -71, 42, 37, -74, 40, -37, -74, 76, 79, 73, -65, 36, -55, 38, 40, -10, -53, -4, 3, 95, -97, 113, 30, -95, 60, -101, -5, 14, 56, -105, 53, 108, -116, -17, 34, -97, 88, -47, -80, 77, -86, -54, 0, -16, 0, 105, -33, -63, -63, 82, -75, 77, 21, -20, 97, -40, -92, -39, 68, 1, 22, 80, -29, 99, -29, 20, 53, 36, 32, 17, -45, 103, 56, -111, 50, 7, -54, -73}
, {-50, -27, -96, -56, 88, -2, -29, 88, 16, 2, -7, -17, 51, 2, 71, -8, 100, -50, 58, -4, 74, 8, 10, -102, 2, 49, -20, 6, -105, 87, -33, 69, -33, -51, -101, 36, -59, -24, -102, -77, 71, 59, -26, 80, 21, -51, 80, -61, 57, 37, -44, -24, -14, -126, -60, 53, 117, -7, 2, -85, 14, 124, -21, 75, -10, -11, 95, 81, -11, 13, -24, -51, -47, -81, -14, 88, 92, -54, -93, -22, 13, 13, 75, -44, -84, 49, 93, 40, -105, 81, -82, -40, 79, 79, 14, 63, 3, -93, 7, 105, 69, 102, -57, -66, 77, 96, 39, 37, 82, -1, -92, 104, 10, -117, 23, -53, -35, -77, -55, -63, 88, 20, -66, 2, -66, -85, 97, -37}
, {8, 81, 31, -50, -74, 8, -43, -53, 6, -19, -70, 55, -55, 98, -16, -6, -77, -85, 18, -70, -1, -19, -83, -67, 86, -61, -74, 59, -60, -40, -23, 14, -62, 120, -48, -96, -64, -40, -13, 69, -2, -60, 14, -76, -58, -67, -42, 8, 46, -34, -37, -86, 52, 36, -48, 55, 15, 32, -16, -49, 30, 28, 31, -89, 57, 0, 55, 79, 55, -73, 95, -32, -11, 7, -129, -34, -65, 79, -69, 13, -83, 0, 78, 13, 45, -77, -3, 60, 82, 4, -7, -75, -38, 61, -62, -58, 105, -58, 105, -108, 26, -86, 66, 11, -4, -127, -34, -4, 81, 9, 101, 29, -72, 4, -17, -85, 5, 32, -42, 56, 49, 50, 97, -93, -37, -49, 1, 25}
, {22, 31, 27, 39, 65, 118, 84, 124, -67, -34, -41, -35, 57, 0, -94, 39, -62, 54, 104, -18, -3, 49, -49, 20, 38, -22, -30, -6, 98, 50, -29, 31, 11, 19, -17, 1, 59, 102, 38, 33, 99, 51, -53, 76, 8, 153, 1, -53, -41, 20, -9, 93, -48, 50, -55, -86, -8, -45, -9, -36, -57, -112, 51, -25, -11, -46, -90, -43, 0, -12, 31, -41, 124, -95, 25, -81, 21, 75, -57, -63, 36, 110, -84, -45, 6, -56, -45, -39, 102, -95, 32, 27, 66, -19, 14, 48, -75, -31, 86, -13, 90, -45, 11, 60, 70, -27, 24, 5, 30, -50, 11, -17, -5, 60, -34, -36, -47, 50, -55, 20, 12, 20, -22, 31, -80, 42, -22, -88}
, {64, 97, 80, 15, 14, -47, -66, -60, 42, 25, 39, 40, -39, 86, 13, 83, -97, 84, 29, -48, -56, 10, -94, 41, 10, -23, 13, -100, -1, -7, 23, 56, -9, 111, 22, 14, -62, 79, -63, 44, -42, -72, 48, -23, -13, -43, 2, 44, 31, 9, 44, 5, -89, -53, 97, 26, -27, 36, 64, 41, -107, -86, 62, 38, -13, -99, -118, -14, 26, 57, 60, -52, 37, 68, -98, 39, 10, 83, -46, -65, 49, -18, 72, -28, 91, -65, 33, -4, 8, -110, 118, -14, -81, 13, -66, -91, -51, 7, 69, 2, 66, -31, -87, -55, -48, -7, 60, -82, 39, 25, -14, 12, 16, 108, -65, -86, -63, 84, 103, -47, -52, 0, 105, -26, -74, 83, -81, 90}
, {-72, 80, -29, 47, 12, 16, 88, 129, -78, 35, -37, 69, -66, 94, -52, 67, -14, -20, -38, 74, -63, 40, -59, 0, -11, 15, -54, 92, -74, 21, 30, 49, -64, -143, -61, -79, -58, 56, 4, 19, 59, 51, -72, -83, 40, -11, -50, -74, 39, 85, -27, 78, 14, -46, 6, 12, -63, -91, -24, -26, 15, -12, -16, -36, 91, 99, -60, -70, 73, -113, 31, -94, 87, 35, -22, 12, 102, 89, -28, 84, -22, -65, 94, 67, -59, 51, 91, -13, -14, 45, -58, -67, -39, 101, 17, -31, 83, 2, 81, -23, 28, 34, -4, 118, -8, -44, -17, 37, -37, 50, -18, 27, -14, -28, 89, 96, -49, -9, 107, -124, 116, 31, -12, -17, 1, -41, 7, 16}
, {-83, -105, 81, -33, 28, -22, -85, 93, 52, -47, 87, -7, 90, 52, 75, -75, 81, 95, 49, 61, 14, 67, 29, -51, -52, -38, 116, -32, -96, 54, 62, -50, 6, 90, 104, 5, -61, 104, 70, -5, 104, 70, 90, 73, 100, 144, -1, 15, -105, -78, -64, 95, -41, 12, -45, -59, 36, 60, -36, 105, -87, 56, -92, 40, 27, -90, -52, 55, -5, 109, -34, 83, -32, -15, -101, 84, 46, -32, 28, 0, 7, 92, -70, 8, -50, 37, 50, -43, -10, -4, 84, 94, 29, -24, -19, -4, 42, 66, 96, -13, -80, -89, 23, 31, -55, -56, 7, -14, -51, -60, -37, 60, -8, 1, -46, 57, -68, 63, 60, -63, -56, -80, 17, -37, 35, 38, -32, 15}
, {20, -12, -49, 67, -44, 116, 25, -26, -31, 21, -10, 41, 13, -21, -42, -33, 30, -61, -22, 34, 62, -74, 75, -35, -56, 29, 11, 16, -48, -71, -54, 62, -37, -121, -75, -5, 54, 25, 14, 12, 28, 64, -74, 15, -46, -99, 8, 34, -9, -38, 66, 3, 58, -75, -3, -95, -29, 44, 12, -93, 57, 119, 15, 28, -42, -35, -48, -1, -70, -6, 47, -69, -14, 5, 37, -46, 91, -77, 0, -12, -53, -115, 37, 15, 32, 85, 90, 17, 79, 57, -16, -61, -72, 0, -17, 59, -71, -14, -17, 83, -45, -5, 1, -13, 34, -44, 39, 76, 11, 48, -85, 20, -24, -43, 40, 55, 71, -74, 33, -11, 30, -74, 49, 72, 100, 8, -1, 64}
, {-29, -39, -35, -71, 52, 83, -77, -24, 104, -49, -28, -26, 21, 19, 24, -84, -31, -55, -6, -3, 52, -97, -39, -106, -15, -58, 39, 0, 56, -61, 54, -59, -91, 0, 2, -23, -70, 60, -3, 33, 49, 40, 48, -73, -3, 2, -35, -51, -72, 47, 13, -88, 5, 7, -54, -111, -76, -77, -56, -26, -42, -51, -99, -27, 92, -12, -7, -45, 40, 92, 45, 16, 94, 43, -99, 47, 74, -24, 23, -81, -3, -34, -52, -43, 41, 73, -17, -117, 20, -75, -109, -16, 13, -49, 33, 18, 71, -19, 78, 77, -51, -14, 12, -8, -4, 92, -42, 9, -15, 2, -87, 43, 34, 14, -12, -69, 124, -8, -2, -99, 24, 36, 61, -104, 104, 1, 4, 72}
, {98, 54, 93, -56, -48, 17, 19, 42, 104, 90, 3, -76, 24, 20, -72, -85, 65, 104, -31, -14, 82, 38, -73, -40, 124, 16, 53, -84, -44, 90, -5, -3, -42, 35, -3, -12, 53, -91, 46, 74, 22, 5, 54, 82, -35, 75, -58, 30, 61, 83, -76, 88, 21, -79, -98, 82, -83, -93, 100, 57, -103, 3, -27, 61, 50, 5, -36, -28, 14, -72, 10, 48, 121, 40, -74, -7, -30, 73, -72, 3, 27, -39, -25, -37, -50, -78, -70, -81, 2, -96, -7, -57, 65, -50, -43, -29, -14, -55, -69, 45, -26, -54, -12, 69, -11, -98, -76, -40, -6, 58, 5, -39, 4, 87, -93, 81, 41, -49, -77, 85, 26, -49, 89, -46, -63, -37, -53, -88}
, {58, -89, 78, 41, 11, 71, -2, 75, 56, -63, -66, 9, -35, 62, 74, 14, 49, -72, -94, 86, -93, 45, 56, 65, -1, 66, 46, 44, 40, -18, -59, -19, -77, 16, -36, 55, -62, 123, 80, 82, -19, -79, 19, 75, 58, -41, -65, -58, 32, 49, 66, -30, -113, 32, 101, -59, 14, 51, -71, -72, 55, -90, 56, 16, 38, -119, 23, 27, -66, -53, 64, -19, 26, -28, -80, -32, 4, -85, 42, -90, 56, -150, 35, 89, 72, -24, -15, -87, -18, 23, 67, -82, -40, 106, 103, -52, 69, 17, -18, -123, -9, -136, -70, 22, 19, 45, -43, 1, -69, -94, 19, -77, 21, -83, -6, -111, 10, 48, 102, 58, -43, -78, 28, -106, 54, -64, -27, -15}
, {-61, 104, -25, -79, -85, -15, -67, -128, -71, 1, 21, 46, -67, -59, 23, 80, 51, 59, -107, -81, 60, -66, 100, 13, -121, 86, 60, 64, -25, 51, 61, 97, 111, 13, -99, -102, -43, -30, 104, 66, -64, 94, 56, -31, -35, -14, 43, 115, -71, 4, 62, -20, 14, 31, -34, 34, -1, -31, -91, -116, -20, 40, -68, 19, 62, 70, -92, 14, -17, -19, -84, 19, 1, 0, -27, -14, -101, 25, -52, 59, 2, -67, 36, -24, 31, 66, -11, -1, -37, 16, 73, -108, -68, 7, 0, -61, 58, 3, 35, 21, 16, 55, 60, -71, -95, 102, -50, 102, -99, -8, 52, 48, -30, -115, -5, 86, -114, 40, -91, -68, -81, 48, 72, 62, -6, 77, 98, 7}
, {67, 35, 8, -67, -3, 64, -34, 62, 11, 6, -59, 16, -45, -46, -14, 74, 38, -31, 37, 34, -3, -62, -35, -4, 1, 28, -20, -12, 5, -59, 45, 81, -10, -87, 75, -86, 0, 12, -81, -69, 58, -59, -110, 71, 65, 5, 65, 32, -72, 70, 79, -92, -29, -108, -1, -71, 7, 51, 84, 89, -19, -96, -22, -37, -14, 62, 91, 81, -106, 42, 80, -32, -37, 4, -43, 49, 1, 71, 76, -104, -33, -42, 0, 34, -52, -86, 41, -101, -21, -90, 30, -40, 15, 101, 56, -69, 79, -39, -43, 60, 28, 26, 35, 72, 8, 38, 63, 77, 45, 54, -19, -6, 95, 22, -35, 64, -15, 60, -41, -66, 77, 63, 11, -53, -19, -32, -29, -14}
, {63, -66, 70, -29, -17, -112, 75, -152, 96, -48, 86, -121, -56, -109, 72, -30, -21, -35, 45, -109, 117, 34, 101, 85, 52, 55, 110, 36, 17, -71, -70, 90, -62, 77, -24, -27, 111, -88, -69, 67, -49, -69, -85, -77, -27, 74, 20, -47, -69, 78, -61, 17, -55, 19, -32, -38, 80, -46, 76, -84, 37, 119, -62, -26, -68, 86, 8, -88, 21, -9, -13, 107, -8, -8, 52, -63, -58, -36, -55, -50, -85, 46, -64, -29, 54, 0, -51, -48, 45, 40, -34, 24, 87, -115, -68, 74, 16, 36, 8, 94, 48, 145, -101, -112, 31, 56, 40, 13, 89, -20, 69, -98, 66, 69, 57, -7, 66, -13, -3, 47, 42, -43, 34, 14, 23, -34, 82, 21}
, {17, -50, -53, 106, 79, -70, -24, -60, 5, 76, 85, -18, 33, -41, 42, 99, 54, 25, 23, 89, 56, -14, -70, 34, -45, -67, 30, -68, 92, 64, -66, -35, -3, 41, 97, 82, 89, 11, -32, 40, 88, -51, 7, -41, 73, -55, -55, -46, -25, 28, -66, 71, -45, 83, -22, -9, -86, 29, -32, 77, 67, -107, 24, -72, 42, -53, -75, 25, -49, -15, 94, -81, 104, -23, 34, 27, 13, -45, 65, -79, 1, -32, -76, -52, -89, 79, -69, 116, 42, -88, 46, -12, 82, -44, -29, 29, 99, 101, -104, 80, -60, -125, 39, -82, -66, 0, -65, -114, -3, 45, -15, 12, -29, -27, -58, 75, -84, 108, 63, 91, -119, 83, -15, 5, -96, 17, 75, 60}
, {114, -67, -26, -45, 64, -158, -4, -163, -67, 28, 91, 52, -7, -91, -14, 64, 1, 110, 81, -38, -19, 66, 7, 42, -47, 48, 51, -23, -75, 25, 13, 17, -17, -26, 109, -24, -46, -5, 32, 42, 53, -32, 59, 35, -113, -2, -16, 9, 8, 24, 25, -78, 79, -58, -33, 20, 40, 20, -53, -22, -52, 16, -43, -9, 14, -69, 44, 55, 2, -11, -89, 96, -14, 75, -8, -121, -60, 82, -95, 60, -59, 67, -39, 40, -1, 53, 54, 84, -58, 97, -32, 72, 49, -20, -101, -50, -38, 60, 43, -46, -65, 28, 61, 6, 34, 12, 88, 89, 99, 5, -4, 94, -75, 30, 21, -46, 77, -31, -66, 80, 30, -6, 8, -45, -58, -82, 1, 95}
, {15, 35, -73, 75, -32, 86, -12, 34, 7, -46, -75, 72, -103, -45, 64, 17, 39, 49, 75, 68, -7, -11, -33, 15, 5, 90, -101, 18, 131, -56, 11, -21, -2, -23, 72, 67, 15, 62, -8, 30, 12, 38, 59, 29, -89, -99, -23, -44, -2, 8, 71, -27, -25, 22, -33, 56, -194, 40, -103, -38, 11, 21, 47, 81, 88, -37, -55, -84, 10, 59, -28, 30, 17, 97, -75, -7, -69, 27, 94, 51, -124, 7, 62, -49, 76, 70, -126, 64, 55, -13, 60, 0, 19, 74, -34, -4, 22, 32, -25, -87, 37, -139, -21, 14, -140, 1, -80, -15, -83, 83, 117, -94, -27, 13, -71, -65, 77, 62, 37, -10, -60, -114, 44, 79, -90, -18, 75, 55}
, {73, -57, 18, 2, 22, 25, 13, 93, 51, -26, 82, -30, 11, 52, -66, 70, -61, 87, 109, 95, 44, -55, -32, -97, 110, -11, 104, 84, -126, -73, 45, -62, -106, 88, 96, -33, 42, 6, -112, 32, 3, -40, -91, 67, -10, -30, -74, -100, 13, -81, -28, 59, -43, -105, -56, -96, 21, -11, -33, 82, 71, 32, -48, -111, 99, 81, -53, -52, 54, 2, -58, -1, -31, 48, 60, 110, -41, 27, -105, -94, -30, -39, 13, -31, -54, -75, 82, -68, -72, 53, -96, 86, -61, -4, 54, -11, -16, 46, 104, -25, -40, -14, 30, 5, 3, 64, 22, 70, -42, 75, 49, 27, 3, -35, -54, 70, 30, 7, 88, -71, 103, 36, 84, -92, 48, -23, 8, 47}
, {-38, 50, 96, 63, -84, 89, 25, -59, -78, -13, 11, -5, 62, 75, 50, -53, 39, -50, -91, -23, 67, 6, 3, -45, 89, 82, -53, -82, 28, -38, 82, -36, -4, -12, 113, -40, 101, -32, 13, -54, -13, -38, 53, -26, -79, -38, -108, 90, 21, 74, -61, -91, 54, 53, 36, -11, -102, -43, -65, -32, 14, -70, -60, 91, 68, -52, -68, 1, 30, 107, -18, -83, 70, 89, -60, -81, 44, 15, 20, 55, 17, -33, 80, -28, 75, -84, -55, 4, -64, -71, 61, 21, 60, -1, -62, 27, 74, -38, -24, -85, 90, -110, 81, 43, -58, -84, -46, -107, 21, 45, 41, -60, -96, 43, 27, -60, 78, 103, 66, -7, -35, 28, -8, 16, -25, -46, -80, 68}
, {24, 72, -28, 64, 36, 18, -31, 131, -78, -85, -21, 51, -119, 124, 25, -34, -93, 13, -63, -23, -108, -8, 67, 5, -48, 58, -109, 23, 37, -58, 115, 21, -31, -33, -38, 65, -98, -34, 84, -88, 37, 26, -59, 32, 53, 79, -1, -55, 99, 74, 58, 19, 47, -110, -53, 79, -61, -47, 34, -40, -65, -94, 94, 38, -91, -93, 92, 95, 89, -87, 50, -6, 1, 20, -71, 112, -42, 35, 74, -51, 14, -53, 82, -80, 73, 100, 76, -85, 64, 29, 58, -3, 31, -36, -3, 28, -17, 5, 92, -34, -35, -67, 16, -63, 9, -49, -49, 27, 41, 87, -48, 54, 27, -98, -72, 51, 92, -37, 63, -60, -84, 32, 60, -9, -50, -27, 2, 41}
, {-72, 12, 59, 29, -28, 69, 1, 41, -57, -71, -20, 109, -50, 120, -71, 65, -84, 38, -32, 100, -44, 25, -73, 60, -3, -15, -119, -87, -20, -27, -5, -31, -70, -78, 37, -31, 20, 100, -5, 67, 7, 21, 78, -94, 117, -45, 68, -42, -48, -37, 69, 54, -80, -51, -16, 3, -1, 88, -91, 34, 52, 27, 53, -58, 20, -80, 33, -54, -65, -1, 70, -119, -86, -109, -45, 103, -89, 28, -44, -35, -33, -155, 70, 0, 36, 78, -57, 12, 15, -77, 83, -40, 58, -24, 79, 40, 40, 71, 25, -97, -73, -122, 25, 129, -68, -111, -92, -51, -15, -105, 56, 43, -44, -31, 19, -28, 108, 102, -3, -17, 21, -21, 3, -91, 60, -51, 10, 39}
, {-85, -82, -70, -55, -66, -11, -48, -43, -63, 47, 51, 78, -1, 56, 40, 66, -26, -41, -68, 99, 67, 81, -6, 6, -71, 15, 48, 109, -110, 9, -67, -8, -61, -120, 17, 5, -118, 97, -69, 11, -91, 40, -5, 59, -3, 28, 87, 68, -50, 63, -15, 43, 17, 44, 60, -97, -53, -31, -40, -100, -6, 85, -29, -36, 8, 70, -32, -25, 7, -14, -20, 70, -71, -35, -25, -36, -61, 42, -60, 58, 85, -75, 23, 3, -41, -1, -46, 46, 23, 71, -119, -94, 3, -47, 29, -20, -22, -66, -38, 72, 60, 45, 34, -23, 101, 77, 59, -59, -43, 72, 49, 32, 26, -24, -46, 2, 68, -94, 41, -22, 33, -21, 70, 30, -62, -60, 0, -21}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS