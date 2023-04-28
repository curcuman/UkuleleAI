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


const int16_t conv1d_57_bias[CONV_FILTERS] = {6, 12, 21, -4, 22, -8, 4, 12, 25, -7, -14, 3, -25, -5, 0, -11, 10, -6, -23, 25, 1, 37, 10, 9, -7, 37, -21, 15, 24, 15, -7, -2}
;

const int16_t conv1d_57_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{47, 25, -5}
, {81, 49, -81}
, {-37, -46, 76}
, {81, 17, -66}
, {-23, -80, 69}
, {33, 62, 101}
, {95, -43, 85}
, {-105, 16, -72}
, {14, -64, -76}
, {92, -83, -39}
, {-35, 94, 83}
, {-10, 87, -7}
, {47, -79, 36}
, {-69, -82, 65}
, {-36, 32, -86}
, {-84, 52, 73}
}
, {{-59, 6, -71}
, {-78, 12, -7}
, {39, 94, -43}
, {28, 34, -83}
, {54, -105, -57}
, {-97, -23, -52}
, {86, 80, -37}
, {-1, -37, 90}
, {-41, 44, -63}
, {84, -63, -33}
, {27, -28, 124}
, {-86, 100, -45}
, {11, 50, -98}
, {-30, -15, -71}
, {-32, 19, -57}
, {-12, 43, -40}
}
, {{0, 67, 86}
, {-11, 84, 100}
, {53, 101, 109}
, {-88, -77, 54}
, {87, -78, 59}
, {104, 3, 52}
, {-31, -65, -94}
, {-60, -101, 63}
, {-81, 39, 2}
, {19, 95, 0}
, {-7, 7, 15}
, {4, 26, 70}
, {-109, 82, -15}
, {56, 82, 56}
, {-90, -81, 91}
, {-4, -65, -37}
}
, {{-78, -59, 7}
, {19, -39, -66}
, {107, -24, 30}
, {84, 34, 39}
, {77, -20, -1}
, {98, -34, -17}
, {29, -2, -96}
, {42, 95, 27}
, {-1, -69, -18}
, {-55, 106, -19}
, {-70, 23, -95}
, {99, 0, -54}
, {-16, 102, 93}
, {-49, -21, 66}
, {75, -33, 29}
, {54, -78, -37}
}
, {{30, 103, -78}
, {68, 77, -91}
, {-77, -25, 74}
, {-44, 57, 64}
, {-109, -67, 54}
, {-86, 51, -60}
, {-16, -80, 80}
, {26, -99, 85}
, {84, 33, -6}
, {-48, 68, -57}
, {-84, -88, -74}
, {100, 89, -90}
, {-94, -96, 57}
, {-48, -81, -40}
, {80, -102, 75}
, {-23, -16, 102}
}
, {{67, 17, -26}
, {-19, -7, -27}
, {-98, -93, -49}
, {110, 71, -40}
, {74, 115, 20}
, {95, 20, -58}
, {21, 70, -78}
, {43, 51, -16}
, {-89, 86, 98}
, {-63, -29, 49}
, {-68, -13, -25}
, {3, -74, 62}
, {-35, -98, 33}
, {121, 35, -53}
, {112, 11, 108}
, {14, 93, 55}
}
, {{92, 0, 103}
, {18, -45, 14}
, {-24, -78, 70}
, {-81, 17, -3}
, {-7, -53, -92}
, {-10, -52, 80}
, {66, 4, 25}
, {-73, -84, 85}
, {-37, 37, 4}
, {-79, -76, -23}
, {-63, -10, 1}
, {7, 78, 110}
, {66, -61, 94}
, {-59, -78, 0}
, {-71, 91, 97}
, {-27, -89, 3}
}
, {{-83, -66, 27}
, {-39, 75, -27}
, {82, -67, -44}
, {107, 8, -28}
, {87, -8, 65}
, {-69, 8, 45}
, {52, 30, -71}
, {-80, 70, 98}
, {60, 84, -65}
, {41, 31, 17}
, {-82, 84, 39}
, {48, 37, 44}
, {-21, 19, -17}
, {74, 19, 24}
, {-49, 32, 21}
, {98, 14, 17}
}
, {{42, -83, 122}
, {-56, -40, -41}
, {-60, -69, 8}
, {52, -62, 60}
, {83, -18, -59}
, {27, 45, -87}
, {-43, 104, -60}
, {74, 103, -78}
, {56, -2, -41}
, {18, 58, 107}
, {-62, -64, 82}
, {-61, 92, 7}
, {-46, 50, 3}
, {21, -18, -18}
, {-93, 63, -94}
, {-57, 5, 53}
}
, {{-16, -89, 82}
, {78, -16, -82}
, {-53, 20, -74}
, {87, -16, 90}
, {-24, -62, 26}
, {-33, 37, 66}
, {106, -87, -71}
, {-40, 35, -32}
, {60, 103, -64}
, {35, -94, -54}
, {5, 28, 71}
, {-62, 29, 83}
, {7, -58, -84}
, {97, 3, 64}
, {46, -65, 113}
, {-40, -57, 82}
}
, {{17, -6, 103}
, {41, 30, -107}
, {55, 105, -57}
, {51, -30, 26}
, {-1, 23, -95}
, {52, -78, -106}
, {-15, 37, -90}
, {95, 56, 31}
, {40, -3, 74}
, {-51, -34, -42}
, {55, 89, 39}
, {100, -103, 38}
, {89, -20, -35}
, {64, 93, -87}
, {-86, 64, 82}
, {64, -95, -77}
}
, {{86, -105, -14}
, {-79, -67, -122}
, {-108, 11, 45}
, {-65, 93, 45}
, {-93, 16, 99}
, {3, -18, -19}
, {49, 81, 22}
, {39, 16, 54}
, {101, 45, 32}
, {8, 49, 97}
, {62, 5, 60}
, {99, -80, 85}
, {74, -75, 73}
, {51, 84, 87}
, {33, -70, -46}
, {45, -17, 15}
}
, {{12, 55, 74}
, {-26, -14, 65}
, {-65, -72, -53}
, {50, -17, -61}
, {61, 46, 93}
, {-15, 54, 70}
, {-55, -18, 23}
, {66, 62, -15}
, {28, -3, 24}
, {-8, -62, 77}
, {-109, -29, -15}
, {-12, 25, 87}
, {-35, 101, -50}
, {-72, 73, -106}
, {-74, 0, 21}
, {-82, -45, 62}
}
, {{-55, -60, 85}
, {113, 39, 101}
, {-42, -71, 80}
, {22, -83, -96}
, {-29, 11, -5}
, {-13, -90, 70}
, {31, -67, -65}
, {105, -88, 0}
, {97, -74, -73}
, {-7, 58, -46}
, {6, 57, 46}
, {-74, -25, 72}
, {93, 102, -107}
, {19, 95, -15}
, {-43, 108, -44}
, {-62, 36, 48}
}
, {{4, 47, 38}
, {45, -27, 106}
, {7, -29, 13}
, {-53, 75, -108}
, {97, 73, -84}
, {-71, -14, 52}
, {-88, -83, -34}
, {32, -22, -35}
, {-74, 48, 59}
, {-87, 54, 74}
, {8, 75, 45}
, {-52, -73, 61}
, {92, 22, -102}
, {66, -53, 45}
, {-62, -67, -18}
, {86, -40, -35}
}
, {{15, 96, -23}
, {-10, 39, -58}
, {-34, 48, 9}
, {76, -42, -55}
, {15, -26, 10}
, {81, 16, -90}
, {106, 52, 13}
, {-37, -43, 7}
, {18, 25, 51}
, {48, 2, -18}
, {14, -22, 77}
, {38, -89, 61}
, {31, -103, 28}
, {-17, -108, 89}
, {8, -91, -40}
, {104, -89, 74}
}
, {{58, 83, 13}
, {-71, 95, -63}
, {60, -32, 22}
, {66, -91, -62}
, {-8, -91, 95}
, {3, -91, -34}
, {12, 10, -1}
, {-112, 48, -55}
, {-18, -14, -8}
, {-66, -72, 48}
, {24, -39, -17}
, {-53, -14, 60}
, {100, -81, 67}
, {-72, -83, 72}
, {77, 59, 94}
, {93, 70, -72}
}
, {{-76, -89, 15}
, {-25, -34, -38}
, {67, 67, -85}
, {21, -50, -41}
, {26, 91, -99}
, {31, -89, -96}
, {-92, 98, 81}
, {61, 13, 68}
, {-60, 75, -48}
, {18, -21, 84}
, {6, 75, -65}
, {-103, -42, -73}
, {69, 93, -40}
, {12, 38, -81}
, {17, 73, 68}
, {-32, -90, 18}
}
, {{77, 84, 70}
, {17, -95, -14}
, {81, -56, 78}
, {97, -70, -104}
, {30, -83, 72}
, {42, -47, 112}
, {40, 32, 104}
, {63, 27, 30}
, {-71, 101, -77}
, {-86, 7, 73}
, {-76, 15, 58}
, {-79, 11, 37}
, {-59, -95, -107}
, {98, -20, -4}
, {50, 31, 49}
, {-4, -23, -37}
}
, {{35, 10, 12}
, {22, 89, 83}
, {43, 3, -12}
, {33, 22, 59}
, {-35, -86, -29}
, {17, 12, -94}
, {-54, -108, 93}
, {31, 95, -38}
, {-18, 71, 24}
, {-87, 86, -97}
, {-27, 54, 68}
, {35, -15, -82}
, {97, -78, 64}
, {-118, 62, 48}
, {29, -7, -4}
, {-87, -75, -38}
}
, {{0, -92, 100}
, {98, -68, 110}
, {71, 57, 3}
, {-40, -27, 99}
, {79, -1, -4}
, {35, -85, -61}
, {-14, -88, -80}
, {-31, 19, 66}
, {4, -13, 73}
, {-17, 22, 78}
, {75, 35, 96}
, {81, -52, -34}
, {-24, -55, 45}
, {-56, -56, 37}
, {-46, 21, -88}
, {-38, -79, -100}
}
, {{-65, -38, 28}
, {-30, -58, 22}
, {-54, -82, 70}
, {-75, -39, 16}
, {9, -34, 71}
, {10, -75, 106}
, {-65, -1, -20}
, {-50, 85, -25}
, {-46, 81, 85}
, {64, 9, -50}
, {6, 29, 0}
, {-3, 90, 0}
, {-8, 71, 4}
, {-27, -51, 105}
, {-40, -36, 18}
, {108, 91, 12}
}
, {{-80, -11, 9}
, {-91, -43, -58}
, {100, -22, 82}
, {-48, 40, -60}
, {-69, 93, 39}
, {-25, 47, 27}
, {101, -104, -58}
, {55, -112, 63}
, {-8, 12, -54}
, {49, -6, -65}
, {-30, 59, 16}
, {-61, -35, 76}
, {53, 71, 102}
, {94, 73, 16}
, {-41, 31, -4}
, {22, -74, 83}
}
, {{95, -59, -90}
, {35, 28, 35}
, {18, -113, 68}
, {43, 93, 121}
, {-92, 76, -60}
, {63, 99, 69}
, {87, -77, -31}
, {-81, 28, -84}
, {-49, 65, 79}
, {12, 80, -98}
, {68, 81, -63}
, {118, 49, -76}
, {47, 6, -34}
, {67, 31, 57}
, {107, -8, 91}
, {-46, -62, -63}
}
, {{27, -85, -20}
, {68, 47, 33}
, {45, 28, 68}
, {-107, -76, -2}
, {60, 72, -97}
, {-100, -11, 18}
, {-100, 96, -71}
, {45, 82, 12}
, {56, -75, -66}
, {83, -70, -56}
, {-71, 91, 14}
, {-29, 39, -93}
, {101, 47, 28}
, {53, 30, 97}
, {54, -57, -11}
, {11, 91, 77}
}
, {{-48, -31, -68}
, {27, 42, -73}
, {0, -47, -63}
, {33, -48, 93}
, {-101, 90, -42}
, {48, 60, -11}
, {-23, 35, 7}
, {90, 44, 48}
, {109, 4, 16}
, {-76, -41, -78}
, {-11, 87, -46}
, {25, -67, 41}
, {53, 10, -93}
, {103, -71, 46}
, {-49, 88, -84}
, {-49, 37, -5}
}
, {{70, -7, -78}
, {103, -60, -16}
, {24, -79, 96}
, {-8, 43, -119}
, {-26, -21, -80}
, {101, -54, -17}
, {-8, 39, -70}
, {-43, -60, 5}
, {100, 17, 58}
, {45, 69, 77}
, {27, -61, -2}
, {-4, -10, -100}
, {28, -63, 5}
, {52, 1, 40}
, {36, 90, -59}
, {-79, 35, 37}
}
, {{98, -5, 49}
, {12, -28, 107}
, {-83, -71, -10}
, {106, -29, -47}
, {-62, -54, -54}
, {-33, -83, -27}
, {-68, -36, -89}
, {89, 73, -93}
, {8, -60, -69}
, {48, -51, 103}
, {-67, 109, 56}
, {8, -32, 51}
, {76, 34, -45}
, {-109, 73, -27}
, {45, 70, 99}
, {57, -83, 20}
}
, {{-80, -63, -86}
, {-8, 8, 106}
, {96, -68, -23}
, {88, 76, -24}
, {9, 112, -23}
, {26, -74, -51}
, {42, 25, -3}
, {82, -47, -7}
, {25, -66, -91}
, {-43, -40, -90}
, {46, 69, -46}
, {11, -21, -9}
, {-63, 38, -3}
, {-26, 6, 69}
, {-85, -88, 35}
, {63, 59, 79}
}
, {{115, -23, 45}
, {62, -39, -89}
, {68, 15, 20}
, {-29, -76, 38}
, {59, -88, 81}
, {71, -57, 14}
, {-54, 108, -63}
, {85, 105, -9}
, {-11, -45, -93}
, {32, -19, 92}
, {-93, -27, 28}
, {-79, -22, 99}
, {61, -37, -34}
, {65, -64, -31}
, {62, -87, -73}
, {-18, 33, 110}
}
, {{-64, 46, 95}
, {93, -57, 12}
, {72, -70, -38}
, {78, -27, -44}
, {-14, -12, -33}
, {-32, 40, 0}
, {-1, 62, -19}
, {-55, -57, -59}
, {-100, -15, 38}
, {82, -7, -44}
, {84, -95, 24}
, {85, -19, -31}
, {-28, 89, 54}
, {90, 58, -57}
, {53, -6, 3}
, {-76, -7, 41}
}
, {{-65, -89, -19}
, {4, -8, -50}
, {-90, -115, 52}
, {-76, -10, 38}
, {-43, 102, 42}
, {22, -57, -27}
, {-78, 11, -32}
, {-12, -34, -11}
, {47, -40, -101}
, {25, 13, -100}
, {-96, 90, 53}
, {-11, -64, -111}
, {1, 32, -12}
, {-76, 105, -100}
, {-60, 24, -41}
, {116, -38, 39}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE