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