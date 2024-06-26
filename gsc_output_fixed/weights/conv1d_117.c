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