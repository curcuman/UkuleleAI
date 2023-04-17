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