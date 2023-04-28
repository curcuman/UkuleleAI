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


const int16_t conv1d_93_bias[CONV_FILTERS] = {17, -16, 0, 20, 30, -21, -14, 26, -4, -10, -14, 32, 0, -7, 29, -4, 22, 47, 30, 1, -8, 4, -2, 24, 14, 13, -17, 0, -22, 38, 11, 2}
;

const int16_t conv1d_93_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{95, -79, -95}
, {-24, 68, 28}
, {-49, -23, 4}
, {-69, -21, -43}
, {-49, -14, 16}
, {78, 85, 10}
, {72, -83, -50}
, {75, 44, -21}
, {38, -3, 107}
, {0, 17, 11}
, {-93, 103, -23}
, {63, -114, 14}
, {76, 24, 33}
, {31, 72, 72}
, {-29, -11, 42}
, {0, 35, 49}
}
, {{67, -23, 36}
, {23, -31, -1}
, {19, -9, 98}
, {42, -88, -99}
, {-45, -42, 96}
, {97, 71, 66}
, {91, 74, -79}
, {-103, 31, -8}
, {-70, 49, -101}
, {-71, 71, 87}
, {79, 106, 52}
, {-82, -87, 11}
, {-15, -95, -40}
, {50, -16, 121}
, {31, 81, -18}
, {-40, -74, -40}
}
, {{-33, -77, 60}
, {111, 6, 17}
, {70, 66, 42}
, {-75, 32, 95}
, {13, 68, -62}
, {71, 109, -61}
, {-41, -94, -2}
, {16, 56, -59}
, {-83, -83, -45}
, {39, 0, 61}
, {102, 107, 4}
, {-79, 5, -37}
, {14, -85, 97}
, {-48, 79, -60}
, {-85, 14, 19}
, {50, -12, -50}
}
, {{-48, 17, -26}
, {-28, 72, -65}
, {67, -103, 70}
, {57, 71, -42}
, {111, -94, -80}
, {-50, -75, -7}
, {-85, 51, 94}
, {79, -70, -16}
, {105, 78, 53}
, {26, 10, 84}
, {-93, -104, -56}
, {-67, 39, 67}
, {81, -44, 37}
, {-20, -57, 21}
, {-66, -67, -61}
, {45, 79, -66}
}
, {{53, -18, 41}
, {-85, -22, -74}
, {-43, -96, 2}
, {111, -63, 95}
, {-24, 100, 20}
, {-74, -73, -53}
, {-62, -35, -101}
, {100, 7, 23}
, {-2, 79, -35}
, {-66, -44, -5}
, {109, -2, 59}
, {63, 69, -14}
, {-43, 57, -70}
, {49, -32, 20}
, {16, 78, -45}
, {-7, 55, 78}
}
, {{-73, 60, -45}
, {-57, -105, -5}
, {4, -37, -66}
, {16, -23, 78}
, {-43, -97, 100}
, {-7, 80, -4}
, {92, -92, -22}
, {105, 71, -98}
, {-69, 72, 36}
, {15, 98, 0}
, {-98, 29, 100}
, {-34, 83, -42}
, {17, 6, -98}
, {51, 12, 84}
, {-42, 35, 54}
, {11, -96, 64}
}
, {{-50, -83, -65}
, {-110, -23, 50}
, {-37, 16, -24}
, {3, 114, 51}
, {36, -64, 19}
, {6, 30, 67}
, {-95, -25, 4}
, {105, -48, -76}
, {-70, -54, -51}
, {81, 22, -134}
, {80, 88, 59}
, {-89, -41, 90}
, {-46, -41, 88}
, {38, 92, -69}
, {12, 74, 83}
, {-17, 59, -9}
}
, {{-42, 118, -47}
, {93, -13, -13}
, {-25, -14, 49}
, {29, 37, -6}
, {-72, -69, -93}
, {26, -73, 108}
, {-75, -83, 74}
, {23, 19, -88}
, {116, 11, -9}
, {-48, 69, 75}
, {-32, -93, 70}
, {57, 91, -53}
, {-69, 8, 17}
, {-24, 12, 61}
, {-7, -6, 45}
, {76, -25, 18}
}
, {{10, 69, -16}
, {-12, 80, 83}
, {-94, -100, -94}
, {-66, 14, -84}
, {61, 86, -75}
, {-33, -12, -16}
, {-101, -92, 108}
, {-38, 61, -25}
, {-33, -104, 57}
, {27, 70, 77}
, {-37, -46, -4}
, {-53, 11, -26}
, {-93, 74, 71}
, {1, -61, 106}
, {-43, 50, -91}
, {58, 54, 33}
}
, {{44, -106, -57}
, {21, 79, -113}
, {-62, -22, 67}
, {67, -55, -97}
, {69, -2, -17}
, {92, 23, 75}
, {97, 53, 87}
, {31, 39, 21}
, {-44, 102, -69}
, {91, 80, -80}
, {58, 14, -30}
, {8, 81, -69}
, {-96, 95, -46}
, {-91, 41, -88}
, {-54, -66, -37}
, {-66, 93, -89}
}
, {{108, -98, -24}
, {37, -13, 78}
, {-83, -11, 52}
, {106, -100, -12}
, {-8, -64, 48}
, {-87, 22, 4}
, {-40, 68, -29}
, {-54, 55, -69}
, {-86, -18, -76}
, {-43, 18, 3}
, {-32, -79, 47}
, {57, -32, -7}
, {-95, -83, -99}
, {-23, 63, 89}
, {71, 59, 102}
, {78, 102, 60}
}
, {{96, 50, -24}
, {15, 46, -20}
, {24, -20, 110}
, {-64, -31, -72}
, {74, -85, 6}
, {-92, 104, -56}
, {-89, 24, -75}
, {104, 54, 81}
, {111, -82, -25}
, {80, 88, 118}
, {-70, 9, 85}
, {-13, 12, -53}
, {-18, -89, -94}
, {-5, 65, -32}
, {27, -42, 59}
, {-4, 42, 85}
}
, {{-23, 92, 20}
, {-11, -48, -94}
, {37, 44, -99}
, {-34, 101, 25}
, {55, 92, -35}
, {7, -78, 47}
, {28, 73, 4}
, {7, 31, -92}
, {96, -9, 75}
, {-4, -69, -70}
, {-68, 53, 78}
, {88, 72, 73}
, {-26, -75, 88}
, {9, -21, -96}
, {101, -68, -33}
, {38, -98, -59}
}
, {{21, 52, -40}
, {-9, 37, -51}
, {23, -21, -82}
, {-71, -39, -17}
, {-24, -80, -98}
, {92, -24, -86}
, {-4, -49, 60}
, {-1, -1, 68}
, {-49, 115, 90}
, {76, 44, 71}
, {7, 17, 84}
, {94, -23, -93}
, {104, -23, -52}
, {-91, -60, 37}
, {84, 21, 111}
, {-52, 41, 49}
}
, {{-56, -64, -67}
, {-96, 47, -48}
, {-21, 13, 60}
, {60, 82, 10}
, {0, 57, -89}
, {-6, 8, -25}
, {-50, -58, 96}
, {34, -68, -72}
, {-84, -48, -50}
, {109, -43, 23}
, {-36, -114, -18}
, {36, 34, 94}
, {-44, -74, -81}
, {114, -51, 46}
, {-18, -50, -40}
, {-72, 4, 7}
}
, {{-67, 52, -91}
, {-93, 0, -46}
, {81, -8, -36}
, {28, -4, -108}
, {-34, -105, 15}
, {-9, 49, 0}
, {-76, -66, -74}
, {69, 76, -26}
, {8, 69, 49}
, {-37, 103, -88}
, {63, 94, -98}
, {52, 79, 108}
, {-40, 10, 101}
, {100, 88, -92}
, {24, -6, 83}
, {-9, 68, 91}
}
, {{104, -93, -103}
, {-80, -12, 5}
, {78, 97, 1}
, {71, -45, -33}
, {-80, 12, -29}
, {88, 31, 117}
, {70, -30, 29}
, {41, 9, 105}
, {72, -90, -79}
, {-91, -39, -30}
, {-65, 12, -59}
, {57, 4, 0}
, {-56, -100, -65}
, {40, 109, -78}
, {11, 18, 42}
, {-13, -14, 95}
}
, {{-62, -73, -18}
, {-51, -88, -25}
, {0, -73, -58}
, {-23, -31, 62}
, {-48, -41, 47}
, {-63, 104, 39}
, {-43, -78, -42}
, {6, 97, 17}
, {-22, 22, -38}
, {-11, 62, -82}
, {-33, 112, 46}
, {53, -53, 107}
, {17, 3, -107}
, {39, -78, 114}
, {97, 48, 105}
, {17, 99, -23}
}
, {{-37, 45, -51}
, {109, 12, -15}
, {1, 81, -27}
, {-43, -67, -12}
, {59, -91, -63}
, {47, 27, -8}
, {108, -8, 94}
, {72, -66, 28}
, {123, 65, -11}
, {47, -8, -1}
, {-81, 96, 12}
, {-22, -61, -24}
, {64, -7, 104}
, {80, 75, -77}
, {-56, -32, -84}
, {-38, 46, 113}
}
, {{26, 6, -2}
, {43, 73, -91}
, {-4, -74, -4}
, {88, -71, -40}
, {53, 10, -80}
, {55, 32, 0}
, {78, 48, -9}
, {-96, 34, -5}
, {-84, 94, 65}
, {-67, -10, 15}
, {-65, -110, 1}
, {-12, 60, -103}
, {29, 95, -84}
, {-67, -92, 80}
, {-83, 46, 33}
, {-39, 95, 74}
}
, {{-51, -25, 44}
, {71, -67, -70}
, {-111, -64, 26}
, {65, -19, 109}
, {-55, -53, -94}
, {50, 85, 50}
, {-17, 106, 53}
, {-86, 9, -19}
, {-88, 58, 46}
, {-19, -85, 46}
, {-81, -29, -63}
, {-92, -62, -80}
, {-48, -49, 16}
, {66, -43, 58}
, {112, -74, 17}
, {-69, 40, -63}
}
, {{98, -55, -3}
, {-30, -16, -86}
, {-89, 22, -48}
, {48, -39, 45}
, {-78, -9, 103}
, {-36, 83, -63}
, {-93, -71, -71}
, {-46, 7, 82}
, {51, -90, -3}
, {-93, -11, 34}
, {80, 107, 20}
, {28, -8, 40}
, {-30, 48, 8}
, {-40, 71, -8}
, {-85, -38, 84}
, {75, -6, 26}
}
, {{-35, -89, 100}
, {53, -80, -46}
, {-54, -49, 80}
, {6, 94, 27}
, {-93, -71, 90}
, {-100, 64, -43}
, {-52, 82, 77}
, {-38, 115, 15}
, {57, 24, -5}
, {20, 69, -16}
, {29, -33, 41}
, {-53, 111, 58}
, {-66, 100, -85}
, {80, -53, 50}
, {103, -97, -74}
, {18, -27, 72}
}
, {{-25, 90, -31}
, {-92, 104, -75}
, {-80, 87, 7}
, {-89, -36, -15}
, {84, 17, -79}
, {75, 109, 18}
, {-90, 100, -70}
, {40, 61, 107}
, {50, -52, 81}
, {1, -59, 3}
, {0, -64, 12}
, {-67, -52, -49}
, {68, 7, -97}
, {90, -50, 48}
, {-45, 0, 24}
, {-50, 61, -30}
}
, {{-86, 51, 7}
, {90, 96, 54}
, {-34, 108, -81}
, {-76, -98, -53}
, {-28, 94, 64}
, {6, 61, 66}
, {-14, -51, 60}
, {103, -15, -4}
, {-94, -36, -54}
, {99, 99, 72}
, {4, -16, 95}
, {26, 68, -63}
, {57, 13, -97}
, {0, -19, 104}
, {-110, -100, -75}
, {-63, -43, -36}
}
, {{-102, -93, 62}
, {33, -9, -74}
, {52, 38, 10}
, {33, 56, -34}
, {-18, -51, -86}
, {-12, 32, 53}
, {-47, -53, 85}
, {90, 63, -26}
, {45, -78, 90}
, {-75, -80, -9}
, {-15, 103, 36}
, {68, 79, -63}
, {84, -6, 19}
, {36, 103, 17}
, {-33, -16, -84}
, {40, -22, 105}
}
, {{-33, -38, 84}
, {-62, 48, 71}
, {-62, 84, 102}
, {48, -52, -30}
, {111, -72, -88}
, {-102, -44, -19}
, {-102, -35, 9}
, {108, -86, -3}
, {77, -64, -71}
, {13, -54, 3}
, {32, 82, -18}
, {1, 22, -22}
, {-41, -21, -7}
, {86, 15, -37}
, {57, -60, 78}
, {37, -50, 104}
}
, {{38, -55, 63}
, {0, -68, -77}
, {22, -29, -18}
, {102, -69, -47}
, {-66, -31, 90}
, {-2, 13, -20}
, {68, 29, 66}
, {87, -16, -82}
, {28, -38, -7}
, {68, -18, 50}
, {71, -1, 44}
, {-113, 45, 111}
, {-99, 81, -95}
, {111, -64, 19}
, {23, -18, 74}
, {-75, 26, 31}
}
, {{47, 81, 29}
, {62, -68, 51}
, {103, -14, 52}
, {8, -76, -4}
, {53, -83, -20}
, {39, -49, -39}
, {-29, 71, 61}
, {94, -50, 10}
, {-4, -54, 94}
, {-46, 24, -102}
, {52, -40, 57}
, {102, 29, -42}
, {50, -113, -40}
, {-62, 98, 38}
, {-46, -40, 6}
, {-85, -88, -14}
}
, {{-39, 15, 115}
, {28, -5, 52}
, {41, -95, 68}
, {-15, 74, -68}
, {78, -46, 48}
, {-83, 109, -90}
, {-14, 106, 95}
, {-28, 50, 0}
, {-32, -17, -73}
, {41, -19, -21}
, {65, 98, -63}
, {3, 101, 10}
, {70, 3, -80}
, {64, -21, -48}
, {4, -75, -73}
, {-46, 38, -106}
}
, {{-16, -84, 41}
, {-79, 32, -82}
, {39, -51, -76}
, {-29, 80, -90}
, {52, -37, 83}
, {-71, 55, -111}
, {34, 29, -101}
, {65, -19, -22}
, {11, -45, 73}
, {-27, 39, -10}
, {100, -85, 18}
, {119, 62, 56}
, {-40, 59, 100}
, {-13, -58, 99}
, {-18, 62, 40}
, {-68, -37, -90}
}
, {{-16, -25, 17}
, {-94, 49, -65}
, {-66, -99, -81}
, {24, 77, 65}
, {44, -98, -9}
, {-107, 59, -63}
, {-41, 85, -28}
, {-46, 13, 47}
, {78, 98, -57}
, {-5, -9, -56}
, {-86, 90, -59}
, {-23, 105, 11}
, {-103, -27, -27}
, {82, 31, 19}
, {-22, 41, -44}
, {-65, 123, 72}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE