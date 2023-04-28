/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Universit� C�te d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_34_bias[CONV_FILTERS] = {-8, 28, 15, -20, -26, 4, -10, 7, 7, -21, 36, 2, -7, 27, 11, -6, -21, -14, -4, 9, -8, -3, -5, 8, -1, -13, -2, 29, 2, 14, 32, -23, -10, -21, 6, 11, -21, 17, 42, 28, -13, 37, -7, -5, 7, 7, 13, -5, -19, -18, 5, 6, 2, -7, 2, 0, -1, -4, -9, 0, 34, 42, -5, -5}
;

const int16_t conv1d_34_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{28, -84, -47}
, {-55, -51, -14}
, {25, 52, -65}
, {-50, 39, 35}
, {70, 49, 30}
, {60, 55, 37}
, {29, -43, 61}
, {49, -66, 3}
, {-3, -28, 23}
, {42, -77, -58}
, {55, 46, -14}
, {49, 28, -17}
, {-10, 49, -19}
, {-34, 18, -34}
, {50, -58, 44}
, {36, -13, -53}
, {-26, 56, 15}
, {-9, -74, 26}
, {39, 51, -18}
, {-46, 25, -46}
, {-84, 56, -51}
, {-27, 25, 48}
, {53, 43, -70}
, {-7, 0, -64}
, {-37, -83, 9}
, {-63, -39, -59}
, {-66, -23, -47}
, {71, 56, 56}
, {-34, 35, 17}
, {-80, 77, 10}
, {-46, 22, 67}
, {-79, -23, -69}
}
, {{18, -92, 56}
, {-25, 54, -39}
, {-5, -24, 18}
, {-45, 51, 35}
, {42, -76, 37}
, {57, -42, -35}
, {-6, -65, 16}
, {-67, 9, 17}
, {-29, -30, -29}
, {-3, 22, -26}
, {41, -6, 46}
, {6, -38, 8}
, {79, -10, -1}
, {-30, 96, 54}
, {-62, -19, 17}
, {-40, 22, -55}
, {38, 31, -43}
, {-18, -40, 18}
, {23, -24, -48}
, {26, 4, -103}
, {41, 118, 132}
, {65, 108, 25}
, {-54, -48, 43}
, {-55, -12, 42}
, {-58, 1, 30}
, {-8, 21, 0}
, {10, 41, 40}
, {82, 43, -53}
, {56, -40, -43}
, {19, 43, -42}
, {20, -11, -23}
, {29, -61, -22}
}
, {{34, -76, -54}
, {57, -12, -75}
, {9, -12, -47}
, {48, 56, -48}
, {-5, 24, -43}
, {58, -44, -32}
, {35, -33, 30}
, {-78, -15, -2}
, {-16, 27, -16}
, {-15, -64, -60}
, {60, 19, -59}
, {36, -21, -12}
, {45, 18, -47}
, {5, 54, 54}
, {68, -39, -41}
, {-61, -77, 16}
, {-26, -84, -55}
, {-16, 31, 26}
, {9, 75, -53}
, {27, -40, 47}
, {-31, 54, 52}
, {11, 15, 138}
, {-27, 46, -50}
, {-2, 62, -26}
, {0, 31, -52}
, {82, -22, 44}
, {41, -15, -48}
, {48, -29, -28}
, {-44, -72, 31}
, {76, -41, 40}
, {-58, 84, 61}
, {-48, 53, -46}
}
, {{-20, -53, -56}
, {55, 78, -56}
, {-44, -60, 22}
, {-8, 75, -70}
, {-3, -49, -72}
, {54, -23, 10}
, {36, -82, 53}
, {-23, -2, 43}
, {-60, -47, 34}
, {-85, -65, 46}
, {15, 0, -13}
, {-67, -25, -52}
, {12, 56, 20}
, {69, -22, -96}
, {16, 70, 15}
, {1, 51, 12}
, {-87, 26, 4}
, {-63, -54, -84}
, {-32, -79, -46}
, {-43, 26, 30}
, {27, -28, 62}
, {-102, 92, 28}
, {-46, -8, -30}
, {30, 49, 59}
, {12, 3, -18}
, {-70, -56, -64}
, {-14, 7, 62}
, {-14, -6, -4}
, {9, 2, 58}
, {23, 43, -45}
, {34, 49, 0}
, {44, -16, 63}
}
, {{26, -30, 34}
, {20, 55, 13}
, {8, 50, -80}
, {-92, -26, -62}
, {57, 71, 14}
, {1, 38, 52}
, {-23, 4, 66}
, {-24, 72, -31}
, {49, 8, 8}
, {34, 33, -34}
, {7, 20, -76}
, {58, 12, 34}
, {-56, 47, -57}
, {54, -72, 50}
, {-46, -56, 1}
, {56, 49, 28}
, {-36, 49, -30}
, {-17, 20, -94}
, {-14, 12, -25}
, {31, 82, 0}
, {-11, -41, 5}
, {-30, 58, 124}
, {-12, 7, 40}
, {4, -20, -55}
, {21, 8, -48}
, {-32, 15, -49}
, {12, 18, 48}
, {69, -35, -28}
, {-54, 13, -48}
, {10, 23, -81}
, {-32, 41, 40}
, {-2, -25, 22}
}
, {{43, -54, 72}
, {-13, -83, -33}
, {74, -17, 12}
, {40, 22, -1}
, {-29, -9, -79}
, {0, -26, -3}
, {32, -71, -46}
, {34, -58, -44}
, {-50, -35, -67}
, {51, -18, 60}
, {93, -36, -83}
, {74, 2, 30}
, {-41, -48, 51}
, {-27, 25, -39}
, {6, -25, 55}
, {30, -68, 78}
, {-1, 78, -48}
, {41, 11, -64}
, {-51, -2, -34}
, {-74, 15, 41}
, {-55, 53, -116}
, {-38, 41, 38}
, {-53, 46, 40}
, {-61, -33, -87}
, {-42, 62, 85}
, {-30, -52, 24}
, {-62, 56, -49}
, {3, -16, 11}
, {-61, 43, 70}
, {15, 51, -30}
, {14, 57, -11}
, {62, -52, -57}
}
, {{30, 15, -81}
, {49, -32, 52}
, {-13, -36, 28}
, {-44, -10, -87}
, {61, 63, 17}
, {34, 23, -6}
, {64, 43, 5}
, {0, -37, -4}
, {20, -40, 56}
, {25, -50, 37}
, {30, 17, -35}
, {-52, -67, 26}
, {-53, 4, 13}
, {-51, 49, -77}
, {36, 78, 74}
, {3, -37, 17}
, {2, -57, 26}
, {-68, -67, -60}
, {39, -58, -76}
, {54, -68, 7}
, {37, -14, -95}
, {30, 91, 67}
, {47, -33, 59}
, {63, -51, 81}
, {-54, -29, 25}
, {-30, -34, 3}
, {57, -34, -26}
, {-40, 39, -46}
, {12, -39, -35}
, {55, 26, -6}
, {-39, 8, -19}
, {-5, 0, -7}
}
, {{48, 3, -6}
, {85, 21, 28}
, {-41, -32, 9}
, {-34, -15, -41}
, {-19, -34, 60}
, {76, 33, -21}
, {91, -88, 46}
, {7, 43, -73}
, {38, -19, 17}
, {26, -13, -18}
, {-58, -39, -8}
, {-7, -12, -51}
, {52, 25, 30}
, {-69, -66, -7}
, {75, 86, -48}
, {31, 73, -32}
, {-25, -12, 71}
, {-27, -73, -96}
, {47, -9, -80}
, {-28, 112, -7}
, {-1, -16, -43}
, {180, 21, -44}
, {-57, -13, 63}
, {1, 56, 53}
, {-49, 26, 21}
, {3, 28, -11}
, {-56, 16, 6}
, {64, -55, 69}
, {59, -5, -25}
, {-93, -49, -1}
, {-23, 35, -6}
, {-24, 20, 14}
}
, {{-19, -6, -17}
, {86, 35, 46}
, {-21, 25, 16}
, {-64, -23, -36}
, {81, 13, -16}
, {33, -25, -51}
, {71, 67, 63}
, {36, 71, -27}
, {-49, 39, -12}
, {-42, 45, 6}
, {18, 53, -36}
, {6, 30, 38}
, {-31, 46, -2}
, {-50, -8, -27}
, {-33, -67, 23}
, {22, -18, -48}
, {-2, 16, 40}
, {-23, -1, -28}
, {-6, -13, 14}
, {-69, -53, -19}
, {-40, -45, 10}
, {121, 61, -15}
, {13, -53, 61}
, {26, -41, 7}
, {-52, 94, -38}
, {-63, -31, 40}
, {-18, -38, 18}
, {-38, 69, -32}
, {30, -33, -9}
, {46, 2, 30}
, {52, 6, -48}
, {-56, -48, -50}
}
, {{16, -80, -14}
, {-6, -56, 63}
, {37, -8, -92}
, {42, -26, 31}
, {-6, -45, -73}
, {-26, 47, -35}
, {-44, 4, 21}
, {26, -42, 71}
, {40, -40, 2}
, {-45, 58, -45}
, {-36, 7, -43}
, {-43, 27, -50}
, {-49, 2, 9}
, {81, -4, -15}
, {42, 30, -67}
, {-32, 0, -40}
, {7, 42, -2}
, {52, -3, -34}
, {-60, 10, 22}
, {-17, -46, -51}
, {-104, 50, -3}
, {-30, -35, 21}
, {24, 19, -50}
, {37, -78, 25}
, {23, -30, 0}
, {8, 6, -14}
, {-45, -1, -71}
, {-64, 34, 68}
, {22, 56, 63}
, {17, -68, -38}
, {-32, 33, 27}
, {-60, -42, 17}
}
, {{-5, 52, 79}
, {22, -7, 31}
, {-58, 2, -91}
, {-64, -16, 36}
, {-46, 35, 22}
, {30, 72, -3}
, {-55, 1, -58}
, {37, -3, -44}
, {-8, 61, -25}
, {8, -49, 61}
, {-53, -39, -62}
, {38, -55, -12}
, {84, 56, -39}
, {-1, 36, -7}
, {-45, -26, 9}
, {-66, 32, 66}
, {-58, -42, -45}
, {-6, -37, 27}
, {-56, 24, 72}
, {7, -3, 70}
, {6, 61, 63}
, {52, 17, 77}
, {64, -9, 31}
, {49, -24, -66}
, {5, 37, 103}
, {28, -69, -40}
, {-8, 40, 43}
, {-6, -59, -49}
, {6, -36, 71}
, {19, 74, -20}
, {-47, 63, -66}
, {22, 63, 16}
}
, {{29, -13, 19}
, {-6, 87, -48}
, {-42, 9, 29}
, {-73, 54, 60}
, {-20, 36, -30}
, {16, -26, 11}
, {18, -59, 15}
, {66, -66, 29}
, {-27, -10, 0}
, {47, -61, 55}
, {-1, -60, -5}
, {0, 13, -20}
, {59, -70, -18}
, {50, 18, 13}
, {-14, 0, 58}
, {-3, 62, 35}
, {-64, 37, 13}
, {42, -25, -13}
, {-28, 53, -22}
, {-10, -55, -10}
, {-25, 19, 35}
, {3, -37, 118}
, {-34, 42, -9}
, {64, -66, -25}
, {-63, -85, -76}
, {-17, 14, -20}
, {38, -70, -11}
, {14, 55, -75}
, {67, 32, -53}
, {-60, -20, -28}
, {-9, 12, 46}
, {35, -64, -21}
}
, {{-16, 71, -18}
, {74, -37, -1}
, {7, -83, -25}
, {37, 41, -13}
, {50, -37, 45}
, {-2, 33, -6}
, {-10, -2, -25}
, {64, -30, -10}
, {-13, 16, -25}
, {18, -23, -30}
, {50, 46, -22}
, {-44, 12, -20}
, {-61, 5, -66}
, {-48, -20, -24}
, {54, -1, -68}
, {17, 53, 6}
, {-4, -18, 2}
, {1, -31, 43}
, {64, -56, -49}
, {-58, 8, -71}
, {5, -14, 37}
, {86, -57, 25}
, {35, 63, 59}
, {77, 5, -11}
, {-54, 4, -3}
, {-89, -56, 40}
, {25, -65, 64}
, {72, -55, 7}
, {56, 35, -8}
, {-52, 21, -5}
, {-68, -30, -76}
, {55, -39, -40}
}
, {{64, -33, -2}
, {-59, 43, 50}
, {-77, 6, 19}
, {-51, 11, 77}
, {23, 29, 32}
, {47, 9, 7}
, {14, 9, -13}
, {18, 35, 49}
, {28, -44, -12}
, {24, 44, 46}
, {-55, -62, -21}
, {39, -59, -33}
, {95, -16, 3}
, {-26, 33, -7}
, {-27, 24, -17}
, {49, -6, -7}
, {49, -46, -44}
, {-18, 37, -16}
, {83, -25, 0}
, {17, -8, -37}
, {115, 15, 27}
, {49, 20, 62}
, {-13, -23, 27}
, {-55, -26, -9}
, {-6, -33, -89}
, {-36, 30, -64}
, {-38, -44, -18}
, {25, 72, -63}
, {-20, 22, 38}
, {-7, 14, 43}
, {42, 46, 23}
, {-59, 43, -95}
}
, {{-5, -74, 6}
, {-24, 51, -34}
, {40, -71, -49}
, {49, 41, 61}
, {4, -60, 56}
, {-56, 16, -55}
, {-8, 55, -72}
, {-73, -42, 65}
, {-51, 15, 41}
, {9, 41, 27}
, {-59, -15, 29}
, {-57, 44, 8}
, {-29, 25, -41}
, {61, -26, -7}
, {17, -9, 15}
, {62, -2, 67}
, {3, -9, -9}
, {81, 75, 40}
, {-53, -34, -75}
, {67, 6, -62}
, {-16, 1, 22}
, {-47, 78, 78}
, {-67, -43, 62}
, {-43, -45, 39}
, {-63, 21, 50}
, {18, -61, -23}
, {40, -60, -68}
, {-21, -36, 72}
, {26, 10, -69}
, {-51, 64, -24}
, {-77, 52, -16}
, {-31, -45, -28}
}
, {{-23, 7, 52}
, {-3, -27, 62}
, {56, 26, 40}
, {58, -52, 69}
, {38, -56, 71}
, {29, 6, 45}
, {10, 13, -62}
, {25, 31, -69}
, {-65, -52, -3}
, {34, -4, 72}
, {-19, -87, -73}
, {79, 9, -15}
, {-27, -39, 47}
, {-36, -38, 18}
, {66, -50, -64}
, {50, -20, -61}
, {-48, 45, -37}
, {-55, 31, -20}
, {45, -30, 17}
, {-36, -58, -11}
, {-47, 30, 13}
, {-17, -40, 48}
, {38, 8, 57}
, {-33, -74, -22}
, {27, -2, 37}
, {-7, -29, 14}
, {-66, 50, 51}
, {-61, -49, 50}
, {-19, -42, 60}
, {-26, -103, -31}
, {19, -74, 79}
, {20, -39, -16}
}
, {{89, 67, -45}
, {-21, 88, 59}
, {57, -66, -25}
, {-84, 23, -54}
, {-1, -44, 20}
, {6, -63, 31}
, {-39, 9, 66}
, {54, 3, 31}
, {11, 50, -32}
, {32, 4, 64}
, {12, -80, -43}
, {43, -1, 29}
, {-46, 28, 14}
, {-31, 25, 53}
, {54, -36, 29}
, {-11, 0, -51}
, {-44, 34, -7}
, {27, -39, -83}
, {-74, -3, 0}
, {13, -18, 68}
, {26, -18, 34}
, {-145, 27, 20}
, {-25, 8, 14}
, {-47, 54, 4}
, {44, 8, -66}
, {-28, 22, -69}
, {19, -53, -21}
, {2, 63, 48}
, {-31, 9, 7}
, {-78, -39, -104}
, {21, -74, -13}
, {-3, 52, 47}
}
, {{43, -59, -51}
, {47, 40, -67}
, {58, -2, 17}
, {41, 41, 65}
, {-48, 9, 15}
, {31, 45, 30}
, {-9, -18, -27}
, {-58, 55, 37}
, {-45, -43, -52}
, {36, 28, -51}
, {-34, 52, 42}
, {-26, 59, 52}
, {-45, -22, -68}
, {-49, 22, -7}
, {-29, -9, 47}
, {57, -55, -74}
, {-72, 54, 72}
, {40, -70, 44}
, {-49, 59, -47}
, {65, -68, 39}
, {0, 60, -63}
, {-7, -63, 45}
, {7, -75, -64}
, {-26, -66, -55}
, {-2, 0, 8}
, {86, 8, 32}
, {-12, 71, -5}
, {12, -32, 24}
, {40, -48, -32}
, {-29, -33, 17}
, {-5, 4, 31}
, {-13, -8, 46}
}
, {{-10, -35, -63}
, {82, 0, -17}
, {-51, 51, 62}
, {-27, -8, -61}
, {-40, 90, 47}
, {-39, 16, 6}
, {77, 84, 43}
, {34, -46, 30}
, {31, 51, -41}
, {-38, -12, 37}
, {-84, 11, 56}
, {-37, 53, 8}
, {-10, 13, 68}
, {59, -61, 11}
, {-47, -25, 21}
, {52, 14, 30}
, {49, -7, 15}
, {-53, -71, -24}
, {13, -27, -78}
, {-19, 66, -26}
, {-25, -21, 23}
, {51, 69, 29}
, {43, 45, 40}
, {16, 70, -28}
, {-37, -55, -3}
, {28, -28, 41}
, {-64, 41, 49}
, {46, 2, 18}
, {-48, -66, -55}
, {-29, -44, 46}
, {-59, -78, -70}
, {57, -41, -64}
}
, {{-24, 15, 43}
, {70, 41, 16}
, {19, 4, -56}
, {-53, -64, 16}
, {54, -16, 32}
, {-39, -15, -55}
, {13, 48, -55}
, {9, -40, 43}
, {-24, -10, 0}
, {8, 25, 8}
, {67, 34, 84}
, {17, -9, -18}
, {18, -37, -47}
, {7, 29, -60}
, {-41, -45, -38}
, {-55, 40, 38}
, {55, -51, 49}
, {42, 67, 40}
, {53, 50, 14}
, {56, -25, 47}
, {49, 9, -26}
, {207, -31, 15}
, {-3, -57, -68}
, {70, -60, -18}
, {46, 13, 61}
, {90, 91, 27}
, {64, 8, -16}
, {25, 6, -13}
, {24, -2, 5}
, {-35, -27, 35}
, {67, -38, 40}
, {-20, 12, 55}
}
, {{-57, 19, -31}
, {-15, -48, -61}
, {-31, 54, 15}
, {-56, -74, -58}
, {-40, 0, 72}
, {-48, 79, -8}
, {25, -49, -80}
, {51, -25, -53}
, {54, 3, -46}
, {-69, -37, 49}
, {-73, 79, -18}
, {-19, 9, -44}
, {-31, 31, -39}
, {42, -46, 0}
, {-33, 16, 61}
, {76, -44, 63}
, {48, 68, 23}
, {-7, 49, -29}
, {-62, -48, 55}
, {24, 7, 59}
, {22, -62, 42}
, {57, 27, 3}
, {-84, -5, -26}
, {-57, -2, 11}
, {26, 17, -26}
, {-39, -18, 77}
, {25, 5, -13}
, {-60, 45, 16}
, {-61, -54, 38}
, {-52, 56, 81}
, {47, 29, -35}
, {-9, -29, 30}
}
, {{-46, -2, -22}
, {-73, 17, -11}
, {-72, 39, 64}
, {-27, 0, -65}
, {97, 62, -13}
, {26, -35, 38}
, {-36, 29, -43}
, {-54, -13, 74}
, {-28, -34, 52}
, {68, 55, -75}
, {5, -74, 11}
, {9, 9, -61}
, {-73, -35, -8}
, {30, -97, -19}
, {-81, -1, -21}
, {8, -20, 61}
, {19, -55, 68}
, {37, 23, -50}
, {-64, -11, 44}
, {-71, -44, -25}
, {35, -70, 70}
, {-9, 34, 8}
, {47, -34, -23}
, {-63, -12, -20}
, {-46, 57, 43}
, {-88, 41, -51}
, {29, -37, -47}
, {73, -26, 3}
, {28, -64, -35}
, {27, -68, -74}
, {3, -64, 10}
, {60, -17, 37}
}
, {{30, -28, 90}
, {-67, 77, -4}
, {7, -38, -1}
, {20, -2, -6}
, {9, -4, 49}
, {-12, 53, 8}
, {12, 65, 47}
, {-58, 9, 42}
, {-35, 49, -19}
, {64, 53, 47}
, {-67, -10, 53}
, {29, 25, -66}
, {37, -48, 44}
, {12, -29, -36}
, {-17, -44, -24}
, {13, 25, 1}
, {1, 49, -43}
, {-41, -78, -63}
, {-45, 0, 42}
, {-38, -28, -35}
, {72, -19, 9}
, {37, 147, 48}
, {-15, 72, 16}
, {38, -51, 19}
, {-26, -23, 0}
, {26, 8, -11}
, {-71, -15, 56}
, {72, 2, 39}
, {-2, 2, -33}
, {-14, -61, -62}
, {62, -38, -12}
, {30, -29, -4}
}
, {{-6, 54, 4}
, {89, 68, -14}
, {-33, 35, 38}
, {9, 89, 60}
, {-5, 3, -100}
, {-89, 4, 14}
, {29, 40, -20}
, {-15, -65, -54}
, {45, -26, -16}
, {-75, -77, 2}
, {-6, -56, 40}
, {-85, -46, 18}
, {-41, 46, 84}
, {-8, 49, -59}
, {-76, -22, -4}
, {-66, -25, 32}
, {8, -9, 42}
, {-20, -65, 30}
, {24, 24, -89}
, {-57, 3, 18}
, {42, -43, 27}
, {86, 21, 132}
, {-28, -29, 77}
, {20, 58, -17}
, {-26, -50, 35}
, {-6, -58, 56}
, {-18, -29, -26}
, {18, -33, 10}
, {-62, -47, 4}
, {14, -72, -15}
, {3, 79, 25}
, {-11, -54, 9}
}
, {{28, 74, 18}
, {-62, -30, 2}
, {-45, -15, 3}
, {-19, -52, 9}
, {-19, 25, -18}
, {-2, 26, -76}
, {72, 28, 10}
, {41, 16, -16}
, {-23, 39, 14}
, {35, 68, -48}
, {57, -24, -90}
, {-36, 11, 32}
, {23, -2, -17}
, {-2, -1, 16}
, {-3, -44, -55}
, {-41, 67, -64}
, {15, 68, 6}
, {-10, 22, 66}
, {0, -50, -71}
, {-41, 76, -51}
, {18, 64, 32}
, {158, 63, -38}
, {19, -41, -23}
, {-19, 30, 60}
, {53, 46, -13}
, {-23, -39, -58}
, {38, 50, -54}
, {1, -8, 82}
, {30, 62, 57}
, {33, -39, -73}
, {6, -38, 51}
, {-42, 7, 55}
}
, {{23, 24, 0}
, {35, -24, -27}
, {-47, 61, -8}
, {53, 14, -26}
, {41, -58, 8}
, {22, -42, 39}
, {40, -18, -65}
, {-26, 68, 43}
, {31, -4, -40}
, {-21, -82, -55}
, {-59, 14, -19}
, {-58, 83, -24}
, {3, 0, 58}
, {-32, 59, 18}
, {-59, 11, -14}
, {14, -31, -71}
, {-14, -38, 41}
, {-50, -41, -61}
, {-8, 56, -60}
, {-9, -6, -57}
, {-24, -32, -26}
, {-70, -48, 14}
, {-18, -29, -11}
, {-65, -57, 4}
, {-68, 65, 1}
, {-26, 65, 5}
, {-52, -35, -55}
, {53, -56, -53}
, {36, -27, -53}
, {9, -32, -53}
, {-44, -38, -52}
, {-38, -50, -51}
}
, {{-53, 67, 67}
, {70, 34, 58}
, {23, -65, -12}
, {27, -60, -86}
, {26, -67, -48}
, {27, -27, 31}
, {17, 37, 82}
, {-12, 60, 19}
, {-25, 14, 14}
, {30, -22, 30}
, {-2, -36, -39}
, {62, -44, -25}
, {46, 50, -61}
, {24, 53, 40}
, {0, 16, -47}
, {60, 34, -25}
, {37, -78, -14}
, {-56, 63, 72}
, {-39, 27, 74}
, {43, 24, 61}
, {52, 33, 63}
, {8, 30, 30}
, {-66, 18, 22}
, {48, -55, 74}
, {-20, 30, -25}
, {37, 21, 39}
, {-53, 49, 0}
, {-20, -8, 11}
, {38, 21, -50}
, {-26, -33, 32}
, {-20, 8, -5}
, {-46, 10, 28}
}
, {{-46, 10, 20}
, {69, 82, -50}
, {-4, 20, 45}
, {72, 105, 12}
, {-7, 32, 73}
, {-53, -70, 49}
, {-69, -2, 3}
, {-29, 9, -26}
, {-11, -32, 21}
, {-80, -52, 61}
, {-79, -50, 11}
, {-62, -60, -32}
, {14, -47, 82}
, {-20, 30, -46}
, {-81, 55, -7}
, {56, -77, -48}
, {-53, -92, -72}
, {71, -1, 20}
, {-12, -45, 2}
, {36, -56, -6}
, {16, 48, 129}
, {32, 103, 134}
, {-6, 22, 79}
, {10, -60, 46}
, {-44, -73, -28}
, {44, 32, 37}
, {3, 24, -53}
, {67, -46, -28}
, {-55, -48, -42}
, {59, 13, -25}
, {-54, 59, 16}
, {-81, 37, 2}
}
, {{-48, -29, 39}
, {-62, 0, 0}
, {15, 27, 29}
, {36, 27, -21}
, {38, -68, 32}
, {23, -84, 48}
, {5, 10, 26}
, {17, 52, -65}
, {24, -73, 11}
, {69, -9, -46}
, {69, 38, 30}
, {3, 31, 20}
, {40, -70, 0}
, {16, -5, 57}
, {-30, -37, 42}
, {2, -20, 13}
, {63, -60, -73}
, {-32, -28, 12}
, {-45, 45, -77}
, {-1, -54, -20}
, {-32, 50, -43}
, {-12, -48, 39}
, {-33, -33, -13}
, {-38, -88, 19}
, {-26, -36, -16}
, {0, -19, 34}
, {50, 39, 25}
, {31, 21, 6}
, {3, -21, -51}
, {48, -61, 55}
, {-72, 66, 23}
, {34, -85, -2}
}
, {{32, -9, 35}
, {8, 39, 47}
, {-40, -64, 45}
, {-18, -70, -9}
, {68, -83, -1}
, {-20, -63, -63}
, {-56, 48, -60}
, {-87, 32, 18}
, {11, 39, 24}
, {36, 2, -54}
, {-48, -16, -27}
, {96, -53, -2}
, {-66, -62, -16}
, {-44, 20, 93}
, {62, -85, -13}
, {-27, 47, 66}
, {60, 27, -44}
, {31, -9, 37}
, {-71, -6, -29}
, {2, -31, -64}
, {41, 83, 67}
, {144, 50, 54}
, {12, -20, -37}
, {-89, -26, -52}
, {37, 58, -103}
, {-11, -12, -69}
, {63, -2, 8}
, {-11, 51, 11}
, {-29, 91, 4}
, {-25, -77, 9}
, {-18, 26, 13}
, {-23, 71, -44}
}
, {{-96, -20, -53}
, {57, 88, 10}
, {-33, 52, 68}
, {20, -6, -4}
, {-44, -38, -16}
, {-14, -43, 30}
, {12, 71, 58}
, {6, 91, -41}
, {69, 100, 83}
, {40, 46, 20}
, {57, -48, 2}
, {-19, -72, -8}
, {10, -39, -41}
, {67, 44, 69}
, {-14, -31, -70}
, {-6, 19, 12}
, {-67, -28, -3}
, {46, 88, -14}
, {32, -47, 100}
, {-25, 9, -44}
, {32, -2, 49}
, {45, 88, -48}
, {-10, 9, 14}
, {57, 46, -53}
, {-61, -20, -29}
, {-43, 54, -39}
, {-52, -5, 70}
, {-32, 57, -11}
, {-32, -18, -70}
, {32, -13, 10}
, {13, -8, 72}
, {-74, -78, -60}
}
, {{61, -82, -70}
, {15, 55, -11}
, {38, -9, 47}
, {68, 14, 53}
, {-7, -7, -46}
, {-34, 27, 45}
, {-55, -59, -63}
, {-45, 66, -55}
, {82, -1, 74}
, {-53, -54, -75}
, {-18, -14, 15}
, {-63, 19, 15}
, {25, 13, 52}
, {-6, 14, 30}
, {42, -58, 44}
, {-57, 35, -20}
, {42, -58, -45}
, {30, -39, 37}
, {66, 63, 39}
, {-14, 34, 28}
, {-8, -1, -78}
, {-46, 16, -25}
, {69, -22, 24}
, {2, 28, -31}
, {9, -63, 39}
, {75, 66, -30}
, {27, -14, 73}
, {-61, -11, 45}
, {-40, -23, 29}
, {75, 67, 60}
, {-52, -34, -47}
, {-26, -3, 14}
}
, {{13, 26, -41}
, {-2, -61, 58}
, {-67, 50, -35}
, {44, -25, -11}
, {29, -67, -33}
, {-66, -10, 26}
, {-73, -8, 33}
, {10, -33, 43}
, {-39, 17, 40}
, {54, 15, -82}
, {37, -74, -83}
, {61, -35, -21}
, {-56, -58, 40}
, {48, 14, -71}
, {6, 64, 6}
, {17, 59, -41}
, {0, 11, -12}
, {-14, -68, 52}
, {-72, -10, 14}
, {-77, -53, -33}
, {-5, -46, 11}
, {79, 34, -46}
, {26, -43, -54}
, {-74, -51, -61}
, {-38, -11, 53}
, {24, -31, 24}
, {39, 69, 3}
, {27, -48, -55}
, {57, 23, -10}
, {-27, 37, -36}
, {-56, -41, 5}
, {-46, 47, 36}
}
, {{-7, -5, 31}
, {1, 61, 20}
, {38, 68, 73}
, {24, 30, 49}
, {91, -64, -28}
, {-43, -35, 37}
, {-28, 19, -20}
, {18, -30, -39}
, {6, 56, 48}
, {59, -3, 35}
, {-2, 49, 62}
, {27, -26, -27}
, {-81, -31, -29}
, {-63, -57, -18}
, {-30, 40, 46}
, {-55, -56, 13}
, {23, -53, -59}
, {5, -3, 15}
, {-3, 9, 41}
, {3, -69, 36}
, {-27, -28, -29}
, {9, 33, -91}
, {20, 26, 5}
, {30, 57, -33}
, {-27, -38, 46}
, {59, -17, 41}
, {-60, 50, -42}
, {10, -43, -7}
, {-4, 62, 38}
, {18, -72, 41}
, {-56, -11, 47}
, {8, -10, -59}
}
, {{-85, -16, -33}
, {30, 33, 38}
, {17, -6, 8}
, {15, 24, -42}
, {28, -62, -9}
, {-20, 90, -21}
, {-78, -20, -6}
, {19, -1, 38}
, {-2, -17, -102}
, {32, 84, -27}
, {-48, 39, -15}
, {6, 36, 70}
, {-74, 44, -97}
, {-4, 37, -9}
, {-59, -50, 14}
, {-57, -19, 72}
, {-11, 61, 21}
, {33, 46, 71}
, {-51, -31, -34}
, {-29, -45, 14}
, {-2, 71, -29}
, {-15, 0, 85}
, {-74, 66, -48}
, {-51, -75, 12}
, {-18, 92, 2}
, {-53, -50, 6}
, {-18, 1, 40}
, {-58, 23, -36}
, {-23, 57, -36}
, {44, 11, 78}
, {68, -60, 85}
, {-58, 40, -9}
}
, {{-30, -47, -46}
, {-17, 33, 41}
, {36, -45, 34}
, {-19, -65, -34}
, {-33, -43, 61}
, {2, 71, -53}
, {-32, -12, 52}
, {4, -2, -29}
, {71, -55, 76}
, {30, -15, 39}
, {26, -40, 9}
, {-78, 17, -35}
, {38, 35, 35}
, {-74, -52, 42}
, {-54, 54, 9}
, {-17, -36, -60}
, {-51, -25, 65}
, {-20, 48, 55}
, {57, 45, 75}
, {13, 56, 35}
, {-39, 44, 60}
, {79, -44, -53}
, {0, -68, 21}
, {43, -59, -38}
, {36, -65, -25}
, {-45, -15, 79}
, {54, 25, -11}
, {-12, 15, -37}
, {-46, -20, 22}
, {55, 48, 53}
, {38, -2, -64}
, {-40, 28, 14}
}
, {{-8, 29, 12}
, {82, 54, 104}
, {68, -65, 7}
, {61, 38, -59}
, {79, 8, 80}
, {-2, 9, 9}
, {71, -26, 42}
, {45, -47, -47}
, {10, 26, 72}
, {73, -62, -7}
, {2, -43, -89}
, {16, 51, -47}
, {7, -5, -1}
, {54, 12, -1}
, {35, 14, -66}
, {-21, -68, 53}
, {-81, 69, 25}
, {-23, 11, -61}
, {48, 47, 63}
, {26, 11, -65}
, {-51, -93, -73}
, {-17, 14, 68}
, {-33, 52, -56}
, {-20, -24, -83}
, {40, -8, -12}
, {44, -22, 27}
, {-8, 71, -23}
, {74, -34, 61}
, {-68, -63, -42}
, {-12, 17, -39}
, {68, -50, -7}
, {-45, -10, 25}
}
, {{-13, 34, 5}
, {-22, -49, 46}
, {84, 62, -63}
, {60, 64, 25}
, {29, -54, -18}
, {-51, -7, 71}
, {-71, 15, -23}
, {-48, -57, 37}
, {7, 20, -13}
, {-77, 31, 26}
, {67, 55, -61}
, {-67, 21, 93}
, {-87, -10, -17}
, {17, -56, -66}
, {1, 14, 78}
, {56, 12, 56}
, {-62, -13, -2}
, {-71, -20, -86}
, {50, -30, -5}
, {-61, -4, 29}
, {-9, -2, 43}
, {106, -38, 111}
, {-10, 61, -55}
, {-39, -5, 33}
, {18, -24, 58}
, {-42, -21, 23}
, {34, 13, 5}
, {4, 20, 39}
, {-65, -98, -35}
, {69, 46, -57}
, {78, -24, -49}
, {-61, 4, 31}
}
, {{-29, -72, 62}
, {-47, -79, 14}
, {-21, 28, 20}
, {-16, 65, 67}
, {32, -53, -79}
, {25, -24, 67}
, {19, 0, 37}
, {-84, 40, 17}
, {-26, -1, -19}
, {-33, -37, 29}
, {-57, 93, 17}
, {23, 36, 42}
, {10, -47, 53}
, {43, -37, -27}
, {-11, -7, 27}
, {-38, 36, -62}
, {79, -35, 55}
, {-40, 28, -16}
, {58, 10, -71}
, {62, 21, -33}
, {-25, 141, 162}
, {86, 123, 90}
, {-50, -39, 0}
, {21, 9, 38}
, {64, -33, -4}
, {-6, 102, 26}
, {-66, 23, 24}
, {-67, 12, 36}
, {7, -43, 81}
, {-6, 5, 17}
, {-43, -10, -72}
, {-19, -60, 29}
}
, {{-22, 2, -51}
, {-86, -29, 66}
, {53, -80, -50}
, {23, -18, -64}
, {-8, 23, -79}
, {0, -4, -50}
, {-15, -10, 7}
, {-75, 33, 31}
, {-41, 29, 8}
, {49, 55, 36}
, {12, 15, -63}
, {-34, 74, 41}
, {-6, 12, 17}
, {-73, -67, 49}
, {16, -51, -40}
, {-1, -68, 66}
, {-4, 60, -19}
, {-27, -23, 54}
, {-50, -16, 5}
, {4, -53, 65}
, {121, 72, 17}
, {61, 0, 45}
, {61, 3, -55}
, {24, 20, -71}
, {-15, 77, 29}
, {15, 41, -1}
, {-35, -34, -72}
, {-105, -75, -15}
, {-64, 72, 80}
, {-42, -24, -9}
, {72, -63, 13}
, {-17, 62, 24}
}
, {{5, 42, 45}
, {-49, -11, 60}
, {-43, 11, 42}
, {-6, -35, -4}
, {-25, -78, -34}
, {-25, -54, 53}
, {14, 27, -30}
, {-2, 33, -13}
, {-63, -24, -21}
, {33, 37, 29}
, {-26, -38, 22}
, {-35, -8, 23}
, {41, 52, -17}
, {60, -79, 48}
, {37, -68, -74}
, {-1, 62, -50}
, {-43, 21, -53}
, {-89, 67, 38}
, {-25, 57, -24}
, {-24, -23, 43}
, {6, 28, -20}
, {-72, -12, 27}
, {15, -13, -39}
, {-6, -74, -20}
, {-42, 73, -13}
, {-38, -53, 7}
, {-60, -8, 2}
, {8, 28, -16}
, {-50, -27, -14}
, {-74, -29, -20}
, {54, 22, 17}
, {35, -6, 51}
}
, {{10, 10, 25}
, {66, 35, 20}
, {-56, -25, -76}
, {-84, -69, 28}
, {29, 25, -54}
, {18, -57, 30}
, {80, 73, 16}
, {42, -86, -55}
, {81, -92, -24}
, {81, -36, 1}
, {-11, 28, -18}
, {-27, -58, -23}
, {34, -37, 82}
, {-44, 83, 0}
, {-31, -28, 28}
, {-56, 51, 60}
, {-30, 0, -24}
, {4, -5, 56}
, {38, -1, -16}
, {-23, -68, 62}
, {128, 38, 3}
, {41, 63, 41}
, {-89, -60, -4}
, {53, -34, -21}
, {15, -56, -33}
, {58, -60, 50}
, {44, 33, 28}
, {57, 58, 93}
, {-36, -15, 56}
, {26, 5, -11}
, {-28, -34, 40}
, {5, -11, -86}
}
, {{-96, -44, 36}
, {-25, 2, -59}
, {46, 68, 4}
, {77, -70, -62}
, {38, 69, -70}
, {62, -68, -35}
, {47, -47, -27}
, {25, -2, 33}
, {5, 37, -45}
, {48, 29, -5}
, {-24, 32, -68}
, {-38, 37, -45}
, {11, 53, -36}
, {-12, 26, -52}
, {-46, 63, -3}
, {-6, 59, 49}
, {11, 38, 42}
, {-23, -68, -37}
, {-68, -51, -1}
, {-35, -28, -46}
, {36, 40, -4}
, {56, 10, -32}
, {63, -32, -28}
, {-24, 44, -8}
, {-58, -38, 1}
, {4, 75, 9}
, {54, -49, -73}
, {38, -34, 1}
, {40, 34, 30}
, {-2, -76, -61}
, {49, 36, -25}
, {-23, 82, -19}
}
, {{1, -32, -45}
, {-52, -19, -21}
, {-44, 35, -21}
, {-3, 4, -12}
, {-53, -40, 14}
, {-25, 20, 20}
, {13, -79, 6}
, {-59, 35, -42}
, {88, 3, 34}
, {-31, -55, -18}
, {65, -65, 55}
, {74, -57, 26}
, {5, 107, 8}
, {84, 1, 10}
, {77, 39, 32}
, {-52, 62, -18}
, {54, 15, -72}
, {17, -59, -76}
, {83, 30, 20}
, {6, 71, -8}
, {-27, 50, -74}
, {-16, -64, 64}
, {-31, -36, -50}
, {-16, -34, 3}
, {43, -46, -31}
, {88, 30, -11}
, {-53, 45, 6}
, {8, -36, 63}
, {-9, 83, -11}
, {-33, 67, -46}
, {-54, 44, -70}
, {-68, -21, 24}
}
, {{-40, 51, 76}
, {41, -21, -12}
, {-5, 49, -64}
, {-58, -25, -63}
, {53, -25, -78}
, {-39, 13, 18}
, {-25, 56, -21}
, {-48, 86, 13}
, {3, 22, 39}
, {-72, -49, -24}
, {30, -52, -48}
, {11, 11, -70}
, {10, -3, 69}
, {56, -4, 36}
, {-63, 24, -51}
, {-1, 3, -32}
, {-31, -80, 40}
, {65, 25, 56}
, {35, -57, -1}
, {6, 64, -5}
, {5, 4, 80}
, {-19, 91, 46}
, {-28, -57, 23}
, {-56, 40, 74}
, {22, 55, -44}
, {104, -29, 40}
, {7, 11, 17}
, {-43, 61, 7}
, {-67, -42, 52}
, {81, 24, -30}
, {27, 48, -73}
, {51, -22, 7}
}
, {{21, 0, -48}
, {89, 40, 49}
, {-49, -29, 20}
, {1, 33, -50}
, {-44, 106, -12}
, {69, 88, -49}
, {12, 85, 72}
, {-42, 72, 2}
, {44, 14, -42}
, {-17, -14, -28}
, {-60, 65, 65}
, {80, -11, -35}
, {31, -19, -32}
, {-59, -53, -60}
, {35, -59, 36}
, {56, -23, -38}
, {8, 70, -52}
, {-3, -48, 24}
, {-14, -2, -28}
, {-36, -30, 85}
, {-84, 21, 39}
, {65, 30, -31}
, {33, 53, -34}
, {8, -65, 100}
, {-54, -34, 32}
, {52, 70, -33}
, {69, 69, -4}
, {-54, 62, -9}
, {12, -65, -59}
, {-29, 45, 60}
, {-11, 77, 38}
, {-35, -34, -45}
}
, {{-14, 26, 22}
, {-57, -67, -87}
, {65, -46, 11}
, {-36, -91, -84}
, {-28, 1, -30}
, {67, 8, -70}
, {-41, -31, 0}
, {49, 3, -23}
, {50, 10, 48}
, {-14, 31, -40}
, {45, 6, -15}
, {47, 30, -28}
, {59, 74, -7}
, {-100, 50, -32}
, {51, 41, -43}
, {-9, -9, -11}
, {50, 56, 30}
, {53, 53, 30}
, {55, 23, 47}
, {46, 64, 31}
, {47, -9, 0}
, {-36, 88, 150}
, {-23, 41, -55}
, {-38, 28, -83}
, {37, 72, 72}
, {59, 24, -22}
, {69, -4, -11}
, {0, -53, -41}
, {20, -4, 64}
, {6, 24, 60}
, {-57, 10, -14}
, {67, 10, 34}
}
, {{-45, -69, -21}
, {30, -47, 37}
, {16, 51, -18}
, {19, 21, 0}
, {27, -17, -51}
, {-14, -78, -1}
, {51, 45, 9}
, {53, -9, 56}
, {28, 55, 10}
, {-46, -74, 24}
, {14, 22, 15}
, {-44, -24, -19}
, {-25, 3, 63}
, {-50, -61, -28}
, {-8, -62, 40}
, {-15, -38, -27}
, {52, 30, -80}
, {27, -41, 57}
, {57, 32, 35}
, {-19, -25, 28}
, {41, 58, -75}
, {28, 62, 39}
, {-48, 19, 63}
, {-20, -78, -76}
, {4, -57, 12}
, {-74, -58, 39}
, {-67, 21, -31}
, {-55, 51, 46}
, {-24, -61, 69}
, {-23, -70, 41}
, {26, -9, -71}
, {15, 44, 29}
}
, {{-12, 81, -13}
, {34, -25, -69}
, {-59, 15, -63}
, {9, -19, 16}
, {-16, 19, -4}
, {-59, -68, -97}
, {-50, -26, 48}
, {23, 13, 57}
, {59, 5, 31}
, {63, 63, 42}
, {42, 37, 80}
, {-52, 70, -30}
, {-9, -89, -9}
, {86, 7, -55}
, {20, 69, -8}
, {-52, 16, -61}
, {-56, -8, -34}
, {29, 1, -23}
, {-29, -30, 32}
, {-2, -26, -58}
, {-61, 46, -52}
, {-53, 21, -9}
, {-68, 14, 47}
, {-57, 71, -85}
, {66, 2, 20}
, {42, 0, 44}
, {-59, 70, 73}
, {79, 62, 6}
, {-9, 4, 35}
, {-66, 43, -55}
, {1, -70, -35}
, {63, -34, 0}
}
, {{-74, -47, -15}
, {-28, 25, -5}
, {42, 23, 60}
, {39, 19, 73}
, {64, -36, -3}
, {22, 39, -72}
, {-14, 44, 38}
, {61, 32, -79}
, {-76, 9, -47}
, {-26, 45, -49}
, {82, 66, 6}
, {-15, -3, -55}
, {-13, -48, 10}
, {-55, -42, 63}
, {18, -19, -50}
, {15, 53, 75}
, {-51, -28, -12}
, {-35, 36, -34}
, {-70, -67, 17}
, {-66, 14, -75}
, {-62, 54, -23}
, {-21, 11, -71}
, {6, -38, 13}
, {39, -36, 11}
, {63, -18, -17}
, {54, 35, -52}
, {-45, 47, -32}
, {-43, -7, 58}
, {-18, 45, 0}
, {63, -47, 26}
, {2, -41, 76}
, {66, -20, 7}
}
, {{18, 96, 7}
, {-38, -3, -25}
, {-5, 51, 31}
, {-47, 31, 0}
, {48, 38, -49}
, {52, -29, -1}
, {-26, 20, 0}
, {-61, -20, 81}
, {-56, 2, -3}
, {-37, 18, -33}
, {30, -35, -46}
, {-37, 14, 27}
, {31, 36, 58}
, {-43, 24, -45}
, {16, 58, 52}
, {43, 60, -61}
, {82, 11, 52}
, {-44, 2, -34}
, {-19, 37, -47}
, {-12, 55, 71}
, {-22, -64, -14}
, {23, 52, -53}
, {-16, -73, -14}
, {-46, 50, -21}
, {-53, 19, -58}
, {69, 18, -15}
, {40, 20, 43}
, {39, 29, -16}
, {-16, -38, 7}
, {4, 23, -20}
, {-54, -20, 5}
, {-51, 34, 46}
}
, {{30, -90, 52}
, {54, -12, -48}
, {-42, 64, -44}
, {-29, -53, -55}
, {-33, -2, 5}
, {34, 29, -9}
, {51, 44, -45}
, {58, -59, 5}
, {-10, 37, -67}
, {-3, 11, -14}
, {48, -40, 59}
, {41, -39, -53}
, {-56, -19, -48}
, {-14, -60, 69}
, {19, -32, -19}
, {-30, -21, -2}
, {-43, 33, 1}
, {56, 14, -42}
, {52, 79, -60}
, {-28, -34, 6}
, {28, -81, -16}
, {134, 41, 24}
, {-59, 67, 34}
, {-31, -48, 8}
, {3, 85, 52}
, {-22, 27, 17}
, {0, 34, -53}
, {-71, -21, -27}
, {35, 63, 80}
, {17, 8, -57}
, {42, 76, -46}
, {72, -14, -28}
}
, {{63, -13, -20}
, {-91, -62, 26}
, {46, 1, 8}
, {95, -18, 39}
, {-7, -75, -41}
, {-8, -14, -61}
, {-21, -62, -84}
, {-69, 8, -67}
, {-43, 57, 64}
, {-72, -53, -45}
, {21, -49, -30}
, {-71, 34, 85}
, {65, -52, 84}
, {35, -27, 48}
, {-40, 0, 28}
, {46, -10, -9}
, {-53, -68, -18}
, {-48, -34, 49}
, {-23, -70, -45}
, {-25, 42, 49}
, {-51, -28, -17}
, {16, -1, 54}
, {70, 63, -39}
, {46, -9, 27}
, {-31, -5, 16}
, {70, -1, 27}
, {43, 53, -2}
, {-4, 78, 56}
, {6, -75, 87}
, {-13, -31, -28}
, {80, -33, 45}
, {62, -56, -41}
}
, {{-63, -70, -4}
, {0, -78, 50}
, {-31, -52, -4}
, {34, 47, 47}
, {53, -4, 62}
, {-24, -29, 26}
, {-51, 13, -60}
, {0, 1, -67}
, {9, -44, 44}
, {-11, 15, 1}
, {-64, 12, 71}
, {-54, -23, -6}
, {49, -25, 57}
, {56, -30, 56}
, {42, 71, 0}
, {-40, 55, -11}
, {-38, -6, 67}
, {4, -59, -21}
, {-52, -46, 73}
, {17, -46, 21}
, {-76, -11, 0}
, {32, 19, -16}
, {11, -28, -50}
, {57, 22, 50}
, {-3, 34, -37}
, {70, -45, 62}
, {57, 24, 64}
, {-42, 59, -35}
, {13, 30, 58}
, {67, 42, 4}
, {-69, 40, 65}
, {23, 39, -3}
}
, {{-49, -10, 33}
, {-19, 22, 63}
, {-29, 11, -96}
, {-21, -24, -66}
, {78, 5, -13}
, {-89, -59, 61}
, {-74, 41, 17}
, {-77, -29, 9}
, {-3, -36, -17}
, {-67, -78, -11}
, {63, 44, -66}
, {13, -37, -29}
, {-17, 33, -61}
, {1, -66, -29}
, {-42, -61, 5}
, {-33, 10, -42}
, {51, -52, 61}
, {47, 44, 9}
, {-28, -77, -63}
, {34, 55, -59}
, {-26, -74, -58}
, {96, 57, -70}
, {-63, 20, -48}
, {-34, 32, 20}
, {1, 18, 66}
, {35, -61, -4}
, {71, 57, -36}
, {-33, -29, -59}
, {24, 32, -18}
, {-17, 63, 48}
, {61, 69, 23}
, {-62, -78, -62}
}
, {{44, -8, 21}
, {-2, 18, 53}
, {-33, 32, 82}
, {-51, -32, -13}
, {12, -32, 33}
, {-72, 39, 53}
, {7, 42, -3}
, {24, 24, -13}
, {-71, -55, -80}
, {-27, 56, -54}
, {-27, 65, -55}
, {40, -26, -76}
, {-16, -21, -40}
, {18, 79, 62}
, {-59, -67, 44}
, {75, -9, 52}
, {59, -14, -28}
, {-1, -29, 29}
, {45, -67, -9}
, {2, -12, -7}
, {76, 29, -23}
, {132, 131, 66}
, {10, -35, 21}
, {-58, 37, 56}
, {40, -32, 47}
, {64, -72, 30}
, {34, -48, -4}
, {-40, -20, -65}
, {13, 71, 38}
, {-56, 1, 0}
, {-84, 37, -32}
, {-12, 29, -17}
}
, {{25, -37, 54}
, {14, -20, -72}
, {-57, 13, -58}
, {2, 57, 24}
, {57, -35, 26}
, {66, -63, 30}
, {0, -7, 51}
, {-9, 67, -27}
, {-28, -80, 38}
, {-2, -59, 32}
, {59, -48, 1}
, {-73, 42, -63}
, {-37, 29, 74}
, {63, -35, -69}
, {34, -18, 64}
, {20, 45, 41}
, {-55, 61, -33}
, {-65, -33, 39}
, {33, -71, -4}
, {28, -55, -38}
, {116, 0, 0}
, {-55, -82, -78}
, {10, 51, 29}
, {42, 58, 56}
, {-61, -74, 5}
, {-64, -49, 11}
, {67, -1, 55}
, {62, -13, 39}
, {43, 42, -61}
, {25, -72, 15}
, {13, 10, -78}
, {34, -27, 35}
}
, {{-41, -65, -31}
, {0, -33, 43}
, {-18, 39, 1}
, {31, -64, 15}
, {-59, -10, 21}
, {25, -68, -45}
, {-28, 72, 15}
, {50, -29, -8}
, {-48, -69, 7}
, {-55, 74, -67}
, {15, -71, -41}
, {74, 35, -73}
, {-6, -64, -25}
, {-59, 12, -28}
, {34, -2, 50}
, {-78, 6, 16}
, {-37, 63, -74}
, {-49, -58, 69}
, {12, -16, -61}
, {46, 60, 37}
, {-71, 23, -59}
, {-47, -19, -47}
, {-61, -71, -88}
, {-66, 46, 32}
, {-42, -17, -29}
, {-27, 37, -71}
, {67, 66, 46}
, {-16, 67, 57}
, {36, 21, 57}
, {-14, 13, -13}
, {-42, -88, 60}
, {54, 5, -12}
}
, {{-25, -57, -60}
, {17, -15, 22}
, {63, -92, 26}
, {-52, -27, -65}
, {56, 10, 30}
, {-12, 77, -72}
, {72, -51, 62}
, {-36, -29, -55}
, {-23, -50, 50}
, {-37, 36, -81}
, {-16, -57, -42}
, {13, -77, -54}
, {73, 49, -37}
, {-60, -51, 37}
, {47, 31, 64}
, {-9, -67, 23}
, {-20, -38, -60}
, {14, 29, 58}
, {75, 37, -41}
, {-9, -58, 10}
, {63, 17, -84}
, {-74, 16, 20}
, {-19, -29, 54}
, {-49, 33, 57}
, {0, -34, 34}
, {29, -34, 20}
, {-1, 1, 39}
, {6, -31, 27}
, {-13, -19, -49}
, {-1, -62, -72}
, {43, -81, 17}
, {67, 12, 21}
}
, {{-41, -14, 4}
, {36, -71, 10}
, {-59, -2, 68}
, {-35, 63, -2}
, {66, 21, -37}
, {10, 57, 28}
, {-7, -30, -15}
, {65, -5, -26}
, {42, 51, -80}
, {67, 21, -41}
, {68, 44, 59}
, {-42, 20, 0}
, {-94, 34, -52}
, {9, -56, -5}
, {40, -76, 42}
, {6, -60, -11}
, {-19, -18, -56}
, {-55, 7, -39}
, {71, 71, -45}
, {-12, -13, -42}
, {-76, -47, 3}
, {-18, -103, 45}
, {-25, -66, -73}
, {38, -29, -13}
, {-27, -32, 62}
, {13, 80, 35}
, {-62, -11, -46}
, {67, 1, -52}
, {-34, 88, 24}
, {-5, 76, 20}
, {66, 50, 18}
, {13, 50, 76}
}
, {{-6, -18, -17}
, {-3, 2, -42}
, {92, 95, -30}
, {51, 9, 30}
, {-74, -41, 9}
, {-63, 35, 54}
, {39, -57, -36}
, {1, 2, 46}
, {-61, 66, 45}
, {41, 31, -21}
, {-22, -1, 34}
, {78, 10, 0}
, {-39, -43, 48}
, {72, 23, -49}
, {34, 44, 68}
, {-44, 52, -60}
, {-15, -41, 24}
, {67, -9, 7}
, {11, 73, 37}
, {36, 27, -20}
, {-24, 34, 128}
, {177, -25, 104}
, {-9, -21, 4}
, {-50, 3, -53}
, {76, 95, 14}
, {104, 71, -25}
, {-16, 55, 48}
, {-19, -61, -61}
, {-30, -27, 28}
, {30, -19, 75}
, {70, 31, 67}
, {-65, -47, -49}
}
, {{52, 67, -77}
, {-82, -19, 11}
, {-48, -56, -36}
, {48, -40, 5}
, {-60, 65, 44}
, {24, 26, 57}
, {-19, 0, 54}
, {0, -62, 59}
, {12, -71, 45}
, {16, 31, 49}
, {81, -57, -58}
, {-43, 9, -10}
, {20, 32, -27}
, {-63, -23, -17}
, {34, 2, 38}
, {-21, 30, 38}
, {37, -15, 2}
, {1, 35, 33}
, {14, -43, 28}
, {-46, 46, 11}
, {91, 2, 110}
, {39, 95, 153}
, {32, -84, -49}
, {-108, -9, -9}
, {92, 16, 53}
, {-35, -44, -39}
, {19, 72, -5}
, {-86, 45, -49}
, {58, 57, 13}
, {85, 9, -68}
, {-68, -42, -63}
, {46, -25, 51}
}
, {{62, -43, -48}
, {24, -39, -14}
, {-4, -68, 35}
, {-11, -45, 36}
, {46, 41, 25}
, {-30, -35, 53}
, {20, 37, 49}
, {29, -35, 30}
, {60, 34, 59}
, {57, -3, 58}
, {-49, 38, -78}
, {30, -73, 9}
, {56, 47, -7}
, {-21, 10, -50}
, {-27, -38, 7}
, {31, 63, 15}
, {-41, -26, -2}
, {-58, -19, -82}
, {17, -69, 28}
, {-72, 71, -45}
, {63, 23, 21}
, {-14, 81, -2}
, {24, 27, 44}
, {-50, 5, -35}
, {36, -71, -42}
, {-50, -75, -32}
, {37, -26, 14}
, {-68, -27, -77}
, {38, 59, -48}
, {-23, 49, 4}
, {-16, 10, -41}
, {58, 22, 2}
}
, {{-2, 8, 27}
, {-16, 67, 30}
, {-22, -79, 63}
, {-54, -6, -25}
, {32, -44, -20}
, {29, -28, -35}
, {-34, -17, -18}
, {-24, 32, -46}
, {35, 23, -25}
, {52, 76, -5}
, {70, -30, -43}
, {0, -33, 26}
, {52, 52, -31}
, {-82, 59, 50}
, {14, -38, 46}
, {-4, -4, 32}
, {-1, 56, 19}
, {-30, -65, -54}
, {26, -20, -30}
, {72, 50, 18}
, {69, 45, 88}
, {117, -19, -75}
, {66, -66, 7}
, {0, -42, -67}
, {-12, 40, -8}
, {-80, -58, -33}
, {-51, -53, -18}
, {-12, 3, -11}
, {-27, 71, -12}
, {-78, -59, 55}
, {46, -58, -13}
, {77, 28, -84}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE