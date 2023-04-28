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


const int16_t conv1d_50_bias[CONV_FILTERS] = {19, 2, -5, -9, 15, 10, 15, -1, 19, 0, 21, 15, 10, -11, 15, -2, 12, -5, 2, 6, 27, 1, -12, -6, -2, -5, 30, -1, -2, 9, 18, 12, -2, 12, 0, 17, 4, -1, -14, -16, 16, 4, 8, -12, -8, 7, -17, -5, 11, 11, 25, -10, -11, 34, 1, -15, 0, 23, -2, -14, -7, -1, 6, 0}
;

const int16_t conv1d_50_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-11, -54, -57}
, {-50, 37, -55}
, {64, 28, 87}
, {5, 22, -56}
, {-38, -72, -15}
, {44, -58, 48}
, {7, 22, -46}
, {-21, -29, -58}
, {-54, 15, -44}
, {17, 22, -65}
, {28, -9, 11}
, {33, -57, -23}
, {24, 16, 33}
, {-48, -24, 50}
, {-60, -43, 8}
, {48, 45, -58}
, {46, -4, 14}
, {53, 70, 17}
, {-7, -7, 58}
, {24, 18, 62}
, {66, 55, 50}
, {82, 21, -71}
, {24, -65, -4}
, {57, -8, -63}
, {-12, 0, 56}
, {-31, 83, 16}
, {-53, 37, 10}
, {-67, -23, 12}
, {9, -27, 31}
, {-89, -45, -11}
, {-4, 41, 55}
, {-6, 32, 94}
}
, {{-45, 74, 48}
, {20, 29, -44}
, {-37, -72, -5}
, {54, -46, 3}
, {-64, 30, -52}
, {24, -26, 21}
, {48, -22, 52}
, {-27, -25, -37}
, {49, 28, -46}
, {7, -5, -8}
, {19, 21, 33}
, {-34, -77, -43}
, {-38, 0, 37}
, {44, 11, -61}
, {-55, -67, -64}
, {59, -19, 13}
, {73, 17, -14}
, {-61, -5, -39}
, {-19, 92, 48}
, {1, 47, 52}
, {23, -1, 71}
, {-32, -28, 18}
, {68, 39, -69}
, {56, 44, 23}
, {4, -11, -60}
, {-5, -58, -64}
, {87, -15, -38}
, {-57, 13, 22}
, {-63, -64, 52}
, {71, 49, 83}
, {-63, 35, 0}
, {-13, 61, 52}
}
, {{14, 65, -21}
, {-19, -59, 45}
, {31, 46, -13}
, {-70, 19, -7}
, {19, -44, 3}
, {13, 51, 14}
, {4, 26, -35}
, {44, 30, -13}
, {-82, 44, 46}
, {55, -47, -63}
, {40, -9, 27}
, {48, -25, 63}
, {29, 69, -38}
, {-68, -78, -3}
, {-43, -44, -51}
, {49, -42, -19}
, {-76, 41, -10}
, {-64, 48, 36}
, {-43, -81, 23}
, {-12, -7, -39}
, {-70, -13, -30}
, {41, -43, -50}
, {-21, 42, -45}
, {33, -27, 69}
, {-11, 55, 20}
, {-53, -46, 10}
, {30, -38, -48}
, {-48, -30, 19}
, {50, 40, 38}
, {-2, 42, 16}
, {65, 35, 9}
, {71, 29, -10}
}
, {{70, 6, 45}
, {75, -45, -23}
, {-18, -67, -21}
, {58, 28, 69}
, {41, 25, -71}
, {49, 38, 4}
, {-47, 86, -52}
, {-80, -24, -74}
, {41, 77, -33}
, {-49, -61, -38}
, {81, -54, 55}
, {-20, -17, 63}
, {-60, -47, -30}
, {-10, -47, -29}
, {-6, -58, 76}
, {65, 25, -9}
, {37, 52, 0}
, {71, 10, 32}
, {-29, -18, -24}
, {-25, 8, -23}
, {-2, -25, -40}
, {26, -28, 71}
, {12, -2, 76}
, {37, -37, -33}
, {23, -45, -15}
, {-5, 35, -75}
, {-62, 59, 32}
, {44, 47, 32}
, {-63, -20, -28}
, {36, -73, 49}
, {14, -76, -1}
, {2, -22, 18}
}
, {{-56, -52, -18}
, {40, -31, -69}
, {1, 13, -13}
, {-16, -15, -26}
, {-54, -27, 45}
, {89, 52, -9}
, {31, -68, -9}
, {-56, -57, 17}
, {-25, 57, 0}
, {-51, -54, -54}
, {44, -27, -41}
, {69, 38, 27}
, {64, -46, -13}
, {-3, 50, 49}
, {-15, -45, 1}
, {63, 0, 23}
, {-26, -9, -39}
, {43, 14, -30}
, {43, 37, -62}
, {6, 43, 74}
, {61, 29, -43}
, {-68, -6, -38}
, {-12, 16, -14}
, {-22, 56, -25}
, {-53, 38, 42}
, {-8, 36, -50}
, {-54, 50, 31}
, {4, 17, -37}
, {-38, 44, 43}
, {-25, -73, -21}
, {35, 14, -20}
, {-52, 53, 75}
}
, {{-41, 29, 66}
, {28, -23, -15}
, {61, -42, 56}
, {10, -8, -61}
, {-68, -86, -20}
, {9, 67, 59}
, {9, 10, -7}
, {51, -33, 17}
, {-55, -66, 36}
, {5, -8, -18}
, {34, -58, 1}
, {-8, 69, -29}
, {29, 1, 47}
, {92, 73, -1}
, {15, -68, 56}
, {-52, -23, 61}
, {51, -48, 58}
, {36, -39, -37}
, {-42, 68, 82}
, {-60, 29, 48}
, {-35, 3, 60}
, {75, -42, 10}
, {16, -71, -49}
, {-2, 46, 26}
, {-76, -19, 7}
, {46, -24, 55}
, {-11, -90, -69}
, {66, 59, 1}
, {-28, 15, 27}
, {-52, 13, -5}
, {-11, 38, -23}
, {-49, -33, 59}
}
, {{48, -4, -40}
, {3, -17, 50}
, {72, 14, 80}
, {-66, -12, -46}
, {-56, 39, -97}
, {-1, 52, -7}
, {67, -25, 8}
, {74, 48, 60}
, {-20, 39, -54}
, {57, 72, -2}
, {-16, 25, 48}
, {-6, 58, 69}
, {0, -49, 17}
, {-11, 41, 25}
, {-39, 2, 51}
, {-37, 84, -48}
, {-12, -36, -57}
, {-21, 51, -21}
, {1, 29, -39}
, {-57, -48, -20}
, {13, 18, -57}
, {86, -5, 65}
, {-46, -60, -99}
, {53, -24, 70}
, {-41, 59, -72}
, {24, 61, 31}
, {-77, 2, 60}
, {-31, 45, -58}
, {75, -32, -65}
, {-63, 23, 11}
, {-49, -48, -58}
, {-20, 47, 66}
}
, {{0, 32, -58}
, {2, 29, 30}
, {-55, 57, 29}
, {-33, 33, 47}
, {-1, -25, -35}
, {16, 3, 42}
, {-63, 2, 47}
, {-50, 32, -16}
, {16, -1, -55}
, {-44, 53, 66}
, {55, 8, 37}
, {27, 39, -31}
, {-59, -27, 11}
, {-19, -27, -15}
, {4, 72, 57}
, {-30, 50, 45}
, {-21, 65, -2}
, {70, 44, 10}
, {-20, -33, -57}
, {-58, -11, 10}
, {-6, -61, 15}
, {72, -1, -22}
, {-28, 62, 5}
, {74, 2, 26}
, {5, -37, -27}
, {49, -67, 62}
, {30, -33, 30}
, {51, -34, 67}
, {21, -64, -27}
, {71, -58, -8}
, {57, 20, -63}
, {-3, -52, -74}
}
, {{-17, 1, 58}
, {-70, -50, 30}
, {-77, -77, -49}
, {-23, -46, 74}
, {61, 24, -17}
, {-56, 4, 10}
, {41, 12, -44}
, {21, 55, -24}
, {59, -39, -63}
, {-37, -60, -64}
, {36, -10, -83}
, {55, -30, 15}
, {45, -48, 32}
, {-51, 4, -26}
, {10, -25, 7}
, {-17, -65, -2}
, {-60, 36, -47}
, {50, -67, -61}
, {43, 56, -56}
, {-12, -49, 52}
, {-26, 1, -27}
, {16, -73, -38}
, {72, -5, 21}
, {72, -8, 18}
, {-25, 41, -82}
, {50, 40, -88}
, {34, -34, -66}
, {-42, 59, -36}
, {45, 38, 71}
, {-31, 54, -22}
, {-25, 69, 60}
, {-45, -54, 46}
}
, {{47, 7, -1}
, {53, -26, -22}
, {1, 1, -42}
, {43, -1, -47}
, {-54, -71, -16}
, {-32, -55, 22}
, {44, 2, -66}
, {12, -51, 80}
, {-22, -31, -17}
, {42, -13, -8}
, {-39, 37, -19}
, {22, 49, 72}
, {42, 10, -41}
, {96, 31, 23}
, {-20, -67, 67}
, {42, 16, -8}
, {1, -75, 2}
, {-13, -20, -61}
, {-42, 6, 65}
, {-59, 42, 69}
, {-64, 53, 65}
, {52, -31, 50}
, {-33, -3, 39}
, {63, -33, -28}
, {52, 47, -44}
, {56, 53, 47}
, {-7, 4, -25}
, {-33, -35, 79}
, {-28, 8, -42}
, {93, -52, -4}
, {31, 43, 54}
, {-45, 0, -95}
}
, {{39, -5, 6}
, {78, -58, 70}
, {-23, -46, 89}
, {81, 62, 42}
, {-28, -30, 12}
, {3, -57, 61}
, {-11, 31, -21}
, {45, -20, -38}
, {8, -10, -35}
, {12, 56, 70}
, {64, -70, 4}
, {1, -60, -26}
, {27, -48, 67}
, {52, 102, -9}
, {-66, -18, -43}
, {49, 59, 8}
, {10, 31, -52}
, {-42, 32, 63}
, {-6, 30, -19}
, {10, 22, -52}
, {-52, -6, 80}
, {79, -70, -29}
, {-56, -86, 47}
, {-17, -51, 64}
, {27, -14, -66}
, {-19, 50, 41}
, {0, 45, -42}
, {41, 5, 52}
, {80, -4, -18}
, {-36, 40, -26}
, {-55, -25, -23}
, {14, 0, -36}
}
, {{-25, 41, 0}
, {69, -17, -14}
, {14, 67, 23}
, {74, 20, -64}
, {-5, -57, -48}
, {-48, 16, 30}
, {46, -33, 74}
, {8, -58, 26}
, {-38, 42, -48}
, {-40, -52, -43}
, {-19, 4, 21}
, {-32, -50, -11}
, {-62, 62, -47}
, {3, 55, 22}
, {25, 12, -59}
, {-22, 51, -25}
, {-18, -66, 11}
, {-59, 27, -29}
, {-59, -19, 18}
, {19, 21, 14}
, {34, 41, -61}
, {42, 39, -50}
, {67, 58, 6}
, {52, -2, -61}
, {83, -9, -2}
, {69, 78, 50}
, {65, 71, -66}
, {-15, 56, 63}
, {-6, -55, -66}
, {-38, 11, -87}
, {-3, -14, -55}
, {10, -16, -12}
}
, {{18, 40, 52}
, {-37, 43, 75}
, {-63, 19, 72}
, {22, 31, -32}
, {-82, -80, -16}
, {48, 4, -73}
, {29, -7, 15}
, {45, 11, 27}
, {-8, -11, -40}
, {21, 22, -22}
, {57, 16, 79}
, {63, -65, 41}
, {-11, 10, 40}
, {41, 71, -9}
, {-21, 42, 25}
, {3, 34, -15}
, {71, -20, 53}
, {6, -50, -58}
, {-83, 30, 63}
, {-45, -62, -5}
, {12, 56, 63}
, {-23, -31, -21}
, {-30, -30, 0}
, {-8, 47, -13}
, {-67, -36, 64}
, {11, -46, -19}
, {46, -25, 62}
, {1, -30, -34}
, {52, -61, -67}
, {6, 27, 65}
, {12, -69, -47}
, {-44, -25, -39}
}
, {{24, -64, -60}
, {-76, 44, -32}
, {85, 55, 58}
, {40, 23, -28}
, {-14, 76, 55}
, {86, -20, 45}
, {-21, -60, -76}
, {42, 48, -38}
, {-36, -75, -8}
, {-3, 45, 33}
, {50, 57, 11}
, {-33, 60, -34}
, {34, -29, 13}
, {0, -17, -56}
, {19, 60, -17}
, {-34, 40, 18}
, {23, 22, 18}
, {-6, 66, 0}
, {26, -19, 0}
, {35, 8, 76}
, {60, 0, -11}
, {-57, -46, 30}
, {26, -8, -55}
, {0, -67, 57}
, {0, -80, -14}
, {0, 51, -8}
, {-25, -70, -24}
, {-65, 32, -15}
, {16, -5, 11}
, {17, -9, 35}
, {21, 43, -41}
, {47, 11, 40}
}
, {{71, 39, 5}
, {5, -39, 19}
, {-71, -39, -41}
, {28, -11, 41}
, {-35, 5, 44}
, {-80, -49, 69}
, {40, 42, 18}
, {32, -28, -17}
, {-4, 79, -42}
, {-40, -61, 56}
, {-69, -68, 0}
, {51, -47, -46}
, {35, 57, 27}
, {2, 101, 17}
, {-74, 5, 0}
, {40, -63, 42}
, {50, -46, 43}
, {-29, 43, 34}
, {32, 2, 18}
, {-31, -42, 76}
, {32, 28, 41}
, {45, 11, -7}
, {-16, 35, 22}
, {68, -56, -18}
, {51, -49, -34}
, {-26, -70, -6}
, {-80, 60, 53}
, {-6, 26, 3}
, {0, -3, 49}
, {-67, -15, 86}
, {-21, 3, 0}
, {-61, -42, 5}
}
, {{74, 6, 30}
, {-17, 18, -29}
, {70, 42, -44}
, {-80, 31, -45}
, {21, 76, 32}
, {-45, 64, -30}
, {4, 0, -73}
, {72, -71, -16}
, {-3, -67, -48}
, {-76, 5, -1}
, {8, -15, 31}
, {-44, -28, 61}
, {0, 3, -46}
, {37, 1, -47}
, {17, 49, -28}
, {16, 79, 21}
, {-65, -88, -96}
, {-23, -24, 19}
, {68, 9, -30}
, {-33, -30, 54}
, {79, 9, -23}
, {-66, 71, 19}
, {17, 5, 23}
, {-72, 56, -37}
, {0, -80, 43}
, {86, 33, 5}
, {9, -39, 38}
, {-9, -71, 38}
, {51, 33, 36}
, {-36, 37, -83}
, {-65, -29, 22}
, {17, 52, 58}
}
, {{70, -56, 61}
, {14, 10, 63}
, {-26, 33, 28}
, {-57, 44, 48}
, {67, 51, -50}
, {-100, 47, -75}
, {41, 8, 24}
, {-7, 52, 78}
, {-63, -6, -10}
, {22, 52, -19}
, {-53, -81, 7}
, {3, 41, 2}
, {-23, -13, 63}
, {58, 45, 55}
, {-67, 45, 30}
, {69, -2, -18}
, {-90, 66, 21}
, {-42, -51, -20}
, {82, -13, -43}
, {-7, 4, 71}
, {79, 83, 8}
, {11, -66, 13}
, {6, 38, -34}
, {-64, -28, 58}
, {-47, -2, 23}
, {53, -42, 64}
, {0, -33, 55}
, {-17, -24, -14}
, {21, -17, 7}
, {-36, 39, -24}
, {-44, 29, 10}
, {53, -58, -88}
}
, {{31, -71, 18}
, {-60, 48, -4}
, {-29, -3, 45}
, {-22, 49, -16}
, {-58, -21, 33}
, {-23, -34, 40}
, {11, 43, -64}
, {63, -47, 53}
, {56, 32, 12}
, {-9, -37, -54}
, {73, 68, 46}
, {-2, -42, -58}
, {-23, 20, 52}
, {-56, 48, -1}
, {39, 74, 31}
, {-15, -22, 13}
, {11, -32, 37}
, {0, 75, -17}
, {-70, 8, 72}
, {-59, 46, -52}
, {72, -59, -68}
, {82, -60, 26}
, {50, 45, -31}
, {-49, 30, -38}
, {74, 46, 12}
, {-59, -50, -30}
, {-49, -35, -17}
, {-63, -10, -44}
, {53, -31, -7}
, {28, 7, 9}
, {31, 6, 65}
, {-65, -70, 41}
}
, {{-58, -11, -33}
, {2, -45, -17}
, {-16, 41, -72}
, {54, -59, -3}
, {0, 43, 31}
, {32, -29, -93}
, {-4, 61, 55}
, {49, 41, 33}
, {-34, -7, 18}
, {28, -69, 22}
, {7, 16, 57}
, {-37, 33, 12}
, {34, -16, -46}
, {-47, 12, -23}
, {-10, -36, -54}
, {-55, 11, -7}
, {78, 0, 16}
, {39, -22, 80}
, {30, 60, 20}
, {63, 63, -35}
, {-64, 22, 50}
, {51, 14, -16}
, {5, 8, 14}
, {58, -20, 73}
, {18, 36, -59}
, {-81, -69, -21}
, {16, 46, -47}
, {69, 24, -21}
, {-20, -8, -19}
, {-29, 60, 74}
, {42, 36, -17}
, {-30, -17, -84}
}
, {{22, -11, 50}
, {69, 9, -39}
, {28, -29, -19}
, {9, 45, -79}
, {34, -70, -90}
, {-6, -73, 43}
, {-38, -12, 9}
, {-12, 6, 82}
, {6, -39, -41}
, {-39, 39, -45}
, {35, -9, 44}
, {-40, -3, 55}
, {-33, -3, 11}
, {84, -42, 89}
, {16, -44, -28}
, {84, 50, 45}
, {-52, -4, -27}
, {20, 67, 65}
, {13, -13, 38}
, {20, -48, -22}
, {14, -5, -4}
, {54, -35, 46}
, {50, -44, 19}
, {0, 61, 34}
, {64, 1, -73}
, {-42, -63, -52}
, {63, -24, -9}
, {-38, 55, 45}
, {-21, -2, -26}
, {-31, 22, -45}
, {-14, -1, 13}
, {36, -50, 20}
}
, {{6, 62, 46}
, {-14, -34, -35}
, {34, 28, 28}
, {-9, 15, -7}
, {-55, -11, 0}
, {3, -21, 52}
, {55, -51, 28}
, {27, 49, 4}
, {73, -11, 51}
, {74, -21, -43}
, {-64, 10, -49}
, {-3, -30, 78}
, {21, -34, -21}
, {84, 74, -44}
, {-24, -42, -4}
, {31, 14, -20}
, {-26, -9, -73}
, {-14, -22, -41}
, {82, 6, 13}
, {-32, 71, 6}
, {-7, 14, -58}
, {1, 67, 52}
, {-27, 70, -54}
, {6, 2, -22}
, {10, 11, 2}
, {9, -3, 81}
, {-10, 4, -19}
, {-11, 19, 10}
, {-60, -55, 40}
, {13, 62, 47}
, {-18, 24, -51}
, {-41, -65, -11}
}
, {{-55, 35, 43}
, {-26, -69, -19}
, {29, 60, 0}
, {-64, 67, -58}
, {-7, -7, 33}
, {49, -27, -62}
, {19, 15, -38}
, {0, -20, 9}
, {-57, -13, 30}
, {-13, 23, -59}
, {-23, 11, -50}
, {50, 71, 39}
, {4, 0, 6}
, {-40, -16, 34}
, {-65, -10, 49}
, {-12, -23, 14}
, {39, 48, -76}
, {34, 61, -32}
, {30, 1, 53}
, {47, -4, -60}
, {-5, -77, -51}
, {-2, -49, 57}
, {12, -8, -32}
, {58, 31, -11}
, {26, 45, 13}
, {-2, 59, -77}
, {22, 64, 66}
, {-13, -21, -16}
, {-54, 56, 33}
, {32, 29, 5}
, {70, 65, 42}
, {31, -18, 29}
}
, {{-36, 25, 59}
, {-36, 22, -87}
, {-78, 81, -61}
, {15, -40, -45}
, {14, -20, 47}
, {41, 18, 50}
, {-6, 22, -55}
, {-10, -62, -11}
, {19, 1, -39}
, {-29, -24, -9}
, {-76, 22, -24}
, {34, -29, 41}
, {-32, -9, -30}
, {64, -55, -5}
, {-7, 49, -15}
, {-81, 38, -28}
, {-15, -10, -48}
, {-12, -12, 25}
, {-24, -60, 76}
, {23, 27, -44}
, {54, -36, 42}
, {18, 85, -45}
, {-51, -48, -75}
, {52, 10, 40}
, {52, 56, 98}
, {23, -8, -63}
, {-3, 15, -38}
, {-12, 19, 51}
, {27, 39, 33}
, {-50, -30, 27}
, {-6, 52, -26}
, {53, -51, 74}
}
, {{65, 48, -17}
, {24, 41, 0}
, {-22, 43, 0}
, {42, 11, -17}
, {9, -14, 51}
, {38, 53, -30}
, {38, 8, -7}
, {-60, -74, -37}
, {-25, -60, -34}
, {-42, 59, -72}
, {36, -32, 64}
, {-14, -14, -66}
, {-70, 6, 4}
, {-31, -70, -86}
, {-50, 63, -3}
, {53, -41, -5}
, {68, 23, 58}
, {7, 69, -8}
, {43, -73, -23}
, {-7, 57, 62}
, {-40, 24, 57}
, {-62, -68, -51}
, {-10, -15, 74}
, {-17, -45, 47}
, {-11, 51, -25}
, {66, 60, 65}
, {65, -23, -49}
, {35, 49, 12}
, {49, 43, 0}
, {47, 13, 1}
, {-61, -17, 51}
, {27, 38, -35}
}
, {{-65, 50, 65}
, {49, 56, 42}
, {-47, 29, 61}
, {-3, 55, 56}
, {-75, 17, 76}
, {-53, 11, 73}
, {-51, 29, -50}
, {-52, -72, 29}
, {32, 65, -65}
, {20, 43, -54}
, {-35, 41, -51}
, {-8, 4, -58}
, {52, 44, 59}
, {16, 53, -32}
, {-52, 60, -69}
, {9, 28, 56}
, {-65, 10, -5}
, {-38, 29, -33}
, {-8, -83, 21}
, {-55, -64, 2}
, {-53, -62, 48}
, {6, 4, 22}
, {5, -55, -48}
, {-29, 16, -61}
, {-30, -77, -18}
, {-18, 47, -25}
, {25, 60, 72}
, {-10, -51, 30}
, {56, 40, 21}
, {-12, 29, -12}
, {-30, 47, 32}
, {44, 11, -28}
}
, {{6, 60, 1}
, {18, 52, -33}
, {24, 22, 9}
, {-8, 16, 30}
, {52, -50, -26}
, {1, -61, -7}
, {42, -60, 0}
, {61, -9, -13}
, {-67, 34, -3}
, {80, 0, -69}
, {-54, -27, 48}
, {-30, 9, 25}
, {57, 46, -24}
, {-36, -25, -80}
, {-28, 51, 90}
, {3, -32, -17}
, {67, -54, 58}
, {22, 49, -4}
, {-72, 25, -46}
, {14, -56, -9}
, {-31, 17, 31}
, {-7, 68, 72}
, {-26, 11, 8}
, {-9, -10, -42}
, {32, 64, 46}
, {-31, -35, -48}
, {-27, 61, 30}
, {10, 22, -53}
, {28, 5, -63}
, {-44, -2, -19}
, {11, 43, 74}
, {31, 5, -13}
}
, {{-24, 8, -29}
, {10, 29, 33}
, {32, 52, 42}
, {70, -69, 73}
, {55, -38, 24}
, {-31, 12, 6}
, {-57, 38, -17}
, {88, 79, -17}
, {-41, 63, -17}
, {27, 36, 73}
, {-39, -26, 15}
, {56, -22, -46}
, {-7, -77, -28}
, {66, 12, 0}
, {-66, 1, 3}
, {32, -19, 49}
, {77, -31, 80}
, {-53, -1, 35}
, {-32, -21, 63}
, {18, 32, 7}
, {97, 94, 50}
, {-7, -51, -12}
, {13, -28, 18}
, {11, 37, 29}
, {-10, -33, 9}
, {-36, -24, 8}
, {65, -35, -53}
, {66, -27, -42}
, {-53, 8, -1}
, {-21, -77, -44}
, {-63, -5, -33}
, {86, -18, 8}
}
, {{-75, 4, -56}
, {58, -8, -52}
, {11, 11, 14}
, {-53, -16, -28}
, {49, -6, -23}
, {-42, -48, -12}
, {-39, -92, -8}
, {57, 37, -45}
, {32, -52, 64}
, {4, 43, 52}
, {61, -9, 57}
, {0, -9, -51}
, {-47, -48, -32}
, {46, 54, 20}
, {-28, -57, 47}
, {-78, 21, 47}
, {-39, 36, -24}
, {-8, 37, 66}
, {-41, -14, 37}
, {15, -50, -25}
, {-33, -6, -13}
, {-42, -1, 39}
, {-10, -56, -73}
, {43, -27, 40}
, {-8, -35, 63}
, {68, -16, -11}
, {45, 78, 83}
, {-33, 3, 48}
, {-51, -35, 45}
, {-50, 14, 17}
, {64, -12, 41}
, {58, 16, 60}
}
, {{-13, -19, 7}
, {-29, 23, -38}
, {-68, 33, 21}
, {35, -52, 61}
, {-19, -18, -20}
, {-52, 63, -35}
, {-76, -20, -15}
, {-28, 37, 6}
, {45, 46, 78}
, {-58, -16, 61}
, {-11, -52, 33}
, {16, 61, 67}
, {63, -26, -79}
, {36, 41, 62}
, {-61, 64, 3}
, {30, 54, -70}
, {61, -16, 61}
, {49, 34, 7}
, {-7, -24, -38}
, {59, 64, 76}
, {-74, 30, -56}
, {-11, -19, -56}
, {15, 68, 53}
, {-6, 72, 57}
, {-4, -27, 69}
, {-65, -29, 28}
, {12, 73, -47}
, {8, 50, -13}
, {62, -45, 17}
, {60, 48, -16}
, {4, -46, 67}
, {-13, -58, -64}
}
, {{15, 3, -8}
, {-45, 54, 30}
, {-55, -5, 46}
, {-59, 26, -69}
, {51, -39, -10}
, {-25, 75, -60}
, {-37, 7, -10}
, {68, -61, 70}
, {-8, -38, 52}
, {-62, -13, 3}
, {67, 10, 49}
, {-28, -32, -3}
, {85, -42, 76}
, {57, 37, -65}
, {-17, -47, -46}
, {-8, 89, -26}
, {26, 7, -79}
, {-48, -20, -60}
, {-48, 61, 50}
, {-65, -60, 30}
, {-79, -13, -3}
, {-57, -54, 69}
, {-70, -36, -55}
, {30, -11, 21}
, {18, -19, 10}
, {37, 81, 17}
, {67, -24, -4}
, {-12, -50, -76}
, {67, 19, -16}
, {18, 59, -8}
, {-24, -59, 84}
, {30, -61, 57}
}
, {{-9, 62, 16}
, {60, -15, 53}
, {-41, 23, -13}
, {53, 47, 0}
, {28, 27, -49}
, {-21, -49, 51}
, {-15, -68, -54}
, {-5, 22, 21}
, {7, 12, 35}
, {-61, -3, -4}
, {-2, -32, -3}
, {26, -67, 48}
, {81, -36, 26}
, {-18, 36, -25}
, {52, -52, -72}
, {-27, -9, 50}
, {-12, 35, 36}
, {-74, -13, 28}
, {-71, 55, -41}
, {-3, -74, 33}
, {45, -28, 54}
, {-56, 29, 55}
, {-14, 69, -70}
, {19, 6, -62}
, {4, 7, 9}
, {-43, -48, -55}
, {-12, -25, 11}
, {14, 39, -11}
, {-22, 3, 35}
, {0, 24, -69}
, {28, 54, 11}
, {63, -25, -42}
}
, {{37, 24, 24}
, {65, 53, -5}
, {-14, -42, -49}
, {-53, 43, -55}
, {-11, 9, 72}
, {30, 36, -49}
, {-39, -35, -25}
, {-72, -33, -14}
, {15, 66, 59}
, {-16, 18, 21}
, {39, -10, -48}
, {65, 71, -48}
, {15, -26, 1}
, {-78, -54, -19}
, {18, 23, 43}
, {8, -51, 46}
, {45, -47, 0}
, {-57, 12, 33}
, {-82, -87, -87}
, {66, -8, 66}
, {-37, 72, -16}
, {-66, -65, 72}
, {-52, -7, 6}
, {-53, 51, 38}
, {49, 17, -15}
, {78, 48, -16}
, {63, 23, -14}
, {-2, -15, -29}
, {-17, -34, 31}
, {-58, -86, -67}
, {-19, -1, -36}
, {-38, 9, -70}
}
, {{13, -21, 68}
, {-56, 71, 6}
, {25, -25, -30}
, {25, -43, 64}
, {-36, -58, -70}
, {-52, 59, -50}
, {10, -70, -30}
, {33, -50, 33}
, {67, -18, 30}
, {56, 58, 70}
, {58, -4, 5}
, {-19, -29, 4}
, {13, 4, 38}
, {91, 34, 70}
, {69, -27, 32}
, {-61, 68, -40}
, {50, 12, 4}
, {49, 53, -5}
, {53, 81, -45}
, {-39, 39, 29}
, {43, -58, 12}
, {-3, -63, 49}
, {-20, 31, -39}
, {-49, -2, 18}
, {24, 24, -31}
, {57, 6, 28}
, {0, -59, -55}
, {20, -53, 24}
, {-63, -34, 40}
, {-16, -2, -62}
, {48, -36, -15}
, {38, -43, 21}
}
, {{34, -22, 66}
, {-66, -43, -62}
, {56, 12, 43}
, {-18, -52, -59}
, {-65, 43, 60}
, {59, -3, 60}
, {-67, 3, -10}
, {36, -36, 4}
, {43, -66, -21}
, {-1, 14, -37}
, {4, 22, -6}
, {9, 77, 32}
, {49, -56, 12}
, {-78, 39, -1}
, {-32, 24, -36}
, {4, -72, 7}
, {-60, 22, 37}
, {-24, 48, 59}
, {-25, 34, -24}
, {43, -41, 60}
, {28, -62, 53}
, {-42, -45, 8}
, {-61, -57, -80}
, {-59, 10, 53}
, {46, -44, -19}
, {55, 5, -57}
, {-32, 27, 0}
, {-71, -14, -40}
, {73, 50, -53}
, {-37, -58, -4}
, {-33, 46, 16}
, {-23, 31, 14}
}
, {{24, -42, 43}
, {-6, -38, 56}
, {21, 46, -47}
, {-60, 69, 21}
, {-16, -33, -58}
, {23, -45, -22}
, {-69, 31, 56}
, {45, -51, 32}
, {64, -62, 50}
, {-27, 32, -22}
, {-70, 45, -65}
, {36, 7, 41}
, {-38, 66, -9}
, {30, 0, -4}
, {18, -50, -58}
, {36, 44, -77}
, {-64, 64, -85}
, {17, 20, -37}
, {34, -68, 42}
, {-27, 25, 51}
, {0, 95, -72}
, {-72, 13, 49}
, {-47, 33, -23}
, {60, 55, -40}
, {74, 33, 50}
, {-20, 30, -16}
, {28, -30, 7}
, {-21, 41, -60}
, {-61, -6, 43}
, {-18, -7, -62}
, {-54, 37, 18}
, {43, -22, -63}
}
, {{32, 15, 43}
, {75, -74, 57}
, {75, -21, -39}
, {-30, 48, 74}
, {60, 15, -55}
, {33, 70, 67}
, {-11, -36, 22}
, {-65, 62, -28}
, {-68, -49, -51}
, {15, 28, -56}
, {-25, -17, -71}
, {-64, 39, 75}
, {80, 0, 9}
, {6, 92, -50}
, {-65, 7, -65}
, {-39, 7, -74}
, {-28, 50, -46}
, {4, -73, -7}
, {71, 56, -80}
, {-8, -59, 39}
, {-39, -2, -20}
, {-12, -56, 59}
, {29, -37, -15}
, {-49, 20, 44}
, {50, 10, 24}
, {39, -4, 23}
, {45, 13, 57}
, {4, -30, 40}
, {75, -32, -32}
, {22, -25, 21}
, {-22, 8, 6}
, {75, -42, 23}
}
, {{61, -38, -70}
, {-26, 28, 71}
, {-63, -5, 21}
, {50, 15, -40}
, {-54, -20, 37}
, {-52, -45, -18}
, {47, 18, 13}
, {13, -44, 51}
, {-63, 30, -32}
, {-32, 48, -34}
, {-3, -4, -7}
, {-3, -39, -15}
, {0, 10, 74}
, {-16, -88, -84}
, {65, -35, 69}
, {-22, 30, -44}
, {-61, 40, 59}
, {15, 80, 65}
, {-69, -22, 3}
, {37, 44, -17}
, {-35, -57, -81}
, {-66, 44, 36}
, {14, 57, 36}
, {-30, 74, 45}
, {-20, -27, 47}
, {6, -71, 64}
, {29, 55, 61}
, {18, 35, -23}
, {20, -72, 39}
, {50, 2, 3}
, {-27, -11, 64}
, {7, -12, -32}
}
, {{-30, -2, 58}
, {-20, 39, 63}
, {10, -78, -53}
, {-33, -7, -18}
, {26, -60, -49}
, {7, -24, -6}
, {22, 62, -79}
, {20, -37, -35}
, {-21, 34, -4}
, {-70, -11, 9}
, {7, 37, -25}
, {32, -69, 37}
, {12, 27, 76}
, {-14, -56, 3}
, {37, 13, 48}
, {29, -9, 26}
, {-67, 38, -45}
, {25, 66, 35}
, {-85, -63, -47}
, {-22, 9, 62}
, {-84, -16, 17}
, {-26, -2, -69}
, {-42, -2, 42}
, {-28, 54, 81}
, {2, -71, -83}
, {60, 37, -55}
, {10, 83, 76}
, {-61, -63, -38}
, {68, 66, -67}
, {-58, 78, -6}
, {-67, 48, 12}
, {77, 2, 29}
}
, {{-62, 75, 70}
, {-58, -24, 55}
, {-45, -13, -13}
, {6, 50, 59}
, {-58, 8, -51}
, {-24, -22, 21}
, {-64, 23, 6}
, {39, 46, 24}
, {55, -26, 65}
, {-16, -12, -42}
, {-55, 28, 35}
, {25, -30, -41}
, {32, 13, 33}
, {-72, -3, 49}
, {35, -27, 68}
, {-77, 31, -16}
, {-22, 10, -61}
, {-2, -17, 51}
, {-49, -17, 1}
, {-45, 2, -28}
, {-37, 10, 29}
, {13, 15, -25}
, {49, 68, 53}
, {-12, 49, 65}
, {65, -21, -49}
, {40, -29, 53}
, {21, -40, 45}
, {-45, 28, -36}
, {-19, 73, -41}
, {-34, 44, 70}
, {30, 18, 0}
, {64, 39, 29}
}
, {{72, 6, -9}
, {-52, 38, -21}
, {-26, -49, 44}
, {1, -65, -26}
, {33, -2, 11}
, {-4, 22, 68}
, {5, 90, -42}
, {34, 22, -70}
, {19, 53, 40}
, {15, 47, -19}
, {-22, 30, 21}
, {-42, 0, 0}
, {-15, -50, -76}
, {-17, 44, -43}
, {-66, 28, -2}
, {1, 20, -76}
, {37, 35, 31}
, {15, 36, -26}
, {-77, 23, -10}
, {7, -32, -33}
, {44, 54, -44}
, {38, -66, 4}
, {-9, 50, -3}
, {54, -48, -34}
, {-26, -10, 73}
, {-27, -2, -44}
, {-33, 50, -8}
, {18, -19, -60}
, {67, 11, -14}
, {84, -77, -69}
, {18, -5, 13}
, {6, -41, 31}
}
, {{55, -5, 2}
, {46, 57, 51}
, {-72, -16, -5}
, {-14, 36, 47}
, {55, -25, -55}
, {2, -77, -31}
, {46, 4, 47}
, {82, 9, -9}
, {53, 27, -30}
, {-61, 0, 12}
, {36, 37, 36}
, {-45, -20, -1}
, {-25, 9, -22}
, {17, -44, -54}
, {16, -42, 9}
, {-6, 6, -18}
, {-49, 3, 52}
, {38, -70, 11}
, {-23, -74, -71}
, {74, 37, -9}
, {-29, 44, 52}
, {-63, -66, -18}
, {37, 14, 61}
, {-68, 63, 48}
, {-60, -55, 48}
, {-59, 21, -26}
, {-7, 37, -21}
, {-37, 13, -48}
, {42, -7, -4}
, {1, 74, 54}
, {-48, 36, -4}
, {0, 47, 17}
}
, {{74, -16, -32}
, {34, 70, -77}
, {-31, 17, -3}
, {53, -27, 45}
, {-15, -34, -77}
, {30, -24, -59}
, {-44, -7, 69}
, {17, 57, 49}
, {32, 0, -36}
, {-62, 56, 18}
, {83, -19, -39}
, {-5, 49, 66}
, {-16, -42, -35}
, {-69, 75, 69}
, {-51, 46, 12}
, {71, 50, 53}
, {-20, -55, -33}
, {-21, 9, -20}
, {-1, -66, 68}
, {-64, -68, -17}
, {-6, -86, -52}
, {-44, -55, -29}
, {-27, -88, -39}
, {6, 67, -6}
, {7, 28, 49}
, {70, 54, -3}
, {28, 4, 56}
, {57, 46, 9}
, {42, 0, -53}
, {-70, -70, 28}
, {5, -12, 28}
, {-7, 56, 36}
}
, {{-60, 58, 47}
, {20, 22, 13}
, {3, -14, 21}
, {18, -7, -3}
, {-64, 46, 4}
, {81, -71, -54}
, {14, -47, 47}
, {-26, 4, 27}
, {42, -14, -62}
, {9, -53, -57}
, {-65, 1, -18}
, {12, 18, -35}
, {-8, -47, -23}
, {-59, 22, -27}
, {-15, -68, -32}
, {51, -32, 12}
, {-22, -70, 10}
, {-36, -59, -27}
, {-38, -77, 1}
, {70, -23, 76}
, {-22, -1, -5}
, {47, -38, 33}
, {-43, 45, 38}
, {56, -32, -61}
, {22, 9, 32}
, {-46, 51, 18}
, {66, -5, -30}
, {22, -1, 11}
, {-71, 2, 44}
, {19, 44, -22}
, {65, 66, 79}
, {4, -66, -2}
}
, {{29, 7, -24}
, {62, 13, 15}
, {-15, -42, 13}
, {38, -39, 23}
, {24, -13, 40}
, {-38, -21, -11}
, {-51, -65, 28}
, {-69, 31, -58}
, {5, 12, 0}
, {51, 47, -2}
, {64, 54, 18}
, {-2, -4, 23}
, {-30, -20, -41}
, {7, -48, 40}
, {58, 6, -57}
, {67, -55, 58}
, {-13, 26, -26}
, {58, 73, 1}
, {64, -67, 63}
, {-15, 0, -12}
, {42, -20, -27}
, {71, 65, 70}
, {7, 54, -23}
, {40, -77, 23}
, {-1, 78, 69}
, {31, -78, -10}
, {36, 35, 16}
, {-31, 8, 12}
, {48, 5, -55}
, {54, -42, 1}
, {-48, -28, 19}
, {83, 29, -2}
}
, {{65, 18, -11}
, {-19, -67, 54}
, {16, 72, -24}
, {-63, 3, 2}
, {46, 71, -33}
, {-64, 65, 11}
, {-69, -9, -85}
, {20, -81, 27}
, {-38, -39, -13}
, {-28, -45, 0}
, {13, -52, 64}
, {-78, 24, 58}
, {-31, 44, 25}
, {17, 6, -60}
, {44, 20, 20}
, {35, 41, -3}
, {-41, -102, 17}
, {-72, 35, -46}
, {58, -44, 37}
, {-13, -6, 62}
, {33, 65, 39}
, {1, -61, 30}
, {48, -94, -62}
, {-22, -37, -27}
, {-40, -31, 16}
, {74, 65, -25}
, {42, -67, -67}
, {2, 7, -71}
, {59, 48, 73}
, {2, 11, 41}
, {23, -72, -36}
, {44, 31, -8}
}
, {{-60, -21, 72}
, {62, 56, 61}
, {-44, -21, -77}
, {80, -31, -59}
, {33, -62, 16}
, {-11, 56, 12}
, {26, 46, -32}
, {38, 23, -67}
, {12, -58, 74}
, {-40, 29, -41}
, {-46, -12, -5}
, {47, -12, -77}
, {73, -52, -21}
, {44, 25, -39}
, {-73, -11, -27}
, {41, -57, -2}
, {-53, 60, 62}
, {-20, 73, 52}
, {-4, 57, 9}
, {30, 61, -75}
, {-18, 48, 29}
, {-52, -39, -43}
, {-3, -37, 54}
, {-68, 10, -11}
, {-45, 41, -34}
, {43, -44, -44}
, {28, -42, 13}
, {52, -32, -40}
, {67, -55, 51}
, {-8, -64, 50}
, {66, 41, 34}
, {-9, -13, 10}
}
, {{-13, 69, -67}
, {-58, -66, -34}
, {34, -67, -52}
, {4, 4, 36}
, {-64, -29, -46}
, {-14, -54, -8}
, {0, -42, 80}
, {-13, 0, 15}
, {-54, -3, 12}
, {74, 25, 25}
, {-2, 67, -70}
, {-58, -47, 61}
, {-7, 2, 20}
, {60, -4, -50}
, {-61, 64, 73}
, {46, -3, -47}
, {32, 54, 32}
, {89, -15, -31}
, {-40, -41, -57}
, {70, -51, -17}
, {-53, 41, 51}
, {-62, 3, 63}
, {-66, 33, -13}
, {-28, -39, 20}
, {-61, 55, -49}
, {74, 41, -74}
, {71, 87, -31}
, {-17, 50, 57}
, {-23, -1, 21}
, {7, 33, -27}
, {29, -22, 68}
, {36, -96, -6}
}
, {{-32, -37, -55}
, {-20, 66, 26}
, {8, -20, -10}
, {-5, 48, 52}
, {-17, -48, 4}
, {-37, 38, -77}
, {-16, 74, -4}
, {32, -56, 20}
, {26, -11, -5}
, {-72, -60, 22}
, {0, -13, 0}
, {47, -55, -9}
, {19, 7, -22}
, {11, 55, 58}
, {-15, 21, -52}
, {-52, 49, -73}
, {32, -55, 43}
, {-26, 50, 19}
, {93, -46, 18}
, {53, -33, 0}
, {6, 48, -51}
, {3, 1, 29}
, {20, 22, -59}
, {0, 39, 52}
, {-26, -49, 47}
, {-50, 34, 20}
, {4, -26, -70}
, {-26, 23, -3}
, {24, 79, 0}
, {-27, -42, 30}
, {68, 46, -78}
, {22, 27, -49}
}
, {{69, -64, 48}
, {-49, -73, 7}
, {24, 54, -68}
, {-63, -8, -25}
, {-81, -46, -23}
, {-26, -62, -73}
, {19, -32, -13}
, {51, 52, -50}
, {-57, 71, -28}
, {-25, 49, -60}
, {-18, 3, -65}
, {3, 24, 11}
, {-28, 62, -17}
, {81, 15, -54}
, {45, -58, -31}
, {-24, 69, -31}
, {-14, -65, 15}
, {11, -3, 23}
, {-1, 34, 87}
, {34, 78, 48}
, {-25, 34, -14}
, {-60, 14, 74}
, {-16, -33, -54}
, {-6, 4, 71}
, {0, 84, -20}
, {-44, 0, 24}
, {-39, 7, 38}
, {31, 68, -10}
, {6, -24, 64}
, {74, 67, 53}
, {60, -69, 32}
, {14, -23, -23}
}
, {{-52, 6, 22}
, {-10, -41, 47}
, {-25, -36, 45}
, {-4, -30, 68}
, {44, -46, -11}
, {38, 43, 46}
, {31, 41, 45}
, {57, 28, -2}
, {-15, 51, 61}
, {-32, -43, -54}
, {-5, 1, -26}
, {38, 45, -58}
, {-20, 67, 23}
, {89, 72, 2}
, {-6, -53, -17}
, {20, 62, -3}
, {-50, -17, 31}
, {-53, 15, 71}
, {-7, 31, -53}
, {27, -82, 1}
, {-54, -16, 73}
, {10, 35, 18}
, {35, -71, -16}
, {51, -48, -23}
, {-59, -21, 71}
, {-1, 65, 88}
, {-44, 10, -19}
, {69, -30, 42}
, {-11, -60, -42}
, {30, 50, -55}
, {19, -63, -62}
, {47, -80, -27}
}
, {{-10, 8, 20}
, {40, 24, 29}
, {-72, 26, 8}
, {4, 46, -4}
, {38, -17, 42}
, {26, -60, -43}
, {55, -21, -45}
, {-30, -22, 73}
, {-40, 15, 86}
, {-2, 71, -41}
, {72, -54, -68}
, {-29, 82, -18}
, {81, -30, -16}
, {-26, 0, 80}
, {3, -52, -36}
, {80, -75, -9}
, {-65, 34, 28}
, {-37, 20, -30}
, {-50, 39, -59}
, {35, -4, 10}
, {59, -42, 6}
, {-13, 2, -31}
, {-1, -6, 27}
, {16, 12, -10}
, {55, -56, 12}
, {-40, -37, -32}
, {-9, -26, 8}
, {46, -36, -52}
, {3, -51, 3}
, {42, -2, 11}
, {-33, -34, 7}
, {40, 5, -24}
}
, {{-49, 40, 26}
, {72, -36, -8}
, {-24, 84, -4}
, {-53, 6, -29}
, {-49, 16, -48}
, {4, 57, -30}
, {-23, 35, -44}
, {-62, -80, -59}
, {24, 17, -35}
, {-65, -29, 60}
, {8, -21, -10}
, {25, -6, -62}
, {19, 61, 80}
, {16, 17, 54}
, {-38, 53, -4}
, {54, -36, -43}
, {-69, 3, -59}
, {-51, -11, 26}
, {18, -20, 56}
, {20, 8, 68}
, {-53, 0, -69}
, {15, 22, -68}
, {63, -96, -8}
, {-19, -6, 42}
, {-22, -32, -30}
, {12, 73, 85}
, {15, -11, -39}
, {-73, -29, -81}
, {43, -24, 77}
, {-15, 28, 27}
, {-41, 38, 66}
, {87, 78, 47}
}
, {{-29, 64, -67}
, {-43, 47, -57}
, {15, -2, 64}
, {12, -42, 54}
, {10, 65, -28}
, {10, 55, 3}
, {-26, -36, -48}
, {-29, -60, 65}
, {54, -14, -25}
, {-18, -48, -8}
, {-34, 30, 30}
, {62, -65, 4}
, {-59, -13, 11}
, {80, 82, -44}
, {41, 13, -3}
, {-69, 91, -11}
, {8, 45, 46}
, {3, -49, -37}
, {4, 30, 63}
, {-8, -5, 70}
, {27, 9, -43}
, {29, 83, 58}
, {-79, -20, 28}
, {20, -8, 57}
, {-59, -80, -18}
, {58, 82, 27}
, {58, -58, -43}
, {-55, 31, -45}
, {-69, -51, 60}
, {-39, -54, -75}
, {38, -61, 16}
, {-15, -25, -37}
}
, {{21, -28, -59}
, {-27, -41, -16}
, {0, -1, 23}
, {70, -33, 2}
, {-64, 34, 46}
, {-3, -47, 21}
, {33, 28, -36}
, {35, 14, 62}
, {-45, -6, 19}
, {75, -76, -36}
, {18, 35, -51}
, {43, -47, -39}
, {-48, -96, 25}
, {18, 33, -36}
, {73, 27, -80}
, {-9, -59, 84}
, {-1, 41, 57}
, {36, 40, -59}
, {-41, -26, -69}
, {68, 68, 12}
, {-42, 15, 2}
, {-72, 0, -65}
, {-54, 74, 48}
, {4, -58, 22}
, {-54, -53, -18}
, {41, 65, 66}
, {-67, -21, -65}
, {81, 55, -20}
, {2, 8, -32}
, {0, -76, -51}
, {-29, 63, 49}
, {-6, 88, 60}
}
, {{10, 68, 59}
, {69, 15, 61}
, {35, 16, 79}
, {74, -2, -2}
, {-67, 0, 57}
, {60, 44, 35}
, {18, -28, -42}
, {22, 78, -21}
, {11, 17, -33}
, {-72, -37, -6}
, {-54, -1, 28}
, {36, -27, 16}
, {-49, -37, -25}
, {42, 26, 92}
, {-36, -24, -59}
, {-28, -65, -74}
, {-9, -18, -57}
, {8, -66, -47}
, {84, 46, 73}
, {-38, -68, 14}
, {32, -58, -18}
, {65, -18, 35}
, {-4, 29, 46}
, {-24, -13, 41}
, {-18, 45, 37}
, {-42, 82, 63}
, {-10, 5, -4}
, {-61, 22, -56}
, {-48, 57, -34}
, {-30, 41, -43}
, {68, 2, -57}
, {-48, 6, -26}
}
, {{71, -67, -11}
, {-23, 44, -29}
, {-42, 46, 7}
, {33, -38, -10}
, {-39, 57, 63}
, {-72, 30, 61}
, {23, -27, -46}
, {29, -37, -41}
, {-37, 74, -67}
, {75, -53, 63}
, {26, -53, -31}
, {-62, 43, 46}
, {-43, -43, -53}
, {36, 2, -38}
, {34, -14, 71}
, {51, 52, -11}
, {46, 23, -71}
, {31, -71, -56}
, {-11, -58, -15}
, {-26, 4, 68}
, {-36, -24, 60}
, {53, -30, -29}
, {-2, 27, -71}
, {73, -8, 56}
, {-55, -24, 4}
, {66, 26, 5}
, {76, -43, -3}
, {-19, -6, -21}
, {59, -45, -64}
, {9, -60, 6}
, {-36, 30, 47}
, {-41, -87, -21}
}
, {{50, -32, -40}
, {46, -59, 53}
, {-72, -30, -73}
, {-9, -45, -65}
, {-35, -71, -41}
, {6, 47, -72}
, {61, -31, 32}
, {-44, -22, -29}
, {-56, -19, 75}
, {-35, 73, 62}
, {11, -53, 24}
, {-13, -57, -35}
, {11, 10, -66}
, {9, -53, -36}
, {-66, 9, 19}
, {32, 61, -51}
, {-63, -22, 66}
, {-1, -26, -7}
, {-21, 45, -34}
, {-11, -57, 75}
, {-6, 62, 42}
, {6, -61, 13}
, {-45, 0, 34}
, {-43, -9, -2}
, {-31, -45, 6}
, {22, 21, 14}
, {53, -31, 5}
, {32, -24, -65}
, {-65, -65, -66}
, {-14, -74, 58}
, {-29, -71, -11}
, {44, -2, -58}
}
, {{-57, 37, -38}
, {-67, 4, 70}
, {28, -42, 13}
, {15, 20, -17}
, {25, 21, 46}
, {82, 10, -43}
, {-34, 5, 62}
, {-10, 1, -51}
, {-32, -44, 2}
, {-33, -21, 53}
, {1, -66, 68}
, {-61, -37, -77}
, {67, 26, -69}
, {-26, 17, 13}
, {44, -22, -30}
, {45, -4, 32}
, {44, -19, -36}
, {-11, -53, 21}
, {48, 30, 65}
, {-63, 26, 70}
, {23, -71, 68}
, {8, -1, -60}
, {22, -21, 56}
, {-72, 71, 35}
, {-58, 5, 37}
, {20, -67, 73}
, {-59, -11, 63}
, {-59, 82, -45}
, {-36, 29, 66}
, {-81, 19, 68}
, {74, -3, 11}
, {55, 29, -32}
}
, {{44, 12, 50}
, {0, 3, 41}
, {-23, 6, -19}
, {21, -63, 10}
, {-26, 40, -74}
, {2, 28, 23}
, {-43, 20, 5}
, {40, 5, -27}
, {36, -66, 59}
, {61, -30, -59}
, {18, 24, 1}
, {9, 23, -78}
, {48, 52, -22}
, {25, -31, 69}
, {-55, -55, -17}
, {-36, 5, 68}
, {-69, 65, 59}
, {56, 12, -47}
, {63, 72, -75}
, {66, 47, -40}
, {1, -30, -59}
, {-2, -60, -2}
, {33, -36, 58}
, {77, 64, 49}
, {38, 50, -52}
, {-35, -61, -58}
, {42, -13, -44}
, {-68, -3, 38}
, {3, -18, 69}
, {-33, 4, -77}
, {-12, -36, 54}
, {90, 44, 16}
}
, {{-10, -8, 2}
, {22, -49, -71}
, {-19, -46, 5}
, {1, 49, 49}
, {70, 48, 13}
, {27, -53, 4}
, {-84, -21, -69}
, {56, -5, -25}
, {-48, 57, -67}
, {23, 12, 63}
, {65, 55, 10}
, {-20, -44, 40}
, {32, -73, 47}
, {-52, -59, -26}
, {52, 27, -60}
, {62, 29, -49}
, {-9, 63, 55}
, {-29, -26, -37}
, {-79, -42, 21}
, {-50, 39, 25}
, {-52, 22, -2}
, {-64, -22, -63}
, {21, 19, -47}
, {-22, 49, 67}
, {-75, -26, 25}
, {25, 38, -76}
, {63, 51, 17}
, {5, -40, 61}
, {26, -43, 42}
, {39, -54, -71}
, {49, -53, -70}
, {11, -71, -62}
}
, {{-41, 19, 25}
, {10, -66, -59}
, {-66, -15, -9}
, {-9, -14, -25}
, {40, -56, 1}
, {-52, -35, -51}
, {-57, -13, 29}
, {-12, -1, -49}
, {-7, 59, 51}
, {-72, -61, -16}
, {48, 60, 62}
, {-35, 1, -38}
, {50, 63, -18}
, {-50, 0, 16}
, {20, 8, 73}
, {39, 67, 44}
, {20, 6, 82}
, {56, 6, 62}
, {-32, 15, -57}
, {-57, -18, -53}
, {-24, -16, 35}
, {67, 55, -67}
, {-12, 83, 75}
, {40, -7, 3}
, {26, -53, -69}
, {-10, 53, 39}
, {51, 44, 17}
, {0, -39, -7}
, {-28, 35, -20}
, {-2, 7, -38}
, {-56, -56, 39}
, {-23, 86, 56}
}
, {{36, -57, 20}
, {14, -63, 55}
, {-31, 20, 31}
, {59, 69, 8}
, {-33, 12, -66}
, {-20, 42, -91}
, {40, 3, -41}
, {-5, -66, -32}
, {-9, -26, 7}
, {3, -41, -6}
, {-67, 2, -53}
, {49, 73, -9}
, {73, -45, 56}
, {61, 19, -49}
, {42, 45, 30}
, {-14, -5, 1}
, {-8, 43, -75}
, {4, 7, -39}
, {44, -32, 55}
, {19, -12, 53}
, {-9, 42, -50}
, {34, 26, 31}
, {-40, 14, -58}
, {69, 39, 36}
, {-19, -33, -31}
, {10, -71, -4}
, {-80, -25, -36}
, {67, -49, 67}
, {-56, -32, 54}
, {-24, 64, -43}
, {71, 13, 17}
, {-98, 0, -36}
}
, {{-46, -36, 41}
, {-37, 49, 15}
, {-62, 30, 8}
, {-43, -81, 71}
, {49, 36, -57}
, {-28, -55, -73}
, {0, -28, -48}
, {60, 57, 6}
, {40, 63, 18}
, {32, 24, 61}
, {-26, -16, 24}
, {-62, 7, -9}
, {19, 14, 9}
, {-41, 46, 79}
, {79, 9, 0}
, {0, 14, -67}
, {59, 64, 58}
, {-16, -45, -3}
, {-25, 84, -10}
, {-38, 64, 1}
, {94, 40, 61}
, {-54, -41, -17}
, {-35, -14, -10}
, {-36, 46, -50}
, {21, 72, 14}
, {68, 19, 16}
, {-55, -74, 2}
, {41, -2, -56}
, {35, -35, -66}
, {-60, -21, -65}
, {8, 73, 5}
, {32, -3, -36}
}
, {{62, 7, 54}
, {8, -3, -38}
, {-55, 37, 57}
, {-6, 30, 53}
, {28, 34, 25}
, {-31, -57, 37}
, {39, -18, 20}
, {28, -73, 14}
, {10, -81, -38}
, {-52, -16, 25}
, {-2, 68, 61}
, {72, 31, 50}
, {77, -15, 33}
, {24, -88, 28}
, {-21, -52, 35}
, {-68, -23, -27}
, {-50, -7, -34}
, {73, -63, 69}
, {43, -19, 0}
, {61, 62, 4}
, {-53, 10, 59}
, {-17, 46, -33}
, {54, -26, 28}
, {-69, -45, 63}
, {66, -11, -21}
, {54, -20, 41}
, {-70, -71, 33}
, {54, -56, -57}
, {-45, -13, 45}
, {42, -39, 34}
, {40, 38, -62}
, {-36, 0, -10}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE