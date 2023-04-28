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


const int16_t conv1d_70_bias[CONV_FILTERS] = {3, 2, 2, 14, -1, 11, 6, -4, 7, 15, 0, 4, -1, -3, 6, -11, 7, 0, 10, -7, -18, 7, 19, -1, 7, -3, 0, 17, 4, 1, -6, 3, 13, -5, 16, -6, -3, -3, 23, 7, -10, 5, 5, 17, 4, -6, 10, 0, 0, 3, 2, 11, 7, 11, -2, -10, 3, -7, 3, 13, -11, -14, -3, 6}
;

const int16_t conv1d_70_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{31, -67, 82}
, {-56, 25, -7}
, {-59, 13, -55}
, {-7, 17, 73}
, {-47, -9, -53}
, {64, 28, -14}
, {73, 31, -41}
, {47, -10, 20}
, {51, -41, 0}
, {17, 49, -40}
, {41, -26, -44}
, {-7, 64, 58}
, {17, 41, 19}
, {51, 26, 4}
, {-76, -46, -21}
, {68, -56, 29}
, {55, -5, 58}
, {-8, -14, 65}
, {-6, -66, 52}
, {-62, 77, -2}
, {-75, 21, -65}
, {7, -34, 7}
, {45, 64, -12}
, {36, 0, -26}
, {-34, 67, -43}
, {-71, 31, -78}
, {14, 31, 73}
, {49, 9, -49}
, {5, -67, -57}
, {-52, -8, -19}
, {-60, 52, -71}
, {-33, 5, -66}
}
, {{-4, 11, -10}
, {-44, 4, 71}
, {-42, 44, 22}
, {5, -39, -9}
, {59, -65, -61}
, {-8, -4, 55}
, {-48, -45, -39}
, {27, -44, 73}
, {60, 67, 12}
, {63, 21, 15}
, {-43, -24, -81}
, {-1, -6, -7}
, {-53, -40, 61}
, {-58, -40, -79}
, {54, 47, 19}
, {-39, 71, 50}
, {8, -30, 1}
, {85, 41, 51}
, {-19, 70, 6}
, {0, -59, 9}
, {-28, -78, 71}
, {53, -37, -63}
, {0, -16, 45}
, {5, 43, 11}
, {-61, 52, -50}
, {23, -76, 38}
, {22, 29, -31}
, {23, 0, 30}
, {-64, -21, 34}
, {-18, -58, -33}
, {77, -32, -63}
, {-56, -57, -23}
}
, {{-6, 20, 8}
, {47, 83, -1}
, {50, -50, 20}
, {36, 55, -17}
, {-21, 54, 62}
, {12, 34, 51}
, {-4, 36, -84}
, {48, 16, -59}
, {75, -44, 60}
, {-62, 59, -55}
, {-56, 28, -69}
, {-41, -47, -30}
, {-41, 28, 67}
, {0, -34, 8}
, {-1, -23, 63}
, {-79, -32, -39}
, {-45, 11, 53}
, {-27, 35, -54}
, {36, -39, -28}
, {-21, 48, 76}
, {8, 33, 40}
, {-61, -69, 15}
, {3, 44, -62}
, {84, -56, 4}
, {-71, -40, -34}
, {55, 4, -10}
, {30, 4, -41}
, {-61, -21, 30}
, {2, 32, 28}
, {6, 38, -47}
, {24, 31, 20}
, {46, -23, -60}
}
, {{36, 23, 4}
, {-4, -64, -36}
, {17, 59, 42}
, {38, 18, -61}
, {42, 26, -35}
, {42, 16, -3}
, {-33, 84, 57}
, {-19, 11, 26}
, {24, 24, 0}
, {65, -47, -16}
, {-56, 18, 76}
, {-47, -19, 72}
, {24, -61, 33}
, {-26, 13, -62}
, {51, -69, -84}
, {1, 71, 66}
, {-14, 49, 33}
, {-44, -25, 50}
, {-8, -68, 15}
, {37, -26, -2}
, {-30, 74, 27}
, {-47, -68, -52}
, {24, 21, -65}
, {-33, -11, -76}
, {-70, -71, -54}
, {-41, 23, -67}
, {1, 95, -4}
, {-14, -26, 36}
, {-31, -32, -33}
, {-43, -67, 74}
, {54, 74, -15}
, {68, 24, -48}
}
, {{48, 20, -16}
, {-61, -49, -13}
, {-29, -32, 15}
, {18, -69, -14}
, {51, 69, 0}
, {-73, 58, 32}
, {-34, -13, -49}
, {-41, -40, -21}
, {29, -59, 74}
, {-51, 62, 18}
, {78, 23, 69}
, {37, -26, 34}
, {-58, 11, -66}
, {36, 17, -13}
, {36, -5, 25}
, {-1, 74, 74}
, {34, 18, -3}
, {4, 2, 4}
, {-29, -27, -12}
, {70, -45, -8}
, {5, -18, 13}
, {65, -62, 31}
, {-53, 15, -26}
, {-33, 25, 1}
, {41, 54, 16}
, {67, -23, -51}
, {-71, 38, -62}
, {35, -65, 70}
, {-59, -37, 8}
, {-49, 2, -61}
, {42, -24, -48}
, {33, 71, 4}
}
, {{-59, -64, 84}
, {17, 65, 7}
, {-57, -34, 39}
, {10, -2, -7}
, {13, -18, 22}
, {34, -19, 6}
, {60, 78, -47}
, {0, 30, -19}
, {-66, -47, -1}
, {54, 67, -5}
, {66, 44, 7}
, {27, 34, 66}
, {-65, 6, 23}
, {0, 31, -59}
, {52, -66, -72}
, {-59, 60, 12}
, {-70, 31, 9}
, {-52, 27, 55}
, {-27, 0, 8}
, {16, -24, -13}
, {-35, -71, -21}
, {-69, 57, 51}
, {45, -59, -39}
, {27, 21, -10}
, {46, -18, 13}
, {-53, -58, -68}
, {-49, 65, -50}
, {4, -29, -13}
, {42, 8, 54}
, {-10, 67, 16}
, {-68, 44, 36}
, {-13, -49, 14}
}
, {{50, 49, 14}
, {-30, 4, -1}
, {-28, 49, -65}
, {-45, -14, -11}
, {1, -45, -58}
, {-4, 66, -37}
, {-22, 79, 23}
, {-25, 8, -20}
, {35, -72, 74}
, {-17, -63, 31}
, {-49, 21, 33}
, {-10, -21, 6}
, {73, 46, 51}
, {-16, -21, -69}
, {22, 65, -19}
, {68, 26, -12}
, {-14, -57, 53}
, {13, -45, 13}
, {62, -12, -22}
, {-55, 20, 88}
, {-12, 7, -35}
, {-39, 16, -27}
, {64, 9, 64}
, {-42, 6, -30}
, {-42, 28, -45}
, {40, -66, 74}
, {42, 59, 41}
, {29, -58, -50}
, {23, 47, -40}
, {52, 15, -19}
, {-38, -11, -20}
, {57, -26, -22}
}
, {{-58, 27, 22}
, {-31, 40, -59}
, {16, 0, 60}
, {-46, -34, 45}
, {-35, -55, 66}
, {-17, -19, 55}
, {-56, 62, -36}
, {65, 2, -14}
, {22, 0, 5}
, {-37, 70, 33}
, {7, -65, 77}
, {45, 17, 61}
, {-66, 45, 30}
, {-15, -52, -72}
, {-3, 63, 38}
, {-28, -16, -40}
, {-3, -23, 24}
, {39, 18, 0}
, {-4, -43, -26}
, {-17, 64, 48}
, {40, -31, -2}
, {8, -69, 26}
, {69, 1, -7}
, {-39, 8, 65}
, {-6, 30, 62}
, {9, -24, 12}
, {19, 27, 51}
, {33, -18, 48}
, {53, 3, -65}
, {-58, -54, 62}
, {41, -44, 65}
, {47, 71, -37}
}
, {{48, 63, -44}
, {10, 31, 52}
, {0, -66, 31}
, {22, 68, -47}
, {55, -59, 48}
, {26, 14, 54}
, {-31, 2, 65}
, {47, -11, -68}
, {-43, -17, 60}
, {-28, 61, -46}
, {8, -74, -72}
, {-8, -23, 71}
, {72, -4, 14}
, {-35, -36, 16}
, {-2, -40, -27}
, {-56, 27, -69}
, {-6, 47, 75}
, {-71, 77, -62}
, {-70, -44, -18}
, {65, -24, -49}
, {-66, 26, 34}
, {-32, -41, 20}
, {0, -48, -61}
, {-13, 57, 23}
, {-68, 76, 45}
, {49, 59, 50}
, {-66, 33, -5}
, {58, -71, -36}
, {-24, 20, 11}
, {30, 65, -1}
, {-75, -2, 46}
, {70, -8, -9}
}
, {{22, -80, 78}
, {-54, 55, -10}
, {2, 32, 16}
, {6, 22, 63}
, {43, 62, -73}
, {26, 22, -59}
, {61, 86, 5}
, {14, 32, -29}
, {36, -73, -29}
, {35, -5, -56}
, {78, -71, 74}
, {34, -9, -24}
, {11, 18, 82}
, {47, 41, 41}
, {-12, -49, -4}
, {10, 74, 21}
, {77, -34, -60}
, {-58, 16, 44}
, {-48, -32, 49}
, {30, 15, -57}
, {-17, 37, -79}
, {7, 40, -33}
, {52, 54, -4}
, {26, -43, -12}
, {-30, -10, -51}
, {-78, -1, -84}
, {-1, 14, -27}
, {-54, -5, -31}
, {11, 36, -47}
, {-18, -45, 65}
, {-63, 0, -46}
, {71, -41, -42}
}
, {{39, 17, 64}
, {45, -39, 44}
, {-39, -78, 1}
, {-39, 15, 4}
, {6, 15, -11}
, {-31, -45, 57}
, {17, -19, -55}
, {5, -12, -27}
, {-22, 24, 52}
, {-38, 7, -38}
, {12, -72, 27}
, {3, 39, 1}
, {-36, 47, -70}
, {-24, -36, 55}
, {60, 63, 20}
, {-16, 9, 8}
, {-12, -61, -29}
, {-37, 25, 38}
, {-18, 23, 67}
, {52, -16, 39}
, {45, 32, -53}
, {46, -19, 43}
, {-56, -20, -58}
, {26, -18, -4}
, {-33, -39, 29}
, {26, -50, 37}
, {0, 51, -68}
, {-39, 25, 10}
, {74, 24, -38}
, {-32, -4, -19}
, {15, 64, 20}
, {34, 43, 61}
}
, {{-35, -37, -65}
, {-17, -28, -6}
, {21, -22, 87}
, {-17, -35, -52}
, {-34, 23, 2}
, {-62, -64, 37}
, {-36, -31, 39}
, {-22, 4, -13}
, {-37, -19, -41}
, {23, -44, 62}
, {1, 12, 70}
, {-39, 3, 19}
, {-43, -30, -57}
, {4, 76, 30}
, {72, 43, 56}
, {-40, 13, -48}
, {14, 46, 20}
, {-29, 51, -50}
, {-11, -16, -13}
, {-55, 26, 32}
, {28, -77, -37}
, {30, 37, 82}
, {28, -29, -26}
, {27, -20, -26}
, {28, 54, 60}
, {-60, -71, -23}
, {-18, -43, -56}
, {12, -64, 45}
, {4, -4, -2}
, {79, 77, 19}
, {-18, -62, -11}
, {-32, 79, 31}
}
, {{-29, -49, 57}
, {29, 41, 0}
, {-77, 8, -27}
, {-19, -26, -63}
, {-27, 39, -62}
, {79, 48, 12}
, {64, -25, -53}
, {-10, -64, 63}
, {-30, 71, 2}
, {-43, 22, 48}
, {-5, -40, 25}
, {36, 24, 52}
, {-75, -71, 44}
, {-22, 37, -23}
, {-57, 24, 60}
, {-32, 59, -49}
, {38, 71, -7}
, {49, -19, -13}
, {49, 23, -88}
, {53, -44, -35}
, {-50, -56, 21}
, {-52, 29, -64}
, {-42, -62, -2}
, {79, 12, -79}
, {-17, -57, 72}
, {-18, 51, 8}
, {3, 22, 9}
, {-19, 41, 30}
, {75, 35, 27}
, {-75, 56, 17}
, {53, 8, 43}
, {-24, 18, 11}
}
, {{27, 4, -28}
, {-38, 50, -32}
, {-72, -46, 70}
, {-64, 19, 34}
, {-77, -28, 57}
, {2, 48, 34}
, {12, 12, 34}
, {-55, -10, 45}
, {-17, -65, -4}
, {-47, 68, -21}
, {49, -44, -28}
, {-70, -86, 65}
, {-22, 32, 54}
, {38, -11, -1}
, {-32, 69, 31}
, {21, 13, 41}
, {-59, 2, -10}
, {25, 43, 2}
, {72, -18, -63}
, {49, 56, 14}
, {-71, -66, 18}
, {-48, -56, -5}
, {49, 21, 16}
, {15, 43, 75}
, {-68, -1, 0}
, {62, 57, 57}
, {41, 72, 28}
, {68, -3, -74}
, {-2, 15, 43}
, {-57, 27, -13}
, {49, 56, -66}
, {-42, 58, -11}
}
, {{-63, 81, -60}
, {-10, -23, 60}
, {-73, -1, 73}
, {-14, -10, 7}
, {-25, -38, -30}
, {0, 33, -23}
, {37, -32, 74}
, {40, -15, 6}
, {71, -50, 41}
, {-11, -25, 60}
, {-48, 82, -62}
, {-61, -27, -58}
, {57, 54, -70}
, {29, -68, -70}
, {41, 34, 26}
, {20, 44, -39}
, {-23, -77, 84}
, {26, -14, -84}
, {13, -4, 19}
, {-33, -77, 45}
, {73, 38, 70}
, {79, 81, -50}
, {24, -40, 26}
, {-38, -12, -63}
, {-36, -47, 10}
, {39, -65, 75}
, {6, 26, -48}
, {-40, -42, 69}
, {-72, 3, -5}
, {-24, 44, -12}
, {6, -42, -62}
, {-71, 0, -44}
}
, {{-26, -68, -54}
, {-51, 58, -64}
, {1, -75, 76}
, {-77, 26, -66}
, {13, -49, -71}
, {-87, 32, -62}
, {53, -57, -6}
, {-49, -8, -34}
, {-73, 43, 69}
, {32, -22, 47}
, {37, 26, 26}
, {0, 70, 28}
, {37, 57, 30}
, {-9, -71, 47}
, {-34, -1, -5}
, {73, -73, 63}
, {-39, -45, 59}
, {25, 33, -57}
, {34, -9, -36}
, {-61, -29, -26}
, {50, 27, 23}
, {29, 24, -51}
, {-46, -44, -57}
, {-23, 1, 17}
, {-25, -77, 47}
, {-4, 9, 23}
, {-104, -6, -51}
, {75, -61, 74}
, {87, 27, 34}
, {73, -78, 73}
, {-66, -26, 32}
, {-6, 52, -59}
}
, {{51, -30, 59}
, {27, -15, -24}
, {-46, 55, -45}
, {-2, -33, -51}
, {21, -24, -40}
, {28, -24, -30}
, {0, 67, 13}
, {-37, 44, 12}
, {-18, -74, -46}
, {-8, -37, -4}
, {-10, 21, 9}
, {-9, 2, 86}
, {59, -51, -63}
, {-19, 21, -36}
, {-11, -23, -31}
, {-13, 58, 9}
, {-14, 6, -9}
, {-18, 35, -10}
, {53, -38, -59}
, {66, 37, 56}
, {30, -60, 72}
, {11, -7, 68}
, {3, -51, 67}
, {-55, 11, 31}
, {66, -11, 36}
, {-6, -46, -28}
, {-16, -6, 46}
, {-17, 73, -38}
, {-15, -18, -27}
, {-34, 68, 17}
, {-57, -27, 15}
, {-28, 63, -69}
}
, {{-19, 17, 22}
, {-35, -3, 60}
, {-6, -2, 11}
, {-18, -35, -25}
, {-1, 67, -10}
, {-46, 80, 71}
, {-48, 44, -64}
, {-23, -32, 22}
, {45, -45, 44}
, {-15, 64, -55}
, {56, -40, 7}
, {33, -16, 1}
, {13, 63, -65}
, {71, 33, -70}
, {10, 15, -5}
, {47, 43, 1}
, {2, 46, 37}
, {-73, 37, -52}
, {-66, -58, -42}
, {-56, -57, 48}
, {-6, -20, 61}
, {-29, 61, -39}
, {-70, 72, 43}
, {8, 22, 0}
, {-2, -61, -66}
, {-64, -43, 22}
, {-60, 37, 77}
, {43, -5, 9}
, {23, -35, 62}
, {54, -39, 27}
, {-36, 77, 10}
, {2, -66, 26}
}
, {{-58, 29, -37}
, {-52, 17, 60}
, {-2, -88, -65}
, {-67, 51, 42}
, {26, 22, 5}
, {-27, 19, 52}
, {-25, -21, -28}
, {-66, 21, -37}
, {-25, -9, 10}
, {10, 27, -63}
, {54, 42, -34}
, {21, -58, 31}
, {-28, 28, 69}
, {38, -16, 20}
, {50, 41, 73}
, {-20, -26, 37}
, {-44, -58, -2}
, {19, -61, -43}
, {33, 73, 20}
, {-26, 77, 55}
, {0, -19, 53}
, {-49, -54, 22}
, {-52, 72, -10}
, {-66, 43, 44}
, {55, -41, -71}
, {-13, 11, 60}
, {-30, -26, 100}
, {47, -64, -6}
, {58, 2, -45}
, {27, -42, -84}
, {-2, 55, 57}
, {0, 26, -68}
}
, {{25, -63, 40}
, {-24, -47, -51}
, {-21, 27, 77}
, {69, -3, 18}
, {23, 48, -73}
, {-69, -15, -58}
, {-48, -36, 48}
, {77, -1, -29}
, {46, -24, 7}
, {-8, 29, 64}
, {-54, -41, -37}
, {22, -77, -32}
, {-45, 43, 38}
, {-2, 73, -44}
, {23, 38, 20}
, {-33, 17, 60}
, {2, 85, 15}
, {-56, -10, 4}
, {37, -15, 14}
, {-14, 45, 2}
, {7, 73, -15}
, {43, 1, 5}
, {59, 54, 50}
, {14, -79, 23}
, {-55, -39, -11}
, {57, 1, 26}
, {-2, -35, 6}
, {-65, 77, 32}
, {-75, 34, 91}
, {-8, 31, 53}
, {74, -55, -9}
, {-37, 18, -11}
}
, {{34, 12, -63}
, {-49, 44, 32}
, {62, -20, 60}
, {63, 23, 52}
, {-27, 49, 11}
, {46, -44, 79}
, {10, 31, 68}
, {73, -63, 61}
, {-60, -1, -8}
, {-53, -14, 63}
, {-20, 63, -51}
, {-20, -10, 46}
, {-34, -27, -25}
, {-87, 26, -17}
, {5, -39, 34}
, {-74, 80, -70}
, {-46, -58, 69}
, {-15, 7, -40}
, {55, 61, -58}
, {69, -37, -47}
, {-33, -22, 24}
, {27, 40, -40}
, {67, 0, 92}
, {46, -59, -56}
, {-42, -37, -69}
, {10, -42, -30}
, {28, -50, 12}
, {25, -56, -25}
, {-13, 12, -9}
, {-39, 76, -19}
, {26, 58, -9}
, {-14, 42, 1}
}
, {{-52, 30, 27}
, {-28, -3, -42}
, {-27, -34, -48}
, {32, 65, 26}
, {-25, -23, 44}
, {-3, -18, -67}
, {10, 34, -58}
, {-33, -53, 67}
, {-7, -16, 24}
, {-59, -15, 32}
, {68, 64, -17}
, {54, 59, -9}
, {39, 27, -21}
, {-70, -41, 10}
, {17, 21, 11}
, {39, 28, 34}
, {71, 62, 28}
, {-56, 15, -30}
, {-47, 23, -27}
, {75, 68, -5}
, {-62, 84, 24}
, {54, -39, 25}
, {52, 34, 9}
, {-52, -39, 28}
, {-56, -26, -22}
, {49, 35, -30}
, {62, -37, 75}
, {-50, -67, 37}
, {-4, 40, -66}
, {79, 34, 55}
, {0, -34, 13}
, {48, -37, -26}
}
, {{-39, -31, 63}
, {-75, 10, -61}
, {77, -48, 4}
, {63, -11, 34}
, {58, 61, 44}
, {-8, 46, -57}
, {-13, -35, 33}
, {-39, 22, 33}
, {-10, 83, -29}
, {-22, -62, -16}
, {75, 14, 41}
, {50, -30, -54}
, {-22, 10, 44}
, {21, -4, 24}
, {4, 22, 27}
, {11, -18, 40}
, {49, -23, 25}
, {12, 26, -48}
, {-21, 15, -57}
, {-1, -10, -25}
, {46, 51, -42}
, {-71, 41, 52}
, {-39, -42, 24}
, {31, 52, 27}
, {4, -63, 54}
, {-31, -10, 36}
, {64, -50, -43}
, {11, -48, 34}
, {72, -29, 11}
, {64, -59, -74}
, {46, -12, 43}
, {19, -16, -1}
}
, {{-72, -27, 31}
, {58, 41, -23}
, {-54, 65, -64}
, {-9, -69, 66}
, {19, 59, 3}
, {50, -38, 0}
, {21, -32, -37}
, {-48, -90, 75}
, {19, 67, 8}
, {-49, 25, -51}
, {1, 30, 53}
, {54, 68, 58}
, {4, 65, -76}
, {32, -52, 64}
, {7, -55, -3}
, {4, 17, 14}
, {48, 57, -33}
, {38, -51, -5}
, {8, 69, -50}
, {54, -21, 55}
, {68, 15, -78}
, {50, -57, -45}
, {68, 36, -18}
, {64, 30, -34}
, {35, 37, 41}
, {75, -56, -7}
, {77, 7, -66}
, {-71, 53, -57}
, {10, 48, -15}
, {35, 52, -59}
, {29, -16, -62}
, {23, -32, -40}
}
, {{27, 2, -11}
, {13, -17, 31}
, {64, 67, 58}
, {6, 19, 70}
, {14, -65, 49}
, {-29, -11, 13}
, {-49, 11, -6}
, {58, 56, -56}
, {35, -5, -38}
, {-46, -34, 58}
, {50, -70, 54}
, {26, 62, 4}
, {3, -32, 51}
, {-38, -79, -69}
, {-63, -35, 67}
, {-58, -59, 76}
, {-11, 28, -50}
, {-3, -61, 23}
, {43, -23, 9}
, {-66, 32, -53}
, {-76, -32, -4}
, {12, 6, 7}
, {45, 79, 66}
, {29, -19, -16}
, {-13, 19, 13}
, {-33, -29, 3}
, {-44, 66, 52}
, {21, 57, 40}
, {-88, 58, 13}
, {42, 1, -35}
, {-60, 72, 54}
, {-39, 13, -65}
}
, {{-58, -13, -18}
, {23, -14, -6}
, {-18, -61, -19}
, {24, -39, -71}
, {62, 0, 29}
, {-22, -17, 28}
, {-59, 34, 52}
, {-19, -62, 32}
, {-55, -8, -11}
, {47, 35, 16}
, {-23, -28, 45}
, {64, 17, -57}
, {-73, 10, -40}
, {72, -29, -38}
, {24, 56, 38}
, {-16, 71, -33}
, {56, 18, -65}
, {-57, -10, 66}
, {30, -22, -1}
, {57, 26, 15}
, {65, 27, 27}
, {-12, -47, -2}
, {51, 18, -55}
, {-50, -9, -33}
, {-46, 40, 59}
, {1, 63, 10}
, {1, 46, 32}
, {-42, -6, -17}
, {62, 6, -45}
, {-57, 47, -47}
, {24, -49, 72}
, {78, -68, 52}
}
, {{-16, -15, 13}
, {25, 13, 57}
, {43, 47, 47}
, {-24, 41, -4}
, {5, -13, 60}
, {25, -62, 51}
, {-69, -15, 66}
, {-55, 18, -40}
, {28, 40, -81}
, {69, -22, -7}
, {68, 71, -35}
, {35, 82, 19}
, {-3, 34, -19}
, {-77, -19, -14}
, {0, 5, 54}
, {-6, 22, -27}
, {-31, 38, -11}
, {-25, 40, 6}
, {0, -69, -71}
, {24, -63, 55}
, {-20, -62, 5}
, {26, -23, 60}
, {42, -16, -12}
, {-38, 22, -37}
, {33, 67, 16}
, {51, -63, -44}
, {-51, 28, -10}
, {-37, -9, -71}
, {-41, 77, -72}
, {30, -51, -12}
, {24, 59, -26}
, {-37, 38, 18}
}
, {{-55, -42, -5}
, {41, -69, -58}
, {21, -37, 20}
, {-34, -43, 66}
, {27, -48, 16}
, {72, -24, -75}
, {-1, -30, -34}
, {-38, 21, -42}
, {47, -72, -51}
, {8, -4, 62}
, {37, 51, 5}
, {38, 31, 12}
, {37, -23, 28}
, {-16, -42, 2}
, {-54, 71, -83}
, {-46, 3, 27}
, {6, 44, -35}
, {-15, 39, -15}
, {-51, 26, -5}
, {-24, -24, -52}
, {8, 49, 7}
, {-14, 74, 28}
, {24, -11, -41}
, {-60, 28, -55}
, {73, -15, 6}
, {5, 19, 25}
, {42, -53, 25}
, {-31, 40, 36}
, {-15, -39, 40}
, {67, -9, -23}
, {-44, 0, -7}
, {25, -18, 58}
}
, {{-1, 4, -69}
, {46, 12, 44}
, {0, 24, 20}
, {21, 17, -61}
, {2, 55, 3}
, {-4, -63, -68}
, {28, 33, 88}
, {-57, 32, 60}
, {59, -21, -71}
, {-23, -68, -50}
, {17, 7, 41}
, {65, -8, -25}
, {7, -22, -74}
, {-77, 32, 59}
, {-27, -78, 52}
, {-49, -64, 60}
, {42, 34, 19}
, {10, 13, -13}
, {43, 49, -17}
, {-2, 30, 42}
, {-1, -82, -23}
, {38, -68, 67}
, {77, 35, 60}
, {61, -74, 41}
, {-42, 42, 26}
, {5, 37, -20}
, {59, 0, 56}
, {15, -63, -45}
, {-48, 42, 48}
, {-66, -13, 5}
, {49, -67, -5}
, {-10, 47, 1}
}
, {{-33, -64, 69}
, {-41, -53, -1}
, {-8, 57, 6}
, {67, -81, -22}
, {-69, 57, -38}
, {54, 30, -54}
, {-29, -15, 51}
, {64, 13, 61}
, {13, -74, -21}
, {56, -72, 62}
, {-5, 10, -32}
, {-73, 11, -52}
, {29, -11, 64}
, {18, 46, -43}
, {-20, -14, -73}
, {71, 42, 53}
, {14, -31, -21}
, {17, -92, -57}
, {-72, -7, -20}
, {42, 66, -56}
, {34, 60, -33}
, {57, 51, 53}
, {-39, -10, 53}
, {73, 36, -78}
, {51, 73, 18}
, {-58, -27, -61}
, {63, -47, 8}
, {0, -21, -56}
, {-28, -27, 20}
, {51, 84, -41}
, {50, -23, -49}
, {53, -16, -31}
}
, {{-6, -23, -32}
, {-34, -2, 27}
, {70, 2, 15}
, {43, 36, 34}
, {-48, 45, 4}
, {-53, 43, -18}
, {-75, -4, -61}
, {40, -36, -66}
, {49, -53, 79}
, {54, 14, -45}
, {30, 22, -11}
, {41, -61, 52}
, {-51, 17, 74}
, {-57, -16, -59}
, {-50, 72, 47}
, {27, 4, -75}
, {-6, -71, 76}
, {-57, 16, -69}
, {23, -50, 85}
, {3, 52, 78}
, {-66, 44, 6}
, {-66, -51, 73}
, {-7, 69, -51}
, {34, 14, -1}
, {-39, 23, -56}
, {71, 18, 59}
, {17, 23, 60}
, {-26, 38, 44}
, {-55, -47, -48}
, {-35, -43, 78}
, {20, 70, 66}
, {-14, -11, 9}
}
, {{-32, -61, 68}
, {49, 70, 25}
, {40, -12, 15}
, {12, 12, 18}
, {80, 45, 56}
, {-2, 80, -5}
, {42, 37, -88}
, {-56, 48, 37}
, {-62, 61, 35}
, {12, -35, 40}
, {-6, -10, 8}
, {-5, -34, -36}
, {0, -31, -68}
, {-50, -55, -29}
, {14, 9, 56}
, {53, -66, 36}
, {30, 55, -56}
, {-22, 24, -63}
, {24, -28, 85}
, {15, -62, 79}
, {7, 69, 7}
, {-40, -48, -59}
, {-37, 25, 59}
, {17, 3, 7}
, {17, -44, -69}
, {6, -11, -30}
, {53, 54, -27}
, {-72, -63, 3}
, {21, 70, 62}
, {29, 5, -61}
, {-32, 79, -45}
, {-56, 48, -71}
}
, {{-32, -41, 17}
, {-34, -10, 64}
, {4, -21, -61}
, {-23, 62, -19}
, {-25, 82, -5}
, {-42, 12, 33}
, {23, -30, -65}
, {71, 34, 67}
, {17, 67, 69}
, {-35, 38, -26}
, {68, 19, -42}
, {-32, -66, 3}
, {22, -32, 25}
, {29, 20, 28}
, {67, 9, -26}
, {23, -52, 24}
, {-54, -42, -61}
, {71, 41, -52}
, {22, -62, 52}
, {7, 4, -26}
, {0, -33, -27}
, {-17, -49, 58}
, {19, -60, -46}
, {-56, 1, 65}
, {-15, 39, -40}
, {65, -48, 20}
, {-20, 31, 25}
, {18, 58, -66}
, {53, -56, -36}
, {27, 18, -3}
, {-57, 14, -24}
, {-35, -69, -54}
}
, {{-40, 53, 58}
, {-30, 40, 22}
, {33, -21, 48}
, {-41, -6, -26}
, {-60, 22, 7}
, {-80, -43, -70}
, {35, 55, 51}
, {37, 0, 46}
, {40, -39, 15}
, {60, -32, -60}
, {8, 70, 26}
, {-29, -8, 84}
, {70, -27, -37}
, {-11, 24, 34}
, {1, -24, -14}
, {-63, -1, 57}
, {-51, -41, -51}
, {-41, 8, -71}
, {-64, 32, -6}
, {-36, -24, -52}
, {-40, 41, 10}
, {45, 44, 77}
, {-62, 71, -9}
, {-63, -52, -41}
, {8, 70, 78}
, {-49, 62, -34}
, {-42, 28, -40}
, {46, -33, -32}
, {25, -34, 37}
, {30, -14, -61}
, {34, -24, -33}
, {-1, -9, -47}
}
, {{14, -33, 84}
, {5, 19, -12}
, {41, 28, 14}
, {25, -75, -23}
, {-68, 75, 13}
, {24, 58, -57}
, {3, -12, -24}
, {-10, 2, 17}
, {44, -50, 37}
, {74, 39, 54}
, {-6, -48, 49}
, {-34, -48, 45}
, {27, -22, -40}
, {20, -34, -60}
, {-7, 17, 68}
, {18, 34, 13}
, {-66, 65, -32}
, {0, -15, -18}
, {-45, -28, -15}
, {-15, 34, -26}
, {-63, 18, 5}
, {-49, 0, -57}
, {5, 8, 0}
, {60, 60, -23}
, {20, -35, -55}
, {15, -64, 31}
, {-15, 62, -13}
, {24, -59, 11}
, {-22, 8, -23}
, {-63, 0, 75}
, {-10, 7, -30}
, {-4, -3, 7}
}
, {{67, -37, 6}
, {15, 38, -19}
, {52, -14, -7}
, {62, 24, 67}
, {-70, -56, -1}
, {-40, 49, -74}
, {31, -9, 17}
, {-35, -14, 82}
, {-48, -67, 36}
, {15, 57, 44}
, {24, 39, 35}
, {79, 58, -52}
, {71, -30, -18}
, {-11, 70, -43}
, {33, 51, 60}
, {-15, 68, 14}
, {70, 43, 2}
, {59, 63, 32}
, {68, 71, 75}
, {70, 2, -56}
, {52, 67, -42}
, {-31, -67, 10}
, {-40, 25, 22}
, {-39, 19, -54}
, {59, -55, 50}
, {0, -9, -60}
, {4, 48, -64}
, {-40, -13, 17}
, {-52, -44, 29}
, {50, 39, -50}
, {-18, -62, 72}
, {-36, -55, 65}
}
, {{82, 0, 14}
, {-47, 62, 55}
, {81, 12, -74}
, {-21, -32, 7}
, {-56, 52, -31}
, {61, -9, -47}
, {-3, -28, 56}
, {-45, 53, 75}
, {50, -34, -14}
, {-45, -4, 7}
, {-22, 18, -56}
, {-50, -48, -36}
, {-16, -76, -67}
, {30, 57, 52}
, {-71, -15, -62}
, {-55, 64, -7}
, {6, -68, -12}
, {13, -38, 27}
, {-50, -34, 28}
, {12, -61, 69}
, {49, 77, -20}
, {-49, 8, 51}
, {-29, 44, 34}
, {-47, 8, 7}
, {39, 16, 19}
, {2, 73, -53}
, {66, 16, 44}
, {63, 22, -3}
, {29, -56, -43}
, {68, -64, 13}
, {-28, -26, -70}
, {-24, -37, 44}
}
, {{-43, 10, 14}
, {60, -28, 62}
, {1, 8, -27}
, {27, 15, 15}
, {-26, 1, -34}
, {-42, -10, -66}
, {-59, -33, 4}
, {47, 11, 18}
, {62, -56, 75}
, {16, 17, 12}
, {-71, 2, 0}
, {-61, -2, 35}
, {-47, 0, -37}
, {-54, 49, 47}
, {-44, 72, -2}
, {21, 18, 18}
, {57, 46, -69}
, {34, -42, 30}
, {28, 22, -11}
, {61, 47, 19}
, {13, 49, -46}
, {-6, 13, 49}
, {-54, 67, 60}
, {12, -79, -11}
, {67, -35, -7}
, {0, 53, -30}
, {-56, -15, 46}
, {-53, -70, -21}
, {-47, -13, 28}
, {-30, -26, -50}
, {41, 70, -59}
, {42, -51, 39}
}
, {{66, -46, 10}
, {-78, -83, 1}
, {67, 30, 54}
, {34, 6, 22}
, {-45, -1, 61}
, {0, -18, -29}
, {-42, -18, 20}
, {-31, -24, -32}
, {-36, -58, 0}
, {-66, -6, -50}
, {64, 85, 12}
, {35, -18, 9}
, {-6, 60, 82}
, {10, 74, -41}
, {-41, -61, 6}
, {19, 43, -6}
, {-23, 21, -71}
, {-22, -59, -58}
, {8, -46, 71}
, {20, -8, 52}
, {60, 14, 6}
, {6, 40, -24}
, {47, -45, -73}
, {80, -80, -59}
, {-3, 2, 51}
, {0, 25, 38}
, {85, -44, -44}
, {0, 17, -52}
, {34, -4, 31}
, {-41, 35, 1}
, {53, -36, -49}
, {28, 50, -41}
}
, {{66, -58, 13}
, {66, 73, -29}
, {56, 16, -43}
, {-48, 74, 4}
, {-57, 56, 21}
, {62, -47, 9}
, {-55, -19, -41}
, {65, 57, 61}
, {-66, 80, 35}
, {-28, -29, -17}
, {-61, 44, 40}
, {-6, 38, 14}
, {34, -56, 45}
, {-14, 5, -3}
, {-25, -36, -31}
, {-34, 58, 60}
, {-25, 62, 60}
, {49, 4, 55}
, {-49, -33, 62}
, {-66, -6, -61}
, {-21, 46, 84}
, {56, 66, -7}
, {-4, -72, -59}
, {69, 2, -1}
, {0, 21, -50}
, {76, -71, -34}
, {-56, -11, 37}
, {-49, -52, 33}
, {-58, 31, 42}
, {3, 47, -35}
, {-26, 18, 14}
, {-74, 20, -47}
}
, {{8, 71, 45}
, {26, 78, -50}
, {31, -72, 64}
, {-39, -33, -79}
, {52, -41, 0}
, {43, -38, 5}
, {-51, 4, 15}
, {53, 47, -18}
, {-17, 9, 23}
, {-1, -76, -37}
, {-79, -70, 41}
, {-10, 34, -32}
, {-59, 10, 4}
, {-40, 25, 17}
, {-49, -4, -6}
, {47, -30, -56}
, {37, 11, -46}
, {64, -6, 33}
, {50, 18, 4}
, {-40, 17, 65}
, {81, -40, -31}
, {8, -64, 0}
, {54, 27, -19}
, {65, 29, 23}
, {-30, -72, 66}
, {-41, -44, 46}
, {-16, -58, 53}
, {-24, -29, 28}
, {16, 74, 1}
, {-17, -45, 51}
, {42, 66, 56}
, {2, 0, 0}
}
, {{-64, -43, 10}
, {5, -56, 37}
, {46, -39, 25}
, {34, 4, 51}
, {44, -41, -21}
, {-14, -48, 38}
, {-11, 4, 57}
, {-37, 47, -34}
, {-41, 52, -19}
, {-29, -4, -53}
, {1, -40, -54}
, {35, -28, 69}
, {4, -13, -63}
, {72, -18, -7}
, {-60, 12, -43}
, {20, 24, -77}
, {11, 3, 34}
, {-27, -2, -56}
, {79, -79, -72}
, {-1, -1, 1}
, {35, -73, 72}
, {-77, -28, 47}
, {46, -60, -57}
, {-45, 12, -18}
, {66, 48, -42}
, {41, 52, 13}
, {-33, 21, -37}
, {61, 18, 51}
, {-40, 20, 28}
, {-14, 49, 31}
, {-46, 58, 51}
, {-53, 15, 60}
}
, {{-5, -14, -1}
, {88, -62, 15}
, {-53, 3, 57}
, {-38, 1, -15}
, {-38, 17, -18}
, {50, -59, -38}
, {42, 24, 63}
, {42, 23, 11}
, {68, 40, -39}
, {26, -36, -27}
, {-79, 7, -41}
, {51, -12, 79}
, {-44, -18, -31}
, {9, 56, 26}
, {-46, -67, 69}
, {-31, -19, 66}
, {70, -65, 32}
, {-32, -28, 59}
, {31, 65, -58}
, {30, -32, -56}
, {-43, 24, 6}
, {10, 72, -1}
, {30, -6, -38}
, {-2, -75, 15}
, {58, 59, 40}
, {-47, 16, 73}
, {-29, 27, 42}
, {-59, 75, -47}
, {-65, -15, -39}
, {30, 69, 14}
, {18, 28, 13}
, {-64, -34, -60}
}
, {{-5, 33, 23}
, {45, -27, -36}
, {72, 45, -22}
, {-44, -64, -28}
, {54, 4, 17}
, {36, -69, -36}
, {5, -61, 46}
, {74, 4, -5}
, {35, -48, -5}
, {-78, -73, -35}
, {55, 24, 33}
, {-30, 63, -68}
, {8, -13, -36}
, {26, -46, 2}
, {-2, 49, 48}
, {-56, 44, 72}
, {-52, 0, 35}
, {-1, -33, 11}
, {-61, 27, 53}
, {2, 44, -60}
, {22, 14, -31}
, {-50, 77, 17}
, {53, -24, 47}
, {-6, -55, -31}
, {31, 27, 4}
, {-58, -28, -31}
, {33, 20, -33}
, {-6, -21, -23}
, {-12, 26, -65}
, {55, 66, -7}
, {58, 1, -12}
, {64, 48, 48}
}
, {{13, 57, 20}
, {-53, -70, 2}
, {-42, 27, -81}
, {-21, -19, 64}
, {47, 0, 15}
, {19, 41, 25}
, {58, -20, -11}
, {-80, 32, 9}
, {-6, -76, -24}
, {9, 45, -11}
, {40, 28, 39}
, {73, 73, 45}
, {19, -32, 18}
, {11, 72, 0}
, {-78, 37, 50}
, {37, 45, -85}
, {-39, -24, -61}
, {-48, 44, 1}
, {8, -15, 22}
, {24, -2, 68}
, {-24, -49, 31}
, {25, -18, -55}
, {-78, 14, -37}
, {-63, 63, 12}
, {-41, 3, -49}
, {-68, 18, 49}
, {7, 28, 78}
, {62, -61, 25}
, {-1, 53, 67}
, {6, 50, 0}
, {58, -27, 41}
, {-6, -27, -28}
}
, {{-17, 44, 76}
, {-66, 0, 71}
, {37, 3, -2}
, {-72, 16, 16}
, {40, -15, 13}
, {-73, -28, 67}
, {39, -8, -36}
, {15, 30, -10}
, {-49, -25, 42}
, {23, 64, -60}
, {29, 49, 61}
, {-63, 1, 33}
, {64, -46, -17}
, {-44, 59, 22}
, {-72, -44, 63}
, {43, -38, -30}
, {-41, -35, 2}
, {17, 15, -76}
, {-20, -31, -25}
, {27, -25, -34}
, {47, 18, -4}
, {20, 9, -56}
, {-50, 16, -19}
, {-3, -26, 55}
, {-40, 39, -12}
, {-23, 33, -51}
, {15, 60, -3}
, {26, -32, 0}
, {21, -66, 20}
, {7, -86, -18}
, {5, 63, 58}
, {-25, 40, 22}
}
, {{-1, -66, -16}
, {-55, -52, 0}
, {9, 20, -23}
, {-34, -23, 31}
, {47, -61, -47}
, {68, -41, 12}
, {32, -20, -80}
, {68, 7, 7}
, {62, 10, 78}
, {-45, 10, -58}
, {52, -59, -32}
, {-31, -52, -29}
, {-35, -5, -40}
, {-5, 27, 18}
, {7, -56, 3}
, {-28, -4, 28}
, {-25, -10, 58}
, {13, -1, 47}
, {-5, 64, 46}
, {-13, 47, 48}
, {85, -68, 0}
, {51, 13, 53}
, {-26, -8, -22}
, {-56, -43, 42}
, {57, 11, -35}
, {-37, 28, -16}
, {-41, 23, -37}
, {-4, 52, -64}
, {83, 63, 84}
, {12, 3, -29}
, {-8, 1, -39}
, {-54, 5, 79}
}
, {{51, -36, 73}
, {47, -38, -53}
, {20, 60, -18}
, {-7, 46, -49}
, {18, -53, -10}
, {-53, -20, -9}
, {-3, 24, -3}
, {-40, 8, 67}
, {47, 51, 2}
, {18, -65, -62}
, {57, 23, 26}
, {-22, -7, -53}
, {26, -32, 67}
, {20, 67, 53}
, {57, -20, 36}
, {4, 29, 41}
, {-44, -54, 68}
, {-50, 28, -24}
, {12, -69, -67}
, {-52, -4, 53}
, {37, -64, 37}
, {-29, -24, 3}
, {63, 63, 26}
, {-64, -21, -30}
, {-12, -39, -19}
, {0, 46, -41}
, {-7, 59, -18}
, {-30, 42, 30}
, {-28, 20, -44}
, {9, -29, 55}
, {-35, -55, 34}
, {13, 19, -25}
}
, {{19, 5, -73}
, {23, -31, -30}
, {49, -61, -63}
, {-60, -22, 31}
, {-47, -54, -29}
, {75, 68, 50}
, {-71, 0, -8}
, {2, 17, 34}
, {72, 27, -68}
, {33, 46, -13}
, {-54, -49, -29}
, {57, 7, -32}
, {17, -65, -69}
, {-57, -72, 14}
, {61, -52, -41}
, {26, -63, -25}
, {30, 80, 33}
, {-31, 56, -1}
, {42, -10, 10}
, {-51, -7, -14}
, {-22, 27, -68}
, {50, 19, 34}
, {25, 6, 7}
, {1, 39, 75}
, {-30, -73, 53}
, {-70, 76, -37}
, {59, -39, -36}
, {-29, 7, -19}
, {-60, 62, -51}
, {12, 22, -83}
, {73, 40, -35}
, {10, 63, 46}
}
, {{-8, 56, -87}
, {-8, -31, -30}
, {-10, -12, -72}
, {-33, 65, 5}
, {22, -48, 72}
, {26, 18, -33}
, {78, 8, -22}
, {7, 28, 18}
, {-56, -55, 0}
, {-75, 27, 7}
, {-64, 0, 47}
, {41, -52, -43}
, {64, -9, 26}
, {-61, -24, -75}
, {-7, 31, -36}
, {-14, -31, -9}
, {27, 4, -1}
, {-59, -28, 53}
, {-60, -50, 0}
, {-2, 19, 0}
, {61, 62, 88}
, {62, 2, -16}
, {-16, 25, 45}
, {-43, 74, -20}
, {70, -58, 42}
, {27, -35, 80}
, {22, -37, 0}
, {-53, -13, -72}
, {63, 64, -22}
, {-68, 12, -55}
, {36, 0, 31}
, {62, 50, -40}
}
, {{-4, -37, 30}
, {7, 81, -21}
, {-81, -69, 41}
, {-7, -4, -62}
, {45, -18, 73}
, {58, 57, -7}
, {60, 25, 31}
, {67, -73, 59}
, {28, 11, -7}
, {-22, -41, 61}
, {-73, 26, 29}
, {-15, 15, 23}
, {-43, 44, 49}
, {-13, -73, 4}
, {-45, -72, -14}
, {69, 2, -60}
, {-53, 43, 69}
, {0, 23, -75}
, {-29, -55, 31}
, {-53, 70, -60}
, {51, 37, 49}
, {-33, 26, -69}
, {77, -1, -51}
, {-31, 13, -59}
, {-45, -48, 63}
, {-6, -67, 58}
, {-26, 90, 0}
, {59, -34, -37}
, {-24, -9, -41}
, {-18, -45, -24}
, {7, 41, -11}
, {14, 7, 39}
}
, {{-37, 54, -57}
, {-29, -62, -20}
, {-35, 40, -33}
, {67, -30, 33}
, {-7, -24, -5}
, {-10, -34, 21}
, {76, 54, -13}
, {60, 69, -40}
, {-9, 28, 7}
, {33, 25, 4}
, {17, 8, 37}
, {-38, 48, 33}
, {50, -23, 63}
, {26, 0, -1}
, {56, -81, 7}
, {71, 32, -59}
, {17, -71, -1}
, {-53, 15, 66}
, {60, -21, 11}
, {-54, 61, 46}
, {-54, -61, -51}
, {-28, 2, 49}
, {30, -47, -11}
, {35, -41, 60}
, {-22, 49, 12}
, {5, -73, 29}
, {1, 15, -19}
, {-47, 69, 7}
, {-77, 17, -16}
, {7, 51, -48}
, {-34, -71, -18}
, {35, -67, 49}
}
, {{-62, 10, 45}
, {8, 60, 27}
, {49, -57, 6}
, {-2, -48, 60}
, {-66, 39, 7}
, {17, -45, -18}
, {67, 55, 33}
, {-38, -24, -51}
, {5, -42, -22}
, {67, 36, 2}
, {53, -41, 15}
, {50, 43, 22}
, {-57, 70, -64}
, {-50, -43, -47}
, {80, 38, -41}
, {71, -17, -2}
, {31, -58, -24}
, {-71, 17, -56}
, {-42, 40, 76}
, {5, -47, -54}
, {26, -20, 40}
, {58, -2, 49}
, {-57, 41, 20}
, {-71, 80, -54}
, {-52, 49, 55}
, {-55, 66, 8}
, {31, -4, 41}
, {-11, -50, 52}
, {42, -15, 62}
, {60, 46, 66}
, {66, -12, 48}
, {-48, 24, -13}
}
, {{39, 21, -28}
, {62, 2, -6}
, {-54, 21, 56}
, {60, -40, 0}
, {35, 35, 20}
, {-22, 30, 60}
, {19, 42, -18}
, {38, 5, 22}
, {73, -34, 80}
, {-36, -22, -62}
, {-54, 29, -28}
, {37, -11, 50}
, {-28, -58, 46}
, {69, -1, -68}
, {-60, 30, -50}
, {-40, 39, -21}
, {28, 71, 38}
, {3, 2, -41}
, {20, 30, 48}
, {-58, -5, 12}
, {-51, -13, 40}
, {-27, -3, 17}
, {68, -52, -18}
, {0, -9, 7}
, {47, -37, 21}
, {-60, -60, -57}
, {45, -41, -70}
, {66, 47, 59}
, {26, -12, -26}
, {-26, 40, 3}
, {51, -43, 32}
, {-29, 66, -13}
}
, {{-64, 59, -60}
, {29, 67, 11}
, {-11, -59, -78}
, {40, -15, 36}
, {-71, 58, 12}
, {-40, -27, 38}
, {53, 56, 54}
, {-45, 22, -60}
, {-28, -46, 36}
, {-4, -7, 60}
, {73, 42, 36}
, {-75, -56, 17}
, {64, -53, -21}
, {7, -51, 2}
, {15, -60, -58}
, {36, -69, 7}
, {-45, 33, -59}
, {47, 51, -78}
, {-28, -32, 65}
, {38, -75, 52}
, {-61, 61, 27}
, {42, 74, -5}
, {-39, 69, -11}
, {38, -23, 62}
, {27, 18, -11}
, {29, 58, 11}
, {-80, -2, 57}
, {0, 51, -33}
, {55, -19, 46}
, {17, 33, -78}
, {47, 4, -35}
, {-15, 34, -15}
}
, {{47, 35, 29}
, {31, -18, 53}
, {44, -1, 1}
, {73, 8, -17}
, {-49, -18, 10}
, {38, 71, -5}
, {42, -61, -69}
, {-26, -66, -65}
, {46, 47, 2}
, {-53, 67, 18}
, {-81, 17, -84}
, {-55, -10, -11}
, {55, -17, -56}
, {-24, -24, 6}
, {-37, -70, 65}
, {-73, -41, -18}
, {-29, -1, 36}
, {21, -12, -18}
, {-4, 40, -42}
, {68, -62, 72}
, {55, -10, -52}
, {43, 34, -72}
, {-26, 42, -49}
, {42, 50, 19}
, {-1, 51, 47}
, {3, -46, 39}
, {-13, 42, 37}
, {51, 53, -47}
, {-34, 18, 65}
, {50, 3, -54}
, {-27, -56, -57}
, {-16, 70, 15}
}
, {{46, -47, 22}
, {-13, 20, 32}
, {-42, -10, 6}
, {13, 4, 52}
, {14, 26, -6}
, {24, 37, 63}
, {66, -33, 50}
, {72, -4, -41}
, {29, 77, 27}
, {-76, 9, -16}
, {57, -51, 18}
, {28, -41, 62}
, {42, -54, 17}
, {47, 38, -23}
, {-57, -33, 37}
, {-14, 35, -29}
, {-70, 2, 62}
, {-5, -67, 29}
, {-37, -40, -49}
, {52, 81, 7}
, {-70, 31, -40}
, {-5, -58, -54}
, {2, -13, -9}
, {-54, -38, -53}
, {-50, 0, 26}
, {-55, 59, 39}
, {21, -81, 22}
, {-52, 62, -35}
, {15, -1, -10}
, {9, -74, -49}
, {-19, 55, 15}
, {10, 66, -50}
}
, {{-45, -51, 2}
, {69, -86, 37}
, {9, -4, -51}
, {-24, -56, 34}
, {-64, -11, -68}
, {-22, 12, -48}
, {18, 2, 79}
, {3, 40, 25}
, {79, 2, -77}
, {-46, -26, -31}
, {-5, -16, 15}
, {2, -67, 65}
, {36, -44, 58}
, {-35, -4, -32}
, {70, -39, -1}
, {9, 10, 44}
, {-47, -75, -18}
, {-59, -46, 7}
, {51, 62, -73}
, {16, 13, -42}
, {-18, 6, 65}
, {-10, 34, 69}
, {12, -8, -7}
, {32, 2, -30}
, {-78, 39, 69}
, {-61, 25, 11}
, {67, -32, 31}
, {-42, -26, 24}
, {-68, -16, 72}
, {-54, -53, -67}
, {38, -25, 35}
, {-70, 66, 46}
}
, {{-14, -71, -11}
, {7, 3, 35}
, {-16, -14, 52}
, {6, 46, 70}
, {27, -33, -38}
, {2, 21, -1}
, {-6, 20, 45}
, {-50, 0, 69}
, {77, 23, 15}
, {64, 72, -10}
, {15, 73, -30}
, {67, -26, 6}
, {55, -16, -62}
, {-57, -70, 12}
, {-37, 23, 64}
, {-4, 17, 36}
, {24, -41, 2}
, {40, -67, -51}
, {-57, -36, -8}
, {36, -45, -52}
, {60, -76, -23}
, {-24, 15, 61}
, {29, -22, -46}
, {25, 59, 0}
, {4, -41, -66}
, {14, 42, 4}
, {3, -68, 34}
, {-54, 22, 8}
, {50, 28, -36}
, {67, 57, 40}
, {43, 53, 74}
, {7, 40, 64}
}
, {{-61, -70, -66}
, {50, -42, 15}
, {-60, 72, -14}
, {-15, -56, -61}
, {16, 62, -48}
, {42, -7, -65}
, {-59, 70, 86}
, {-33, -66, 8}
, {-42, -11, -24}
, {15, -16, 23}
, {-58, 4, 6}
, {-43, 82, 24}
, {-27, -41, 5}
, {-63, 28, -7}
, {58, -81, -17}
, {33, -15, -31}
, {-65, 73, -61}
, {85, -64, 17}
, {-3, -32, -44}
, {68, 71, 39}
, {67, -62, 41}
, {47, 31, 62}
, {-39, -73, -7}
, {86, 8, -32}
, {-38, -26, 72}
, {-23, 44, -32}
, {29, 15, -68}
, {55, 77, 72}
, {65, 24, -22}
, {28, 68, -15}
, {-40, 49, -51}
, {-40, -51, -18}
}
, {{37, 24, 19}
, {71, 3, -10}
, {43, 0, 65}
, {76, -29, 0}
, {-63, -13, 29}
, {-65, -62, 44}
, {51, -64, 8}
, {64, 32, -31}
, {-72, 42, 13}
, {-8, -1, 4}
, {-70, 77, 65}
, {-2, -5, 17}
, {21, 10, 48}
, {28, -64, -7}
, {77, -17, 29}
, {-62, 70, 57}
, {-34, 28, 30}
, {-31, -81, -34}
, {-44, 44, -49}
, {-19, 4, -45}
, {-64, 26, 70}
, {-52, 52, -31}
, {47, -26, 45}
, {67, -14, 8}
, {43, 34, -5}
, {0, -29, 56}
, {43, -76, -13}
, {-19, 69, 9}
, {70, -40, -76}
, {54, -4, -54}
, {62, -76, 30}
, {35, -50, 9}
}
, {{-17, -21, -52}
, {12, -54, -65}
, {65, 38, -12}
, {4, -19, -39}
, {71, -46, -20}
, {-39, 15, 7}
, {79, -15, 51}
, {74, -29, 10}
, {-77, -13, 17}
, {-54, 12, -38}
, {3, 40, -38}
, {35, -49, -29}
, {-6, -55, 35}
, {51, -8, 45}
, {67, -34, 56}
, {29, 19, -12}
, {-7, -39, -46}
, {36, 11, 7}
, {52, 38, -36}
, {-5, -66, 58}
, {31, -26, 39}
, {-6, -59, 33}
, {12, -39, 55}
, {2, 46, 62}
, {-41, -23, -30}
, {78, 17, 28}
, {-7, 66, 55}
, {65, -49, -6}
, {35, 46, 80}
, {-37, -56, -52}
, {-17, 59, -18}
, {-48, -55, -34}
}
, {{-49, -52, 12}
, {-11, -65, -13}
, {-22, -62, 59}
, {47, -40, -13}
, {61, -21, 41}
, {-26, 52, 49}
, {-54, 52, 0}
, {45, 35, -64}
, {63, -48, 54}
, {53, 45, 3}
, {0, -22, 62}
, {39, -57, -22}
, {-2, -55, 6}
, {-85, 2, -39}
, {25, 15, 46}
, {-55, 59, -28}
, {76, -18, -63}
, {52, -54, -69}
, {42, 14, 55}
, {19, 25, 33}
, {33, 46, -12}
, {27, 57, -71}
, {56, -62, -67}
, {23, -13, 12}
, {-29, 13, 28}
, {-14, -25, -13}
, {41, -57, -8}
, {74, -26, -50}
, {69, -62, 46}
, {65, -16, -24}
, {-71, -60, -59}
, {4, -14, 70}
}
, {{-17, 53, -39}
, {29, 16, -49}
, {-35, 52, 10}
, {-5, -32, 56}
, {-35, 6, 62}
, {60, 30, -23}
, {-78, 38, -91}
, {-64, -51, -13}
, {-23, -39, 49}
, {33, 52, -38}
, {50, -42, -30}
, {20, -59, -49}
, {65, 18, -54}
, {-10, -31, 43}
, {1, 49, -7}
, {-41, 39, -35}
, {1, 49, 23}
, {-19, -38, -18}
, {70, -7, -40}
, {-65, 88, -73}
, {48, 79, 26}
, {69, -45, -45}
, {10, 83, -4}
, {-22, -2, 86}
, {-68, -20, 27}
, {26, 48, -66}
, {45, 36, 29}
, {-25, -54, 21}
, {-15, -39, -62}
, {-36, 66, 50}
, {-46, 68, 23}
, {79, -44, -56}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE