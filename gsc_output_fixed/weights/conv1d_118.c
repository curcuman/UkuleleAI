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


const int16_t conv1d_118_bias[CONV_FILTERS] = {8, 11, 15, -4, 12, 10, 0, -6, 10, -13, 10, 6, -1, 21, 19, 10, 8, 11, -1, 16, -1, -6, -20, -4, 16, 21, 22, -3, 14, 0, 18, 15, 6, 11, -10, 14, -3, -1, 0, 24, 4, 0, -18, 8, 11, -10, 0, 36, 23, 6, 0, 14, 0, 1, 4, 26, 4, 14, 15, -18, -7, -12, -15, 9}
;

const int16_t conv1d_118_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{19, -55, -41}
, {64, 68, 77}
, {-62, -1, -7}
, {20, 41, -4}
, {-51, 86, 40}
, {49, 35, -54}
, {-82, 59, -52}
, {-66, 47, -4}
, {37, -67, 48}
, {12, -41, -1}
, {68, 1, -35}
, {-71, 31, 28}
, {40, -37, 49}
, {-60, 57, 52}
, {-15, 44, 46}
, {60, 63, 18}
, {60, -32, -74}
, {-48, 75, 58}
, {-57, -21, -3}
, {59, -9, 41}
, {-23, -4, -65}
, {36, 47, -50}
, {26, -44, -25}
, {-25, -65, 72}
, {-74, 70, 59}
, {-48, 11, -64}
, {33, -63, -9}
, {52, -63, 39}
, {22, -19, -41}
, {28, -38, -52}
, {57, 9, 47}
, {62, -62, 71}
}
, {{36, 19, 43}
, {72, 42, 59}
, {8, -19, 59}
, {-40, -61, -32}
, {8, -12, 14}
, {-73, -43, -61}
, {46, -32, -59}
, {18, -21, 19}
, {20, 13, 64}
, {-21, -45, 62}
, {65, 11, 16}
, {70, 9, 45}
, {52, -18, -25}
, {54, 31, -64}
, {36, 28, -31}
, {3, 67, 69}
, {-69, 54, -65}
, {-19, 68, -77}
, {-46, -18, 20}
, {7, -14, 19}
, {11, -22, -35}
, {-39, -8, 34}
, {-4, 4, 58}
, {61, -55, -43}
, {65, 4, 46}
, {74, 75, -33}
, {59, -46, 53}
, {-40, -73, -2}
, {-30, -23, -71}
, {58, 12, 60}
, {40, 69, 12}
, {62, -75, -36}
}
, {{41, 52, 66}
, {-4, -8, 41}
, {-21, 77, -26}
, {62, 39, -91}
, {14, -23, -52}
, {3, 53, -28}
, {-22, -6, 53}
, {32, 30, -43}
, {21, -2, 70}
, {9, -50, -9}
, {-32, -39, 38}
, {-53, -1, 7}
, {57, -24, 44}
, {-48, 44, 4}
, {-36, 12, 18}
, {24, -62, -36}
, {-31, 6, -52}
, {52, 12, -56}
, {-50, -49, 30}
, {25, -47, 0}
, {-40, -1, -26}
, {20, -39, 20}
, {10, -38, 40}
, {27, -4, -62}
, {43, 7, 52}
, {-46, -45, 33}
, {72, 33, 16}
, {72, 61, -48}
, {64, -4, -51}
, {-24, -72, 65}
, {1, -23, -3}
, {-24, -20, -52}
}
, {{-1, 66, -6}
, {27, 23, 31}
, {64, -32, -68}
, {-39, 50, -62}
, {-13, -23, 39}
, {10, 41, 12}
, {29, 58, 45}
, {-77, -74, 2}
, {8, -60, 2}
, {16, -31, -13}
, {47, 62, 29}
, {71, -17, -4}
, {46, -55, 6}
, {32, 22, -46}
, {7, 43, -30}
, {-73, -69, 47}
, {15, 33, 52}
, {-40, -20, 0}
, {-67, -75, 61}
, {40, -30, -39}
, {67, 46, -9}
, {-80, -45, -25}
, {27, 40, -47}
, {-36, -49, -42}
, {-33, 40, 37}
, {-57, -56, 71}
, {23, 28, -68}
, {-47, 7, 0}
, {37, -64, 56}
, {24, -34, 62}
, {-19, 68, -25}
, {-2, -58, -55}
}
, {{2, 44, -12}
, {-59, -19, 23}
, {16, 57, -44}
, {-74, -56, -48}
, {57, 43, 33}
, {63, -31, -34}
, {-23, 21, 6}
, {43, 8, 9}
, {37, -51, 32}
, {30, 5, 5}
, {53, 50, -4}
, {13, 0, 27}
, {13, 46, 27}
, {39, -9, -50}
, {0, 45, -43}
, {-40, -77, 41}
, {-8, 0, -59}
, {-42, -45, -79}
, {-45, 22, 18}
, {63, -74, -16}
, {64, -5, -16}
, {-59, 52, 36}
, {-69, -59, -63}
, {54, -70, -58}
, {32, 8, -76}
, {-69, -55, -39}
, {58, -12, -3}
, {-30, -2, 3}
, {-17, -37, 14}
, {80, 75, 40}
, {19, -66, 64}
, {-36, -39, -8}
}
, {{-61, 61, -25}
, {7, -40, -62}
, {-36, -19, -30}
, {-25, 13, 24}
, {-65, -35, 47}
, {-45, 19, 57}
, {25, 7, -34}
, {-67, -24, -71}
, {-63, 70, 31}
, {58, 11, 78}
, {-51, -61, 56}
, {33, 29, 0}
, {74, 0, -34}
, {-4, 14, 48}
, {8, -57, 52}
, {-13, 53, 14}
, {-51, 28, -18}
, {-22, -62, -30}
, {-21, -15, 4}
, {-40, 76, 76}
, {-1, 66, 17}
, {-21, 60, -14}
, {-16, -46, 62}
, {39, 34, -37}
, {79, 24, -19}
, {-31, 30, 22}
, {15, 64, 27}
, {-46, 62, 47}
, {74, -51, 57}
, {-69, -4, -2}
, {11, 4, 69}
, {1, -39, 14}
}
, {{-34, -68, -64}
, {38, 29, 30}
, {70, 68, 74}
, {29, 49, -23}
, {-44, 49, -29}
, {-36, 45, -71}
, {48, -25, -71}
, {-23, -56, 33}
, {18, -52, -73}
, {28, -15, -4}
, {64, -9, -27}
, {61, 66, 57}
, {-65, 19, 41}
, {-39, -4, 61}
, {-73, 28, -39}
, {-9, -34, 63}
, {62, -23, 57}
, {22, -47, 14}
, {-15, 11, -39}
, {24, -19, 26}
, {-51, -47, -29}
, {-68, 33, -65}
, {-36, 19, -62}
, {50, -3, 17}
, {-69, -43, 69}
, {-47, 15, -58}
, {64, 64, 59}
, {-64, 15, -20}
, {50, -23, 0}
, {71, -39, -1}
, {2, 21, -47}
, {44, -1, -28}
}
, {{22, -48, -54}
, {-25, 49, 47}
, {49, -30, 53}
, {-7, 28, 50}
, {-20, 5, -61}
, {23, -12, -60}
, {2, 27, -63}
, {75, 15, 54}
, {45, 20, -1}
, {20, -11, -36}
, {62, 38, -21}
, {69, -51, -69}
, {-48, 10, -33}
, {39, 24, -22}
, {-3, -30, -21}
, {34, 33, 23}
, {23, 39, 2}
, {51, -50, -53}
, {-28, -79, 82}
, {65, -42, -29}
, {-47, -56, -4}
, {51, -33, 40}
, {-45, 67, -62}
, {70, -65, -54}
, {-64, -66, -62}
, {18, -74, -39}
, {-29, 8, 42}
, {28, -79, 46}
, {21, 26, 1}
, {74, -18, 35}
, {-24, 20, 69}
, {-13, 0, -36}
}
, {{-4, 8, -63}
, {63, 81, -70}
, {69, -69, 41}
, {-43, 57, 85}
, {-38, 15, -41}
, {-50, -53, -69}
, {-75, -42, -71}
, {-43, 38, 31}
, {28, -26, 72}
, {0, -65, -32}
, {16, 52, -19}
, {43, 15, 12}
, {-57, 60, -56}
, {4, -29, -8}
, {37, 38, 60}
, {66, -41, -21}
, {-36, 9, 20}
, {43, 1, -34}
, {12, 31, 73}
, {-36, -23, 54}
, {-8, -17, 53}
, {-64, -9, 27}
, {5, -8, -29}
, {-24, 47, 3}
, {-33, -70, 1}
, {-72, -71, 38}
, {-9, 2, 40}
, {65, 55, -35}
, {-53, -73, 34}
, {57, 41, 21}
, {-23, 13, -11}
, {40, -22, 12}
}
, {{67, -6, -16}
, {-81, -69, -12}
, {-27, 44, -50}
, {-53, -27, -7}
, {25, -28, 50}
, {-61, -47, 67}
, {58, 50, 44}
, {56, 34, -1}
, {-47, -9, 26}
, {-42, 81, -57}
, {-3, -12, -23}
, {-12, -61, -9}
, {-49, -8, 75}
, {70, -8, -49}
, {-47, -62, -59}
, {-41, -44, -32}
, {-43, -4, -42}
, {17, -27, 16}
, {15, -7, -70}
, {-84, 80, -2}
, {-30, -70, -10}
, {-8, 30, -28}
, {18, 13, 20}
, {75, 14, 45}
, {-11, -38, 28}
, {59, 52, -20}
, {-7, -54, -25}
, {35, -13, -22}
, {-40, 26, 43}
, {-26, 51, -3}
, {37, 27, -39}
, {15, -60, -58}
}
, {{-4, -21, -5}
, {-10, 79, 72}
, {69, -59, -32}
, {6, -20, -29}
, {27, 83, -5}
, {27, -48, -65}
, {-30, -1, -53}
, {13, -77, -49}
, {-53, -88, 31}
, {26, -47, 17}
, {-34, 33, -3}
, {-69, -36, -53}
, {31, -24, 30}
, {15, 15, -35}
, {-81, -59, 40}
, {38, 56, -75}
, {-26, -69, -64}
, {31, 37, 11}
, {56, 51, -43}
, {39, 54, 63}
, {36, 31, 61}
, {49, -71, -67}
, {-15, 6, 24}
, {42, 19, 68}
, {-33, -32, -15}
, {39, -27, 82}
, {87, 16, -15}
, {-63, 83, -48}
, {41, 39, 4}
, {24, -22, -66}
, {4, 26, 75}
, {-55, 56, -29}
}
, {{-28, 3, -38}
, {-67, 0, 8}
, {57, -24, 35}
, {4, -39, -9}
, {5, -29, 79}
, {-71, 40, -9}
, {43, -44, 23}
, {-14, 16, -36}
, {-45, 13, 35}
, {25, 0, -33}
, {75, -2, 52}
, {-69, -28, 41}
, {68, 18, -56}
, {-61, -51, 51}
, {-58, 1, 28}
, {62, -5, -67}
, {48, 42, 36}
, {75, 29, 71}
, {-47, -61, -26}
, {-60, -50, 46}
, {21, -49, -48}
, {-31, -12, 6}
, {-23, -61, 44}
, {20, 24, 20}
, {42, -66, -12}
, {6, 30, 32}
, {-16, 34, 71}
, {39, 16, 28}
, {-46, 21, 8}
, {74, -17, 14}
, {-36, -50, 39}
, {-27, -52, -7}
}
, {{-54, 35, -4}
, {-64, -2, 34}
, {66, 7, -8}
, {-53, 0, -16}
, {36, -52, 5}
, {-60, 4, -55}
, {13, 66, 14}
, {1, 52, -17}
, {-31, 13, 50}
, {-50, -53, 10}
, {25, -19, -20}
, {-28, 7, -37}
, {-69, -56, 53}
, {-61, -36, -54}
, {-43, 31, -66}
, {-63, -10, 24}
, {43, 39, 32}
, {59, -35, -42}
, {-66, -2, -74}
, {-70, -64, 39}
, {-63, -32, 0}
, {44, -27, 53}
, {-15, 26, -58}
, {11, -38, -60}
, {70, -44, 58}
, {55, 0, -11}
, {-41, -28, -66}
, {-48, -6, 27}
, {2, -17, -47}
, {-50, -43, 1}
, {-38, -52, -21}
, {-72, 66, 25}
}
, {{9, 44, 31}
, {36, 29, -28}
, {-52, -57, 61}
, {-51, 21, -33}
, {59, -13, 3}
, {-59, -62, -37}
, {53, 42, 14}
, {-45, -28, 69}
, {33, 45, -22}
, {34, -26, -13}
, {-13, 33, 19}
, {-24, 51, -40}
, {-34, 36, -42}
, {64, -44, -55}
, {23, 38, -24}
, {-40, -31, 32}
, {44, 37, 77}
, {-6, -13, 40}
, {58, 38, 41}
, {-59, -42, -72}
, {8, -29, -54}
, {-10, -47, 15}
, {24, 23, 14}
, {32, 1, 0}
, {20, -67, -35}
, {-59, 5, -3}
, {47, -25, -45}
, {-67, -55, -37}
, {14, 22, 18}
, {55, -61, 39}
, {-26, -48, -58}
, {36, -46, -61}
}
, {{-55, -53, -74}
, {-35, 49, -76}
, {-26, -63, -34}
, {59, 45, -76}
, {65, 37, 11}
, {-21, -50, 45}
, {63, -31, 67}
, {32, -75, 57}
, {62, 24, -37}
, {-22, -66, -3}
, {0, 74, 66}
, {-54, 38, 64}
, {-26, -34, -77}
, {-5, -48, 56}
, {13, 6, 74}
, {-2, 52, 2}
, {62, 29, 16}
, {-75, -23, -20}
, {47, 45, 79}
, {32, 13, 63}
, {-57, -59, 48}
, {-67, -5, 54}
, {79, -32, -36}
, {64, -5, -29}
, {-9, -61, 42}
, {68, 56, -38}
, {22, 57, 18}
, {-25, 7, -31}
, {-8, -72, 0}
, {-27, -26, 0}
, {19, -37, -3}
, {10, 0, -65}
}
, {{4, -38, 8}
, {-18, -9, 50}
, {74, -19, 25}
, {-53, -8, 72}
, {29, 5, -60}
, {-12, -73, 11}
, {52, -9, -6}
, {21, -22, -36}
, {70, -6, 48}
, {71, -29, -44}
, {-49, 35, -72}
, {64, 26, 27}
, {-28, 23, -48}
, {65, 13, 1}
, {-3, 52, 14}
, {-7, -15, -48}
, {-54, -56, 26}
, {-70, 48, 7}
, {49, 29, 2}
, {-7, 30, 54}
, {39, -42, 17}
, {-66, 32, 66}
, {4, 52, 31}
, {-27, 34, 55}
, {16, 28, 66}
, {50, -31, 6}
, {-62, -24, 50}
, {-62, 36, -17}
, {-6, 48, 21}
, {-32, 66, 7}
, {-4, 8, -73}
, {-82, 1, -24}
}
, {{18, 37, -55}
, {45, 33, -53}
, {-39, 15, -4}
, {4, -35, 23}
, {-36, -40, -84}
, {-21, -48, 61}
, {3, 64, 74}
, {40, -80, 12}
, {-48, 66, -30}
, {-22, 58, -66}
, {-51, 47, 27}
, {-55, -5, 52}
, {-46, 67, -1}
, {37, 59, 2}
, {28, -15, -38}
, {48, 52, -48}
, {-42, -38, -35}
, {48, -23, -55}
, {-7, 16, 73}
, {-15, 45, 40}
, {45, -5, -43}
, {23, -80, 53}
, {-52, -60, 32}
, {-41, -17, 25}
, {49, -43, -66}
, {39, 17, -12}
, {-50, 71, 27}
, {72, 13, -53}
, {24, -53, 0}
, {16, 43, 55}
, {15, -65, -28}
, {34, -52, 7}
}
, {{-27, -1, -25}
, {-25, 69, -12}
, {-15, 74, -64}
, {63, -44, 8}
, {1, -28, 38}
, {16, -44, 18}
, {47, -24, -36}
, {85, 73, -60}
, {-33, 44, 27}
, {74, -29, -64}
, {-42, -50, 46}
, {-5, -13, -53}
, {-23, 89, -4}
, {62, -19, -36}
, {85, -37, -66}
, {-21, 4, 63}
, {-70, 25, 27}
, {8, 76, -45}
, {-5, 18, -2}
, {13, -86, -26}
, {-5, 17, -56}
, {-19, -43, -62}
, {10, 74, 27}
, {-11, -3, 16}
, {47, 22, 32}
, {15, 48, -49}
, {-6, -23, -1}
, {-10, 38, 15}
, {44, 1, -60}
, {-12, -37, 72}
, {-22, -16, -13}
, {3, 0, -66}
}
, {{-32, 37, 41}
, {-16, -59, -48}
, {36, -13, -66}
, {18, -78, -28}
, {0, -10, 52}
, {1, 33, -34}
, {32, 44, -38}
, {45, -38, -29}
, {-64, 37, -1}
, {67, 13, -2}
, {60, 54, 74}
, {-58, -70, 35}
, {22, 21, -21}
, {9, -42, 2}
, {47, 52, 55}
, {13, 12, 48}
, {-18, -3, -10}
, {-6, -23, 73}
, {26, 9, -56}
, {-34, 36, -46}
, {-38, -4, -62}
, {-1, 70, 45}
, {60, 59, 9}
, {54, -10, -8}
, {-35, -47, -47}
, {-67, 62, -12}
, {-61, -42, 5}
, {-20, 67, 56}
, {66, -50, 55}
, {-12, -2, 41}
, {-11, -10, -60}
, {12, 17, -36}
}
, {{-15, 38, -14}
, {-22, 21, -18}
, {-48, 3, -32}
, {24, -26, 73}
, {54, 70, -22}
, {-18, 29, 13}
, {-17, 78, -24}
, {10, 20, -51}
, {-21, 56, -52}
, {13, -33, -44}
, {-68, 42, 83}
, {27, 25, -31}
, {43, -28, 11}
, {-40, 66, 54}
, {14, -28, -24}
, {24, 29, -46}
, {3, 35, -62}
, {80, -57, -17}
, {59, 42, -55}
, {50, 12, 1}
, {-66, -20, -48}
, {26, -33, 17}
, {-28, 23, -6}
, {68, 15, -37}
, {52, 49, 71}
, {-29, -48, 33}
, {54, -49, -24}
, {26, -46, 42}
, {23, 16, 51}
, {42, -21, 48}
, {-55, -2, -13}
, {-21, -39, 21}
}
, {{-19, 18, 8}
, {47, -39, -33}
, {22, -35, 4}
, {69, 24, 29}
, {-43, -56, -23}
, {-26, 36, -75}
, {20, 1, 41}
, {81, 60, 8}
, {-36, -40, -8}
, {47, -37, 65}
, {-56, 1, -26}
, {-57, 34, 12}
, {-60, 44, -28}
, {-63, 34, -53}
, {60, 44, 10}
, {-36, 43, -36}
, {45, -35, -48}
, {29, 66, 58}
, {31, 34, 30}
, {0, 46, -5}
, {-33, -50, 42}
, {-17, 51, -40}
, {40, 0, -2}
, {-11, 22, 18}
, {-29, -7, -40}
, {32, -57, 47}
, {45, 53, 58}
, {-20, 10, -31}
, {-52, 61, 68}
, {-64, 3, -53}
, {39, 34, 45}
, {-18, 49, 58}
}
, {{-56, -36, -37}
, {-65, 45, 55}
, {-62, -8, 61}
, {-43, -40, 20}
, {50, -55, 39}
, {-23, 49, 39}
, {10, -64, -62}
, {28, -63, -28}
, {-15, -21, -94}
, {23, -31, 8}
, {-22, -13, 28}
, {-38, 18, 57}
, {67, 16, -14}
, {-52, 19, -36}
, {-38, 1, -55}
, {71, 42, -46}
, {3, 20, 4}
, {16, -66, -37}
, {-63, -49, -75}
, {51, 37, -6}
, {18, 64, 41}
, {-12, -39, -50}
, {-51, -23, -25}
, {-26, 26, -75}
, {-26, 42, -13}
, {51, -27, 11}
, {50, 68, 69}
, {2, 55, -40}
, {36, -30, 54}
, {-68, -55, 12}
, {74, 66, -73}
, {-60, 30, 25}
}
, {{65, 68, -66}
, {-15, 41, 38}
, {6, 58, -35}
, {-30, 10, -54}
, {-22, -43, 26}
, {-65, 61, -24}
, {27, -68, 49}
, {28, 57, 64}
, {12, -19, 52}
, {-40, -79, -79}
, {27, 69, 38}
, {-31, -80, 43}
, {54, 12, -62}
, {40, 57, -42}
, {-70, -56, 22}
, {9, -43, -35}
, {54, 0, -74}
, {-54, -32, 0}
, {-36, 62, -43}
, {56, 29, -53}
, {22, -33, 25}
, {-6, 53, 59}
, {29, 32, 60}
, {30, -37, -26}
, {66, -74, -1}
, {6, -71, -32}
, {-66, -62, -65}
, {-5, -65, 40}
, {56, 67, 69}
, {53, -20, 51}
, {44, 46, -34}
, {34, 25, -23}
}
, {{-46, 69, 45}
, {27, -50, -59}
, {-45, -46, -26}
, {-26, 5, -14}
, {74, 82, 38}
, {-10, 35, -16}
, {61, -50, -49}
, {2, -42, -55}
, {-12, 28, 53}
, {10, 78, -23}
, {54, 8, -35}
, {1, -28, 0}
, {31, 4, -50}
, {44, 19, -34}
, {-22, -6, -3}
, {-69, -75, -2}
, {-17, -50, -35}
, {46, 56, 21}
, {1, -6, 50}
, {-14, -72, 17}
, {-55, 36, 41}
, {0, -15, -54}
, {-47, 26, 9}
, {-12, 0, 17}
, {6, 64, -57}
, {-36, 34, 26}
, {55, -62, -1}
, {62, 35, 77}
, {-64, -21, -21}
, {-64, 35, -14}
, {28, -32, 68}
, {-8, -4, 46}
}
, {{71, -23, -17}
, {-12, 78, 43}
, {-25, -24, -62}
, {-36, 10, -47}
, {-36, 35, 68}
, {40, 26, -33}
, {7, -45, -12}
, {-43, -13, -7}
, {43, 0, 0}
, {-34, -61, 27}
, {42, -27, 28}
, {-40, -53, 60}
, {60, -44, -49}
, {34, -4, 81}
, {-75, -19, -55}
, {-40, 3, 37}
, {-68, -72, -10}
, {18, 63, 24}
, {-6, -18, 46}
, {28, 5, -41}
, {46, -63, 55}
, {-8, -16, -11}
, {-68, 26, -89}
, {12, 57, -21}
, {-8, -2, 23}
, {-2, 41, 3}
, {59, 30, -56}
, {24, 52, 39}
, {23, 2, 1}
, {11, 11, 12}
, {11, -50, 75}
, {45, -70, 42}
}
, {{-1, -1, 0}
, {4, 75, 60}
, {74, 28, 62}
, {5, -7, -59}
, {47, -10, -7}
, {-17, -4, -17}
, {-55, 0, -45}
, {-32, -87, 67}
, {-64, 66, -61}
, {43, 2, -43}
, {-42, -34, 46}
, {-4, -59, 73}
, {50, 87, -18}
, {-1, 53, -21}
, {-26, -65, -64}
, {55, -16, 1}
, {29, -14, 49}
, {-53, -47, -36}
, {-56, -30, 22}
, {38, 7, 93}
, {-56, -35, -4}
, {11, -23, 63}
, {86, -12, -46}
, {19, 18, 25}
, {-33, 60, 72}
, {3, -24, -66}
, {-3, 52, 60}
, {29, 10, 1}
, {-28, 65, -24}
, {-78, 49, 14}
, {67, -49, 36}
, {-51, -18, -71}
}
, {{62, 60, -61}
, {-7, 37, 26}
, {-54, -46, 15}
, {-46, 70, -40}
, {22, 4, -50}
, {7, -43, -6}
, {54, 68, 58}
, {60, 66, -26}
, {54, 24, 44}
, {15, -58, 36}
, {8, -60, 9}
, {16, 54, 34}
, {-40, 54, -1}
, {-33, -4, 6}
, {-23, -26, -41}
, {-66, -67, -32}
, {-23, 3, -27}
, {78, -3, -52}
, {27, -16, 31}
, {80, 33, 17}
, {37, 16, -27}
, {-14, 46, 13}
, {12, 57, 90}
, {83, -34, -59}
, {50, 4, -49}
, {47, -49, 34}
, {81, -40, 67}
, {-35, 3, 14}
, {-54, -78, -12}
, {-46, 3, -66}
, {60, 58, -60}
, {-1, -70, -14}
}
, {{-24, -66, -31}
, {69, 73, -11}
, {-2, 40, -76}
, {-22, -23, 50}
, {-26, -3, 32}
, {37, -25, -2}
, {29, 22, 71}
, {-42, -44, 47}
, {-79, -24, -87}
, {31, 4, -59}
, {-37, 18, 58}
, {-32, 40, 53}
, {13, -34, 26}
, {10, 19, 4}
, {39, -44, -71}
, {-15, 40, 64}
, {-24, -7, -1}
, {44, 56, 21}
, {-60, -65, -27}
, {19, -12, 17}
, {-49, -44, 75}
, {1, 43, 59}
, {36, -38, 46}
, {-10, 42, 22}
, {-61, -53, 1}
, {47, 63, 57}
, {4, 18, 78}
, {-1, 22, -30}
, {51, -37, 7}
, {32, -11, -71}
, {62, -4, 70}
, {32, 21, 28}
}
, {{53, -47, -33}
, {-7, -42, 18}
, {56, 17, 23}
, {3, 11, -60}
, {-61, 0, -49}
, {-45, -23, -39}
, {-12, 42, -41}
, {-45, -84, 9}
, {-45, 2, 36}
, {0, -46, -31}
, {-47, -57, -66}
, {53, -21, 46}
, {52, -29, 5}
, {-82, -3, -65}
, {32, -14, 12}
, {-6, -42, -15}
, {45, 23, -51}
, {63, -16, 5}
, {6, -40, 17}
, {-58, 41, 5}
, {-48, -33, -47}
, {-31, 31, 41}
, {80, 8, 43}
, {31, 56, 48}
, {-11, 10, -46}
, {-11, -24, 29}
, {80, 15, 75}
, {70, -40, -63}
, {0, 12, 33}
, {31, 3, 62}
, {6, -5, 55}
, {-25, -37, 58}
}
, {{-4, 2, -33}
, {66, 6, -48}
, {-27, -3, 26}
, {65, -25, 7}
, {50, 29, 17}
, {55, -59, 28}
, {-43, 47, 19}
, {1, -50, 72}
, {72, 76, 34}
, {-4, 45, -9}
, {-51, -55, 36}
, {-42, 63, 57}
, {60, 50, 17}
, {-65, -26, -68}
, {-51, 54, 32}
, {-36, -49, -61}
, {-58, 36, -35}
, {73, -62, -25}
, {-73, -24, 18}
, {8, 45, -64}
, {-47, 57, 29}
, {-80, -27, -43}
, {7, 44, -63}
, {16, 39, 50}
, {36, 11, 18}
, {72, 56, -45}
, {-72, 71, 36}
, {-23, -38, -75}
, {-16, -35, -24}
, {47, 62, -58}
, {39, 26, 10}
, {-54, 38, 53}
}
, {{-22, 31, 8}
, {-4, -9, -5}
, {-53, 21, -9}
, {-38, 80, -16}
, {-2, -34, -60}
, {10, 9, 18}
, {-58, -12, 41}
, {-25, -22, -21}
, {0, -9, -42}
, {21, 6, 31}
, {59, 2, 58}
, {69, 35, 24}
, {-1, 41, -47}
, {-58, -25, -5}
, {-50, 15, 27}
, {-38, -62, -5}
, {-47, 37, 78}
, {12, -43, -57}
, {73, 38, 69}
, {-67, -13, -7}
, {-36, -33, -77}
, {5, 6, -43}
, {54, -37, -23}
, {3, -63, 38}
, {0, -24, -59}
, {43, -57, 29}
, {30, -80, -32}
, {-80, 60, 10}
, {-35, 50, 46}
, {76, -36, -22}
, {59, 50, -52}
, {12, -43, -77}
}
, {{-7, -49, -37}
, {34, -14, -54}
, {53, -4, 21}
, {87, -48, -12}
, {14, 56, -32}
, {15, 45, 24}
, {-31, -61, 32}
, {-55, -36, 20}
, {65, 53, 20}
, {-44, -43, 0}
, {-66, 24, -37}
, {38, 56, -66}
, {-62, 32, 36}
, {-23, 75, -37}
, {-45, 26, -55}
, {-33, 0, 49}
, {-41, 53, 5}
, {83, -22, 12}
, {37, 8, 61}
, {-26, -60, -44}
, {-32, 65, -46}
, {38, 34, 4}
, {73, 48, -21}
, {-28, -56, -2}
, {-55, 14, -10}
, {-24, 6, -50}
, {-40, -42, 66}
, {86, 57, 8}
, {30, -4, 22}
, {11, -60, 66}
, {-22, -8, -70}
, {-3, -22, -51}
}
, {{-7, -48, 39}
, {20, 39, 10}
, {-44, -23, -1}
, {-11, 4, 54}
, {-47, -3, 31}
, {-43, -36, 79}
, {56, -50, 45}
, {35, -32, -18}
, {37, 26, -83}
, {4, 19, 83}
, {-68, 32, 39}
, {61, -21, -37}
, {-34, 44, -72}
, {-47, -46, -58}
, {70, 47, 25}
, {75, 64, 58}
, {-26, 66, 43}
, {-15, 0, -28}
, {-74, 55, -46}
, {-8, 57, -55}
, {48, -31, 10}
, {35, 24, 75}
, {17, -42, 85}
, {54, -41, -38}
, {-68, -60, 52}
, {74, -26, 11}
, {60, 0, 8}
, {-35, 29, 85}
, {-28, -8, 78}
, {31, 47, 3}
, {-24, -62, -71}
, {58, -37, -66}
}
, {{-15, 55, 6}
, {-29, -4, 45}
, {5, 21, 44}
, {66, 0, -5}
, {-68, -63, 39}
, {3, -57, 62}
, {-30, -35, 5}
, {-66, -16, 43}
, {-51, -32, 43}
, {1, -52, -11}
, {-63, 52, -42}
, {51, 70, 41}
, {31, 60, 4}
, {-16, -70, -33}
, {-61, -21, 54}
, {-10, -58, 0}
, {-60, -10, -3}
, {4, 5, -42}
, {60, 62, -74}
, {-59, -32, -32}
, {-32, 2, 41}
, {78, 15, 6}
, {-12, 19, 59}
, {79, 40, -46}
, {24, 43, -46}
, {56, -35, 44}
, {65, 35, 41}
, {60, 4, -64}
, {-63, -72, 7}
, {11, -20, 18}
, {-48, 58, 62}
, {-36, -35, -2}
}
, {{38, 40, -15}
, {-31, -69, -69}
, {-3, -22, -55}
, {61, 60, -3}
, {65, 53, 59}
, {38, -62, 43}
, {-50, 53, -50}
, {-12, 41, 34}
, {50, -37, 37}
, {44, -17, 25}
, {37, 18, 11}
, {-67, -60, -23}
, {-54, 75, -57}
, {61, 2, 56}
, {-30, -39, -26}
, {-23, -33, -56}
, {-31, -30, -8}
, {70, 88, -25}
, {62, 19, 11}
, {49, 13, 42}
, {-31, -1, 39}
, {-28, 42, -9}
, {-25, -39, -49}
, {-66, 63, -12}
, {54, 10, 55}
, {5, 59, 76}
, {-1, -3, -64}
, {4, -43, 10}
, {-58, 26, -6}
, {0, -66, -19}
, {7, -16, -8}
, {-6, 15, 63}
}
, {{-16, -68, 22}
, {-8, 60, 49}
, {31, -32, 58}
, {-46, -59, 68}
, {6, 11, 67}
, {-9, -69, 15}
, {10, -39, -15}
, {39, 16, 69}
, {72, 9, 87}
, {-61, 52, 26}
, {-33, -6, -11}
, {78, -23, -71}
, {-21, 31, 25}
, {-47, -66, 26}
, {12, 21, -17}
, {-23, 73, 54}
, {-25, 52, -23}
, {-38, -60, 19}
, {51, -18, -16}
, {52, 23, 18}
, {-61, -36, -21}
, {49, 20, -13}
, {-35, 70, 0}
, {44, -24, 9}
, {-77, -59, 75}
, {54, 64, -47}
, {-13, -73, -69}
, {63, -56, -35}
, {-21, 38, -24}
, {50, -20, -29}
, {-48, -7, 0}
, {58, -17, 33}
}
, {{-45, -9, -57}
, {26, 41, -47}
, {14, 79, 62}
, {-27, -17, 61}
, {-45, 72, 65}
, {26, 58, -61}
, {-2, 53, -65}
, {47, 28, 60}
, {4, -6, -14}
, {-17, 75, -33}
, {-60, 58, 30}
, {-17, 30, 35}
, {30, 21, 65}
, {51, 15, -4}
, {36, -61, 20}
, {-82, -25, 48}
, {1, 72, 70}
, {-9, -55, -65}
, {-43, 45, 63}
, {25, -50, -15}
, {-65, -46, 37}
, {14, 34, -59}
, {-42, 49, -31}
, {-19, -37, 59}
, {-16, -16, -11}
, {34, -39, -55}
, {-68, -45, -36}
, {64, 15, -61}
, {-7, 24, -51}
, {-9, -52, 31}
, {22, -46, -44}
, {-79, 43, 33}
}
, {{56, 29, 24}
, {26, 55, -61}
, {-14, -38, -25}
, {-18, 26, 49}
, {47, -62, 51}
, {-8, -31, 45}
, {-32, 40, -66}
, {39, -64, 49}
, {6, -36, -51}
, {39, 37, 56}
, {-20, -15, 33}
, {-1, 32, 60}
, {10, -53, 54}
, {-52, -2, 25}
, {63, 79, 17}
, {70, 60, -29}
, {37, -69, -21}
, {-63, 36, -83}
, {-55, 69, 64}
, {-14, 0, 49}
, {55, -65, -54}
, {-21, 0, 40}
, {50, -35, -10}
, {-70, 46, -20}
, {50, 12, 66}
, {68, -63, -55}
, {24, -34, 11}
, {18, 50, 62}
, {8, 73, -62}
, {-24, 10, -69}
, {52, -51, 1}
, {68, -34, 43}
}
, {{8, -20, 68}
, {26, 31, -5}
, {-46, -15, -63}
, {-8, -66, 36}
, {12, 90, -23}
, {37, 44, -66}
, {-63, 32, -22}
, {-66, -59, 30}
, {-56, -78, 47}
, {-23, 8, 20}
, {48, -50, 66}
, {-65, 29, -12}
, {11, 48, -9}
, {46, 68, 0}
, {-22, -82, 53}
, {5, -28, -20}
, {-66, -67, 34}
, {16, 68, 5}
, {-66, -41, -26}
, {20, 63, -62}
, {-11, -32, -42}
, {26, -73, -19}
, {-59, 31, -42}
, {11, 53, 22}
, {24, -2, -6}
, {-2, 77, 33}
, {-43, 30, -10}
, {51, 27, 10}
, {34, -33, 60}
, {18, -48, -19}
, {53, -2, -9}
, {48, 62, -21}
}
, {{-45, 37, 9}
, {-72, 46, 19}
, {-29, 4, 67}
, {52, 38, 3}
, {-21, 77, 45}
, {40, 12, 14}
, {-38, -34, -34}
, {11, 37, -64}
, {-63, 35, -46}
, {15, 49, 44}
, {70, -28, -60}
, {-32, -50, 57}
, {-8, -67, 35}
, {-69, -64, 36}
, {18, 50, 15}
, {-35, -60, 14}
, {-53, 79, 63}
, {21, -46, -65}
, {71, 66, 30}
, {18, 43, 30}
, {-21, 3, 54}
, {71, 10, -19}
, {-78, 39, 73}
, {-20, -23, 49}
, {-59, 35, 30}
, {-4, 0, 61}
, {59, 17, -9}
, {-54, -51, -49}
, {-66, 62, 50}
, {-18, -53, -6}
, {-31, -3, -39}
, {32, 48, -65}
}
, {{-2, -38, 56}
, {-49, -40, -74}
, {-52, -60, 66}
, {49, -1, -46}
, {-12, 64, -4}
, {-43, 32, -31}
, {7, 44, -62}
, {-37, 20, 69}
, {-47, 27, -71}
, {-11, 9, -22}
, {0, -70, -81}
, {-42, 17, 47}
, {-61, -23, -47}
, {24, 35, 12}
, {-56, 49, -10}
, {-5, -16, -32}
, {-52, 38, 35}
, {69, -13, 17}
, {64, -35, 6}
, {-35, -11, 28}
, {-68, 37, 65}
, {10, 14, -27}
, {-25, -42, -59}
, {43, 64, -54}
, {0, 13, 6}
, {-35, -7, -46}
, {-61, 46, -37}
, {31, -11, -20}
, {34, -14, 57}
, {70, 66, -69}
, {49, 38, -62}
, {-44, -1, 52}
}
, {{-7, 11, 55}
, {30, -21, 9}
, {12, 34, 7}
, {-62, -54, 70}
, {19, -33, 1}
, {71, 24, 42}
, {-62, 74, -12}
, {-18, 26, 52}
, {18, 45, -8}
, {-46, 30, -30}
, {-47, -55, 4}
, {-18, 40, 58}
, {69, 60, 18}
, {-28, -68, -12}
, {46, -49, -62}
, {-51, -19, -29}
, {28, 49, 53}
, {3, -45, -64}
, {-41, 34, -17}
, {-13, -40, 25}
, {80, -41, 57}
, {53, 50, -17}
, {-28, 62, -40}
, {-14, -44, 0}
, {26, -76, -9}
, {-66, -9, -31}
, {-51, 57, 2}
, {-29, -35, -27}
, {53, -35, 53}
, {-7, 25, 31}
, {-61, -13, -65}
, {40, -19, -38}
}
, {{-33, -18, -41}
, {-49, -65, -71}
, {13, 66, 59}
, {30, 1, -20}
, {-4, -6, 11}
, {-26, -45, 45}
, {13, 9, -27}
, {33, 57, 51}
, {45, 46, 9}
, {-10, 69, 26}
, {-14, -12, -71}
, {-36, 31, -43}
, {-62, 40, 8}
, {55, 37, -58}
, {79, -13, -19}
, {43, 55, 43}
, {72, -27, 29}
, {-35, -5, 37}
, {22, -49, 70}
, {-35, 28, 20}
, {-65, 60, 47}
, {43, 59, 29}
, {-55, -1, 31}
, {23, -24, 45}
, {-35, -72, -79}
, {-11, -38, -36}
, {-23, 3, 8}
, {38, 21, -52}
, {-56, 76, 20}
, {12, 7, 29}
, {-9, 67, 16}
, {72, -24, 71}
}
, {{58, 43, 48}
, {63, 29, -23}
, {-70, 13, 16}
, {-19, -30, -23}
, {26, 0, 7}
, {36, 44, -76}
, {-38, 62, -36}
, {1, -43, -37}
, {-9, 23, -65}
, {-50, -17, 39}
, {-68, 67, -43}
, {14, 49, 63}
, {30, -56, 0}
, {20, 0, 34}
, {46, 12, 16}
, {44, 68, -67}
, {-45, 32, 28}
, {74, -70, 32}
, {44, -51, 30}
, {-41, -79, -33}
, {33, 29, 16}
, {-19, -60, 24}
, {56, -69, -38}
, {50, -4, -61}
, {51, -61, -9}
, {49, -55, -25}
, {-30, 26, -12}
, {68, 30, -47}
, {56, -66, -65}
, {-73, 49, 8}
, {33, -37, 7}
, {-32, 49, 5}
}
, {{20, -43, -32}
, {-47, -49, 56}
, {-71, 72, 70}
, {-9, -18, -65}
, {52, -14, -5}
, {-18, -7, 64}
, {18, -26, 59}
, {75, 50, -44}
, {-30, -43, -61}
, {66, 1, 74}
, {10, 60, 32}
, {52, -63, -25}
, {0, -15, 85}
, {-33, -25, -57}
, {74, 48, 59}
, {-18, -27, -32}
, {-4, -36, 43}
, {63, -49, -53}
, {-2, 66, -67}
, {25, -78, -22}
, {17, -46, -53}
, {77, -40, -20}
, {-61, 47, 74}
, {-28, -31, 56}
, {-45, -10, -9}
, {30, 26, -41}
, {-58, 59, 17}
, {5, -9, 12}
, {-73, 61, -68}
, {-45, 26, 8}
, {-60, 11, 11}
, {50, 46, -4}
}
, {{-45, 50, 18}
, {-82, -7, 26}
, {-50, 10, -12}
, {-18, -42, 2}
, {-64, -55, 9}
, {11, 7, -61}
, {-79, 48, 39}
, {42, -21, 55}
, {64, 33, -21}
, {43, -46, 9}
, {-4, -6, 23}
, {33, 65, -34}
, {-12, 84, -23}
, {-67, 55, -37}
, {20, -39, 55}
, {16, -17, 71}
, {75, 8, 20}
, {40, -3, 36}
, {50, -52, -33}
, {-68, 34, 55}
, {-24, 29, 66}
, {-59, 51, -6}
, {52, -7, 13}
, {23, -36, -62}
, {37, 86, 46}
, {-71, -60, -30}
, {37, -42, 32}
, {6, 3, 15}
, {-7, 5, 74}
, {-51, 39, 41}
, {-71, -40, 13}
, {65, 75, 42}
}
, {{31, 53, -37}
, {-19, -60, 0}
, {36, 45, 64}
, {0, -11, -3}
, {48, 22, -70}
, {53, 30, 48}
, {0, -41, 2}
, {-71, -39, -65}
, {-59, 14, 55}
, {-51, 65, -63}
, {25, -54, 62}
, {-56, -65, -18}
, {-48, 13, 23}
, {38, 47, 22}
, {41, 38, -76}
, {-16, -64, 22}
, {3, 65, 62}
, {12, 23, -4}
, {-12, -3, -24}
, {-18, -32, -43}
, {-60, -8, 44}
, {-41, 36, -4}
, {-48, -61, 40}
, {-73, 17, -42}
, {-61, -30, 8}
, {-27, 55, 1}
, {37, 10, 54}
, {-9, -38, -4}
, {66, -35, 15}
, {62, 4, 49}
, {-7, -18, -67}
, {33, 23, -51}
}
, {{52, 59, 39}
, {-24, 50, 12}
, {-4, 52, -3}
, {-16, 26, 101}
, {-57, -21, 62}
, {-12, 23, 29}
, {42, 65, 46}
, {63, -45, -20}
, {67, 66, -40}
, {31, 5, -55}
, {16, 67, -66}
, {-35, 10, -17}
, {36, 15, -35}
, {-29, -20, -69}
, {-56, 4, -36}
, {-8, 37, -50}
, {26, -65, 58}
, {-68, -18, 6}
, {-59, 44, -40}
, {25, 29, 68}
, {22, -59, -21}
, {-23, -41, 11}
, {-27, 53, 46}
, {-56, 38, 42}
, {49, 3, -46}
, {-17, -8, 1}
, {38, 18, 0}
, {-45, 80, -21}
, {-26, 31, -78}
, {14, 3, 7}
, {-69, 5, -12}
, {70, -59, 65}
}
, {{-27, -2, 35}
, {62, -68, 66}
, {-28, 34, 45}
, {-35, 46, -24}
, {3, 5, 60}
, {-33, -14, 37}
, {-31, -35, -63}
, {-8, -38, 59}
, {-41, 23, -7}
, {36, -30, -13}
, {-58, 20, 1}
, {-61, -9, 15}
, {18, -26, 20}
, {41, 20, -57}
, {-33, -12, 6}
, {-62, -60, 59}
, {26, 39, -61}
, {8, -40, 61}
, {52, -42, 48}
, {-1, 43, 34}
, {8, 28, -18}
, {72, -56, 53}
, {-13, 41, 28}
, {-40, 23, 57}
, {-74, -18, 21}
, {27, -56, -4}
, {-43, -3, -62}
, {-79, 56, 5}
, {-17, -3, 4}
, {39, -11, -34}
, {-4, -43, -14}
, {27, -7, 40}
}
, {{48, -13, -68}
, {37, 66, -33}
, {-17, 2, 28}
, {-66, -55, -30}
, {-61, 18, -62}
, {76, 69, -16}
, {25, -52, 22}
, {57, 52, 37}
, {34, 0, 30}
, {-26, -21, -44}
, {17, -19, -62}
, {35, 43, 43}
, {-11, -60, 47}
, {6, 12, 32}
, {6, -41, -27}
, {49, -46, 29}
, {20, -52, 26}
, {-65, 7, 10}
, {-34, -61, 54}
, {-59, 59, -19}
, {37, -46, 68}
, {43, 25, -40}
, {9, 41, -57}
, {-27, 34, 41}
, {-52, -26, 25}
, {64, -24, 16}
, {37, 12, -8}
, {-26, 43, -60}
, {-61, 66, 39}
, {-49, -70, 19}
, {-42, -65, -32}
, {65, -4, 41}
}
, {{6, -43, 50}
, {15, 0, -34}
, {-31, -28, -69}
, {-29, 71, 45}
, {53, -65, -9}
, {70, 51, 2}
, {-68, -50, -32}
, {65, -59, -29}
, {39, 0, 34}
, {49, 17, 23}
, {36, 53, 47}
, {-20, -8, 55}
, {66, -76, 69}
, {10, -8, -68}
, {-13, 18, 6}
, {68, 1, -48}
, {-12, -9, 26}
, {-61, 63, -54}
, {-44, 1, 68}
, {43, 54, 29}
, {60, -48, 21}
, {3, -66, -45}
, {63, 32, -29}
, {-5, -11, 13}
, {-46, 25, 24}
, {-15, -78, -24}
, {47, -68, -51}
, {-53, -79, -12}
, {60, 61, 19}
, {-40, -8, 21}
, {-11, 0, 63}
, {45, 10, -9}
}
, {{61, 14, 38}
, {-38, 40, -72}
, {-41, -53, -76}
, {-40, -26, -44}
, {45, -57, 38}
, {-42, 31, 51}
, {-16, -1, 54}
, {36, -21, 68}
, {71, -39, -86}
, {67, -29, -71}
, {-69, 72, -70}
, {-3, -35, 46}
, {53, 5, -76}
, {-29, -23, -71}
, {-47, -27, 48}
, {-38, 18, -62}
, {66, -19, 51}
, {30, 41, 40}
, {-52, -58, 26}
, {60, 46, 38}
, {-59, -44, 11}
, {16, 27, -17}
, {93, -86, 57}
, {-28, -36, -3}
, {78, 18, -51}
, {-66, -20, 51}
, {84, -56, 65}
, {82, -62, 0}
, {46, 37, 31}
, {-29, 17, 28}
, {7, -5, -45}
, {22, -71, 75}
}
, {{48, -72, 5}
, {15, -35, 28}
, {-36, -9, 50}
, {54, -27, 49}
, {-74, -45, 49}
, {-41, 44, -65}
, {-13, -60, -4}
, {0, -51, 68}
, {31, -66, -42}
, {12, 68, -30}
, {-74, -33, -31}
, {-36, 43, -39}
, {-44, 60, -16}
, {26, 59, -59}
, {28, -29, -27}
, {29, -77, 15}
, {5, -13, 62}
, {-49, 57, 34}
, {23, 7, 13}
, {50, -45, 49}
, {30, 1, 64}
, {-54, 30, -4}
, {-62, 68, 4}
, {70, 11, 66}
, {24, -10, 9}
, {-31, -69, 46}
, {39, 12, -20}
, {-60, 2, 9}
, {-31, -19, 59}
, {12, 33, 71}
, {-4, 19, 53}
, {-10, -63, -16}
}
, {{74, 79, 51}
, {60, -34, -1}
, {64, -19, -45}
, {-72, -74, -50}
, {32, 67, 14}
, {75, 6, -28}
, {14, -17, -26}
, {38, -47, 23}
, {-48, 70, -25}
, {-31, -70, -59}
, {51, 28, 18}
, {-57, 53, -14}
, {28, -50, 54}
, {48, 89, 80}
, {-70, 42, 1}
, {-34, -39, 49}
, {-58, 21, -77}
, {13, 36, -45}
, {59, 28, -25}
, {-51, 30, 36}
, {-62, 30, 49}
, {-83, -58, -32}
, {48, -34, -54}
, {-70, -28, 65}
, {-30, -45, 77}
, {-52, 15, -52}
, {-55, -9, -61}
, {25, -68, 1}
, {-31, -13, 28}
, {-34, -6, -44}
, {-14, 57, 77}
, {39, -8, -26}
}
, {{7, 4, -17}
, {39, -50, 91}
, {-24, 61, 38}
, {8, -36, 42}
, {-37, -22, 43}
, {-11, 56, -30}
, {-58, -63, 54}
, {83, -46, 17}
, {-15, -44, 71}
, {-2, 46, 43}
, {12, 59, 71}
, {78, -50, 28}
, {4, -39, 29}
, {28, -24, -6}
, {32, -49, -17}
, {56, 66, -5}
, {47, 48, 27}
, {-31, 26, 41}
, {80, 72, 2}
, {-93, -27, 7}
, {-75, -36, -70}
, {-17, -19, -78}
, {-61, 52, -19}
, {78, -66, -53}
, {74, 4, 82}
, {-28, 11, 73}
, {-26, -73, -50}
, {-28, 35, -73}
, {-53, 5, -41}
, {54, 45, 32}
, {-46, -47, -49}
, {77, -31, 26}
}
, {{-53, 72, 49}
, {-30, -67, 49}
, {-60, -61, 59}
, {86, -63, -3}
, {53, 61, 58}
, {-69, -32, -8}
, {-53, -29, -41}
, {34, -15, 13}
, {-15, 42, 49}
, {-57, 60, 1}
, {39, -32, 51}
, {32, -72, -81}
, {33, 14, 34}
, {-5, -14, 76}
, {-33, 41, 1}
, {-49, 74, 76}
, {-12, 43, 62}
, {49, -48, 20}
, {-53, -78, -13}
, {47, -74, 2}
, {-14, 70, -34}
, {-63, -45, 1}
, {78, -3, -53}
, {-58, -56, -45}
, {39, -21, -65}
, {37, 4, -64}
, {-48, 52, 6}
, {-48, -51, -7}
, {-31, -18, 8}
, {-19, 67, 35}
, {74, 16, -28}
, {-59, 42, -8}
}
, {{-50, 50, 25}
, {-55, 37, -29}
, {-11, -29, 17}
, {22, -37, 0}
, {-23, -43, 27}
, {55, 21, 75}
, {8, -70, 41}
, {47, -31, -61}
, {-69, -33, -52}
, {-35, 75, 63}
, {65, -14, 62}
, {37, 30, 1}
, {-25, -45, 46}
, {30, 35, 67}
, {57, -79, 14}
, {32, 20, -45}
, {-45, 20, -38}
, {79, 34, -20}
, {-76, 50, 40}
, {-20, -21, 10}
, {-68, -40, 34}
, {16, 37, 32}
, {36, 1, -53}
, {0, 12, 62}
, {12, -24, 35}
, {2, -27, 57}
, {42, -19, 10}
, {27, 60, 83}
, {8, 70, 55}
, {-60, 15, 39}
, {64, -2, -63}
, {-38, -61, 44}
}
, {{-17, -28, -69}
, {-30, 57, 62}
, {0, 28, -29}
, {87, -10, -39}
, {30, -43, 33}
, {13, 37, 28}
, {-44, 21, 3}
, {18, -60, -23}
, {52, 60, -74}
, {-2, 50, -18}
, {-72, 33, 59}
, {-24, -50, 54}
, {15, -21, -52}
, {45, 44, 1}
, {10, -11, -68}
, {58, -47, 21}
, {26, 75, -53}
, {67, 34, -22}
, {39, -71, 55}
, {31, 31, 29}
, {86, -42, 37}
, {62, -59, 38}
, {12, 28, -22}
, {-29, 62, 71}
, {-34, 49, -65}
, {-47, -66, -3}
, {-47, -17, -27}
, {77, 62, -47}
, {4, -11, -41}
, {-61, 36, 42}
, {-73, -56, -27}
, {-12, -86, 55}
}
, {{-1, -70, 12}
, {-55, -26, -60}
, {-68, -37, 37}
, {-76, -78, -9}
, {-26, -2, 22}
, {4, 64, 21}
, {-61, -23, 0}
, {79, 73, 59}
, {-23, -50, -33}
, {-36, -6, -3}
, {65, 68, -54}
, {8, 23, 70}
, {50, -67, -75}
, {-2, 7, -4}
, {43, -23, -18}
, {-38, 48, -25}
, {-58, -38, -5}
, {6, -75, 13}
, {51, 65, 66}
, {62, 64, 34}
, {9, 53, -34}
, {35, 57, -16}
, {-90, -25, 2}
, {29, -74, 25}
, {56, -62, 2}
, {-38, 63, -65}
, {3, -27, 39}
, {17, 19, 72}
, {-54, 35, -30}
, {10, -2, -5}
, {-48, -68, -71}
, {51, -48, 36}
}
, {{-43, -26, 47}
, {-63, 73, -53}
, {26, 17, -42}
, {-74, 65, 11}
, {-54, 41, 68}
, {-9, 64, -69}
, {-60, 27, -11}
, {-3, 15, 57}
, {-6, -28, 79}
, {-42, -33, -36}
, {-50, 6, 68}
, {15, -65, 28}
, {-58, -18, -10}
, {-41, 63, -57}
, {43, -23, 67}
, {-24, 64, -66}
, {24, 51, 14}
, {48, 51, 25}
, {13, -24, -26}
, {-46, -5, 34}
, {75, 13, 1}
, {-36, -32, -26}
, {13, -42, -70}
, {51, 42, 28}
, {-2, -53, -15}
, {-42, 61, 24}
, {24, 33, 27}
, {57, -36, -68}
, {46, 28, 59}
, {-5, 26, -26}
, {69, -24, 51}
, {12, -28, -17}
}
, {{-18, -38, 7}
, {64, -14, -53}
, {54, -58, 21}
, {44, -69, 70}
, {33, 6, 23}
, {-53, 42, -29}
, {-16, 68, 76}
, {58, -60, 0}
, {22, 20, 50}
, {-35, -3, -28}
, {48, 18, 63}
, {20, -2, 11}
, {-68, 5, 27}
, {-43, 73, 31}
, {62, 10, -60}
, {40, -21, 24}
, {-12, 60, -59}
, {72, 75, 33}
, {-13, 5, 31}
, {-84, 42, -31}
, {54, -43, 11}
, {40, 16, -1}
, {-25, 28, -2}
, {55, -17, 67}
, {-79, -13, 39}
, {58, -56, -45}
, {-21, 10, -53}
, {45, -41, -25}
, {-12, 24, -1}
, {7, 29, -56}
, {22, -66, -31}
, {64, -76, 19}
}
, {{18, 74, 42}
, {65, 67, -56}
, {37, 58, 57}
, {-27, 46, -75}
, {-67, 63, 49}
, {61, 71, -30}
, {59, 21, 67}
, {66, 3, -48}
, {56, 63, -2}
, {19, 39, -46}
, {-34, 40, 71}
, {72, -56, 51}
, {23, -41, 68}
, {35, 30, 15}
, {-73, 56, 55}
, {-11, -69, 6}
, {20, 10, -63}
, {-7, 42, -27}
, {9, 28, -10}
, {-77, -26, -59}
, {-11, -34, -6}
, {66, -62, 36}
, {52, -3, -49}
, {-48, -12, -14}
, {0, -54, 34}
, {-13, -13, -22}
, {15, -16, 30}
, {17, 50, 5}
, {8, -2, -30}
, {64, 54, 61}
, {36, -17, 57}
, {-68, 1, 81}
}
, {{-52, 59, -58}
, {28, -31, -80}
, {-57, -6, -14}
, {-58, 17, 63}
, {-69, -4, -6}
, {-7, 70, 50}
, {-69, -64, 16}
, {31, -43, 55}
, {41, -74, 49}
, {-45, 47, 78}
, {13, 23, 62}
, {-1, -68, 34}
, {-4, -43, -11}
, {33, 3, -30}
, {-20, 8, 18}
, {-25, -64, 63}
, {-49, 14, 0}
, {47, 33, 17}
, {13, -73, 44}
, {38, 14, 20}
, {-4, 70, -23}
, {-54, -23, 68}
, {-58, 49, 4}
, {38, 38, -30}
, {0, 43, 13}
, {-23, -21, -43}
, {79, 7, -39}
, {-23, 11, 60}
, {33, 6, 7}
, {-53, -18, -32}
, {3, 22, 56}
, {-32, 21, -25}
}
, {{50, 10, -58}
, {64, -28, 19}
, {39, 32, -26}
, {-31, 69, 39}
, {28, -63, -9}
, {-30, 11, -15}
, {-51, -29, -5}
, {-5, -12, 57}
, {18, 27, 48}
, {57, -37, -57}
, {-50, 71, -34}
, {-72, -9, 10}
, {39, -17, 0}
, {28, -38, -68}
, {18, 15, 74}
, {7, -68, -64}
, {77, 45, -31}
, {25, -43, 33}
, {15, 78, 49}
, {-44, -64, 25}
, {-76, 47, 13}
, {-73, -13, -25}
, {-48, 4, 64}
, {39, -6, 0}
, {15, 66, -11}
, {-59, -40, -52}
, {-15, -63, -22}
, {62, 30, -41}
, {48, 25, 5}
, {-77, 58, 34}
, {-7, 68, 52}
, {2, -75, -16}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE