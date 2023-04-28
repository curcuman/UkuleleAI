/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    1
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  10


const int16_t conv1d_52_bias[CONV_FILTERS] = {9, -13, 23, -1, -2, 18, 24, -1, 0, 19, 21, 24, 9, 28, -14, 4}
;

const int16_t conv1d_52_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{36, -78, 56, 92, -43, -53, 77, -60, 60, 22}
}
, {{-17, -60, -18, -10, 97, -29, -12, 0, 62, 27}
}
, {{-84, -5, 19, -86, -25, 21, -63, -108, 15, -8}
}
, {{-90, 0, 22, 4, -36, 109, 64, -40, 2, 26}
}
, {{108, 75, 57, -92, -109, -86, -24, -59, 25, 10}
}
, {{72, 73, 11, 14, -67, 81, -30, 61, 61, -12}
}
, {{-8, -115, 6, -65, -108, 0, -5, -69, -45, -33}
}
, {{78, -39, 60, -41, -9, 68, -63, 40, 1, 2}
}
, {{3, -90, 27, 71, 94, 111, 61, 100, 104, 20}
}
, {{-60, 18, 105, 77, 6, 63, 48, 51, -39, 18}
}
, {{15, -13, 34, 46, 27, 13, 86, 32, -15, -86}
}
, {{-6, -54, 0, -44, 7, -44, 51, 101, 61, -78}
}
, {{56, 94, 38, 63, -59, 17, 68, -89, 38, 94}
}
, {{-136, -55, 31, 96, 82, 122, 110, 43, -8, 22}
}
, {{58, 30, 73, 57, -53, 48, 86, -75, -99, 17}
}
, {{117, 24, 97, 40, -99, -93, -94, -126, 0, -72}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE