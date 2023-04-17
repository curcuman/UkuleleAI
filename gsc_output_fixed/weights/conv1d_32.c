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


const int16_t conv1d_32_bias[CONV_FILTERS] = {2, -5, -16, -15, 33, -25, -39, -7, 7, -23, -16, -40, 21, -29, -6, -12}
;

const int16_t conv1d_32_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{110, 0, 23, -52, 88, 64, -56, -83, 89, 78}
}
, {{86, -17, 15, -24, 0, -31, -51, -66, -41, -107}
}
, {{51, -72, -6, -27, -92, 49, -91, 26, 39, 56}
}
, {{-11, -51, 56, 28, -87, -17, -80, -82, 66, 99}
}
, {{-5, 0, -39, -2, 0, 50, -61, -23, -72, -99}
}
, {{-102, -30, 37, -17, -69, -49, 58, 46, 71, 92}
}
, {{27, 103, 53, -9, -1, -62, 4, -50, 22, -101}
}
, {{-40, 0, -45, 32, 13, 3, 7, -91, 20, -91}
}
, {{87, -50, -58, -24, 83, -70, -28, 51, -34, -27}
}
, {{74, -1, 65, 54, -46, -4, 10, 32, 95, 105}
}
, {{-66, -45, -65, -78, 10, -27, -55, -63, -29, 61}
}
, {{-90, -76, -21, -6, 70, -16, -35, -14, 14, 74}
}
, {{-17, 84, 4, 46, 96, -32, -6, 19, 82, 83}
}
, {{69, 74, -25, 48, -95, -96, 53, 37, 0, 9}
}
, {{-44, 91, 92, 16, 84, -59, -89, -95, -100, -27}
}
, {{-85, -55, -56, 0, -3, -46, 55, 80, -53, 54}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE