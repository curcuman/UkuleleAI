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


const int16_t conv1d_4_bias[CONV_FILTERS] = {-8, 14, -11, 9, 6, -15, -8, -1, -8, -4, 10, 5, -10, 1, 11, 5}
;

const int16_t conv1d_4_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-72, 70, -40, 84, 71, 55, 6, 72, 62, 92}
}
, {{12, -70, -24, -36, -39, -25, -68, 74, -25, -10}
}
, {{87, 97, 3, 5, 34, 33, -65, 37, 51, -98}
}
, {{10, -82, 0, 87, 11, -22, 79, -54, -84, 41}
}
, {{-38, 32, 61, 81, -51, 17, -72, -6, 41, -64}
}
, {{49, 55, -13, 22, 17, 14, 56, -57, 85, 61}
}
, {{85, 54, -88, -63, 13, -68, -48, 9, 57, 31}
}
, {{51, 81, 53, -35, -47, 68, 71, 43, 69, -82}
}
, {{19, 73, -84, -10, -72, 37, 10, -53, -62, -99}
}
, {{-11, 85, 21, -22, -90, -45, 51, 8, 69, -20}
}
, {{68, 41, -62, 58, 82, 12, -30, 61, -22, 16}
}
, {{35, -65, 50, -2, 85, -69, 0, 79, -87, 59}
}
, {{31, -94, 95, 82, -49, 12, 49, 12, -30, -24}
}
, {{-84, 35, 59, -59, -58, -45, -33, 40, -18, -12}
}
, {{84, -84, -48, -91, -69, -9, 42, 23, -40, -85}
}
, {{76, -20, 92, 98, 90, -43, -27, 14, 44, -62}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE