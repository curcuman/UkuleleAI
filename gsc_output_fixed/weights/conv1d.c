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


const int16_t conv1d_bias[CONV_FILTERS] = {-23, -19, -24, -16, 10, 3, 5, -20, -17, -16, -2, -4, -25, -26, -9, -30}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{79, 59, -39, 4, -70, -40, 15, -38, -110, -20}
}
, {{-46, 12, 90, -15, -71, -55, -56, -73, 57, 24}
}
, {{113, -3, -50, 27, -48, -50, 30, -58, -24, -27}
}
, {{108, 61, -49, -100, -70, -97, 50, -2, 102, 79}
}
, {{-21, 90, 44, 94, -51, -83, 4, -11, -55, -34}
}
, {{96, 15, 9, -56, 73, -95, 67, 78, -5, -18}
}
, {{-83, -38, 73, -58, 30, 41, -40, -35, -6, 108}
}
, {{26, 79, 96, -49, 18, 104, 48, -5, -4, -110}
}
, {{-100, -60, -4, 30, -8, -45, -64, -70, -62, -45}
}
, {{45, 52, -101, 0, -5, -36, 87, -48, -38, 79}
}
, {{61, -53, -34, 14, -82, -105, -57, 79, -55, 126}
}
, {{-51, -81, -45, -24, -38, 82, 94, 26, -12, -54}
}
, {{-32, -54, 52, -24, 63, -91, 23, 70, 53, -98}
}
, {{39, 22, -96, -96, -6, 52, 71, -70, 81, 2}
}
, {{-66, -62, -18, -66, -59, 12, -2, 43, 1, 124}
}
, {{-41, -81, 56, 61, -17, 63, 9, -65, -12, -2}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE