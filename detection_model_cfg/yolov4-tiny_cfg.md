Interpreter has 248 tensors and 152 nodes
Inputs: 0
Outputs: 233 212

Tensor   0 input_1              kTfLiteFloat32  kTfLiteArenaRw    2076672 bytes ( 2.0 MB)  1 416 416 3
Tensor   1 functional_1/tf_op_layer_AddV2_2/AddV2/y;StatefulPartitionedCall/functional_1/tf_op_layer_AddV2_2/AddV2/y kTfLiteFloat32   kTfLiteMmapRo       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor   2 functional_1/tf_op_layer_AddV2_5/AddV2_3/y;StatefulPartitionedCall/functional_1/tf_op_layer_AddV2_5/AddV2_3/y kTfLiteFloat32   kTfLiteMmapRo       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor   3 functional_1/tf_op_layer_Mul_16/Mul_16/y;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_16/Mul_16/y kTfLiteFloat32   kTfLiteMmapRo          8 bytes ( 0.0 MB)  2
Tensor   4 functional_1/tf_op_layer_Mul_17/Mul_17/y;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_17/Mul_17/y kTfLiteFloat32   kTfLiteMmapRo          8 bytes ( 0.0 MB)  2
Tensor   5 functional_1/tf_op_layer_Mul_23/Mul_23/y;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_23/Mul_23/y kTfLiteFloat32   kTfLiteMmapRo          4 bytes ( 0.0 MB) 
Tensor   6 functional_1/tf_op_layer_Mul_3/Mul_3/y;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_3/Mul_3/y kTfLiteFloat32   kTfLiteMmapRo          8 bytes ( 0.0 MB)  2
Tensor   7 functional_1/tf_op_layer_Mul_4/Mul_4/y;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_4/Mul_4/y kTfLiteFloat32   kTfLiteMmapRo          8 bytes ( 0.0 MB)  2
Tensor   8 functional_1/tf_op_layer_Mul_5/Mul_5/y;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_5/Mul_5/y kTfLiteFloat32   kTfLiteMmapRo          8 bytes ( 0.0 MB)  2
Tensor   9 functional_1/tf_op_layer_Mul_8/Mul_8/y;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_8/Mul_8/y kTfLiteFloat32   kTfLiteMmapRo          4 bytes ( 0.0 MB) 
Tensor  10 functional_1/tf_op_layer_Mul_9/Mul_9/y;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_9/Mul_9/y kTfLiteFloat32   kTfLiteMmapRo          4 bytes ( 0.0 MB) 
Tensor  11 functional_1/tf_op_layer_Reshape_8/Reshape_8/shape;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_8/Reshape_8/shape kTfLiteInt32   kTfLiteMmapRo         12 bytes ( 0.0 MB)  3
Tensor  12 functional_1/tf_op_layer_Reshape_9/Reshape_9/shape;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_9/Reshape_9/shape kTfLiteInt32   kTfLiteMmapRo         12 bytes ( 0.0 MB)  3
Tensor  13 functional_1/tf_op_layer_ResizeBilinear/ResizeBilinear/size;StatefulPartitionedCall/functional_1/tf_op_layer_ResizeBilinear/ResizeBilinear/size kTfLiteInt32   kTfLiteMmapRo          8 bytes ( 0.0 MB)  2
Tensor  14 functional_1/tf_op_layer_Sub_5/Sub_5/y;StatefulPartitionedCall/functional_1/tf_op_layer_Sub_5/Sub_5/y kTfLiteFloat32   kTfLiteMmapRo          4 bytes ( 0.0 MB) 
Tensor  15 functional_1/tf_op_layer_split_4/split_4/size_splits;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_4/size_splits kTfLiteInt32   kTfLiteMmapRo         36 bytes ( 0.0 MB)  9
Tensor  16 functional_1/tf_op_layer_split_4/split_4/split_dim;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_4/split_dim kTfLiteInt32   kTfLiteMmapRo          4 bytes ( 0.0 MB) 
Tensor  17 functional_1/zero_padding2d_1/Pad/paddings;StatefulPartitionedCall/functional_1/zero_padding2d_1/Pad/paddings kTfLiteInt32   kTfLiteMmapRo         32 bytes ( 0.0 MB)  4 2
Tensor  18 functional_1/conv2d/Conv2D;StatefulPartitionedCall/functional_1/conv2d/Conv2D kTfLiteFloat32   kTfLiteMmapRo       3456 bytes ( 0.0 MB)  32 3 3 3
Tensor  19 functional_1/conv2d_1/Conv2D;StatefulPartitionedCall/functional_1/conv2d_1/Conv2D kTfLiteFloat32   kTfLiteMmapRo      73728 bytes ( 0.1 MB)  64 3 3 32
Tensor  20 functional_1/conv2d_2/Conv2D;StatefulPartitionedCall/functional_1/conv2d_2/Conv2D kTfLiteFloat32   kTfLiteMmapRo     147456 bytes ( 0.1 MB)  64 3 3 64
Tensor  21 functional_1/conv2d_3/Conv2D;StatefulPartitionedCall/functional_1/conv2d_3/Conv2D kTfLiteFloat32   kTfLiteMmapRo      36864 bytes ( 0.0 MB)  32 3 3 32
Tensor  22 functional_1/conv2d_4/Conv2D;StatefulPartitionedCall/functional_1/conv2d_4/Conv2D kTfLiteFloat32   kTfLiteMmapRo      36864 bytes ( 0.0 MB)  32 3 3 32
Tensor  23 functional_1/conv2d_5/Conv2D;StatefulPartitionedCall/functional_1/conv2d_5/Conv2D kTfLiteFloat32   kTfLiteMmapRo      16384 bytes ( 0.0 MB)  64 1 1 64
Tensor  24 functional_1/conv2d_6/Conv2D;StatefulPartitionedCall/functional_1/conv2d_6/Conv2D kTfLiteFloat32   kTfLiteMmapRo     589824 bytes ( 0.6 MB)  128 3 3 128
Tensor  25 functional_1/conv2d_7/Conv2D;StatefulPartitionedCall/functional_1/conv2d_7/Conv2D kTfLiteFloat32   kTfLiteMmapRo     147456 bytes ( 0.1 MB)  64 3 3 64
Tensor  26 functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D kTfLiteFloat32   kTfLiteMmapRo     147456 bytes ( 0.1 MB)  64 3 3 64
Tensor  27 functional_1/conv2d_9/Conv2D;StatefulPartitionedCall/functional_1/conv2d_9/Conv2D kTfLiteFloat32   kTfLiteMmapRo      65536 bytes ( 0.1 MB)  128 1 1 128
Tensor  28 functional_1/conv2d_10/Conv2D;StatefulPartitionedCall/functional_1/conv2d_10/Conv2D kTfLiteFloat32   kTfLiteMmapRo    2359296 bytes ( 2.2 MB)  256 3 3 256
Tensor  29 functional_1/conv2d_11/Conv2D;StatefulPartitionedCall/functional_1/conv2d_11/Conv2D kTfLiteFloat32   kTfLiteMmapRo     589824 bytes ( 0.6 MB)  128 3 3 128
Tensor  30 functional_1/conv2d_12/Conv2D;StatefulPartitionedCall/functional_1/conv2d_12/Conv2D kTfLiteFloat32   kTfLiteMmapRo     589824 bytes ( 0.6 MB)  128 3 3 128
Tensor  31 functional_1/conv2d_13/Conv2D;StatefulPartitionedCall/functional_1/conv2d_13/Conv2D kTfLiteFloat32   kTfLiteMmapRo     262144 bytes ( 0.2 MB)  256 1 1 256
Tensor  32 functional_1/conv2d_14/Conv2D;StatefulPartitionedCall/functional_1/conv2d_14/Conv2D kTfLiteFloat32   kTfLiteMmapRo    9437184 bytes ( 9.0 MB)  512 3 3 512
Tensor  33 functional_1/conv2d_15/Conv2D;StatefulPartitionedCall/functional_1/conv2d_15/Conv2D kTfLiteFloat32   kTfLiteMmapRo     524288 bytes ( 0.5 MB)  256 1 1 512
Tensor  34 functional_1/conv2d_16/Conv2D;StatefulPartitionedCall/functional_1/conv2d_16/Conv2D kTfLiteFloat32   kTfLiteMmapRo    4718592 bytes ( 4.5 MB)  512 3 3 256
Tensor  35 functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D kTfLiteFloat32   kTfLiteMmapRo     131072 bytes ( 0.1 MB)  128 1 1 256
Tensor  36 functional_1/conv2d_19/Conv2D;StatefulPartitionedCall/functional_1/conv2d_19/Conv2D kTfLiteFloat32   kTfLiteMmapRo    3538944 bytes ( 3.4 MB)  256 3 3 384
Tensor  37 functional_1/conv2d_17/Conv2D;StatefulPartitionedCall/functional_1/conv2d_17/Conv2D kTfLiteFloat32   kTfLiteMmapRo     522240 bytes ( 0.5 MB)  255 1 1 512
Tensor  38 functional_1/conv2d_20/Conv2D;StatefulPartitionedCall/functional_1/conv2d_20/Conv2D kTfLiteFloat32   kTfLiteMmapRo     261120 bytes ( 0.2 MB)  255 1 1 256
Tensor  39 functional_1/batch_normalization/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization/FusedBatchNormV3;functional_1/conv2d_4/Conv2D;StatefulPartitionedCall/functional_1/conv2d_4/Conv2D;functional_1/conv2d/Conv2D;StatefulPartitionedCall/functional_1/conv2d/Conv2D kTfLiteFloat32   kTfLiteMmapRo        128 bytes ( 0.0 MB)  32
Tensor  40 functional_1/batch_normalization_1/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_1/FusedBatchNormV3;functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D;functional_1/conv2d_1/Conv2D;StatefulPartitionedCall/functional_1/conv2d_1/Conv2D kTfLiteFloat32   kTfLiteMmapRo        256 bytes ( 0.0 MB)  64
Tensor  41 functional_1/batch_normalization_2/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_2/FusedBatchNormV3;functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D;functional_1/conv2d_2/Conv2D;StatefulPartitionedCall/functional_1/conv2d_2/Conv2D kTfLiteFloat32   kTfLiteMmapRo        256 bytes ( 0.0 MB)  64
Tensor  42 functional_1/batch_normalization_3/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_3/FusedBatchNormV3;functional_1/conv2d_4/Conv2D;StatefulPartitionedCall/functional_1/conv2d_4/Conv2D;functional_1/conv2d_3/Conv2D;StatefulPartitionedCall/functional_1/conv2d_3/Conv2D kTfLiteFloat32   kTfLiteMmapRo        128 bytes ( 0.0 MB)  32
Tensor  43 functional_1/batch_normalization_4/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_4/FusedBatchNormV3;functional_1/conv2d_4/Conv2D;StatefulPartitionedCall/functional_1/conv2d_4/Conv2D kTfLiteFloat32   kTfLiteMmapRo        128 bytes ( 0.0 MB)  32
Tensor  44 functional_1/batch_normalization_5/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_5/FusedBatchNormV3;functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D;functional_1/conv2d_5/Conv2D;StatefulPartitionedCall/functional_1/conv2d_5/Conv2D kTfLiteFloat32   kTfLiteMmapRo        256 bytes ( 0.0 MB)  64
Tensor  45 functional_1/batch_normalization_6/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_6/FusedBatchNormV3;functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D;functional_1/conv2d_6/Conv2D;StatefulPartitionedCall/functional_1/conv2d_6/Conv2D kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  46 functional_1/batch_normalization_7/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_7/FusedBatchNormV3;functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D;functional_1/conv2d_7/Conv2D;StatefulPartitionedCall/functional_1/conv2d_7/Conv2D kTfLiteFloat32   kTfLiteMmapRo        256 bytes ( 0.0 MB)  64
Tensor  47 functional_1/batch_normalization_8/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_8/FusedBatchNormV3;functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D kTfLiteFloat32   kTfLiteMmapRo        256 bytes ( 0.0 MB)  64
Tensor  48 functional_1/batch_normalization_9/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_9/FusedBatchNormV3;functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D;functional_1/conv2d_9/Conv2D;StatefulPartitionedCall/functional_1/conv2d_9/Conv2D kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  49 functional_1/batch_normalization_10/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_10/FusedBatchNormV3;functional_1/conv2d_19/Conv2D;StatefulPartitionedCall/functional_1/conv2d_19/Conv2D;functional_1/conv2d_10/Conv2D;StatefulPartitionedCall/functional_1/conv2d_10/Conv2D kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  50 functional_1/batch_normalization_11/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_11/FusedBatchNormV3;functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D;functional_1/conv2d_11/Conv2D;StatefulPartitionedCall/functional_1/conv2d_11/Conv2D kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  51 functional_1/batch_normalization_12/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_12/FusedBatchNormV3;functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D;functional_1/conv2d_12/Conv2D;StatefulPartitionedCall/functional_1/conv2d_12/Conv2D kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  52 functional_1/batch_normalization_13/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_13/FusedBatchNormV3;functional_1/conv2d_19/Conv2D;StatefulPartitionedCall/functional_1/conv2d_19/Conv2D;functional_1/conv2d_13/Conv2D;StatefulPartitionedCall/functional_1/conv2d_13/Conv2D kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  53 functional_1/batch_normalization_14/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_14/FusedBatchNormV3;functional_1/conv2d_16/Conv2D;StatefulPartitionedCall/functional_1/conv2d_16/Conv2D;functional_1/conv2d_14/Conv2D;StatefulPartitionedCall/functional_1/conv2d_14/Conv2D kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  54 functional_1/batch_normalization_15/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_15/FusedBatchNormV3;functional_1/conv2d_19/Conv2D;StatefulPartitionedCall/functional_1/conv2d_19/Conv2D;functional_1/conv2d_15/Conv2D;StatefulPartitionedCall/functional_1/conv2d_15/Conv2D kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  55 functional_1/batch_normalization_16/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_16/FusedBatchNormV3;functional_1/conv2d_16/Conv2D;StatefulPartitionedCall/functional_1/conv2d_16/Conv2D kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  56 functional_1/batch_normalization_17/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_17/FusedBatchNormV3;functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  57 functional_1/batch_normalization_18/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_18/FusedBatchNormV3;functional_1/conv2d_19/Conv2D;StatefulPartitionedCall/functional_1/conv2d_19/Conv2D kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  58 functional_1/conv2d_17/BiasAdd;StatefulPartitionedCall/functional_1/conv2d_17/BiasAdd;functional_1/conv2d_20/Conv2D;StatefulPartitionedCall/functional_1/conv2d_20/Conv2D;functional_1/conv2d_17/Conv2D;StatefulPartitionedCall/functional_1/conv2d_17/Conv2D;unknown_95 kTfLiteFloat32   kTfLiteMmapRo       1020 bytes ( 0.0 MB)  255
Tensor  59 functional_1/conv2d_20/BiasAdd;StatefulPartitionedCall/functional_1/conv2d_20/BiasAdd;functional_1/conv2d_20/Conv2D;StatefulPartitionedCall/functional_1/conv2d_20/Conv2D;unknown_97 kTfLiteFloat32   kTfLiteMmapRo       1020 bytes ( 0.0 MB)  255
Tensor  60 functional_1/tf_op_layer_strided_slice_5/strided_slice_5;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_5/strided_slice_5 kTfLiteInt32   kTfLiteMmapRo         16 bytes ( 0.0 MB)  4
Tensor  61 functional_1/tf_op_layer_strided_slice_5/strided_slice_5;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_5/strided_slice_51 kTfLiteInt32   kTfLiteMmapRo         16 bytes ( 0.0 MB)  4
Tensor  62 functional_1/tf_op_layer_strided_slice_5/strided_slice_5;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_5/strided_slice_52 kTfLiteInt32   kTfLiteMmapRo         16 bytes ( 0.0 MB)  4
Tensor  63 functional_1/zero_padding2d/Pad;StatefulPartitionedCall/functional_1/zero_padding2d/Pad kTfLiteFloat32  kTfLiteArenaRw    2086668 bytes ( 2.0 MB)  1 417 417 3
Tensor  64 functional_1/batch_normalization/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization/FusedBatchNormV3;functional_1/conv2d_4/Conv2D;StatefulPartitionedCall/functional_1/conv2d_4/Conv2D;functional_1/conv2d/Conv2D;StatefulPartitionedCall/functional_1/conv2d/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw    5537792 bytes ( 5.3 MB)  1 208 208 32
Tensor  65 functional_1/tf_op_layer_LeakyRelu/LeakyRelu;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu/LeakyRelu kTfLiteFloat32  kTfLiteArenaRw    5537792 bytes ( 5.3 MB)  1 208 208 32
Tensor  66 functional_1/zero_padding2d_1/Pad;StatefulPartitionedCall/functional_1/zero_padding2d_1/Pad kTfLiteFloat32  kTfLiteArenaRw    5591168 bytes ( 5.3 MB)  1 209 209 32
Tensor  67 functional_1/batch_normalization_1/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_1/FusedBatchNormV3;functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D;functional_1/conv2d_1/Conv2D;StatefulPartitionedCall/functional_1/conv2d_1/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw    2768896 bytes ( 2.6 MB)  1 104 104 64
Tensor  68 functional_1/tf_op_layer_LeakyRelu_1/LeakyRelu_1;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_1/LeakyRelu_1 kTfLiteFloat32  kTfLiteArenaRw    2768896 bytes ( 2.6 MB)  1 104 104 64
Tensor  69 functional_1/batch_normalization_2/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_2/FusedBatchNormV3;functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D;functional_1/conv2d_2/Conv2D;StatefulPartitionedCall/functional_1/conv2d_2/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw    2768896 bytes ( 2.6 MB)  1 104 104 64
Tensor  70 functional_1/tf_op_layer_LeakyRelu_2/LeakyRelu_2;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_2/LeakyRelu_2 kTfLiteFloat32  kTfLiteArenaRw    2768896 bytes ( 2.6 MB)  1 104 104 64
Tensor  71 functional_1/tf_op_layer_split/split;StatefulPartitionedCall/functional_1/tf_op_layer_split/split kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 104 104 32
Tensor  72 functional_1/tf_op_layer_split/split;StatefulPartitionedCall/functional_1/tf_op_layer_split/split1 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 104 104 32
Tensor  73 functional_1/batch_normalization_3/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_3/FusedBatchNormV3;functional_1/conv2d_4/Conv2D;StatefulPartitionedCall/functional_1/conv2d_4/Conv2D;functional_1/conv2d_3/Conv2D;StatefulPartitionedCall/functional_1/conv2d_3/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 104 104 32
Tensor  74 functional_1/tf_op_layer_LeakyRelu_3/LeakyRelu_3;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_3/LeakyRelu_3 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 104 104 32
Tensor  75 functional_1/batch_normalization_4/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_4/FusedBatchNormV3;functional_1/conv2d_4/Conv2D;StatefulPartitionedCall/functional_1/conv2d_4/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 104 104 32
Tensor  76 functional_1/tf_op_layer_LeakyRelu_4/LeakyRelu_4;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_4/LeakyRelu_4 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 104 104 32
Tensor  77 functional_1/tf_op_layer_concat/concat;StatefulPartitionedCall/functional_1/tf_op_layer_concat/concat kTfLiteFloat32  kTfLiteArenaRw    2768896 bytes ( 2.6 MB)  1 104 104 64
Tensor  78 functional_1/batch_normalization_5/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_5/FusedBatchNormV3;functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D;functional_1/conv2d_5/Conv2D;StatefulPartitionedCall/functional_1/conv2d_5/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw    2768896 bytes ( 2.6 MB)  1 104 104 64
Tensor  79 functional_1/tf_op_layer_LeakyRelu_5/LeakyRelu_5;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_5/LeakyRelu_5 kTfLiteFloat32  kTfLiteArenaRw    2768896 bytes ( 2.6 MB)  1 104 104 64
Tensor  80 functional_1/tf_op_layer_concat_1/concat_1;StatefulPartitionedCall/functional_1/tf_op_layer_concat_1/concat_1 kTfLiteFloat32  kTfLiteArenaRw    5537792 bytes ( 5.3 MB)  1 104 104 128
Tensor  81 functional_1/max_pooling2d/MaxPool;StatefulPartitionedCall/functional_1/max_pooling2d/MaxPool kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 52 52 128
Tensor  82 functional_1/batch_normalization_6/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_6/FusedBatchNormV3;functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D;functional_1/conv2d_6/Conv2D;StatefulPartitionedCall/functional_1/conv2d_6/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 52 52 128
Tensor  83 functional_1/tf_op_layer_LeakyRelu_6/LeakyRelu_6;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_6/LeakyRelu_6 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 52 52 128
Tensor  84 functional_1/tf_op_layer_split_1/split_1;StatefulPartitionedCall/functional_1/tf_op_layer_split_1/split_1 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 52 52 64
Tensor  85 functional_1/tf_op_layer_split_1/split_1;StatefulPartitionedCall/functional_1/tf_op_layer_split_1/split_11 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 52 52 64
Tensor  86 functional_1/batch_normalization_7/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_7/FusedBatchNormV3;functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D;functional_1/conv2d_7/Conv2D;StatefulPartitionedCall/functional_1/conv2d_7/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 52 52 64
Tensor  87 functional_1/tf_op_layer_LeakyRelu_7/LeakyRelu_7;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_7/LeakyRelu_7 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 52 52 64
Tensor  88 functional_1/batch_normalization_8/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_8/FusedBatchNormV3;functional_1/conv2d_8/Conv2D;StatefulPartitionedCall/functional_1/conv2d_8/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 52 52 64
Tensor  89 functional_1/tf_op_layer_LeakyRelu_8/LeakyRelu_8;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_8/LeakyRelu_8 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 52 52 64
Tensor  90 functional_1/tf_op_layer_concat_2/concat_2;StatefulPartitionedCall/functional_1/tf_op_layer_concat_2/concat_2 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 52 52 128
Tensor  91 functional_1/batch_normalization_9/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_9/FusedBatchNormV3;functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D;functional_1/conv2d_9/Conv2D;StatefulPartitionedCall/functional_1/conv2d_9/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 52 52 128
Tensor  92 functional_1/tf_op_layer_LeakyRelu_9/LeakyRelu_9;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_9/LeakyRelu_9 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 52 52 128
Tensor  93 functional_1/tf_op_layer_concat_3/concat_3;StatefulPartitionedCall/functional_1/tf_op_layer_concat_3/concat_3 kTfLiteFloat32  kTfLiteArenaRw    2768896 bytes ( 2.6 MB)  1 52 52 256
Tensor  94 functional_1/max_pooling2d_1/MaxPool;StatefulPartitionedCall/functional_1/max_pooling2d_1/MaxPool kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 26 26 256
Tensor  95 functional_1/batch_normalization_10/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_10/FusedBatchNormV3;functional_1/conv2d_19/Conv2D;StatefulPartitionedCall/functional_1/conv2d_19/Conv2D;functional_1/conv2d_10/Conv2D;StatefulPartitionedCall/functional_1/conv2d_10/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 26 26 256
Tensor  96 functional_1/tf_op_layer_LeakyRelu_10/LeakyRelu_10;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_10/LeakyRelu_10 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 26 26 256
Tensor  97 functional_1/tf_op_layer_split_2/split_2;StatefulPartitionedCall/functional_1/tf_op_layer_split_2/split_2 kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 26 26 128
Tensor  98 functional_1/tf_op_layer_split_2/split_2;StatefulPartitionedCall/functional_1/tf_op_layer_split_2/split_21 kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 26 26 128
Tensor  99 functional_1/batch_normalization_11/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_11/FusedBatchNormV3;functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D;functional_1/conv2d_11/Conv2D;StatefulPartitionedCall/functional_1/conv2d_11/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 26 26 128
Tensor 100 functional_1/tf_op_layer_LeakyRelu_11/LeakyRelu_11;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_11/LeakyRelu_11 kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 26 26 128
Tensor 101 functional_1/batch_normalization_12/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_12/FusedBatchNormV3;functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D;functional_1/conv2d_12/Conv2D;StatefulPartitionedCall/functional_1/conv2d_12/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 26 26 128
Tensor 102 functional_1/tf_op_layer_LeakyRelu_12/LeakyRelu_12;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_12/LeakyRelu_12 kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 26 26 128
Tensor 103 functional_1/tf_op_layer_concat_4/concat_4;StatefulPartitionedCall/functional_1/tf_op_layer_concat_4/concat_4 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 26 26 256
Tensor 104 functional_1/batch_normalization_13/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_13/FusedBatchNormV3;functional_1/conv2d_19/Conv2D;StatefulPartitionedCall/functional_1/conv2d_19/Conv2D;functional_1/conv2d_13/Conv2D;StatefulPartitionedCall/functional_1/conv2d_13/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 26 26 256
Tensor 105 functional_1/tf_op_layer_LeakyRelu_13/LeakyRelu_13;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_13/LeakyRelu_13 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 26 26 256
Tensor 106 functional_1/tf_op_layer_concat_5/concat_5;StatefulPartitionedCall/functional_1/tf_op_layer_concat_5/concat_5 kTfLiteFloat32  kTfLiteArenaRw    1384448 bytes ( 1.3 MB)  1 26 26 512
Tensor 107 functional_1/max_pooling2d_2/MaxPool;StatefulPartitionedCall/functional_1/max_pooling2d_2/MaxPool kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 13 13 512
Tensor 108 functional_1/batch_normalization_14/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_14/FusedBatchNormV3;functional_1/conv2d_16/Conv2D;StatefulPartitionedCall/functional_1/conv2d_16/Conv2D;functional_1/conv2d_14/Conv2D;StatefulPartitionedCall/functional_1/conv2d_14/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 13 13 512
Tensor 109 functional_1/tf_op_layer_LeakyRelu_14/LeakyRelu_14;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_14/LeakyRelu_14 kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 13 13 512
Tensor 110 functional_1/batch_normalization_15/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_15/FusedBatchNormV3;functional_1/conv2d_19/Conv2D;StatefulPartitionedCall/functional_1/conv2d_19/Conv2D;functional_1/conv2d_15/Conv2D;StatefulPartitionedCall/functional_1/conv2d_15/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw     173056 bytes ( 0.2 MB)  1 13 13 256
Tensor 111 functional_1/tf_op_layer_LeakyRelu_15/LeakyRelu_15;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_15/LeakyRelu_15 kTfLiteFloat32  kTfLiteArenaRw     173056 bytes ( 0.2 MB)  1 13 13 256
Tensor 112 functional_1/batch_normalization_16/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_16/FusedBatchNormV3;functional_1/conv2d_16/Conv2D;StatefulPartitionedCall/functional_1/conv2d_16/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 13 13 512
Tensor 113 functional_1/tf_op_layer_LeakyRelu_16/LeakyRelu_16;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_16/LeakyRelu_16 kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 13 13 512
Tensor 114 functional_1/batch_normalization_17/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_17/FusedBatchNormV3;functional_1/conv2d_18/Conv2D;StatefulPartitionedCall/functional_1/conv2d_18/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw      86528 bytes ( 0.1 MB)  1 13 13 128
Tensor 115 functional_1/tf_op_layer_LeakyRelu_17/LeakyRelu_17;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_17/LeakyRelu_17 kTfLiteFloat32  kTfLiteArenaRw      86528 bytes ( 0.1 MB)  1 13 13 128
Tensor 116 functional_1/tf_op_layer_ResizeBilinear/ResizeBilinear;StatefulPartitionedCall/functional_1/tf_op_layer_ResizeBilinear/ResizeBilinear kTfLiteFloat32  kTfLiteArenaRw     346112 bytes ( 0.3 MB)  1 26 26 128
Tensor 117 functional_1/tf_op_layer_concat_6/concat_6;StatefulPartitionedCall/functional_1/tf_op_layer_concat_6/concat_6 kTfLiteFloat32  kTfLiteArenaRw    1038336 bytes ( 1.0 MB)  1 26 26 384
Tensor 118 functional_1/batch_normalization_18/FusedBatchNormV3;StatefulPartitionedCall/functional_1/batch_normalization_18/FusedBatchNormV3;functional_1/conv2d_19/Conv2D;StatefulPartitionedCall/functional_1/conv2d_19/Conv2D1 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 26 26 256
Tensor 119 functional_1/tf_op_layer_LeakyRelu_18/LeakyRelu_18;StatefulPartitionedCall/functional_1/tf_op_layer_LeakyRelu_18/LeakyRelu_18 kTfLiteFloat32  kTfLiteArenaRw     692224 bytes ( 0.7 MB)  1 26 26 256
Tensor 120 functional_1/conv2d_17/BiasAdd;StatefulPartitionedCall/functional_1/conv2d_17/BiasAdd;functional_1/conv2d_20/Conv2D;StatefulPartitionedCall/functional_1/conv2d_20/Conv2D;functional_1/conv2d_17/Conv2D;StatefulPartitionedCall/functional_1/conv2d_17/Conv2D;unknown_951 kTfLiteFloat32  kTfLiteArenaRw     172380 bytes ( 0.2 MB)  1 13 13 255
Tensor 121 functional_1/tf_op_layer_split_4/split_4;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_4 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 122 functional_1/tf_op_layer_split_4/split_4;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_41 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 123 functional_1/tf_op_layer_split_4/split_4;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_42 kTfLiteFloat32  kTfLiteArenaRw      54756 bytes ( 0.1 MB)  1 13 13 81
Tensor 124 functional_1/tf_op_layer_split_4/split_4;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_43 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 125 functional_1/tf_op_layer_split_4/split_4;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_44 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 126 functional_1/tf_op_layer_split_4/split_4;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_45 kTfLiteFloat32  kTfLiteArenaRw      54756 bytes ( 0.1 MB)  1 13 13 81
Tensor 127 functional_1/tf_op_layer_split_4/split_4;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_46 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 128 functional_1/tf_op_layer_split_4/split_4;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_47 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 129 functional_1/tf_op_layer_split_4/split_4;StatefulPartitionedCall/functional_1/tf_op_layer_split_4/split_48 kTfLiteFloat32  kTfLiteArenaRw      54756 bytes ( 0.1 MB)  1 13 13 81
Tensor 130 functional_1/tf_op_layer_Exp_3/Exp_3;StatefulPartitionedCall/functional_1/tf_op_layer_Exp_3/Exp_3 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 131 functional_1/tf_op_layer_Mul_15/Mul_15;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_15/Mul_15 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 132 functional_1/tf_op_layer_Reshape_12/Reshape_12;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_12/Reshape_12 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 169 2
Tensor 133 functional_1/tf_op_layer_Exp_4/Exp_4;StatefulPartitionedCall/functional_1/tf_op_layer_Exp_4/Exp_4 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 134 functional_1/tf_op_layer_Mul_16/Mul_16;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_16/Mul_16 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 135 functional_1/tf_op_layer_Reshape_13/Reshape_13;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_13/Reshape_13 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 169 2
Tensor 136 functional_1/tf_op_layer_Exp_5/Exp_5;StatefulPartitionedCall/functional_1/tf_op_layer_Exp_5/Exp_5 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 137 functional_1/tf_op_layer_Mul_17/Mul_17;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_17/Mul_17 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 138 functional_1/tf_op_layer_Reshape_14/Reshape_14;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_14/Reshape_14 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 169 2
Tensor 139 functional_1/tf_op_layer_concat_12/concat_12;StatefulPartitionedCall/functional_1/tf_op_layer_concat_12/concat_12 kTfLiteFloat32  kTfLiteArenaRw       4056 bytes ( 0.0 MB)  1 507 2
Tensor 140 functional_1/tf_op_layer_Sigmoid_10/Sigmoid_10;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_10/Sigmoid_10 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 141 functional_1/tf_op_layer_Mul_20/Mul_20;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_20/Mul_20 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 142 functional_1/tf_op_layer_Sub_4/Sub_4;StatefulPartitionedCall/functional_1/tf_op_layer_Sub_4/Sub_4 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 143 functional_1/tf_op_layer_AddV2_4/AddV2_4;StatefulPartitionedCall/functional_1/tf_op_layer_AddV2_4/AddV2_4 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 144 functional_1/tf_op_layer_Mul_21/Mul_21;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_21/Mul_21 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 145 functional_1/tf_op_layer_Reshape_16/Reshape_16;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_16/Reshape_16 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 169 2
Tensor 146 functional_1/tf_op_layer_Sigmoid_11/Sigmoid_11;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_11/Sigmoid_11 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 147 functional_1/tf_op_layer_Mul_22/Mul_22;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_22/Mul_22 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 148 functional_1/tf_op_layer_Sub_5/Sub_5;StatefulPartitionedCall/functional_1/tf_op_layer_Sub_5/Sub_5 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 149 functional_1/tf_op_layer_AddV2_5/AddV2_5;StatefulPartitionedCall/functional_1/tf_op_layer_AddV2_5/AddV2_5 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 150 functional_1/tf_op_layer_Mul_23/Mul_23;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_23/Mul_23 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 151 functional_1/tf_op_layer_Reshape_17/Reshape_17;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_17/Reshape_17 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 169 2
Tensor 152 functional_1/tf_op_layer_Sigmoid_6/Sigmoid_6;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_6/Sigmoid_6 kTfLiteFloat32  kTfLiteArenaRw      54756 bytes ( 0.1 MB)  1 13 13 81
Tensor 153 functional_1/tf_op_layer_strided_slice_6/strided_slice_6;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_6/strided_slice_6 kTfLiteFloat32  kTfLiteArenaRw        676 bytes ( 0.0 MB)  1 13 13 1
Tensor 154 functional_1/tf_op_layer_strided_slice_7/strided_slice_7;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_7/strided_slice_7 kTfLiteFloat32  kTfLiteArenaRw      54080 bytes ( 0.1 MB)  1 13 13 80
Tensor 155 functional_1/tf_op_layer_Mul_12/Mul_12;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_12/Mul_12 kTfLiteFloat32  kTfLiteArenaRw      54080 bytes ( 0.1 MB)  1 13 13 80
Tensor 156 functional_1/tf_op_layer_Reshape_9/Reshape_9;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_9/Reshape_9 kTfLiteFloat32  kTfLiteArenaRw      54080 bytes ( 0.1 MB)  1 169 80
Tensor 157 functional_1/tf_op_layer_Sigmoid_7/Sigmoid_7;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_7/Sigmoid_7 kTfLiteFloat32  kTfLiteArenaRw      54756 bytes ( 0.1 MB)  1 13 13 81
Tensor 158 functional_1/tf_op_layer_strided_slice_8/strided_slice_8;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_8/strided_slice_8 kTfLiteFloat32  kTfLiteArenaRw        676 bytes ( 0.0 MB)  1 13 13 1
Tensor 159 functional_1/tf_op_layer_strided_slice_9/strided_slice_9;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_9/strided_slice_9 kTfLiteFloat32  kTfLiteArenaRw      54080 bytes ( 0.1 MB)  1 13 13 80
Tensor 160 functional_1/tf_op_layer_Mul_13/Mul_13;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_13/Mul_13 kTfLiteFloat32  kTfLiteArenaRw      54080 bytes ( 0.1 MB)  1 13 13 80
Tensor 161 functional_1/tf_op_layer_Reshape_10/Reshape_10;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_10/Reshape_10 kTfLiteFloat32  kTfLiteArenaRw      54080 bytes ( 0.1 MB)  1 169 80
Tensor 162 functional_1/tf_op_layer_Sigmoid_8/Sigmoid_8;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_8/Sigmoid_8 kTfLiteFloat32  kTfLiteArenaRw      54756 bytes ( 0.1 MB)  1 13 13 81
Tensor 163 functional_1/tf_op_layer_strided_slice_10/strided_slice_10;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_10/strided_slice_10 kTfLiteFloat32  kTfLiteArenaRw        676 bytes ( 0.0 MB)  1 13 13 1
Tensor 164 functional_1/tf_op_layer_strided_slice_11/strided_slice_11;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_11/strided_slice_11 kTfLiteFloat32  kTfLiteArenaRw      54080 bytes ( 0.1 MB)  1 13 13 80
Tensor 165 functional_1/tf_op_layer_Mul_14/Mul_14;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_14/Mul_14 kTfLiteFloat32  kTfLiteArenaRw      54080 bytes ( 0.1 MB)  1 13 13 80
Tensor 166 functional_1/tf_op_layer_Reshape_11/Reshape_11;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_11/Reshape_11 kTfLiteFloat32  kTfLiteArenaRw      54080 bytes ( 0.1 MB)  1 169 80
Tensor 167 functional_1/tf_op_layer_concat_11/concat_11;StatefulPartitionedCall/functional_1/tf_op_layer_concat_11/concat_11 kTfLiteFloat32  kTfLiteArenaRw     162240 bytes ( 0.2 MB)  1 507 80
Tensor 168 functional_1/tf_op_layer_Sigmoid_9/Sigmoid_9;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_9/Sigmoid_9 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 169 functional_1/tf_op_layer_Mul_18/Mul_18;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_18/Mul_18 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 170 functional_1/tf_op_layer_Sub_3/Sub_3;StatefulPartitionedCall/functional_1/tf_op_layer_Sub_3/Sub_3 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 171 functional_1/tf_op_layer_AddV2_3/AddV2_3;StatefulPartitionedCall/functional_1/tf_op_layer_AddV2_3/AddV2_3 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 172 functional_1/tf_op_layer_Mul_19/Mul_19;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_19/Mul_19 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 13 13 2
Tensor 173 functional_1/tf_op_layer_Reshape_15/Reshape_15;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_15/Reshape_15 kTfLiteFloat32  kTfLiteArenaRw       1352 bytes ( 0.0 MB)  1 169 2
Tensor 174 functional_1/tf_op_layer_concat_13/concat_13;StatefulPartitionedCall/functional_1/tf_op_layer_concat_13/concat_13 kTfLiteFloat32  kTfLiteArenaRw       4056 bytes ( 0.0 MB)  1 507 2
Tensor 175 functional_1/tf_op_layer_concat_14/concat_14;StatefulPartitionedCall/functional_1/tf_op_layer_concat_14/concat_14 kTfLiteFloat32  kTfLiteArenaRw       8112 bytes ( 0.0 MB)  1 507 4
Tensor 176 functional_1/conv2d_20/BiasAdd;StatefulPartitionedCall/functional_1/conv2d_20/BiasAdd;functional_1/conv2d_20/Conv2D;StatefulPartitionedCall/functional_1/conv2d_20/Conv2D;unknown_971 kTfLiteFloat32  kTfLiteArenaRw     689520 bytes ( 0.7 MB)  1 26 26 255
Tensor 177 functional_1/tf_op_layer_split_3/split_3;StatefulPartitionedCall/functional_1/tf_op_layer_split_3/split_3 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 178 functional_1/tf_op_layer_split_3/split_3;StatefulPartitionedCall/functional_1/tf_op_layer_split_3/split_31 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 179 functional_1/tf_op_layer_split_3/split_3;StatefulPartitionedCall/functional_1/tf_op_layer_split_3/split_32 kTfLiteFloat32  kTfLiteArenaRw     219024 bytes ( 0.2 MB)  1 26 26 81
Tensor 180 functional_1/tf_op_layer_split_3/split_3;StatefulPartitionedCall/functional_1/tf_op_layer_split_3/split_33 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 181 functional_1/tf_op_layer_split_3/split_3;StatefulPartitionedCall/functional_1/tf_op_layer_split_3/split_34 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 182 functional_1/tf_op_layer_split_3/split_3;StatefulPartitionedCall/functional_1/tf_op_layer_split_3/split_35 kTfLiteFloat32  kTfLiteArenaRw     219024 bytes ( 0.2 MB)  1 26 26 81
Tensor 183 functional_1/tf_op_layer_split_3/split_3;StatefulPartitionedCall/functional_1/tf_op_layer_split_3/split_36 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 184 functional_1/tf_op_layer_split_3/split_3;StatefulPartitionedCall/functional_1/tf_op_layer_split_3/split_37 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 185 functional_1/tf_op_layer_split_3/split_3;StatefulPartitionedCall/functional_1/tf_op_layer_split_3/split_38 kTfLiteFloat32  kTfLiteArenaRw     219024 bytes ( 0.2 MB)  1 26 26 81
Tensor 186 functional_1/tf_op_layer_Exp/Exp;StatefulPartitionedCall/functional_1/tf_op_layer_Exp/Exp kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 187 functional_1/tf_op_layer_Mul_3/Mul_3;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_3/Mul_3 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 188 functional_1/tf_op_layer_Reshape_3/Reshape_3;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_3/Reshape_3 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 676 2
Tensor 189 functional_1/tf_op_layer_Exp_1/Exp_1;StatefulPartitionedCall/functional_1/tf_op_layer_Exp_1/Exp_1 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 190 functional_1/tf_op_layer_Mul_4/Mul_4;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_4/Mul_4 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 191 functional_1/tf_op_layer_Reshape_4/Reshape_4;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_4/Reshape_4 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 676 2
Tensor 192 functional_1/tf_op_layer_Exp_2/Exp_2;StatefulPartitionedCall/functional_1/tf_op_layer_Exp_2/Exp_2 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 193 functional_1/tf_op_layer_Mul_5/Mul_5;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_5/Mul_5 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 194 functional_1/tf_op_layer_Reshape_5/Reshape_5;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_5/Reshape_5 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 676 2
Tensor 195 functional_1/tf_op_layer_concat_8/concat_8;StatefulPartitionedCall/functional_1/tf_op_layer_concat_8/concat_8 kTfLiteFloat32  kTfLiteArenaRw      16224 bytes ( 0.0 MB)  1 2028 2
Tensor 196 functional_1/tf_op_layer_Sigmoid/Sigmoid;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid/Sigmoid kTfLiteFloat32  kTfLiteArenaRw     219024 bytes ( 0.2 MB)  1 26 26 81
Tensor 197 functional_1/tf_op_layer_strided_slice/strided_slice;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice/strided_slice kTfLiteFloat32  kTfLiteArenaRw       2704 bytes ( 0.0 MB)  1 26 26 1
Tensor 198 functional_1/tf_op_layer_strided_slice_1/strided_slice_1;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_1/strided_slice_1 kTfLiteFloat32  kTfLiteArenaRw     216320 bytes ( 0.2 MB)  1 26 26 80
Tensor 199 functional_1/tf_op_layer_Mul/Mul;StatefulPartitionedCall/functional_1/tf_op_layer_Mul/Mul kTfLiteFloat32  kTfLiteArenaRw     216320 bytes ( 0.2 MB)  1 26 26 80
Tensor 200 functional_1/tf_op_layer_Reshape/Reshape;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape/Reshape kTfLiteFloat32  kTfLiteArenaRw     216320 bytes ( 0.2 MB)  1 676 80
Tensor 201 functional_1/tf_op_layer_Sigmoid_1/Sigmoid_1;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_1/Sigmoid_1 kTfLiteFloat32  kTfLiteArenaRw     219024 bytes ( 0.2 MB)  1 26 26 81
Tensor 202 functional_1/tf_op_layer_strided_slice_2/strided_slice_2;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_2/strided_slice_2 kTfLiteFloat32  kTfLiteArenaRw       2704 bytes ( 0.0 MB)  1 26 26 1
Tensor 203 functional_1/tf_op_layer_strided_slice_3/strided_slice_3;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_3/strided_slice_3 kTfLiteFloat32  kTfLiteArenaRw     216320 bytes ( 0.2 MB)  1 26 26 80
Tensor 204 functional_1/tf_op_layer_Mul_1/Mul_1;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_1/Mul_1 kTfLiteFloat32  kTfLiteArenaRw     216320 bytes ( 0.2 MB)  1 26 26 80
Tensor 205 functional_1/tf_op_layer_Reshape_1/Reshape_1;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_1/Reshape_1 kTfLiteFloat32  kTfLiteArenaRw     216320 bytes ( 0.2 MB)  1 676 80
Tensor 206 functional_1/tf_op_layer_Sigmoid_2/Sigmoid_2;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_2/Sigmoid_2 kTfLiteFloat32  kTfLiteArenaRw     219024 bytes ( 0.2 MB)  1 26 26 81
Tensor 207 functional_1/tf_op_layer_strided_slice_4/strided_slice_4;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_4/strided_slice_4 kTfLiteFloat32  kTfLiteArenaRw       2704 bytes ( 0.0 MB)  1 26 26 1
Tensor 208 functional_1/tf_op_layer_strided_slice_5/strided_slice_5;StatefulPartitionedCall/functional_1/tf_op_layer_strided_slice_5/strided_slice_53 kTfLiteFloat32  kTfLiteArenaRw     216320 bytes ( 0.2 MB)  1 26 26 80
Tensor 209 functional_1/tf_op_layer_Mul_2/Mul_2;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_2/Mul_2 kTfLiteFloat32  kTfLiteArenaRw     216320 bytes ( 0.2 MB)  1 26 26 80
Tensor 210 functional_1/tf_op_layer_Reshape_2/Reshape_2;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_2/Reshape_2 kTfLiteFloat32  kTfLiteArenaRw     216320 bytes ( 0.2 MB)  1 676 80
Tensor 211 functional_1/tf_op_layer_concat_7/concat_7;StatefulPartitionedCall/functional_1/tf_op_layer_concat_7/concat_7 kTfLiteFloat32  kTfLiteArenaRw     648960 bytes ( 0.6 MB)  1 2028 80
Tensor 212 Identity_1           kTfLiteFloat32  kTfLiteArenaRw     811200 bytes ( 0.8 MB)  1 2535 80
Tensor 213 functional_1/tf_op_layer_Sigmoid_3/Sigmoid_3;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_3/Sigmoid_3 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 214 functional_1/tf_op_layer_Mul_6/Mul_6;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_6/Mul_6 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 215 functional_1/tf_op_layer_Sub/Sub;StatefulPartitionedCall/functional_1/tf_op_layer_Sub/Sub kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 216 functional_1/tf_op_layer_AddV2/AddV2;StatefulPartitionedCall/functional_1/tf_op_layer_AddV2/AddV2 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 217 functional_1/tf_op_layer_Mul_7/Mul_7;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_7/Mul_7 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 218 functional_1/tf_op_layer_Reshape_6/Reshape_6;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_6/Reshape_6 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 676 2
Tensor 219 functional_1/tf_op_layer_Sigmoid_4/Sigmoid_4;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_4/Sigmoid_4 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 220 functional_1/tf_op_layer_Mul_8/Mul_8;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_8/Mul_8 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 221 functional_1/tf_op_layer_Sub_1/Sub_1;StatefulPartitionedCall/functional_1/tf_op_layer_Sub_1/Sub_1 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 222 functional_1/tf_op_layer_AddV2_1/AddV2_1;StatefulPartitionedCall/functional_1/tf_op_layer_AddV2_1/AddV2_1 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 223 functional_1/tf_op_layer_Mul_9/Mul_9;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_9/Mul_9 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 224 functional_1/tf_op_layer_Reshape_7/Reshape_7;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_7/Reshape_7 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 676 2
Tensor 225 functional_1/tf_op_layer_Sigmoid_5/Sigmoid_5;StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid_5/Sigmoid_5 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 226 functional_1/tf_op_layer_Mul_10/Mul_10;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_10/Mul_10 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 227 functional_1/tf_op_layer_Sub_2/Sub_2;StatefulPartitionedCall/functional_1/tf_op_layer_Sub_2/Sub_2 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 228 functional_1/tf_op_layer_AddV2_2/AddV2_2;StatefulPartitionedCall/functional_1/tf_op_layer_AddV2_2/AddV2_2 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 229 functional_1/tf_op_layer_Mul_11/Mul_11;StatefulPartitionedCall/functional_1/tf_op_layer_Mul_11/Mul_11 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 26 26 2
Tensor 230 functional_1/tf_op_layer_Reshape_8/Reshape_8;StatefulPartitionedCall/functional_1/tf_op_layer_Reshape_8/Reshape_8 kTfLiteFloat32  kTfLiteArenaRw       5408 bytes ( 0.0 MB)  1 676 2
Tensor 231 functional_1/tf_op_layer_concat_9/concat_9;StatefulPartitionedCall/functional_1/tf_op_layer_concat_9/concat_9 kTfLiteFloat32  kTfLiteArenaRw      16224 bytes ( 0.0 MB)  1 2028 2
Tensor 232 functional_1/tf_op_layer_concat_10/concat_10;StatefulPartitionedCall/functional_1/tf_op_layer_concat_10/concat_10 kTfLiteFloat32  kTfLiteArenaRw      32448 bytes ( 0.0 MB)  1 2028 4
Tensor 233 Identity             kTfLiteFloat32  kTfLiteArenaRw      40560 bytes ( 0.0 MB)  1 2535 4
Tensor 234 (null)               kTfLiteFloat32  kTfLiteArenaRw    4672512 bytes ( 4.5 MB)  1 208 208 27
Tensor 235 (null)               kTfLiteFloat32  kTfLiteArenaRw   12460032 bytes (11.9 MB)  1 104 104 288
Tensor 236 (null)               kTfLiteFloat32  kTfLiteArenaRw   24920064 bytes (23.8 MB)  1 104 104 576
Tensor 237 (null)               kTfLiteFloat32  kTfLiteArenaRw   12460032 bytes (11.9 MB)  1 104 104 288
Tensor 238 (null)               kTfLiteFloat32  kTfLiteArenaRw   12460032 bytes (11.9 MB)  1 104 104 288
Tensor 239 (null)               kTfLiteFloat32  kTfLiteArenaRw   12460032 bytes (11.9 MB)  1 52 52 1152
Tensor 240 (null)               kTfLiteFloat32  kTfLiteArenaRw    6230016 bytes ( 5.9 MB)  1 52 52 576
Tensor 241 (null)               kTfLiteFloat32  kTfLiteArenaRw    6230016 bytes ( 5.9 MB)  1 52 52 576
Tensor 242 (null)               kTfLiteFloat32  kTfLiteArenaRw    6230016 bytes ( 5.9 MB)  1 26 26 2304
Tensor 243 (null)               kTfLiteFloat32  kTfLiteArenaRw    3115008 bytes ( 3.0 MB)  1 26 26 1152
Tensor 244 (null)               kTfLiteFloat32  kTfLiteArenaRw    3115008 bytes ( 3.0 MB)  1 26 26 1152
Tensor 245 (null)               kTfLiteFloat32  kTfLiteArenaRw    3115008 bytes ( 3.0 MB)  1 13 13 4608
Tensor 246 (null)               kTfLiteFloat32  kTfLiteArenaRw    1557504 bytes ( 1.5 MB)  1 13 13 2304
Tensor 247 (null)               kTfLiteFloat32  kTfLiteArenaRw    9345024 bytes ( 8.9 MB)  1 26 26 3456

Node   0 Operator Builtin Code  34 PAD
  Inputs: 0 17
  Outputs: 63
Node   1 Operator Builtin Code   3 CONV_2D
  Inputs: 63 18 39
  Outputs: 64
  Temporaries: 234
Node   2 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 64
  Outputs: 65
Node   3 Operator Builtin Code  34 PAD
  Inputs: 65 17
  Outputs: 66
Node   4 Operator Builtin Code   3 CONV_2D
  Inputs: 66 19 40
  Outputs: 67
  Temporaries: 235
Node   5 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 67
  Outputs: 68
Node   6 Operator Builtin Code   3 CONV_2D
  Inputs: 68 20 41
  Outputs: 69
  Temporaries: 236
Node   7 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 69
  Outputs: 70
Node   8 Operator Builtin Code  49 SPLIT
  Inputs: 16 70
  Outputs: 71 72
Node   9 Operator Builtin Code   3 CONV_2D
  Inputs: 72 21 42
  Outputs: 73
  Temporaries: 237
Node  10 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 73
  Outputs: 74
Node  11 Operator Builtin Code   3 CONV_2D
  Inputs: 74 22 43
  Outputs: 75
  Temporaries: 238
Node  12 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 75
  Outputs: 76
Node  13 Operator Builtin Code   2 CONCATENATION
  Inputs: 76 74
  Outputs: 77
Node  14 Operator Builtin Code   3 CONV_2D
  Inputs: 77 23 44
  Outputs: 78
Node  15 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 78
  Outputs: 79
Node  16 Operator Builtin Code   2 CONCATENATION
  Inputs: 70 79
  Outputs: 80
Node  17 Operator Builtin Code  17 MAX_POOL_2D
  Inputs: 80
  Outputs: 81
Node  18 Operator Builtin Code   3 CONV_2D
  Inputs: 81 24 45
  Outputs: 82
  Temporaries: 239
Node  19 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 82
  Outputs: 83
Node  20 Operator Builtin Code  49 SPLIT
  Inputs: 16 83
  Outputs: 84 85
Node  21 Operator Builtin Code   3 CONV_2D
  Inputs: 85 25 46
  Outputs: 86
  Temporaries: 240
Node  22 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 86
  Outputs: 87
Node  23 Operator Builtin Code   3 CONV_2D
  Inputs: 87 26 47
  Outputs: 88
  Temporaries: 241
Node  24 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 88
  Outputs: 89
Node  25 Operator Builtin Code   2 CONCATENATION
  Inputs: 89 87
  Outputs: 90
Node  26 Operator Builtin Code   3 CONV_2D
  Inputs: 90 27 48
  Outputs: 91
Node  27 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 91
  Outputs: 92
Node  28 Operator Builtin Code   2 CONCATENATION
  Inputs: 83 92
  Outputs: 93
Node  29 Operator Builtin Code  17 MAX_POOL_2D
  Inputs: 93
  Outputs: 94
Node  30 Operator Builtin Code   3 CONV_2D
  Inputs: 94 28 49
  Outputs: 95
  Temporaries: 242
Node  31 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 95
  Outputs: 96
Node  32 Operator Builtin Code  49 SPLIT
  Inputs: 16 96
  Outputs: 97 98
Node  33 Operator Builtin Code   3 CONV_2D
  Inputs: 98 29 50
  Outputs: 99
  Temporaries: 243
Node  34 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 99
  Outputs: 100
Node  35 Operator Builtin Code   3 CONV_2D
  Inputs: 100 30 51
  Outputs: 101
  Temporaries: 244
Node  36 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 101
  Outputs: 102
Node  37 Operator Builtin Code   2 CONCATENATION
  Inputs: 102 100
  Outputs: 103
Node  38 Operator Builtin Code   3 CONV_2D
  Inputs: 103 31 52
  Outputs: 104
Node  39 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 104
  Outputs: 105
Node  40 Operator Builtin Code   2 CONCATENATION
  Inputs: 96 105
  Outputs: 106
Node  41 Operator Builtin Code  17 MAX_POOL_2D
  Inputs: 106
  Outputs: 107
Node  42 Operator Builtin Code   3 CONV_2D
  Inputs: 107 32 53
  Outputs: 108
  Temporaries: 245
Node  43 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 108
  Outputs: 109
Node  44 Operator Builtin Code   3 CONV_2D
  Inputs: 109 33 54
  Outputs: 110
Node  45 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 110
  Outputs: 111
Node  46 Operator Builtin Code   3 CONV_2D
  Inputs: 111 34 55
  Outputs: 112
  Temporaries: 246
Node  47 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 112
  Outputs: 113
Node  48 Operator Builtin Code   3 CONV_2D
  Inputs: 111 35 56
  Outputs: 114
Node  49 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 114
  Outputs: 115
Node  50 Operator Builtin Code  23 RESIZE_BILINEAR
  Inputs: 115 13
  Outputs: 116
Node  51 Operator Builtin Code   2 CONCATENATION
  Inputs: 116 105
  Outputs: 117
Node  52 Operator Builtin Code   3 CONV_2D
  Inputs: 117 36 57
  Outputs: 118
  Temporaries: 247
Node  53 Operator Builtin Code  98 LEAKY_RELU
  Inputs: 118
  Outputs: 119
Node  54 Operator Builtin Code   3 CONV_2D
  Inputs: 113 37 58
  Outputs: 120
Node  55 Operator Builtin Code 102 SPLIT_V
  Inputs: 120 15 16
  Outputs: 121 122 123 124 125 126 127 128 129
Node  56 Operator Builtin Code  47 EXP
  Inputs: 122
  Outputs: 130
Node  57 Operator Builtin Code  18 MUL
  Inputs: 130 8
  Outputs: 131
Node  58 Operator Builtin Code  22 RESHAPE
  Inputs: 131 11
  Outputs: 132
Node  59 Operator Builtin Code  47 EXP
  Inputs: 125
  Outputs: 133
Node  60 Operator Builtin Code  18 MUL
  Inputs: 133 3
  Outputs: 134
Node  61 Operator Builtin Code  22 RESHAPE
  Inputs: 134 11
  Outputs: 135
Node  62 Operator Builtin Code  47 EXP
  Inputs: 128
  Outputs: 136
Node  63 Operator Builtin Code  18 MUL
  Inputs: 136 4
  Outputs: 137
Node  64 Operator Builtin Code  22 RESHAPE
  Inputs: 137 11
  Outputs: 138
Node  65 Operator Builtin Code   2 CONCATENATION
  Inputs: 132 135 138
  Outputs: 139
Node  66 Operator Builtin Code  14 LOGISTIC
  Inputs: 124
  Outputs: 140
Node  67 Operator Builtin Code  18 MUL
  Inputs: 140 9
  Outputs: 141
Node  68 Operator Builtin Code  41 SUB
  Inputs: 141 14
  Outputs: 142
Node  69 Operator Builtin Code   0 ADD
  Inputs: 142 2
  Outputs: 143
Node  70 Operator Builtin Code  18 MUL
  Inputs: 143 5
  Outputs: 144
Node  71 Operator Builtin Code  22 RESHAPE
  Inputs: 144 11
  Outputs: 145
Node  72 Operator Builtin Code  14 LOGISTIC
  Inputs: 127
  Outputs: 146
Node  73 Operator Builtin Code  18 MUL
  Inputs: 146 9
  Outputs: 147
Node  74 Operator Builtin Code  41 SUB
  Inputs: 147 14
  Outputs: 148
Node  75 Operator Builtin Code   0 ADD
  Inputs: 148 2
  Outputs: 149
Node  76 Operator Builtin Code  18 MUL
  Inputs: 149 5
  Outputs: 150
Node  77 Operator Builtin Code  22 RESHAPE
  Inputs: 150 11
  Outputs: 151
Node  78 Operator Builtin Code  14 LOGISTIC
  Inputs: 123
  Outputs: 152
Node  79 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 152 61 60 62
  Outputs: 153
Node  80 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 152 60 61 62
  Outputs: 154
Node  81 Operator Builtin Code  18 MUL
  Inputs: 153 154
  Outputs: 155
Node  82 Operator Builtin Code  22 RESHAPE
  Inputs: 155 12
  Outputs: 156
Node  83 Operator Builtin Code  14 LOGISTIC
  Inputs: 126
  Outputs: 157
Node  84 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 157 61 60 62
  Outputs: 158
Node  85 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 157 60 61 62
  Outputs: 159
Node  86 Operator Builtin Code  18 MUL
  Inputs: 158 159
  Outputs: 160
Node  87 Operator Builtin Code  22 RESHAPE
  Inputs: 160 12
  Outputs: 161
Node  88 Operator Builtin Code  14 LOGISTIC
  Inputs: 129
  Outputs: 162
Node  89 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 162 61 60 62
  Outputs: 163
Node  90 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 162 60 61 62
  Outputs: 164
Node  91 Operator Builtin Code  18 MUL
  Inputs: 163 164
  Outputs: 165
Node  92 Operator Builtin Code  22 RESHAPE
  Inputs: 165 12
  Outputs: 166
Node  93 Operator Builtin Code   2 CONCATENATION
  Inputs: 156 161 166
  Outputs: 167
Node  94 Operator Builtin Code  14 LOGISTIC
  Inputs: 121
  Outputs: 168
Node  95 Operator Builtin Code  18 MUL
  Inputs: 168 9
  Outputs: 169
Node  96 Operator Builtin Code  41 SUB
  Inputs: 169 14
  Outputs: 170
Node  97 Operator Builtin Code   0 ADD
  Inputs: 170 2
  Outputs: 171
Node  98 Operator Builtin Code  18 MUL
  Inputs: 171 5
  Outputs: 172
Node  99 Operator Builtin Code  22 RESHAPE
  Inputs: 172 11
  Outputs: 173
Node 100 Operator Builtin Code   2 CONCATENATION
  Inputs: 173 145 151
  Outputs: 174
Node 101 Operator Builtin Code   2 CONCATENATION
  Inputs: 174 139
  Outputs: 175
Node 102 Operator Builtin Code   3 CONV_2D
  Inputs: 119 38 59
  Outputs: 176
Node 103 Operator Builtin Code 102 SPLIT_V
  Inputs: 176 15 16
  Outputs: 177 178 179 180 181 182 183 184 185
Node 104 Operator Builtin Code  47 EXP
  Inputs: 178
  Outputs: 186
Node 105 Operator Builtin Code  18 MUL
  Inputs: 186 6
  Outputs: 187
Node 106 Operator Builtin Code  22 RESHAPE
  Inputs: 187 11
  Outputs: 188
Node 107 Operator Builtin Code  47 EXP
  Inputs: 181
  Outputs: 189
Node 108 Operator Builtin Code  18 MUL
  Inputs: 189 7
  Outputs: 190
Node 109 Operator Builtin Code  22 RESHAPE
  Inputs: 190 11
  Outputs: 191
Node 110 Operator Builtin Code  47 EXP
  Inputs: 184
  Outputs: 192
Node 111 Operator Builtin Code  18 MUL
  Inputs: 192 8
  Outputs: 193
Node 112 Operator Builtin Code  22 RESHAPE
  Inputs: 193 11
  Outputs: 194
Node 113 Operator Builtin Code   2 CONCATENATION
  Inputs: 188 191 194
  Outputs: 195
Node 114 Operator Builtin Code  14 LOGISTIC
  Inputs: 179
  Outputs: 196
Node 115 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 196 61 60 62
  Outputs: 197
Node 116 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 196 60 61 62
  Outputs: 198
Node 117 Operator Builtin Code  18 MUL
  Inputs: 197 198
  Outputs: 199
Node 118 Operator Builtin Code  22 RESHAPE
  Inputs: 199 12
  Outputs: 200
Node 119 Operator Builtin Code  14 LOGISTIC
  Inputs: 182
  Outputs: 201
Node 120 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 201 61 60 62
  Outputs: 202
Node 121 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 201 60 61 62
  Outputs: 203
Node 122 Operator Builtin Code  18 MUL
  Inputs: 202 203
  Outputs: 204
Node 123 Operator Builtin Code  22 RESHAPE
  Inputs: 204 12
  Outputs: 205
Node 124 Operator Builtin Code  14 LOGISTIC
  Inputs: 185
  Outputs: 206
Node 125 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 206 61 60 62
  Outputs: 207
Node 126 Operator Builtin Code  45 STRIDED_SLICE
  Inputs: 206 60 61 62
  Outputs: 208
Node 127 Operator Builtin Code  18 MUL
  Inputs: 207 208
  Outputs: 209
Node 128 Operator Builtin Code  22 RESHAPE
  Inputs: 209 12
  Outputs: 210
Node 129 Operator Builtin Code   2 CONCATENATION
  Inputs: 200 205 210
  Outputs: 211
Node 130 Operator Builtin Code   2 CONCATENATION
  Inputs: 211 167
  Outputs: 212
Node 131 Operator Builtin Code  14 LOGISTIC
  Inputs: 177
  Outputs: 213
Node 132 Operator Builtin Code  18 MUL
  Inputs: 213 9
  Outputs: 214
Node 133 Operator Builtin Code  41 SUB
  Inputs: 214 14
  Outputs: 215
Node 134 Operator Builtin Code   0 ADD
  Inputs: 215 1
  Outputs: 216
Node 135 Operator Builtin Code  18 MUL
  Inputs: 216 10
  Outputs: 217
Node 136 Operator Builtin Code  22 RESHAPE
  Inputs: 217 11
  Outputs: 218
Node 137 Operator Builtin Code  14 LOGISTIC
  Inputs: 180
  Outputs: 219
Node 138 Operator Builtin Code  18 MUL
  Inputs: 219 9
  Outputs: 220
Node 139 Operator Builtin Code  41 SUB
  Inputs: 220 14
  Outputs: 221
Node 140 Operator Builtin Code   0 ADD
  Inputs: 221 1
  Outputs: 222
Node 141 Operator Builtin Code  18 MUL
  Inputs: 222 10
  Outputs: 223
Node 142 Operator Builtin Code  22 RESHAPE
  Inputs: 223 11
  Outputs: 224
Node 143 Operator Builtin Code  14 LOGISTIC
  Inputs: 183
  Outputs: 225
Node 144 Operator Builtin Code  18 MUL
  Inputs: 225 9
  Outputs: 226
Node 145 Operator Builtin Code  41 SUB
  Inputs: 226 14
  Outputs: 227
Node 146 Operator Builtin Code   0 ADD
  Inputs: 227 1
  Outputs: 228
Node 147 Operator Builtin Code  18 MUL
  Inputs: 228 10
  Outputs: 229
Node 148 Operator Builtin Code  22 RESHAPE
  Inputs: 229 11
  Outputs: 230
Node 149 Operator Builtin Code   2 CONCATENATION
  Inputs: 218 224 230
  Outputs: 231
Node 150 Operator Builtin Code   2 CONCATENATION
  Inputs: 231 195
  Outputs: 232
Node 151 Operator Builtin Code   2 CONCATENATION
  Inputs: 232 175
  Outputs: 233


**default partitions (Fallback : ADD, MUL, SPILT, SPILT_V)**

0 1 2 3 4 5 6 7 
9 10 11 12 13 14 15 16 17 18 19 
21 22 23 24 25 26 27 28 29 30 31 
33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 102 
56 59 62 66 72 78 79 80 83 84 85 88 89 90 94 104 107 110 114 115 116 119 120 121 124 125 126 131 137 143 
58 61 64 65 68 74 82 87 92 93 96 106 109 112 113 118 123 128 129 130 133 139 145 
71 77 99 100 101 136 142 148 149 150 151 
