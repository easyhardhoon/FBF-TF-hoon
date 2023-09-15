[CPU INTERETER STATE] 
Interpreter has 210 tensors and 76 nodes
Inputs: 0
Outputs: 198 199 200 201

Tensor   0 normalized_input_image_tensor kTfLiteFloat32  kTfLiteArenaRw    1228800 bytes ( 1.2 MB)  1 320 320 3
Tensor   1 anchors              kTfLiteFloat32   kTfLiteMmapRo      32544 bytes ( 0.0 MB)  2034 4
Tensor   2 BoxPredictor_0/Reshape_1/shape kTfLiteInt32   kTfLiteMmapRo         12 bytes ( 0.0 MB)  3
Tensor   3 BoxPredictor_0/Reshape/shape kTfLiteInt32   kTfLiteMmapRo         16 bytes ( 0.0 MB)  4
Tensor   4 BoxPredictor_0/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor   5 BoxPredictor_0/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor   6 BoxPredictor_1/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       4096 bytes ( 0.0 MB)  1024
Tensor   7 BoxPredictor_1/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       4096 bytes ( 0.0 MB)  1024
Tensor   8 BoxPredictor_2/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor   9 BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  10 BoxPredictor_3/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  11 BoxPredictor_3/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  12 BoxPredictor_4/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  13 BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  14 BoxPredictor_5/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  15 BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  16 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        128 bytes ( 0.0 MB)  32
Tensor  17 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  18 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  19 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  20 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  21 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  22 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       4096 bytes ( 0.0 MB)  1024
Tensor  23 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       4096 bytes ( 0.0 MB)  1024
Tensor  24 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       4096 bytes ( 0.0 MB)  1024
Tensor  25 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  26 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  27 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  28 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        256 bytes ( 0.0 MB)  64
Tensor  29 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  30 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  31 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  32 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  33 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        128 bytes ( 0.0 MB)  32
Tensor  34 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        256 bytes ( 0.0 MB)  64
Tensor  35 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        256 bytes ( 0.0 MB)  64
Tensor  36 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  37 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  38 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  39 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo        512 bytes ( 0.0 MB)  128
Tensor  40 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  41 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  42 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  43 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       1024 bytes ( 0.0 MB)  256
Tensor  44 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  45 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  46 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  47 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  48 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  49 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  50 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3 kTfLiteFloat32   kTfLiteMmapRo       2048 bytes ( 0.0 MB)  512
Tensor  51 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D kTfLiteFloat32   kTfLiteMmapRo       3456 bytes ( 0.0 MB)  32 3 3 3
Tensor  52 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo       8192 bytes ( 0.0 MB)  64 1 1 32
Tensor  53 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo      32768 bytes ( 0.0 MB)  128 1 1 64
Tensor  54 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo      65536 bytes ( 0.1 MB)  128 1 1 128
Tensor  55 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo     131072 bytes ( 0.1 MB)  256 1 1 128
Tensor  56 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo     262144 bytes ( 0.2 MB)  256 1 1 256
Tensor  57 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo     524288 bytes ( 0.5 MB)  512 1 1 256
Tensor  58 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)  512 1 1 512
Tensor  59 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)  512 1 1 512
Tensor  60 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)  512 1 1 512
Tensor  61 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)  512 1 1 512
Tensor  62 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)  512 1 1 512
Tensor  63 BoxPredictor_0/BoxEncodingPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo      24576 bytes ( 0.0 MB)  12 1 1 512
Tensor  64 BoxPredictor_0/ClassPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo     559104 bytes ( 0.5 MB)  273 1 1 512
Tensor  65 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo    2097152 bytes ( 2.0 MB)  1024 1 1 512
Tensor  66 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D kTfLiteFloat32   kTfLiteMmapRo    4194304 bytes ( 4.0 MB)  1024 1 1 1024
Tensor  67 BoxPredictor_1/BoxEncodingPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo      98304 bytes ( 0.1 MB)  24 1 1 1024
Tensor  68 BoxPredictor_1/ClassPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo    2236416 bytes ( 2.1 MB)  546 1 1 1024
Tensor  69 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Conv2D kTfLiteFloat32   kTfLiteMmapRo    1048576 bytes ( 1.0 MB)  256 1 1 1024
Tensor  70 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Conv2D kTfLiteFloat32   kTfLiteMmapRo    4718592 bytes ( 4.5 MB)  512 3 3 256
Tensor  71 BoxPredictor_2/BoxEncodingPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo      49152 bytes ( 0.0 MB)  24 1 1 512
Tensor  72 BoxPredictor_2/ClassPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo    1118208 bytes ( 1.1 MB)  546 1 1 512
Tensor  73 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Conv2D kTfLiteFloat32   kTfLiteMmapRo     262144 bytes ( 0.2 MB)  128 1 1 512
Tensor  74 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Conv2D kTfLiteFloat32   kTfLiteMmapRo    1179648 bytes ( 1.1 MB)  256 3 3 128
Tensor  75 BoxPredictor_3/BoxEncodingPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo      24576 bytes ( 0.0 MB)  24 1 1 256
Tensor  76 BoxPredictor_3/ClassPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo     559104 bytes ( 0.5 MB)  546 1 1 256
Tensor  77 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/Conv2D kTfLiteFloat32   kTfLiteMmapRo     131072 bytes ( 0.1 MB)  128 1 1 256
Tensor  78 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/Conv2D kTfLiteFloat32   kTfLiteMmapRo    1179648 bytes ( 1.1 MB)  256 3 3 128
Tensor  79 BoxPredictor_4/BoxEncodingPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo      24576 bytes ( 0.0 MB)  24 1 1 256
Tensor  80 BoxPredictor_4/ClassPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo     559104 bytes ( 0.5 MB)  546 1 1 256
Tensor  81 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/Conv2D kTfLiteFloat32   kTfLiteMmapRo      65536 bytes ( 0.1 MB)  64 1 1 256
Tensor  82 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Conv2D kTfLiteFloat32   kTfLiteMmapRo     294912 bytes ( 0.3 MB)  128 3 3 64
Tensor  83 BoxPredictor_5/BoxEncodingPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo      12288 bytes ( 0.0 MB)  24 1 1 128
Tensor  84 BoxPredictor_5/ClassPredictor/Conv2D kTfLiteFloat32   kTfLiteMmapRo     279552 bytes ( 0.3 MB)  546 1 1 128
Tensor  85 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       1152 bytes ( 0.0 MB)  1 3 3 32
Tensor  86 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/Conv2D kTfLiteFloat32   kTfLiteMmapRo       2304 bytes ( 0.0 MB)  1 3 3 64
Tensor  87 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise;BoxPredictor_5/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       4608 bytes ( 0.0 MB)  1 3 3 128
Tensor  88 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise;BoxPredictor_5/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       4608 bytes ( 0.0 MB)  1 3 3 128
Tensor  89 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise;BoxPredictor_4/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       9216 bytes ( 0.0 MB)  1 3 3 256
Tensor  90 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise;BoxPredictor_4/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       9216 bytes ( 0.0 MB)  1 3 3 256
Tensor  91 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)  1 3 3 512
Tensor  92 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)  1 3 3 512
Tensor  93 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)  1 3 3 512
Tensor  94 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)  1 3 3 512
Tensor  95 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)  1 3 3 512
Tensor  96 BoxPredictor_0/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_0/BoxEncodingPredictor_depthwise/depthwise;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)  1 3 3 512
Tensor  97 BoxPredictor_0/BoxEncodingPredictor/BiasAdd;BoxPredictor_0/BoxEncodingPredictor/Conv2D;BoxPredictor_0/BoxEncodingPredictor/biases kTfLiteFloat32   kTfLiteMmapRo         48 bytes ( 0.0 MB)  12
Tensor  98 BoxPredictor_0/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_0/ClassPredictor_depthwise/depthwise;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)  1 3 3 512
Tensor  99 BoxPredictor_0/ClassPredictor/BiasAdd;BoxPredictor_0/ClassPredictor/Conv2D;BoxPredictor_0/ClassPredictor/biases kTfLiteFloat32   kTfLiteMmapRo       1092 bytes ( 0.0 MB)  273
Tensor 100 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)  1 3 3 512
Tensor 101 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise;BoxPredictor_1/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      36864 bytes ( 0.0 MB)  1 3 3 1024
Tensor 102 BoxPredictor_1/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/BoxEncodingPredictor_depthwise/depthwise;BoxPredictor_1/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      36864 bytes ( 0.0 MB)  1 3 3 1024
Tensor 103 BoxPredictor_1/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_1/BoxEncodingPredictor/Conv2D;BoxPredictor_1/BoxEncodingPredictor/biases kTfLiteFloat32   kTfLiteMmapRo         96 bytes ( 0.0 MB)  24
Tensor 104 BoxPredictor_1/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      36864 bytes ( 0.0 MB)  1 3 3 1024
Tensor 105 BoxPredictor_1/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_1/ClassPredictor/Conv2D;BoxPredictor_1/ClassPredictor/biases kTfLiteFloat32   kTfLiteMmapRo       2184 bytes ( 0.0 MB)  546
Tensor 106 BoxPredictor_2/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/BoxEncodingPredictor_depthwise/depthwise;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)  1 3 3 512
Tensor 107 BoxPredictor_2/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_2/BoxEncodingPredictor/Conv2D;BoxPredictor_2/BoxEncodingPredictor/biases kTfLiteFloat32   kTfLiteMmapRo         96 bytes ( 0.0 MB)  24
Tensor 108 BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo      18432 bytes ( 0.0 MB)  1 3 3 512
Tensor 109 BoxPredictor_2/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_2/ClassPredictor/Conv2D;BoxPredictor_2/ClassPredictor/biases kTfLiteFloat32   kTfLiteMmapRo       2184 bytes ( 0.0 MB)  546
Tensor 110 BoxPredictor_3/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_3/BoxEncodingPredictor_depthwise/depthwise;BoxPredictor_4/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       9216 bytes ( 0.0 MB)  1 3 3 256
Tensor 111 BoxPredictor_3/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_3/BoxEncodingPredictor/Conv2D;BoxPredictor_3/BoxEncodingPredictor/biases kTfLiteFloat32   kTfLiteMmapRo         96 bytes ( 0.0 MB)  24
Tensor 112 BoxPredictor_3/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_3/ClassPredictor_depthwise/depthwise;BoxPredictor_4/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       9216 bytes ( 0.0 MB)  1 3 3 256
Tensor 113 BoxPredictor_3/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_3/ClassPredictor/Conv2D;BoxPredictor_3/ClassPredictor/biases kTfLiteFloat32   kTfLiteMmapRo       2184 bytes ( 0.0 MB)  546
Tensor 114 BoxPredictor_4/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/BoxEncodingPredictor_depthwise/depthwise;BoxPredictor_4/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       9216 bytes ( 0.0 MB)  1 3 3 256
Tensor 115 BoxPredictor_4/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_4/BoxEncodingPredictor/Conv2D;BoxPredictor_4/BoxEncodingPredictor/biases kTfLiteFloat32   kTfLiteMmapRo         96 bytes ( 0.0 MB)  24
Tensor 116 BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       9216 bytes ( 0.0 MB)  1 3 3 256
Tensor 117 BoxPredictor_4/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_4/ClassPredictor/Conv2D;BoxPredictor_4/ClassPredictor/biases kTfLiteFloat32   kTfLiteMmapRo       2184 bytes ( 0.0 MB)  546
Tensor 118 BoxPredictor_5/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/BoxEncodingPredictor_depthwise/depthwise;BoxPredictor_5/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       4608 bytes ( 0.0 MB)  1 3 3 128
Tensor 119 BoxPredictor_5/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_5/BoxEncodingPredictor/biases kTfLiteFloat32   kTfLiteMmapRo         96 bytes ( 0.0 MB)  24
Tensor 120 Squeeze              kTfLiteInt32   kTfLiteMmapRo         12 bytes ( 0.0 MB)  3
Tensor 121 BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/depthwise kTfLiteFloat32   kTfLiteMmapRo       4608 bytes ( 0.0 MB)  1 3 3 128
Tensor 122 BoxPredictor_5/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_5/ClassPredictor/biases kTfLiteFloat32   kTfLiteMmapRo       2184 bytes ( 0.0 MB)  546
Tensor 123 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D kTfLiteFloat32  kTfLiteArenaRw    3276800 bytes ( 3.1 MB)  1 160 160 32
Tensor 124 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw    3276800 bytes ( 3.1 MB)  1 160 160 32
Tensor 125 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/Conv2D;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw    6553600 bytes ( 6.2 MB)  1 160 160 64
Tensor 126 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/Conv2D kTfLiteFloat32  kTfLiteArenaRw    1638400 bytes ( 1.6 MB)  1 80 80 64
Tensor 127 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw    3276800 bytes ( 3.1 MB)  1 80 80 128
Tensor 128 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw    3276800 bytes ( 3.1 MB)  1 80 80 128
Tensor 129 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw    3276800 bytes ( 3.1 MB)  1 80 80 128
Tensor 130 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 40 40 128
Tensor 131 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw    1638400 bytes ( 1.6 MB)  1 40 40 256
Tensor 132 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw    1638400 bytes ( 1.6 MB)  1 40 40 256
Tensor 133 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw    1638400 bytes ( 1.6 MB)  1 40 40 256
Tensor 134 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     409600 bytes ( 0.4 MB)  1 20 20 256
Tensor 135 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 136 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 137 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 138 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 139 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 140 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 141 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 142 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 143 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 144 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 145 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 146 BoxPredictor_0/BoxEncodingPredictor_depthwise/Relu6;BoxPredictor_0/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;BoxPredictor_0/BoxEncodingPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 147 BoxPredictor_0/BoxEncodingPredictor/BiasAdd;BoxPredictor_0/BoxEncodingPredictor/Conv2D;BoxPredictor_0/BoxEncodingPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw      19200 bytes ( 0.0 MB)  1 20 20 12
Tensor 148 BoxPredictor_0/Reshape kTfLiteFloat32  kTfLiteArenaRw      19200 bytes ( 0.0 MB)  1 1200 1 4
Tensor 149 BoxPredictor_0/ClassPredictor_depthwise/Relu6;BoxPredictor_0/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_0/ClassPredictor_depthwise/depthwise;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     819200 bytes ( 0.8 MB)  1 20 20 512
Tensor 150 BoxPredictor_0/ClassPredictor/BiasAdd;BoxPredictor_0/ClassPredictor/Conv2D;BoxPredictor_0/ClassPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw     436800 bytes ( 0.4 MB)  1 20 20 273
Tensor 151 BoxPredictor_0/Reshape_1 kTfLiteFloat32  kTfLiteArenaRw     436800 bytes ( 0.4 MB)  1 1200 91
Tensor 152 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     204800 bytes ( 0.2 MB)  1 10 10 512
Tensor 153 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw     409600 bytes ( 0.4 MB)  1 10 10 1024
Tensor 154 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     409600 bytes ( 0.4 MB)  1 10 10 1024
Tensor 155 FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D kTfLiteFloat32  kTfLiteArenaRw     409600 bytes ( 0.4 MB)  1 10 10 1024
Tensor 156 BoxPredictor_1/BoxEncodingPredictor_depthwise/Relu6;BoxPredictor_1/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/ClassPredictor_depthwise/depthwise;BoxPredictor_1/BoxEncodingPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     409600 bytes ( 0.4 MB)  1 10 10 1024
Tensor 157 BoxPredictor_1/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_1/BoxEncodingPredictor/Conv2D;BoxPredictor_1/BoxEncodingPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw       9600 bytes ( 0.0 MB)  1 10 10 24
Tensor 158 BoxPredictor_1/Reshape kTfLiteFloat32  kTfLiteArenaRw       9600 bytes ( 0.0 MB)  1 600 1 4
Tensor 159 BoxPredictor_1/ClassPredictor_depthwise/Relu6;BoxPredictor_1/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_1/ClassPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw     409600 bytes ( 0.4 MB)  1 10 10 1024
Tensor 160 BoxPredictor_1/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_1/ClassPredictor/Conv2D;BoxPredictor_1/ClassPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw     218400 bytes ( 0.2 MB)  1 10 10 546
Tensor 161 BoxPredictor_1/Reshape_1 kTfLiteFloat32  kTfLiteArenaRw     218400 bytes ( 0.2 MB)  1 600 91
Tensor 162 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Relu6;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Conv2D kTfLiteFloat32  kTfLiteArenaRw     102400 bytes ( 0.1 MB)  1 10 10 256
Tensor 163 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Relu6;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Conv2D kTfLiteFloat32  kTfLiteArenaRw      51200 bytes ( 0.0 MB)  1 5 5 512
Tensor 164 BoxPredictor_2/BoxEncodingPredictor_depthwise/Relu6;BoxPredictor_2/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise;BoxPredictor_2/BoxEncodingPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw      51200 bytes ( 0.0 MB)  1 5 5 512
Tensor 165 BoxPredictor_2/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_2/BoxEncodingPredictor/Conv2D;BoxPredictor_2/BoxEncodingPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw       2400 bytes ( 0.0 MB)  1 5 5 24
Tensor 166 BoxPredictor_2/Reshape kTfLiteFloat32  kTfLiteArenaRw       2400 bytes ( 0.0 MB)  1 150 1 4
Tensor 167 BoxPredictor_2/ClassPredictor_depthwise/Relu6;BoxPredictor_2/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_2/ClassPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw      51200 bytes ( 0.0 MB)  1 5 5 512
Tensor 168 BoxPredictor_2/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_2/ClassPredictor/Conv2D;BoxPredictor_2/ClassPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw      54600 bytes ( 0.1 MB)  1 5 5 546
Tensor 169 BoxPredictor_2/Reshape_1 kTfLiteFloat32  kTfLiteArenaRw      54600 bytes ( 0.1 MB)  1 150 91
Tensor 170 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Relu6;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Conv2D kTfLiteFloat32  kTfLiteArenaRw      12800 bytes ( 0.0 MB)  1 5 5 128
Tensor 171 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Relu6;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Conv2D kTfLiteFloat32  kTfLiteArenaRw       9216 bytes ( 0.0 MB)  1 3 3 256
Tensor 172 BoxPredictor_3/BoxEncodingPredictor_depthwise/Relu6;BoxPredictor_3/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise;BoxPredictor_3/BoxEncodingPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw       9216 bytes ( 0.0 MB)  1 3 3 256
Tensor 173 BoxPredictor_3/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_3/BoxEncodingPredictor/Conv2D;BoxPredictor_3/BoxEncodingPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw        864 bytes ( 0.0 MB)  1 3 3 24
Tensor 174 BoxPredictor_3/Reshape kTfLiteFloat32  kTfLiteArenaRw        864 bytes ( 0.0 MB)  1 54 1 4
Tensor 175 BoxPredictor_3/ClassPredictor_depthwise/Relu6;BoxPredictor_3/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise;BoxPredictor_3/ClassPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw       9216 bytes ( 0.0 MB)  1 3 3 256
Tensor 176 BoxPredictor_3/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_3/ClassPredictor/Conv2D;BoxPredictor_3/ClassPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw      19656 bytes ( 0.0 MB)  1 3 3 546
Tensor 177 BoxPredictor_3/Reshape_1 kTfLiteFloat32  kTfLiteArenaRw      19656 bytes ( 0.0 MB)  1 54 91
Tensor 178 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/Relu6;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/Conv2D kTfLiteFloat32  kTfLiteArenaRw       4608 bytes ( 0.0 MB)  1 3 3 128
Tensor 179 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/Relu6;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/Conv2D kTfLiteFloat32  kTfLiteArenaRw       4096 bytes ( 0.0 MB)  1 2 2 256
Tensor 180 BoxPredictor_4/BoxEncodingPredictor_depthwise/Relu6;BoxPredictor_4/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise;BoxPredictor_4/BoxEncodingPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw       4096 bytes ( 0.0 MB)  1 2 2 256
Tensor 181 BoxPredictor_4/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_4/BoxEncodingPredictor/Conv2D;BoxPredictor_4/BoxEncodingPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw        384 bytes ( 0.0 MB)  1 2 2 24
Tensor 182 BoxPredictor_4/Reshape kTfLiteFloat32  kTfLiteArenaRw        384 bytes ( 0.0 MB)  1 24 1 4
Tensor 183 BoxPredictor_4/ClassPredictor_depthwise/Relu6;BoxPredictor_4/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_4/ClassPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw       4096 bytes ( 0.0 MB)  1 2 2 256
Tensor 184 BoxPredictor_4/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_4/ClassPredictor/Conv2D;BoxPredictor_4/ClassPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw       8736 bytes ( 0.0 MB)  1 2 2 546
Tensor 185 BoxPredictor_4/Reshape_1 kTfLiteFloat32  kTfLiteArenaRw       8736 bytes ( 0.0 MB)  1 24 91
Tensor 186 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/Relu6;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNormV3;FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/Conv2D kTfLiteFloat32  kTfLiteArenaRw       1024 bytes ( 0.0 MB)  1 2 2 64
Tensor 187 FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/depthwise;FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Conv2D kTfLiteFloat32  kTfLiteArenaRw        512 bytes ( 0.0 MB)  1 1 1 128
Tensor 188 BoxPredictor_5/BoxEncodingPredictor_depthwise/Relu6;BoxPredictor_5/BoxEncodingPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/depthwise;BoxPredictor_5/BoxEncodingPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw        512 bytes ( 0.0 MB)  1 1 1 128
Tensor 189 BoxPredictor_5/BoxEncodingPredictor/BiasAdd;BoxPredictor_5/BoxEncodingPredictor/Conv2D;BoxPredictor_5/BoxEncodingPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw         96 bytes ( 0.0 MB)  1 1 1 24
Tensor 190 BoxPredictor_5/Reshape kTfLiteFloat32  kTfLiteArenaRw         96 bytes ( 0.0 MB)  1 6 1 4
Tensor 191 concat               kTfLiteFloat32  kTfLiteArenaRw      32544 bytes ( 0.0 MB)  1 2034 1 4
Tensor 192 Squeeze1             kTfLiteFloat32  kTfLiteArenaRw      32544 bytes ( 0.0 MB)  1 2034 4
Tensor 193 BoxPredictor_5/ClassPredictor_depthwise/Relu6;BoxPredictor_5/ClassPredictor_depthwise/BatchNorm/FusedBatchNormV3;BoxPredictor_5/ClassPredictor_depthwise/depthwise kTfLiteFloat32  kTfLiteArenaRw        512 bytes ( 0.0 MB)  1 1 1 128
Tensor 194 BoxPredictor_5/ClassPredictor/BiasAdd;BoxPredictor_5/ClassPredictor/Conv2D;BoxPredictor_5/ClassPredictor/biases1 kTfLiteFloat32  kTfLiteArenaRw       2184 bytes ( 0.0 MB)  1 1 1 546
Tensor 195 BoxPredictor_5/Reshape_1 kTfLiteFloat32  kTfLiteArenaRw       2184 bytes ( 0.0 MB)  1 6 91
Tensor 196 concat_1             kTfLiteFloat32  kTfLiteArenaRw     740376 bytes ( 0.7 MB)  1 2034 91
Tensor 197 convert_scores       kTfLiteFloat32  kTfLiteArenaRw     740376 bytes ( 0.7 MB)  1 2034 91
Tensor 198 TFLite_Detection_PostProcess kTfLiteFloat32  kTfLiteArenaRw        160 bytes ( 0.0 MB)  1 10 4
Tensor 199 TFLite_Detection_PostProcess:1 kTfLiteFloat32  kTfLiteArenaRw         40 bytes ( 0.0 MB)  1 10
Tensor 200 TFLite_Detection_PostProcess:2 kTfLiteFloat32  kTfLiteArenaRw         40 bytes ( 0.0 MB)  1 10
Tensor 201 TFLite_Detection_PostProcess:3 kTfLiteFloat32  kTfLiteArenaRw          4 bytes ( 0.0 MB)  1
Tensor 202 (null)               kTfLiteFloat32  kTfLiteArenaRw      32544 bytes ( 0.0 MB)  2034 4
Tensor 203 (null)               kTfLiteFloat32  kTfLiteArenaRw     740376 bytes ( 0.7 MB)  2034 91
Tensor 204 (null)               kTfLiteUInt8  kTfLiteArenaRw       2034 bytes ( 0.0 MB)  2034
Tensor 205 (null)               kTfLiteFloat32  kTfLiteArenaRw    2764800 bytes ( 2.6 MB)  1 160 160 27
Tensor 206 (null)               kTfLiteFloat32  kTfLiteArenaRw     230400 bytes ( 0.2 MB)  1 5 5 2304
Tensor 207 (null)               kTfLiteFloat32  kTfLiteArenaRw      41472 bytes ( 0.0 MB)  1 3 3 1152
Tensor 208 (null)               kTfLiteFloat32  kTfLiteArenaRw      18432 bytes ( 0.0 MB)  1 2 2 1152
Tensor 209 (null)               kTfLiteFloat32  kTfLiteArenaRw       2304 bytes ( 0.0 MB)  1 1 1 576

Node   0 Operator Builtin Code   3 CONV_2D
  Inputs: 0 51 16
  Outputs: 123
  Temporaries: 205
Node   1 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 123 85 33
  Outputs: 124
Node   2 Operator Builtin Code   3 CONV_2D
  Inputs: 124 52 34
  Outputs: 125
Node   3 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 125 86 35
  Outputs: 126
Node   4 Operator Builtin Code   3 CONV_2D
  Inputs: 126 53 36
  Outputs: 127
Node   5 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 127 87 37
  Outputs: 128
Node   6 Operator Builtin Code   3 CONV_2D
  Inputs: 128 54 38
  Outputs: 129
Node   7 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 129 88 39
  Outputs: 130
Node   8 Operator Builtin Code   3 CONV_2D
  Inputs: 130 55 40
  Outputs: 131
Node   9 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 131 89 41
  Outputs: 132
Node  10 Operator Builtin Code   3 CONV_2D
  Inputs: 132 56 42
  Outputs: 133
Node  11 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 133 90 43
  Outputs: 134
Node  12 Operator Builtin Code   3 CONV_2D
  Inputs: 134 57 44
  Outputs: 135
Node  13 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 135 91 45
  Outputs: 136
Node  14 Operator Builtin Code   3 CONV_2D
  Inputs: 136 58 46
  Outputs: 137
Node  15 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 137 92 47
  Outputs: 138
Node  16 Operator Builtin Code   3 CONV_2D
  Inputs: 138 59 48
  Outputs: 139
Node  17 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 139 93 49
  Outputs: 140
Node  18 Operator Builtin Code   3 CONV_2D
  Inputs: 140 60 50
  Outputs: 141
Node  19 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 141 94 17
  Outputs: 142
Node  20 Operator Builtin Code   3 CONV_2D
  Inputs: 142 61 18
  Outputs: 143
Node  21 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 143 95 19
  Outputs: 144
Node  22 Operator Builtin Code   3 CONV_2D
  Inputs: 144 62 20
  Outputs: 145
Node  23 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 145 96 4
  Outputs: 146
Node  24 Operator Builtin Code   3 CONV_2D
  Inputs: 146 63 97
  Outputs: 147
Node  25 Operator Builtin Code  22 RESHAPE
  Inputs: 147 3
  Outputs: 148
Node  26 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 145 98 5
  Outputs: 149
Node  27 Operator Builtin Code   3 CONV_2D
  Inputs: 149 64 99
  Outputs: 150
Node  28 Operator Builtin Code  22 RESHAPE
  Inputs: 150 2
  Outputs: 151
Node  29 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 145 100 21
  Outputs: 152
Node  30 Operator Builtin Code   3 CONV_2D
  Inputs: 152 65 22
  Outputs: 153
Node  31 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 153 101 23
  Outputs: 154
Node  32 Operator Builtin Code   3 CONV_2D
  Inputs: 154 66 24
  Outputs: 155
Node  33 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 155 102 6
  Outputs: 156
Node  34 Operator Builtin Code   3 CONV_2D
  Inputs: 156 67 103
  Outputs: 157
Node  35 Operator Builtin Code  22 RESHAPE
  Inputs: 157 3
  Outputs: 158
Node  36 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 155 104 7
  Outputs: 159
Node  37 Operator Builtin Code   3 CONV_2D
  Inputs: 159 68 105
  Outputs: 160
Node  38 Operator Builtin Code  22 RESHAPE
  Inputs: 160 2
  Outputs: 161
Node  39 Operator Builtin Code   3 CONV_2D
  Inputs: 155 69 25
  Outputs: 162
Node  40 Operator Builtin Code   3 CONV_2D
  Inputs: 162 70 29
  Outputs: 163
  Temporaries: 206
Node  41 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 163 106 8
  Outputs: 164
Node  42 Operator Builtin Code   3 CONV_2D
  Inputs: 164 71 107
  Outputs: 165
Node  43 Operator Builtin Code  22 RESHAPE
  Inputs: 165 3
  Outputs: 166
Node  44 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 163 108 9
  Outputs: 167
Node  45 Operator Builtin Code   3 CONV_2D
  Inputs: 167 72 109
  Outputs: 168
Node  46 Operator Builtin Code  22 RESHAPE
  Inputs: 168 2
  Outputs: 169
Node  47 Operator Builtin Code   3 CONV_2D
  Inputs: 163 73 26
  Outputs: 170
Node  48 Operator Builtin Code   3 CONV_2D
  Inputs: 170 74 30
  Outputs: 171
  Temporaries: 207
Node  49 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 171 110 10
  Outputs: 172
Node  50 Operator Builtin Code   3 CONV_2D
  Inputs: 172 75 111
  Outputs: 173
Node  51 Operator Builtin Code  22 RESHAPE
  Inputs: 173 3
  Outputs: 174
Node  52 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 171 112 11
  Outputs: 175
Node  53 Operator Builtin Code   3 CONV_2D
  Inputs: 175 76 113
  Outputs: 176
Node  54 Operator Builtin Code  22 RESHAPE
  Inputs: 176 2
  Outputs: 177
Node  55 Operator Builtin Code   3 CONV_2D
  Inputs: 171 77 27
  Outputs: 178
Node  56 Operator Builtin Code   3 CONV_2D
  Inputs: 178 78 31
  Outputs: 179
  Temporaries: 208
Node  57 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 179 114 12
  Outputs: 180
Node  58 Operator Builtin Code   3 CONV_2D
  Inputs: 180 79 115
  Outputs: 181
Node  59 Operator Builtin Code  22 RESHAPE
  Inputs: 181 3
  Outputs: 182
Node  60 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 179 116 13
  Outputs: 183
Node  61 Operator Builtin Code   3 CONV_2D
  Inputs: 183 80 117
  Outputs: 184
Node  62 Operator Builtin Code  22 RESHAPE
  Inputs: 184 2
  Outputs: 185
Node  63 Operator Builtin Code   3 CONV_2D
  Inputs: 179 81 28
  Outputs: 186
Node  64 Operator Builtin Code   3 CONV_2D
  Inputs: 186 82 32
  Outputs: 187
  Temporaries: 209
Node  65 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 187 118 14
  Outputs: 188
Node  66 Operator Builtin Code   3 CONV_2D
  Inputs: 188 83 119
  Outputs: 189
Node  67 Operator Builtin Code  22 RESHAPE
  Inputs: 189 3
  Outputs: 190
Node  68 Operator Builtin Code   2 CONCATENATION
  Inputs: 148 158 166 174 182 190
  Outputs: 191
Node  69 Operator Builtin Code  22 RESHAPE
  Inputs: 191 120
  Outputs: 192
Node  70 Operator Builtin Code   4 DEPTHWISE_CONV_2D
  Inputs: 187 121 15
  Outputs: 193
Node  71 Operator Builtin Code   3 CONV_2D
  Inputs: 193 84 122
  Outputs: 194
Node  72 Operator Builtin Code  22 RESHAPE
  Inputs: 194 2
  Outputs: 195
Node  73 Operator Builtin Code   2 CONCATENATION
  Inputs: 151 161 169 177 185 195
  Outputs: 196
Node  74 Operator Builtin Code  14 LOGISTIC
  Inputs: 196
  Outputs: 197
Node  75 Operator Custom Name TFLite_Detection_PostProcess
  Inputs: 192 197 1
  Outputs: 198 199 200 201
  Temporaries: 202 203 204

