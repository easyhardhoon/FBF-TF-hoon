**APP_DOT**

modified (FBF-TF::Work_Hoon) + yolo_output_parsing + get_mAP + DOT(IEIE)

## Change delegation policy in tensorflowlite

1. use YOLOv4-tiny (not simple CNNS like Lenet with MNIST)
2. change TFLite's default delegation mechnanism (always choose FirstNLargest partition)
3. can choose each partition flexibly 
4. It would be better to choose "partition with more params" than to "partition with more layers"
5. Large partitions are not always the most parameterized
6. Evaluate each case by [yolo_output_parsing + get_mAP]
