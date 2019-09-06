
# Deepstream 4.0 Faster RCNN, SSD, and mask RCNN UFF model object detector

* `deepstream_custom_uff.c` is the test app.
    > Pipeline: filesrc->h264parse->nvv4l2decoder->streammux->nvinfer(frcnn/ssd)->nvosd->nveglglesink

* `nvdsinfer_customparser_frcnn_uff` is the lib to parse the output layers of frcnn detector
* `nvdsinfer_customparser_ssd_uff` is the lib to parse the output layers of ssd detector
* `nvdsinfer_customparser_mrcnn_uff` is the lib to parse the output layers of mask rcnn
* `pgie_frcnn_uff_config.txt` is the nvinfer config file
* `pgie_ssd_uff_config.txt` is the nvinfer config file
* `pgie_mrcnn_uff_config.txt` is the nvinfer config file
* `frnn_label.txt` includes 4 classes
* `ssd_label.txt` includes 3 classes
* `mrcnn_label.txt` includes 81 classes
* Support `Tesla/Tegra` platforms both

## Prequisites:

* [Deepstream 4.0](https://developer.nvidia.com/deepstream-sdk)

* [TensorRT OSS](https://github.com/NVIDIA/TensorRT)
 * Complile a new `libnvinfer_plugin.so` and replace your system libnvinfer_plugin.so. This library implements 2 IPlugins "`cropAndResizePlugin"`and `"proposalPlugin`" for frcnn network, and `"firstDimTile`" for ssd network. "ResizeNearest_TRT" "ProposalLayer_TRT" "PyramidROIAlign_TRT" "DetectionLayer_TRT" "SpecialSlice_TRT" for mask rcnn
 * Download frcnn / ssd uff model from [NGC](https://ngc.nvidia.com/)

## Complie and run
 * $ cd nvdsinfer_customparser_frcnn_uff or nvdsinfer_customparser_ssd_uff or nvdsinfer_customparser_mrcnn_uff
 * $ make
 * $ cd ..
 * $ make
 * Modify config file "pgie_frcnn(ssd or mrcnn)_uff_config.txt" Set "uff-file","model-engine-file" to be your path
 * $ `./deepstream-custom-uff config_file [H264_file]`
   

## Notes
* For frcnn fp16 mode, it needs the below patch and rebuild libnvds_infer.so
```
--- a/src/utils/nvdsinfer/nvdsinfer_context_impl.cpp
+++ b/src/utils/nvdsinfer/nvdsinfer_context_impl.cpp
@@ -1851,7 +1851,7 @@ NvDsInferContextImpl::generateTRTModel(
         }

         if (!uffParser->parse(initParams.uffFilePath,
-                    *network, modelDataType))
+                    *network, DataType::kFLOAT))
```

* For ssd, don't forget to set your own keep_count, keep_top_k in nvdsinfer_custombboxparser_ssd_uff.cpp for NMS layer.
* In "attach_metadata_detector()"
 * frame scale_ratio_x/scale_ratio_y is "network width/height" / "streammux width/height"
 * Some objs will be filtered because its width/height/top/left is beyond the source size (streammux is as source)
* We have to generate tensorRT engine file for frcnn model because there is a work around in network construction and we can't get "dense_class_4/Softmax" layer output directly. -- This issue has been fixed in convert_to_uff python conversion. Uff can be used directly.
* For "mask rcnn", app can show bbox but cannot show mask in present. User can dump mask in nvdsinfer_customparser_mrcnn_uff/nvdsinfer_custombboxparser_mrcnn_uff.cpp -> "out_mask"
