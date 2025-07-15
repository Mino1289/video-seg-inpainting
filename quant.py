from pathlib import Path
from ultralytics import YOLO
import shutil
import openvino as ov
import nncf
import argparse

from zipfile import ZipFile

from ultralytics.data.utils import DATASETS_DIR
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ops

from notebook_utils import download_file

# parameters for quantization (to cli)
# to_quantize = True  # Set to True if you want to quantize the model
model_id = [
    "yolo11n-seg",
    "yolo11s-seg",
    "yolo11m-seg",
    "yolo11l-seg",
    "yolo11x-seg",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a YOLO-seg model.")
    parser.add_argument('--model-name', type=str, default='yolo11s-seg', choices=model_id, help='YOLO model name')
    parser.add_argument('--openvino', action='store_true', help='Export model to OpenVINO format.')
    parser.add_argument('--quantize', action='store_true', help='Quantize the OpenVINO model.')
    
    args = parser.parse_args()

    if args.quantize and not args.openvino:
        parser.error("--quantize requires --openvino.")

    SEG_MODEL_NAME = args.model_name

    seg_model = YOLO(f"{SEG_MODEL_NAME}.pt")
    label_map = seg_model.model.names

    # instance segmentation model
    seg_model_path = Path(f"{SEG_MODEL_NAME}_openvino_model/{SEG_MODEL_NAME}.xml")
    if args.openvino and not seg_model_path.exists():
        print(f"Exporting {SEG_MODEL_NAME} to OpenVINO format...")
        seg_model.export(format="openvino", dynamic=True, half=True)
        print("Export complete.")
        
    if args.quantize:
        int8_model_seg_path = Path(f"{SEG_MODEL_NAME}_openvino_model_int8/{SEG_MODEL_NAME}.xml")
        quantized_seg_model = None
        

        if not int8_model_seg_path.exists():
            print("Starting quantization process...")
            DATA_URL = "http://images.cocodataset.org/zips/val2017.zip"
            LABELS_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
            CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/v8.1.0/ultralytics/cfg/datasets/coco.yaml"

            OUT_DIR = DATASETS_DIR

            DATA_PATH = OUT_DIR / "val2017.zip"
            LABELS_PATH = OUT_DIR / "coco2017labels-segments.zip"
            CFG_PATH = OUT_DIR / "coco.yaml"

            if not (OUT_DIR / "coco/labels").exists():
                download_file(DATA_URL, DATA_PATH.name, DATA_PATH.parent)
                download_file(LABELS_URL, LABELS_PATH.name, LABELS_PATH.parent)
                download_file(CFG_URL, CFG_PATH.name, CFG_PATH.parent)
                with ZipFile(LABELS_PATH, "r") as zip_ref:
                    zip_ref.extractall(OUT_DIR)
                with ZipFile(DATA_PATH, "r") as zip_ref:
                    zip_ref.extractall(OUT_DIR / "coco/images")

            args = get_cfg(cfg=DEFAULT_CFG)
            args.data = str(CFG_PATH)
            seg_validator = seg_model.task_map[seg_model.task]["validator"](args=args)
            seg_validator.data = check_det_dataset(args.data)
            seg_validator.stride = 32
            seg_data_loader = seg_validator.get_dataloader(OUT_DIR / "coco/", 1)

            seg_validator.is_coco = True
            seg_validator.class_map = coco80_to_coco91_class()
            seg_validator.names = label_map
            seg_validator.metrics.names = seg_validator.names
            seg_validator.nc = 80
            seg_validator.nm = 32
            seg_validator.process = ops.process_mask
            seg_validator.plot_masks = []

            def transform_fn(data_item: dict):
                """
                Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
                Parameters:
                data_item: Dict with data item produced by DataLoader during iteration
                Returns:
                    input_tensor: Input data for quantization
                """
                input_tensor = seg_validator.preprocess(data_item)["img"].numpy()
                return input_tensor

            quantization_dataset = nncf.Dataset(seg_data_loader, transform_fn)

            core = ov.Core()
            seg_ov_model = core.read_model(seg_model_path)


            if not int8_model_seg_path.exists():
                ignored_scope = nncf.IgnoredScope(  # post-processing
                    subgraphs=[
                        nncf.Subgraph(inputs=[f"__module.model.{22 if 'v8' in SEG_MODEL_NAME else 23}/aten::cat/Concat",
                                            f"__module.model.{22 if 'v8' in SEG_MODEL_NAME else 23}/aten::cat/Concat_1",
                                            f"__module.model.{22 if 'v8' in SEG_MODEL_NAME else 23}/aten::cat/Concat_2",
                                            f"__module.model.{22 if 'v8' in SEG_MODEL_NAME else 23}/aten::cat/Concat_7"],
                                    outputs=[f"__module.model.{22 if 'v8' in SEG_MODEL_NAME else 23}/aten::cat/Concat_8"])
                    ]
                )

                # Segmentation model
                print("Quantizing segmentation model...")
                quantized_seg_model = nncf.quantize(
                    seg_ov_model,
                    quantization_dataset,
                    preset=nncf.QuantizationPreset.MIXED,
                    ignored_scope=ignored_scope
                )

                print(f"Quantized segmentation model will be saved to {int8_model_seg_path}")
                ov.save_model(quantized_seg_model, str(int8_model_seg_path))
                shutil.copy(seg_model_path.parent / "metadata.yaml", int8_model_seg_path.parent / "metadata.yaml")
                print("Quantization complete.")
            else:
                print(f"Quantized model already exists at {int8_model_seg_path}. Skipping quantization.")
    
    print("Script finished.")