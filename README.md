# Segment Anything (SAM): faster inference fork

> **You are on `fp16-accumulate`.** Research fork of [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) (base `dca509f`; `main` mirrors upstream). Branches: `portable` = plain-PyTorch changes only (ONNX / TensorRT exportable), `fp16-accumulate` = `portable` with PyTorch fp16 GEMM accumulation enabled by default (no custom kernel; output precision differs), `pr/*` = the small changes that could go upstream, each alone on upstream `main`; `realistic-prs` = all of them together.
> Licence as upstream: [Apache 2.0](LICENSE) (code); checkpoints under their upstream licence (not included, link only). On `portable` and `fp16-accumulate` no upstream file is modified except this README; the only new file is `segment_anything/fast.py`, which carries a notice that it was added in this fork (Apache-2.0 section 4b). The `pr/*` branches are upstream-PR candidates and keep upstream's file headers.

```python
from segment_anything import sam_model_registry
from segment_anything.fast import FastSamPredictor   # same API as SamPredictor
pred = FastSamPredictor(sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth").cuda())  # on `fp16-accumulate`: PyTorch fp16 accumulation on by default
pred.set_image(image)                                  # uint8 HxWx3
masks, scores, logits = pred.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
```

**How much of the speed-up could be merged upstream?** One harness for all rows: `set_image` + one-point `predict` (3 masks to the host), 1297x840 image, RTX 5060 Ti 16 GB, torch 2.14.1+cu130, 2026-10-12, cold process, median of 20 runs after 3 warm-up calls. Memory = torch peak allocated / reserved over load + runs (includes weights). Accuracy = mean mask IoU vs upstream fp32 over 336 masks (21 Mip-NeRF 360 images x 6 prompts).

ViT-B:

| level | ms | speed-up | peak MB alloc / reserved | mean mask IoU vs upstream fp32 |
|---|---|---|---|---|
| upstream default (fp32) | 172.2 | 1.00x | 2794 / 3148 | reference |
| upstream + recommended fast flags (fp16 autocast around `set_image`, `cudnn.benchmark`, TF32) | 101.3 | 1.70x | 1781 / 2632 | 0.9906 |
| + realistic upstream PRs (the 3 small PRs below, as one subset, with the same flags) | 53.7 | 3.21x | 407 / 488 | 0.9894 |
| `portable` (all plain-PyTorch changes, exportable) | 47.3 | 3.64x | 361 / 492 | 0.9894 |
| `fp16-accumulate` (PyTorch fp16 GEMM accumulation) | 41.5 | 4.15x | 361 / 492 | 0.9902 |

**Realistically mergeable upstream: 3.21x of 4.15x** (ViT-B). ViT-L: 3.25x of 4.37x; ViT-H: 3.20x of 4.31x.

| ms (speed-up) / MB alloc | ViT-L | ViT-H |
|---|---|---|
| upstream fp32 | 410.8 / 4425 | 735.9 / 5762 |
| upstream + recommended fast flags | 199.3 (2.06x) / 3413 | 308.0 (2.39x) / 5366 |
| + realistic PRs | 126.4 (3.25x) / 1197 | 230.1 (3.20x) / 2518 |
| `portable` | 114.7 (3.58x) / 1197 | 214.0 (3.44x) / 2518 |
| `fp16-accumulate` | 93.9 (4.37x) / 1197 | 170.7 (4.31x) / 2518 |
| mean mask IoU vs fp32 (flags / PRs / fast / kernels) | 0.9949 / 0.9955 / 0.9956 / 0.9936 | 0.9979 / 0.9974 / 0.9974 / 0.9964 |

Startup (model load) 0.9 s for ViT-B (fork = upstream). First-call overhead: upstream +0.5 s; fork rows +1.0 s (cudnn autotune of the convs, `cudnn.benchmark`; the flags row pays it too). New image size: +0 ms (eager PyTorch, nothing compiled per size; upstream +0 ms). SAM masks of ambiguous single points move with tiny numeric changes (a 1-LSB resize difference alone gives 0.976 mean IoU), so the 0.989 to 0.990 mean IoU of every non-reference row is at the noise level of the model; the precision-only flags row is already at 0.9906.

Realistic PRs (each alone on upstream `main`, ViT-B, ms saved on the preceding row; drafts only, nothing posted):
- `pr/sdpa-relpos-attention`: relative-position bias folded into a fused SDPA call, no N x N attention matrix. 101.3 to 63.2 ms (-38.1 ms, with fp16 autocast), peak 1781 to 736 MB alloc. Draft `01_PR_sdpa_relpos_attention.md`.
- `pr/half-encoder`: `SamPredictor(..., encoder_dtype=torch.float16)` + `LayerNorm2d` via `F.layer_norm`. 172.2 to 83.5 ms alone (vs 101.3 ms for autocast: -17.8 ms), peak 1444 MB. Draft `02_PR_half_encoder.md`.
- `pr/gpu-preprocess`: opt-in `gpu_preprocess=True`, PIL-style two-pass resize on the GPU. 101.3 to 93.9 ms (-7.4 ms). Draft `03_PR_gpu_preprocess.md`.

Fork-only: `fast.py` replaces the whole encoder forward (window partition, cached one-hot tables, fused neck) and `fp16-accumulate` flips the global PyTorch fp16-accumulation flag: a different code path next to the existing one, and a numerics trade-off, so neither is a small PR.
Raw results, commands and per-run times: `results/2026-10-12_sam-b_release/` in the research repo (bench harness `rel_bench.py`, `queue.sh`).

---

## Latest updates -- SAM 2: Segment Anything in Images and Videos

Please check out our new release on [**Segment Anything Model 2 (SAM 2)**](https://github.com/facebookresearch/segment-anything-2).

* SAM 2 code: https://github.com/facebookresearch/segment-anything-2
* SAM 2 demo: https://sam2.metademolab.com/
* SAM 2 paper: https://arxiv.org/abs/2408.00714

 ![SAM 2 architecture](https://github.com/facebookresearch/segment-anything-2/blob/main/assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)** is a foundation model towards solving promptable visual segmentation in images and videos. We extend SAM to video by considering images as a video with a single frame. The model design is a simple transformer architecture with streaming memory for real-time video processing. We build a model-in-the-loop data engine, which improves model and data via user interaction, to collect [**our SA-V dataset**](https://ai.meta.com/datasets/segment-anything-video), the largest video segmentation dataset to date. SAM 2 trained on our data provides strong performance across a wide range of tasks and visual domains.

# Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM design](assets/model_diagram.png?raw=true)

The **Segment Anything Model (SAM)** produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt:

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

Additionally, masks can be generated for images from the command line:

```
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

See the examples notebooks on [using SAM with prompts](/notebooks/predictor_example.ipynb) and [automatically generating masks](/notebooks/automatic_mask_generator_example.ipynb) for more details.

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## ONNX Export

SAM's lightweight mask decoder can be exported to ONNX format so that it can be run in any environment that supports ONNX runtime, such as in-browser as showcased in the [demo](https://segment-anything.com/demo). Export the model with

```
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

See the [example notebook](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb) for details on how to combine image preprocessing via SAM's backbone with mask prediction using the ONNX model. It is recommended to use the latest stable version of PyTorch for ONNX export.

### Web demo

The `demo/` folder has a simple one page React app which shows how to run mask prediction with the exported ONNX model in a web browser with multithreading. Please see [`demo/README.md`](https://github.com/facebookresearch/segment-anything/blob/main/demo/README.md) for more details.

## <a name="Models"></a>Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Dataset

See [here](https://ai.facebook.com/datasets/segment-anything/) for an overview of the datastet. The dataset can be downloaded [here](https://ai.facebook.com/datasets/segment-anything-downloads/). By downloading the datasets you agree that you have read and accepted the terms of the SA-1B Dataset Research License.

We save masks per image as a json file. It can be loaded as a dictionary in python in the below format.

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # Image id
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
}

annotation {
    "id"                    : int,              # Annotation id
    "segmentation"          : dict,             # Mask saved in COCO RLE format.
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "area"                  : int,              # The area in pixels of the mask
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "stability_score"       : float,            # A measure of the mask's quality
    "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
    "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
}
```

Image ids can be found in sa_images_ids.txt which can be downloaded using the above [link](https://ai.facebook.com/datasets/segment-anything-downloads/) as well.

To decode a mask in COCO RLE format into binary:

```
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

See [here](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py) for more instructions to manipulate masks stored in RLE format.

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The Segment Anything project was made possible with the help of many contributors (alphabetical):

Aaron Adcock, Vaibhav Aggarwal, Morteza Behrooz, Cheng-Yang Fu, Ashley Gabriel, Ahuva Goldstand, Allen Goodman, Sumanth Gurram, Jiabo Hu, Somya Jain, Devansh Kukreja, Robert Kuo, Joshua Lane, Yanghao Li, Lilian Luong, Jitendra Malik, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, Bram Wasti, Zachary Winstrom

## Citing Segment Anything

If you use SAM or SA-1B in your research, please use the following BibTeX entry.

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
