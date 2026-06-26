# CoS-V2X: Cooperative Vehicle-Infrastructure End-to-End Planning

CoS-V2X is the vehicle-infrastructure cooperative (V2X) end-to-end planning model
from **VIPS** (ECCV 2026). It extends
[SparseDrive](https://github.com/swc-17/SparseDrive) to the cooperative V2X
setting on the [V2X-Real](https://mobility-lab.seas.ucla.edu/v2x-real/) dataset,
fusing vehicle and infrastructure camera streams under a bandwidth-limited
(top-100 anchor) transmission scheme.

Hoonhee Cho, Jae-Young Kang, Giwon Lee, Hyemin Yang, Heejun Park, Kuk-jin Yoon<br>
KAIST, Visual Intelligence Lab

- **Project page:** https://vips2026.github.io/
- **VIPS benchmark + evaluation code:** https://github.com/mickeykang16/VIPS
- **Paper / arXiv:** _coming soon_

CoS-V2X is evaluated with the VIPS two-stage pseudo-simulation benchmark — see the
VIPS repository for the reported numbers and the cooperative evaluation protocol.

## Architecture

<p align="center">
  <img src="resources/cos_v2x_architecture.png" width="820" alt="CoS-V2X architecture">
</p>

## Installation

CoS-V2X uses an mmcv / mmdet (CUDA) stack, separate from the VIPS `vips` env.

```bash
conda create -n cos_v2x python=3.8 -y
conda activate cos_v2x

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirement.txt

# compile the deformable-aggregation CUDA op
cd projects/mmdet3d_plugin/ops && python setup.py develop && cd ../../../
```

## Data

CoS-V2X trains and evaluates on the cooperative
[V2X-Real](https://mobility-lab.seas.ucla.edu/v2x-real/) dataset. Point the
configs at your prepared V2X-Real data (configs default to `data/v2xreal/`):

```bash
export V2XREAL_DATA_ROOT=/path/to/v2xreal/data
```

See the [VIPS repository](https://github.com/mickeykang16/VIPS) for the V2X-Real
evaluation assets and the cooperative data layout.

## Checkpoints

Pretrained CoS-V2X weights are hosted on Hugging Face:

```bash
hf download mickeykang/CoS-V2X --local-dir checkpoints
# then point the config / test script at the downloaded .pth
```

## Training

_Coming soon._

## Open-loop evaluation

_Coming soon._

## Use with the VIPS benchmark

To run CoS-V2X inside the VIPS pseudo-simulation benchmark, clone this repo into
the VIPS `models/` directory and follow the VIPS "Evaluation → CoS-V2X" section:

```bash
git clone https://github.com/mickeykang16/CoS-V2X models/CoS-V2X
```

## Citation

```bibtex
@inproceedings{cho2026vips,
  title     = {{VIPS}: Vehicle-Infrastructure Cooperative Planning Benchmark via Pseudo-Simulation},
  author    = {Cho, Hoonhee and Kang, Jae-Young and Lee, Giwon and Yang, Hyemin and Park, Heejun and Yoon, Kuk-jin},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

CoS-V2X builds on SparseDrive — please also cite:

```bibtex
@article{sun2024sparsedrive,
  title={SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation},
  author={Sun, Wenchao and Lin, Xuewu and Shi, Yining and Zhang, Chuang and Wu, Haoran and Zheng, Sifa},
  journal={arXiv preprint arXiv:2405.19620},
  year={2024}
}
```

## License

Released under the MIT License (see [LICENSE](LICENSE)). The original SparseDrive
code is © swc-17 (MIT); the V2X-cooperative modifications are © KAIST Visual
Intelligence Lab.

## Acknowledgement

Built on [SparseDrive](https://github.com/swc-17/SparseDrive),
[Sparse4D](https://github.com/HorizonRobotics/Sparse4D), and
[mmdetection3d](https://github.com/open-mmlab/mmdetection3d); evaluated on
[V2X-Real](https://mobility-lab.seas.ucla.edu/v2x-real/).
