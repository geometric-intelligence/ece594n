# 3D segmentation

Author: Ana María Cárdenas
Repository: https://github.com/anuzk13/pointcloud_project
Slides: https://github.com/anuzk13/pointcloud_project
Project exploring 3D segmentation of pointclouds.


## Models for semantic segmentation:

- [Open3D ML](https://github.com/isl-org/Open3D-ML/tree/main?tab=readme-ov-file#model-zoo): 
Clone the library and follow instructions for installation

# Models
- RandLA-Net ([github](https://github.com/QingyongHu/RandLA-Net)) Hu, Qingyong, et al. "Randla-net: Efficient semantic segmentation of large-scale point clouds." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
- Point Transformer ([github](https://github.com/Pointcept/Pointcept))
Zhao, Hengshuang, et al. "Point transformer." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
- Point Net ([github](https://github.com/charlesq34/pointnet)) Qi, Charles Ruizhongtai, et al. "Pointnet++: Deep hierarchical feature learning on point sets in a metric space." Advances in neural information processing systems 30 (2017).

# Projects referenced:
- https://github.com/carlos-argueta/open3d_experiments
- https://towardsdatascience.com/point-net-for-semantic-segmentation-3eea48715a62 

# Requirements:
Open3D ML is not supported in windows

```
pip install requirements.txt
```

## Other software

### Colmap

Use colmap to create pointclouds from images https://colmap.github.io/

### Meshlab

Use meshlab to clean and align pointclouds https://www.meshlab.net/

## Gaussian Splats

Use luma-ai (https://lumalabs.ai/) or the original  Gaussian Splatting paper to create Gaussian Splats (https://github.com/graphdeco-inria/gaussian-splatting)

- [Clay Splat](https://lumalabs.ai/embed/4ae1a520-3c74-485f-85a2-c3138121914a?mode=sparkles&background=%23ffffff&color=%23000000&showTitle=true&loadBg=true&logoPosition=bottom-left&infoPosition=bottom-right&cinematicVideo=undefined&showMenu=false)

- [Living Room Splat](https://lumalabs.ai/embed/27aaa0e4-6042-448f-ae50-5db06eff379b?mode=sparkles&background=%23ffffff&color=%23000000&showTitle=true&loadBg=true&logoPosition=bottom-left&infoPosition=bottom-right&cinematicVideo=undefined&showMenu=false)