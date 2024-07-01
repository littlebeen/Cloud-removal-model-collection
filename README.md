# Cloud-removal-model-collection
A collection of the existing end-to-end cloud removal models

This project includes the code for the MDSA, CVAE, AMGAN, SPAGAN, and MemoryNet models designed for cloud removal. Additionally, it incorporates the data loaders for RICE, WHU, T-Cloud, and a custom dataset named CUHK-CR (located in the 'dataloader' directory under the file named 'My.py'). The respective papers for these models and the download links for the datasets are provided below.

🔥 Congradualation to my collaborators and myself! This paper has been accepted by IEEE Transactions on Geoscience and Remote Sensing! It's a new start! The code for [Diffusion Enhancement for Cloud Removal in Ultra-Resolution Remote Sensing Imagery](https://ieeexplore.ieee.org/abstract/document/10552304/) could be found at [https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal](https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal)

**datasets**

* RICE [paper](https://arxiv.org/abs/1901.00600)  [download link](https://github.com/BUPTLdy/RICE_DATASET)

* T-Cloud [paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Ding_Uncertainty-Based_Thin_Cloud_Removal_Network_via_Conditional_Variational_Autoencoders_ACCV_2022_paper.pdf) [download link](https://pan.baidu.com/s/1LtkcdxMbJQTgEr-JvTM1Ug) pick up code:t63d

* WHU [download link](http://gpcv.whu.edu.cn/data/WHU_Cloud_Dataset.html)

* CUHK-CR [paper](https://ieeexplore.ieee.org/abstract/document/10552304/) [download link](https://github.com/littlebeen/Diffusion-Enhancement-for-CR/tree/main)

**models**

* cvae : [Uncertainty-Based Thin Cloud Removal Network via Conditional Variational Autoencoders](https://openaccess.thecvf.com/content/ACCV2022/papers/Ding_Uncertainty-Based_Thin_Cloud_Removal_Network_via_Conditional_Variational_Autoencoders_ACCV_2022_paper.pdf)

* mdsa : [Cloud Removal in Optical Remote Sensing Imagery Using Multiscale Distortion-Aware Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9686746)

* amgan : [Attention mechanism-based generative adversarial networks for cloud removal in Landsat images](https://pdf.sciencedirectassets.com/271745/1-s2.0-S0034425722X00023/1-s2.0-S0034425722000165/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjELz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIDHn2mggQjzwK94UWF3WGy%2BZ3gTAV5D%2BIBxWUiJfNrL3AiEAxtqfXP2BeavsQVt4Ddbsmuxgi%2FXJwVTFgOZ1GtYxUpEqswUIFRAFGgwwNTkwMDM1NDY4NjUiDG%2BS84EPQIzeA2GgvSqQBUqZxUKSDwoGfTGnMd7RMZkMWwPJiGTLZOFgMKYqqNCgBvh6bKxqFKy1AHXp7%2BfQ2QL0EtqBykCemk52DOQ6cr%2BqsZsKKMNnN3dCApxHKs89YqjkcFLBhNxkOBdqeUpSayqbUbMm696nUxozun%2FMUw0qGjevZiDLQt2JRxrhQzqsQ%2FG%2FK7IhQ46xErNe4Bkq2VYP7pO8VnKEcGOz8NOlHJSsxWH5E58O%2BX9pHoVgxno%2Bylg%2FeCtD%2FYNpw23nuYZYgv90gatRUY0SXFFZi3c46d4pIbsmReUaXHScJHRrEzJAji4cXC%2BARVnTToeT4rE3T%2B%2FyuQef5kBCNby5ZKwndZpfmpohA%2BUszTr%2F3JR%2BAd39ibREHNy1LIQzWKRd13f5kfdksSpIquN055OwNWfTHFt6cC8ONAjQ70QXONt74r5xGl1cs7Q38Lo%2BtEAdHiPMs1WepDcg5k5%2B5djwMKs2reU5q3bX0g5liOQJW0lkoDLL9nfkdgJU%2BxW0sSERG4EuFcQXPcB2Vj2K00wKVOzegODrJdHNxGpCA2USGvqdIq1P3Xp94vvPuWFSq%2BatFrbr9i1fDY8hGm%2FFET69qlhRLnOUeOVLftZSxhjD%2Fn25VJrHDmDedN2h6vuvc%2FGFZm2sEGnk6SeWLSgCRBpz206gJOJDtBIPIuIbIAuv5OGyr%2F3sTJhczzmQ6zawDyGqbx9J%2FIjI4Ff0gDgNb2bT7tYx0WkOQmNna8l6yRtTrGdMtbonbJXOodqYjekym8r8ICgLAlp8Dtwf8bTePBlrz6xwxLUbSn9eWkjur%2Fov4QNpfjKG2lOJ7fjGINeoSwWwNb1PS1HJJcqCGbjGHzFByfTBWo05P4AcmQOfI3y4o7fu3dyRMIS18qoGOrEB5yNN41QNih5ob9GjOgQqct4sWY%2B5YMkJjtoYOipoqvFM346ZiQLZQelhOEUphl%2FSZGQ26mjtieazDBq1Cp6qwQdIZ7BsHp0myWRlmixsXkfgOET61wUPh9dlPt2IlfW42mfbYawv4tNNGNTnT4c8ImkGE%2FDFScm7HaQGL6YlOx2F58QBUQiGEy%2FI0m7CUxil%2FHP03qQiolSdn0n7j6Dyth2NMNR9F%2Fgt1bAaiqUSOHNs&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231121T130048Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYS6O5Y6OY%2F20231121%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8265152c8e2cce0eecdd9b4cbcfc6171fd00c5e0395661002bb663b0c01ed813&hash=e265975663913221d38df8c949f7700a5e5cdc0e39bfbb9554c8569d3daaaffc&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0034425722000165&tid=spdf-f997a002-1d50-4a41-aca5-8a3bd7f23c36&sid=d73831005a11c44746296ac-ca44c29026dfgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=19085e5557565304515256&rr=82991e237abf9e62&cc=cn)

* spagan : [Cloud Removal for Remote Sensing Imagery vai Spatial Attention Generative Adversarial Network](https://arxiv.org/ftp/arxiv/papers/2009/2009.13015.pdf)

* memorynet : [Memory augment is All You Need for image restoration](https://arxiv.org/abs/2309.01377)

 # Usage

**Train**

1. Add your output dir path in the config and choose the model you need (mn, mdsa, cvae, spagan, amgan)
2. Change the dataset path in the dataload/xx.py
3. python train.py

**Test**

1. Put the pretained model in the pre_train dir and change the config
2. python test.py (metrics include SSIM, PSNR and LPIPS)

# Cite

If this project is useful to you, please cite this paper :)

```
@article{sui2024diffusion,
  author={Sui, Jialu and Ma, Yiyang and Yang, Wenhan and Zhang, Xiaokang and Pun, Man-On and Liu, Jiaying},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Diffusion Enhancement for Cloud Removal in Ultra-Resolution Remote Sensing Imagery}, 
  year={2024},
  volume={62},
  pages={1-14},
  doi={10.1109/TGRS.2024.3411671}}
```

If you have any questions, be free to contact me!
