# SMDR-IS

This Repo includes the testing codes of our SMDR-IS. (PyTorch Version).

If you use our code, please cite our paper and hit the star at the top-right corner. Thanks!


# Requirement
```
Python 3.7, Pytorch 1.11.0.
```


# Testing
```
1. Download the code
2. Put your testing images in the "data/input" folder
3. Python test.py
4. Find the result in "data/result" folder
```markdown
5. You can find all the pre-trained model in *[Google Drive](https://drive.google.com/file/d/1yC2lw6J4WQfuycWNlwHgIhy87xSQJftg/view?usp=drive_link)* 
```
Note that the PSNR_SSIM.py provide the metrics code adopted our paper.
```

```
The validation data are in the "data/input" folder (underwater images), "data/gt" folder (grount truth images).
```

# Bibtex

```
@article{zhang2023synergistic,
  title={Synergistic Multiscale Detail Refinement via Intrinsic Supervision for Underwater Image Enhancement},
  author={Zhang, Dehuan and Zhou, Jingchun and Zhang, Weishi and Guo, ChunLe and Li, Chongyi},
  journal={arXiv preprint arXiv:2308.11932},
  year={2023}
}
```
#  License
The code is made available for academic research purpose only. This project is open sourced under MIT license.

# Contact
If you have any questions, please contact Jingchun Zhou at zhoujingchun03@qq.com.

