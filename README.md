## FrameSeg 

Fast, responsive semantic segmentation tool for image sequences.

This tool is part of the code release for the paper **Semi-supervised Segmentation of Aerial Videos with Iterative Label Propagation**, ACCV 2020 oral.

Paper: https://arxiv.org/abs/2010.01910

Project page: https://sites.google.com/site/aerialimageunderstanding/semantics-through-time-semi-supervised-segmentation-of-aerial-videos


## Features
- send to back
- hybrid click/drag segmentation
- can iterate ovverlapping polygon
- responsive for high resolutions
- numpy map ouptut with one channel per class
- customizable classes [ `classConfig.yaml` - mandatory ] / language [ `languageConfig.yaml` - optional ]
- python / cross platform / executables available for Linux/Windows (tested on Ubuntu 20.04 and Windows 10)

## Getting started

- the fastest way is to grab an executable + the `classConfig.yaml` file from the release folder
- test on the `sampleData` image
- manual way:

#### Ubuntu 20.04
```bash

conda create -n frameSeg python=3.8
conda activate frameSeg
pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04/wxPython-4.1.1-cp38-cp38-linux_x86_64.whl
pip install opencv-python==4.4.0.46
python frameSeg.py
```
To pack/generate executables:
```bash
conda install -c conda-forge pyinstaller 
pyinstaller -F --noconsole frameSeg.py
```

## Video tutorial

https://youtu.be/N_xnHNkRd_c

## Known Issues

Ubuntu

Q: ```wx._core.PyNoAppError: The wx.App object must be created first!```

A: Lauch from [Visual Studio Code](https://code.visualstudio.com/) (Ctrl+F5)


