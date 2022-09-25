Directory Structure is:
.
├── Code
│   ├── Q1
│   │   ├── CNN Layers
│   │   │   ├── 3_layers.py
│   │   │   ├── 4_layers.py
│   │   │   └── 5_layers.py
│   │   ├── Filters
│   │   │   ├── 64_128_256.py
│   │   │   ├── 64_256_256.py
│   │   │   └── 64_256_512.py
│   │   ├── Kernel Size for Max-Pooling
│   │   │   ├── kernel_2.py
│   │   │   ├── kernel_3.py
│   │   │   └── kernel_4.py
│   │   ├── Linear Layers
│   │   │   ├── 1_layer.py
│   │   │   ├── 2_layers.py
│   │   │   ├── 3_layers.py
│   │   │   └── 4_layers.py
│   │   ├── Stride for Pooling
│   │   │   └── Stride.py
│   │   ├── Training Time
│   │   │   └── Training_time.py
│   │   └── bestmodel.py
│   └── Q2
│       ├── Q2.1
│       │   └── occlusion.py
│       └── Q2.2
│           ├── Q2.2.1
│           │   └── filter_identify.py
│           └── Q2.2.2
│               └── filtermodify.py
├── Data
│   ├── filters
│   │   ├── Filter_conv1_filter44.png
│   │   ├── Filter_conv1_filter45.png
│   │   ├── Filter_conv2_filter117.png
│   │   ├── Filter_conv2_filter37.png
│   │   ├── Filter_conv3_filter209.png
│   │   ├── Filter_conv3_filter254.png
│   │   ├── Filter_conv4_filter227.png
│   │   └── Filter_conv4_filter62.png
│   ├── occlusion
│   │   ├── ant
│   │   │   └── n0221948600001022.jpg
│   │   ├── catamaran
│   │   │   └── n0298179200000895.jpg
│   │   ├── coral_reef
│   │   │   └── n0925647900000977.jpg
│   │   ├── goose
│   │   │   └── n0185567200000907.jpg
│   │   ├── green_mamba
│   │   │   └── n0174993900000946.jpg
│   │   ├── lion
│   │   │   └── n0212916500000951.jpg
│   │   ├── miniature_poodle
│   │   │   └── n0211371200000915.jpg
│   │   ├── organ
│   │   │   └── n0385406500001044.jpg
│   │   ├── spider_web
│   │   │   └── n0427554800000946.jpg
│   │   └── toucan
│   │       └── n0184338300000933.jpg
│   └── plots
│       ├── ant.png
│       ├── catamaran.png
│       ├── coral_reef.png
│       ├── goose.png
│       ├── green_mamba.png
│       ├── lion.png
│       ├── miniature_poodle.png
│       ├── organ.png
│       ├── spider_web.png
│       └── toucan.png
├── README.txt
├── REPORT.pdf
└── bestmodel
    └── bestmodel.pth

plots directory contain the heat-maps for the images in occlusion directory.
bestmodel.pth is the saved weights for best model
filters contain the randomly chosen filters
Q1 directory contains all the codes used to perform hyper-parameter experiments.