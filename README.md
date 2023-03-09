# FP8 Quantization: The Power of the Exponent
This repository contains the implementation and experiments for the paper presented in

**Andrey Kuzmin<sup>\*1</sup>, Mart van Baalen<sup>\*1</sup>,  Yuwei Ren<sup>1</sup>, 
Markus Nagel<sup>1</sup>, Jorn Peters<sup>1</sup>, Tijmen Blankevoort<sup>1</sup> "FP8 Quantization: The Power of the Exponent", NeurIPS 
2022.** [[ArXiv]](https://arxiv.org/abs/2208.09225)

*Equal contribution
<sup>1</sup> Qualcomm AI Research (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.)

You can use this code to recreate the results in the paper.

## Method and Results

In this repository we share the code to reproduce analytical and experimental results on performance of FP8 format with different mantissa/exponent division versus INT8. The first part of the repository allows the user to reproduce 
analytical computations of SQNR for uniform, Gaussian, and Student's-t distibutions. Varying the mantissa/exponent bit-width division changes the trade-off between accurate representation of the data around mean of the distribution, 
and the ability to capture its tails. The more outliers are present in the data, the more exponent bits is useful to allocate for the best results. In the second part we provide the code to reproduce the post-training quantization (PTQ) 
results for MobileNetV2, and Resnet-18 pre-trained on ImageNet.
 


## How to install
Make sure to have Python ≥3.8 (tested with Python 3.8.10) and 
ensure the latest version of `pip` (tested with 21.3.1):
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade --no-deps pip
```

Next, install PyTorch 1.11.0 with the appropriate CUDA version (tested with CUDA 10.0):
```bash
pip install torch==1.11.0 torchvision==0.12.0
```

Finally, install the remaining dependencies using pip:
```bash
pip install -r requirements.txt
```
## Running experiments
### Analytical expected SQNR computations
The main run file to compute the expected SQNR for different distributions using different formats is 
`compute_quant_error.py`. The script takes no input arguments and computes the SQNR for different distributions and formats:
```bash
python compute_quant_error.py
```
### ImageNet experiments
The main run file to reproduce the ImageNet experiments is `image_net.py`. 
It contains commands for validating models quantized with post-training quantization.
You can see the full list of options for each command using `python image_net.py [COMMAND] --help`.
```bash
Usage: image_net.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  validate-quantized
```

To reproduce the experiments run:
```bash
python image_net.py validate-quantized --images-dir </PATH/TO/IMAGENET> 
--architecture <ARCHITECTURE_NAME> --batch-size 64 --seed 10
--model-dir </PATH/TO/PRETRAINED/MODEL> # only needed for MobileNet-V2
--n-bits 8  --cuda --load-type fp32 --quant-setup all --qmethod fp_quantizer --per-channel 
--fp8-mantissa-bits=5 --fp8-set-maxval --no-fp8-mse-include-mantissa-bits
--weight-quant-method=current_minmax --act-quant-method=allminmax --num-est-batches=1 
 ```

where <ARCHITECTURE_NAME> can be mobilenet_v2_quantized or resnet18_quantized. 
Please note that only MobileNet-V2 requires pre-trained weights that can be downloaded here (the tar file is used as it is without a need to untar):
- [MobileNetV2](https://drive.google.com/open?id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR)

## Reference
If you find our work useful, please cite
```
@article{kuzmin2022fp8,
  title={FP8 Quantization: The Power of the Exponent},
  author={Kuzmin, Andrey and Van Baalen, Mart and Ren, Yuwei and Nagel, Markus and Peters, Jorn and Blankevoort, Tijmen},
  journal={arXiv preprint arXiv:2208.09225},
  year={2022}
}
```
