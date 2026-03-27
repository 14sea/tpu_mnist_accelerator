# TPU MNIST Accelerator (MLP + CNN) for AX301 FPGA

INT8 inference accelerator using a 4x4 systolic array on Cyclone IV.

## Hardware Target

- Board: Heijin AX301
- FPGA: Altera Cyclone IV EP4CE6F17C8

## Project Layout

- `rtl/`: TPU RTL (`systolic_array_4x4`, UART, inference engine)
- `quartus/`: Quartus project files
- `host_infer.py`: MLP host inference client
- `cnn/host_cnn_infer.py`: CNN host inference (conv on host, FC on FPGA)
- `model/` and `cnn/model/`: training + quantization scripts
- `weights/` and `cnn/weights/`: exported model weights and test vectors
- `sim/`: simulation testbenches

## Prerequisites

- Quartus Prime Lite
- Python 3 + `pyserial`, `numpy`, `scikit-learn`, `pillow`
- `openFPGALoader`

## Build and Program Bitstream

```bash
cd quartus
quartus_sh --flow compile tpu_demo
quartus_cpf -c -o bitstream_compression=off tpu_demo.sof ../tpu_demo.rbf
cd ..
openFPGALoader -c usb-blaster tpu_demo.rbf
```

## Run Inference

```bash
# MLP path
python3 host_infer.py --port /dev/ttyUSB0 --samples 10

# CNN path
python3 cnn/host_cnn_infer.py --port /dev/ttyUSB0 --samples 10
```

## Optional: Regenerate Weights

```bash
python3 model/train.py
python3 cnn/model/train_cnn.py --epochs 20
```

## Notes

- Keep generated Quartus reports and bitstreams out of source control.
- If sharing prebuilt bitstreams, publish them in Releases with matching commit tags.
