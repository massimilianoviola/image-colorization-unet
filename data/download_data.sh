#!/bin/bash

echo "Starting dataset downloads..."
cd data
bash download_bsds500.sh
bash download_div2k_lr.sh
bash download_sbd.sh
echo "All datasets downloaded!"