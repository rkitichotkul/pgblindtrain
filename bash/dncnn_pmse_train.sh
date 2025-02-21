python train/main.py train \
    --datadir data \
    --modeltype dncnn \
    --modeldir result/dncnn_pmse_0/dncnn_pmse_0.pth \
    --logdir result/dncnn_pmse_0 \
    --numlayers 17 \
    --sigma 0 \
    --alpha 0.01 \
    --loss mse \
    --learnrate 1e-3 \
    --epochs 15 \
    --logevery 50
