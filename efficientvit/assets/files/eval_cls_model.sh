# Top1 Acc=79.390, Top5 Acc=94.346
python eval_cls_model.py --model b1-r224 --image_size 224

# Top1 Acc=79.918, Top5 Acc=94.704
python eval_cls_model.py --model b1-r256 --image_size 256

# Top1 Acc=80.410, Top5 Acc=94.984
python eval_cls_model.py --model b1-r288 --image_size 288

# Top1 Acc=82.100, Top5 Acc=95.782
python eval_cls_model.py --model b2-r224 --image_size 224 --crop_ratio 1.0

# Top1 Acc=82.698, Top5 Acc=96.096
python eval_cls_model.py --model b2-r256 --image_size 256 --crop_ratio 1.0

# Top1 Acc=83.086, Top5 Acc=96.302
python eval_cls_model.py --model b2-r288 --image_size 288 --crop_ratio 1.0

# Top1 Acc=83.468, Top5 Acc=96.356
python eval_cls_model.py --model b3-r224 --image_size 224 --crop_ratio 1.0

# Top1 Acc=83.806, Top5 Acc=96.514
python eval_cls_model.py --model b3-r256 --image_size 256 --crop_ratio 1.0

# Top1 Acc=84.150, Top5 Acc=96.732
python eval_cls_model.py --model b3-r288 --image_size 288 --crop_ratio 1.0

# Top1 Acc=84.484, Top5 Acc=96.862
python eval_cls_model.py --model l1-r224 --image_size 224 --crop_ratio 1.0

# Top1 Acc=85.050, Top5 Acc=97.090
python eval_cls_model.py --model l2-r224 --image_size 224 --crop_ratio 1.0

# Top1 Acc=85.366, Top5 Acc=97.216
python eval_cls_model.py --model l2-r256 --image_size 256 --crop_ratio 1.0

# Top1 Acc=85.630, Top5 Acc=97.364
python eval_cls_model.py --model l2-r288 --image_size 288 --crop_ratio 1.0

# Top1 Acc=85.734, Top5 Acc=97.438
python eval_cls_model.py --model l2-r320 --image_size 320 --crop_ratio 1.0

# Top1 Acc=85.868, Top5 Acc=97.516
python eval_cls_model.py --model l2-r352 --image_size 352 --crop_ratio 1.0

# Top1 Acc=85.978, Top5 Acc=97.518
python eval_cls_model.py --model l2-r384 --image_size 384 --crop_ratio 1.0
