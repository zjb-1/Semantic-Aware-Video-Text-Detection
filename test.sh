CUDA_VISIBLE_DEVICES=6 python /lustre/home/jbzhang/experiment/Swin-Track/tools/test_video.py mask_track_rcnn_r50_fpn.py work_dirs/epoch_12.pth --out output_icdar.pkl --eval segm \
--show --show-dir "/lustre/home/jbzhang/experiment/data/vis_test_one/"
