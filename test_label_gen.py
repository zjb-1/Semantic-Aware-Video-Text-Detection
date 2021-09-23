import pdb
import os
import numpy as np
import glob
import cv2
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)



# if __name__ == '__main__':
#     # data_path = '/lustre/data_sharing/jbzhang/nlpr_video_dataset/train_dataset/'
#     path_list = os.listdir('/lustre/data_sharing/jbzhang/nlpr_video_dataset/train_dataset/')
#     with open('/lustre/data_sharing/wfeng/ch3_test.json', 'r') as f:
#         js = json.load(f)
#     out_anno = dict()
#     out_anno['info'], out_anno['categories'] = js['info'], js['categories']
#     out_anno['videos'] = []
#     ins_num = 0
#     ins_cnt = 1
#     for id, path in enumerate(path_list[100:200]):
#         path = os.path.join('/lustre/data_sharing/jbzhang/nlpr_video_dataset/train_dataset/', path)
#         print(id, path)
#         img_list = glob.glob(os.path.join(path, '*.jpg'))
#
#         img_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
#         img_num = len(img_list)
#         # print(img_num)
#         dict_v = {}
#         dict_v['length'], dict_v['id'] = img_num, id + 1
#         dict_v['file_names'] = []
#         ins_list = []
#
#         for i, img_name in enumerate(img_list):
#             print(id, i, img_name)
#             if i == 0:
#                 img_0 = cv2.imread(img_name)
#                 height, width, _ = img_0.shape
#                 dict_v['height'], dict_v['width'] = height, width
#
#             dict_v['file_names'].append(img_name.split('/', 6)[-1])
#
#         out_anno['videos'].append(dict_v)
#     with open("/lustre/home/mbzhao/experiments/masktrackrcnn/nlpr_video_test_100-200.json", "w") as ff:
#         json.dump(out_anno, ff, cls=NpEncoder)
#     print("加载入文件完成...")


if __name__ == '__main__':
    data_path = '/lustre/data_sharing/jbzhang/nlpr_video_dataset/test_dataset/'
    path_list = os.listdir('/lustre/data_sharing/jbzhang/nlpr_video_dataset/test_dataset/')

    anno = dict()
    for idx, video_name in enumerate(path_list[:]):
        path = os.path.join(data_path, video_name)
        print(idx, path)
        img_list = glob.glob(os.path.join(path, '*.jpg'))
        img_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        dict_v = dict()

        ins_list = []
        for id, img_name in enumerate(img_list):
            print(idx, id, img_name)
            gt_name = img_name.replace('.jpg', '.txt')
            with open(gt_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                strs = line.strip().split('\t')
                if (not strs[-1].isdigit()) or int(strs[-1]) > 1999:
                    continue
                poly = strs[0].replace(',', '_')
                txt, ins_idx = strs[-2], int(strs[-1])
                state = 'LOW' if '###' in txt else 'MODERATE'
                s = str(id+1) + ',' + txt + ',' + state + ',' + poly

                if not(ins_idx in ins_list):
                    ins_list.append(ins_idx)
                    dict_v[str(ins_idx)] = {}
                    dict_v[str(ins_idx)]['track'] = []
                    dict_v[str(ins_idx)]['track'].append(s)
                    dict_v[str(ins_idx)]['trans'] = txt
                else:
                    dict_v[str(ins_idx)]['track'].append(s)
                    if txt == '###':
                        dict_v[str(ins_idx)]['trans'] = txt
        anno[video_name] = dict_v
    with open('/lustre/home/mbzhao/experiments/masktrackrcnn/nlpr_video_test_mot.json', 'w', encoding='utf-8') as ff:
        json.dump(anno, ff)
    print("加载入文件完成...")


































