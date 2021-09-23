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

def create_new_anno_dict(num, ins_id, video_id, poly, bbox, area):
    dict_ins = {}
    dict_ins['id'], dict_ins['video_id'] = ins_id, video_id
    dict_ins['category_id'], dict_ins['iscrowd'] = 1, 0
    # print('num=%d'%num)
    dict_ins['segmentations'], dict_ins['bboxes'], dict_ins['areas'] = ['None'] * num, ['None'] * num, ['None'] * num
    dict_ins['segmentations'][0], dict_ins['bboxes'][0], dict_ins['areas'][0] = poly, bbox, area
    return dict_ins


if __name__ == '__main__' :
    # data_path = '/lustre/data_sharing/jbzhang/nlpr_video_dataset/train_dataset/'
    path_list = os.listdir('/lustre/data_sharing/jbzhang/nlpr_video_dataset/train_dataset/')
    with open('/lustre/data_sharing/wfeng/ch3_train_ignore.json', 'r') as f:
        js = json.load(f)
    out_anno = dict()
    out_anno['info'], out_anno['categories'] = js['info'], js['categories']
    out_anno['annotations'], out_anno['videos'] = [], []
    ins_num = 0
    ins_cnt = 1
    for id, path in enumerate(path_list[:100]):
        path = os.path.join('/lustre/data_sharing/jbzhang/nlpr_video_dataset/train_dataset/', path)
        print(id, path)
        img_list = glob.glob(os.path.join(path, '*.jpg'))
        # import pdb; pdb.set_trace()

        img_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        img_num = len(img_list)
        # print(img_num)
        dict_v = {}
        dict_v['length'], dict_v['id'] = img_num, id + 1
        dict_v['file_names'] = []
        ins_list = []

        for i, img_name in enumerate(img_list):
            print(id, i, img_name)
            if i == 0:
                img_0 = cv2.imread(img_name)
                height, width, _ = img_0.shape
                dict_v['height'], dict_v['width'] = height, width
            
            dict_v['file_names'].append(img_name.split('/', 6)[-1])
            txt_name = img_name.replace('.jpg', '.txt')
            with open(txt_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                strs = line.strip().split(',', 7)
                temps = strs[-1].split('\t')
                if (not temps[-1].isdigit()) or int(temps[-1]) > 1999:
                    continue
                ins_idx = int(temps[-1])
                poly, bbox = [], []
                for s in strs[: -1]:
                    poly.append(int(s))
                poly.append(int(temps[0]))
                poly_ = np.array(poly, np.int64).reshape([4, 2])
                poly_[poly_[:, 0] >= width, 0] = width - 1
                poly_[poly_[:, 1] >= height, 1] = height - 1
                poly_[poly_[:, 0] < 0, 0] = 0
                poly_[poly_[:, 1] < 0, 1] = 0
                x_max, x_min, y_max, y_min = poly_[:, 0].max(), poly_[:, 0].min(), poly_[:, 1].max(), poly_[:, 1].min()
                w, h = x_max - x_min, y_max - y_min
                bbox = [x_min, y_min, w, h]
                area = w*h
                poly = poly_.reshape(-1).tolist()
                if not (ins_idx in ins_list):
                    ins_list.append(ins_idx)
                    out_anno['annotations'].append(create_new_anno_dict(img_num, ins_cnt, id+1, poly, bbox, area))
                    ins_cnt += 1
                    # print(ins_idx, img_num)
                    # print(len(out_anno['annotations'][ins_list.index(ins_idx)]['segmentations']))
                else:
                    index = ins_list.index(ins_idx)
                    # print(ins_idx)
                    # print(index, i)
                    # print(img_num)
                    # print(len(out_anno['annotations']))
                    # print(len(out_anno['annotations'][index]['segmentations']))
                    out_anno['annotations'][ins_num + index]['segmentations'][i] = poly
                    out_anno['annotations'][ins_num + index]['bboxes'][i] = bbox
                    out_anno['annotations'][ins_num + index]['areas'][i] = area
        ins_num += len(ins_list)
        out_anno['videos'].append(dict_v)
    with open("/lustre/home/mbzhao/experiments/masktrackrcnn/nlpr_video_train_100.json", "w") as ff:
        json.dump(out_anno, ff, cls=NpEncoder)
    print("加载入文件完成...")




        
        

                                

                
            
            
        
    
    
    
    