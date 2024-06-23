from pycocotools.coco import COCO
import numpy as np
import cv2, os, argparse, json, datetime, random
from collections import defaultdict

class CocoToSingleJson:
    def __init__(self, ann_dir, visulizaiton_save_dir, image_save_dir):
        self.ann_dir = ann_dir
        self.visulizaiton_save_dir = os.path.join(visulizaiton_save_dir, 'visulizaiton')
        self.image_save_dir = image_save_dir
        if not os.path.exists(self.visulizaiton_save_dir):
            os.makedirs(self.visulizaiton_save_dir)
    
    def load_json(self, file_path):
        with open(file_path, 'r') as load_f:
            load_dict = json.load(load_f)
        load_f.close()
        return load_dict
    
    def get_json_file_list(self):
        json_file_names = os.listdir(self.ann_dir)
        json_file_names.sort()
        return json_file_names
        
    def get_ann(self):
        json_file_names = self.get_json_file_list()
        for json_file_name in json_file_names:
            json_file_path = os.path.join(self.ann_dir, json_file_name)
            dict_f = self.load_json(json_file_path)
            id_cls = {}
            catIds = dict_f['categories']
            for catId in catIds:
                id_cls[catId['id']] = catId['name']
            
            images = dict_f['images']
            image_name = images[0]['file_name']
            img = cv2.imread(os.path.join(self.image_save_dir, image_name))
            annotations = dict_f['annotations']
            for annotation in annotations:
                cls = annotation['category_id']
                segmentation = annotation['segmentation']
                segs_np = []
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for seg in segmentation:
                    seg_np = np.array(seg, dtype=np.int32).reshape([-1,2])
                    segs_np.append(seg_np)
                img_s = cv2.drawContours(img, segs_np, -1, color, 3)
                cv2.imwrite(os.path.join(self.visulizaiton_save_dir, image_name), img_s)
            
            
        
        

    def generate_train_label(self):
        self.get_ann()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", default='/root/code/gbicc_data_processing/temp/jsonfile/annotations',
                        type=str, help="the dir of josn label")
    parser.add_argument("--visulizaiton_save_dir", default='/root/code/gbicc_data_processing/temp', type=str,
                        help="the path to save generate label")
    parser.add_argument("--image_save_dir", default='/root/code/gbicc_data_processing/temp/try_label_segdata', type=str,
                        help="the path to save generate label")
    cfg = parser.parse_args()

    gbicc = CocoToSingleJson(cfg.ann_dir, cfg.visulizaiton_save_dir, cfg.image_save_dir)
    gbicc.generate_train_label()