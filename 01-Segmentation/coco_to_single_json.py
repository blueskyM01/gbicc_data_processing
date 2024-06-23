from pycocotools.coco import COCO
import numpy as np
import cv2, os, argparse, json, datetime
from collections import defaultdict

class CocoToSingleJson:
    def __init__(self, ann_dir, ann_name, label_save_dir):
        self.ann_dir = ann_dir
        self.ann_name = ann_name
        self.label_save_dir = label_save_dir
        self.nava_catIds = self.load_json("./configure/nava_catIds.json")
        self.nava_supercategory = self.load_json("./configure/nava_supercategory.json")
        self.info, self.licenses = self.get_coco_info_licenses()
        self.info["description"] = "InstanceSegment"
        self.info["version"] = "0.1.0"
        self.info["year"] = 2024
        self.info["contributor"] = "GBICC"
        self.licenses[0]["id"] = 1
        self.licenses[0]["name"] = "License"
    
    def load_json(self, file_path):
        with open(file_path, 'r') as load_f:
            load_dict = json.load(load_f)
        load_f.close()
        return load_dict
    
    def get_coco_info_licenses(self):
        annFile = os.path.join(self.ann_dir, self.ann_name)
        with open(annFile, 'r') as load_f:
            load_dict = json.load(load_f)
        load_f.close()
        print(load_dict["categories"])
        info = load_dict["info"]
        licenses = load_dict["licenses"]
        return info, licenses
        
    def get_ann(self):
        name_seg_id = defaultdict(list)  # 创建一个字典，值的type是list
        annFile = os.path.join(self.ann_dir, self.ann_name)
        # initialize COCO api for instance annotations
        coco = COCO(annFile)
        # display COCO categories and supercategories
        CatIds = sorted(coco.getCatIds())  # 获得满足给定过滤条件的category的id
        # 使用指定的id加载category
        # [{}, {}, ...], {'id': 1, 'name': 'car', 'supercategory': ''}
        cats = coco.loadCats(CatIds)
        
        source_cat_names = {}
        # 将原ann中种类的id与name存成dict，id为key， name为value，如下：
        # {1: 'person', 2: 'car', 3: 'scarf', 4: 'schoolbag'}
        for cat in cats:
            source_cat_names[cat['id']] = cat['name']
        
        with open('./configure/gbicc_catIds.json', "w") as f:
            json.dump(source_cat_names, f)
        f.close()

        # 找出所有category_id的image_id, 参数没有给定的话，指的是数据集中所有图像id
        # image_id = [1, 2, 3, 4, 5,......]
        imgIds = []
        for CatId in CatIds:
            imgIds.extend(list(coco.getImgIds(catIds=[CatId])))
        imgIds = list(set(imgIds))

        # 使用给定imgIds加载image
        # imgs = [{}, {}, ....], {} = {'id': 1, 'width': 4000, 'height': 3000, 'file_name': 'MVIMG_20201022_095333.jpg',
        #                              'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': 0}
        imgs = coco.loadImgs(imgIds)

        # single json file save dir
        single_json_file_save_dir = os.path.join(self.label_save_dir, 'annotations')
        if not os.path.exists(single_json_file_save_dir):
            os.makedirs(single_json_file_save_dir)
        for idx, img in enumerate(imgs):
            single_json_file = {}
            single_json_file_name_suffix = img['file_name'].split('/')[-1].split('.')[-1]
            single_json_file_name = img['file_name'].split('/')[-1].replace(single_json_file_name_suffix, 'json')
            
            # ---------------------------------------- get "info" ----------------------------------------
            date_created = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%MS')
            self.info["date_created"] = date_created
            single_json_file['info'] = self.info
            
            # ---------------------------------------- get "licenses" ----------------------------------------
            single_json_file['licenses'] = self.licenses
            
            
            
            # ---------------------------------------- get "annotations" ----------------------------------------
            # coco中每个标注实例都对应一个id，如果一张图像中有多个实例，也就有多个ann
            # annIds = [id1, id2, ....]
            annIds = coco.getAnnIds(imgIds=img['id'])

            # 使用给定的annIds加载annotation
            # anns = [{}, {}], {} = {'id': 5, 'image_id': 5, 'category_id': 1, 'segmentation': [], 'area': 109903.41049999993,
            #                        'bbox': [2614.17, 188.47, 368.05, 298.61], 'iscrowd': 0, 'attributes': {'occluded': False}}
            anns = coco.loadAnns(annIds)
            
            nava_anns = []
            linkid_list = []
            seg_link = []
            area = 0
            for i, ann in enumerate(anns):
                nava_ann = {}
                if ann["attributes"]["linkid"] not in linkid_list:
                    if ann["attributes"]["islinked"] == '1':
                        linkid = ann["attributes"]["linkid"]
                        linkid_list.append(linkid)
                        for ann2 in anns:
                            seg = ann2['segmentation']
                            if ann2["attributes"]["linkid"] == linkid:
                                seg_link += seg
                                area += ann2['area']
                    else:
                        area = ann['area']
                        seg = ann['segmentation']
                        seg_link += seg
                else:
                    continue
                nava_ann["segmentation"] = seg_link  
                ann_cat = ann['category_id']
                nava_ann["category_id"] = self.nava_catIds[source_cat_names[ann_cat]]
                nava_ann["area"] = area
                nava_ann["iscrowd"] = int(ann["attributes"]["iscrowd"])
                nava_ann["bbox"] = ann["bbox"]
                nava_ann["id"] = ann["id"]
                nava_anns.append(nava_ann)
                seg_link = []
                area = 0
            single_json_file["annotations"] = nava_anns
            
            # ---------------------------------------- get "categories" ----------------------------------------
            for single_image_categorie in cats:
                single_image_categorie['id'] = self.nava_catIds[single_image_categorie['name']]
                single_image_categorie['supercategory'] = self.nava_supercategory[str(single_image_categorie['id'])]   
            single_json_file["categories"] = cats
             
            # ---------------------------------------- get "images" ---------------------------------------------
            img['file_name'] = img['file_name'].split('/')[-1]
            img["date_captured"] = date_created
            images = [img]  
            single_json_file["images"] = images
            
            # ---------------------------------------- save -----------------------------------------------------
            with open(os.path.join(single_json_file_save_dir, single_json_file_name), "w") as final_f:
                json.dump(single_json_file, final_f)
            final_f.close()
            
            print('Process the ', idx+1, 'th image')
        return name_seg_id

    def generate_train_label(self):
        name_seg_id = self.get_ann()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", default='/root/code/gbicc_data_processing/temp',
                        type=str, help="the dir of josn label")
    parser.add_argument("--ann_name", default='instances_Train.json', type=str, help="the name of josn label")
    parser.add_argument("--label_save_dir", default='/root/code/gbicc_data_processing/temp/jsonfile', type=str,
                        help="the path to save generate label")
    cfg = parser.parse_args()

    gbicc = CocoToSingleJson(cfg.ann_dir, cfg.ann_name, cfg.label_save_dir)
    gbicc.generate_train_label()