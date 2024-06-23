# gbicc_data_processing

## 1. Instance segmentation
- coco package install  
    ```
    sudo apt-get install pip
    sudo pip install pycocotools==2.0.0
    ```

- convert to single json file
    ```
    cd 01-Segmentation
    python coco_to_single_json.py --ann_dir xxx/json_file_dir --ann_name xxx.json --label_save_dir xxx/single_json_file_save_dir
    参数说明：
    --ann_dir 从标注平台上下载下来的coco 1.0的json文件存储目录
    --ann_name 从标注平台上下载下来的coco 1.0的json文件名称
    --label_save_dir 生成的单个json文件存储目录，会在该目录下生成一个名为annotations的文件夹，里面存储着生成的单个json文件

    举例：
    python coco_to_single_json.py --ann_dir /root/code/gbicc_data_processing/temp --ann_name instances_Train.json --label_save_dir /root/code/gbicc_data_processing/temp
    ```

- `01-Segmentation/configure`文件说明
    - gbicc_catIds.json是根据下载下来的标注文件自动生成的，里面是标注商对类别的定义
    - nava_catIds.json和nava_supercategory.json是客户对类别的定义文件，根据客户需求修改

- show result
    ```
    cd 01-Segmentation
    python coco_to_single_json_show.py --ann_dir 生成的单个json文件存储目录 --visulizaiton_save_dir 可视化结果存储的目录 --image_save_dir 标注图片存储的目录

    举例：
    python coco_to_single_json_show.py --ann_dir /root/code/gbicc_data_processing/temp/annotations --visulizaiton_save_dir /root/code/gbicc_data_processing/temp --image_save_dir /root/code/gbicc_data_processing/temp/try_label_segdata

    最后会在--visulizaiton_save_dir下生成一个visulizaiton文件夹，里面存储着标注可视化图片
    ```
