import os.path as op
from typing import List
import random
import os

from utils.iotools import read_json
from .bases import BaseDataset


class ORBENCH(BaseDataset):
    """
    ORBench
    """
    dataset_dir = 'ORBench_PRCV'

    def __init__(self, root='', test_size=150/400 ,verbose=True):
        super(ORBENCH, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'train/')

        self.anno_path = op.join(self.dataset_dir, 'train/text_annos.json')
        # self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos, id_num = self._split_anno(self.anno_path,test_size=test_size)

        self.train, self.train_id_container = self._process_anno(self.train_annos,True,first_pid=0)
        self.test, self.test_id_container = self._process_anno(self.test_annos,False,first_pid = int(id_num*(1-test_size)))
        self.val, self.val_id_container = self._process_anno(self.val_annos,False,first_pid=id_num)

        if verbose:
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str, test_size):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        # 获取所有唯一的 ID
        all_ids = list(set(anno['id'] for anno in annos))
        # 随机打乱所有的 ID
        random.shuffle(all_ids)
        id_num = len(all_ids)
        train_size = int(len(all_ids)*(1-test_size))
        train_ids = set(all_ids[:train_size])
        test_ids = set(all_ids[train_size:id_num])
        val_ids = set(all_ids[id_num:])  # 可以根据需要调整验证集的分配
        for anno in annos:
            if anno['id'] in train_ids:
                train_annos.append(anno)
            elif anno['id'] in test_ids:
                test_annos.append(anno)
            elif anno['id'] in val_ids:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos, id_num

  
    def _process_anno(self, annos: List[dict], training=False, first_pid = 0):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            pid_map = {}  # 用于映射每个 id 到连续的 pid
            current_pid = 0
            for anno in annos:
                pid = int(anno['id'])  # 获取原始 pid
                if pid not in pid_map:
                    pid_map[pid] = current_pid
                    current_pid += 1
                pid = pid_map[pid]  # 获取连续的 pid
                pid_container.add(pid)
                vis_img_path = os.path.join(self.img_dir, anno['file_path'])
                pid_str = anno['file_path'].split('/')[1] 
                cp_img_dir = os.path.join(self.img_dir, 'cp', pid_str)  # 彩铅图片文件夹路径
                sk_img_dir = os.path.join(self.img_dir, 'sk', pid_str)  # 素描图片文件夹路径
                nir_img_dir = os.path.join(self.img_dir, 'nir', pid_str)  # 红外图片文件夹路径
                cp_path = random.choice(os.listdir(cp_img_dir)) if os.path.exists(cp_img_dir) else ""
                sk_path = random.choice(os.listdir(sk_img_dir)) if os.path.exists(sk_img_dir) else ""
                nir_path = random.choice(os.listdir(nir_img_dir)) if os.path.exists(nir_img_dir) else ""
                cp_path = os.path.join(cp_img_dir, cp_path) if cp_path else ""
                sk_path = os.path.join(sk_img_dir, sk_path) if sk_path else ""
                nir_path = os.path.join(nir_img_dir, nir_path) if nir_path else ""

                # 获取每一行的caption（这里只取一个）
                caption = anno['caption']

                # 将数据添加到dataset中
                dataset.append((pid, image_id, vis_img_path, cp_path, sk_path, nir_path, caption))
                image_id += 1
            
            return dataset, pid_container


        else:  # 验证集
            dataset = {
                "image_pids": [],
                "caption_pids": [],
                "vis_img_paths": [],
                "cp_paths": [],
                "sk_paths": [],
                "nir_paths": [],
                "captions": []
            }

            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            val_pids = sorted(list(set([int(anno['id']) for anno in annos])))

            pid_mapping = {pid: first_pid + idx for idx, pid in enumerate(val_pids)}
            for anno in annos:
                pid = int(anno['id'])
                if pid in pid_mapping:  # 只处理验证集的 pid
                    new_pid = pid_mapping[pid]  # 将原 pid 映射到 350-399 范围内
                    pid_container.add(new_pid)
                    vis_img_path = os.path.join(self.img_dir, anno['file_path'])
                    pid_str = anno['file_path'].split('/')[1]  # 提取路径中的pid部分
                    cp_img_dir = os.path.join(self.img_dir, 'cp', pid_str)  # 彩铅图片文件夹路径
                    sk_img_dir = os.path.join(self.img_dir, 'sk', pid_str)  # 素描图片文件夹路径
                    nir_img_dir = os.path.join(self.img_dir, 'nir', pid_str)  # 红外图片文件夹路径
                    cp_path = random.choice(os.listdir(cp_img_dir)) if os.path.exists(cp_img_dir) else ""
                    sk_path = random.choice(os.listdir(sk_img_dir)) if os.path.exists(sk_img_dir) else ""
                    nir_path = random.choice(os.listdir(nir_img_dir)) if os.path.exists(nir_img_dir) else ""
                    cp_path = os.path.join(cp_img_dir, cp_path) if cp_path else ""
                    sk_path = os.path.join(sk_img_dir, sk_path) if sk_path else ""
                    nir_path = os.path.join(nir_img_dir, nir_path) if nir_path else ""
                    caption = anno['caption']  # 每行只有一个caption（字符串）

                    img_paths.append(vis_img_path)
                    image_pids.append(new_pid)  # 使用映射后的 pid
                    caption_pids.append(new_pid)  # 使用映射后的 pid
                    captions.append(caption)
                    dataset["vis_img_paths"].append(vis_img_path)
                    dataset["cp_paths"].append(cp_path)
                    dataset["sk_paths"].append(sk_path)
                    dataset["nir_paths"].append(nir_path)

            dataset["image_pids"] = image_pids
            dataset["caption_pids"] = caption_pids
            dataset["captions"] = captions

            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))


