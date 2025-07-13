from utils.iotools import read_json

train_annos, test_annos, val_annos = [], [], []
annos = read_json("data_files/ORBench_PRCV/train/text_annos.json")
for anno in annos:
    if anno['split'] == 'train':
        train_annos.append(anno)
    elif anno['split'] == 'test':
        test_annos.append(anno)
    else:
        val_annos.append(anno)
print(len(train_annos),len(test_annos),len(val_annos))
print(len(annos))