from utils.iotools import read_json
from collections import Counter

train_annos, test_annos, val_annos = [], [], []
annos = read_json("data_files/ORBench_PRCV/val/val_queries.json")
query_type_counts = Counter(anno['query_type'] for anno in annos)

for query_type, count in query_type_counts.items():
    print(f"query_type: {query_type}, count: {count}")
print("total: ",len(annos))