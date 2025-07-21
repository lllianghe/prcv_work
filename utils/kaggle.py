import json

def get_query_type_idx_range(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    query_type_ranges = []
    current_query_type = data[0]['query_type']
    begin_idx = 0
    for idx, entry in enumerate(data):
        if idx==0: continue
        query_type= entry['query_type']
        if query_type != current_query_type:
            if current_query_type is not None:
                # 如果 query_type 发生变化，记录前一个 query_type 的范围
                query_type_ranges.append((current_query_type, begin_idx, idx - 1))
            # 更新当前的 query_type 和起始索引
            current_query_type = query_type
            begin_idx = idx
        
        # 最后一项，结束时需要加入
        if idx == len(data) - 1:
            query_type_ranges.append((current_query_type, begin_idx, idx))
    
    return query_type_ranges