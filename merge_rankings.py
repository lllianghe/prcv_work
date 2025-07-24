import pandas as pd
import argparse
import os
from collections import defaultdict
import ast

def merge_rankings(file1, file2, output_file):
    """
    根据用户指定的新格式，合并两个ReID排名列表CSV文件。
    新的格式假定：
    - 第一列是查询ID。
    - 第三列是名为 'ranking_list_idx' 的列，其内容为字符串形式的排名列表。

    参数:
    file1 (str): 第一个输入文件的路径。
    file2 (str): 第二个输入文件的路径。
    output_file (str): 合并后的输出文件路径。
    """
    print(f"开始合并文件: {file1} 和 {file2}")
    
    # 读取CSV文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    print("文件读取成功。")
    print(f"文件1 '{os.path.basename(file1)}' 的维度: {df1.shape}")
    print(f"文件2 '{os.path.basename(file2)}' 的维度: {df2.shape}")

    # 检查文件是否为空
    if df1.empty or df2.empty:
        print("错误: 一个或两个输入文件为空。")
        return

    # 检查关键列是否存在
    if len(df1.columns) < 3 or len(df2.columns) < 3:
        print(f"错误: 文件列数不足3，无法找到排名列表。预期的列结构：查询ID, query_type, ranking_list_idx。")
        return

    query_col = df1.columns[0]
    ranking_col = df1.columns[2]
    print(f"将使用第一列 '{query_col}' 作为查询ID，第三列 '{ranking_col}' 作为排名列表。")

    # 检查两个文件中的查询ID顺序是否完全一致
    if not df1[query_col].equals(df2[query_col]):
        print("警告: 两个文件中的查询ID顺序不一致。将基于查询ID进行合并，这可能会稍慢一些。")
        df2 = df2.set_index(query_col).loc[df1[query_col]].reset_index()
        print("已根据文件1的查询ID顺序重新排列文件2。")

    merged_results = []
    
    # 逐行处理查询
    print("开始逐行合并排名...")
    for index, row1 in df1.iterrows():
        row2 = df2.iloc[index]
        query_id = row1[query_col]
        
        try:
            # 解析字符串格式的列表
            ranked_list1 = ast.literal_eval(row1[ranking_col])
            ranked_list2 = ast.literal_eval(row2[ranking_col])
        except (ValueError, SyntaxError) as e:
            print(f"错误: 在处理查询ID {query_id} 时解析排名列表失败: {e}")
            continue

        # 从解析后的列表中创建 gallery_id -> rank 的字典
        ranks1 = {gallery_id: rank + 1 for rank, gallery_id in enumerate(ranked_list1)}
        ranks2 = {gallery_id: rank + 1 for rank, gallery_id in enumerate(ranked_list2)}
        
        # 合并排名
        combined_ranks = defaultdict(int)
        all_gallery_ids = set(ranks1.keys()) | set(ranks2.keys())

        for gid in all_gallery_ids:
            rank1 = ranks1.get(gid, 101) # 如果ID不在排名中，给一个较大的排名值
            rank2 = ranks2.get(gid, 101)
            combined_ranks[gid] = rank1 + rank2
            
        # 根据合并后的排名总和进行排序
        sorted_ranks = sorted(combined_ranks.items(), key=lambda item: item[1])
        
        # 提取排序后的前100个gallery ID
        top_100 = [gid for gid, rank in sorted_ranks[:100]]
        
        # 准备写入新文件的行数据（与输入格式一致）
        query_type_col = df1.columns[1]
        query_type = row1[query_type_col]
        new_row = {
            query_col: query_id,
            query_type_col: query_type,
            ranking_col: str(top_100) # 将列表转换为字符串
        }
        merged_results.append(new_row)

    if not merged_results:
        print("没有生成任何合并结果。")
        return

    print("排名合并完成，正在生成与输入格式一致的输出文件...")
    # 创建新的DataFrame
    final_df = pd.DataFrame(merged_results)
    
    # 确保列顺序与输入文件一致
    if not final_df.empty:
        final_df = final_df[[query_col, df1.columns[1], ranking_col]]
    
    # 保存到CSV
    final_df.to_csv(output_file, index=False)
    print(f"合并后的排名列表已保存到: {output_file}")


if __name__ == "__main__":
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="合并两个ReID排名列表文件。")
    
    # 定义--file1参数，并设置默认值
    parser.add_argument("--file1", type=str, default="/SSD_Data01/myf/research/PRCV/fgclip_model/prcv_work_branch/logs/ORBench/20250723_181814_irra/ranking_list.csv", help="第一个 ranking_list.csv 文件的路径")
    
    # 定义--file2参数，并设置默认值
    parser.add_argument("--file2", type=str, default="/SSD_Data01/myf/research/PRCV/fgclip_model/prcv_work_branch/logs/ORBench/20250723_181204_irra/ranking_list.csv", help="第二个 ranking_list.csv 文件的路径")
    
    # 定义--output参数，并设置默认值
    parser.add_argument("--output", type=str, default="merged_ranking_list.csv", help="输出合并后的文件路径")
    
    # 解析命令行传入的参数
    args = parser.parse_args()
    
    # --- 路径处理 ---
    # 获取脚本文件所在的绝对目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 如果提供的文件路径是相对路径，则将其转换为基于脚本目录的绝对路径
    file1_path = args.file1 if os.path.isabs(args.file1) else os.path.join(script_dir, args.file1)
    file2_path = args.file2 if os.path.isabs(args.file2) else os.path.join(script_dir, args.file2)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)

    # --- 执行合并 ---
    print("--- 开始执行合并脚本 ---")
    print(f"文件1: {file1_path}")
    print(f"文件2: {file2_path}")
    print(f"输出文件: {output_path}")
    
    merge_rankings(file1_path, file2_path, output_path)
    
    print("--- 脚本执行完毕 ---")