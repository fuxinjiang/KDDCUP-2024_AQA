# rrf code 主要用于对不同retriever推荐的结果进行ensemble
import sys
from tqdm import tqdm
from collections import defaultdict

def rrf_score(rank, k=60):
    return 1.0/(k + rank)

def aggregate_results(result_list, top_n=20, save_path=''):
    """
    使用 RRF 汇总和排序多个检索器的结果。
    
    参数:
        result_list (list): 检索结果列表，每个子列表包含不同检索器的结果。
        top_n (int, 可选): 返回的前 top_n 个结果。默认为 20。
        save_path (str, 可选): 保存汇总结果的文件路径。默认为空字符串。
    """
    with open(save_path, 'w') as f:
        for results in tqdm(zip(*result_list), total=len(result_list[0])):
            result_dict = defaultdict(float)
            for retriever_results in results:
                for rank, paper_id in enumerate(retriever_results, start=1):
                    result_dict[paper_id] += rrf_score(rank)
            
            top_results = sorted(result_dict, key=result_dict.get, reverse=True)[:top_n]
            f.write(','.join(top_results) + '\n')

if __name__ == "__main__":

    pass
