import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def process_csv(input_path, test_ratio, output_split_path, output_test_path):
    # 1. 读取原始 CSV
    df = pd.read_csv(input_path)
    
    # 检查必要的列是否存在
    required_cols = ['rgb_path', 'xyz_path', 'fps', 'height', 'width', 'caption']
    for col in required_cols:
        if col not in df.columns:
            print(f"警告: 原始 CSV 中缺少列 '{col}'")

    # 2. 划分训练集和测试集
    # 使用 shuffle=True 确保数据被打乱，random_state 保证结果可复现
    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42, shuffle=True)

    # 3. 创建带有 'split' 列的新 DataFrame
    # 先在各自的副本上打标签
    train_df_with_label = train_df.copy()
    train_df_with_label['split'] = 'training'
    
    test_df_with_label = test_df.copy()
    test_df_with_label['split'] = 'test'
    
    # 合并后保存到 output_split_path
    full_df_with_split = pd.concat([train_df_with_label, test_df_with_label])
    full_df_with_split.to_csv(output_split_path, index=False)
    print(f"已保存包含 split 列的完整数据至: {output_split_path}")

    # 4. 仅保存测试集内容（不含 split 列）到 output_test_path
    # test_df 本身就不包含 split 列
    test_df.to_csv(output_test_path, index=False)
    print(f"已保存测试集数据至: {output_test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据比例拆分 CSV 数据集并打标签。")
    
    parser.add_argument("--input", type=str, required=True, help="原始 CSV 文件路径")
    parser.add_argument("--ratio", type=float, default=0.2, help="测试集所占比例 (例如 0.2)")
    parser.add_argument("--out_split", type=str, required=True, help="输出带有 split 列的 CSV 路径")
    parser.add_argument("--out_test", type=str, required=True, help="输出仅包含测试集的 CSV 路径")

    args = parser.parse_args()

    process_csv(args.input, args.ratio, args.out_split, args.out_test)