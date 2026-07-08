import pandas as pd

# 读取所有文件
still_table = pd.read_csv("still_table.csv")
still_hand = pd.read_csv("still_hand.csv")
walk_hand = pd.read_csv("walk_hand.csv")
walk_pocket = pd.read_csv("walk_pocket.csv")

# ---- 加入旧数据 ----
old_still = pd.read_csv("still.csv")      # 旧静止数据
old_walk = pd.read_csv("walk.csv")        # 旧行走数据

# 打标签
still_table['activity'] = 'still'
still_hand['activity'] = 'still'
walk_hand['activity'] = 'walk'
walk_pocket['activity'] = 'walk'
old_still['activity'] = 'still'
old_walk['activity'] = 'walk'

# 合并
merged_df = pd.concat([still_table, still_hand, walk_hand, walk_pocket, old_still, old_walk], ignore_index=True)

# 整理列名
merged_df = merged_df[['Time (s)', 'Linear Acceleration x (m/s^2)',
                       'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)',
                       'activity']]
merged_df.columns = ['time', 'acc_x', 'acc_y', 'acc_z', 'activity']

# 保存
merged_df.to_csv('sensor_data.csv', index=False)
print(f"合并完成！总共 {len(merged_df)} 行数据")
print(f"各类别数量：\n{merged_df['activity'].value_counts()}")