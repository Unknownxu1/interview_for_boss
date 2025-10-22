# 浏览列表（list）
# 查看职位（detail）
# 立即开聊消息（addfriend）
# 收到回复（replay）
# 面试邀请（interview）
# 现有一份用户行为日志数据表（job_seeker_behavior），字段如下：
# user_id：求职者唯一标识
# behavior：行为类型（仅包含上述 5 种）
# timestamp：行为发生时间（毫秒级时间戳）
# position_id：职位 ID
# 一份用户基本信息数据表
# geek_info ：city（所在城市）、work_exp（工作年限，如 0-1 年 / 1-3 年 / 3-5 年）、edu（学历，如大专 / 本科 / 硕士）
# 任务：
# 计算分职位的漏斗转化率：按position_id分组，计算每个职位的转化率（注：需基于 “完成前序行为的用户” 计算，且用户行为必须按时间顺序发生；login 是所有行为的前置条件，即后续行为必须在 login 之后）。
# 对用户进行聚类，总结每个用户群的核心特征

import pandas as pd

df1 = pd.read_csv('job_seeker_behavior.csv')
behavior = ['list','detail','addfriend','replay','interview']

positions = df1['position_id'].unique()
results = []
# 每个职位每个动作的转化率
for position in positions:
    df_position = df1[df1['position_id'] == position]
    df_position = df_position.sort_values(['user_id', 'timestamp'])
    
    # 存储截止到每个阶段，满足行为序列顺序的用户
    # 因为behavior里没有login，假设每个记录之前都login了
    list_users = set()
    detail_users = set()
    addfriend_users = set()
    replay_users = set()
    interview_users = set()
    
    # 按用户分组验证行为序列
    for user_id, user_data in df_position.groupby('user_id'):
        user_behaviors = user_data['behavior'].tolist()
        
        # 检查用户是否按顺序完成了行为（中间可以有其他行为）
        current_step_index = -1  
        actuaL_behavior = []
        
        for b in user_behaviors:
            # b是实际行为
            b_index = behavior.index(b)
                
            # 如果b是下一步行为，更新当前index
            if b_index == current_step_index+1:
                current_step_index = b_index
                actuaL_behavior.append(b)
    
        # 把用户分配到对应的满足行为集合当中
        if 'list' in actuaL_behavior:
            list_users.add(user_id)
        if 'detail' in actuaL_behavior:
            detail_users.add(user_id)
        if 'addfriend' in actuaL_behavior:
            addfriend_users.add(user_id)
        if 'replay' in actuaL_behavior:
            replay_users.add(user_id)
        if 'interview' in actuaL_behavior:
            interview_users.add(user_id)
    
    # 计算转化率
    list_count = len(list_users)
    detail_count = len(detail_users)
    addfriend_count = len(addfriend_users)
    replay_count = len(replay_users)
    interview_count = len(interview_users)
    
    # 计算转化率
    if list_count > 0:
        list_to_detail_rate = detail_count / list_count
    else:
        list_to_detail_rate = 0
    
    if detail_count > 0:
        detail_to_addfriend_rate = addfriend_count / detail_count
    else:
        detail_to_addfriend_rate = 0
    
    if addfriend_count > 0:
        addfriend_to_replay_rate = replay_count / addfriend_count
    else:
        addfriend_to_replay_rate = 0
    
    if replay_count > 0:
        replay_to_interview_rate = interview_count / replay_count
    else:
        replay_to_interview_rate = 0
    
    # results记录每个职位对应的结果
    results.append([
        position,
        list_to_detail_rate, detail_to_addfriend_rate, addfriend_to_replay_rate, replay_to_interview_rate
    ])
print(results)

# 对用户进行聚类，总结每个用户群的核心特征
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df2 = pd.read_csv('geek_info.csv')
# df1是job_seeker_behavior

# 对'work_exp','edu','city'进行编码
df2[['work_exp_encode','edu_encode','city_encode']] = OrdinalEncoder().fit_transform(df2[['work_exp','edu''city']])

all_users = df1['user_id'].unique()

user_info = []
for user_id in all_users:
    user_data = df1[df1['user_id'] == user_id]
    
    # 按时间排序
    user_data = user_data.sort_values('timestamp')
    
    # 计算用户行为特征
    total_num = len(user_data)
    
    #对于单个用户，直接看整体行为比例，漏斗会丢失部分信息
    list_count = len(user_data[user_data['behavior'] == 'list'])
    detail_count = len(user_data[user_data['behavior'] == 'detail'])
    addfriend_count = len(user_data[user_data['behavior'] == 'addfriend'])
    replay_count = len(user_data[user_data['behavior'] == 'replay'])
    interview_count = len(user_data[user_data['behavior'] == 'interview'])

    list_ratio = list_count / total_num if total_num > 0 else 0
    detail_ratio = detail_count / total_num if total_num > 0 else 0
    addfriend_ratio = addfriend_count / total_num if total_num > 0 else 0
    replay_ratio = replay_count / total_num if total_num > 0 else 0
    interview_ratio = interview_count / total_num if total_num > 0 else 0

    active_days = len(user_data['timestamp'].dt.date.unique()) # 活跃天数
    work_exp_encode = df2.loc[df2['user_id'] == user_id, 'work_exp_encode'].values[0]
    edu_encode = df2.loc[df2['user_id'] == user_id, 'edu_encode'].values[0]
    city_encode = df2.loc[df2['user_id'] == user_id, 'city_encode'].values[0]
    user_info.append([user_id,list_ratio, detail_ratio, addfriend_ratio, replay_ratio, interview_ratio,active_days,work_exp_encode,edu_encode,city_encode])

user_columns = ['user_id','list_ratio', 'detail_ratio', 'addfriend_ratio', 'replay_ratio', 'interview_ratio','active_days','work_exp_encode','edu_encode','city_encode']
features_df = pd.DataFrame(user_info, columns=user_columns)

# 采用kmeans进行聚类
best_score = -1
best_k = 2

for k in range(2, 12):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_df)
    score = silhouette_score(features_df, cluster_labels)
    
    if score > best_score:
        best_score = score
        best_k = k

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
features_df['cluster'] = kmeans.fit_predict(features_df)

work_exp_mapping = dict(enumerate(df2['work_exp'].unique()))
edu_mapping = dict(enumerate(df2['edu'].unique()))
city_mapping = dict(enumerate(df2['city'].unique()))

# 分析每个cluster的特征
for cluster_id in range(best_k):
    cluster_data = features_df[features_df['cluster'] == cluster_id]
    
    print(f"\nCluster {cluster_id} (共 {len(cluster_data)} 用户):")
    
    list_ratio_mean = cluster_data['list_ratio'].mean()
    detail_ratio_mean = cluster_data['detail_ratio'].mean()
    addfriend_ratio_mean = cluster_data['addfriend_ratio'].mean()
    replay_ratio_mean = cluster_data['replay_ratio'].mean()
    interview_ratio_mean = cluster_data['interview_ratio'].mean()
    active_days_mean = cluster_data['active_days'].mean()
    
    # 计算分类特征的众数
    work_exp_mode = cluster_data['work_exp_encode'].mode().iloc[0] if len(cluster_data['work_exp_encode'].mode()) > 0 else 0
    edu_mode = cluster_data['edu_encode'].mode().iloc[0] if len(cluster_data['edu_encode'].mode()) > 0 else 0
    city_mode = cluster_data['city_encode'].mode().iloc[0] if len(cluster_data['city_encode'].mode()) > 0 else 0
    
    # 将编码映射回原始标签
    work_exp_label = work_exp_mapping.get(work_exp_mode)
    edu_label = edu_mapping.get(edu_mode)
    city_label = city_mapping.get(city_mode)
    
    # 对每一类用户进行数据统计
    features = []
    # 每个簇内的用户的行为特征
    if list_ratio_mean > features_df['list_ratio'].mean():
        features.append("高频浏览")
    
    if detail_ratio_mean > features_df['detail_ratio'].mean():
        features.append("频繁查看职位")
    
    if addfriend_ratio_mean > features_df['addfriend_ratio'].mean():
        features.append("积极开聊天")
    
    if replay_ratio_mean > features_df['replay_ratio'].mean():
        features.append("高回复")
    
    if interview_ratio_mean > features_df['interview_ratio'].mean():
        features.append("高面邀")
    
    # 根据活跃天数判断用户粘性
    if active_days_mean > features_df['active_days'].mean():
        features.append("长期活跃")
    else:
        features.append("短期活跃")
    
    # 根据工作年限和学历判断用户背景
    if work_exp_mode > features_df['work_exp_encode'].mean():
        features.append("工作经验较多")
    else:
        features.append("缺乏工作经验")
    
    if edu_mode > features_df['edu_encode'].mean():
        features.append("学历较高")
    else:
        features.append("普通学历")
    
    print(f"  核心特征: {', '.join(features)}")
    print(f"  基本信息: 典型工作年限: {work_exp_label}, 典型学历: {edu_label}, 典型城市: {city_label}")
    print(f"  行为特征:")
    print(f"  平均活跃天数: {active_days_mean:.1f}")
    print(f"  行为比例 - 浏览: {list_ratio_mean:.4f}, 查看详情: {detail_ratio_mean:.4f},行为比例 - 开聊: {addfriend_ratio_mean:.4f}, 回复: {replay_ratio_mean:.4f}, 面试: {interview_ratio_mean:.4f}")


