import pandas as pd

df1 = pd.read_csv(r'c:\Users\25936\Desktop\Code\chainsight\test_outputs\BC_S5\run_20260119_110937\orchestrator\open_deployment_20261006.csv')
df2 = pd.read_csv(r'c:\Users\25936\Desktop\Code\chainsight\outputs\BC_S5_src\run_20260119_111553\orchestrator\open_deployment_20261006.csv')

# 按key排序
df1_sorted = df1.sort_values(['material', 'sending', 'receiving', 'ori_deployment_uid']).reset_index(drop=True)
df2_sorted = df2.sort_values(['material', 'sending', 'receiving', 'ori_deployment_uid']).reset_index(drop=True)

# 比较deployed_qty
print('deployed_qty比较:')
diff_count = 0
for i in range(len(df1_sorted)):
    v1 = df1_sorted.loc[i, 'deployed_qty']
    v2 = df2_sorted.loc[i, 'deployed_qty']
    if v1 != v2:
        diff_count += 1
        print(f'  差异 行{i}: code_vo={v1}, src={v2}')
        print(f'    material: {df1_sorted.loc[i, "material"]}')
        print(f'    sending: {df1_sorted.loc[i, "sending"]}')
        print(f'    receiving: {df1_sorted.loc[i, "receiving"]}')
        print(f'    ori_deployment_uid: {df1_sorted.loc[i, "ori_deployment_uid"]}')
        print()

print(f'总差异数: {diff_count}')
