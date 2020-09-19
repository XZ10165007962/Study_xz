import pandas as pd
pd.set_option('display.max_columns',None)

train_data = pd.read_excel('E:/jiangxi/deluxe_train_data_0914.xlsx')
#test_data = pd.read_excel(r'E:\江西权益潜客挖掘\deluxe_test_data_0914.xlsx')

print(train_data.describe())
#print(test_data.describe())
