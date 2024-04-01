import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#計算平均跟標準差
def cal_mean_std(data):
    # Calculate mean and standard deviation
    
    mean1 = np.mean(data, axis=0)
    std1 = np.std(data, axis=0)
    std1[std1 == 0] = 1e-120
    return std1, mean1

#計算機率與分組
def cal_prob(data):

    group1_data = []
    group2_data = []
    group1 = data[data['cluster'] == 1].index
    for i in group1:
        row = data.iloc[i]  # 这里的索引是从 0 开始的，因此选择的是第二行
        group1_data.append(data_check(row[0],row[1],row[2],row[3],row[4],row[5]))

    
        
    group2 = data[data['cluster'] == 2].index
    for i in group2:
        row = data.iloc[i]  # 这里的索引是从 0 开始的，因此选择的是第二行
        group2_data.append(data_check(row[0],row[1],row[2],row[3],row[4],row[5]))

   

    p1 = len(group1_data) / 50
    p2 = len(group2_data) / 50
    
    return p1, p2, group1_data, group2_data

#轉dataframe資料
def dataframe_change(data):
    matclass = pd.DataFrame({
        'd1': [x[0] for x in data],
        'c1': [x[1] for x in data],
        'd2': [x[2] for x in data],
        'c2': [x[3] for x in data],
        'd3': [x[4] for x in data],
        'c3': [x[5] for x in data]
    })

    return matclass

#計算高斯函數
def gaussian_func( data1, data2, mu1, mu2, sigma1, sigma2):
    
    
    data =dataframe_change(data1 + data2)
    
    m1 = mu1
    m2 = mu2
    s1 = sigma1
    s2 = sigma2
    data_new = data.copy()
    
    new_cluster = update_group(data,data_new,m1,m2,s1,s2)
    data['cluster'] = new_cluster
   
    return data

#計算kmean
def update_group(g1, g2, mu1, mu2, sigma1, sigma2):
    
    g1_d1 = b_gau(g1['d1'], mu1[0], sigma1[0])
    g1_c1 = b_gau(g1['c1'], mu1[1], sigma1[1])
    g1_d2 = b_gau(g1['d2'], mu1[2], sigma1[2])
    g1_c2 = b_gau(g1['c2'], mu1[3], sigma1[3])
    g1_d3 = b_gau(g1['d3'], mu1[4], sigma1[4])
    g1_c3 = b_gau(g1['c3'], mu1[5], sigma1[5])
    
    # Calculate Gaussian probabilities for g2
    g2_d1 = b_gau(g2['d1'], mu2[0], sigma2[0])
    g2_c1 = b_gau(g2['c1'], mu2[1], sigma2[1])
    g2_d2 = b_gau(g2['d2'], mu2[2], sigma2[2])
    g2_c2 = b_gau(g2['c2'], mu2[3], sigma2[3])
    g2_d3 = b_gau(g2['d3'], mu2[4], sigma2[4])
    g2_c3 = b_gau(g2['c3'], mu2[5], sigma2[5])

    # Calculate b-value for g1 and g2
    g1_b = (((((g1_d1 * g1_c1) * g1_d2) * g1_c2) * g1_d3) * g1_c3)
    g2_b = (((((g2_d1 * g2_c1) * g2_d2) * g2_c2) * g2_d3) * g2_c3)
  
    new_cluster = []
    for i in range(len(g1_b)):
        if g1_b[i] > g2_b[i]:
            new_cluster.append(1)
        
        if g1_b[i]<g2_b[i]:
            new_cluster.append(2)


        

    
    return new_cluster

#取資料轉陣列
def split_data(data):
    result = []
    for row in data.iterrows():
        result.append(data_check(row['Dim1'], row['Dim2'], row['Dim3'], row['Dim4'], row['Dim5'], row['Dim6']))
    return result

def data_check(Dim1, Dim2, Dim3, Dim4, Dim5, Dim6):
    num = [Dim1, Dim2, Dim3, Dim4, Dim5, Dim6]
    return num


def b_gau(data, mu, sigma):
    for i in range(0,len(data)):
        data[i] = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((data[i] - mu)**2) / (2 * sigma**2))
   
    return data

#計算b_value
def cal_b(g1, g2, mu1, mu2, sigma1, sigma2, c1, c2):
    # Calculate Gaussian probabilities for g1
    g1 = dataframe_change(g1)
    g2 = dataframe_change(g2)

    g1_d1 = b_gau(g1['d1'], mu1[0], sigma1[0])
    g1_c1 = b_gau(g1['c1'], mu1[1], sigma1[1])
    g1_d2 = b_gau(g1['d2'], mu1[2], sigma1[2])
    g1_c2 = b_gau(g1['c2'], mu1[3], sigma1[3])
    g1_d3 = b_gau(g1['d3'], mu1[4], sigma1[4])
    g1_c3 = b_gau(g1['c3'], mu1[5], sigma1[5])
   
    # Calculate Gaussian probabilities for g2
    g2_d1 = b_gau(g2['d1'], mu2[0], sigma2[0])
    g2_c1 = b_gau(g2['c1'], mu2[1], sigma2[1])
    g2_d2 = b_gau(g2['d2'], mu2[2], sigma2[2])
    g2_c2 = b_gau(g2['c2'], mu2[3], sigma2[3])
    g2_d3 = b_gau(g2['d3'], mu2[4], sigma2[4])
    g2_c3 = b_gau(g2['c3'], mu2[5], sigma2[5])
    
    # Calculate b-value for g1 and g2
    g1_b = (((((g1_d1 * g1_c1) * g1_d2) * g1_c2) * g1_d3) * g1_c3)
    g2_b = (((((g2_d1 * g2_c1) * g2_d2) * g2_c2) * g2_d3) * g2_c3)
    # Make lengths of g1_b and g2_b equal by padding with zeros
    max_length = max(len(g1_b), len(g2_b))
   
    if len(g1_b) < max_length:
        g1_b = np.append(g1_b, np.zeros(max_length - len(g1_b)))
        
    
    if len(g2_b) < max_length:
        g2_b = np.append(g2_b, np.zeros(max_length - len(g2_b)))
    
    # Calculate the final b-value
    b_value = (c1 * g1_b) + (c2 * g2_b)
    
    return b_value

file_path = "GMM_data.csv"
data1 = pd.read_csv("class1.csv")
data2 = pd.read_csv("class2.csv")
data3 = pd.read_csv("class3.csv")
data4 = pd.read_csv("class4.csv")
data5 = pd.read_csv("class5.csv")
data6 = pd.read_csv("class6.csv")
data7 = pd.read_csv("class7.csv")
data8 = pd.read_csv("class8.csv")
data9 = pd.read_csv("class9.csv")
data10 = pd.read_csv("class10.csv")
data11 = pd.read_csv("class11.csv")
data12 = pd.read_csv("class12.csv")
data13 = pd.read_csv("class13.csv")
data14 = pd.read_csv("class14.csv")
data15 = pd.read_csv("class15.csv")
data16 = pd.read_csv("class16.csv")

#計算結果
def result_cal(data1):
    new_bvalue = []
    max_iterations = 0  # Set maximum number of iterations
        
    while max_iterations <= 8:
        class1_p1, class1_p2, class1_g1, class1_g2 = cal_prob(data1)

                #class1 = np.vstack((class1_g1, class1_g2))
                
        new_std1, new_mu1 = cal_mean_std(class1_g1)
        new_std2, new_mu2 = cal_mean_std(class1_g2)
        print(new_mu1)
        print(new_std1)
        print(new_mu2)
        print(new_std2)
        
        new_class1 = gaussian_func(class1_g1, class1_g2, new_mu1, new_mu2, new_std1, new_std2)
        data1 = new_class1
       
        b_value = cal_b(class1_g1, class1_g2, new_mu1, new_mu2, new_std1, new_std2, class1_p1, class1_p2)

        b_value = np.concatenate((b_value, np.zeros(50 - len(b_value))))

        new_bvalue.append(b_value)

        #if max_iterations == 10:
        #   break
        
    
        max_iterations += 1



    for i in range(0,len(new_bvalue)):
        new_bvalue[i]= [np.log(np.max(new_bvalue[i]))]
        
   



        
    P_value = []

    for i in range(0, len(new_bvalue)):
        for j in range(0,1):
            P_value.append(new_bvalue[i][j])
         
    return P_value
print(result_cal(data4))          
#fig, axs = plt.subplots(4, 4, figsize=(16, 16))
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
"""
axs[0,0].plot(np.arange(len(result_cal(data1))), result_cal(data1))
axs[0,0].set(xlabel='iteration', ylabel='Likelyhood')

axs[0,1].plot(np.arange(len(result_cal(data2))), result_cal(data2))
axs[0,1].set(xlabel='iteration', ylabel='Likelyhood')


axs[0,2].plot(np.arange(len(result_cal(data3))), result_cal(data3))
axs[0,2].set(xlabel='iteration', ylabel='Likelyhood')
"""
axs.plot(np.arange(len(result_cal(data4))), result_cal(data4))
axs.set(xlabel='iteration', ylabel='Likelyhood')
"""
axs[1,0].plot(np.arange(len(result_cal(data5))), result_cal(data5))
axs[1,0].set(xlabel='iteration', ylabel='Likelyhood')

axs[1, 1].plot(np.arange(len(result_cal(data6))), result_cal(data6))
axs[1, 1].set(xlabel='iteration', ylabel='Likelyhood')

axs[1,2].plot(np.arange(len(result_cal(data7))), result_cal(data7))
axs[1,2].set(xlabel='iteration', ylabel='Likelyhood')

axs[1,3].plot(np.arange(len(result_cal(data8))), result_cal(data8))
axs[1,3].set(xlabel='iteration', ylabel='Likelyhood')

axs[2,0].plot(np.arange(len(result_cal(data9))), result_cal(data9))
axs[2,0].set(xlabel='iteration', ylabel='Likelyhood')

axs[2,1].plot(np.arange(len(result_cal(data10))), result_cal(data10))
axs[2,1].set(xlabel='iteration', ylabel='Likelyhood')

axs[2,2].plot(np.arange(len(result_cal(data11))), result_cal(data11))
axs[2,2].set(xlabel='iteration', ylabel='Likelyhood')

axs[2,3].plot(np.arange(len(result_cal(data12))), result_cal(data12))
axs[2,3].set(xlabel='iteration', ylabel='Likelyhood')

axs[3,0].plot(np.arange(len(result_cal(data13))), result_cal(data13))
axs[3,0].set(xlabel='iteration', ylabel='Likelyhood')

axs[3,1].plot(np.arange(len(result_cal(data14))), result_cal(data14))
axs[3,1].set(xlabel='iteration', ylabel='Likelyhood')

axs[3,2].plot(np.arange(len(result_cal(data15))), result_cal(data15))
axs[3,2].set(xlabel='iteration', ylabel='Likelyhood')

axs[3,3].plot(np.arange(len(result_cal(data16))), result_cal(data16))
axs[3,3].set(xlabel='iteration', ylabel='Likelyhood')
"""
plt.tight_layout()
plt.show()


 





