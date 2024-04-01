import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_check(Dim1, Dim2, Dim3, Dim4, Dim5, Dim6):
    num = [Dim1, Dim2, Dim3, Dim4, Dim5, Dim6]
    return num

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

    
    return group1_data, group2_data

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
def b_gau(data, mu, sigma):
    for i in range(0,len(data)):
        data[i] = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((data[i] - mu)**2) / (2 * sigma**2))
   
    return data
#計算b_value
def cal_b(g1, g2, mu1, mu2, sigma1, sigma2):
    # Calculate Gaussian probabilities for g1
    g1 = dataframe_change(g1)
    g2 = dataframe_change(g2)

    g1['d1'] = b_gau(g1['d1'], mu1[0], sigma1[0])
    g1['c1'] = b_gau(g1['c1'], mu1[1], sigma1[1])
    g1['d2'] = b_gau(g1['d2'], mu1[2], sigma1[2])
    g1['c2'] = b_gau(g1['c2'], mu1[3], sigma1[3])
    g1['d3'] = b_gau(g1['d3'], mu1[4], sigma1[4])
    g1['c3'] = b_gau(g1['c3'], mu1[5], sigma1[5])
   
    # Calculate Gaussian probabilities for g2
    g2['d1'] = b_gau(g2['d1'], mu2[0], sigma2[0])
    g2['c1'] = b_gau(g2['c1'], mu2[1], sigma2[1])
    g2['d2'] = b_gau(g2['d2'], mu2[2], sigma2[2])
    g2['c2'] = b_gau(g2['c2'], mu2[3], sigma2[3])
    g2['d3'] = b_gau(g2['d3'], mu2[4], sigma2[4])
    g2['c3'] = b_gau(g2['c3'], mu2[5], sigma2[5])
    g = pd.concat([g1, g2], ignore_index=True)
    return g

def ROC_test(data,delta):
    p_count = 0
    n_count = 0
    for _,row in data.iterrows():
        for i in range(0,6):
            if row[i] >= delta:
                p_count += 1
            else:
                n_count += 1
    
    return p_count,n_count

data1 = pd.read_csv("class1_remain.csv")
data2 = pd.read_csv("class2_remain.csv")
data3 = pd.read_csv("class3_remain.csv")
data4 = pd.read_csv("class4_remain.csv")
data5 = pd.read_csv("class5_remain.csv")
data6 = pd.read_csv("class6_remain.csv")
data7 = pd.read_csv("class7_remain.csv")
data8 = pd.read_csv("class8_remain.csv")
data9 = pd.read_csv("class9_remain.csv")
data10 = pd.read_csv("class10_remain.csv")
data11 = pd.read_csv("class11_remain.csv")
data12 = pd.read_csv("class12_remain.csv")
data13 = pd.read_csv("class13_remain.csv")
data14 = pd.read_csv("class14_remain.csv")
data15 = pd.read_csv("class15_remain.csv")
data16 = pd.read_csv("class16_remain.csv")
    
mean1_w1=[0.03161989, 1.50204894, 0.04344565, 2.58000926, 0.09925531, 2.65418923]
std1_w1=[0.01174088, 0.56230394, 0.01565543, 0.96827543, 0.04020455, 0.90631517]
mean2_w1=[0.01654705, 1.17048071, 0.03346184, 0.73459924, 0.01135217, 1.64054629]
std2_w1=[1.64239031e-02, 7.98536879e-01, 2.07385368e-02, 9.84131754e-01, 1.20772874e-17, 1.49063979e+00]
w1_b=38.49572266964815

mean1_w2=[8.06002687e+000, 2.86678120e+000, 5.42146742e+001, 2.03687280e+000, 3.98942280e+119, 1.93661272e+000]
std1_w2=[2.60152433e+000, 1.11786324e+000, 2.43846318e+001, 7.39486035e-001, 4.30030990e+104, 7.38836045e-001]
mean2_w2=[10.17011084,  2.95947188, 51.36897693,  2.28120296,  0,  2.25629523]
std2_w2=[1.11332905e+000, 4.17337968e-001, 3.08798632e+001, 2.94480442e-001, 1.00000000e-120, 2.55197643e-001]
w2_b=285.36408361669476

class1_g1, class1_g2 = cal_prob(data1)
Gau_data1 = cal_b(class1_g1, class1_g2,mean1_w1,mean2_w1,std1_w1,std2_w1)

class2_g1, class2_g2 = cal_prob(data3)
Gau_data2 = cal_b(class2_g1, class2_g2,mean1_w2,mean2_w2,std1_w2,std2_w2)

PN_data = pd.concat([Gau_data1,Gau_data2],ignore_index=True)
TPR = []
FPR = []
for delta in np.arange(0.13,10.1,0.01):
    P_count_P,N_count_P = ROC_test(Gau_data1.iloc[:,0:6],delta)
    P_count_N,N_count_N = ROC_test(Gau_data2.iloc[:,0:6],delta)
    TPR.append(P_count_P)
    FPR.append(N_count_P)


plt.scatter(FPR, TPR, label="ROC")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()


