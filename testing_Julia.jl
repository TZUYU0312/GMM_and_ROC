using Clustering, Plots,Distances
using CSV
using LinearAlgebra
using DataFrames
using Random

data1 = CSV.read("class1_remain.csv", DataFrame)
data2 = CSV.read("class2_remain.csv", DataFrame)
data3 = CSV.read("class3_remain.csv", DataFrame)
data4 = CSV.read("class4_remain.csv", DataFrame)
data5 = CSV.read("class5_remain.csv", DataFrame)
data6 = CSV.read("class6_remain.csv", DataFrame)
data7 = CSV.read("class7_remain.csv", DataFrame)
data8 = CSV.read("class8_remain.csv", DataFrame)
data9 = CSV.read("class9_remain.csv", DataFrame)
data10 = CSV.read("class10_remain.csv", DataFrame)
data11 = CSV.read("class11_remain.csv", DataFrame)
data12 = CSV.read("class12_remain.csv", DataFrame)
data13 = CSV.read("class13_remain.csv", DataFrame)
data14 = CSV.read("class14_remain.csv", DataFrame)
data15 = CSV.read("class15_remain.csv", DataFrame)
data16 = CSV.read("class16_remain.csv", DataFrame)
#先跳過
#lambda_data = CSV.read("")
std1_w1=[1.0e-10, 0.17624175234970793, 1.0e-10, 0.09845899565359698, 1.0e-10, 0.10448018217939968]
mean1_w1=[1.0e-10, 0.15000960015140946, 1.0e-10, 0.704699861777352, 1.0e-10, 0.016748982993515484]
std2_w1=[1.0e-10, 0.1859780735695148, 1.0e-10, 0.09018984830405535, 1.0e-10, 0.3291923234690998]
mean2_w1=[1.0e-10, 0.2879308261154027, 1.0e-10, 0.7283210904172436, 1.0e-10, 0.41523313951666885]
w1_b=66.16143230929652

mean1_w2=[0.0003002386775015857, 0.11642995260454227, 0.000221231348980533, 0.030530239580821354, 1.0e-10, 1.0e-10]
std1_w2=[2.7020918516949386e-6, 0.2432029496885184, 1.2951415798251294e-6, 0.1414684249858152, 1.0e-10, 1.0e-10]
mean2_w2=[0.0002954092812236708, 0.18644847708774823, 0.00021639932765212444, 0.41050074719026086, 1.0e-10, 1.0e-10]
std2_w2=[2.1028422307699e-6, 0.24173170860587914, 1.422769781780434e-7, 0.29257827001637265, 1.0e-10, 1.0e-10]
w2_b=87.12147717499859
#算機率與分組
function cal_prob(data)   
    
    group1_data = filter(row -> row[:cluster] == 1, data)
    
      
   
    group2_data = filter(row -> row[:cluster] == 2, data)

    
    return group1_data,group2_data
end
function b_Gau(data,mu,sigma)
    for i in 1: size(data,1)
        data[i] = 1 / sqrt(2 * π * sigma^2) * exp(-( data[i]- mu)^2 / (2 * sigma^2))
    end
    return data
end

#計算B value
function cal_b(g1,g2,mu1,mu2,sigma1,sigma2)
    g1.d1 = b_Gau(g1.d1,mu1[1],sigma1[1])
    g1.c1 = b_Gau(g1.c1,mu1[2],sigma1[2])
    g1.d2 = b_Gau(g1.d2,mu1[3],sigma1[3])
    g1.c2 = b_Gau(g1.c2,mu1[4],sigma1[4])
    g1.d3 = b_Gau(g1.d3,mu1[5],sigma1[5])
    g1.c3 = b_Gau(g1.c3,mu1[6],sigma1[6])

    g2.d1 = b_Gau(g2.d1,mu2[1],sigma2[1])
    g2.c1 = b_Gau(g2.c1,mu2[2],sigma2[2])
    g2.d2 = b_Gau(g2.d2,mu2[3],sigma2[3])
    g2.c2 = b_Gau(g2.c2,mu2[4],sigma2[4])
    g2.d3 = b_Gau(g2.d3,mu2[5],sigma2[5])
    g2.c3 = b_Gau(g2.c3,mu2[6],sigma2[6])
    g_data = vcat(g1,g2)
    return g_data
end

function ROC_test(data,delta)
    global p_count = 0
    global n_count = 0
    for row in eachrow(data)
        for i in 1:6
            if row[i] >= delta
                p_count += 1
            else
                n_count += 1
            end
        end
    end
    return p_count,n_count
end

#先用w2跟w3的資料
class1_g1, class1_g2 = cal_prob(data6[:, 1:7])
Gau_data1 = cal_b(class1_g1,class1_g2,mean1_w1,mean2_w1,std1_w1,std2_w1)

class2_g1, class2_g2 = cal_prob(data12[:, 1:7])
Gau_data2 = cal_b(class2_g1,class2_g2,mean1_w2,mean2_w2,std1_w2,std2_w2)

PN_data = vcat(Gau_data1,Gau_data2)

#P = w1 N = w2 delta=0.7  data>0.7 = p  data<0.7 =N
global TPR = []
global FPR = []
for delta in 0.75:0.001:10
    P_count_P,N_count_P = ROC_test(Gau_data1[:,1:6],delta)
    P_count_N,N_count_N = ROC_test(Gau_data2[:,1:6],delta)
    push!(TPR,P_count_P)
    push!(FPR,P_count_N)
    #println(P_count_P/P_count_N)
end

plot!()
scatter!(FPR, TPR, label="ROC")
