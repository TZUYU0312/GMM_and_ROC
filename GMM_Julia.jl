using Clustering, Plots,Distances
using CSV,Statistics
using LinearAlgebra
using DataFrames
using Random,PrettyTables

function cal_mean_std(data)
    #算平均與標準差
    mean1 = [mean(data[!, col]) for col in names(data)]
    std1 = [std(data[!, col]) for col in names(data)]
    std1[std1 .== 0] .= 1e-10
    mean1[mean1 .== 0] .= 1e-10
    return std1,mean1
end

#算機率與分組
function cal_prob(data)   
    group1 = count(row -> row[:cluster] == 1, eachrow(data))
    group1_data = filter(row -> row[:cluster] == 1, data)
    
    p1 = group1/50  
    group2 = count(row -> row[:cluster] == 2, eachrow(data))
    group2_data = filter(row -> row[:cluster] == 2, data)

    p2 = group2/50
    return p1,p2,group1_data,group2_data
end

#計算Gaussian函數
function Gaussian_func(data1,data2,mu1,mu2,sigma1,sigma2)
       
    
    data = vcat(data1,data2)
    
   
    new_cluster = group_update(data,data,mu1,mu2,sigma1,sigma2)
    
    data[!, :cluster] = new_cluster
    
    
    return data
end

#將平均值與標準差帶入資料集並選機率最大的重新分組
function group_update(data1,data2,mu1,mu2,sigma1,sigma2)
    data1.d1 = b_Gau(data1.d1,mu1[1],sigma1[1])
    data1.c1 = b_Gau(data1.c1,mu1[2],sigma1[2])
    data1.d2 = b_Gau(data1.d2,mu1[3],sigma1[3])
    data1.c2 = b_Gau(data1.c2,mu1[4],sigma1[4])
    data1.d3 = b_Gau(data1.d3,mu1[5],sigma1[5])
    data1.c3 = b_Gau(data1.c3,mu1[6],sigma1[6])
    g1_b =((((data1.d1 .* data1.c1) .*data1.d2) .* data1.c2) .* data1.d3) .* data1.c3
    data2.d1 = b_Gau(data2.d1,mu2[1],sigma2[1])
    data2.c1 = b_Gau(data2.c1,mu2[2],sigma2[2])
    data2.d2 = b_Gau(data2.d2,mu2[3],sigma2[3])
    data2.c2 = b_Gau(data2.c2,mu2[4],sigma2[4])
    data2.d3 = b_Gau(data2.d3,mu2[5],sigma2[5])
    data2.c3 = b_Gau(data2.c3,mu2[6],sigma2[6])
    g2_b =((((data2.d1 .* data2.c1) .*data2.d2) .* data2.c2) .* data2.d3) .* data2.c3
    
    
  
    new_cluster = [ifelse(g1_b[i] > g2_b[i], 1, 2) for i in 1:length(g1_b)]
    
    
    
    return new_cluster




end

#取資料
function split_Data(data)
    global result = []
    for row in eachrow(data)
        result =push!(result, data_check(row[1],row[2],row[3],row[4],row[5],row[6]))
    end
    return result
end
function data_check(Dim1,Dim2,Dim3,Dim4,Dim5,Dim6) #Dim1[:,1]
    num = [Dim1,Dim2,Dim3,Dim4,Dim5,Dim6] 
    return num
end
function b_Gau(data,mu,sigma)
    for i in 1: size(data,1)
        data[i] = 1 / sqrt(2 * π * sigma^2) * exp(-( data[i]- mu)^2 / (2 * sigma^2))
    end
    return data
end

#計算B value
function cal_b(g1,g2,mu1,mu2,sigma1,sigma2,c1,c2)
    g1.d1 = b_Gau(g1.d1,mu1[1],sigma1[1])
    g1.c1 = b_Gau(g1.c1,mu1[2],sigma1[2])
    g1.d2 = b_Gau(g1.d2,mu1[3],sigma1[3])
    g1.c2 = b_Gau(g1.c2,mu1[4],sigma1[4])
    g1.d3 = b_Gau(g1.d3,mu1[5],sigma1[5])
    g1.c3 = b_Gau(g1.c3,mu1[6],sigma1[6])
    g1_b =((((g1.d1 .* g1.c1) .*g1.d2) .* g1.c2) .* g1.d3) .* g1.c3
    g2.d1 = b_Gau(g2.d1,mu2[1],sigma2[1])
    g2.c1 = b_Gau(g2.c1,mu2[2],sigma2[2])
    g2.d2 = b_Gau(g2.d2,mu2[3],sigma2[3])
    g2.c2 = b_Gau(g2.c2,mu2[4],sigma2[4])
    g2.d3 = b_Gau(g2.d3,mu2[5],sigma2[5])
    g2.c3 = b_Gau(g2.c3,mu2[6],sigma2[6])
    g2_b =((((g2.d1 .* g2.c1) .*g2.d2) .* g2.c2) .* g2.d3) .* g2.c3
    max_length = max(length(g1_b),length(g2_b))
    if max_length != length(g1_b)
        g1_b = vcat(g1_b, fill(0, max_length - length(g1_b)))
    end
    if max_length != length(g2_b)
        g2_b = vcat(g2_b, fill(0, max_length - length(g2_b)))
    end        
    b_value = ((c1 * g1_b) .+ (c2 * g2_b))
    
    return b_value
end   

#轉dataframe
function dataframe_change(data)
    matclass = DataFrame(d1 = map(x -> x[1],data),c1 = map(x ->x[2],data),
                        d2 = map(x -> x[3],data),c2 = map(x -> x[4],data),
                        d3 = map(x->x[5],data),c3 = map(x -> x[6],data))
    
    return matclass
end
file_path = "GMM_data.csv"
data1 = CSV.read("class1.csv", DataFrame)
data2 = CSV.read("class2.csv", DataFrame)
data3 = CSV.read("class3.csv", DataFrame)
data4 = CSV.read("class4.csv", DataFrame)
data5 = CSV.read("class5.csv", DataFrame)
data6 = CSV.read("class6.csv", DataFrame)
data7 = CSV.read("class7.csv", DataFrame)
data8 = CSV.read("class8.csv", DataFrame)
data9 = CSV.read("class9.csv", DataFrame)
data10 = CSV.read("class10.csv", DataFrame)
data11 = CSV.read("class11.csv", DataFrame)
data12 = CSV.read("class12.csv", DataFrame)
data13 = CSV.read("class13.csv", DataFrame)
data14 = CSV.read("class14.csv", DataFrame)
data15 = CSV.read("class15.csv", DataFrame)
data16 = CSV.read("class16.csv", DataFrame)
df = CSV.read(file_path, DataFrame)
#data = df[:,2:7]
function result_cal(data1)
    
    global new_bvalue = []
    global max_iterations = 0# 設置最大迭代次數
    
    while max_iterations <= 5
        
        class1_p1, class1_p2, class1_g1, class1_g2 = cal_prob(data1[:, 1:7])
        

        new_std1, new_mu1 = cal_mean_std(class1_g1[:, 1:6])
        new_std2, new_mu2 = cal_mean_std(class1_g2[:, 1:6])
        println(new_std1)
        println(new_mu1)
        println(new_std2)
        println(new_mu2)
        new_class1= Gaussian_func( class1_g1[:, 1:6], class1_g2[:, 1:6], new_mu1, new_mu2, new_std1, new_std2)
        b_value = cal_b(class1_g1, class1_g2, new_mu1, new_mu2, new_std1, new_std2, class1_p1, class1_p2)
        #b_value = vcat(b_value, fill(1, 50 - length(b_value)))
        
        global new_bvalue = push!(new_bvalue , maximum(b_value))
       
    
        
        
       
        
    # println(p)
        data1 = new_class1
        global  max_iterations += 1
        
    
    end

    for i in 1:length(new_bvalue)
        new_bvalue[i] =log.( maximum(new_bvalue[i]))
    end

    
    #for i in 2:length(new_bvalue)
        
     #   global P_value = map(*, P_value, new_bvalue[i])
    #end
    println(new_bvalue)
    return new_bvalue
end


#=
#(Likely[2:end])
new_bvalue1 = result_cal(data1)
p1 = plot(collect(1:length(new_bvalue1)), new_bvalue1, xlabel="iteration", ylabel="Likelyhood")


new_bvalue2 = result_cal(data2)
p2 = plot(collect(1:length(new_bvalue2)), new_bvalue2, xlabel="iteration", ylabel="Likelyhood")


new_bvalue3 = result_cal(data3)
p3 = plot(collect(1:length(new_bvalue3)), new_bvalue3, xlabel="iteration", ylabel="Likelyhood")


new_bvalue4 = result_cal(data4)
p4 = plot(collect(1:length(new_bvalue4)), new_bvalue4, xlabel="iteration", ylabel="Likelyhood")


new_bvalue5 = result_cal(data5)
p5 = plot(collect(1:length(new_bvalue5)), new_bvalue5, xlabel="iteration", ylabel="Likelyhood")

new_bvalue6 = result_cal(data6)
p6 = plot(collect(1:length(new_bvalue6)), new_bvalue6, xlabel="iteration", ylabel="Likelyhood")


new_bvalue7 = result_cal(data7)
p7 = plot(collect(1:length(new_bvalue7)), new_bvalue7, xlabel="iteration", ylabel="Likelyhood")


new_bvalue8 = result_cal(data8)
p8 = plot(collect(1:length(new_bvalue8)), new_bvalue8, xlabel="iteration", ylabel="Likelyhood")


new_bvalue9 = result_cal(data9)
p9 = plot(collect(1:length(new_bvalue9)), new_bvalue9, xlabel="iteration", ylabel="Likelyhood")


new_bvalue10 = result_cal(data10)
p10 = plot(collect(1:length(new_bvalue10)), new_bvalue10, xlabel="iteration", ylabel="Likelyhood")


new_bvalue11 = result_cal(data11)
p11 = plot(collect(1:length(new_bvalue11)), new_bvalue11, xlabel="iteration", ylabel="Likelyhood")
=#

new_bvalue12 = result_cal(data12)
p12 = plot(collect(1:length(new_bvalue12)), new_bvalue12, xlabel="iteration", ylabel="Likelyhood")

#=
new_bvalue13 = result_cal(data13)
p13 = plot(collect(1:length(new_bvalue13)), new_bvalue13, xlabel="iteration", ylabel="Likelyhood")


new_bvalue14 = result_cal(data14)
p14 = plot(collect(1:length(new_bvalue14)), new_bvalue14, xlabel="iteration", ylabel="Likelyhood")


new_bvalue15 = result_cal(data15)
p15 = plot(collect(1:length(new_bvalue15)), new_bvalue15, xlabel="iteration", ylabel="Likelyhood")
=#
#=
new_bvalue16 = result_cal(data16)
p16 = plot(collect(1:length(new_bvalue16)), new_bvalue16, xlabel="iteration", ylabel="Likelyhood")
#plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,layout=(4,4))
=#