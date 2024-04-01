using Clustering, Plots,Distances
using CSV
using LinearAlgebra
using DataFrames
using Random

function new_data_set(df)
    
    return df[:, 1:end-1]
end

#取資料
function data_check(Dim1,Dim2,Dim3,Dim4,Dim5,Dim6) #Dim1[:,1]
    num = [Dim1,Dim2,Dim3,Dim4,Dim5,Dim6] 
    return num
end

#轉dataframe
function dataframe_change(data)
    matclass = DataFrame(d1 = map(x -> x[1],data),c1 = map(x ->x[2],data),
                        d2 = map(x -> x[3],data),c2 = map(x -> x[4],data),
                        d3 = map(x->x[5],data),c3 = map(x -> x[6],data))
    return matclass
end

function cal_kmean(data)
    feature = collect(Matrix(data))
    result = kmeans(feature', 2)
    assignments = result.assignments
    return result.centers,assignments

end
function paste_data(data1,data2)
    result = vcat([data1],[data2])
    return result
end
file_path = "dye.csv"
df = CSV.read(file_path, DataFrame)
global class_matrix = df[:,end] #class

#設定16class
global df_matclass1 = []
global df_matclass2 = []
global df_matclass3 = []
global df_matclass4 = []
global df_matclass5 = []
global df_matclass6 = []
global df_matclass7 = []
global df_matclass8 = []
global df_matclass9 = []
global df_matclass10 = []
global df_matclass11 = []
global df_matclass12 = []
global df_matclass13 = []
global df_matclass14 = []
global df_matclass15 = []
global df_matclass16 = []
global df = new_data_set(df)
global new_data = []

#將dataframe的數據轉變為陣列型態
for row in eachrow(df)  
    new_data1 = data_check(row[1],row[2],row[3],row[4],row[5],row[6])
    push!(new_data,new_data1)  
end

#利用class的數據來分組
for i in 1:size(new_data,1)
    if class_matrix[i] == 1
        push!(df_matclass1,new_data[i])
    end
    if class_matrix[i] == 2
        push!(df_matclass2,new_data[i])
    end
    if class_matrix[i] == 3
        push!(df_matclass3,new_data[i])
    end
    if class_matrix[i] == 4
        push!(df_matclass4,new_data[i])
    end
    if class_matrix[i] == 5
        push!(df_matclass5,new_data[i])
    end
    if class_matrix[i] == 6
        push!(df_matclass6,new_data[i])
    end
    if class_matrix[i] == 7
        push!(df_matclass7,new_data[i])
    end
    if class_matrix[i] == 8
        push!(df_matclass8,new_data[i])
    end
    if class_matrix[i] == 9
        push!(df_matclass9,new_data[i])
    end
    if class_matrix[i] == 10
        push!(df_matclass10,new_data[i])
    end
    if class_matrix[i] == 11
        push!(df_matclass11,new_data[i])
    end
    if class_matrix[i] == 12
        push!(df_matclass12,new_data[i])
    end
    if class_matrix[i] == 13
        push!(df_matclass13,new_data[i])
    end
    if class_matrix[i] == 14
        push!(df_matclass14,new_data[i])
    end
    if class_matrix[i] == 15
        push!(df_matclass15,new_data[i])
    end
    if class_matrix[i] == 16
        push!(df_matclass16,new_data[i])
    end
end

#隨機取50資料、並存取剩餘資料
matclass1 = df_matclass1[randperm(length(df_matclass1))[1:50]]
remaining_data_1 = filter!(x -> !(x in matclass1), df_matclass1)
matclass2 = df_matclass2[randperm(length(df_matclass2))[1:50]]
remaining_data_2 = filter!(x -> !(x in matclass2), df_matclass2)
matclass3 = df_matclass3[randperm(length(df_matclass3))[1:50]]
remaining_data_3 = filter!(x -> !(x in matclass3), df_matclass3)
matclass4 = df_matclass4[randperm(length(df_matclass4))[1:50]]
remaining_data_4 = filter!(x -> !(x in matclass4), df_matclass4)
matclass5 = df_matclass5[randperm(length(df_matclass5))[1:50]]
remaining_data_5 = filter!(x -> !(x in matclass5), df_matclass5)
matclass6 = df_matclass6[randperm(length(df_matclass6))[1:50]]
remaining_data_6 = filter!(x -> !(x in matclass6), df_matclass6)
matclass7 = df_matclass7[randperm(length(df_matclass7))[1:50]]
remaining_data_7 = filter!(x -> !(x in matclass7), df_matclass7)
matclass8 = df_matclass8[randperm(length(df_matclass8))[1:50]]
remaining_data_8 = filter!(x -> !(x in matclass8), df_matclass8)
matclass9 = df_matclass9[randperm(length(df_matclass9))[1:50]]
remaining_data_9 = filter!(x -> !(x in matclass9), df_matclass9)
matclass10 = df_matclass10[randperm(length(df_matclass10))[1:50]]
remaining_data_10 = filter!(x -> !(x in matclass10), df_matclass10)
matclass11 = df_matclass11[randperm(length(df_matclass11))[1:50]]
remaining_data_11 = filter!(x -> !(x in matclass11), df_matclass11)
matclass12 = df_matclass12[randperm(length(df_matclass12))[1:50]]
remaining_data_12 = filter!(x -> !(x in matclass12), df_matclass12)
matclass13 = df_matclass13[randperm(length(df_matclass13))[1:50]]
remaining_data_13 = filter!(x -> !(x in matclass13), df_matclass13)
matclass14 = df_matclass14[randperm(length(df_matclass14))[1:50]]
remaining_data_14 = filter!(x -> !(x in matclass14), df_matclass14)
matclass15 = df_matclass15[randperm(length(df_matclass15))[1:50]]
remaining_data_15 = filter!(x -> !(x in matclass15), df_matclass15)
matclass16 = df_matclass16[randperm(length(df_matclass16))[1:50]]
remaining_data_16 = filter!(x -> !(x in matclass16), df_matclass16)



#轉dataframe，存訓練資料
df_mat1 = dataframe_change(matclass1)
center1,group1 = cal_kmean(df_mat1)
df_mat1[!, :cluster] = group1
CSV.write("class1.csv", df_mat1)
df_remaining1 = dataframe_change(remaining_data_1)
center1_re,group1_re = cal_kmean(df_remaining1)
df_remaining1[!, :cluster] = group1_re
CSV.write("class1_remain.csv",df_remaining1)

df_mat2 = dataframe_change(matclass2)
center2,group2 = cal_kmean(df_mat2)
df_mat2[!, :cluster] = group2
CSV.write("class2.csv", df_mat2)
df_remaining2 = dataframe_change(remaining_data_2)
center2_re,group2_re = cal_kmean(df_remaining2)
df_remaining2[!, :cluster] = group2_re
CSV.write("class2_remain.csv",df_remaining2)

df_mat3 = dataframe_change(matclass3)
center3,group3 = cal_kmean(df_mat3)
df_mat3[!, :cluster] = group3
CSV.write("class3.csv", df_mat3)
df_remaining3 = dataframe_change(remaining_data_3)
center3_re,group3_re = cal_kmean(df_remaining3)
df_remaining3[!, :cluster] = group3_re
CSV.write("class3_remain.csv",df_remaining3)

df_mat4 = dataframe_change(matclass4)
center4,group4 = cal_kmean(df_mat4)
df_mat4[!, :cluster] = group4
CSV.write("class4.csv", df_mat4)
df_remaining4 = dataframe_change(remaining_data_4)
center4_re,group4_re = cal_kmean(df_remaining4)
df_remaining4[!, :cluster] = group4_re
CSV.write("class4_remain.csv",df_remaining4)

df_mat5 = dataframe_change(matclass5)
center5,group5 = cal_kmean(df_mat5)
df_mat5[!, :cluster] = group5
CSV.write("class5.csv", df_mat5)
df_remaining5 = dataframe_change(remaining_data_5)
center5_re,group5_re = cal_kmean(df_remaining5)
df_remaining5[!, :cluster] = group5_re
CSV.write("class5_remain.csv",df_remaining5)

df_mat6 = dataframe_change(matclass6)
center6,group6 = cal_kmean(df_mat6)
df_mat6[!, :cluster] = group6
CSV.write("class6.csv", df_mat6)
df_remaining6 = dataframe_change(remaining_data_6)
center6_re,group6_re = cal_kmean(df_remaining6)
df_remaining6[!, :cluster] = group6_re
CSV.write("class6_remain.csv",df_remaining6)

df_mat7 = dataframe_change(matclass7)
center7,group7 = cal_kmean(df_mat7)
df_mat7[!, :cluster] = group7
CSV.write("class7.csv", df_mat7)
df_remaining7 = dataframe_change(remaining_data_7)
center7_re,group7_re = cal_kmean(df_remaining7)
df_remaining7[!, :cluster] = group7_re
CSV.write("class7_remain.csv",df_remaining7)

df_mat8 = dataframe_change(matclass8)
center8,group8 = cal_kmean(df_mat8)
df_mat8[!, :cluster] = group8
CSV.write("class8.csv", df_mat8)
df_remaining8 = dataframe_change(remaining_data_8)
center8_re,group8_re = cal_kmean(df_remaining8)
df_remaining8[!, :cluster] = group8_re
CSV.write("class8_remain.csv",df_remaining8)

df_mat9 = dataframe_change(matclass9)
center9,group9 = cal_kmean(df_mat9)
df_mat9[!, :cluster] = group9
CSV.write("class9.csv", df_mat9)
df_remaining9 = dataframe_change(remaining_data_9)
center9_re,group9_re = cal_kmean(df_remaining9)
df_remaining9[!, :cluster] = group9_re
CSV.write("class9_remain.csv",df_remaining9)

df_mat10 = dataframe_change(matclass10)
center10,group10 = cal_kmean(df_mat10)
df_mat10[!, :cluster] = group10
CSV.write("class10.csv", df_mat10)
df_remaining10 = dataframe_change(remaining_data_10)
center10_re,group10_re = cal_kmean(df_remaining10)
df_remaining10[!, :cluster] = group10_re
CSV.write("class10_remain.csv",df_remaining10)

df_mat11 = dataframe_change(matclass11)
center11,group11 = cal_kmean(df_mat11)
df_mat11[!, :cluster] = group11
CSV.write("class11.csv", df_mat11)
df_remaining11 = dataframe_change(remaining_data_11)
center11_re,group11_re = cal_kmean(df_remaining11)
df_remaining11[!, :cluster] = group11_re
CSV.write("class11_remain.csv",df_remaining11)

df_mat12 = dataframe_change(matclass12)
center12,group12 = cal_kmean(df_mat12)
df_mat12[!, :cluster] = group12
CSV.write("class12.csv", df_mat12)
df_remaining12 = dataframe_change(remaining_data_12)
center12_re,group12_re = cal_kmean(df_remaining12)
df_remaining12[!, :cluster] = group12_re
CSV.write("class12_remain.csv",df_remaining12)

df_mat13 = dataframe_change(matclass13)
center13,group13 = cal_kmean(df_mat13)
df_mat13[!, :cluster] = group13
CSV.write("class13.csv", df_mat13)
df_remaining13 = dataframe_change(remaining_data_13)
center13_re,group13_re = cal_kmean(df_remaining13)
df_remaining13[!, :cluster] = group13_re
CSV.write("class13_remain.csv",df_remaining13)

df_mat14 = dataframe_change(matclass14)
center14,group14 = cal_kmean(df_mat14)
df_mat14[!, :cluster] = group14
CSV.write("class14.csv", df_mat14)
df_remaining14 = dataframe_change(remaining_data_14)
center14_re,group14_re = cal_kmean(df_remaining14)
df_remaining14[!, :cluster] = group14_re
CSV.write("class14_remain.csv",df_remaining14)

df_mat15 = dataframe_change(matclass15)
center15,group15 = cal_kmean(df_mat15)
df_mat15[!, :cluster] = group15
CSV.write("class15.csv", df_mat15)
df_remaining15 = dataframe_change(remaining_data_15)
center15_re,group15_re = cal_kmean(df_remaining15)
df_remaining15[!, :cluster] = group15_re
CSV.write("class15_remain.csv",df_remaining15)

df_mat16 = dataframe_change(matclass16)
center16,group16 = cal_kmean(df_mat16)
df_mat16[!, :cluster] = group16
CSV.write("class16.csv", df_mat16)
df_remaining16 = dataframe_change(remaining_data_16)
center16_re,group16_re = cal_kmean(df_remaining16)
df_remaining16[!, :cluster] = group16_re
CSV.write("class16_remain.csv",df_remaining16)

combined_df_remain = vcat(df_remaining1, df_remaining2, df_remaining3, df_remaining4,df_remaining5,
                            df_remaining6,df_remaining7,df_remaining8,df_remaining9,df_remaining10,
                            df_remaining11,df_remaining12,df_remaining13,df_remaining14,df_remaining15,df_remaining16)

CSV.write("test.csv", combined_df_remain)
#取中心點做資料黏貼
global swap = []
data1 = paste_data(center1[:,1],center1[:,2])
swap = vcat(swap,data1)
data2 = paste_data(center2[:,1],center2[:,1])
swap = vcat(swap,data2)
data3 = paste_data(center3[:,1],center3[:,2])
swap = vcat(swap,data3)
data4 = paste_data(center4[:,1],center4[:,1])
swap = vcat(swap,data4)
data5 = paste_data(center5[:,1],center5[:,2])
swap = vcat(swap,data5)
data6 = paste_data(center6[:,1],center6[:,1])
swap = vcat(swap,data6)
data7 = paste_data(center7[:,1],center7[:,2])
swap = vcat(swap,data7)
data8 = paste_data(center8[:,1],center8[:,1])
swap = vcat(swap,data8)
data9 = paste_data(center9[:,1],center9[:,2])
swap = vcat(swap,data9)
data10 = paste_data(center10[:,1],center10[:,1])
swap = vcat(swap,data10)
data11 = paste_data(center11[:,1],center11[:,2])
swap = vcat(swap,data11)
data12 = paste_data(center12[:,1],center12[:,1])
swap = vcat(swap,data12)
data13 = paste_data(center13[:,1],center13[:,2])
swap = vcat(swap,data13)
data14 = paste_data(center14[:,1],center14[:,1])
swap = vcat(swap,data14)
data15 = paste_data(center15[:,1],center15[:,2])
swap = vcat(swap,data15)
data16 = paste_data(center16[:,1],center16[:,1])
swap = vcat(swap,data16)
result = vcat(dataframe_change(swap))

#加上前面1,2欄位
new_col = [[1],[2],[1],[2],[1],[2],[1],[2],[1],[2],
            [1],[2],[1],[2],[1],[2],[1],[2],[1],[2],
            [1],[2],[1],[2],[1],[2],[1],[2],[1],[2],[1],[2]]
new_col = DataFrame(col = map(x -> x[1],new_col))
ans = hcat(new_col,result)
CSV.write("GMM_data.csv", ans)









