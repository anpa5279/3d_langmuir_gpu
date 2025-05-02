function fluctuation_xy(a)
    a_avg_xy = Statistics.mean(a, dims=(1, 2))
    a_fluctuation = a .- a_avg_xy 
    return a_fluctuation
end