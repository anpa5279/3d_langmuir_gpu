using Pkg
using Plots
using Printf
using JLD2
using Oceananigans
using Measures
using Statistics
model = 0 # 0 for box model, 1 for carbonate diffeq test, 2 for 0d case, 3 for testing nonhydrostatic SplitCCRungeKutta3 in oceananigans, 4 for 3D nonhydrostatic space testing 
# opening oceananigans output file
if model == 0
     fld_file="outputs/box_model.jld2"
     image = "outputs/box_model.png"
     pd_image = "outputs/percent_difference_box.png"
     f = jldopen(fld_file)
     # reading the fields
     t_index = keys(f["timeseries/t"])
     CO₂_oc = Float64[] #FieldTimeSeries(fld_file, "CO₂"; backend=OnDisk())
     CO₃_oc = Float64[] #FieldTimeSeries(fld_file, "CO₃"; backend=OnDisk())
     HCO₃_oc = Float64[] #FieldTimeSeries(fld_file, "HCO₃"; backend=OnDisk())
     OH_oc = Float64[] #FieldTimeSeries(fld_file, "OH"; backend=OnDisk())
     BOH₃_oc = Float64[] #FieldTimeSeries(fld_file, "BOH₃"; backend=OnDisk())
     BOH₄_oc = Float64[] #FieldTimeSeries(fld_file, "BOH₄"; backend=OnDisk())
     t = Float64[] #
     for i in t_index
          #@show i
          push!(t, f["timeseries/t"][i])
          push!(CO₂_oc,   f["timeseries/CO₂"][i][1, 1, 1])
          push!(HCO₃_oc,  f["timeseries/HCO₃"][i][1, 1, 1])
          push!(CO₃_oc,   f["timeseries/CO₃"][i][1, 1, 1])
          push!(BOH₃_oc,  f["timeseries/BOH₃"][i][1, 1, 1])
          push!(BOH₄_oc,  f["timeseries/BOH₄"][i][1, 1, 1])
          push!(OH_oc,    f["timeseries/OH"][i][1, 1, 1])
     end
     t1_end = t[end]
     close(f)
     #opening second output file
     fld_file="outputs/box_model1.jld2"
     f = jldopen(fld_file)
     # reading the fields
     t_index = keys(f["timeseries/t"])
     for i in t_index
          #@show i
          push!(t, f["timeseries/t"][i] + t1_end)
          push!(CO₂_oc,   f["timeseries/CO₂"][i][1, 1, 1])
          push!(HCO₃_oc,  f["timeseries/HCO₃"][i][1, 1, 1])
          push!(CO₃_oc,   f["timeseries/CO₃"][i][1, 1, 1])
          push!(BOH₃_oc,  f["timeseries/BOH₃"][i][1, 1, 1])
          push!(BOH₄_oc,  f["timeseries/BOH₄"][i][1, 1, 1])
          push!(OH_oc,    f["timeseries/OH"][i][1, 1, 1])
     end
elseif model == 1
     @load "outputs/carbonate-diffeq-test.jld2" t u 
     image = "outputs/carbonate-diffeq-test.png"
     pd_image = "outputs/percent_difference_carbonate-diffeq-test.png"
     u = reduce(hcat, u)'
     CO₂_oc = u[:, 1]
     HCO₃_oc = u[:, 2]
     CO₃_oc = u[:, 3]
     H_oc = u[:, 4]
     OH_oc = u[:, 5]
     BOH₃_oc = u[:, 6]
     BOH₄_oc = u[:, 7]
     dt = 0.05
elseif model == 2
     @load "outputs/0d-case.jld2" t u 
     image = "outputs/0d-case-.png"
     pd_image = "outputs/percent_difference_0d-case-.png"
     u = reduce(hcat, u)'
     CO₂_oc = u[:, 1]
     HCO₃_oc = u[:, 2]
     CO₃_oc = u[:, 3]
     OH_oc = u[:, 4]
     BOH₃_oc = u[:, 5]
     BOH₄_oc = u[:, 6]
     dt = 0.05
elseif model == 3
     fld_file="split-testing.jld2"
     image = "outputs/split-testing.png"
     pd_image = "outputs/percent_difference_split.png"
     f = jldopen(fld_file)
     # reading the fields
     t_index = keys(f["timeseries/t"])
     CO₂_oc = Float64[] #FieldTimeSeries(fld_file, "CO₂"; backend=OnDisk())
     CO₃_oc = Float64[] #FieldTimeSeries(fld_file, "CO₃"; backend=OnDisk())
     HCO₃_oc = Float64[] #FieldTimeSeries(fld_file, "HCO₃"; backend=OnDisk())
     OH_oc = Float64[] #FieldTimeSeries(fld_file, "OH"; backend=OnDisk())
     BOH₃_oc = Float64[] #FieldTimeSeries(fld_file, "BOH₃"; backend=OnDisk())
     BOH₄_oc = Float64[] #FieldTimeSeries(fld_file, "BOH₄"; backend=OnDisk())
     t = Float64[] #
     for i in t_index
          #@show i
          push!(t, f["timeseries/t"][i])
          push!(CO₂_oc,   f["timeseries/CO₂"][i][1, 1, 1])
          push!(HCO₃_oc,  f["timeseries/HCO₃"][i][1, 1, 1])
          push!(CO₃_oc,   f["timeseries/CO₃"][i][1, 1, 1])
          push!(BOH₃_oc,  f["timeseries/BOH₃"][i][1, 1, 1])
          push!(BOH₄_oc,  f["timeseries/BOH₄"][i][1, 1, 1])
          push!(OH_oc,    f["timeseries/OH"][i][1, 1, 1])
     end
     close(f)
elseif model == 4
     fld_file="langmuir_turbulence_fields.jld2"
     image = "outputs/3d.png"
     pd_image = "outputs/percent_difference_3d.png"
     f = jldopen(fld_file)
     # reading the fields
     t_index = keys(f["timeseries/t"])
     CO₂_oc = Float64[] #FieldTimeSeries(fld_file, "CO₂"; backend=OnDisk())
     CO₃_oc = Float64[] #FieldTimeSeries(fld_file, "CO₃"; backend=OnDisk())
     HCO₃_oc = Float64[] #FieldTimeSeries(fld_file, "HCO₃"; backend=OnDisk())
     OH_oc = Float64[] #FieldTimeSeries(fld_file, "OH"; backend=OnDisk())
     BOH₃_oc = Float64[] #FieldTimeSeries(fld_file, "BOH₃"; backend=OnDisk())
     BOH₄_oc = Float64[] #FieldTimeSeries(fld_file, "BOH₄"; backend=OnDisk())
     t = Float64[] #
     for i in t_index
          #@show i
          push!(t, f["timeseries/t"][i]) # average over the whole domain
          CO₂_avg = mean(f["timeseries/CO₂"][i][3:end-3, 3:end-3, 3:end-3])
          push!(CO₂_oc,   CO₂_avg)
          HCO₃_avg = mean(f["timeseries/HCO₃"][i][3:end-3, 3:end-3, 3:end-3])
          push!(HCO₃_oc,  HCO₃_avg)
          CO₃_avg = mean(f["timeseries/CO₃"][i][3:end-3, 3:end-3, 3:end-3])
          push!(CO₃_oc,   CO₃_avg)
          BOH₃_avg = mean(f["timeseries/BOH₃"][i][3:end-3, 3:end-3, 3:end-3])
          push!(BOH₃_oc,  BOH₃_avg)
          BOH₄_avg = mean(f["timeseries/BOH₄"][i][3:end-3, 3:end-3, 3:end-3]) # average over the whole domain
          push!(BOH₄_oc,  BOH₄_avg)
          OH_avg = mean(f["timeseries/OH"][i][3:end-3, 3:end-3, 3:end-3])
          push!(OH_oc,    OH_avg)
     end
     close(f)
end 
N = length(t)
# opening fortran output file
fortran_file = "outputs/cc-smalltimestep.hst"
f = open(fortran_file)

t_f = Float64[]
CO₂_f = Float64[]
CO₃_f = Float64[]
HCO₃_f = Float64[]
OH_f = Float64[]
BOH₃_f = Float64[]
BOH₄_f = Float64[]

CO₂_d = Float64[]
CO₃_d = Float64[]
HCO₃_d = Float64[]
OH_d = Float64[]
BOH₃_d = Float64[]
BOH₄_d = Float64[]

line_count = 0              

for lines in readlines(f)
     if N > line_count
          values = parse.(Float64, split(lines))
          global line_count += 1 
          # Store in respective arrays (assuming correct ordering in file)
          push!(t_f, values[1])
          push!(CO₂_f,   values[2])
          push!(HCO₃_f,  values[3])
          push!(CO₃_f,   values[4])
          push!(BOH₃_f,  values[5])
          push!(BOH₄_f,  values[6])
          push!(OH_f,    values[7])
          push!(CO₂_d,  (CO₂_oc[line_count] - CO₂_f[line_count]) / CO₂_f[line_count] * 100)
          push!(CO₃_d,  (CO₃_oc[line_count] - CO₃_f[line_count]) / CO₃_f[line_count] * 100)
          push!(HCO₃_d, (HCO₃_oc[line_count] - HCO₃_f[line_count]) / HCO₃_f[line_count] * 100)
          push!(OH_d,   (OH_oc[line_count] - OH_f[line_count]) / OH_f[line_count] * 100) 
          push!(BOH₃_d, (BOH₃_oc[line_count] - BOH₃_f[line_count]) / BOH₃_f[line_count] * 100)
          push!(BOH₄_d, (BOH₄_oc[line_count] - BOH₄_f[line_count]) / BOH₄_f[line_count] * 100)
     else
          break 
     end 
end

# plotting results
starting = 1
stopping = N
# CO₂
CO₂p = plot(t[starting:stopping], [CO₂_oc[starting:stopping] CO₂_f[starting:stopping]],
     xlabel = "t (s)", ylabel = "CO₂ (micromol/kg)",
     title = "CO₂")
# CO₃
CO₃p = plot(t[starting:stopping], [CO₃_oc[starting:stopping] CO₃_f[starting:stopping]],
     xlabel = "t (s)", ylabel = "CO₃ (micromol/kg)",
     title = "CO₃")
# HCO₃
HCO₃p = plot(t[starting:stopping], [HCO₃_oc[starting:stopping] HCO₃_f[starting:stopping]],
     xlabel = "t (s)", ylabel = "HCO₃ (micromol/kg)",
     title = "HCO₃")
# OH
OHp = plot(t[starting:stopping], [OH_oc[starting:stopping] OH_f[starting:stopping]],
     xlabel = "t (s)", ylabel = "OH (micromol/kg)",
     title = "OH")
# BOH₃
BOH₃p = plot(t[starting:stopping], [BOH₃_oc[starting:stopping] BOH₃_f[starting:stopping]],
     xlabel = "t (s)", ylabel = "B(OH)₃ (micromol/kg)",
     title = "B(OH)₃")
# BOH₄
BOH₄p = plot(t[starting:stopping], [BOH₄_oc[starting:stopping] BOH₄_f[starting:stopping]],
     xlabel = "t (s)", ylabel = "B(OH)₄ (micromol/kg)",
     title = "B(OH)₄")
# layout 
gr()
pf = plot(CO₂p, CO₃p, HCO₃p, OHp, BOH₃p, BOH₄p, layout = (3, 2), label = ["Oceananigans" "Fortran Miniapp"], legend = :outertop,         # << shared legend at the top
           legendcolumns = 2,
           size = (1600, 1200),
           left_margin = 30mm,
           bottom_margin = 10mm,
           plot_margin = 5mm, 
           titlefont = font(12), 
           tickfont = font(9), 
           labelfont = font(10), 
           xlabelfont = font(10), 
           ylabelfont = font(10), 
           titlefontsize = 12)
png(pf, image)

#plotting percent differences
#CO₂
CO₂p = plot(t[starting:stopping], CO₂_d[starting:stopping],
     xlabel = "t (s)", ylabel = "CO₂ %",
     title = "CO₂")
# CO₃
CO₃p = plot(t[starting:stopping], CO₃_d[starting:stopping],
     xlabel = "t (s)", ylabel = "CO₃ %",
     title = "CO₃")
# HCO₃
HCO₃p = plot(t[starting:stopping], HCO₃_d[starting:stopping],
     xlabel = "t (s)", ylabel = "HCO₃ %",
     title = "HCO₃")
# OH
OHp = plot(t[starting:stopping], OH_d[starting:stopping],
     xlabel = "t (s)", ylabel = "OH %",
     title = "OH")
# BOH₃
BOH₃p = plot(t[starting:stopping], BOH₃_d[starting:stopping],
     xlabel = "t (s)", ylabel = "B(OH)₃ %",
     title = "B(OH)₃")
# BOH₄
BOH₄p = plot(t[starting:stopping], BOH₄_d[starting:stopping],
     xlabel = "t (s)", ylabel = "B(OH)₄ %",
     title = "B(OH)₄")
# layout 
gr()
pf = plot(CO₂p, CO₃p, HCO₃p, OHp, BOH₃p, BOH₄p, layout = (3, 2), legend = false,         # << shared legend at the top
           size = (1600, 1200),
           left_margin = 30mm,
           bottom_margin = 10mm,
           plot_margin = 5mm, 
           titlefont = font(12), 
           tickfont = font(9), 
           labelfont = font(10), 
           xlabelfont = font(10), 
           ylabelfont = font(10), 
           titlefontsize = 12)
png(pf, pd_image)