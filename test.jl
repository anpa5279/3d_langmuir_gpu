bᵢ(x, y, z) = z > - initial_mixed_layer_depth ? 0.0 : g_Earth * β * dTdz * (z + initial_mixed_layer_depth) 
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth) 

for z in grid.z.cᵃᵃᶜ
    b = bᵢ(0.0, 0.0, z)
    T = Tᵢ(0.0, 0.0, z)
    Ttob = (T - T0)*g_Earth*β
    b_diff = b - Ttob
    println("z = $z, b = $(b), T to b = $(Ttob), diff = $(b_diff)")
end