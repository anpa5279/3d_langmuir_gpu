function finite_diff(i, j, k, grid, field, dir, loc)
    if loc == "c"
        m = 1
    elseif loc == "f"
        m = -1
    end 
    if dir =="x"
        return (field[i + m, j, k] + field[i, j, k]) / grid.Δxᶠᵃᵃ
    elseif dir =="y"
        return (field[i, j + m, k] + field[i, j, k]) / grid.Δyᵃᶠᵃ
    elseif dir =="z"
        return (field[i, j, k + m] + field[i, j, k]) / grid.z.Δᵃᵃᶠ
    end
end

function finite_diff_nofield(a, b, diff)
    return (a + b) / diff
end

function avg(a, b) #ℑxyᶜᶜᵃ
    return 0.5 * (a+b)
end 

function strain(i, j, k, grid, velocities) 
    u = velocities.u 
    v = velocities.v 
    w = velocities.w 
    strain_tensor = zeros(3, 3)
    dir = ("x", "y", "z")
    loc = "c"
    for i = 1:3
        vel = velocities[keys(velocities)[i]]
        strain_tensor[i, i] = 0.5 * (finite_diff(i, j, k, grid, vel, dir[i], loc) + finite_diff(i, j, k, grid, vel, dir[i], loc))
    end
    strain_tensor[1, 2] = 0.5 * (finite_diff(i, j, k, grid, u, "y", "f") + finite_diff(i, j, k, grid, v, "x", "f"))
    strain_tensor[2, 3] = 0.5 * (finite_diff(i, j, k, grid, v, "z", "f") + finite_diff(i, j, k, grid, w, "y", "f"))
    strain_tensor[3, 1] = 0.5 * (finite_diff(i, j, k, grid, w, "x", "f") + finite_diff(i, j, k, grid, u, "z", "f"))
    strain_tensor[2, 1] = strain_tensor[1, 2]
    strain_tensor[3, 2] = strain_tensor[2, 3]
    strain_tensor[1, 3] = strain_tensor[3, 1]
    return strain_tensor
end 