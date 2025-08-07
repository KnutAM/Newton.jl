using StaticArrays
using BenchmarkTools
using Newton
using RecursiveFactorization
import CairoMakie as Plt

function timeit()
    dims = [1:20..., 25:5:60...]
    inv_time = Float64[]
    sinv_time = Float64[]
    rinv_time = Float64[]
    for n in dims
        a = rand(SMatrix{n, n})
        println("SMatrix{$n, $n}: Equality check passed: ", inv(a) ≈ Newton.sinv(a))
        push!(inv_time,  minimum((@benchmark  inv($(rand(SMatrix{n, n})))).times))
        push!(sinv_time, minimum((@benchmark Newton.sinv($(rand(SMatrix{n, n})))).times))
        cache = NewtonCache(zeros(n); linsolver = Newton.RecursiveFactorizationLinsolver())
        push!(rinv_time, minimum((@benchmark Newton.inv!($(rand(n, n)), $cache)).times))
        println("inv: ", last(inv_time), ", sinv: ", last(sinv_time), ", speedup: ", (last(inv_time), last(rinv_time)) ./ last(sinv_time))
        println()
    end
    return dims, inv_time, sinv_time, rinv_time
end

# Default values based on ran benchmark
function plotit(;
    dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 55, 60], 
    inv_time = [1.708, 1.917, 3.625, 10.208, 94.66069546891465, 165.90170380078638, 237.36674816625916, 342.1658986175115, 540.5585106382979, 748.9495798319327, 1045.8, 1320.8, 1579.1, 2217.5555555555557, 2611.1111111111113, 3093.75, 3395.875, 3713.5, 4148.857142857143, 4339.285714285715, 6896.0, 9958.0, 13500.0, 16875.0, 22667.0, 27916.0, 34750.0, 39250.0], 
    sinv_time = [1.666, 1.917, 3.625, 6.625, 15.239478957915832, 25.14156626506024, 41.33097880928355, 61.41692150866463, 101.82271762208067, 155.2240932642487, 196.41754385964913, 274.9077490774908, 366.26341463414633, 396.25, 500.42783505154637, 579.0514285714286, 807.8777777777777, 1054.1, 1208.3, 1287.5, 2615.777777777778, 4053.5714285714284, 10917.0, 13584.0, 19458.0, 26916.0, 33083.0, 48000.0], 
    rinv_time = [57.026476578411405, 101.06951871657753, 125.28058361391695, 173.33285714285714, 236.30536130536132, 306.19747899159665, 416.6683417085427, 482.51295336787564, 611.7514450867052, 748.5966386554621, 894.8863636363636, 918.8947368421053, 1175.0, 1416.6, 1608.3, 1741.6, 2000.0, 2310.222222222222, 2587.8888888888887, 2703.777777777778, 4589.285714285715, 7062.5, 10041.0, 13041.0, 17417.0, 21791.0, 27667.0, 33250.0]
    )
    return plotit(dims, inv_time, sinv_time, rinv_time)
end

function plotit(dims, inv_time, sinv_time, rinv_time)
    fig = Plt.Figure()
    ax = Plt.Axis(fig[1,1]; xlabel = "dim", ylabel = "time [ns]", yscale = log10)
    Plt.lines!(ax, dims, inv_time; label = "inv")
    Plt.lines!(ax, dims, rinv_time; label = "rinv")
    Plt.lines!(ax, dims, sinv_time; label = "sinv")
    Plt.axislegend(ax; position = :rb)
    ax2 = Plt.Axis(fig[1,2]; xlabel = "dim", ylabel = "relative time")
    Plt.lines!(ax2, dims, sinv_time ./ inv_time; label = "inv")
    Plt.lines!(ax2, dims, sinv_time ./ rinv_time; label = "rinv")
    return fig
end
