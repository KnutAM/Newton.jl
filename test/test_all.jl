using Newton

Newton.logging_mode(;enable=true)
run(`julia -e 'using Pkg; Pkg.activate("."); Pkg.test()'`)
Newton.logging_mode(;enable=false)
run(`julia -e 'using Pkg; Pkg.activate("."); Pkg.test()'`)
