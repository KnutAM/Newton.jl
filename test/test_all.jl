using Newton

Newton.logging(;enable=true)
run(`julia -i -e 'using Pkg; Pkg.activate("."); Pkg.test()'`)
Newton.logging(;enable=false)
run(`julia -i -e 'using Pkg; Pkg.activate("."); Pkg.test()'`)