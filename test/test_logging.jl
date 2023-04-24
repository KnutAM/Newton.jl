using Preferences

if Newton.LOGGING    
    @testset "logging_mode" begin
        @test Preferences.load_preference(Newton, "log_iterations")
        Newton.logging_mode(;enable=false)
        @test !(Preferences.load_preference(Newton, "log_iterations"))
        Newton.logging_mode(;enable=true) # Test that it sets even if already enabled 
        @test Preferences.load_preference(Newton, "log_iterations")
    end
end