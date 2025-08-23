add_executable(option_pricer_tests
    test_black_scholes.cpp
    test_models.cpp
    test_monte_carlo.cpp
)

target_link_libraries(option_pricer_tests
    PRIVATE
        option_pricer_core
)

add_test(NAME option_pricer_tests COMMAND option_pricer_tests)
