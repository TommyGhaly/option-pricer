add_library(option_pricer_core
    binomial_tree.cpp
    black_scholes.cpp
    monte_carlo.cpp
    greeks.cpp
)

target_include_directories(option_pricer_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Installation
install(TARGETS option_pricer_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    binomial_tree.h
    black_scholes.h
    monte_carlo.h
    greeks.h
    DESTINATION include/option_pricer_core
)
