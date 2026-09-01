find_package(OpenMP)

if (SPH_EXA_WITH_GRACKLE)
    add_library(cooler grackle_cooler.cpp)
    target_include_directories(cooler PRIVATE ${CSTONE_DIR} ${COOLING_DIR})
    target_link_libraries(cooler PRIVATE grackle OpenMP::OpenMP_CXX)
    target_compile_definitions(cooler PUBLIC SPH_EXA_HAVE_GRACKLE)
    target_compile_definitions(cooler PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_compile_definitions(cooler PRIVATE CONFIG_BFLOAT_8=ON)
endif ()


if (SPH_EXA_WITH_CHANGA_COOLING)
    add_library(changa_cosmo_cooler changa_cosmo_cooler.cpp)
    target_include_directories(changa_cosmo_cooler PRIVATE ${CSTONE_DIR} ${COOLING_DIR})
    target_link_libraries(changa_cosmo_cooler PRIVATE changa_cooling_cosmo OpenMP::OpenMP_CXX)
    target_compile_definitions(changa_cosmo_cooler PUBLIC SPH_EXA_HAVE_CHANGA_COOLING)
endif ()
