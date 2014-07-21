
#Require a minimum version of CMake
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

#This project is C++ based.
project(hpcg CXX)

# This adds the HPX cmake configuration directory to the search path.

message(STATUS
${HPX_ROOT}/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}/Modules)
set(CMAKE_MODULE_PATH
${HPX_ROOT}/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}/Modules)

# add our cmake path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

###########################
# Include boost and hpx
#
find_package(HPX_Boost)

hpx_include_sys_directories("${BOOST_INCLUDE_DIR}")
hpx_link_sys_directories("${BOOST_LIBRARY_DIR}")


#Instruct cmake to find the HPX settings


if(NOT HPCG_NOHPX)
   find_package(HPX REQUIRED)
   set(HPX_RPATH "${CMAKE_BINARY_DIR}/lib/hpx:${HPX_RPATH}:${CMAKE_INSTALL_PREFIX}/lib/hpx")
   include_directories(${HPX_INCLUDE_DIR})
   link_directories(${HPX_LIBRARY_DIR})
else()
   add_definitions("-DHPCG_NOHPX")   
endif()


if(NOT HPCG_NOOPENMP)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
else()
    add_definitions("-DHPCG_NOOPENMP")
endif()

if(NOT HPCG_NOMPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
else()
    add_definitions("-DHPCG_NOMPI")
endif()




########################################################################
# configurations
# 


########################################################################
# further includes, depending on config
# 

#########################
# Builds
# 


if(HPCG_DEBUG)
    add_definitions("-DHPCG_DEBUG")
endif()

set(SOURCES
            src/CG.cpp
            src/parallel_for.cpp
            src/CG_ref.cpp
            src/TestCG.cpp
            src/ComputeResidual.cpp
            src/ExchangeHalo.cpp
            src/GenerateGeometry.cpp
            src/GenerateProblem.cpp
            src/OptimizeProblem.cpp
            src/ReadHpcgDat.cpp
            src/ReportResults.cpp
            src/SetupHalo.cpp
            src/TestSymmetry.cpp
            src/TestNorms.cpp
            src/WriteProblem.cpp
            src/YAML_Doc.cpp
            src/YAML_Element.cpp
            src/ComputeDotProduct.cpp
            src/ComputeDotProduct_ref.cpp
            src/mytimer.cpp
            src/ComputeSPMV.cpp
            src/ComputeSPMV_ref.cpp
            src/ComputeSYMGS.cpp
            src/ComputeSYMGS_ref.cpp
            src/ComputeWAXPBY.cpp
            src/ComputeWAXPBY_ref.cpp
            src/ComputeMG_ref.cpp
            src/ComputeMG.cpp
            src/ComputeProlongation_ref.cpp
            src/ComputeRestriction_ref.cpp
            src/GenerateCoarseProblem.cpp
            src/init.cpp
            src/finalize.cpp
            testing/main.cpp
)




if(HPCG_NOHPX)
  add_executable(hpcg
    ${SOURCES}
  )
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  if(NOT HPCG_NOMPI)
     target_link_libraries(hpcg ${MPI_LIBRARIES})
     if(MPI_COMPILE_FLAGS)
         set_target_properties(hpcg PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
     endif()
     if(MPI_LINK_FLAGS)
         set_target_properties(hpcg PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
     endif()
  endif()
else()
  add_hpx_executable(hpcg
   ESSENTIAL
   SOURCES  
    ${SOURCES}
   HEADERS  
   COMPONENT_DEPENDENCIES iostreams 
   DEPENDENCIES
  )
endif()


