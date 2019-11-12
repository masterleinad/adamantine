# This function set the macro ADAMANTINE_WITH_CUDA and set ADAMANTINE_CUDA_LIBRARIES with
# the list of CUDA libraries that we are using
FUNCTION(SET_CUDA_LIBRARIES)
  ADD_DEFINITIONS(-DADAMANTINE_WITH_CUDA)
  SET(ADAMANTINE_CUDA_LIBRARIES
    "cusparse"
    "cusolver"
    PARENT_SCOPE
    )
  # cuSolver needs OpenMP
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}"
      PARENT_SCOPE
      )
    SET(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}"
      PARENT_SCOPE
      )
  ELSE()
    MESSAGE(SEND_ERROR "Could not find OpenMP required by cuSolver")
  ENDIF()
ENDFUNCTION()


