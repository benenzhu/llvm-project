# For a better template to copy, see examples/standalone
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_toy_chapter(toyc-ch0
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  DEPENDS
  ToyCh0OpsIncGen

  )
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(toyc-ch0
  PRIVATE
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)
