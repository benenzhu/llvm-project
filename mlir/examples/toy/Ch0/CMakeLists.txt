# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh0CombineIncGen)

add_toy_chapter(toyc-ch0
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp

  DEPENDS
  ToyCh0OpsIncGen
  ToyCh0ShapeInferenceInterfaceIncGen
  ToyCh0CombineIncGen
  )

# this line is new here
include_directories(${CMAKE_CURRENT_BINARY_DIR}) 
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch0
  PRIVATE
    MLIRAnalysis
    MLIRCastInterfaces
    MLIRCallInterfaces
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    # here add this MLIRpass why?
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTransforms)

